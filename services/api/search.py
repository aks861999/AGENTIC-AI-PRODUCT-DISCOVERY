"""
Search functions - three strategies exposed to the API layer.

All OpenSearch calls are executed in a thread pool via asyncio.to_thread()
so that the async FastAPI event loop is never blocked.

Functions:
    lexical_search()  — BM25 multi_match + function_score rating boost
    semantic_search() — kNN approximate nearest-neighbour query
    hybrid_search()   — Combined BM25 + kNN with normalization pipeline
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from opensearchpy import OpenSearch

from api.models import ProductHit, SearchMode, SearchResponse
from ingestion.schema import HYBRID_PIPELINE_ID

logger = logging.getLogger(__name__)


# ── Query builders ────────────────────────────────────────────────────────────

def _build_filter_clauses(
    category: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_rating: Optional[float] = None,
) -> list[dict]:
    """Return a list of OpenSearch filter clauses from optional params."""
    filters: list[dict] = []
    if category:
        filters.append({"term": {"main_category": category}})
    price_range: dict[str, float] = {}
    if min_price is not None:
        price_range["gte"] = min_price
    if max_price is not None:
        price_range["lte"] = max_price
    if price_range:
        filters.append({"range": {"price": price_range}})
    if min_rating is not None:
        filters.append({"range": {"average_rating": {"gte": min_rating}}})
    return filters


def _build_lexical_query(
    query: str,
    filters: list[dict],
    boost_terms: Optional[list[str]] = None,
    size: int = 10,
    from_: int = 0,
) -> dict:
    """
    Build a BM25 multi_match query with function_score rating boost.

    Title is boosted 3×, features 2×, description 1×.
    Rating and log-normalised rating_number are used as multiplicative boosts.
    """
    should_clauses: list[dict] = []
    if boost_terms:
        for term in boost_terms:
            should_clauses.append(
                {"multi_match": {"query": term, "fields": ["title^2", "features"], "boost": 3}}
            )

    base_query: dict = {
        "function_score": {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^3", "features^2", "description^1"],
                                "type": "best_fields",
                                "tie_breaker": 0.3,
                                "fuzziness": "AUTO",
                            }
                        }
                    ],
                    "should": should_clauses,
                    "filter": filters,
                }
            },
            "functions": [
                {
                    "field_value_factor": {
                        "field": "average_rating",
                        "factor": 0.5,
                        "modifier": "none",
                        "missing": 1,
                    }
                },
                {
                    "field_value_factor": {
                        "field": "rating_number",
                        "factor": 0.1,
                        "modifier": "log1p",
                        "missing": 1,
                    }
                },
            ],
            "score_mode": "sum",
            "boost_mode": "multiply",
        }
    }

    return {
        "from": from_,
        "size": size,
        "_source": {"excludes": ["embedding_vector", "embedding_text"]},
        "query": base_query,
    }


def _build_semantic_query(
    query_vector: list[float],
    filters: list[dict],
    k: int = 100,
    size: int = 10,
) -> dict:
    """
    Build a kNN query for approximate nearest-neighbour search.

    Pre-filter is applied before kNN (supported from OpenSearch 2.14+).
    """
    body: dict = {
        "size": size,
        "_source": {"excludes": ["embedding_vector", "embedding_text"]},
        "query": {
            "knn": {
                "embedding_vector": {
                    "vector": query_vector,
                    "k": k,
                }
            }
        },
    }
    if filters:
        body["query"] = {
            "knn": {
                "embedding_vector": {
                    "vector": query_vector,
                    "k": k,
                    "filter": {"bool": {"filter": filters}},
                }
            }
        }
    return body


def _build_hybrid_query(
    query: str,
    query_vector: list[float],
    filters: list[dict],
    boost_terms: Optional[list[str]] = None,
    size: int = 10,
    from_: int = 0,
) -> dict:
    """
    Build a hybrid query combining BM25 and kNN sub-queries.

    The normalization_processor pipeline (registered at startup) handles
    min-max normalisation and arithmetic_mean combination with weights [0.4, 0.6].
    """
    should_clauses: list[dict] = []
    if boost_terms:
        for term in boost_terms:
            should_clauses.append(
                {"multi_match": {"query": term, "fields": ["title^2", "features"], "boost": 3}}
            )

    bm25_sub: dict = {
        "bool": {
            "must": [
                {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^3", "features^2", "description^1"],
                        "type": "best_fields",
                        "tie_breaker": 0.3,
                    }
                }
            ],
            "should": should_clauses,
            "filter": filters,
        }
    }

    knn_sub: dict = {
        "knn": {
            "embedding_vector": {
                "vector": query_vector,
                "k": 100,
            }
        }
    }
    if filters:
        knn_sub["knn"]["embedding_vector"]["filter"] = {"bool": {"filter": filters}}

    return {
        "from": from_,
        "size": size,
        "_source": {"excludes": ["embedding_vector", "embedding_text"]},
        "query": {
            "hybrid": {
                "queries": [bm25_sub, knn_sub],
            }
        },
    }


# ── Response parsing ──────────────────────────────────────────────────────────

def _parse_response(raw: dict, mode: SearchMode) -> SearchResponse:
    """Convert a raw OpenSearch response dict into a SearchResponse."""
    total_value = raw.get("hits", {}).get("total", {})
    if isinstance(total_value, dict):
        total = total_value.get("value", 0)
    else:
        total = int(total_value or 0)

    hits = [ProductHit.from_hit(h) for h in raw.get("hits", {}).get("hits", [])]
    took_ms = raw.get("took", 0)
    return SearchResponse(total=total, hits=hits, took_ms=took_ms, mode=mode)


# ── Public async search functions ─────────────────────────────────────────────

async def lexical_search(
    client: OpenSearch,
    query: str,
    index_name: str,
    category: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_rating: Optional[float] = None,
    boost_terms: Optional[list[str]] = None,
    size: int = 10,
    from_: int = 0,
) -> SearchResponse:
    """Execute a BM25 lexical search with optional filters and rating boost."""
    filters = _build_filter_clauses(category, min_price, max_price, min_rating)
    body = _build_lexical_query(query, filters, boost_terms, size, from_)

    logger.debug("lexical_search query body: %s", body)
    raw = await asyncio.to_thread(
        client.search, index=index_name, body=body
    )
    return _parse_response(raw, SearchMode.lexical)


async def semantic_search(
    client: OpenSearch,
    query_vector: list[float],
    index_name: str,
    category: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_rating: Optional[float] = None,
    size: int = 10,
    k: int = 100,
) -> SearchResponse:
    """Execute a kNN semantic search using a pre-encoded query vector."""
    filters = _build_filter_clauses(category, min_price, max_price, min_rating)
    body = _build_semantic_query(query_vector, filters, k=k, size=size)

    logger.debug("semantic_search query body: %s", body)
    raw = await asyncio.to_thread(
        client.search, index=index_name, body=body
    )
    return _parse_response(raw, SearchMode.semantic)


async def hybrid_search(
    client: OpenSearch,
    query: str,
    query_vector: list[float],
    index_name: str,
    category: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_rating: Optional[float] = None,
    boost_terms: Optional[list[str]] = None,
    size: int = 10,
    from_: int = 0,
) -> SearchResponse:
    """
    Execute a hybrid BM25+kNN search via the normalization pipeline.

    Falls back to lexical-only if the hybrid query type is unsupported
    by the running OpenSearch version.
    """
    filters = _build_filter_clauses(category, min_price, max_price, min_rating)
    body = _build_hybrid_query(
        query, query_vector, filters, boost_terms, size, from_
    )

    params = {"search_pipeline": HYBRID_PIPELINE_ID}
    logger.debug("hybrid_search query body: %s", body)

    try:
        raw = await asyncio.to_thread(
            client.search,
            index=index_name,
            body=body,
            params=params,
        )
        return _parse_response(raw, SearchMode.hybrid)
    except Exception as exc:
        logger.warning(
            "Hybrid query failed (%s); falling back to lexical search.", exc
        )
        filters2 = _build_filter_clauses(category, min_price, max_price, min_rating)
        body2 = _build_lexical_query(query, filters2, boost_terms, size, from_)
        raw = await asyncio.to_thread(client.search, index=index_name, body=body2)
        return _parse_response(raw, SearchMode.lexical)
