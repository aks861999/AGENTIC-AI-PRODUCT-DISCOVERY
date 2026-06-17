"""
Search Implementation executing native lexical, semantic, and hybrid workflows.
Optimized for dynamic Home Depot EAV attribute filtering.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional
from opensearchpy import OpenSearch
from api.models import ProductHit, SearchMode, SearchResponse
from ingestion.schema import HYBRID_PIPELINE_ID

logger = logging.getLogger(__name__)


def _build_filter_clauses(required_attributes: Optional[dict[str, str]] = None) -> list[dict]:
    """
    Dynamically build nested filters for Home Depot attributes.

    Uses keyword matching on both name and value. The value field
    has a .keyword subfield for exact matching AND text analysis
    for more flexible matching (e.g., "8.5 in" vs "8.5 inch").
    """
    filters: list[dict] = []

    if required_attributes:
        for attr_name, attr_value in required_attributes.items():
            filters.append({
                "nested": {
                    "path": "attributes",
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"attributes.name": attr_name}},
                                {
                                    "bool": {
                                        "should": [
                                            # Try exact keyword match first
                                            {"term": {"attributes.value.keyword": attr_value}},
                                            # Fallback: match on analyzed text (handles "8.5 in" vs "8.5 inch")
                                            {"match": {"attributes.value": attr_value}},
                                        ],
                                        "minimum_should_match": 1
                                    }
                                }
                            ]
                        }
                    }
                }
            })
    return filters

def _build_lexical_query(query: str, filters: list[dict], boost_terms: Optional[list[str]] = None, size: int = 10, from_: int = 0) -> dict:
    should_clauses: list[dict] = [
        {
            "nested": {
                "path": "attributes",
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["attributes.name", "attributes.value"],
                        "type": "best_fields",
                        "tie_breaker": 0.3
                    }
                },
                "score_mode": "max"
            }
        }
    ]

    if boost_terms:
        for term in boost_terms:
            should_clauses.append({"multi_match": {"query": term, "fields": ["title^2", "product_description", "bullet_points"], "boost": 3}})



    base_query: dict = {
        "bool": {
            "must": [
                {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^3", "product_description^2"],
                        "type": "best_fields",
                        "tie_breaker": 0.3,
                        "fuzziness": "AUTO",
                    }
                }
            ],
            "should": should_clauses,
            "filter": filters,
        }
    }

    return {
        "from": from_,
        "size": size,
        "_source": {"excludes": ["embedding_vector"]},
        "query": base_query,
    }

def _build_semantic_query(query_vector: list[float], filters: list[dict], k: int = 100, size: int = 10, from_: int = 0) -> dict:
    body: dict = {
        "from": from_,
        "size": size,
        "_source": {"excludes": ["embedding_vector"]},
        "query": {"knn": {"embedding_vector": {"vector": query_vector, "k": k}}}
    }
    if filters:
        body["query"]["knn"]["embedding_vector"]["filter"] = {"bool": {"filter": filters}}
    return body

def _build_hybrid_query(query: str, query_vector: list[float], filters: list[dict], boost_terms: Optional[list[str]] = None, size: int = 10, from_: int = 0) -> dict:
    should_clauses: list[dict] = [
        {
            "nested": {
                "path": "attributes",
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["attributes.name", "attributes.value"]
                    }
                },
                "score_mode": "max"
            }
        }
    ]

    if boost_terms:
        for term in boost_terms:
            should_clauses.append({"multi_match": {"query": term, "fields": ["title^2", "product_description", "bullet_points"], "boost": 3}})

    bm25_sub: dict = {
        "bool": {
            "must": [
                {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^3", "product_description^2"],
                        "type": "best_fields",
                        "tie_breaker": 0.3,
                    }
                }
            ],
            "should": should_clauses,
            "filter": filters,
        }
    }

    knn_sub: dict = {"knn": {"embedding_vector": {"vector": query_vector, "k": 50}}}
    if filters:
        knn_sub["knn"]["embedding_vector"]["filter"] = {"bool": {"filter": filters}}

    return {
        "from": from_,
        "size": size,
        "_source": {"excludes": ["embedding_vector"]},
        "query": {"hybrid": {"queries": [bm25_sub, knn_sub]}},
    }

def _parse_response(raw: dict, mode: SearchMode) -> SearchResponse:
    total_value = raw.get("hits", {}).get("total", {})
    total = total_value.get("value", 0) if isinstance(total_value, dict) else int(total_value or 0)
    hits = [ProductHit.from_hit(h) for h in raw.get("hits", {}).get("hits", [])]
    return SearchResponse(total=total, hits=hits, took_ms=raw.get("took", 0), mode=mode)

async def lexical_search(client: OpenSearch, query: str, index_name: str, required_attributes: Optional[dict[str, str]] = None, boost_terms: Optional[list[str]] = None, size: int = 10, from_: int = 0) -> SearchResponse:
    filters = _build_filter_clauses(required_attributes)
    body = _build_lexical_query(query, filters, boost_terms, size, from_)
    raw = await asyncio.to_thread(client.search, index=index_name, body=body, request_timeout=30)
    return _parse_response(raw, SearchMode.lexical)

async def semantic_search(client: OpenSearch, query_vector: list[float], index_name: str, required_attributes: Optional[dict[str, str]] = None, size: int = 10, from_: int = 0, k: int = 50) -> SearchResponse:
    filters = _build_filter_clauses(required_attributes)
    body = _build_semantic_query(query_vector, filters, k=k, size=size, from_=from_)
    raw = await asyncio.to_thread(client.search, index=index_name, body=body, request_timeout=25)
    return _parse_response(raw, SearchMode.semantic)

async def hybrid_search(client: OpenSearch, query: str, query_vector: list[float], index_name: str, required_attributes: Optional[dict[str, str]] = None, boost_terms: Optional[list[str]] = None, size: int = 10, from_: int = 0) -> SearchResponse:
    filters = _build_filter_clauses(required_attributes)
    body = _build_hybrid_query(query, query_vector, filters, boost_terms, size, from_)
    params = {"search_pipeline": HYBRID_PIPELINE_ID, "timeout": "15s"}

    try:
        raw = await asyncio.to_thread(
            client.search,
            index=index_name,
            body=body,
            params=params,
            request_timeout=45,
        )
        return _parse_response(raw, SearchMode.hybrid)
    except Exception as exc:
        logger.warning("Hybrid execution failed (%s); reverting to lexical fallback.", exc)
        body2 = _build_lexical_query(query, filters, boost_terms, size, from_)
        raw = await asyncio.to_thread(
            client.search,
            index=index_name,
            body=body2,
            request_timeout=15,
        )
        return _parse_response(raw, SearchMode.lexical)
