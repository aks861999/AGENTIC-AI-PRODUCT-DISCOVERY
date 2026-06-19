"""
Natural language query chain using LangChain 0.3 LCEL.

Architecture:
    NL query string
        → LLM (Gemini Flash / fallback stub)
        → JSON parse → ParsedNLQuery
        → SchemaLinker.resolve() — attribute name resolution via embedding cosine sim
          + value verification via direct OpenSearch aggregation query
        → hybrid_search() with resolved params
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Optional

import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from api.models import NLFilters, NLSearchResponse, ParsedNLQuery, ProductHit, SearchMode
from api.search import hybrid_search
from shared.config.settings import get_settings

import os
from pathlib import Path

SCHEMA_CACHE_PATH = Path(__file__).resolve().parent / "schema_cache.npz"

logger = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────
# (unchanged — keep your existing SYSTEM_PROMPT and USER_PROMPT)

SYSTEM_PROMPT = """You are a product search query parser for a home improvement product catalog.
Your task is to extract structured search parameters from the user's natural language query.

Output ONLY valid JSON matching this exact schema — no markdown, no explanation:
{{
  "semantic_query": "<string: the core product description to search for>",
  "filters": {{
    "attributes": {{"<AttributeName>": "<AttributeValue>"}}
  }},
  "boost_terms": ["<keyword1>", "<keyword2>"],
  "search_mode": "hybrid"
}}

CRITICAL RULES FOR ATTRIBUTES:
- Preserve the EXACT attribute names and values as they appear in the user's query.
  Example: user says "clamp with Assembled Depth (in.) 8.5 in" -> key MUST be "Assembled Depth (in.)", value MUST be "8.5 in".
  Example: user says "Clamp Type Pipe" -> key is "Clamp Type", value is "Pipe".
- Do NOT rename, generalize, or normalize attribute names. If the user wrote "Assembled Depth (in.)",
  do NOT change it to "Product Depth" or "Depth" — use it EXACTLY.
- Only infer an attribute name if the user did NOT specify it explicitly.
  Example: "8.5 inch deep clamp" -> you may infer "Assembled Depth (in.)": "8.5 in".
- Preserve units in the value (e.g., "8.5 in", "600 lb", "12 gauge").
- Use the EXACT spelling, capitalization, and punctuation from the query for attribute keys.

Other rules:
- semantic_query: strip out attribute/value phrases, keep only the product concept (e.g., "clamps")
- boost_terms: quality keywords like "galvanized", "heavy-duty", "structural"
- If no attributes are mentioned, use {{}} for attributes
- search_mode is always "hybrid"
"""

USER_PROMPT = "User query: {query}"



# ── Schema Linker ─────────────────────────────────────────────────────────────

class SchemaLinker:
    """
    Resolves LLM-extracted attribute name mentions to canonical indexed names.

    Implements the Text-to-SQL community standard for schema linking:
    1. Embedding cosine similarity — semantic matching over all indexed attr names
    2. Value verification — direct OpenSearch query confirms the resolved name
    actually carries the requested value (eliminates false positives)
        
    The canonical_names + canonical_embeddings are populated once at startup
    via a terms aggregation and cached in memory. TTL-based refresh prevents
    stale schema after re-indexing.
    """

    _CACHE_TTL_SEC = 600   # refresh schema cache every 10 minutes
    _EMBED_THRESHOLD = 0.65  # minimum cosine similarity to accept a candidate
    _TOP_K = 3               # how many embedding candidates to pass to value verifier
    _AGG_SIZE = 5000

    def __init__(self) -> None:
        self._canonical_names: list[str] = []
        self._canonical_embeddings: Optional[np.ndarray] = None
        self._cache_ts: float = 0.0
        self._lock = asyncio.Lock()

    # ── Cache management ──────────────────────────────────────────────────────

    async def _ensure_cache(self, client, index_name: str, model) -> None:
        """Populate or refresh the attribute name cache if stale."""
        now = time.monotonic()
        if self._canonical_embeddings is not None and (now - self._cache_ts) < self._CACHE_TTL_SEC:
            return

        async with self._lock:
            # Double-check after acquiring lock
            if self._canonical_embeddings is not None and (now - self._cache_ts) < self._CACHE_TTL_SEC:
                return

            await self._refresh_cache(client, index_name, model)

    async def _refresh_cache(self, client, index_name: str, model) -> None:
        # ── Fast path: load from disk if available ───────────────────────────
        if SCHEMA_CACHE_PATH.exists():
            try:
                data = np.load(SCHEMA_CACHE_PATH, allow_pickle=True)
                names = list(data["names"])
                embeddings = data["embeddings"].astype(np.float32)
                if names and embeddings.shape[0] == len(names):
                    self._canonical_names = names
                    self._canonical_embeddings = embeddings
                    self._cache_ts = time.monotonic()
                    logger.info(
                        "SchemaLinker loaded %d attribute names from disk cache: %s",
                        len(names), SCHEMA_CACHE_PATH,
                    )
                    return
                else:
                    logger.warning("SchemaLinker disk cache malformed, falling back to OpenSearch.")
            except Exception as exc:
                logger.warning("SchemaLinker disk cache load failed: %s — falling back to OpenSearch.", exc)

        
        try:
            raw = await asyncio.to_thread(
                client.search,
                index=index_name,
                body={
                    "size": 0,
                    "aggs": {
                        "attr_names": {
                            "nested": {"path": "attributes"},
                            "aggs": {
                                "names": {
                                    "terms": {
                                        "field": "attributes.name",
                                        "size": self._AGG_SIZE,
                                        "order": {"_count": "desc"},
                                    }
                                }
                            },
                        }
                    },
                },
                request_timeout=15,
            )
            buckets = (
                raw.get("aggregations", {})
                .get("attr_names", {})
                .get("names", {})
                .get("buckets", [])
            )
            names = [b["key"] for b in buckets if b.get("key")]


            # ── Guard: warn if aggregation was truncated ──────────────────────────
            if len(names) >= self._AGG_SIZE:
                logger.warning(
                    "SchemaLinker: aggregation returned exactly %d names — this equals "
                    "_AGG_SIZE=%d, meaning the vocabulary may be truncated. "
                    "Increase _AGG_SIZE in SchemaLinker.",
                    len(names), self._AGG_SIZE,
                )
            else:
                logger.info(
                    "SchemaLinker cache refreshed: %d canonical attribute names indexed "
                    "(well below cap of %d — no truncation).",
                    len(names), self._AGG_SIZE,
                )

            if not names:
                logger.warning("SchemaLinker: terms aggregation returned no attribute names.")
                return

            # Embed all canonical names in one batch — cheap (50-200 names)
            embeddings: np.ndarray = await asyncio.to_thread(
                lambda: model.encode(names, normalize_embeddings=True, show_progress_bar=False)
            )

            self._canonical_names = names
            self._canonical_embeddings = embeddings
            self._cache_ts = time.monotonic()

            logger.info(
                "SchemaLinker cache refreshed: %d canonical attribute names indexed.",
                len(names),
            )

        except Exception as exc:
            logger.warning("SchemaLinker cache refresh failed: %s", exc)

    # ── Value verification ────────────────────────────────────────────────────

    @staticmethod
    async def _verify_value(
        client, index_name: str, attr_name: str, attr_value: str
    ) -> bool:
        """
        Direct OpenSearch query: does attr_name carry attr_value anywhere
        in the index? Uses a nested bool query with a match (not term) on value
        so it handles partial strings and analysis (e.g., "Moth Gray" vs "moth gray").
        One query, deterministic result — not dependent on KNN sampling.
        """
        try:
            resp = await asyncio.to_thread(
                client.search,
                index=index_name,
                body={
                    "size": 0,
                    "query": {
                        "nested": {
                            "path": "attributes",
                            "query": {
                                "bool": {
                                    "must": [
                                        {"term": {"attributes.name": attr_name}},
                                        {"match": {"attributes.value": attr_value}},
                                    ]
                                }
                            },
                        }
                    },
                },
                request_timeout=5,
            )
            count = resp.get("hits", {}).get("total", {}).get("value", 0)
            logger.debug(
                "Value verify: attr='%s' value='%s' → %d hits", attr_name, attr_value, count
            )
            return count > 0
        except Exception as exc:
            logger.warning("Value verification query failed: %s", exc)
            return True  # Fail open — don't block the filter if verification errors

    # ── Public resolve method ─────────────────────────────────────────────────

    async def resolve(
        self,
        client,
        index_name: str,
        model,
        extracted_name: str,
        extracted_value: str,
    ) -> str:
        """
        Resolve an LLM-extracted attribute name to its canonical indexed name.

        Resolution order:
        1. Exact match against cached canonical names (case-insensitive)
        2. Embedding cosine similarity → top-K candidates
        3. Value verification for each candidate → return first confirmed
        4. Fallback: return best embedding candidate even without value confirmation
        5. Final fallback: return original extracted name unchanged
        
        Args:
            client:         OpenSearch client
            index_name:     Index to search against
            model:          SentenceTransformer model (same instance used for indexing)
            extracted_name: The attribute name the LLM extracted (e.g., "Color")
            extracted_value: The attribute value the LLM extracted (e.g., "Moth Gray")

        Returns:
            The canonical indexed attribute name (e.g., "Color/Finish")
        """
        await self._ensure_cache(client, index_name, model)

        # ── Stage 1: Exact match (case-insensitive) ──────────────────────────
        lower_extracted = extracted_name.lower().strip()
        for canonical in self._canonical_names:
            if canonical.lower() == lower_extracted:
                logger.debug("SchemaLinker exact match: '%s' → '%s'", extracted_name, canonical)
                return canonical


        # ── Stage 2: Embedding cosine similarity ─────────────────────────────
        if self._canonical_embeddings is None or not self._canonical_names:
            logger.warning("SchemaLinker: no cache available, returning extracted name unchanged.")
            return extracted_name

        query_vec: np.ndarray = await asyncio.to_thread(
            lambda: model.encode([extracted_name], normalize_embeddings=True)[0]
        )
        # Cosine similarity = dot product (embeddings are L2-normalised)
        scores: np.ndarray = self._canonical_embeddings @ query_vec  # shape: (N,)

        # Get top-K candidates above threshold
        top_k_indices = scores.argsort()[::-1][: self._TOP_K]
        candidates: list[tuple[str, float]] = [
            (self._canonical_names[i], float(scores[i]))
            for i in top_k_indices
            if float(scores[i]) >= self._EMBED_THRESHOLD
        ]

        if not candidates:
            logger.info(
                "SchemaLinker: no embedding candidate above threshold %.2f for '%s' "
                "(best score: %.3f for '%s'). Returning extracted name.",
                self._EMBED_THRESHOLD,
                extracted_name,
                float(scores[top_k_indices[0]]) if len(top_k_indices) > 0 else 0.0,
                self._canonical_names[top_k_indices[0]] if len(top_k_indices) > 0 else "N/A",
            )
            return extracted_name

        logger.debug(
            "SchemaLinker embedding candidates for '%s': %s",
            extracted_name,
            [(name, f"{score:.3f}") for name, score in candidates],
        )

        # ── Stage 3: Value verification ───────────────────────────────────────
        for candidate_name, embed_score in candidates:
            confirmed = await self._verify_value(client, index_name, candidate_name, extracted_value)
            if confirmed:
                logger.info(
                    "SchemaLinker resolved: '%s' → '%s' (embed_score=%.3f, value_verified=True)",
                    extracted_name,
                    candidate_name,
                    embed_score,
                )
                return candidate_name

        # ── Stage 4: No value confirmed — trust best embedding candidate ──────
        best_name, best_score = candidates[0]
        logger.info(
            "SchemaLinker: value verification failed for all candidates of '%s'. "
            "Using best embedding match '%s' (score=%.3f) unverified.",
            extracted_name,
            best_name,
            best_score,
        )
        return best_name

    async def resolve_all(
        self,
        client,
        index_name: str,
        model,
        required_attributes: dict[str, str],
    ) -> dict[str, str]:
        """Resolve all extracted attribute names in parallel."""
        if not required_attributes:
            return {}

        # Ensure cache is warm before firing parallel resolution
        await self._ensure_cache(client, index_name, model)

        tasks = {
            extracted_name: self.resolve(
                client, index_name, model, extracted_name, value
            )
            for extracted_name, value in required_attributes.items()
        }

        resolved: dict[str, str] = {}
        for extracted_name, coro in tasks.items():
            resolved_name = await coro
            resolved[resolved_name] = required_attributes[extracted_name]

        if resolved != required_attributes:
            logger.info("SchemaLinker resolved all: %s → %s", required_attributes, resolved)

        return resolved


# ── Module-level singleton ────────────────────────────────────────────────────
_schema_linker = SchemaLinker()


# ── Keep your existing JSON extraction helpers ─────────────────────────────────

def _extract_json(text: str) -> dict:
    cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    logger.warning("Could not parse LLM JSON output: %r", text[:200])
    return {}


def _dict_to_parsed_query(data: dict) -> ParsedNLQuery:
    raw_filters = data.get("filters") or {}
    if not isinstance(raw_filters, dict):
        raw_filters = {}
    attr_filters = raw_filters.get("attributes") or {}
    if not isinstance(attr_filters, dict):
        attr_filters = {}
    filters = NLFilters(attributes=attr_filters)
    mode_str = data.get("search_mode", "hybrid")
    try:
        mode = SearchMode(mode_str)
    except ValueError:
        mode = SearchMode.hybrid
    return ParsedNLQuery(
        semantic_query=data.get("semantic_query", ""),
        filters=filters,
        boost_terms=data.get("boost_terms") or [],
        search_mode=mode,
    )


# ── LLM chain ─────────────────────────────────────────────────────────────────
# (keep your existing _build_llm_chain, get_nl_chain, _stub_parse, parse_nl_query unchanged)
# ...

_attr_name_cache: dict[str, tuple[list[str], float]] = {}
_ATTR_CACHE_TTL_SEC = 300  # 5 minutes



def _build_llm_chain():
    """
    Build and return a LangChain LCEL chain for query parsing.

    Attempts to use Google Gemini if GOOGLE_API_KEY is set,
    falls back to a lightweight stub chain for offline/test use.
    """
    settings = get_settings()
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    )

    if settings.google_api_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                google_api_key=settings.google_api_key,
            )
            logger.info("NL chain: using Google Gemini (%s)", settings.llm_model)
            return prompt | llm | StrOutputParser()
        except Exception as exc:
            logger.warning("Failed to init Gemini LLM: %s , using stub.", exc)

    # Stub chain: echo-based minimal parser for offline testing
    logger.info("NL chain: using stub parser (no GOOGLE_API_KEY set).")
    return None



_chain = None

def get_nl_chain():
    global _chain
    if _chain is None:
        _chain = _build_llm_chain()
    return _chain


def _stub_parse(query: str) -> ParsedNLQuery:
    """Minimal offline query parser — preserves exact attribute names from user query."""
    import re

    text = query.lower()
    attr_filters: dict[str, str] = {}

    # ── Exact attribute pattern: "AttrName value" or "AttrName: value" ──
    # Matches patterns like "Assembled Depth (in.) 8.5 in", "Clamp Type Pipe"
    # This captures the EXACT attribute name the user typed
    exact_attr_pattern = r'([A-Z][A-Za-z\s()]+?)\s+([A-Za-z0-9][A-Za-z0-9\s./()-]*)'
    for match in re.finditer(exact_attr_pattern, query):
        attr_name = match.group(1).strip()
        attr_value = match.group(2).strip()
        # Validate: attribute name should look real (contains letters, reasonable length)
        if len(attr_name) >= 3 and len(attr_name) <= 60 and len(attr_value) >= 1:
            attr_filters[attr_name] = attr_value

    # ── Fallback: dimension extraction with INFERRED attribute names ──
    # Only runs if no exact attributes were found above
    if not attr_filters:
        # "X inch deep" / "X in. deep" -> infer "Assembled Depth (in.)"
        m = re.search(r'(\d+(?:[\-\s/]*\d+)*)\s*(?:inch|in\.?)\s*(?:deep|depth)', text)
        if m:
            attr_filters["Assembled Depth (in.)"] = f"{m.group(1)} in"

        # "X inch wide" / "X in. wide" -> infer "Assembled Width (in.)"
        m = re.search(r'(\d+(?:[\-\s/]*\d+)*)\s*(?:inch|in\.?)\s*(?:wide|width)', text)
        if m:
            attr_filters["Assembled Width (in.)"] = f"{m.group(1)} in"

        # "X inch tall" / "X in. tall" / "X inch high" -> infer "Assembled Height (in.)"
        m = re.search(r'(\d+(?:[\-\s/]*\d+)*)\s*(?:inch|in\.?)\s*(?:tall|high|height)', text)
        if m:
            attr_filters["Assembled Height (in.)"] = f"{m.group(1)} in"

        # "X gauge" -> infer "Gauge"
        m = re.search(r'(\d+)\s*-?\s*gauge\b', text)
        if m:
            attr_filters["Gauge"] = m.group(1)

        # "stainless steel", "galvanized steel" -> infer "Material"
        m = re.search(r'(galvanized|stainless|zinc\s*coated|powder\s*coated)\s+steel', text)
        if m:
            attr_filters["Material"] = f"{m.group(1).title()} Steel"

    # ── Strip all attribute phrases to get semantic core ──
    semantic = query
    # Remove "AttrName value" patterns
    semantic = re.sub(r'[A-Z][A-Za-z\s()]+?\s+[A-Za-z0-9][A-Za-z0-9\s./()-]*', '', semantic)
    # Remove dimension phrases
    semantic = re.sub(r'(\d+(?:[\-\s/]*\d+)*)\s*(?:inch|in\.?)\s*(?:deep|depth|wide|width|tall|high|height)?', '', semantic, flags=re.IGNORECASE)
    semantic = re.sub(r'(\d+)\s*-?\s*gauge\b', '', semantic, flags=re.IGNORECASE)
    semantic = re.sub(r'(galvanized|stainless|zinc\s*coated)\s+steel', '', semantic, flags=re.IGNORECASE)
    semantic = re.sub(r'\s+', ' ', semantic).strip()

    # Boost terms: quality descriptors
    boost_pattern = r'\b(galvanized|heavy-duty|structural|outdoor|corrosion-resistant|zmax|strong-drive)\b'
    boost_terms = re.findall(boost_pattern, query, re.IGNORECASE)

    return ParsedNLQuery(
        semantic_query=semantic or query,
        filters=NLFilters(attributes=attr_filters),
        boost_terms=boost_terms,
        search_mode=SearchMode.hybrid,
    )


async def parse_nl_query(query: str) -> ParsedNLQuery:
    """
    Parse a natural language query into a structured ParsedNLQuery.

    Uses LLM if available, otherwise falls back to stub parser.
    """
    chain = get_nl_chain()
    if chain is None:
        return _stub_parse(query)

    try:
        raw_output: str = await chain.ainvoke({"query": query})


        logger.debug(
            "[DEBUG][parse_nl_query] Raw LLM output: %r",
            raw_output
        )



        parsed_dict = _extract_json(raw_output)

        logger.debug(
            "[DEBUG][parse_nl_query] Extracted JSON dict: %s",
            parsed_dict
        )


        logger.info(
            "[DEBUG][parse_nl_query] FINAL ParsedNLQuery => semantic_query=%r | attributes=%s | boost_terms=%s | mode=%s",
            parsed_dict.get("semantic_query"),
            parsed_dict.get("filters", {}).get("attributes", {}),
            parsed_dict.get("boost_terms"),
            parsed_dict.get("search_mode"),
        )




        return _dict_to_parsed_query(parsed_dict)
    except Exception as exc:
        logger.error("LLM chain failed: %s , falling back to stub parser.", exc)
        return _stub_parse(query)



# ── Public _resolve_attribute_filters (now delegates to SchemaLinker) ─────────

async def _resolve_attribute_filters(
    client,
    index_name: str,
    semantic_query: str,
    model,
    required_attributes: Optional[dict[str, str]],
) -> Optional[dict[str, str]]:
    """
    Replaces the old KNN-product-sampling based resolver.
    Now delegates to SchemaLinker which operates over the schema vocabulary
    directly, not over product documents.
    """
    if not required_attributes:
        return None
    return await _schema_linker.resolve_all(client, index_name, model, required_attributes)


# ── nl_search (keep your existing implementation unchanged) ──────────────────
async def nl_search(
    client,
    model,
    query: str,
    index_name: str,
    size: int = 10,
) -> NLSearchResponse:
    parsed = await parse_nl_query(query)
    logger.info(
        "NL query parsed: semantic_query=%r filters=%s boost_terms=%s",
        parsed.semantic_query, parsed.filters, parsed.boost_terms,
    )
    semantic_q = parsed.semantic_query or query
    vector: list[float] = await asyncio.to_thread(
        lambda: model.encode(semantic_q, normalize_embeddings=True).tolist()
    )
    resolved_attrs = await _resolve_attribute_filters(
        client=client,
        index_name=index_name,
        semantic_query=semantic_q,
        model=model,
        required_attributes=parsed.filters.attributes if parsed.filters.attributes else None,
    )
    result = await hybrid_search(
        client=client,
        query=semantic_q,
        query_vector=vector,
        index_name=index_name,
        required_attributes=resolved_attrs,
        boost_terms=parsed.boost_terms if parsed.boost_terms else None,
        size=size,
    )
    return NLSearchResponse(
        total=result.total,
        hits=result.hits,
        took_ms=result.took_ms,
        parsed_query=parsed,
    )