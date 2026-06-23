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

settings = get_settings()

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
- - If the user mentions a specific feature like "sound insulation", "soft close",
  "waterproof", add it to attributes with key "Features"
  e.g., "Features": "sound insulation". Never omit product features.
Other rules:
- semantic_query: strip out attribute/value phrases, keep only the product concept (e.g., "clamps")
- If no attributes are mentioned, use {{}} for attributes
- search_mode is always "hybrid"
"""

USER_PROMPT = "User query: {query}"


import re as _re

def _normalize_attr_value(value: str) -> str:
    """
    Strip natural-language unit suffixes from numeric attribute values.
    Generalizes over the full EAV space: "2 doors" → "2", "30 inches" → "30".
    Non-numeric values (e.g. "Frigidaire", "Frost Free") pass through unchanged.
    """
    if not value or not isinstance(value, str):
        return value
    val = value.strip()
    # Only apply to strings that START with a number
    if not val or not val[0].isdigit():
        return val
    # Strip known unit tokens after the numeric part
    numeric_with_units = _re.compile(
        r'^(\d+(?:\.\d+)?)'           # leading number (int or float)
        r'\s*'
        r'(?:'
        r'cu\.?\s*ft\.?|cubic\s*feet?' # cubic feet
        r'|in(?:ch(?:es)?)?\.?'        # inches / in / in.
        r'|ft\.?|feet?'                # feet
        r'|lb\.?s?|pounds?'            # pounds
        r'|gal(?:lons?)?'              # gallons
        r'|doors?'                     # "2 doors"
        r'|shelves?|shelfs?'           # "3 shelves"
        r'|drawers?'                   # "2 drawers"
        r'|volts?|v\b'                 # voltage
        r'|watts?|w\b'                 # wattage
        r'|amps?|a\b'                  # amperage
        r')?$',
        _re.IGNORECASE,
    )
    m = numeric_with_units.match(val)
    if m and m.group(1) != val:        # only strip if something was actually stripped
        return m.group(1)
    return val



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
    _TOP_K = 5               # how many embedding candidates to pass to value verifier
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
    async def _verify_value(client, index_name: str, attr_name: str, attr_value: str) -> bool:
        normalized = _normalize_attr_value(attr_value)
        values_to_try = list(dict.fromkeys([attr_value, normalized]))  # deduplicated, original first

        for val in values_to_try:
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
                                        "must": [{"term": {"attributes.name": attr_name}}],
                                        "should": [
                                            {"match_phrase": {"attributes.value": {"query": val, "slop": 2}}},
                                            {"fuzzy": {"attributes.value.keyword": {
                                                "value": val, "fuzziness": "AUTO", "prefix_length": 1
                                            }}},
                                        ],
                                        "minimum_should_match": 1,
                                    }
                                }
                            }
                        }
                    },
                    request_timeout=5,
                )
                if resp["hits"]["total"]["value"] > 0:
                    return True
            except Exception as exc:
                logger.warning("Value verification failed: %s", exc)
                return True  # fail-open on errors

        return False

    # ── Public resolve method ─────────────────────────────────────────────────


    async def resolve_one(
        self,
        client,
        index_name: str,
        model,
        raw_key: str,
    ) -> tuple[str, float] | None:
        """
        Resolve a single raw attribute key → (canonical_name, cosine_score).
        Returns None if no match above threshold.
        """
        await self._refresh_cache(client, index_name, model)
        if self._schema_keys is None:
            return None

        raw_vec = await asyncio.to_thread(
            lambda: model.encode(raw_key, normalize_embeddings=True)
        )
        sims = np.dot(self._schema_vecs, raw_vec)  # (N,)
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score < self.sim_threshold:
            return (self._schema_keys[best_idx], best_score)  # returned even if below threshold so frontend can show the warning

        canonical = self._schema_keys[best_idx]
        logger.info("SchemaLinker resolve_one: %r → %r (%.3f)", raw_key, canonical, best_score)
        return (canonical, best_score)

    async def resolve(self, client, index_name, model, extracted_name, extracted_value):
        await self._ensure_cache(client, index_name, model)
        if self._canonical_embeddings is None or not self._canonical_names:
            return extracted_name  # fail-open: return original, never guess

        stage1_rejected: set[str] = set()
        lower_extracted = extracted_name.lower().strip()
        normalized_value = _normalize_attr_value(extracted_value)

        # ── Stage 1: Exact match + value verification ─────────────────────────
        for canonical in self._canonical_names:
            if canonical.lower() == lower_extracted:
                if await self._verify_value(client, index_name, canonical, normalized_value):
                    logger.info("SchemaLinker exact+verified: '%s' → '%s'", extracted_name, canonical)
                    return canonical
                stage1_rejected.add(canonical)
                break  # only one exact match possible

        # ── Stage 2: Substring containment + value verification ──────────────
        # "Brand" is contained in "MFG Brand Name" → prioritize over embedding
        # Guard: only consider substrings ≥ 3 chars to avoid "in", "or" noise
        if len(lower_extracted) >= 3:
            substr_candidates = []
            for canonical in self._canonical_names:
                if canonical in stage1_rejected:
                    continue
                c_lower = canonical.lower()
                if lower_extracted in c_lower:
                    # Score: how much of the canonical name is the extracted name
                    score = len(lower_extracted) / len(c_lower)
                    substr_candidates.append((canonical, score))
            substr_candidates.sort(key=lambda x: x[1], reverse=True)

            for candidate_name, _ in substr_candidates:
                if await self._verify_value(client, index_name, candidate_name, normalized_value):
                    logger.info("SchemaLinker substring+verified: '%s' → '%s'", extracted_name, candidate_name)
                    return candidate_name

        # ── Stage 3: Embedding similarity + value verification ───────────────
        query_vec = np.array(
            await asyncio.to_thread(lambda: model.encode(extracted_name, normalize_embeddings=True))
        )
        scores = self._canonical_embeddings @ query_vec
        top_k_indices = scores.argsort()[-self._TOP_K:][::-1]
        candidates = [
            (self._canonical_names[i], float(scores[i]))
            for i in top_k_indices
            if float(scores[i]) >= self._EMBED_THRESHOLD
            and self._canonical_names[i] not in stage1_rejected
        ]

        for candidate_name, embed_score in candidates:
            if await self._verify_value(client, index_name, candidate_name, normalized_value):
                logger.info("SchemaLinker embed+verified: '%s' → '%s' (%.3f)", extracted_name, candidate_name, embed_score)
                return candidate_name

        # ── Stage 4: Fail-open — never return an unverified embedding guess ──
        # An unverified guess guarantees 0 results. The original name at least
        # has a chance of matching if the LLM happened to use the right name.
        logger.warning(
            "SchemaLinker: no verified match for '%s'. Failing open to original name.",
            extracted_name,
        )
        return extracted_name

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
        search_mode=mode,
    )


_attr_name_cache: dict[str, tuple[list[str], float]] = {}
_ATTR_CACHE_TTL_SEC = 300  # 5 minutes


# ── LLM factory — Gemini primary, Groq fallback ──────────────────────────────

def _build_gemini(settings):
    """Try to build a Gemini LLM. Returns llm or None."""
    if not settings.google_api_key:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            google_api_key=settings.google_api_key,
            max_retries=settings.max_retries, 
        )
        logger.info("NL chain: using Google Gemini (%s)", settings.llm_model)
        return llm
    except Exception as exc:
        logger.warning("Failed to init Gemini LLM: %s", exc)
        return None


def _build_groq(settings):
    """Try to build a Groq LLM. Returns llm or None."""
    if not settings.groq_api_key:
        return None
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=settings.groq_model,
            temperature=settings.llm_temperature,
            openai_api_key=settings.groq_api_key,
            openai_api_base="https://api.groq.com/openai/v1",
        )
        logger.info("NL chain: using Groq (%s)", settings.groq_model)
        return llm
    except Exception as exc:
        logger.warning("Failed to init Groq LLM: %s", exc)
        return None

def _build_llm_chain():
    """Build and return a LangChain LCEL chain for query parsing.
    Escalation order: Gemini → Groq → stub (None).
    """
    settings = get_settings()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human",  USER_PROMPT),
    ])

    llm = _build_gemini(settings) or _build_groq(settings)
    if llm is not None:
        return prompt | llm | StrOutputParser()

    logger.info("NL chain: using stub parser (no API key configured).")
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

    return ParsedNLQuery(
        semantic_query=semantic or query,
        filters=NLFilters(attributes=attr_filters),
        search_mode=SearchMode.hybrid,
    )

async def parse_nl_query(query: str, client=None, index_name=None, model=None) -> ParsedNLQuery:
    settings = get_settings()

    # ── NEW: Inject top candidate schema names into the prompt ──────────────
    schema_hint = ""
    if client and index_name and model and _schema_linker._canonical_names:
        await _schema_linker._ensure_cache(client, index_name, model)
        if _schema_linker._canonical_embeddings is not None:
            q_vec = await asyncio.to_thread(
                lambda: model.encode(query, normalize_embeddings=True)
            )
            scores = _schema_linker._canonical_embeddings @ q_vec
            top_indices = scores.argsort()[-20:][::-1]
            top_names = [_schema_linker._canonical_names[i] for i in top_indices]
            schema_hint = (
                "\n\nKNOWN ATTRIBUTE NAMES IN THIS CATALOG (use these exact strings):\n"
                + "\n".join(f"  - {n}" for n in top_names)
                + "\n\nWhen the user mentions a brand, check this list for the exact brand attribute name."
                + " When the user mentions width or doors, use the exact name from this list."
            )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT + schema_hint),
        ("human", USER_PROMPT),
    ])

    llm_factories = []
    if settings.google_api_key:
        llm_factories.append(("gemini", lambda: _build_gemini(settings)))
    if settings.groq_api_key:
        llm_factories.append(("groq",   lambda: _build_groq(settings)))

    for name, factory in llm_factories:
        try:
            chain      = prompt | factory() | StrOutputParser()
            raw_output = str(await chain.ainvoke({"query": query}))
            logger.info("NL chain: response from %s", name)
            parsed_dict = _extract_json(raw_output)
            logger.info(
                "[DEBUG][parse_nl_query] FINAL ParsedNLQuery => semantic_query=%r | attributes=%s | mode=%s",
                parsed_dict.get("semantic_query"),
                parsed_dict.get("filters", {}).get("attributes", {}),
                parsed_dict.get("search_mode"),
            )
            return _dict_to_parsed_query(parsed_dict)
        except Exception as exc:
            logger.warning("NL chain %s failed: %s — trying next provider.", name, exc)

    logger.warning("All LLM providers failed — falling back to stub parser.")
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
    parsed = await parse_nl_query(query,client=client, index_name=settings.index_name, model=model,
)
    logger.info(
        "NL query parsed: semantic_query=%r filters=%s",
        parsed.semantic_query, parsed.filters,
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
        size=size,
    )
    return NLSearchResponse(
        total=result.total,
        hits=result.hits,
        took_ms=result.took_ms,
        parsed_query=parsed,
    )