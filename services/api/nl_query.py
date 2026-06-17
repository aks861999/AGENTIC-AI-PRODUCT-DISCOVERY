"""
Natural language query chain using LangChain 0.3 LCEL.

Architecture:
    NL query string
        → LLM (Gemini Flash / fallback stub)
        → JSON parse → ParsedNLQuery
        → hybrid_search() with extracted params

The chain is built with LCEL (pipe operator) and is fully async.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import re
import time
from typing import Any, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from api.models import NLFilters, NLSearchResponse, ParsedNLQuery, ProductHit, SearchMode
from api.search import hybrid_search
from shared.config.settings import get_settings

logger = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────

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


# ── JSON extraction helper ────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """
    Robustly extract a JSON object from an LLM response string.

    Handles markdown code fences and extra surrounding text.
    """
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Find first {...} block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse LLM JSON output: %r", text[:200])
    return {}


def _dict_to_parsed_query(data: dict) -> ParsedNLQuery:
    """Convert extracted JSON dict to a ParsedNLQuery model."""
    raw_filters = data.get("filters") or {}
    if not isinstance(raw_filters, dict):
        raw_filters = {}

    # Home Depot: only attribute-based filters (no price, rating, category)
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


# ── LLM chain factory ─────────────────────────────────────────────────────────

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


# ── Main async function ───────────────────────────────────────────────────────

_chain = None  # Module-level cached chain

# Cache for attribute names fetched from candidate products
_attr_name_cache: dict[str, tuple[list[str], float]] = {}
_ATTR_CACHE_TTL_SEC = 300  # 5 minutes


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
        parsed_dict = _extract_json(raw_output)
        return _dict_to_parsed_query(parsed_dict)
    except Exception as exc:
        logger.error("LLM chain failed: %s , falling back to stub parser.", exc)
        return _stub_parse(query)


# ═══════════════════════════════════════════════════════════════════════════════
# Targeted attribute name resolution (from candidate products, not ALL attributes)
# ═══════════════════════════════════════════════════════════════════════════════

async def _get_candidate_attribute_names(
    client,
    index_name: str,
    semantic_query: str,
    model,
) -> list[str]:
    """
    Fetch attribute names from candidate products matching the semantic query.

    Instead of fetching ALL attribute names from the index (naive),
    we do a lightweight semantic search to find products matching the query,
    then extract attribute names only from those products.
    This gives a targeted, relevant subset of attribute names.
    """
    # 1. Encode the semantic query
    query_vector: list[float] = await asyncio.to_thread(
        lambda: model.encode(semantic_query, normalize_embeddings=True).tolist()
    )

    # 2. Do a lightweight kNN search (no filters) to get candidate products
    try:
        knn_query = {
            "size": 20,  # Small number — we just need representative attribute names
            "_source": ["attributes.name"],  # Only fetch attribute names, nothing else
            "query": {
                "knn": {
                    "embedding_vector": {
                        "vector": query_vector,
                        "k": 50
                    }
                }
            }
        }
        raw = await asyncio.to_thread(
            client.search,
            index=index_name,
            body=knn_query,
            request_timeout=10,
        )
    except Exception as exc:
        logger.warning("Failed to fetch candidate attributes: %s", exc)
        return []

    # 3. Extract unique attribute names from candidate products
    attr_names: set[str] = set()
    hits = raw.get("hits", {}).get("hits", [])
    for hit in hits:
        source = hit.get("_source", {})
        for attr in source.get("attributes", []):
            name = attr.get("name", "")
            if name:
                attr_names.add(name)

    logger.debug(
        "Extracted %d unique attribute names from %d candidate products for query '%s'",
        len(attr_names), len(hits), semantic_query
    )
    return sorted(attr_names)


def _find_best_attr_name(extracted: str, actual_names: list[str]) -> str | None:
    """Resolve an LLM-extracted attribute name to the best match from candidates."""
    if not actual_names:
        return None

    # Exact match (case-sensitive)
    if extracted in actual_names:
        return extracted

    # Case-insensitive exact match
    lower_map = {n.lower(): n for n in actual_names}
    if extracted.lower() in lower_map:
        return lower_map[extracted.lower()]

    # Fuzzy match — Ratcliff/Obershelp similarity (cutoff=0.6 for precision)
    matches = difflib.get_close_matches(extracted, actual_names, n=1, cutoff=0.6)
    if matches:
        logger.debug("Fuzzy-matched attribute '%s' -> '%s'", extracted, matches[0])
        return matches[0]

    return None


async def _resolve_attribute_filters(
    client,
    index_name: str,
    semantic_query: str,
    model,
    required_attributes: Optional[dict[str, str]],
) -> Optional[dict[str, str]]:
    """
    Map LLM-extracted attribute names to actual indexed attribute names.

    Uses candidate-product-based resolution instead of fetching ALL attributes.
    Called ONCE in nl_search() — NOT repeated in search.py functions.
    """
    if not required_attributes:
        return None

    # Get relevant attribute names from candidate products
    actual_names = await _get_candidate_attribute_names(
        client, index_name, semantic_query, model
    )
    if not actual_names:
        return required_attributes  # passthrough if we can't fetch candidates

    resolved: dict[str, str] = {}
    for extracted_name, value in required_attributes.items():
        best = _find_best_attr_name(extracted_name, actual_names)
        if best:
            resolved[best] = value
        else:
            resolved[extracted_name] = value  # keep original as fallback

    if resolved != required_attributes:
        logger.info("Resolved filters %s -> %s", required_attributes, resolved)

    return resolved


async def nl_search(
    client,
    model,
    query: str,
    index_name: str,
    size: int = 10,
) -> NLSearchResponse:
    """
    End-to-end natural language search:
        1. Parse NL query → ParsedNLQuery (via LLM or stub)
        2. Encode semantic_query → vector
        3. Resolve attribute names against candidate products
        4. Call hybrid_search() with resolved filters and boost_terms
        5. Return NLSearchResponse

    Args:
        client:     OpenSearch client instance
        model:      SentenceTransformer model instance
        query:      Raw NL query from the user
        index_name: Target OpenSearch index
        size:       Number of results to return
    """
    # Step 1: Parse NL query
    parsed = await parse_nl_query(query)
    logger.info(
        "NL query parsed: semantic_query=%r filters=%s boost_terms=%s",
        parsed.semantic_query,
        parsed.filters,
        parsed.boost_terms,
    )

    # Use original query as fallback if parsing yields nothing
    semantic_q = parsed.semantic_query or query

    # Step 2: Encode query vector
    vector: list[float] = await asyncio.to_thread(
        lambda: model.encode(
            semantic_q,
            normalize_embeddings=True,
        ).tolist()
    )

    # Step 3: Resolve attribute names against candidate products
    # This maps LLM-extracted names (e.g., "Product Depth") to actual indexed names
    # (e.g., "Assembled Depth (in.)") by looking at attributes from similar products
    resolved_attrs = await _resolve_attribute_filters(
        client=client,
        index_name=index_name,
        semantic_query=semantic_q,
        model=model,
        required_attributes=parsed.filters.attributes if parsed.filters.attributes else None,
    )

    # Step 4: Execute hybrid search with resolved attributes
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
