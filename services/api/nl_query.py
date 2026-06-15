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
import json
import logging
import re
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

Rules:
- semantic_query must be a clean, descriptive phrase (not the raw user input)
- Extract product attributes from phrases like "12 gauge steel", "galvanized", "3 inch width"
  and map them to the attributes filter. Common Home Depot attributes: Gauge, Material,
  Product Width, Product Height, Product Depth, MFG Brand Name.
- boost_terms are important qualitative keywords from the query (e.g. "galvanized", "heavy-duty")
- If no attributes are mentioned, use an empty object {{}} for attributes
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


def get_nl_chain():
    global _chain
    if _chain is None:
        _chain = _build_llm_chain()
    return _chain


def _stub_parse(query: str) -> ParsedNLQuery:
    """Minimal offline query parser for Home Depot hardware products."""
    import re

    text = query.lower()
    attr_filters: dict[str, str] = {}

    # Attribute extraction for hardware products
    # Gauge extraction: "12 gauge", "12-gauge"
    m = re.search(r"(\d+)(?:\s*-?\s*gauge)\b", text)
    if m:
        attr_filters["Gauge"] = m.group(1)

    # Material extraction: "galvanized steel", "stainless steel", "zinc coated"
    m = re.search(r"(galvanized|stainless|zinc\s*coated|powder\s*coated)\s+steel", text)
    if m:
        attr_filters["Material"] = f"{m.group(1).title()} Steel"

    # Dimension extraction: "3 inch", "3 in.", "1-1/2 in"
    m = re.search(r"(\d+(?:[\-\s/]*\d+)*)\s*(?:inch|in\.?)\s*(?:x\s*(\d+(?:[\-\s/]*\d+)*)\s*(?:inch|in\.?))?\s*(?:x\s*(\d+(?:[\-\s/]*\d+)*)\s*(?:inch|in\.?))?", text)
    if m:
        # Simplified: just capture width hint
        attr_filters["Product Width (in.)"] = m.group(1)

    # Brand extraction
    m = re.search(r"(simpson strong-tie|everbilt|husky|ryobi|dewalt)", text, re.IGNORECASE)
    if m:
        attr_filters["MFG Brand Name"] = m.group(1).title()

    # Strip attribute phrases to get semantic core
    semantic = re.sub(r"(\d+\s*-?\s*gauge)", "", query, flags=re.IGNORECASE)
    semantic = re.sub(r"(galvanized|stainless|zinc\s*coated)\s+steel", "", semantic, flags=re.IGNORECASE)
    semantic = re.sub(r"(\d+(?:[\-\s/]*\d+)*)\s*(?:inch|in\.?)\s*(?:x\s*\d+.*?inch)?", "", semantic, flags=re.IGNORECASE)
    semantic = re.sub(r"\s+", " ", semantic).strip()

    # Boost terms: hardware-specific quality descriptors
    boost_pattern = r"\b(galvanized|heavy-duty|structural|outdoor|corrosion-resistant|zmax|strong-drive)\b"
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
        3. Call hybrid_search() with extracted filters and boost_terms
        4. Return NLSearchResponse

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

    # Step 3: Execute hybrid search
    result = await hybrid_search(
            client=client,
            query=semantic_q,
            query_vector=vector,
            index_name=index_name,
            required_attributes=parsed.filters.attributes if parsed.filters.attributes else None,
            boost_terms=parsed.boost_terms if parsed.boost_terms else None,
            size=size,
        )

    return NLSearchResponse(
        total=result.total,
        hits=result.hits,
        took_ms=result.took_ms,
        parsed_query=parsed,
    )
