"""
Pydantic v2 models for all FastAPI request/response contracts.

All models use model_config = ConfigDict(from_attributes=True)
as required by the project spec.
"""

from __future__ import annotations

import uuid as _uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ── Enumerations ──────────────────────────────────────────────────────────────

class SearchMode(str, Enum):
    lexical  = "lexical"
    semantic = "semantic"
    hybrid   = "hybrid"

# ── Request models ────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    q:          str            = Field(..., min_length=1, max_length=500, description="Search query text")
    category:   Optional[str]  = Field(default=None, description="Filter by main_category")
    min_price:  Optional[float]= Field(default=None, ge=0, description="Minimum price filter")
    max_price:  Optional[float]= Field(default=None, ge=0, description="Maximum price filter")
    min_rating: Optional[float]= Field(default=None, ge=0, le=5, description="Minimum average_rating")
    mode:       SearchMode     = Field(default=SearchMode.hybrid, description="Search strategy")
    size:       int            = Field(default=10, ge=1, le=100, description="Number of results to return")
    from_:      int            = Field(default=0, ge=0, alias="from", description="Pagination offset")

    @field_validator("max_price")
    @classmethod
    def max_price_gte_min(cls, v: Optional[float], info) -> Optional[float]:
        min_p = info.data.get("min_price")
        if v is not None and min_p is not None and v < min_p:
            raise ValueError("max_price must be >= min_price")
        return v


class NLSearchRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    query: str = Field(..., min_length=3, max_length=1000, description="Natural language search query")
    size:  int = Field(default=10, ge=1, le=100, description="Number of results to return")


class SuggestRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    prefix: str = Field(..., min_length=1, max_length=100, description="Autocomplete prefix")
    size:   int = Field(default=5, ge=1, le=20, description="Number of suggestions")


# ── NEW: Chat models ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    message:    str            = Field(..., min_length=1, max_length=2000, description="User message")
    session_id: Optional[str]  = Field(
        default_factory=lambda: str(_uuid.uuid4()),
        description="Chat session ID — generate once, reuse across turns",
    )


class ChatResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    session_id: str
    reply:      str
    results:    list[dict] = Field(default_factory=list, description="Search results if a search was triggered")


# ── Product document model ────────────────────────────────────────────────────

class ProductHit(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:               str            = Field(..., description="Document ID (parent_asin)")
    score:            float          = Field(..., description="Relevance score from OpenSearch")
    title:            str            = Field(default="", description="Product title")
    description:      str            = Field(default="", description="Product description")
    features:         str            = Field(default="", description="Product features")
    main_category:    str            = Field(default="", description="Main product category")
    sub_category:     str            = Field(default="", description="Sub category")
    store:            str            = Field(default="", description="Seller / store name")
    price:            Optional[float]= Field(default=None, description="Product price in USD")
    average_rating:   Optional[float]= Field(default=None, description="Average customer rating")
    rating_number:    Optional[int]  = Field(default=None, description="Number of ratings")
    primary_image_url:str            = Field(default="", description="Primary product image URL")
    parent_asin:      str            = Field(default="", description="Amazon parent ASIN")

    @classmethod
    def from_hit(cls, hit: dict) -> "ProductHit":
        """Construct a ProductHit from a raw OpenSearch hit dict."""
        source = hit.get("_source", {})
        return cls(
            id=hit["_id"],
            score=hit.get("_score") or 0.0,
            title=source.get("title", ""),
            description=source.get("description", ""),
            features=source.get("features", ""),
            main_category=source.get("main_category", ""),
            sub_category=source.get("sub_category", ""),
            store=source.get("store", ""),
            price=source.get("price"),
            average_rating=source.get("average_rating"),
            rating_number=source.get("rating_number"),
            primary_image_url=source.get("primary_image_url", ""),
            parent_asin=source.get("parent_asin", hit["_id"]),
        )


# ── Response models ───────────────────────────────────────────────────────────

class SearchResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    total:   int           = Field(..., description="Total matching documents")
    hits:    list[ProductHit] = Field(..., description="Result documents")
    took_ms: int           = Field(..., description="OpenSearch query latency in ms")
    mode:    SearchMode    = Field(..., description="Search mode used")
    cached:  bool          = Field(default=False, description="True if result was served from cache")


class NLSearchResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    total:        int
    hits:         list[ProductHit]
    took_ms:      int
    parsed_query: "ParsedNLQuery"
    cached:       bool = False


class SuggestResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    suggestions: list[str] = Field(..., description="Autocomplete suggestions")
    prefix:      str


class HealthResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    status:    str
    opensearch:str
    redis:     str
    version:   str = "1.0.0"


# ── NL query parsed structure ─────────────────────────────────────────────────

class ParsedNLQuery(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    semantic_query: str            = Field(default="", description="Extracted semantic search string")
    filters:        "NLFilters"    = Field(default_factory=lambda: NLFilters())
    boost_terms:    list[str]      = Field(default_factory=list)
    search_mode:    SearchMode     = Field(default=SearchMode.hybrid)


class NLFilters(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    main_category: Optional[str]   = None
    price_min:     Optional[float] = None
    price_max:     Optional[float] = None
    rating_min:    Optional[float] = None


# ── Update forward refs ───────────────────────────────────────────────────────
NLSearchResponse.model_rebuild()
ParsedNLQuery.model_rebuild()
