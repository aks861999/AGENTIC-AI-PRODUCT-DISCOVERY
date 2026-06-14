"""
Pydantic v2 models for all FastAPI request/response contracts.
Strictly bound to the Home Depot EAV schema.
"""
from __future__ import annotations
import uuid as _uuid
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field

class SearchMode(str, Enum):
    lexical  = "lexical"
    semantic = "semantic"
    hybrid   = "hybrid"

class SearchRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    q:          str        = Field(..., min_length=1, max_length=500)
    attributes: dict[str, str] = Field(default_factory=dict, description="Key-value specs to filter by (e.g., {'Material': 'Steel'})")
    mode:       SearchMode = Field(default=SearchMode.hybrid)
    size:       int        = Field(default=10, ge=1, le=100)
    from_:      int        = Field(default=0, ge=0, alias="from")

class NLSearchRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    query: str = Field(..., min_length=3, max_length=1000)
    size:  int = Field(default=10, ge=1, le=100)

class SuggestRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    prefix: str = Field(..., min_length=1, max_length=100)
    size:   int = Field(default=5, ge=1, le=20)

class ChatRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    message:    str           = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = Field(default_factory=lambda: str(_uuid.uuid4()))

class ChatResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    session_id: str
    reply:      str
    results:    list[dict] = Field(default_factory=list)

class ProductAttribute(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    name:  str
    value: str

class ProductHit(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id:                  str                    = Field(..., description="Document ID (product_uid)")
    score:               float                  = Field(..., description="Relevance score from OpenSearch")
    product_uid:         str                    = Field(..., description="Home Depot Product UID")
    title:               str                    = Field(default="")
    product_description: str                    = Field(default="")
    attributes:          list[ProductAttribute] = Field(default_factory=list)

    @classmethod
    def from_hit(cls, hit: dict) -> "ProductHit":
        source = hit.get("_source", {})
        return cls(
            id=hit["_id"],
            score=hit.get("_score") or 0.0,
            product_uid=source.get("product_uid", hit["_id"]),
            title=source.get("title", ""),
            product_description=source.get("product_description", ""),
            attributes=source.get("attributes", []),
        )

class SearchResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    total:   int
    hits:    list[ProductHit]
    took_ms: int
    mode:    SearchMode
    cached:  bool = False

class NLSearchResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    total:        int
    hits:         list[ProductHit]
    took_ms:      int
    parsed_query: "ParsedNLQuery"
    cached:       bool = False

class SuggestResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    suggestions: list[str]
    prefix:      str

class HealthResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    status:    str
    opensearch:str
    redis:     str
    version:   str = "1.0.0"

class ParsedNLQuery(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    semantic_query: str        = Field(default="")
    filters:        "NLFilters" = Field(default_factory=lambda: NLFilters())
    boost_terms:    list[str]  = Field(default_factory=list)
    search_mode:    SearchMode = Field(default=SearchMode.hybrid)

class NLFilters(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    attributes: dict[str, str] = Field(default_factory=dict)

NLSearchResponse.model_rebuild()
ParsedNLQuery.model_rebuild()