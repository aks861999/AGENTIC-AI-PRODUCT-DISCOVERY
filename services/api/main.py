"""
FastAPI application entry point.
"""
from __future__ import annotations
from pydantic import BaseModel as _BM
import asyncio
import logging
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import Body, Depends, FastAPI, HTTPException, Query, status, Request

from fastapi.middleware.cors import CORSMiddleware
from opensearchpy import OpenSearch, RequestsHttpConnection
from sentence_transformers import SentenceTransformer

from api.middleware import add_middleware
from api.models import (
    ChatRequest, ChatResponse, HealthResponse, NLSearchRequest,
    NLSearchResponse, ProductHit, SearchMode, SearchResponse, SuggestResponse,
)
from api.nl_query import _schema_linker
from api.nl_query import nl_search, _resolve_attribute_filters
from api.search import hybrid_search, lexical_search, semantic_search
from api.chatbot import chat_turn
import api.dependencies as _deps
from shared.config.settings import get_settings

import asyncio


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _log_queue
    _log_queue = asyncio.Queue(maxsize=500)



    global _os_client, _embedding_model
    logger.info("Starting up Search API...")
    _os_client = _create_os_client()
    _deps._os_client = _os_client

    logger.info("Loading embedding model '%s'...", settings.embedding_model)
    _embedding_model = await asyncio.to_thread(SentenceTransformer, settings.embedding_model)
    _deps._embedding_model = _embedding_model

    try:
        await _schema_linker._refresh_cache(_os_client, settings.index_name, _embedding_model)
        logger.info("Schema linker cache warmed at startup.")
    except Exception as exc:
        logger.warning("Schema linker warm-up failed (non-fatal): %s", exc)

    yield
    logger.info("Shutdown complete.")






logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s %(message)s")
logger = logging.getLogger(__name__)
settings = get_settings()



_os_client: Optional[OpenSearch] = None
_embedding_model: Optional[SentenceTransformer] = None

def _create_os_client() -> OpenSearch:
    http_auth = (settings.opensearch_user, settings.opensearch_password) if settings.opensearch_user else None
    return OpenSearch(
        hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
        http_auth=http_auth,
        use_ssl=settings.opensearch_use_ssl,
        verify_certs=settings.opensearch_verify_certs,
        ssl_show_warn=False,
        timeout=60,
        retry_on_timeout=True,
        max_retries=3,
        connection_class=RequestsHttpConnection,
        pool_maxsize=10,
        pool_block=False,
    )







app = FastAPI(title="Home Depot Product Search API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
add_middleware(app)



@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health_check(client: OpenSearch = Depends(_deps.get_os_client)) -> HealthResponse:
    try:
        info = await asyncio.to_thread(client.cluster.health)
        os_status = info.get("status", "unknown")
    except Exception:
        os_status = "unreachable"
    return HealthResponse(status="ok" if os_status in ("green", "yellow") else "degraded", opensearch=os_status, redis="disabled")

@app.get("/search", response_model=SearchResponse, tags=["search"])
async def structured_search(
    q: str = Query(..., min_length=1, max_length=500),
    attributes: Optional[str] = Query(default=None, description="JSON string mapping required EAV attributes, e.g., '{\"Material\": \"Steel\"}'"),
    mode: SearchMode = Query(default=SearchMode.hybrid),
    size: int = Query(default=10, ge=1, le=100),
    from_: int = Query(default=0, ge=0, alias="from"),
    client: OpenSearch = Depends(_deps.get_os_client),
    model: SentenceTransformer = Depends(_deps.get_model),
) -> SearchResponse:

    # Parse dynamic JSON filters
    parsed_attrs = None
    if attributes:
        try:
            parsed_attrs = json.loads(attributes)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format in attributes parameter")

    try:
        if mode == SearchMode.lexical:
            result = await lexical_search(
                client=client, query=q, index_name=settings.index_name,
                required_attributes=parsed_attrs, size=size, from_=from_,
            )
        elif mode == SearchMode.semantic:
            vector = await asyncio.to_thread(lambda: model.encode(q, normalize_embeddings=True).tolist())
            result = await semantic_search(
                client=client, query_vector=vector, index_name=settings.index_name,
                required_attributes=parsed_attrs, size=size, from_=from_
            )
        else:
            vector = await asyncio.to_thread(lambda: model.encode(q, normalize_embeddings=True).tolist())
            # Resolve attributes against candidate products for hybrid mode
            resolved_attrs = await _resolve_attribute_filters(
                client=client,
                index_name=settings.index_name,
                semantic_query=q,
                model=model,
                required_attributes=parsed_attrs,
            ) if parsed_attrs else None
            result = await hybrid_search(
                client=client, query=q, query_vector=vector,
                index_name=settings.index_name, required_attributes=resolved_attrs or parsed_attrs,
                size=size, from_=from_,
            )
    except Exception as exc:
        logger.exception("Search error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}")

    return result

@app.post("/nl-search", response_model=NLSearchResponse, tags=["search"])
async def natural_language_search(request: NLSearchRequest, client: OpenSearch = Depends(_deps.get_os_client), model: SentenceTransformer = Depends(_deps.get_model)) -> NLSearchResponse:
    try:
        result = await nl_search(client=client, model=model, query=request.query, index_name=settings.index_name, size=request.size)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result

@app.get("/suggest", response_model=SuggestResponse, tags=["search"])
async def suggest(prefix: str = Query(..., min_length=1, max_length=100), size: int = Query(default=5, ge=1, le=20), client: OpenSearch = Depends(_deps.get_os_client)) -> SuggestResponse:
    try:
        raw = await asyncio.to_thread(
            client.search, index=settings.index_name, body={"size": size, "_source": ["title"], "query": {"match_phrase_prefix": {"title": {"query": prefix, "max_expansions": 20}}}}
        )
        hits = raw.get("hits", {}).get("hits", [])
        suggestions = list(dict.fromkeys(h.get("_source", {}).get("title", "") for h in hits if h.get("_source", {}).get("title")))[:size]
    except Exception:
        suggestions = []
    return SuggestResponse(suggestions=suggestions, prefix=prefix)

@app.post("/index/product", status_code=status.HTTP_201_CREATED, tags=["admin"])
async def index_product(product: ProductHit, client: OpenSearch = Depends(_deps.get_os_client), model: SentenceTransformer = Depends(_deps.get_model)) -> dict:

    # Construct embedding specific to Home Depot schema constraints
    attr_text = " ".join([f"{a.name}: {a.value}" for a in product.attributes])
    embedding_text = " ".join(filter(None, [product.title, product.product_description, attr_text]))

    vector = await asyncio.to_thread(lambda: model.encode(embedding_text, normalize_embeddings=True).tolist())

    doc = product.model_dump(exclude={"id", "score"})
    doc["embedding_vector"] = vector
    doc["embedding_text"] = embedding_text

    try:
        resp = await asyncio.to_thread(client.index, index=settings.index_name, body=doc, id=product.id, refresh="wait_for")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"result": resp.get("result"), "id": resp.get("_id")}

@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: ChatRequest, client: OpenSearch = Depends(_deps.get_os_client), model: SentenceTransformer = Depends(_deps.get_model)) -> ChatResponse:
    try:
        result = await chat_turn(session_id=request.session_id or str(uuid.uuid4()), user_message=request.message, client=client, model=model)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    reply = result.get("reply", "")
    if isinstance(reply, list):
        reply = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in reply)
    result["reply"] = reply
    return ChatResponse(**result)

