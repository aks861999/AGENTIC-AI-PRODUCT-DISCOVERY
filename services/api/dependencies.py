"""
FastAPI dependency providers.

All injectors live here so that:
- api/main.py stays focused on route definitions
- tests can patch `api.dependencies._os_client` / `api.dependencies._model`
  from a single well-known location
- future services (e.g. a background worker) can reuse the same singletons

Usage in endpoint signatures:
    from api.dependencies import ClientDep, ModelDep
    async def my_endpoint(client: ClientDep, model: ModelDep): ...
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, status
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

# Module-level singletons — set during application lifespan by api/main.py
_os_client: OpenSearch | None = None
_embedding_model: SentenceTransformer | None = None


def get_os_client() -> OpenSearch:
    """Dependency: returns the shared OpenSearch client or raises 503."""
    if _os_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenSearch client not initialised.",
        )
    return _os_client


def get_model() -> SentenceTransformer:
    """Dependency: returns the shared SentenceTransformer model or raises 503."""
    if _embedding_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding model not initialised.",
        )
    return _embedding_model


# ── Typed dependency aliases (FastAPI Annotated pattern) ─────────────────────
ClientDep = Annotated[OpenSearch, Depends(get_os_client)]
ModelDep = Annotated[SentenceTransformer, Depends(get_model)]
