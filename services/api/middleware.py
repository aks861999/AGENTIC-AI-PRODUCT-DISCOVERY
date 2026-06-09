"""
api/middleware.py , ASGI middleware for the Search API.

Middleware applied (in order, outermost first):
    1. RequestTimingMiddleware  , adds X-Process-Time-Ms response header
                                   and logs method + path + status + latency
    2. CorrelationIdMiddleware  , generates/forwards X-Correlation-ID
                                   so distributed traces can be joined

Usage in api/main.py:
    from api.middleware import add_middleware
    add_middleware(app)
"""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

_SKIP_PATHS = frozenset({"/health", "/docs", "/redoc", "/openapi.json"})


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """
    Measures wall-clock latency for every request and:
      - adds an ``X-Process-Time-Ms`` header to every response
      - logs a one-liner at INFO level: METHOD /path → status  NNms
    """

    def __init__(self, app: ASGIApp, skip_paths: frozenset[str] = _SKIP_PATHS) -> None:
        super().__init__(app)
        self.skip_paths = skip_paths

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response: Response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

        response.headers["X-Process-Time-Ms"] = str(elapsed_ms)

        if request.url.path not in self.skip_paths:
            logger.info(
                "%s %s → %d  %.1fms",
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
            )

        return response


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """
    Reads ``X-Correlation-ID`` from incoming requests (or generates one),
    attaches it to the response, and makes it available as
    ``request.state.correlation_id`` for downstream handlers.

    Useful for correlating API gateway logs with service logs.
    """

    HEADER = "X-Correlation-ID"

    async def dispatch(self, request: Request, call_next) -> Response:
        correlation_id = request.headers.get(self.HEADER) or str(uuid.uuid4())
        request.state.correlation_id = correlation_id

        response: Response = await call_next(request)
        response.headers[self.HEADER] = correlation_id
        return response


def add_middleware(app) -> None:
    """
    Register all custom middleware on the FastAPI app.

    Call this once during app construction, before any routes are defined.
    Middleware is applied in reverse registration order (last added = outermost).
    """
    # CorrelationId registered last → runs outermost (first to see the request)
    app.add_middleware(CorrelationIdMiddleware)
    # Timing registered first → runs innermost (closest to the handler)
    app.add_middleware(RequestTimingMiddleware)
