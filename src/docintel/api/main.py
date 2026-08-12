"""FastAPI application entrypoint, lifecycle, health and request observability."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from docintel.api.container import AppContainer
from docintel.api.routers import ingestion, knowledge_bases, query
from docintel.core.config import get_settings

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    container = AppContainer(settings)
    app.state.container = container
    try:
        yield
    finally:
        await container.close()


async def _request_observability_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    start = time.perf_counter()
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
    )
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request_failed", duration_ms=round((time.perf_counter() - start) * 1000, 2))
        raise
    finally:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "request_complete",
            status_code=getattr(locals().get("response"), "status_code", 500),
            duration_ms=duration_ms,
        )
        structlog.contextvars.clear_contextvars()

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = str(duration_ms)
    return response


def create_app() -> FastAPI:
    app = FastAPI(
        title="docintel",
        description="Modular, domain-agnostic Document Intelligence / RAG platform.",
        version="0.1.0",
        lifespan=lifespan,
    )

    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.api_cors_origins] if settings.api_cors_origins != "*" else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.middleware("http")(_request_observability_middleware)

    app.include_router(knowledge_bases.router)
    app.include_router(ingestion.router)
    app.include_router(query.router)

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", tags=["health"])
    async def ready(request: Request) -> Response:
        """Report whether critical runtime dependencies are reachable."""
        container: AppContainer | None = getattr(request.app.state, "container", None)
        if container is None:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "dependencies": {"container": "unavailable"}},
            )

        try:
            await container.qdrant_client.get_collections()
        except Exception:  # noqa: BLE001 -- readiness must never leak provider internals
            logger.exception("readiness_dependency_failed", dependency="qdrant")
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "dependencies": {"qdrant": "unavailable"}},
            )

        return JSONResponse(
            status_code=200,
            content={"status": "ready", "dependencies": {"qdrant": "ok"}},
        )

    return app


app = create_app()
