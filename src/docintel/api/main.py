"""
FastAPI application entrypoint. Builds the AppContainer once at startup
(lifespan), attaches it to app.state, and tears it down on shutdown.

Run with: uvicorn docintel.api.main:app --host 0.0.0.0 --port 8000
(see docker-compose.yml / Dockerfile for the containerized equivalent).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from docintel.api.container import AppContainer
from docintel.api.routers import ingestion, knowledge_bases, query
from docintel.core.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    container = AppContainer(settings)
    app.state.container = container
    try:
        yield
    finally:
        await container.close()


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

    app.include_router(knowledge_bases.router)
    app.include_router(ingestion.router)
    app.include_router(query.router)

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
