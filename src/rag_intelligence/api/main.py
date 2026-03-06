from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_intelligence.logging import setup_logging
from rag_intelligence.providers import ProviderRegistry
from rag_intelligence.settings import AppSettings

from .middleware import RequestIDMiddleware
from .routes import health

LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: AppSettings = app.state.settings
    setup_logging(log_level=settings.log_level, json_logs=settings.log_json)

    registry = ProviderRegistry(settings)
    app.state.registry = registry

    LOGGER.info("RAG Intelligence API started (default_llm=%s)", settings.default_llm)
    yield
    LOGGER.info("RAG Intelligence API shutting down")


def create_app(settings: AppSettings | None = None) -> FastAPI:
    settings = settings or AppSettings.from_env()

    app = FastAPI(title="RAG Intelligence", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings

    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    app.include_router(health.router, tags=["health"])

    return app


app = create_app()


def run() -> None:
    import uvicorn

    settings = AppSettings.from_env()
    uvicorn.run(
        "rag_intelligence.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
    )
