from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_intelligence.logging import setup_logging
from rag_intelligence.providers import ProviderRegistry
from rag_intelligence.settings import AppSettings

from .middleware import RequestIDMiddleware
from .routes import health, rag, search

LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: AppSettings = app.state.settings
    setup_logging(log_level=settings.log_level, json_logs=settings.log_json)

    registry = ProviderRegistry(settings)
    app.state.registry = registry

    if settings.otel_enabled:
        LOGGER.info("OpenTelemetry enabled (endpoint=%s)", settings.otel_endpoint)

    LOGGER.info("RAG Intelligence API started (default_llm=%s)", settings.default_llm)
    yield
    LOGGER.info("RAG Intelligence API shutting down")


def create_app(settings: AppSettings | None = None) -> FastAPI:
    settings = settings or AppSettings.from_env()

    # OTEL must be initialised BEFORE the app is created so that
    # FastAPIInstrumentor can wrap middleware at the right layer.
    if settings.otel_enabled:
        from rag_intelligence.telemetry import setup_telemetry

        setup_telemetry(
            service_name="rag-intelligence",
            otel_endpoint=settings.otel_endpoint,
        )

    app = FastAPI(title="RAG Intelligence", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings

    if settings.otel_enabled:
        from rag_intelligence.telemetry import instrument_fastapi

        instrument_fastapi(app)

    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    app.include_router(health.router, tags=["health"])
    app.include_router(search.router, tags=["search"])
    app.include_router(rag.router, tags=["rag"])

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
