from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_intelligence.providers import ProviderRegistry
from rag_intelligence.settings import AppSettings

from .routes import health

LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    settings = AppSettings.from_env()
    registry = ProviderRegistry(settings)

    app.state.settings = settings
    app.state.registry = registry

    LOGGER.info("RAG Intelligence API started (default_llm=%s)", settings.default_llm)
    yield
    LOGGER.info("RAG Intelligence API shutting down")


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Intelligence", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
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
