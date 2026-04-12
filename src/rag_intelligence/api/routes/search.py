"""POST /search and POST /search/hybrid — retrieval endpoints."""

from __future__ import annotations

import logging
from time import perf_counter

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from rag_intelligence.retrieval import SearchRequest, search_events

from ..deps import RegistryDep, SettingsDep

LOGGER = logging.getLogger(__name__)

router = APIRouter()


class SearchBody(BaseModel):
    query: str
    embedding_run_id: str | None = None
    top_k: int = Field(default=5, gt=0)
    event_type: str | None = None
    map_name: str | None = None
    file_name: str | None = None
    round_number: int | None = None


class HybridSearchBody(BaseModel):
    query: str
    embedding_run_id: str | None = Field(default="pipeline-docs")
    top_k: int = Field(default=5, gt=0)
    include_semantic: bool = True
    include_lexical: bool = True
    model_filter: str | None = None


@router.post("/search")
async def search(body: SearchBody, settings: SettingsDep, registry: RegistryDep):
    run_id = body.embedding_run_id or settings.default_embedding_run_id
    if not run_id:
        raise HTTPException(
            status_code=422,
            detail=(
                "embedding_run_id is required"
                " (provide it in the body or set DEFAULT_EMBEDDING_RUN_ID)"
            ),
        )

    request = SearchRequest(
        query=body.query,
        embedding_run_id=run_id,
        top_k=body.top_k,
        event_type=body.event_type,
        map_name=body.map_name,
        file_name=body.file_name,
        round_number=body.round_number,
    )
    response = search_events(
        request,
        app_settings_factory=lambda: settings,
        registry_factory=lambda _s: registry,
    )
    return response.to_dict()


@router.post("/search/hybrid")
async def hybrid_search(body: HybridSearchBody, settings: SettingsDep, registry: RegistryDep):
    t0 = perf_counter()
    semantic_results: list[dict] = []
    lexical_results: list[dict] = []

    if body.include_semantic:
        run_id = body.embedding_run_id or "pipeline-docs"
        try:
            request = SearchRequest(
                query=body.query,
                embedding_run_id=run_id,
                top_k=body.top_k,
            )
            response = search_events(
                request,
                app_settings_factory=lambda: settings,
                registry_factory=lambda _s: registry,
            )
            semantic_results = response.to_dict().get("results", [])
        except Exception:
            LOGGER.warning("Semantic search failed", exc_info=True)

    if body.include_lexical:
        try:
            from rag_intelligence.lexical_retrieval import lexical_search

            hits = lexical_search(
                body.query,
                top_k=body.top_k,
                model_filter=body.model_filter,
                settings=settings,
            )
            lexical_results = [h.to_dict() for h in hits]
        except Exception:
            LOGGER.warning("Lexical search failed", exc_info=True)

    retrieval_ms = int((perf_counter() - t0) * 1000)
    return {
        "semantic_results": semantic_results,
        "lexical_results": lexical_results,
        "retrieval_ms": retrieval_ms,
    }
