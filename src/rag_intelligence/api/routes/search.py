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
    pipeline_phase: str | None = None


class HybridSearchBody(BaseModel):
    query: str = Field(examples=["o que acontece na silver?"])
    embedding_run_id: str | None = Field(default="pipeline-docs", examples=["pipeline-docs"])
    top_k: int = Field(default=5, gt=0, examples=[3])
    include_semantic: bool = Field(default=True, examples=[True])
    include_lexical: bool = Field(default=True, examples=[False])
    model_filter: str | None = Field(default=None, examples=[None])
    pipeline_phase: str | None = Field(default=None, examples=["silver"])


class SemanticResultBody(BaseModel):
    rank: int
    score: float | None = None
    doc_id: str
    text: str
    event_type: str | None = None
    map: str | None = None
    file: str | None = None
    round: int | str | None = None
    source_file: str | None = None
    metadata: dict = Field(default_factory=dict)


class LexicalResultBody(BaseModel):
    rank: int
    score: float
    run_id: str
    model_name: str
    roc_auc: float | None = None
    f1: float | None = None
    balanced_accuracy: float | None = None
    log_loss_val: float | None = None
    brier: float | None = None
    feature_importances: dict | None = None
    text_summary: str
    created_at: str | None = None


class SearchResponseBody(BaseModel):
    query: str
    embedding_run_id: str
    top_k: int
    filters: dict
    results_returned: int
    retrieval_ms: int
    results: list[SemanticResultBody]


class HybridSearchResponseBody(BaseModel):
    semantic_results: list[SemanticResultBody]
    lexical_results: list[LexicalResultBody]
    retrieval_ms: int


HYBRID_SEARCH_EXAMPLE = {
    "query": "arquitetura ml training logistic regression roc auc",
    "embedding_run_id": "pipeline-docs",
    "top_k": 3,
    "include_semantic": True,
    "include_lexical": True,
    "pipeline_phase": None,
}


@router.post("/search", response_model=SearchResponseBody)
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
        pipeline_phase=body.pipeline_phase,
    )
    response = search_events(
        request,
        app_settings_factory=lambda: settings,
        registry_factory=lambda _s: registry,
    )
    return response.to_dict()


@router.post(
    "/search/hybrid",
    response_model=HybridSearchResponseBody,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "example": HYBRID_SEARCH_EXAMPLE,
                }
            }
        }
    },
)
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
                document_tier="pipeline_doc",
                pipeline_phase=body.pipeline_phase,
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
