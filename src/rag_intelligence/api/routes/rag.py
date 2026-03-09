"""POST /rag/query — retrieval + LLM synthesis endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag_intelligence.rag import RAGRequest, rag_query, rag_query_stream

from ..deps import RegistryDep, SettingsDep

router = APIRouter(prefix="/rag")


class RAGBody(BaseModel):
    query: str
    embedding_run_id: str | None = None
    top_k: int = Field(default=10, gt=0)
    event_type: str | None = None
    map_name: str | None = None
    stream: bool = True
    llm_key: str | None = None


@router.post("/query")
async def query(body: RAGBody, settings: SettingsDep, registry: RegistryDep):
    run_id = body.embedding_run_id or settings.default_embedding_run_id
    if not run_id:
        raise HTTPException(
            status_code=422,
            detail=(
                "embedding_run_id is required"
                " (provide it in the body or set DEFAULT_EMBEDDING_RUN_ID)"
            ),
        )

    request = RAGRequest(
        query=body.query,
        embedding_run_id=run_id,
        top_k=body.top_k,
        event_type=body.event_type,
        map_name=body.map_name,
        llm_key=body.llm_key,
    )

    settings_factory = lambda: settings  # noqa: E731
    registry_factory = lambda _s: registry  # noqa: E731

    if not body.stream:
        response = rag_query(
            request,
            app_settings_factory=settings_factory,
            registry_factory=registry_factory,
        )
        return {
            "query": response.query,
            "answer": response.answer,
            "sources": response.sources,
            "retrieval_ms": response.retrieval_ms,
            "generation_ms": response.generation_ms,
        }

    return StreamingResponse(
        rag_query_stream(
            request,
            app_settings_factory=settings_factory,
            registry_factory=registry_factory,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
