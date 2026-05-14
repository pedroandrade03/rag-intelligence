"""POST /rag/query — retrieval + LLM synthesis endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag_intelligence.rag import RAGRequest, rag_query, rag_query_stream

from ..deps import RegistryDep, SettingsDep

router = APIRouter(prefix="/rag")
root_router = APIRouter()


class RAGBody(BaseModel):
    query: str = Field(
        examples=["o que acontece na silver?"],
        description="Question to answer using retrieved context.",
    )
    embedding_run_id: str | None = Field(
        default=None,
        examples=["pipeline-docs"],
        description="Embedding run to search. Use pipeline-docs for project documentation.",
    )
    top_k: int = Field(default=3, gt=0, examples=[3])
    event_type: str | None = Field(default=None, examples=[None])
    map_name: str | None = Field(default=None, examples=[None])
    stream: bool = Field(
        default=False,
        description="Use false in Swagger/JSON clients. true returns text/event-stream.",
        examples=[False],
    )
    llm_key: str | None = Field(default=None, examples=[None])


RAG_EXAMPLE: dict[str, Any] = {
    "query": "o que acontece na silver?",
    "embedding_run_id": "pipeline-docs",
    "top_k": 3,
    "event_type": None,
    "map_name": None,
    "stream": False,
    "llm_key": None,
}

RAG_JSON_RESPONSE: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "RAG answer. With stream=false this is JSON; with stream=true this is SSE.",
        "content": {
            "application/json": {
                "example": {
                    "query": "o que acontece na silver?",
                    "answer": "A Silver limpa e normaliza os CSVs da Bronze...",
                    "sources": [],
                    "retrieval_ms": 25,
                    "generation_ms": 1200,
                }
            },
            "text/event-stream": {
                "example": "event: sources\\ndata: {...}\\n\\nevent: chunk\\ndata: {...}\\n\\n"
            },
        },
    }
}


RAG_OPENAPI_EXTRA = {
    "requestBody": {
        "content": {
            "application/json": {
                "example": RAG_EXAMPLE,
            }
        }
    }
}


@router.post("/query", responses=RAG_JSON_RESPONSE, openapi_extra=RAG_OPENAPI_EXTRA)
@root_router.post("/query", responses=RAG_JSON_RESPONSE, openapi_extra=RAG_OPENAPI_EXTRA)
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
