"""POST /search — retrieval-only endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from rag_intelligence.retrieval import SearchRequest, search_events

from ..deps import RegistryDep, SettingsDep

router = APIRouter()


class SearchBody(BaseModel):
    query: str
    embedding_run_id: str | None = None
    top_k: int = Field(default=5, gt=0)
    event_type: str | None = None
    map_name: str | None = None
    file_name: str | None = None
    round_number: int | None = None


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
