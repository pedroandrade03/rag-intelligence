"""GET /metadata — dataset lineage and service metadata."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from rag_intelligence.db import build_pgvector_data_table_name
from rag_intelligence.lexical_retrieval import get_latest_training_run
from rag_intelligence.metadata import (
    STAGE_ORDER,
    LineageAuditError,
    MetadataSettings,
    RunRecord,
    get_latest_run,
    get_run,
    get_run_lineage,
)

from ..deps import SettingsDep

router = APIRouter()


def _latest_training_payload(settings: Any, model_filter: str | None = None) -> dict[str, Any]:
    rows = get_latest_training_run(model_filter=model_filter, settings=settings)
    if not rows:
        return {"run_id": None, "created_at": None, "models": [], "count": 0}

    return {
        "run_id": rows[0].run_id,
        "created_at": rows[0].created_at,
        "models": [row.to_dict() for row in rows],
        "count": len(rows),
    }


def _metadata_settings_from_app(settings: Any) -> MetadataSettings:
    return MetadataSettings(
        pg_host=settings.pg_host,
        pg_port=settings.pg_port,
        pg_user=settings.pg_user,
        pg_password=settings.pg_password,
        pg_database=settings.pg_database,
    )


def _record_to_dict(record: RunRecord) -> dict[str, Any]:
    payload = asdict(record)
    created_at = payload.get("created_at")
    if isinstance(created_at, datetime):
        payload["created_at"] = created_at.isoformat()
    return payload


@router.get("/metadata/training")
async def training_metadata(
    settings: SettingsDep,
    latest: bool = Query(default=True, description="Return the latest training run."),
    model_filter: str | None = Query(
        default=None,
        description="Optional model_name filter, e.g. logistic_regression.",
    ),
):
    """Return structured ML training metadata for agent/tool use and demos."""
    if not latest:
        raise HTTPException(status_code=422, detail="Only latest=true is currently supported")

    try:
        return _latest_training_payload(settings, model_filter=model_filter)
    except Exception as exc:  # pragma: no cover - depends on live DB/driver errors
        raise HTTPException(status_code=503, detail="Training metadata store unavailable") from exc


@router.get("/metadata")
async def metadata(
    settings: SettingsDep,
    stage: str | None = Query(default=None, description="Pipeline stage to inspect."),
    run_id: str | None = Query(default=None, description="Specific run_id to inspect."),
    lineage: bool = Query(
        default=False,
        description="Return full upstream lineage for stage/run_id.",
    ),
):
    """Return service metadata, latest run metadata, or lineage metadata.

    - No query params: service/storage metadata for documentation and demos.
    - stage only: latest completed run for that stage.
    - stage + run_id: specific run metadata.
    - stage + run_id + lineage=true: upstream lineage chain.
    """
    if stage is None:
        return {
            "service": "rag-intelligence",
            "metadata_store": "PostgreSQL dataset_runs",
            "vector_store": "PostgreSQL pgvector",
            "lexical_search": "PostgreSQL Full-Text Search",
            "stages": list(STAGE_ORDER),
            "default_embedding_run_id": settings.default_embedding_run_id,
            "embedding_model": settings.default_embed_model,
            "llm": settings.default_llm,
            "vector_table": f"public.{build_pgvector_data_table_name(settings.pg_table_name)}",
            "embedding_dimension": settings.pg_embed_dim,
        }

    if stage not in STAGE_ORDER:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported stage '{stage}'. Expected one of: {', '.join(STAGE_ORDER)}",
        )

    if lineage and not run_id:
        raise HTTPException(status_code=422, detail="run_id is required when lineage=true")

    md_settings = _metadata_settings_from_app(settings)
    try:
        if lineage:
            report = get_run_lineage(md_settings, stage=stage, run_id=run_id or "")
            return report.to_dict()

        record = (
            get_run(md_settings, stage=stage, run_id=run_id)
            if run_id
            else get_latest_run(md_settings, stage)
        )
    except LineageAuditError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - depends on live DB/driver errors
        raise HTTPException(status_code=503, detail="Metadata store unavailable") from exc

    if record is None:
        target = f"stage={stage} run_id={run_id}" if run_id else f"stage={stage}"
        raise HTTPException(status_code=404, detail=f"No metadata found for {target}")

    return _record_to_dict(record)
