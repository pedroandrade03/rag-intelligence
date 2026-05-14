from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

from fastapi.testclient import TestClient

from rag_intelligence.api.main import create_app
from rag_intelligence.lexical_retrieval import LexicalSearchResult
from rag_intelligence.metadata import RunRecord
from rag_intelligence.settings import AppSettings


def _settings() -> AppSettings:
    return AppSettings.from_env(
        {
            "PG_HOST": "db",
            "PG_PORT": "5432",
            "PG_USER": "raguser",
            "PG_PASSWORD": "ragpassword",
            "PG_DATABASE": "ragdb",
            "PG_TABLE_NAME": "rag_embeddings",
            "PG_EMBED_DIM": "768",
            "DEFAULT_EMBEDDING_RUN_ID": "pipeline-docs",
        }
    )


def _record(stage: str = "gold", run_id: str = "run-1") -> RunRecord:
    return RunRecord(
        id=1,
        run_id=run_id,
        stage=stage,
        status="completed",
        dataset_prefix="csgo-matchmaking-damage",
        bucket=stage,
        source_run_id="parent-run",
        events_key="events.csv",
        artifact_prefix="artifact/prefix",
        manifest_key="manifest.json",
        quality_report_key="quality.json",
        files_processed=2,
        rows_read=100,
        rows_output=90,
        quality_summary={"valid": True},
        created_at=datetime(2026, 5, 14, tzinfo=UTC),
    )


def test_metadata_service_summary():
    app = create_app(settings=_settings())
    with TestClient(app) as client:
        resp = client.get("/metadata")

    assert resp.status_code == 200
    data = resp.json()
    assert data["metadata_store"] == "PostgreSQL dataset_runs"
    assert data["vector_store"] == "PostgreSQL pgvector"
    assert data["lexical_search"] == "PostgreSQL Full-Text Search"
    assert data["vector_table"] == "public.data_rag_embeddings"
    assert data["default_embedding_run_id"] == "pipeline-docs"


def test_metadata_latest_stage():
    app = create_app(settings=_settings())
    with (
        patch("rag_intelligence.api.routes.metadata.get_latest_run", return_value=_record()) as fn,
        TestClient(app) as client,
    ):
        resp = client.get("/metadata", params={"stage": "gold"})

    assert resp.status_code == 200
    assert resp.json()["stage"] == "gold"
    fn.assert_called_once()


def test_metadata_specific_run():
    app = create_app(settings=_settings())
    with (
        patch("rag_intelligence.api.routes.metadata.get_run", return_value=_record()) as fn,
        TestClient(app) as client,
    ):
        resp = client.get("/metadata", params={"stage": "gold", "run_id": "run-1"})

    assert resp.status_code == 200
    assert resp.json()["run_id"] == "run-1"
    fn.assert_called_once()


def test_metadata_latest_training():
    app = create_app(settings=_settings())
    training_rows = [
        LexicalSearchResult(
            rank=1,
            score=1.0,
            run_id="train-run-1",
            model_name="logistic_regression",
            roc_auc=0.68,
            f1=0.65,
            balanced_accuracy=0.66,
            log_loss_val=0.60,
            brier=0.22,
            feature_importances={"eq_diff": 0.2},
            text_summary="logistic regression roc_auc 0.68",
            created_at="2026-05-14T20:17:00+00:00",
        )
    ]
    with (
        patch(
            "rag_intelligence.api.routes.metadata.get_latest_training_run",
            return_value=training_rows,
        ) as fn,
        TestClient(app) as client,
    ):
        resp = client.get("/metadata/training")

    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == "train-run-1"
    assert data["models"][0]["model_name"] == "logistic_regression"
    fn.assert_called_once()


def test_metadata_invalid_stage_returns_422():
    app = create_app(settings=_settings())
    with TestClient(app) as client:
        resp = client.get("/metadata", params={"stage": "invalid"})

    assert resp.status_code == 422


def test_metadata_lineage_requires_run_id():
    app = create_app(settings=_settings())
    with TestClient(app) as client:
        resp = client.get("/metadata", params={"stage": "gold", "lineage": "true"})

    assert resp.status_code == 422
