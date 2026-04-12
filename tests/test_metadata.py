from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from conftest import FakeConnection, FakeCursor
from rag_intelligence.config import ConfigError
from rag_intelligence.metadata import (
    LineageAuditError,
    MetadataSettings,
    RunRecord,
    ensure_schema,
    get_latest_run,
    get_run,
    get_run_lineage,
    register_run,
)

_VALID_PG_ENV = {
    "PG_HOST": "localhost",
    "PG_PORT": "5432",
    "PG_USER": "raguser",
    "PG_PASSWORD": "ragpassword",
    "PG_DATABASE": "ragdb",
}


def _make_settings() -> MetadataSettings:
    return MetadataSettings.from_env(_VALID_PG_ENV)


def _build_row(
    *,
    row_id: int,
    run_id: str,
    stage: str,
    dataset_prefix: str = "csgo",
    bucket: str | None = None,
    source_run_id: str | None = None,
    events_key: str | None = None,
    artifact_prefix: str | None = None,
    manifest_key: str | None = None,
    quality_report_key: str | None = None,
    files_processed: int = 1,
    rows_read: int = 100,
    rows_output: int = 90,
    quality_summary: dict[str, object] | None = None,
) -> tuple[object, ...]:
    return (
        row_id,
        run_id,
        stage,
        "completed",
        dataset_prefix,
        bucket or stage,
        source_run_id,
        events_key,
        artifact_prefix,
        manifest_key,
        quality_report_key,
        files_processed,
        rows_read,
        rows_output,
        quality_summary,
        datetime(2025, 1, 15, 12, row_id, 0, tzinfo=UTC),
    )


class MappingCursor(FakeCursor):
    def __init__(self, rows_by_stage_run: dict[tuple[str, str], tuple[object, ...]]) -> None:
        super().__init__()
        self._rows_by_stage_run = rows_by_stage_run

    def execute(self, query: str, params: tuple[object, ...] = ()) -> None:
        super().execute(query, params)
        if "WHERE stage = %s AND run_id = %s" in query:
            row = self._rows_by_stage_run.get((str(params[0]), str(params[1])))
            self.set_result([row] if row is not None else [])


class MappingConnection(FakeConnection):
    def __init__(self, rows_by_stage_run: dict[tuple[str, str], tuple[object, ...]]) -> None:
        super().__init__(cursor=MappingCursor(rows_by_stage_run))


class TestMetadataSettings:
    def test_from_env_valid(self) -> None:
        s = MetadataSettings.from_env(_VALID_PG_ENV)
        assert s.pg_host == "localhost"
        assert s.pg_port == 5432
        assert s.pg_user == "raguser"
        assert s.pg_password == "ragpassword"
        assert s.pg_database == "ragdb"

    def test_from_env_missing_host(self) -> None:
        env = {**_VALID_PG_ENV, "PG_HOST": ""}
        with pytest.raises(ConfigError, match="Missing required PG"):
            MetadataSettings.from_env(env)

    def test_from_env_invalid_port(self) -> None:
        env = {**_VALID_PG_ENV, "PG_PORT": "abc"}
        with pytest.raises(ConfigError, match="Invalid PG_PORT"):
            MetadataSettings.from_env(env)

    def test_from_env_default_port(self) -> None:
        env = {k: v for k, v in _VALID_PG_ENV.items() if k != "PG_PORT"}
        s = MetadataSettings.from_env(env)
        assert s.pg_port == 5432


class TestEnsureSchema:
    def test_creates_table_and_commits(self) -> None:
        conn = FakeConnection()
        settings = _make_settings()
        ensure_schema(settings, conn_factory=lambda _s: conn)
        assert conn.committed
        assert conn.closed
        queries = conn.cursor().queries
        assert len(queries) == 5
        assert "CREATE TABLE IF NOT EXISTS dataset_runs" in queries[0][0]
        assert "ADD COLUMN IF NOT EXISTS artifact_prefix" in queries[1][0]
        assert "ADD COLUMN IF NOT EXISTS manifest_key" in queries[2][0]
        assert "DROP CONSTRAINT IF EXISTS dataset_runs_stage_check" in queries[3][0]
        assert (
            "CHECK (stage IN ('bronze', 'silver', 'gold', 'documents', 'embeddings'))"
            in queries[4][0]
        )


class TestRegisterRun:
    def test_inserts_and_returns_record(self) -> None:
        created = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
        cursor = FakeCursor()
        cursor.set_result([(42, created)])
        conn = FakeConnection(cursor=cursor)

        settings = _make_settings()
        record = RunRecord(
            run_id="20250115T120000Z",
            stage="gold",
            dataset_prefix="csgo-matchmaking-damage",
            bucket="gold",
            source_run_id="20250115T110000Z",
            events_key="csgo/20250115T120000Z/events.parquet",
            artifact_prefix="csgo/20250115T120000Z/documents/",
            manifest_key="csgo/20250115T120000Z/documents/manifest.json",
            quality_report_key="csgo/20250115T120000Z/quality_report.json",
            files_processed=3,
            rows_read=1000,
            rows_output=950,
            quality_summary={"missing_weapon": 10},
        )
        result = register_run(settings, record, conn_factory=lambda _s: conn)

        assert result.id == 42
        assert result.created_at == created
        assert result.run_id == "20250115T120000Z"
        assert result.stage == "gold"
        assert conn.committed
        assert conn.closed

        query, params = cursor.queries[0]
        assert "INSERT INTO dataset_runs" in query
        assert params[0] == "20250115T120000Z"
        assert params[1] == "gold"
        assert params[7] == "csgo/20250115T120000Z/documents/"
        assert params[8] == "csgo/20250115T120000Z/documents/manifest.json"
        assert params[13] == json.dumps({"missing_weapon": 10})

    def test_null_quality_summary(self) -> None:
        cursor = FakeCursor()
        cursor.set_result([(1, datetime.now(tz=UTC))])
        conn = FakeConnection(cursor=cursor)

        record = RunRecord(
            run_id="r1",
            stage="bronze",
            dataset_prefix="test",
            bucket="bronze",
        )
        register_run(_make_settings(), record, conn_factory=lambda _s: conn)
        _query, params = cursor.queries[0]
        assert params[13] is None


class TestGetLatestRun:
    def test_returns_record_when_exists(self) -> None:
        created = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
        cursor = FakeCursor()
        cursor.set_result(
            [
                (
                    42,
                    "20250115T120000Z",
                    "gold",
                    "completed",
                    "csgo",
                    "gold",
                    "src-run",
                    "events.parquet",
                    "artifacts/",
                    "manifest.json",
                    "quality_report.json",
                    3,
                    1000,
                    950,
                    json.dumps({"missing_weapon": 10}),
                    created,
                )
            ]
        )
        conn = FakeConnection(cursor=cursor)

        result = get_latest_run(_make_settings(), "gold", conn_factory=lambda _s: conn)

        assert result is not None
        assert result.id == 42
        assert result.run_id == "20250115T120000Z"
        assert result.stage == "gold"
        assert result.artifact_prefix == "artifacts/"
        assert result.manifest_key == "manifest.json"
        assert result.quality_summary == {"missing_weapon": 10}
        assert conn.closed

    def test_returns_none_when_empty(self) -> None:
        conn = FakeConnection()
        result = get_latest_run(_make_settings(), "gold", conn_factory=lambda _s: conn)
        assert result is None
        assert conn.closed

    def test_handles_dict_quality_summary(self) -> None:
        cursor = FakeCursor()
        cursor.set_result(
            [
                (
                    1,
                    "r1",
                    "silver",
                    "completed",
                    "ds",
                    "silver",
                    None,
                    None,
                    None,
                    None,
                    "qr.json",
                    1,
                    100,
                    90,
                    {"dup": 5},
                    datetime.now(tz=UTC),
                )
            ]
        )
        conn = FakeConnection(cursor=cursor)
        result = get_latest_run(_make_settings(), "silver", conn_factory=lambda _s: conn)
        assert result is not None
        assert result.quality_summary == {"dup": 5}


class TestGetRun:
    def test_returns_exact_run_when_exists(self) -> None:
        cursor = FakeCursor()
        cursor.set_result(
            [
                _build_row(
                    row_id=7,
                    run_id="20250115T120000Z",
                    stage="documents",
                    bucket="gold",
                    source_run_id="20250115T110000Z",
                    manifest_key="docs/manifest.json",
                )
            ]
        )
        conn = FakeConnection(cursor=cursor)

        result = get_run(
            _make_settings(),
            stage="documents",
            run_id="20250115T120000Z",
            conn_factory=lambda _s: conn,
        )

        assert result is not None
        assert result.stage == "documents"
        assert result.run_id == "20250115T120000Z"
        assert result.manifest_key == "docs/manifest.json"

    def test_returns_none_when_run_is_missing(self) -> None:
        conn = FakeConnection()
        result = get_run(
            _make_settings(),
            stage="embeddings",
            run_id="missing",
            conn_factory=lambda _s: conn,
        )
        assert result is None


class TestGetRunLineage:
    def test_builds_full_lineage_chain_for_embeddings(self) -> None:
        rows = {
            ("embeddings", "embed-run"): _build_row(
                row_id=5,
                run_id="embed-run",
                stage="embeddings",
                bucket="reports",
                source_run_id="docs-run",
                artifact_prefix="csgo/embed-run/embeddings/",
                manifest_key="csgo/embed-run/embeddings/manifest.json",
                quality_report_key="csgo/embed-run/embeddings/quality_report.json",
                quality_summary={"documents_indexed": 90},
            ),
            ("documents", "docs-run"): _build_row(
                row_id=4,
                run_id="docs-run",
                stage="documents",
                bucket="gold",
                source_run_id="gold-run",
                artifact_prefix="csgo/docs-run/documents/",
                manifest_key="csgo/docs-run/documents/manifest.json",
                quality_report_key="csgo/docs-run/documents/quality_report.json",
                quality_summary={"documents_generated": 90},
            ),
            ("gold", "gold-run"): _build_row(
                row_id=3,
                run_id="gold-run",
                stage="gold",
                bucket="gold",
                source_run_id="silver-run",
                events_key="csgo/gold-run/curated/events.csv",
                artifact_prefix="csgo/gold-run/curated/",
                quality_report_key="csgo/gold-run/quality_report.json",
                quality_summary={"rows_output": 90},
            ),
            ("silver", "silver-run"): _build_row(
                row_id=2,
                run_id="silver-run",
                stage="silver",
                bucket="silver",
                source_run_id="bronze-run",
                artifact_prefix="csgo/silver-run/cleaned/",
                quality_report_key="csgo/silver-run/quality_report.json",
                quality_summary={"rows_output": 100},
            ),
            ("bronze", "bronze-run"): _build_row(
                row_id=1,
                run_id="bronze-run",
                stage="bronze",
                bucket="bronze",
                artifact_prefix="csgo/bronze-run/",
                quality_summary={"raw_objects": 1, "extracted_objects": 2},
            ),
        }

        report = get_run_lineage(
            _make_settings(),
            stage="embeddings",
            run_id="embed-run",
            conn_factory=lambda _s: MappingConnection(rows),
        )

        assert [node.stage for node in report.resolved_chain] == [
            "embeddings",
            "documents",
            "gold",
            "silver",
            "bronze",
        ]
        assert report.integrity_status == "ok"
        payload = report.to_dict()
        assert payload["summary"]["chain_complete"] is True
        assert payload["summary"]["hops"] == 4
        assert payload["resolved_chain"][0]["evidence"]["manifest_key"].endswith("manifest.json")
        assert payload["resolved_chain"][-1]["evidence"]["quality_summary"]["raw_objects"] == 1

    def test_builds_lineage_from_intermediate_stage(self) -> None:
        rows = {
            ("gold", "gold-run"): _build_row(
                row_id=3,
                run_id="gold-run",
                stage="gold",
                source_run_id="silver-run",
                dataset_prefix="csgo",
            ),
            ("silver", "silver-run"): _build_row(
                row_id=2,
                run_id="silver-run",
                stage="silver",
                source_run_id="bronze-run",
                dataset_prefix="csgo",
            ),
            ("bronze", "bronze-run"): _build_row(
                row_id=1,
                run_id="bronze-run",
                stage="bronze",
                dataset_prefix="csgo",
            ),
        }

        report = get_run_lineage(
            _make_settings(),
            stage="gold",
            run_id="gold-run",
            conn_factory=lambda _s: MappingConnection(rows),
        )

        assert [node.stage for node in report.resolved_chain] == ["gold", "silver", "bronze"]
        assert report.to_dict()["summary"]["chain_complete"] is True

    def test_raises_when_parent_run_is_missing(self) -> None:
        rows = {
            ("documents", "docs-run"): _build_row(
                row_id=4,
                run_id="docs-run",
                stage="documents",
                source_run_id="gold-run",
            )
        }

        with pytest.raises(LineageAuditError, match="run_not_found@gold:gold-run"):
            get_run_lineage(
                _make_settings(),
                stage="documents",
                run_id="docs-run",
                conn_factory=lambda _s: MappingConnection(rows),
            )

    def test_raises_when_dataset_prefix_breaks_lineage(self) -> None:
        rows = {
            ("silver", "silver-run"): _build_row(
                row_id=2,
                run_id="silver-run",
                stage="silver",
                source_run_id="bronze-run",
                dataset_prefix="csgo-a",
            ),
            ("bronze", "bronze-run"): _build_row(
                row_id=1,
                run_id="bronze-run",
                stage="bronze",
                dataset_prefix="csgo-b",
            ),
        }

        with pytest.raises(LineageAuditError, match="dataset_prefix_mismatch@bronze:bronze-run"):
            get_run_lineage(
                _make_settings(),
                stage="silver",
                run_id="silver-run",
                conn_factory=lambda _s: MappingConnection(rows),
            )

    def test_raises_when_cycle_is_detected(self) -> None:
        rows = {
            ("silver", "silver-run"): _build_row(
                row_id=2,
                run_id="silver-run",
                stage="silver",
                source_run_id="bronze-run",
            ),
            ("bronze", "bronze-run"): _build_row(
                row_id=1,
                run_id="bronze-run",
                stage="bronze",
                source_run_id="silver-run",
            ),
        }

        with pytest.raises(LineageAuditError, match="cycle_detected@bronze:bronze-run"):
            get_run_lineage(
                _make_settings(),
                stage="silver",
                run_id="silver-run",
                conn_factory=lambda _s: MappingConnection(rows),
            )
