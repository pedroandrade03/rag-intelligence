from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from tests.conftest import FakeConnection, FakeCursor

from rag_intelligence.config import ConfigError
from rag_intelligence.metadata import (
    MetadataSettings,
    RunRecord,
    ensure_schema,
    get_latest_run,
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
        assert len(queries) == 1
        assert "CREATE TABLE IF NOT EXISTS dataset_runs" in queries[0][0]


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
        assert params[11] == json.dumps({"missing_weapon": 10})

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
        assert params[11] is None


class TestGetLatestRun:
    def test_returns_record_when_exists(self) -> None:
        created = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
        cursor = FakeCursor()
        cursor.set_result([
            (
                42,
                "20250115T120000Z",
                "gold",
                "completed",
                "csgo",
                "gold",
                "src-run",
                "events.parquet",
                "quality_report.json",
                3,
                1000,
                950,
                json.dumps({"missing_weapon": 10}),
                created,
            )
        ])
        conn = FakeConnection(cursor=cursor)

        result = get_latest_run(_make_settings(), "gold", conn_factory=lambda _s: conn)

        assert result is not None
        assert result.id == 42
        assert result.run_id == "20250115T120000Z"
        assert result.stage == "gold"
        assert result.quality_summary == {"missing_weapon": 10}
        assert conn.closed

    def test_returns_none_when_empty(self) -> None:
        conn = FakeConnection()
        result = get_latest_run(_make_settings(), "gold", conn_factory=lambda _s: conn)
        assert result is None
        assert conn.closed

    def test_handles_dict_quality_summary(self) -> None:
        cursor = FakeCursor()
        cursor.set_result([
            (
                1, "r1", "silver", "completed", "ds", "silver",
                None, None, "qr.json",
                1, 100, 90,
                {"dup": 5},
                datetime.now(tz=UTC),
            )
        ])
        conn = FakeConnection(cursor=cursor)
        result = get_latest_run(_make_settings(), "silver", conn_factory=lambda _s: conn)
        assert result is not None
        assert result.quality_summary == {"dup": 5}
