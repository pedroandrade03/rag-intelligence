from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psycopg2

from rag_intelligence.config import ConfigError

_STAGE_CHECK = "'bronze', 'silver', 'gold', 'documents', 'embeddings'"

_CREATE_TABLE = f"""\
CREATE TABLE IF NOT EXISTS dataset_runs (
    id                 BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id             TEXT        NOT NULL,
    stage              TEXT        NOT NULL
                                   CHECK (stage IN ({_STAGE_CHECK})),
    status             TEXT        NOT NULL DEFAULT 'completed'
                                   CHECK (status IN ('completed', 'failed')),
    dataset_prefix     TEXT        NOT NULL,
    bucket             TEXT        NOT NULL,
    source_run_id      TEXT,
    events_key         TEXT,
    artifact_prefix    TEXT,
    manifest_key       TEXT,
    quality_report_key TEXT,
    files_processed    INTEGER     NOT NULL DEFAULT 0,
    rows_read          INTEGER     NOT NULL DEFAULT 0,
    rows_output        INTEGER     NOT NULL DEFAULT 0,
    quality_summary    JSONB,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, stage)
);
"""

_MIGRATIONS = (
    "ALTER TABLE dataset_runs ADD COLUMN IF NOT EXISTS artifact_prefix TEXT;",
    "ALTER TABLE dataset_runs ADD COLUMN IF NOT EXISTS manifest_key TEXT;",
    "ALTER TABLE dataset_runs DROP CONSTRAINT IF EXISTS dataset_runs_stage_check;",
    f"ALTER TABLE dataset_runs ADD CONSTRAINT dataset_runs_stage_check "
    f"CHECK (stage IN ({_STAGE_CHECK}));",
)

_UPSERT = """\
INSERT INTO dataset_runs (
    run_id, stage, status, dataset_prefix, bucket,
    source_run_id, events_key, artifact_prefix, manifest_key, quality_report_key,
    files_processed, rows_read, rows_output, quality_summary
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (run_id, stage) DO UPDATE SET
    status             = EXCLUDED.status,
    dataset_prefix     = EXCLUDED.dataset_prefix,
    bucket             = EXCLUDED.bucket,
    source_run_id      = EXCLUDED.source_run_id,
    events_key         = EXCLUDED.events_key,
    artifact_prefix    = EXCLUDED.artifact_prefix,
    manifest_key       = EXCLUDED.manifest_key,
    quality_report_key = EXCLUDED.quality_report_key,
    files_processed    = EXCLUDED.files_processed,
    rows_read          = EXCLUDED.rows_read,
    rows_output        = EXCLUDED.rows_output,
    quality_summary    = EXCLUDED.quality_summary
RETURNING id, created_at;
"""

_SELECT_LATEST = """\
SELECT id, run_id, stage, status, dataset_prefix, bucket,
       source_run_id, events_key, artifact_prefix, manifest_key, quality_report_key,
       files_processed, rows_read, rows_output, quality_summary, created_at
FROM dataset_runs
WHERE stage = %s AND status = 'completed'
ORDER BY created_at DESC
LIMIT 1;
"""


@dataclass(frozen=True)
class MetadataSettings:
    pg_host: str
    pg_port: int
    pg_user: str
    pg_password: str
    pg_database: str

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> MetadataSettings:
        raw = dict(os.environ if env is None else env)
        pg_host = raw.get("PG_HOST", "").strip()
        pg_user = raw.get("PG_USER", "").strip()
        pg_password = raw.get("PG_PASSWORD", "").strip()
        pg_database = raw.get("PG_DATABASE", "").strip()
        if not all([pg_host, pg_user, pg_password, pg_database]):
            raise ConfigError(
                "Missing required PG environment variable(s): "
                "PG_HOST, PG_USER, PG_PASSWORD, PG_DATABASE"
            )
        pg_port_str = raw.get("PG_PORT", "5432").strip()
        try:
            pg_port = int(pg_port_str)
        except ValueError as err:
            raise ConfigError(f"Invalid PG_PORT value: {pg_port_str}") from err
        return cls(
            pg_host=pg_host,
            pg_port=pg_port,
            pg_user=pg_user,
            pg_password=pg_password,
            pg_database=pg_database,
        )


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    stage: str
    dataset_prefix: str
    bucket: str
    status: str = "completed"
    source_run_id: str | None = None
    events_key: str | None = None
    artifact_prefix: str | None = None
    manifest_key: str | None = None
    quality_report_key: str | None = None
    files_processed: int = 0
    rows_read: int = 0
    rows_output: int = 0
    quality_summary: dict[str, Any] | None = None
    created_at: datetime | None = None
    id: int | None = None


def _default_conn_factory(settings: MetadataSettings) -> Any:
    return psycopg2.connect(
        host=settings.pg_host,
        port=settings.pg_port,
        user=settings.pg_user,
        password=settings.pg_password,
        dbname=settings.pg_database,
    )


def ensure_schema(settings: MetadataSettings, *, conn_factory: Any = None) -> None:
    factory = conn_factory or _default_conn_factory
    conn = factory(settings)
    try:
        cur = conn.cursor()
        cur.execute(_CREATE_TABLE)
        for query in _MIGRATIONS:
            cur.execute(query)
        conn.commit()
        cur.close()
    finally:
        conn.close()


def register_run(
    settings: MetadataSettings,
    record: RunRecord,
    *,
    conn_factory: Any = None,
) -> RunRecord:
    factory = conn_factory or _default_conn_factory
    conn = factory(settings)
    try:
        cur = conn.cursor()
        quality_json = json.dumps(record.quality_summary) if record.quality_summary else None
        cur.execute(
            _UPSERT,
            (
                record.run_id,
                record.stage,
                record.status,
                record.dataset_prefix,
                record.bucket,
                record.source_run_id,
                record.events_key,
                record.artifact_prefix,
                record.manifest_key,
                record.quality_report_key,
                record.files_processed,
                record.rows_read,
                record.rows_output,
                quality_json,
            ),
        )
        row = cur.fetchone()
        conn.commit()
        cur.close()
    finally:
        conn.close()

    assert row is not None
    return RunRecord(
        id=row[0],
        created_at=row[1],
        run_id=record.run_id,
        stage=record.stage,
        status=record.status,
        dataset_prefix=record.dataset_prefix,
        bucket=record.bucket,
        source_run_id=record.source_run_id,
        events_key=record.events_key,
        artifact_prefix=record.artifact_prefix,
        manifest_key=record.manifest_key,
        quality_report_key=record.quality_report_key,
        files_processed=record.files_processed,
        rows_read=record.rows_read,
        rows_output=record.rows_output,
        quality_summary=record.quality_summary,
    )


def get_latest_run(
    settings: MetadataSettings,
    stage: str,
    *,
    conn_factory: Any = None,
) -> RunRecord | None:
    factory = conn_factory or _default_conn_factory
    conn = factory(settings)
    try:
        cur = conn.cursor()
        cur.execute(_SELECT_LATEST, (stage,))
        row = cur.fetchone()
        cur.close()
    finally:
        conn.close()

    if row is None:
        return None

    quality_raw = row[14]
    quality_summary = json.loads(quality_raw) if isinstance(quality_raw, str) else quality_raw

    return RunRecord(
        id=row[0],
        run_id=row[1],
        stage=row[2],
        status=row[3],
        dataset_prefix=row[4],
        bucket=row[5],
        source_run_id=row[6],
        events_key=row[7],
        artifact_prefix=row[8],
        manifest_key=row[9],
        quality_report_key=row[10],
        files_processed=row[11],
        rows_read=row[12],
        rows_output=row[13],
        quality_summary=quality_summary,
        created_at=row[15],
    )
