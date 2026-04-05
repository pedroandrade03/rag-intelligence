from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import psycopg2

from rag_intelligence.config import ConfigError

STAGE_ORDER = ("bronze", "silver", "gold", "documents", "embeddings")
PARENT_STAGE_BY_STAGE = {
    "silver": "bronze",
    "gold": "silver",
    "documents": "gold",
    "embeddings": "documents",
}
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

_SELECT_RUN = """\
SELECT id, run_id, stage, status, dataset_prefix, bucket,
       source_run_id, events_key, artifact_prefix, manifest_key, quality_report_key,
       files_processed, rows_read, rows_output, quality_summary, created_at
FROM dataset_runs
WHERE stage = %s AND run_id = %s
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


@dataclass(frozen=True)
class RunEvidence:
    bucket: str
    source_run_id: str | None
    events_key: str | None
    artifact_prefix: str | None
    manifest_key: str | None
    quality_report_key: str | None
    files_processed: int
    rows_read: int
    rows_output: int
    quality_summary: dict[str, Any] | None
    created_at: str | None


@dataclass(frozen=True)
class LineageNode:
    stage: str
    run_id: str
    dataset_prefix: str
    status: str
    evidence: RunEvidence


@dataclass(frozen=True)
class LineageIntegrityIssue:
    code: str
    message: str
    stage: str
    run_id: str


@dataclass(frozen=True)
class RunLineageReport:
    requested_stage: str
    requested_run_id: str
    resolved_chain: list[LineageNode]
    integrity_status: str
    integrity_issues: list[LineageIntegrityIssue]
    evidence: list[RunEvidence]

    def to_dict(self) -> dict[str, Any]:
        resolved_chain = [asdict(node) for node in self.resolved_chain]
        summary = {
            "stage_start": self.resolved_chain[-1].stage if self.resolved_chain else None,
            "stage_end": self.resolved_chain[0].stage if self.resolved_chain else None,
            "hops": max(len(self.resolved_chain) - 1, 0),
            "chain_length": len(self.resolved_chain),
            "chain_complete": _is_chain_complete(self.resolved_chain),
            "artifacts_available": {
                node.stage: _artifact_flags(node.evidence) for node in self.resolved_chain
            },
        }
        return {
            "requested_stage": self.requested_stage,
            "requested_run_id": self.requested_run_id,
            "integrity_status": self.integrity_status,
            "integrity_issues": [asdict(issue) for issue in self.integrity_issues],
            "summary": summary,
            "resolved_chain": resolved_chain,
            "evidence": [asdict(item) for item in self.evidence],
        }


class LineageAuditError(ValueError):
    pass


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

    return _build_run_record(row, quality_summary=quality_summary)


def get_run(
    settings: MetadataSettings,
    *,
    stage: str,
    run_id: str,
    conn_factory: Any = None,
) -> RunRecord | None:
    factory = conn_factory or _default_conn_factory
    conn = factory(settings)
    try:
        cur = conn.cursor()
        cur.execute(_SELECT_RUN, (stage, run_id))
        row = cur.fetchone()
        cur.close()
    finally:
        conn.close()

    if row is None:
        return None

    quality_raw = row[14]
    quality_summary = json.loads(quality_raw) if isinstance(quality_raw, str) else quality_raw
    return _build_run_record(row, quality_summary=quality_summary)


def get_run_lineage(
    settings: MetadataSettings,
    *,
    stage: str,
    run_id: str,
    conn_factory: Any = None,
) -> RunLineageReport:
    if stage not in STAGE_ORDER:
        raise LineageAuditError(f"Unsupported stage for lineage audit: {stage}")

    issues: list[LineageIntegrityIssue] = []
    resolved_chain: list[LineageNode] = []
    evidence: list[RunEvidence] = []
    visited: set[tuple[str, str]] = set()

    current_stage = stage
    current_run_id = run_id
    expected_dataset_prefix: str | None = None

    while True:
        key = (current_stage, current_run_id)
        if key in visited:
            issues.append(
                LineageIntegrityIssue(
                    code="cycle_detected",
                    message=(
                        "Lineage traversal encountered the same stage/run twice, "
                        "indicating a cycle."
                    ),
                    stage=current_stage,
                    run_id=current_run_id,
                )
            )
            break
        visited.add(key)

        record = get_run(
            settings,
            stage=current_stage,
            run_id=current_run_id,
            conn_factory=conn_factory,
        )
        if record is None:
            issues.append(
                LineageIntegrityIssue(
                    code="run_not_found",
                    message="No metadata record was found for the requested stage/run.",
                    stage=current_stage,
                    run_id=current_run_id,
                )
            )
            break

        if expected_dataset_prefix is None:
            expected_dataset_prefix = record.dataset_prefix
        elif record.dataset_prefix != expected_dataset_prefix:
            issues.append(
                LineageIntegrityIssue(
                    code="dataset_prefix_mismatch",
                    message=(
                        "Resolved parent run uses a different dataset_prefix than the "
                        "child lineage chain."
                    ),
                    stage=current_stage,
                    run_id=current_run_id,
                )
            )

        node = _build_lineage_node(record)
        resolved_chain.append(node)
        evidence.append(node.evidence)

        parent_stage = PARENT_STAGE_BY_STAGE.get(current_stage)
        if parent_stage is None:
            if record.source_run_id:
                issues.append(
                    LineageIntegrityIssue(
                        code="cycle_detected",
                        message=(
                            "Terminal lineage stage unexpectedly points to another run, "
                            "indicating a cyclic or malformed chain."
                        ),
                        stage=current_stage,
                        run_id=current_run_id,
                    )
                )
            break

        if not record.source_run_id:
            issues.append(
                LineageIntegrityIssue(
                    code="missing_source_run_id",
                    message="Lineage chain stopped because source_run_id is missing.",
                    stage=current_stage,
                    run_id=current_run_id,
                )
            )
            break

        current_stage = parent_stage
        current_run_id = record.source_run_id

    integrity_status = "ok" if not issues and _is_chain_complete(resolved_chain) else "broken"
    if not resolved_chain:
        raise LineageAuditError(
            f"Unable to build lineage for stage={stage} run_id={run_id}: no metadata found"
        )
    if issues:
        issue_summary = "; ".join(f"{issue.code}@{issue.stage}:{issue.run_id}" for issue in issues)
        raise LineageAuditError(
            f"Lineage integrity check failed for stage={stage} run_id={run_id}: {issue_summary}"
        )

    return RunLineageReport(
        requested_stage=stage,
        requested_run_id=run_id,
        resolved_chain=resolved_chain,
        integrity_status=integrity_status,
        integrity_issues=issues,
        evidence=evidence,
    )


def _build_run_record(
    row: tuple[Any, ...],
    *,
    quality_summary: dict[str, Any] | None,
) -> RunRecord:
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


def _build_lineage_node(record: RunRecord) -> LineageNode:
    return LineageNode(
        stage=record.stage,
        run_id=record.run_id,
        dataset_prefix=record.dataset_prefix,
        status=record.status,
        evidence=RunEvidence(
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
            created_at=record.created_at.isoformat() if record.created_at else None,
        ),
    )


def _artifact_flags(evidence: RunEvidence) -> dict[str, bool]:
    return {
        "has_events_key": bool(evidence.events_key),
        "has_artifact_prefix": bool(evidence.artifact_prefix),
        "has_manifest_key": bool(evidence.manifest_key),
        "has_quality_report_key": bool(evidence.quality_report_key),
        "has_quality_summary": bool(evidence.quality_summary),
    }


def _is_chain_complete(chain: list[LineageNode]) -> bool:
    if not chain:
        return False
    start_index = STAGE_ORDER.index(chain[0].stage)
    expected = tuple(reversed(STAGE_ORDER[: start_index + 1]))
    actual = tuple(node.stage for node in chain)
    return actual == expected and chain[-1].stage == "bronze"
