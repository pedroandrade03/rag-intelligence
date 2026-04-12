"""PostgreSQL full-text search layer for ML training results."""

from __future__ import annotations

import json
import logging
from typing import Any

from rag_intelligence.db import default_conn_factory
from rag_intelligence.settings import AppSettings

LOGGER = logging.getLogger(__name__)

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS training_runs (
    run_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    experiment TEXT NOT NULL,
    roc_auc DOUBLE PRECISION,
    f1 DOUBLE PRECISION,
    balanced_accuracy DOUBLE PRECISION,
    log_loss_val DOUBLE PRECISION,
    brier DOUBLE PRECISION,
    train_rows INTEGER,
    test_rows INTEGER,
    feature_count INTEGER,
    test_size DOUBLE PRECISION,
    feature_importances JSONB,
    map_segment_metrics JSONB,
    half_segment_metrics JSONB,
    params JSONB,
    search_text TEXT NOT NULL,
    search_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', search_text)) STORED,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE (run_id, model_name)
);
"""

_CREATE_INDEXES = [
    (
        "CREATE INDEX IF NOT EXISTS training_runs_search_idx"
        " ON training_runs USING GIN (search_vector);"
    ),
    (
        "CREATE INDEX IF NOT EXISTS training_runs_model_name_idx"
        " ON training_runs USING BTREE (model_name);"
    ),
]

_INSERT_SQL = """\
INSERT INTO training_runs (
    run_id, model_name, experiment,
    roc_auc, f1, balanced_accuracy, log_loss_val, brier,
    train_rows, test_rows, feature_count, test_size,
    feature_importances, map_segment_metrics, half_segment_metrics,
    params, search_text
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (run_id, model_name) DO UPDATE SET
    experiment = EXCLUDED.experiment,
    roc_auc = EXCLUDED.roc_auc,
    f1 = EXCLUDED.f1,
    balanced_accuracy = EXCLUDED.balanced_accuracy,
    log_loss_val = EXCLUDED.log_loss_val,
    brier = EXCLUDED.brier,
    train_rows = EXCLUDED.train_rows,
    test_rows = EXCLUDED.test_rows,
    feature_count = EXCLUDED.feature_count,
    test_size = EXCLUDED.test_size,
    feature_importances = EXCLUDED.feature_importances,
    map_segment_metrics = EXCLUDED.map_segment_metrics,
    half_segment_metrics = EXCLUDED.half_segment_metrics,
    params = EXCLUDED.params,
    search_text = EXCLUDED.search_text;
"""


def ensure_training_schema(
    settings: AppSettings | None = None,
    *,
    conn_factory: Any = None,
) -> None:
    """Create the training_runs table and indexes if they don't exist."""
    factory = conn_factory or default_conn_factory
    s = settings or AppSettings.from_env()
    conn = factory(s)
    try:
        cursor = conn.cursor()
        cursor.execute(_CREATE_TABLE)
        for idx_sql in _CREATE_INDEXES:
            cursor.execute(idx_sql)
        conn.commit()
        cursor.close()
    finally:
        conn.close()


def _build_search_text(
    model_name: str,
    metrics: dict[str, float],
    feature_columns: list[str],
    segments: list[str],
) -> str:
    """Assemble full-text search payload from model metadata."""
    parts = [model_name.replace("_", " ")]
    for k, v in metrics.items():
        parts.append(f"{k} {v:.4f}")
    parts.extend(feature_columns)
    parts.extend(segments)
    return " ".join(parts)


def store_training_result(
    *,
    training_output: Any,
    feature_importances: dict[str, dict[str, float]],
    experiment_name: str,
    run_id: str,
    test_size: float,
    model_filter: str | None = None,
    settings: AppSettings | None = None,
    conn_factory: Any = None,
) -> None:
    """Store training metrics in PostgreSQL for FTS retrieval."""
    factory = conn_factory or default_conn_factory
    s = settings or AppSettings.from_env()
    conn = factory(s)
    try:
        cursor = conn.cursor()
        for _, row in training_output.metrics.iterrows():
            model_name = row["model"]
            metrics = {
                "roc_auc": float(row["roc_auc"]),
                "f1": float(row["f1"]),
                "balanced_accuracy": float(row["balanced_accuracy"]),
                "log_loss": float(row["log_loss"]),
                "brier": float(row["brier"]),
            }

            segments: list[str] = []
            if model_name in training_output.map_segment_metrics:
                map_df = training_output.map_segment_metrics[model_name]
                if not map_df.empty:
                    segments.extend(map_df["segment"].tolist())
            if model_name in training_output.half_segment_metrics:
                half_df = training_output.half_segment_metrics[model_name]
                if not half_df.empty:
                    segments.extend(half_df["segment"].tolist())

            search_text = _build_search_text(
                model_name, metrics, training_output.feature_columns, segments
            )

            map_seg_json = None
            if model_name in training_output.map_segment_metrics:
                map_seg_json = json.dumps(
                    training_output.map_segment_metrics[model_name].to_dict(orient="records")
                )
            half_seg_json = None
            if model_name in training_output.half_segment_metrics:
                half_seg_json = json.dumps(
                    training_output.half_segment_metrics[model_name].to_dict(orient="records")
                )

            fi_json = json.dumps(feature_importances.get(model_name, {}))

            cursor.execute(
                _INSERT_SQL,
                (
                    run_id,
                    model_name,
                    experiment_name,
                    metrics["roc_auc"],
                    metrics["f1"],
                    metrics["balanced_accuracy"],
                    metrics["log_loss"],
                    metrics["brier"],
                    training_output.train_rows,
                    training_output.test_rows,
                    len(training_output.feature_columns),
                    test_size,
                    fi_json,
                    map_seg_json,
                    half_seg_json,
                    json.dumps({"model_filter": model_filter}),
                    search_text,
                ),
            )
        conn.commit()
        cursor.close()
        LOGGER.info("Stored %s training results in training_runs", len(training_output.metrics))
    finally:
        conn.close()
