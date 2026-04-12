"""Lexical (full-text) search over ML training metadata in PostgreSQL."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from rag_intelligence.db import default_conn_factory
from rag_intelligence.settings import AppSettings


@dataclass(frozen=True)
class LexicalSearchResult:
    rank: int
    score: float
    run_id: str
    model_name: str
    roc_auc: float | None
    f1: float | None
    balanced_accuracy: float | None
    log_loss_val: float | None
    brier: float | None
    feature_importances: dict[str, float] | None
    text_summary: str
    created_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SEARCH_SQL = """\
SELECT
    run_id,
    model_name,
    roc_auc, f1, balanced_accuracy, log_loss_val, brier,
    feature_importances,
    search_text,
    created_at,
    ts_rank(search_vector, websearch_to_tsquery('english', %s)) AS rank_score
FROM training_runs
WHERE search_vector @@ websearch_to_tsquery('english', %s)
{model_filter_clause}
ORDER BY rank_score DESC, created_at DESC, model_name ASC
LIMIT %s;
"""


def lexical_search(
    query: str,
    *,
    top_k: int = 5,
    model_filter: str | None = None,
    settings: AppSettings | None = None,
    conn_factory: Any = None,
) -> list[LexicalSearchResult]:
    """Run full-text search over training_runs using ts_rank."""
    import json

    factory = conn_factory or default_conn_factory
    s = settings or AppSettings.from_env()
    conn = factory(s)

    model_filter_clause = ""
    params: list[Any] = [query, query]
    if model_filter:
        model_filter_clause = "AND model_name = %s"
        params.append(model_filter)
    params.append(top_k)

    sql = _SEARCH_SQL.format(model_filter_clause=model_filter_clause)

    try:
        cursor = conn.cursor()
        cursor.execute(sql, tuple(params))
        rows = cursor.fetchall()
        cursor.close()
    finally:
        conn.close()

    results: list[LexicalSearchResult] = []
    for rank, row in enumerate(rows, start=1):
        fi_raw = row[7]
        if isinstance(fi_raw, str):
            fi_raw = json.loads(fi_raw)
        created_at = row[9]
        if isinstance(created_at, datetime):
            created_at = created_at.isoformat()

        results.append(
            LexicalSearchResult(
                rank=rank,
                score=float(row[10]),
                run_id=row[0],
                model_name=row[1],
                roc_auc=float(row[2]) if row[2] is not None else None,
                f1=float(row[3]) if row[3] is not None else None,
                balanced_accuracy=float(row[4]) if row[4] is not None else None,
                log_loss_val=float(row[5]) if row[5] is not None else None,
                brier=float(row[6]) if row[6] is not None else None,
                feature_importances=fi_raw if isinstance(fi_raw, dict) else None,
                text_summary=row[8],
                created_at=created_at if isinstance(created_at, str) else None,
            )
        )
    return results
