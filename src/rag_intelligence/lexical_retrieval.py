"""Lexical (full-text) search over ML training metadata in PostgreSQL."""

from __future__ import annotations

import json
import re
import unicodedata
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

_FALLBACK_SQL = """\
SELECT
    run_id,
    model_name,
    roc_auc, f1, balanced_accuracy, log_loss_val, brier,
    feature_importances,
    search_text,
    created_at
FROM training_runs
{model_filter_clause}
ORDER BY created_at DESC, model_name ASC
LIMIT %s;
"""

_STOP_WORDS = {
    "a",
    "an",
    "as",
    "best",
    "better",
    "compare",
    "comparacao",
    "comparar",
    "comparison",
    "de",
    "do",
    "dos",
    "e",
    "em",
    "foi",
    "maior",
    "mais",
    "melhor",
    "modelo",
    "model",
    "o",
    "os",
    "para",
    "performou",
    "performance",
    "qual",
    "que",
    "resultado",
    "top",
}


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.lower())
    without_accents = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", without_accents).strip()


def _query_tokens(query: str) -> set[str]:
    return {token for token in _normalize_text(query).split() if token not in _STOP_WORDS}


def _performance_intent(query: str) -> bool:
    normalized = _normalize_text(query)
    comparative_phrases = (
        "mais performou",
        "melhor desempenho",
        "melhor modelo",
        "best model",
        "best performer",
        "highest roc auc",
        "highest f1",
    )
    if any(phrase in normalized for phrase in comparative_phrases):
        return True

    tokens = set(normalized.split())
    return bool(
        {
            "best",
            "better",
            "desempenho",
            "melhor",
            "performou",
            "performance",
            "ranking",
            "top",
        }
        & tokens
    )


def _metric_preference(query: str) -> tuple[int, bool]:
    normalized = _normalize_text(query)
    tokens = set(normalized.split())
    if "brier" in tokens:
        return 6, False
    if "log loss" in normalized or {"log", "loss"} <= tokens:
        return 5, False
    if "balanced accuracy" in normalized or {"balanced", "accuracy"} <= tokens:
        return 4, True
    if "f1" in tokens:
        return 3, True
    if "roc auc" in normalized or "roc" in tokens or "auc" in tokens:
        return 2, True
    return 2, True


def _model_token_overlap(query_tokens: set[str], model_name: str) -> int:
    model_tokens = set(_normalize_text(model_name).split())
    return len(query_tokens & model_tokens)


def _search_token_overlap(query_tokens: set[str], search_text: str, model_name: str) -> int:
    searchable_tokens = set(_normalize_text(f"{model_name} {search_text}").split())
    return len(query_tokens & searchable_tokens)


def _row_to_result(
    row: tuple[Any, ...],
    *,
    rank: int,
    score: float,
) -> LexicalSearchResult:
    fi_raw = row[7]
    if isinstance(fi_raw, str):
        fi_raw = json.loads(fi_raw)
    created_at = row[9]
    if isinstance(created_at, datetime):
        created_at = created_at.isoformat()

    return LexicalSearchResult(
        rank=rank,
        score=score,
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


def _fallback_search(
    rows: list[tuple[Any, ...]],
    *,
    query: str,
    top_k: int,
) -> list[LexicalSearchResult]:
    query_tokens = _query_tokens(query)
    performance_query = _performance_intent(query)
    metric_index, higher_is_better = _metric_preference(query)

    scored_rows = [
        (
            row,
            _model_token_overlap(query_tokens, str(row[1])),
            _search_token_overlap(query_tokens, str(row[8]), str(row[1])),
        )
        for row in rows
    ]

    explicit_model_rows = [item for item in scored_rows if item[1] > 0]
    candidate_rows = explicit_model_rows or scored_rows

    if performance_query:

        def performance_key(item: tuple[tuple[Any, ...], int, int]) -> tuple[float, int, str]:
            row, model_overlap, text_overlap = item
            metric_value = row[metric_index]
            if metric_value is None:
                primary = float("-inf")
            else:
                metric_score = float(metric_value)
                primary = metric_score if higher_is_better else -metric_score
            return (primary, model_overlap + text_overlap, str(row[1]))

        ranked_rows = sorted(candidate_rows, key=performance_key, reverse=True)
        results: list[LexicalSearchResult] = []
        for rank, (row, model_overlap, text_overlap) in enumerate(ranked_rows[:top_k], start=1):
            metric_value = row[metric_index]
            if metric_value is None:
                score = float(model_overlap + text_overlap)
            elif higher_is_better:
                score = float(metric_value)
            else:
                score = 1.0 / (1.0 + float(metric_value))
            results.append(_row_to_result(row, rank=rank, score=score))
        return results

    ranked_rows = sorted(
        [item for item in candidate_rows if item[2] > 0],
        key=lambda item: (item[2], item[1], str(item[0][9]), str(item[0][1])),
        reverse=True,
    )
    return [
        _row_to_result(row, rank=rank, score=float(text_overlap))
        for rank, (row, _model_overlap, text_overlap) in enumerate(ranked_rows[:top_k], start=1)
    ]


def lexical_search(
    query: str,
    *,
    top_k: int = 5,
    model_filter: str | None = None,
    settings: AppSettings | None = None,
    conn_factory: Any = None,
) -> list[LexicalSearchResult]:
    """Run full-text search over training_runs using ts_rank."""
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
        if not rows:
            fallback_sql = _FALLBACK_SQL.format(model_filter_clause=model_filter_clause)
            fallback_params: list[Any] = []
            if model_filter:
                fallback_params.append(model_filter)
            fallback_params.append(max(top_k * 10, 20))
            cursor.execute(fallback_sql, tuple(fallback_params))
            rows = cursor.fetchall()
        cursor.close()
    finally:
        conn.close()

    if rows and len(rows[0]) == 11:
        return [
            _row_to_result(row, rank=rank, score=float(row[10]))
            for rank, row in enumerate(rows[:top_k], start=1)
        ]

    return _fallback_search(rows, query=query, top_k=top_k)
