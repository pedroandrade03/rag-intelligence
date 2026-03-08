from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_intelligence.db import build_pgvector_data_table_name, default_conn_factory
from rag_intelligence.settings import AppSettings


@dataclass(frozen=True)
class EmbeddingRunProgress:
    existing_rows: int
    distinct_rows: int
    min_doc_line: int | None
    max_doc_line: int | None


def get_embedding_run_progress(
    app_settings: AppSettings,
    embedding_run_id: str,
    document_run_id: str,
    *,
    conn_factory: Any = None,
) -> EmbeddingRunProgress:
    factory = conn_factory or default_conn_factory
    conn = factory(app_settings)
    try:
        cursor = conn.cursor()
        table_name = build_pgvector_data_table_name(app_settings.pg_table_name)
        cursor.execute(
            (
                f"SELECT COUNT(*), COUNT(DISTINCT node_id), "
                "MIN(split_part(node_id, ':', 2)::bigint), "
                "MAX(split_part(node_id, ':', 2)::bigint) "
                f"FROM public.{table_name} "
                "WHERE metadata_->>'embedding_run_id' = %s "
                "AND node_id LIKE %s;"
            ),
            (embedding_run_id, f"{document_run_id}:%"),
        )
        row = cursor.fetchone()
        cursor.close()
    finally:
        conn.close()

    if row is None:
        return EmbeddingRunProgress(
            existing_rows=0,
            distinct_rows=0,
            min_doc_line=None,
            max_doc_line=None,
        )

    return EmbeddingRunProgress(
        existing_rows=int(row[0] or 0),
        distinct_rows=int(row[1] or 0),
        min_doc_line=int(row[2]) if row[2] is not None else None,
        max_doc_line=int(row[3]) if row[3] is not None else None,
    )


def _validate_resume_progress(progress: EmbeddingRunProgress) -> int:
    if progress.existing_rows <= 0:
        return 0
    if progress.min_doc_line != 1:
        raise ValueError(
            "Cannot resume embedding run because existing rows do not start at document 1"
        )
    if progress.distinct_rows != progress.existing_rows:
        raise ValueError("Cannot resume embedding run because existing rows contain duplicates")
    if progress.max_doc_line != progress.existing_rows:
        raise ValueError("Cannot resume embedding run because existing rows are not contiguous")
    if progress.max_doc_line is None:
        raise ValueError("Cannot resume embedding run because the existing progress is incomplete")
    return progress.max_doc_line


def _ensure_vector_store_initialized(vector_store: Any) -> None:
    initialize = getattr(vector_store, "_initialize", None)
    if callable(initialize):
        initialize()


def delete_embeddings_for_run(
    app_settings: AppSettings,
    embedding_run_id: str,
    *,
    conn_factory: Any = None,
) -> int:
    factory = conn_factory or default_conn_factory
    conn = factory(app_settings)
    try:
        cursor = conn.cursor()
        table_name = build_pgvector_data_table_name(app_settings.pg_table_name)
        query = f"DELETE FROM public.{table_name} WHERE metadata_->>'embedding_run_id' = %s;"
        cursor.execute(query, (embedding_run_id,))
        deleted_rows = int(getattr(cursor, "rowcount", 0) or 0)
        conn.commit()
        cursor.close()
    finally:
        conn.close()
    return deleted_rows
