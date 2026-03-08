from __future__ import annotations

import pytest

from conftest import FakeConnection, FakeCursor
from rag_intelligence.db import (
    PGVECTOR_METADATA_INDEXED_KEYS,
    build_pgvector_data_table_name,
    build_pgvector_metadata_indexes,
    create_vector_store,
    ensure_pgvector_storage_contract,
    pgvector_data_table_exists,
)
from rag_intelligence.settings import AppSettings


def build_app_settings() -> AppSettings:
    return AppSettings.from_env(
        {
            "PG_HOST": "localhost",
            "PG_PORT": "5432",
            "PG_USER": "raguser",
            "PG_PASSWORD": "ragpassword",
            "PG_DATABASE": "ragdb",
            "PG_TABLE_NAME": "rag_embeddings",
            "PG_EMBED_DIM": "768",
            "DEFAULT_EMBED_MODEL": "ollama/nomic-embed-text",
            "DEFAULT_LLM": "ollama/qwen2.5",
            "OLLAMA_BASE_URL": "http://localhost:11434",
        }
    )


def test_build_pgvector_data_table_name_uses_data_prefix() -> None:
    assert build_pgvector_data_table_name("rag_embeddings") == "data_rag_embeddings"


def test_build_pgvector_data_table_name_rejects_invalid_identifier() -> None:
    with pytest.raises(ValueError, match="Invalid SQL identifier"):
        build_pgvector_data_table_name("rag-embeddings")


def test_build_pgvector_metadata_indexes_returns_five_deterministic_indexes() -> None:
    indexes = build_pgvector_metadata_indexes("rag_embeddings")

    assert [index.name for index in indexes] == [
        "data_rag_embeddings_meta_embedding_run_id_idx",
        "data_rag_embeddings_meta_event_type_idx",
        "data_rag_embeddings_meta_map_idx",
        "data_rag_embeddings_meta_file_idx",
        "data_rag_embeddings_meta_round_int_idx",
    ]
    assert indexes[0].legacy_index_name == "rag_embeddings_idx_embedding_run_id_text"


def test_round_index_uses_safe_partial_integer_cast() -> None:
    round_index = build_pgvector_metadata_indexes("rag_embeddings")[-1]

    assert "((metadata_->>'round')::integer)" in round_index.create_sql
    assert "WHERE (metadata_->>'round') ~ '^-?[0-9]+$'" in round_index.create_sql


def test_create_vector_store_declares_indexed_metadata_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_from_params(**kwargs: object) -> str:
        captured.update(kwargs)
        return "fake-store"

    monkeypatch.setattr("rag_intelligence.db.PGVectorStore.from_params", fake_from_params)

    result = create_vector_store(build_app_settings())

    assert result == "fake-store"
    assert captured["indexed_metadata_keys"] == set(PGVECTOR_METADATA_INDEXED_KEYS)
    assert captured["perform_setup"] is True
    assert captured["use_jsonb"] is True


def test_create_vector_store_can_disable_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_from_params(**kwargs: object) -> str:
        captured.update(kwargs)
        return "fake-store"

    monkeypatch.setattr("rag_intelligence.db.PGVectorStore.from_params", fake_from_params)

    create_vector_store(build_app_settings(), perform_setup=False)

    assert captured["perform_setup"] is False


def test_ensure_pgvector_storage_contract_executes_idempotent_bootstrap() -> None:
    cursor = FakeCursor()
    conn = FakeConnection(cursor=cursor)

    ensured = ensure_pgvector_storage_contract(
        build_app_settings(),
        conn_factory=lambda _settings: conn,
    )

    assert ensured == [
        "data_rag_embeddings_meta_embedding_run_id_idx",
        "data_rag_embeddings_meta_event_type_idx",
        "data_rag_embeddings_meta_map_idx",
        "data_rag_embeddings_meta_file_idx",
        "data_rag_embeddings_meta_round_int_idx",
    ]
    assert conn.committed is True
    assert conn.closed is True
    executed_queries = [query for query, _params in cursor.queries]
    assert executed_queries[0] == (
        "DROP INDEX IF EXISTS public.rag_embeddings_idx_embedding_run_id_text;"
    )
    assert (
        "CREATE INDEX IF NOT EXISTS data_rag_embeddings_meta_round_int_idx "
        "ON public.data_rag_embeddings USING btree "
        "(((metadata_->>'round')::integer)) "
        "WHERE (metadata_->>'round') ~ '^-?[0-9]+$';"
    ) in executed_queries
    assert len(executed_queries) == 10


def test_pgvector_data_table_exists_checks_regclass() -> None:
    cursor = FakeCursor()
    cursor.set_result(rows=[("public.data_rag_embeddings",)])
    conn = FakeConnection(cursor=cursor)

    exists = pgvector_data_table_exists(
        build_app_settings(),
        conn_factory=lambda _settings: conn,
    )

    assert exists is True
    query, params = cursor.queries[0]
    assert query == "SELECT to_regclass(%s);"
    assert params == ("public.data_rag_embeddings",)
    assert conn.closed is True


def test_pgvector_data_table_exists_returns_false_when_missing() -> None:
    cursor = FakeCursor()
    cursor.set_result(rows=[(None,)])
    conn = FakeConnection(cursor=cursor)

    exists = pgvector_data_table_exists(
        build_app_settings(),
        conn_factory=lambda _settings: conn,
    )

    assert exists is False
