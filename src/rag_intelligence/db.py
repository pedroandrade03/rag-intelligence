from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import psycopg2
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.vector_stores.postgres.base import PGType

if TYPE_CHECKING:
    from rag_intelligence.settings import AppSettings


PGVECTOR_SCHEMA_NAME = "public"
PGVECTOR_METADATA_INDEXED_KEYS: set[tuple[str, PGType]] = {
    ("embedding_run_id", "text"),
    ("event_type", "text"),
    ("map", "text"),
    ("file", "text"),
    ("round", "integer"),
    ("document_tier", "text"),
    ("weapon", "text"),
}


@dataclass(frozen=True)
class PGVectorMetadataIndex:
    name: str
    create_sql: str
    legacy_index_name: str


def _validate_identifier(value: str, *, label: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        raise ValueError(f"{label} cannot be empty")
    if normalized[0].isdigit():
        raise ValueError(f"Invalid SQL identifier for {label}: {value}")
    if not all(character.isalnum() or character == "_" for character in normalized):
        raise ValueError(f"Invalid SQL identifier for {label}: {value}")
    return normalized


def build_pgvector_data_table_name(pg_table_name: str) -> str:
    normalized_table_name = _validate_identifier(pg_table_name, label="PG_TABLE_NAME")
    return f"data_{normalized_table_name}"


def build_pgvector_metadata_indexes(
    pg_table_name: str,
) -> tuple[PGVectorMetadataIndex, ...]:
    normalized_table_name = _validate_identifier(pg_table_name, label="PG_TABLE_NAME")
    data_table_name = build_pgvector_data_table_name(normalized_table_name)
    legacy_prefix = f"{normalized_table_name}_idx"
    index_definitions = [
        PGVectorMetadataIndex(
            name=f"{data_table_name}_meta_embedding_run_id_idx",
            create_sql=(
                f"CREATE INDEX IF NOT EXISTS {data_table_name}_meta_embedding_run_id_idx "
                f"ON {PGVECTOR_SCHEMA_NAME}.{data_table_name} "
                "USING btree ((metadata_->>'embedding_run_id'));"
            ),
            legacy_index_name=f"{legacy_prefix}_embedding_run_id_text",
        ),
        PGVectorMetadataIndex(
            name=f"{data_table_name}_meta_event_type_idx",
            create_sql=(
                f"CREATE INDEX IF NOT EXISTS {data_table_name}_meta_event_type_idx "
                f"ON {PGVECTOR_SCHEMA_NAME}.{data_table_name} "
                "USING btree ((metadata_->>'event_type'));"
            ),
            legacy_index_name=f"{legacy_prefix}_event_type_text",
        ),
        PGVectorMetadataIndex(
            name=f"{data_table_name}_meta_map_idx",
            create_sql=(
                f"CREATE INDEX IF NOT EXISTS {data_table_name}_meta_map_idx "
                f"ON {PGVECTOR_SCHEMA_NAME}.{data_table_name} "
                "USING btree ((metadata_->>'map'));"
            ),
            legacy_index_name=f"{legacy_prefix}_map_text",
        ),
        PGVectorMetadataIndex(
            name=f"{data_table_name}_meta_file_idx",
            create_sql=(
                f"CREATE INDEX IF NOT EXISTS {data_table_name}_meta_file_idx "
                f"ON {PGVECTOR_SCHEMA_NAME}.{data_table_name} "
                "USING btree ((metadata_->>'file'));"
            ),
            legacy_index_name=f"{legacy_prefix}_file_text",
        ),
        PGVectorMetadataIndex(
            name=f"{data_table_name}_meta_round_int_idx",
            create_sql=(
                f"CREATE INDEX IF NOT EXISTS {data_table_name}_meta_round_int_idx "
                f"ON {PGVECTOR_SCHEMA_NAME}.{data_table_name} "
                "USING btree (((metadata_->>'round')::integer)) "
                "WHERE (metadata_->>'round') ~ '^-?[0-9]+$';"
            ),
            legacy_index_name=f"{legacy_prefix}_round_integer",
        ),
        PGVectorMetadataIndex(
            name=f"{data_table_name}_meta_document_tier_idx",
            create_sql=(
                f"CREATE INDEX IF NOT EXISTS {data_table_name}_meta_document_tier_idx "
                f"ON {PGVECTOR_SCHEMA_NAME}.{data_table_name} "
                "USING btree ((metadata_->>'document_tier'));"
            ),
            legacy_index_name=f"{legacy_prefix}_document_tier_text",
        ),
        PGVectorMetadataIndex(
            name=f"{data_table_name}_meta_weapon_idx",
            create_sql=(
                f"CREATE INDEX IF NOT EXISTS {data_table_name}_meta_weapon_idx "
                f"ON {PGVECTOR_SCHEMA_NAME}.{data_table_name} "
                "USING btree ((metadata_->>'weapon'));"
            ),
            legacy_index_name=f"{legacy_prefix}_weapon_text",
        ),
    ]
    return tuple(index_definitions)


def default_conn_factory(settings: AppSettings) -> Any:
    return psycopg2.connect(
        host=settings.pg_host,
        port=settings.pg_port,
        user=settings.pg_user,
        password=settings.pg_password,
        dbname=settings.pg_database,
    )


def ensure_pgvector_storage_contract(
    settings: AppSettings,
    *,
    conn_factory: Any = None,
) -> list[str]:
    indexes = build_pgvector_metadata_indexes(settings.pg_table_name)
    factory = conn_factory or default_conn_factory
    conn = factory(settings)
    try:
        cursor = conn.cursor()
        for definition in indexes:
            cursor.execute(
                f"DROP INDEX IF EXISTS {PGVECTOR_SCHEMA_NAME}.{definition.legacy_index_name};"
            )
            cursor.execute(definition.create_sql)
        conn.commit()
        cursor.close()
    finally:
        conn.close()
    return [d.name for d in indexes]


def pgvector_data_table_exists(
    settings: AppSettings,
    *,
    conn_factory: Any = None,
) -> bool:
    factory = conn_factory or default_conn_factory
    conn = factory(settings)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT to_regclass(%s);",
            (f"{PGVECTOR_SCHEMA_NAME}.{build_pgvector_data_table_name(settings.pg_table_name)}",),
        )
        row = cursor.fetchone()
        cursor.close()
    finally:
        conn.close()
    return bool(row and row[0])


def create_vector_store(
    settings: AppSettings,
    *,
    perform_setup: bool = True,
) -> PGVectorStore:
    return PGVectorStore.from_params(
        host=settings.pg_host,
        port=str(settings.pg_port),
        database=settings.pg_database,
        user=settings.pg_user,
        password=settings.pg_password,
        table_name=settings.pg_table_name,
        embed_dim=settings.pg_embed_dim,
        perform_setup=perform_setup,
        use_jsonb=True,
        indexed_metadata_keys=PGVECTOR_METADATA_INDEXED_KEYS,
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )
