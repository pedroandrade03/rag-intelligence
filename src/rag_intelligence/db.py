from __future__ import annotations

from typing import TYPE_CHECKING

from llama_index.vector_stores.postgres import PGVectorStore

if TYPE_CHECKING:
    from rag_intelligence.settings import AppSettings


def create_vector_store(settings: AppSettings) -> PGVectorStore:
    return PGVectorStore.from_params(
        host=settings.pg_host,
        port=str(settings.pg_port),
        database=settings.pg_database,
        user=settings.pg_user,
        password=settings.pg_password,
        table_name=settings.pg_table_name,
        embed_dim=settings.pg_embed_dim,
        use_jsonb=True,
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )
