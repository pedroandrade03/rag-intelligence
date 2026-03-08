from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
from llama_index.core.schema import Document

from conftest import FakeConnection, FakeCursor, FakeMinio
from rag_intelligence.config import EmbeddingSettings
from rag_intelligence.db import build_pgvector_data_table_name
from rag_intelligence.embeddings import (
    build_document_from_json,
    build_embedding_manifest_key,
    build_embedding_object_prefix,
    build_embedding_quality_report_key,
    delete_embeddings_for_run,
    run_embedding_ingest,
)
from rag_intelligence.settings import AppSettings


def build_settings(
    batch_size: int = 2,
    *,
    max_documents: int | None = None,
) -> EmbeddingSettings:
    return EmbeddingSettings(
        minio_endpoint="localhost:9000",
        minio_access_key="minioadmin",
        minio_secret_key="minioadmin",
        minio_secure=False,
        document_bucket="gold",
        embedding_report_bucket="gold",
        document_dataset_prefix="csgo-matchmaking-damage",
        embedding_dataset_prefix="csgo-matchmaking-damage",
        document_source_run_id="20260308T180500Z",
        embedding_run_id="20260308T181000Z",
        embedding_batch_size=batch_size,
        embedding_max_documents=max_documents,
    )


def build_app_settings(*, embed_dim: int = 3, table_name: str = "rag_embeddings") -> AppSettings:
    return AppSettings.from_env(
        {
            "PG_HOST": "localhost",
            "PG_PORT": "5432",
            "PG_USER": "raguser",
            "PG_PASSWORD": "ragpassword",
            "PG_DATABASE": "ragdb",
            "PG_TABLE_NAME": table_name,
            "PG_EMBED_DIM": str(embed_dim),
            "DEFAULT_EMBED_MODEL": "fake/embed",
            "DEFAULT_LLM": "ollama/qwen2.5",
            "OLLAMA_BASE_URL": "http://localhost:11434",
        }
    )


def _document_manifest_key() -> str:
    return "csgo-matchmaking-damage/20260308T180500Z/documents/manifest.json"


def _document_part_key(index: int) -> str:
    return f"csgo-matchmaking-damage/20260308T180500Z/documents/part-{index:05d}.jsonl"


def _document_payload(
    doc_id: str,
    *,
    event_type: str = "damage",
    file_name: str = "demo_1",
    round_value: int = 1,
    source_file: str = "damage.csv",
) -> bytes:
    record = {
        "doc_id": doc_id,
        "text": f"Evento {event_type} em map=de_mirage file={file_name} round={round_value}.",
        "metadata": {
            "doc_id": doc_id,
            "source_run_id": "20260306T025119Z",
            "document_run_id": "20260308T180500Z",
            "dataset_prefix": "csgo-matchmaking-damage",
            "event_type": event_type,
            "file": file_name,
            "round": round_value,
            "map": "de_mirage",
            "source_file": source_file,
            "source_line_number": 2,
        },
    }
    return (json.dumps(record) + "\n").encode("utf-8")


@dataclass
class FakeNode:
    id_: str
    metadata: dict[str, object]
    embedding: list[float]


class FakeEmbedModel:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        base = float((len(text) % 7) + 1)
        return [base + float(index) for index in range(self.dimension)]


class FakeRegistry:
    def __init__(self, _settings: AppSettings, *, dimension: int = 3) -> None:
        self.dimension = dimension

    def get_embed_model(self, _key: str | None = None) -> FakeEmbedModel:
        return FakeEmbedModel(self.dimension)


class FakeVectorStore:
    def __init__(self) -> None:
        self.initialized = False
        self.added_nodes: list[FakeNode] = []

    def _initialize(self) -> None:
        self.initialized = True

    def add(self, nodes: list[FakeNode]) -> None:
        self.added_nodes.extend(nodes)


class FakePipeline:
    def __init__(
        self,
        *,
        transformations: list[FakeEmbedModel],
        vector_store: FakeVectorStore | None = None,
        disable_cache: bool = False,
    ) -> None:
        del disable_cache
        self.embed_model = transformations[0]
        self.vector_store = vector_store

    def run(self, documents: list[Document]) -> list[FakeNode]:
        nodes = [
            FakeNode(
                id_=str(document.doc_id),
                metadata=dict(document.metadata),
                embedding=self.embed_model.embed(document.text),
            )
            for document in documents
        ]
        if self.vector_store is not None:
            self.vector_store.add(nodes)
        return nodes


def test_build_embedding_keys_use_embeddings_prefix() -> None:
    assert (
        build_embedding_object_prefix("csgo-matchmaking-damage", "20260308T181000Z")
        == "csgo-matchmaking-damage/20260308T181000Z/embeddings/"
    )
    assert (
        build_embedding_manifest_key("csgo-matchmaking-damage", "20260308T181000Z")
        == "csgo-matchmaking-damage/20260308T181000Z/embeddings/manifest.json"
    )
    assert (
        build_embedding_quality_report_key("csgo-matchmaking-damage", "20260308T181000Z")
        == "csgo-matchmaking-damage/20260308T181000Z/embeddings/quality_report.json"
    )
    assert build_pgvector_data_table_name("rag_embeddings") == "data_rag_embeddings"


def test_build_document_from_json_preserves_doc_id_and_augments_metadata() -> None:
    settings = build_settings()
    document = build_document_from_json(
        {
            "doc_id": "20260308T180500Z:1",
            "text": "Evento damage em map=de_mirage file=demo_1 round=1.",
            "metadata": {
                "doc_id": "20260308T180500Z:1",
                "event_type": "damage",
                "file": "demo_1",
                "round": 1,
                "map": "de_mirage",
                "source_file": "damage.csv",
            },
        },
        settings,
        embed_model_name="fake/embed",
        part_key=_document_part_key(1),
        line_number=1,
    )

    assert document.doc_id == "20260308T180500Z:1"
    assert document.metadata["embedding_run_id"] == "20260308T181000Z"
    assert document.metadata["document_source_run_id"] == "20260308T180500Z"
    assert document.metadata["dataset_prefix"] == "csgo-matchmaking-damage"
    assert document.metadata["embed_model"] == "fake/embed"


def test_delete_embeddings_for_run_executes_delete_on_single_pgvector_table() -> None:
    cursor = FakeCursor()
    cursor.rowcount = 7
    conn = FakeConnection(cursor=cursor)

    deleted = delete_embeddings_for_run(
        build_app_settings(),
        "20260308T181000Z",
        conn_factory=lambda _settings: conn,
    )

    assert deleted == 7
    query, params = cursor.queries[0]
    assert "DELETE FROM public.data_rag_embeddings" in query
    assert params == ("20260308T181000Z",)
    assert conn.committed
    assert conn.closed


def test_run_embedding_ingest_reads_manifest_indexes_documents_and_uploads_reports() -> None:
    manifest = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "artifact_prefix": "csgo-matchmaking-damage/20260308T180500Z/documents/",
        "parts": [
            {
                "object_key": _document_part_key(1),
                "rows": 2,
                "first_doc_id": "20260308T180500Z:1",
                "last_doc_id": "20260308T180500Z:2",
            },
            {
                "object_key": _document_part_key(2),
                "rows": 1,
                "first_doc_id": "20260308T180500Z:3",
                "last_doc_id": "20260308T180500Z:3",
            },
        ],
    }
    fake_minio = FakeMinio(
        "localhost:9000",
        "minioadmin",
        "minioadmin",
        False,
        initial_objects={
            "gold": {
                _document_manifest_key(): json.dumps(manifest).encode("utf-8"),
                _document_part_key(1): (
                    _document_payload("20260308T180500Z:1")
                    + _document_payload(
                        "20260308T180500Z:2",
                        event_type="kill",
                        file_name="demo_2",
                        round_value=2,
                        source_file="kills.csv",
                    )
                ),
                _document_part_key(2): _document_payload(
                    "20260308T180500Z:3",
                    event_type="round_meta",
                    file_name="demo_3",
                    round_value=3,
                    source_file="meta.csv",
                ),
            }
        },
        existing_buckets={"gold"},
    )
    vector_store = FakeVectorStore()
    delete_calls: list[str] = []
    ensured_indexes = [
        "data_rag_embeddings_meta_embedding_run_id_idx",
        "data_rag_embeddings_meta_event_type_idx",
        "data_rag_embeddings_meta_map_idx",
        "data_rag_embeddings_meta_file_idx",
        "data_rag_embeddings_meta_round_int_idx",
    ]

    result = run_embedding_ingest(
        build_settings(batch_size=2),
        minio_factory=lambda **kwargs: fake_minio,
        app_settings_factory=lambda: build_app_settings(embed_dim=3),
        registry_factory=lambda settings: FakeRegistry(settings, dimension=3),
        vector_store_factory=lambda _settings: vector_store,
        pipeline_factory=FakePipeline,
        delete_embeddings_for_run_fn=lambda _settings, run_id: delete_calls.append(run_id) or 4,
        ensure_pgvector_storage_contract_fn=lambda _settings: ensured_indexes,
    )

    assert vector_store.initialized is True
    assert delete_calls == ["20260308T181000Z"]
    assert result.files_processed == 2
    assert result.rows_read == 3
    assert result.rows_output == 3
    assert result.quality_summary["deleted_existing_rows"] == 4
    assert result.quality_summary["pg_table_name"] == "rag_embeddings"
    assert result.quality_summary["pg_data_table_name"] == "data_rag_embeddings"
    assert result.quality_summary["ensured_indexes"] == ensured_indexes

    added_doc_ids = [node.metadata["doc_id"] for node in vector_store.added_nodes]
    assert added_doc_ids == [
        "20260308T180500Z:1",
        "20260308T180500Z:2",
        "20260308T180500Z:3",
    ]
    assert all(
        node.metadata["embedding_run_id"] == "20260308T181000Z"
        for node in vector_store.added_nodes
    )

    gold_objects = fake_minio.objects["gold"]
    manifest_key = "csgo-matchmaking-damage/20260308T181000Z/embeddings/manifest.json"
    report_key = "csgo-matchmaking-damage/20260308T181000Z/embeddings/quality_report.json"
    assert manifest_key in gold_objects
    assert report_key in gold_objects

    embedding_manifest = json.loads(gold_objects[manifest_key].decode("utf-8"))
    assert embedding_manifest["total_documents_read"] == 3
    assert embedding_manifest["total_embeddings_indexed"] == 3
    assert embedding_manifest["total_parts"] == 2
    assert embedding_manifest["pg_data_table_name"] == "data_rag_embeddings"
    assert embedding_manifest["ensured_indexes"] == ensured_indexes

    embedding_report = json.loads(gold_objects[report_key].decode("utf-8"))
    assert embedding_report["ensured_indexes"] == ensured_indexes
    assert embedding_report["summary"]["rows_read"] == 3
    assert embedding_report["summary"]["rows_output"] == 3
    assert embedding_report["summary"]["documents_indexed"] == 3
    assert embedding_report["summary"]["ensured_indexes"] == ensured_indexes


def test_run_embedding_ingest_supports_smoke_limit() -> None:
    manifest = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "artifact_prefix": "csgo-matchmaking-damage/20260308T180500Z/documents/",
        "parts": [
            {
                "object_key": _document_part_key(1),
                "rows": 3,
                "first_doc_id": "20260308T180500Z:1",
                "last_doc_id": "20260308T180500Z:3",
            }
        ],
    }
    fake_minio = FakeMinio(
        "localhost:9000",
        "minioadmin",
        "minioadmin",
        False,
        initial_objects={
            "gold": {
                _document_manifest_key(): json.dumps(manifest).encode("utf-8"),
                _document_part_key(1): (
                    _document_payload("20260308T180500Z:1")
                    + _document_payload("20260308T180500Z:2", event_type="kill")
                    + _document_payload("20260308T180500Z:3", event_type="round_meta")
                ),
            }
        },
        existing_buckets={"gold"},
    )
    vector_store = FakeVectorStore()

    result = run_embedding_ingest(
        build_settings(batch_size=2, max_documents=2),
        minio_factory=lambda **kwargs: fake_minio,
        app_settings_factory=lambda: build_app_settings(embed_dim=3),
        registry_factory=lambda settings: FakeRegistry(settings, dimension=3),
        vector_store_factory=lambda _settings: vector_store,
        pipeline_factory=FakePipeline,
        delete_embeddings_for_run_fn=lambda _settings, _run_id: 0,
        ensure_pgvector_storage_contract_fn=lambda _settings: ["idx_a", "idx_b"],
    )

    manifest_key = "csgo-matchmaking-damage/20260308T181000Z/embeddings/manifest.json"
    report_key = "csgo-matchmaking-damage/20260308T181000Z/embeddings/quality_report.json"
    embedding_manifest = json.loads(fake_minio.objects["gold"][manifest_key].decode("utf-8"))
    embedding_report = json.loads(fake_minio.objects["gold"][report_key].decode("utf-8"))

    assert result.rows_read == 2
    assert result.rows_output == 2
    assert len(vector_store.added_nodes) == 2
    assert embedding_manifest["max_documents"] == 2
    assert embedding_manifest["total_embeddings_indexed"] == 2
    assert embedding_report["max_documents"] == 2
    assert embedding_report["summary"]["max_documents"] == 2
    assert embedding_report["summary"]["documents_indexed"] == 2


def test_run_embedding_ingest_fails_when_embedding_dimension_does_not_match() -> None:
    manifest = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "artifact_prefix": "csgo-matchmaking-damage/20260308T180500Z/documents/",
        "parts": [
            {
                "object_key": _document_part_key(1),
                "rows": 1,
                "first_doc_id": "20260308T180500Z:1",
                "last_doc_id": "20260308T180500Z:1",
            }
        ],
    }
    fake_minio = FakeMinio(
        "localhost:9000",
        "minioadmin",
        "minioadmin",
        False,
        initial_objects={
            "gold": {
                _document_manifest_key(): json.dumps(manifest).encode("utf-8"),
                _document_part_key(1): _document_payload("20260308T180500Z:1"),
            }
        },
        existing_buckets={"gold"},
    )

    with pytest.raises(ValueError, match="PG_EMBED_DIM mismatch"):
        run_embedding_ingest(
            build_settings(batch_size=1),
            minio_factory=lambda **kwargs: fake_minio,
            app_settings_factory=lambda: build_app_settings(embed_dim=3),
            registry_factory=lambda settings: FakeRegistry(settings, dimension=2),
            vector_store_factory=lambda _settings: FakeVectorStore(),
            pipeline_factory=FakePipeline,
            delete_embeddings_for_run_fn=lambda _settings, _run_id: 0,
            ensure_pgvector_storage_contract_fn=lambda _settings: [],
        )
