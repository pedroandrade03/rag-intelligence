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
    EmbeddingRunProgress,
    _download_source_object,
    build_document_from_json,
    build_embedding_manifest_key,
    build_embedding_object_prefix,
    build_embedding_quality_report_key,
    delete_embeddings_for_run,
    get_embedding_run_progress,
    run_embedding_ingest,
)
from rag_intelligence.embeddings_pipeline import _embed_document_batch
from rag_intelligence.settings import AppSettings


def build_settings(
    batch_size: int = 2,
    *,
    num_workers: int = 4,
    parallel_batches: int = 4,
    max_documents: int | None = None,
    resume: bool = False,
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
        embedding_num_workers=num_workers,
        embedding_parallel_batches=parallel_batches,
        embedding_max_documents=max_documents,
        embedding_resume=resume,
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
    document_tier: str | None = None,
    file_name: str = "demo_1",
    round_value: int = 1,
    source_file: str = "damage.csv",
) -> bytes:
    tier = document_tier or (
        "round_type_profile" if event_type == "round_meta" else "weapon_map_profile"
    )
    record = {
        "doc_id": doc_id,
        "text": f"Evento {event_type} em map=de_mirage file={file_name} round={round_value}.",
        "metadata": {
            "doc_id": doc_id,
            "source_run_id": "20260306T025119Z",
            "document_run_id": "20260308T180500Z",
            "dataset_prefix": "csgo-matchmaking-damage",
            "event_type": event_type,
            "document_tier": tier,
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


class FlakyDownloadMinio(FakeMinio):
    def __init__(self, *args, flaky_object: str, failures: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.flaky_object = flaky_object
        self.failures_remaining = failures

    def fget_object(self, bucket_name: str, object_name: str, file_path: str) -> None:
        if object_name == self.flaky_object and self.failures_remaining > 0:
            self.failures_remaining -= 1
            raise RuntimeError("simulated incomplete read")
        super().fget_object(bucket_name, object_name, file_path)


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

    def run(self, documents: list[Document], **_: object) -> list[FakeNode]:
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


class NaNFailingPipeline(FakePipeline):
    def run(self, documents: list[Document], **kwargs: object) -> list[FakeNode]:
        del kwargs
        if any("nan-trigger" in document.text for document in documents):
            raise RuntimeError("failed to encode response: json: unsupported value: NaN")
        return super().run(documents)


class GenericFailingPipeline(FakePipeline):
    def run(self, documents: list[Document], **kwargs: object) -> list[FakeNode]:
        del documents
        del kwargs
        raise RuntimeError("simulated transport failure")


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
                "document_tier": "weapon_map_profile",
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
    assert document.metadata["document_tier"] == "weapon_map_profile"
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


def test_get_embedding_run_progress_returns_contiguous_state() -> None:
    cursor = FakeCursor()
    cursor.set_result([(5376, 5376, 1, 5376)])
    conn = FakeConnection(cursor=cursor)

    progress = get_embedding_run_progress(
        build_app_settings(),
        "20260308T181000Z",
        "20260308T180500Z",
        conn_factory=lambda _settings: conn,
    )

    assert progress.existing_rows == 5376
    assert progress.distinct_rows == 5376
    assert progress.min_doc_line == 1
    assert progress.max_doc_line == 5376

    query, params = cursor.queries[0]
    assert "COUNT(DISTINCT node_id)" in query
    assert params == ("20260308T181000Z", "20260308T180500Z:%")
    assert conn.closed


def test_download_source_object_retries_transient_failures(tmp_path) -> None:
    object_key = _document_part_key(1)
    fake_minio = FlakyDownloadMinio(
        "localhost:9000",
        "minioadmin",
        "minioadmin",
        False,
        flaky_object=object_key,
        failures=2,
        initial_objects={"gold": {object_key: _document_payload("20260308T180500Z:1")}},
        existing_buckets={"gold"},
    )

    downloaded_path = _download_source_object(
        fake_minio,
        "gold",
        object_key,
        tmp_path / "part-00001.jsonl",
        label="Document part",
    )

    assert downloaded_path.read_text(encoding="utf-8").startswith('{"doc_id"')
    assert fake_minio.failures_remaining == 0


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
    assert result.quality_summary["num_workers"] == 4
    assert result.quality_summary["parallel_batches"] == 4

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
    assert embedding_manifest["num_workers"] == 4
    assert embedding_manifest["parallel_batches"] == 4

    embedding_report = json.loads(gold_objects[report_key].decode("utf-8"))
    assert embedding_report["ensured_indexes"] == ensured_indexes
    assert embedding_report["num_workers"] == 4
    assert embedding_report["parallel_batches"] == 4
    assert embedding_report["summary"]["rows_read"] == 3
    assert embedding_report["summary"]["rows_output"] == 3
    assert embedding_report["summary"]["documents_indexed"] == 3
    assert embedding_report["summary"]["ensured_indexes"] == ensured_indexes
    assert embedding_report["summary"]["num_workers"] == 4
    assert embedding_report["summary"]["parallel_batches"] == 4
    assert embedding_report["summary"]["resume_enabled"] is False


def test_embed_document_batch_handles_nan_error_by_splitting_and_skipping_bad_doc() -> None:
    documents = [
        Document(text="ok-one", doc_id="20260308T180500Z:1", metadata={"doc_id": "1"}),
        Document(
            text="nan-trigger",
            doc_id="20260308T180500Z:2",
            metadata={"doc_id": "2"},
        ),
        Document(text="ok-three", doc_id="20260308T180500Z:3", metadata={"doc_id": "3"}),
    ]

    batch = _embed_document_batch(
        documents,
        app_settings=build_app_settings(embed_dim=3),
        embed_model_name="fake/embed",
        registry_factory=lambda settings: FakeRegistry(settings, dimension=3),
        pipeline_factory=NaNFailingPipeline,
        num_workers=4,
    )

    assert batch.rows_indexed == 2
    assert [node.id_ for node in batch.nodes] == [
        "20260308T180500Z:1",
        "20260308T180500Z:3",
    ]


def test_embed_document_batch_raises_for_non_nan_pipeline_failures() -> None:
    documents = [
        Document(text="ok-one", doc_id="20260308T180500Z:1", metadata={"doc_id": "1"}),
    ]

    with pytest.raises(RuntimeError, match="simulated transport failure"):
        _embed_document_batch(
            documents,
            app_settings=build_app_settings(embed_dim=3),
            embed_model_name="fake/embed",
            registry_factory=lambda settings: FakeRegistry(settings, dimension=3),
            pipeline_factory=GenericFailingPipeline,
            num_workers=4,
        )


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
    assert embedding_manifest["parallel_batches"] == 4
    assert embedding_report["max_documents"] == 2
    assert embedding_report["parallel_batches"] == 4
    assert embedding_report["summary"]["max_documents"] == 2
    assert embedding_report["summary"]["documents_indexed"] == 2


def test_run_embedding_ingest_can_resume_without_deleting_existing_rows() -> None:
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
                "rows": 2,
                "first_doc_id": "20260308T180500Z:3",
                "last_doc_id": "20260308T180500Z:4",
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
                    + _document_payload("20260308T180500Z:2")
                ),
                _document_part_key(2): (
                    _document_payload("20260308T180500Z:3", file_name="demo_3")
                    + _document_payload("20260308T180500Z:4", file_name="demo_4")
                ),
            }
        },
        existing_buckets={"gold"},
    )
    vector_store = FakeVectorStore()
    delete_calls: list[str] = []

    result = run_embedding_ingest(
        build_settings(batch_size=2, resume=True),
        minio_factory=lambda **kwargs: fake_minio,
        app_settings_factory=lambda: build_app_settings(embed_dim=3),
        registry_factory=lambda settings: FakeRegistry(settings, dimension=3),
        vector_store_factory=lambda _settings: vector_store,
        pipeline_factory=FakePipeline,
        delete_embeddings_for_run_fn=lambda _settings, run_id: delete_calls.append(run_id) or 0,
        get_embedding_run_progress_fn=lambda *_args, **_kwargs: EmbeddingRunProgress(
            existing_rows=2,
            distinct_rows=2,
            min_doc_line=1,
            max_doc_line=2,
        ),
        ensure_pgvector_storage_contract_fn=lambda _settings: ["idx_a"],
    )

    assert delete_calls == []
    assert [node.id_ for node in vector_store.added_nodes] == [
        "20260308T180500Z:3",
        "20260308T180500Z:4",
    ]
    assert result.rows_read == 4
    assert result.rows_output == 4
    assert result.quality_summary["execution_rows_read"] == 2
    assert result.quality_summary["execution_rows_output"] == 2
    assert result.quality_summary["resume_enabled"] is True
    assert result.quality_summary["resume_existing_rows"] == 2
    assert result.quality_summary["resume_from_doc_line"] == 2
    assert result.quality_summary["resume_skipped_parts"] == 1
    assert result.quality_summary["resume_skipped_documents"] == 2

    report_key = "csgo-matchmaking-damage/20260308T181000Z/embeddings/quality_report.json"
    embedding_report = json.loads(fake_minio.objects["gold"][report_key].decode("utf-8"))
    assert embedding_report["resume_enabled"] is True
    assert embedding_report["summary"]["rows_output"] == 4
    assert embedding_report["summary"]["execution_rows_output"] == 2


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
