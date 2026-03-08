from __future__ import annotations

import codecs
import json
import logging
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg2
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import Document
from minio import Minio
from minio.error import S3Error

from rag_intelligence.config import EmbeddingSettings
from rag_intelligence.db import (
    build_pgvector_data_table_name,
    create_vector_store,
    ensure_pgvector_storage_contract,
)
from rag_intelligence.documents import build_document_manifest_key
from rag_intelligence.minio_utils import ensure_bucket
from rag_intelligence.providers import ProviderRegistry
from rag_intelligence.settings import AppSettings

LOGGER = logging.getLogger(__name__)

REQUIRED_DOCUMENT_FIELDS = {"doc_id", "text", "metadata"}
REQUIRED_DOCUMENT_METADATA_FIELDS = {"event_type", "file", "round", "map", "source_file"}
SCALAR_METADATA_TYPES = (str, int, float, bool, type(None))


@dataclass(frozen=True)
class ProcessedEmbeddingPart:
    object_key: str
    rows_read: int
    rows_output: int
    first_doc_id: str
    last_doc_id: str


@dataclass(frozen=True)
class EmbeddingIngestResult:
    uploaded_objects: list[str]
    artifact_prefix: str
    manifest_key: str
    quality_report_key: str
    files_processed: int
    rows_read: int
    rows_output: int
    quality_summary: dict[str, Any]


def build_embedding_object_prefix(dataset_prefix: str, run_id: str) -> str:
    normalized_prefix = dataset_prefix.strip("/")
    normalized_run_id = run_id.strip("/")
    return f"{normalized_prefix}/{normalized_run_id}/embeddings/"


def build_embedding_manifest_key(dataset_prefix: str, run_id: str) -> str:
    return f"{build_embedding_object_prefix(dataset_prefix, run_id)}manifest.json"


def build_embedding_quality_report_key(dataset_prefix: str, run_id: str) -> str:
    return f"{build_embedding_object_prefix(dataset_prefix, run_id)}quality_report.json"


def _stream_text_lines(response: Any, *, chunk_size: int = 64 * 1024) -> Iterator[str]:
    decoder = codecs.getincrementaldecoder("utf-8-sig")(errors="replace")
    buffer = ""

    for chunk in response.stream(chunk_size):
        if not chunk:
            continue
        buffer += decoder.decode(chunk)
        while True:
            newline_index = buffer.find("\n")
            if newline_index == -1:
                break
            yield buffer[: newline_index + 1]
            buffer = buffer[newline_index + 1 :]

    buffer += decoder.decode(b"", final=True)
    if buffer:
        yield buffer


def _load_source_response(client: Minio, bucket: str, object_key: str, *, label: str) -> Any:
    try:
        return client.get_object(bucket, object_key)
    except KeyError as exc:
        raise FileNotFoundError(f"{label} not found: {bucket}/{object_key}") from exc
    except S3Error as exc:
        if exc.code in {"NoSuchKey", "NoSuchObject", "NoSuchBucket"}:
            raise FileNotFoundError(f"{label} not found: {bucket}/{object_key}") from exc
        raise


def load_document_manifest(client: Minio, settings: EmbeddingSettings) -> dict[str, Any]:
    manifest_key = build_document_manifest_key(
        settings.document_dataset_prefix,
        settings.document_source_run_id,
    )
    response = _load_source_response(
        client,
        settings.document_bucket,
        manifest_key,
        label="Document manifest",
    )
    try:
        try:
            manifest = json.loads(response.read().decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Document manifest is invalid JSON: "
                f"{settings.document_bucket}/{manifest_key}"
            ) from exc
    finally:
        response.close()
        response.release_conn()

    if not isinstance(manifest, dict):
        raise ValueError("Document manifest must be a JSON object")
    parts = manifest.get("parts")
    if not isinstance(parts, list):
        raise ValueError("Document manifest must contain a 'parts' array")
    return manifest


def _extract_part_keys(manifest: dict[str, Any]) -> list[str]:
    part_keys: list[str] = []
    for index, part in enumerate(manifest.get("parts", []), start=1):
        if not isinstance(part, dict):
            raise ValueError(f"Document manifest part #{index} must be an object")
        object_key = part.get("object_key")
        if not isinstance(object_key, str) or not object_key.strip():
            raise ValueError(f"Document manifest part #{index} is missing object_key")
        part_keys.append(object_key)
    if not part_keys:
        raise ValueError("Document manifest does not contain any part-*.jsonl objects")
    return part_keys


def _validate_flat_metadata(metadata: dict[str, Any], *, part_key: str, line_number: int) -> None:
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise ValueError(
                f"Invalid metadata key in {part_key}:{line_number}: expected string key"
            )
        if not isinstance(value, SCALAR_METADATA_TYPES):
            raise ValueError(
                f"Invalid metadata value for '{key}' in {part_key}:{line_number}: "
                "expected flat scalar metadata"
            )


def build_document_from_json(
    payload: dict[str, Any],
    settings: EmbeddingSettings,
    *,
    embed_model_name: str,
    part_key: str,
    line_number: int,
) -> Document:
    missing_fields = sorted(REQUIRED_DOCUMENT_FIELDS - set(payload))
    if missing_fields:
        joined_missing = ", ".join(missing_fields)
        raise ValueError(
            f"Malformed JSONL in {part_key}:{line_number}: missing fields {joined_missing}"
        )

    doc_id = payload["doc_id"]
    text = payload["text"]
    metadata = payload["metadata"]
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError(f"Malformed JSONL in {part_key}:{line_number}: invalid doc_id")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Malformed JSONL in {part_key}:{line_number}: invalid text")
    if not isinstance(metadata, dict):
        raise ValueError(f"Malformed JSONL in {part_key}:{line_number}: invalid metadata")

    typed_metadata = dict(metadata)
    _validate_flat_metadata(typed_metadata, part_key=part_key, line_number=line_number)

    missing_metadata = sorted(REQUIRED_DOCUMENT_METADATA_FIELDS - set(typed_metadata))
    if missing_metadata:
        joined_missing = ", ".join(missing_metadata)
        raise ValueError(
            f"Malformed JSONL in {part_key}:{line_number}: missing metadata {joined_missing}"
        )

    typed_metadata["doc_id"] = doc_id
    typed_metadata["embedding_run_id"] = settings.embedding_run_id
    typed_metadata["document_source_run_id"] = settings.document_source_run_id
    typed_metadata["dataset_prefix"] = settings.embedding_dataset_prefix
    typed_metadata["embed_model"] = embed_model_name

    return Document(text=text, metadata=typed_metadata, doc_id=doc_id)


def preview_document_batch(
    client: Minio,
    settings: EmbeddingSettings,
    part_keys: Sequence[str],
    *,
    embed_model_name: str,
    limit: int,
) -> list[Document]:
    preview: list[Document] = []
    for part_key in part_keys:
        response = _load_source_response(
            client,
            settings.document_bucket,
            part_key,
            label="Document part",
        )
        try:
            for line_number, raw_line in enumerate(_stream_text_lines(response), start=1):
                if not raw_line.strip():
                    continue
                try:
                    payload = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSONL in {part_key}:{line_number}") from exc
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"Malformed JSONL in {part_key}:{line_number}: expected object"
                    )
                preview.append(
                    build_document_from_json(
                        payload,
                        settings,
                        embed_model_name=embed_model_name,
                        part_key=part_key,
                        line_number=line_number,
                    )
                )
                if len(preview) >= limit:
                    return preview
        finally:
            response.close()
            response.release_conn()
    return preview


def _ensure_vector_store_initialized(vector_store: Any) -> None:
    initialize = getattr(vector_store, "_initialize", None)
    if callable(initialize):
        initialize()


def _default_delete_conn_factory(app_settings: AppSettings) -> Any:
    return psycopg2.connect(
        host=app_settings.pg_host,
        port=app_settings.pg_port,
        user=app_settings.pg_user,
        password=app_settings.pg_password,
        dbname=app_settings.pg_database,
    )


def delete_embeddings_for_run(
    app_settings: AppSettings,
    embedding_run_id: str,
    *,
    conn_factory: Any = None,
) -> int:
    factory = conn_factory or _default_delete_conn_factory
    conn = factory(app_settings)
    try:
        cursor = conn.cursor()
        table_name = build_pgvector_data_table_name(app_settings.pg_table_name)
        query = (
            f"DELETE FROM public.{table_name} "
            "WHERE metadata_->>'embedding_run_id' = %s;"
        )
        cursor.execute(query, (embedding_run_id,))
        deleted_rows = int(getattr(cursor, "rowcount", 0) or 0)
        conn.commit()
        cursor.close()
    finally:
        conn.close()
    return deleted_rows


def _validate_embedding_dimensions(
    documents: Sequence[Document],
    embed_model: Any,
    *,
    expected_dim: int,
    pipeline_factory: Any = IngestionPipeline,
) -> None:
    pipeline = pipeline_factory(transformations=[embed_model], disable_cache=True)
    nodes = pipeline.run(documents=list(documents))
    if not nodes:
        raise ValueError("Embedding preview batch did not produce any nodes")

    first_embedding = getattr(nodes[0], "embedding", None)
    if not isinstance(first_embedding, list) or not first_embedding:
        raise ValueError("Embedding preview batch did not produce a valid embedding")

    actual_dim = len(first_embedding)
    if actual_dim != expected_dim:
        raise ValueError(
            "PG_EMBED_DIM mismatch: "
            f"expected {expected_dim}, got {actual_dim}"
        )


def _index_document_batch(pipeline: Any, documents: Sequence[Document]) -> int:
    if not documents:
        return 0
    nodes = pipeline.run(documents=list(documents))
    return len(nodes)


def run_embedding_ingest(
    settings: EmbeddingSettings,
    *,
    minio_factory=Minio,
    app_settings_factory=AppSettings.from_env,
    registry_factory=ProviderRegistry,
    vector_store_factory=create_vector_store,
    pipeline_factory=IngestionPipeline,
    delete_embeddings_for_run_fn=delete_embeddings_for_run,
    ensure_pgvector_storage_contract_fn=ensure_pgvector_storage_contract,
) -> EmbeddingIngestResult:
    client = minio_factory(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )
    ensure_bucket(client, settings.embedding_report_bucket)

    document_manifest_key = build_document_manifest_key(
        settings.document_dataset_prefix,
        settings.document_source_run_id,
    )
    document_manifest = load_document_manifest(client, settings)
    part_keys = _extract_part_keys(document_manifest)

    app_settings = app_settings_factory()
    registry = registry_factory(app_settings)
    embed_model_name = app_settings.default_embed_model
    embed_model = registry.get_embed_model(embed_model_name)

    preview_batch = preview_document_batch(
        client,
        settings,
        part_keys,
        embed_model_name=embed_model_name,
        limit=(
            settings.embedding_batch_size
            if settings.embedding_max_documents is None
            else min(
                settings.embedding_batch_size,
                settings.embedding_max_documents,
            )
        ),
    )
    if not preview_batch:
        raise ValueError("Document manifest parts did not contain any documents")

    _validate_embedding_dimensions(
        preview_batch,
        embed_model,
        expected_dim=app_settings.pg_embed_dim,
        pipeline_factory=pipeline_factory,
    )

    vector_store = vector_store_factory(app_settings)
    _ensure_vector_store_initialized(vector_store)
    ensured_indexes = ensure_pgvector_storage_contract_fn(app_settings)
    deleted_existing_rows = delete_embeddings_for_run_fn(
        app_settings,
        settings.embedding_run_id,
    )
    pipeline = pipeline_factory(
        transformations=[embed_model],
        vector_store=vector_store,
        disable_cache=True,
    )

    uploaded_objects: list[str] = []
    processed_parts: list[ProcessedEmbeddingPart] = []
    rows_read = 0
    rows_output = 0
    limit_reached = False

    for part_key in part_keys:
        response = _load_source_response(
            client,
            settings.document_bucket,
            part_key,
            label="Document part",
        )
        batch: list[Document] = []
        part_rows_read = 0
        part_rows_output = 0
        first_doc_id = ""
        last_doc_id = ""

        try:
            for line_number, raw_line in enumerate(_stream_text_lines(response), start=1):
                if (
                    settings.embedding_max_documents is not None
                    and rows_read >= settings.embedding_max_documents
                ):
                    limit_reached = True
                    break
                if not raw_line.strip():
                    continue
                try:
                    payload = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSONL in {part_key}:{line_number}") from exc
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"Malformed JSONL in {part_key}:{line_number}: expected object"
                    )

                document = build_document_from_json(
                    payload,
                    settings,
                    embed_model_name=embed_model_name,
                    part_key=part_key,
                    line_number=line_number,
                )
                batch.append(document)
                part_rows_read += 1
                rows_read += 1

                current_doc_id = str(document.doc_id)
                if not first_doc_id:
                    first_doc_id = current_doc_id
                last_doc_id = current_doc_id

                if len(batch) >= settings.embedding_batch_size:
                    indexed_rows = _index_document_batch(pipeline, batch)
                    part_rows_output += indexed_rows
                    rows_output += indexed_rows
                    batch.clear()
        finally:
            response.close()
            response.release_conn()

        if batch:
            indexed_rows = _index_document_batch(pipeline, batch)
            part_rows_output += indexed_rows
            rows_output += indexed_rows

        processed_parts.append(
            ProcessedEmbeddingPart(
                object_key=part_key,
                rows_read=part_rows_read,
                rows_output=part_rows_output,
                first_doc_id=first_doc_id,
                last_doc_id=last_doc_id,
            )
        )
        if limit_reached:
            break

    if rows_output <= 0:
        raise ValueError("No embeddings were indexed from document parts")

    artifact_prefix = build_embedding_object_prefix(
        settings.embedding_dataset_prefix,
        settings.embedding_run_id,
    )

    with tempfile.TemporaryDirectory(prefix="embedding-ingest-") as temp_dir:
        temp_path = Path(temp_dir)
        manifest = {
            "generated_at": datetime.now(UTC).isoformat(),
            "source_bucket": settings.document_bucket,
            "source_manifest_key": document_manifest_key,
            "report_bucket": settings.embedding_report_bucket,
            "artifact_prefix": artifact_prefix,
            "document_source_run_id": settings.document_source_run_id,
            "embedding_run_id": settings.embedding_run_id,
            "dataset_prefix": settings.embedding_dataset_prefix,
            "embed_model": embed_model_name,
            "pg_table_name": app_settings.pg_table_name,
            "pg_data_table_name": build_pgvector_data_table_name(app_settings.pg_table_name),
            "batch_size": settings.embedding_batch_size,
            "max_documents": settings.embedding_max_documents,
            "ensured_indexes": ensured_indexes,
            "total_documents_read": rows_read,
            "total_embeddings_indexed": rows_output,
            "total_parts": len(processed_parts),
            "parts": [asdict(part) for part in processed_parts],
        }
        manifest_file = temp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest_key = build_embedding_manifest_key(
            settings.embedding_dataset_prefix,
            settings.embedding_run_id,
        )
        client.fput_object(
            bucket_name=settings.embedding_report_bucket,
            object_name=manifest_key,
            file_path=str(manifest_file),
            content_type="application/json",
        )
        uploaded_objects.append(manifest_key)

        quality_summary = {
            "files_processed": len(processed_parts),
            "rows_read": rows_read,
            "rows_output": rows_output,
            "documents_indexed": rows_output,
            "pg_table_name": app_settings.pg_table_name,
            "pg_data_table_name": build_pgvector_data_table_name(app_settings.pg_table_name),
            "embed_model": embed_model_name,
            "batch_size": settings.embedding_batch_size,
            "max_documents": settings.embedding_max_documents,
            "deleted_existing_rows": deleted_existing_rows,
            "ensured_indexes": ensured_indexes,
        }
        quality_report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "source_bucket": settings.document_bucket,
            "source_manifest_key": document_manifest_key,
            "report_bucket": settings.embedding_report_bucket,
            "document_source_run_id": settings.document_source_run_id,
            "embedding_run_id": settings.embedding_run_id,
            "document_dataset_prefix": settings.document_dataset_prefix,
            "embedding_dataset_prefix": settings.embedding_dataset_prefix,
            "artifact_prefix": artifact_prefix,
            "ensured_indexes": ensured_indexes,
            "max_documents": settings.embedding_max_documents,
            "summary": quality_summary,
            "parts": [asdict(part) for part in processed_parts],
        }
        quality_report_file = temp_path / "quality_report.json"
        quality_report_file.write_text(json.dumps(quality_report, indent=2), encoding="utf-8")
        quality_report_key = build_embedding_quality_report_key(
            settings.embedding_dataset_prefix,
            settings.embedding_run_id,
        )
        client.fput_object(
            bucket_name=settings.embedding_report_bucket,
            object_name=quality_report_key,
            file_path=str(quality_report_file),
            content_type="application/json",
        )
        uploaded_objects.append(quality_report_key)

    LOGGER.info(
        (
            "Embedding ingest completed: run_id=%s documents=%s indexed=%s "
            "parts=%s table=%s"
        ),
        settings.embedding_run_id,
        rows_read,
        rows_output,
        len(processed_parts),
        app_settings.pg_table_name,
    )

    return EmbeddingIngestResult(
        uploaded_objects=uploaded_objects,
        artifact_prefix=artifact_prefix,
        manifest_key=manifest_key,
        quality_report_key=quality_report_key,
        files_processed=len(processed_parts),
        rows_read=rows_read,
        rows_output=rows_output,
        quality_summary=quality_summary,
    )
