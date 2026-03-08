from __future__ import annotations

import json
import logging
import tempfile
import threading
from collections.abc import Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from typing import Any

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import Document
from minio import Minio
from minio.error import S3Error

from rag_intelligence.config import EmbeddingSettings
from rag_intelligence.db import (
    build_pgvector_data_table_name,
    create_vector_store,
    default_conn_factory,
    ensure_pgvector_storage_contract,
)
from rag_intelligence.documents import build_document_manifest_key
from rag_intelligence.minio_utils import ensure_bucket, load_minio_object, stream_text_lines
from rag_intelligence.providers import ProviderRegistry
from rag_intelligence.settings import AppSettings

LOGGER = logging.getLogger(__name__)
THREAD_LOCAL = threading.local()

REQUIRED_DOCUMENT_FIELDS = {"doc_id", "text", "metadata"}
REQUIRED_DOCUMENT_METADATA_FIELDS = {"event_type", "file", "round", "map", "source_file"}
SCALAR_METADATA_TYPES = (str, int, float, bool, type(None))
PART_DOWNLOAD_MAX_ATTEMPTS = 3
PART_DOWNLOAD_RETRY_DELAY_SECONDS = 2.0


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


@dataclass(frozen=True)
class EmbeddedBatch:
    nodes: list[Any]
    rows_indexed: int


@dataclass(frozen=True)
class DocumentManifestPart:
    object_key: str
    rows: int | None
    first_doc_id: str | None
    last_doc_id: str | None
    first_doc_line: int | None
    last_doc_line: int | None


@dataclass(frozen=True)
class EmbeddingRunProgress:
    existing_rows: int
    distinct_rows: int
    min_doc_line: int | None
    max_doc_line: int | None


def build_embedding_object_prefix(dataset_prefix: str, run_id: str) -> str:
    normalized_prefix = dataset_prefix.strip("/")
    normalized_run_id = run_id.strip("/")
    return f"{normalized_prefix}/{normalized_run_id}/embeddings/"


def build_embedding_manifest_key(dataset_prefix: str, run_id: str) -> str:
    return f"{build_embedding_object_prefix(dataset_prefix, run_id)}manifest.json"


def build_embedding_quality_report_key(dataset_prefix: str, run_id: str) -> str:
    return f"{build_embedding_object_prefix(dataset_prefix, run_id)}quality_report.json"




def _download_source_object(
    client: Minio,
    bucket: str,
    object_key: str,
    destination_path: Path,
    *,
    label: str,
    max_attempts: int = PART_DOWNLOAD_MAX_ATTEMPTS,
) -> Path:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            if destination_path.exists():
                destination_path.unlink()
            client.fget_object(bucket, object_key, str(destination_path))
            return destination_path
        except KeyError as exc:
            raise FileNotFoundError(f"{label} not found: {bucket}/{object_key}") from exc
        except S3Error as exc:
            if exc.code in {"NoSuchKey", "NoSuchObject", "NoSuchBucket"}:
                raise FileNotFoundError(f"{label} not found: {bucket}/{object_key}") from exc
            last_error = exc
        except Exception as exc:
            last_error = exc

        if attempt < max_attempts:
            LOGGER.warning(
                "Retrying %s download after attempt %s/%s failed: %s",
                label,
                attempt,
                max_attempts,
                last_error,
            )
            sleep(PART_DOWNLOAD_RETRY_DELAY_SECONDS)

    raise RuntimeError(
        f"Failed to download {label.lower()} after {max_attempts} attempts: "
        f"{bucket}/{object_key}"
    ) from last_error


def _stream_text_file_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        yield from handle


def load_document_manifest(client: Minio, settings: EmbeddingSettings) -> dict[str, Any]:
    manifest_key = build_document_manifest_key(
        settings.document_dataset_prefix,
        settings.document_source_run_id,
    )
    response = load_minio_object(
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


def _parse_doc_line_number(doc_id: str, *, label: str) -> int:
    normalized = doc_id.strip()
    if ":" not in normalized:
        raise ValueError(f"Invalid {label}: expected '<run_id>:<line_number>'")
    line_fragment = normalized.rsplit(":", maxsplit=1)[1].strip()
    try:
        line_number = int(line_fragment)
    except ValueError as exc:
        raise ValueError(f"Invalid {label}: expected numeric line number") from exc
    if line_number <= 0:
        raise ValueError(f"Invalid {label}: line number must be greater than zero")
    return line_number


def _extract_manifest_parts(manifest: dict[str, Any]) -> list[DocumentManifestPart]:
    manifest_parts: list[DocumentManifestPart] = []
    for index, part in enumerate(manifest.get("parts", []), start=1):
        if not isinstance(part, dict):
            raise ValueError(f"Document manifest part #{index} must be an object")
        object_key = part.get("object_key")
        if not isinstance(object_key, str) or not object_key.strip():
            raise ValueError(f"Document manifest part #{index} is missing object_key")

        rows_raw = part.get("rows")
        rows: int | None
        if rows_raw is None:
            rows = None
        elif isinstance(rows_raw, int) and rows_raw >= 0:
            rows = rows_raw
        else:
            raise ValueError(f"Document manifest part #{index} has invalid rows")

        first_doc_id_raw = part.get("first_doc_id")
        last_doc_id_raw = part.get("last_doc_id")
        if first_doc_id_raw is None and last_doc_id_raw is None:
            first_doc_id = None
            last_doc_id = None
            first_doc_line = None
            last_doc_line = None
        else:
            if (
                not isinstance(first_doc_id_raw, str)
                or not first_doc_id_raw.strip()
                or not isinstance(last_doc_id_raw, str)
                or not last_doc_id_raw.strip()
            ):
                raise ValueError(
                    f"Document manifest part #{index} must contain both "
                    "first_doc_id and last_doc_id"
                )
            first_doc_id = first_doc_id_raw.strip()
            last_doc_id = last_doc_id_raw.strip()
            first_doc_line = _parse_doc_line_number(
                first_doc_id,
                label=f"document manifest part #{index} first_doc_id",
            )
            last_doc_line = _parse_doc_line_number(
                last_doc_id,
                label=f"document manifest part #{index} last_doc_id",
            )
            if last_doc_line < first_doc_line:
                raise ValueError(
                    f"Document manifest part #{index} has last_doc_id before first_doc_id"
                )

        manifest_parts.append(
            DocumentManifestPart(
                object_key=object_key.strip(),
                rows=rows,
                first_doc_id=first_doc_id,
                last_doc_id=last_doc_id,
                first_doc_line=first_doc_line,
                last_doc_line=last_doc_line,
            )
        )

    if not manifest_parts:
        raise ValueError("Document manifest does not contain any part-*.jsonl objects")
    return manifest_parts


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
        raise ValueError(
            "Cannot resume embedding run because existing rows contain duplicates"
        )
    if progress.max_doc_line != progress.existing_rows:
        raise ValueError(
            "Cannot resume embedding run because existing rows are not contiguous"
        )
    if progress.max_doc_line is None:
        raise ValueError(
            "Cannot resume embedding run because the existing progress is incomplete"
        )
    return progress.max_doc_line


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
        response = load_minio_object(
            client,
            settings.document_bucket,
            part_key,
            label="Document part",
        )
        try:
            for line_number, raw_line in enumerate(stream_text_lines(response), start=1):
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
    num_workers: int,
    pipeline_factory: Any = IngestionPipeline,
) -> None:
    pipeline = pipeline_factory(transformations=[embed_model], disable_cache=True)
    nodes = pipeline.run(documents=list(documents), num_workers=num_workers)
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


def _index_document_batch(
    pipeline: Any,
    documents: Sequence[Document],
    *,
    num_workers: int,
) -> int:
    if not documents:
        return 0
    nodes = pipeline.run(documents=list(documents), num_workers=num_workers)
    return len(nodes)


def _get_thread_embedding_pipeline(
    *,
    app_settings: AppSettings,
    embed_model_name: str,
    registry_factory: Any,
    pipeline_factory: Any,
) -> Any:
    cache_key = (
        app_settings.ollama_base_url,
        app_settings.ollama_embed_batch_size,
        embed_model_name,
        pipeline_factory,
    )
    cached_key = getattr(THREAD_LOCAL, "embedding_pipeline_key", None)
    cached_pipeline = getattr(THREAD_LOCAL, "embedding_pipeline", None)
    if cached_key == cache_key and cached_pipeline is not None:
        return cached_pipeline

    registry = registry_factory(app_settings)
    embed_model = registry.get_embed_model(embed_model_name)
    pipeline = pipeline_factory(transformations=[embed_model], disable_cache=True)
    THREAD_LOCAL.embedding_pipeline_key = cache_key
    THREAD_LOCAL.embedding_pipeline = pipeline
    return pipeline


def _embed_document_batch(
    documents: Sequence[Document],
    *,
    app_settings: AppSettings,
    embed_model_name: str,
    registry_factory: Any,
    pipeline_factory: Any,
    num_workers: int,
) -> EmbeddedBatch:
    pipeline = _get_thread_embedding_pipeline(
        app_settings=app_settings,
        embed_model_name=embed_model_name,
        registry_factory=registry_factory,
        pipeline_factory=pipeline_factory,
    )
    nodes = pipeline.run(documents=list(documents), num_workers=num_workers)
    return EmbeddedBatch(nodes=list(nodes), rows_indexed=len(nodes))


def _submit_embedding_batch(
    executor: ThreadPoolExecutor,
    documents: Sequence[Document],
    *,
    app_settings: AppSettings,
    embed_model_name: str,
    registry_factory: Any,
    pipeline_factory: Any,
    num_workers: int,
) -> Future[EmbeddedBatch]:
    submitted_documents = list(documents)
    return executor.submit(
        _embed_document_batch,
        submitted_documents,
        app_settings=app_settings,
        embed_model_name=embed_model_name,
        registry_factory=registry_factory,
        pipeline_factory=pipeline_factory,
        num_workers=num_workers,
    )


def _flush_completed_embedding_batches(
    pending_futures: list[Future[EmbeddedBatch]],
    *,
    vector_store: Any,
) -> tuple[list[Future[EmbeddedBatch]], int]:
    if not pending_futures:
        return [], 0

    done, not_done = wait(pending_futures, return_when=FIRST_COMPLETED)
    rows_indexed = 0
    for future in done:
        embedded_batch = future.result()
        if embedded_batch.nodes:
            vector_store.add(embedded_batch.nodes)
        rows_indexed += embedded_batch.rows_indexed
    return list(not_done), rows_indexed


def run_embedding_ingest(
    settings: EmbeddingSettings,
    *,
    minio_factory=Minio,
    app_settings_factory=AppSettings.from_env,
    registry_factory=ProviderRegistry,
    vector_store_factory=create_vector_store,
    pipeline_factory=IngestionPipeline,
    delete_embeddings_for_run_fn=delete_embeddings_for_run,
    get_embedding_run_progress_fn=get_embedding_run_progress,
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
    manifest_parts = _extract_manifest_parts(document_manifest)
    part_keys = [part.object_key for part in manifest_parts]

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
        num_workers=settings.embedding_num_workers,
        pipeline_factory=pipeline_factory,
    )

    vector_store = vector_store_factory(app_settings)
    _ensure_vector_store_initialized(vector_store)
    ensured_indexes = ensure_pgvector_storage_contract_fn(app_settings)
    resume_progress = EmbeddingRunProgress(
        existing_rows=0,
        distinct_rows=0,
        min_doc_line=None,
        max_doc_line=None,
    )
    resume_from_doc_line = 0
    deleted_existing_rows = 0
    if settings.embedding_resume:
        resume_progress = get_embedding_run_progress_fn(
            app_settings,
            settings.embedding_run_id,
            settings.document_source_run_id,
        )
        resume_from_doc_line = _validate_resume_progress(resume_progress)
    else:
        deleted_existing_rows = delete_embeddings_for_run_fn(
            app_settings,
            settings.embedding_run_id,
        )

    uploaded_objects: list[str] = []
    processed_parts: list[ProcessedEmbeddingPart] = []
    rows_read = 0
    rows_output = 0
    resume_skipped_parts = 0
    resume_skipped_documents = 0
    limit_reached = False

    with tempfile.TemporaryDirectory(prefix="embedding-parts-") as part_temp_dir:
        part_temp_path = Path(part_temp_dir)
        with ThreadPoolExecutor(max_workers=settings.embedding_parallel_batches) as executor:
            for part_index, manifest_part in enumerate(manifest_parts, start=1):
                part_key = manifest_part.object_key
                if (
                    resume_from_doc_line > 0
                    and manifest_part.last_doc_line is not None
                    and manifest_part.last_doc_line <= resume_from_doc_line
                ):
                    resume_skipped_parts += 1
                    if manifest_part.rows is not None:
                        resume_skipped_documents += manifest_part.rows
                    elif manifest_part.first_doc_line is not None:
                        resume_skipped_documents += (
                            manifest_part.last_doc_line - manifest_part.first_doc_line + 1
                        )
                    continue

                local_part_path = _download_source_object(
                    client,
                    settings.document_bucket,
                    part_key,
                    part_temp_path / f"part-{part_index:05d}.jsonl",
                    label="Document part",
                )
                batch: list[Document] = []
                pending_futures: list[Future[EmbeddedBatch]] = []
                part_rows_read = 0
                part_rows_output = 0
                first_doc_id = ""
                last_doc_id = ""
                part_skip_remaining = 0
                if (
                    resume_from_doc_line > 0
                    and manifest_part.first_doc_line is not None
                    and manifest_part.last_doc_line is not None
                    and manifest_part.first_doc_line <= resume_from_doc_line
                    < manifest_part.last_doc_line
                ):
                    part_skip_remaining = (
                        resume_from_doc_line - manifest_part.first_doc_line + 1
                    )

                try:
                    for line_number, raw_line in enumerate(
                        _stream_text_file_lines(local_part_path),
                        start=1,
                    ):
                        if (
                            settings.embedding_max_documents is not None
                            and rows_read >= settings.embedding_max_documents
                        ):
                            limit_reached = True
                            break
                        if not raw_line.strip():
                            continue
                        if part_skip_remaining > 0:
                            part_skip_remaining -= 1
                            resume_skipped_documents += 1
                            continue
                        try:
                            payload = json.loads(raw_line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"Malformed JSONL in {part_key}:{line_number}"
                            ) from exc
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
                        document_line = _parse_doc_line_number(
                            str(document.doc_id),
                            label=f"document doc_id in {part_key}:{line_number}",
                        )
                        if document_line <= resume_from_doc_line:
                            resume_skipped_documents += 1
                            continue
                        batch.append(document)
                        part_rows_read += 1
                        rows_read += 1

                        current_doc_id = str(document.doc_id)
                        if not first_doc_id:
                            first_doc_id = current_doc_id
                        last_doc_id = current_doc_id

                        if len(batch) >= settings.embedding_batch_size:
                            pending_futures.append(
                                _submit_embedding_batch(
                                    executor,
                                    batch,
                                    app_settings=app_settings,
                                    embed_model_name=embed_model_name,
                                    registry_factory=registry_factory,
                                    pipeline_factory=pipeline_factory,
                                    num_workers=settings.embedding_num_workers,
                                )
                            )
                            batch = []
                        if len(pending_futures) >= settings.embedding_parallel_batches:
                            pending_futures, indexed_rows = (
                                _flush_completed_embedding_batches(
                                    pending_futures,
                                    vector_store=vector_store,
                                )
                            )
                            part_rows_output += indexed_rows
                            rows_output += indexed_rows
                finally:
                    local_part_path.unlink(missing_ok=True)

                if batch:
                    pending_futures.append(
                        _submit_embedding_batch(
                            executor,
                            batch,
                            app_settings=app_settings,
                            embed_model_name=embed_model_name,
                            registry_factory=registry_factory,
                            pipeline_factory=pipeline_factory,
                            num_workers=settings.embedding_num_workers,
                        )
                    )

                while pending_futures:
                    pending_futures, indexed_rows = _flush_completed_embedding_batches(
                        pending_futures,
                        vector_store=vector_store,
                    )
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

    execution_rows_read = rows_read
    execution_rows_output = rows_output
    total_rows_read = resume_progress.existing_rows + execution_rows_read
    total_rows_output = resume_progress.existing_rows + execution_rows_output

    if total_rows_output <= 0:
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
            "num_workers": settings.embedding_num_workers,
            "parallel_batches": settings.embedding_parallel_batches,
            "resume_enabled": settings.embedding_resume,
            "resume_existing_rows": resume_progress.existing_rows,
            "resume_from_doc_line": resume_from_doc_line or None,
            "resume_skipped_parts": resume_skipped_parts,
            "resume_skipped_documents": resume_skipped_documents,
            "max_documents": settings.embedding_max_documents,
            "ensured_indexes": ensured_indexes,
            "execution_documents_read": execution_rows_read,
            "execution_embeddings_indexed": execution_rows_output,
            "total_documents_read": total_rows_read,
            "total_embeddings_indexed": total_rows_output,
            "execution_parts_processed": len(processed_parts),
            "total_parts": len(manifest_parts),
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
            "rows_read": total_rows_read,
            "rows_output": total_rows_output,
            "documents_indexed": total_rows_output,
            "execution_rows_read": execution_rows_read,
            "execution_rows_output": execution_rows_output,
            "pg_table_name": app_settings.pg_table_name,
            "pg_data_table_name": build_pgvector_data_table_name(app_settings.pg_table_name),
            "embed_model": embed_model_name,
            "batch_size": settings.embedding_batch_size,
            "num_workers": settings.embedding_num_workers,
            "parallel_batches": settings.embedding_parallel_batches,
            "resume_enabled": settings.embedding_resume,
            "resume_existing_rows": resume_progress.existing_rows,
            "resume_from_doc_line": resume_from_doc_line or None,
            "resume_skipped_parts": resume_skipped_parts,
            "resume_skipped_documents": resume_skipped_documents,
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
            "num_workers": settings.embedding_num_workers,
            "parallel_batches": settings.embedding_parallel_batches,
            "resume_enabled": settings.embedding_resume,
            "resume_existing_rows": resume_progress.existing_rows,
            "resume_from_doc_line": resume_from_doc_line or None,
            "resume_skipped_parts": resume_skipped_parts,
            "resume_skipped_documents": resume_skipped_documents,
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
        total_rows_read,
        total_rows_output,
        len(processed_parts),
        app_settings.pg_table_name,
    )

    return EmbeddingIngestResult(
        uploaded_objects=uploaded_objects,
        artifact_prefix=artifact_prefix,
        manifest_key=manifest_key,
        quality_report_key=quality_report_key,
        files_processed=len(processed_parts),
        rows_read=total_rows_read,
        rows_output=total_rows_output,
        quality_summary=quality_summary,
    )
