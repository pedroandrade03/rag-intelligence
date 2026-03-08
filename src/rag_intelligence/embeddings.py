from __future__ import annotations

import json
import logging
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import Document
from minio import Minio

from rag_intelligence.config import EmbeddingSettings
from rag_intelligence.db import (
    build_pgvector_data_table_name,
    create_vector_store,
    ensure_pgvector_storage_contract,
)
from rag_intelligence.documents import build_document_manifest_key
from rag_intelligence.embeddings_manifest import (
    _download_source_object,
    _extract_manifest_parts,
    _parse_doc_line_number,
    _stream_text_file_lines,
    build_document_from_json,
    load_document_manifest,
    preview_document_batch,
)
from rag_intelligence.embeddings_pipeline import (
    EmbeddedBatch,
    _flush_completed_embedding_batches,
    _submit_embedding_batch,
    _validate_embedding_dimensions,
)
from rag_intelligence.embeddings_storage import (
    EmbeddingRunProgress,
    _ensure_vector_store_initialized,
    _validate_resume_progress,
    delete_embeddings_for_run,
    get_embedding_run_progress,
)
from rag_intelligence.minio_utils import ensure_bucket
from rag_intelligence.providers import ProviderRegistry
from rag_intelligence.settings import AppSettings

LOGGER = logging.getLogger(__name__)


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
                    and manifest_part.first_doc_line
                    <= resume_from_doc_line
                    < manifest_part.last_doc_line
                ):
                    part_skip_remaining = resume_from_doc_line - manifest_part.first_doc_line + 1

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
                            pending_futures, indexed_rows = _flush_completed_embedding_batches(
                                pending_futures,
                                vector_store=vector_store,
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
        ("Embedding ingest completed: run_id=%s documents=%s indexed=%s parts=%s table=%s"),
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
