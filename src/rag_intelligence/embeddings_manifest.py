from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Any

from llama_index.core.schema import Document
from minio import Minio
from minio.error import S3Error

from rag_intelligence.config import EmbeddingSettings
from rag_intelligence.documents import build_document_manifest_key
from rag_intelligence.minio_utils import load_minio_object, stream_text_lines

LOGGER = logging.getLogger(__name__)

REQUIRED_DOCUMENT_FIELDS = {"doc_id", "text", "metadata"}
REQUIRED_DOCUMENT_METADATA_FIELDS = {"event_type", "file", "round", "map", "source_file"}
SCALAR_METADATA_TYPES = (str, int, float, bool, type(None))
PART_DOWNLOAD_MAX_ATTEMPTS = 3
PART_DOWNLOAD_RETRY_DELAY_SECONDS = 2.0


@dataclass(frozen=True)
class DocumentManifestPart:
    object_key: str
    rows: int | None
    first_doc_id: str | None
    last_doc_id: str | None
    first_doc_line: int | None
    last_doc_line: int | None


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
