from __future__ import annotations

import codecs
import csv
import json
import logging
import tempfile
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from minio import Minio
from minio.error import S3Error

from rag_intelligence.config import DocumentSettings
from rag_intelligence.gold import build_gold_events_key
from rag_intelligence.minio_utils import clean_cell, ensure_bucket

LOGGER = logging.getLogger(__name__)

REQUIRED_GOLD_COLUMNS = {"event_type", "file", "round", "map", "source_file"}
NUMERIC_METADATA_FIELDS = {
    "round",
    "hp_dmg",
    "arm_dmg",
    "att_pos_x",
    "att_pos_y",
    "vic_pos_x",
    "vic_pos_y",
    "tick",
    "seconds",
    "start_seconds",
    "end_seconds",
    "att_rank",
    "vic_rank",
    "ct_eq_val",
    "t_eq_val",
    "ct_alive",
    "t_alive",
    "nade_land_x",
    "nade_land_y",
    "avg_match_rank",
    "source_line_number",
}
BOOLEAN_METADATA_FIELDS = {"is_bomb_planted"}
EVENT_TYPE_DAMAGE = "damage"
EVENT_TYPE_GRENADE = "grenade"
EVENT_TYPE_KILL = "kill"
EVENT_TYPE_ROUND_META = "round_meta"


@dataclass(frozen=True)
class DocumentPart:
    object_key: str
    rows: int
    first_doc_id: str
    last_doc_id: str


@dataclass(frozen=True)
class DocumentBuildResult:
    uploaded_objects: list[str]
    artifact_prefix: str
    manifest_key: str
    quality_report_key: str
    files_processed: int
    rows_read: int
    rows_output: int
    quality_summary: dict[str, Any]


def build_document_object_prefix(dataset_prefix: str, run_id: str) -> str:
    normalized_prefix = dataset_prefix.strip("/")
    normalized_run_id = run_id.strip("/")
    return f"{normalized_prefix}/{normalized_run_id}/documents/"


def build_document_part_key(dataset_prefix: str, run_id: str, part_number: int) -> str:
    return (
        f"{build_document_object_prefix(dataset_prefix, run_id)}"
        f"part-{part_number:05d}.jsonl"
    )


def build_document_manifest_key(dataset_prefix: str, run_id: str) -> str:
    return f"{build_document_object_prefix(dataset_prefix, run_id)}manifest.json"


def build_document_quality_report_key(dataset_prefix: str, run_id: str) -> str:
    return f"{build_document_object_prefix(dataset_prefix, run_id)}quality_report.json"


def build_doc_id(document_run_id: str, line_number: int) -> str:
    return f"{document_run_id}:{line_number}"


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


def _parse_number(value: str) -> int | float | str:
    try:
        number = Decimal(value)
    except InvalidOperation:
        return value

    if not number.is_finite():
        return value
    if number == number.to_integral_value():
        return int(number)
    return float(number)


def _parse_bool(value: str) -> bool | str:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return value


def _coerce_metadata_value(field: str, value: str) -> Any:
    if field in BOOLEAN_METADATA_FIELDS:
        return _parse_bool(value)
    if field in NUMERIC_METADATA_FIELDS:
        return _parse_number(value)
    return value


def _append_token(tokens: list[str], key: str, value: str | None) -> None:
    cleaned = clean_cell(value)
    if cleaned is None:
        return
    tokens.append(f"{key}={cleaned}")


def _append_position_tokens(tokens: list[str], row: dict[str, str | None]) -> None:
    fields = (
        "att_pos_x",
        "att_pos_y",
        "vic_pos_x",
        "vic_pos_y",
        "nade_land_x",
        "nade_land_y",
    )
    for field in fields:
        _append_token(tokens, field, row.get(field))


def build_document_text(row: dict[str, str | None]) -> str:
    event_type = clean_cell(row.get("event_type")) or "event"
    map_value = clean_cell(row.get("map")) or "unknown_map"
    file_value = clean_cell(row.get("file")) or "unknown_file"
    round_value = clean_cell(row.get("round")) or "unknown_round"

    tokens: list[str] = []
    if event_type == EVENT_TYPE_DAMAGE:
        lead = f"Evento damage em map={map_value} file={file_value} round={round_value}."
        for field in (
            "weapon",
            "hp_dmg",
            "arm_dmg",
            "hitbox",
            "att_team",
            "vic_team",
            "att_side",
            "vic_side",
            "tick",
            "seconds",
            "source_file",
        ):
            _append_token(tokens, field, row.get(field))
        _append_position_tokens(tokens, row)
        return " ".join([lead, *tokens]).strip()

    if event_type == EVENT_TYPE_GRENADE:
        lead = f"Evento grenade em map={map_value} file={file_value} round={round_value}."
        for field in (
            "weapon",
            "nade",
            "hp_dmg",
            "arm_dmg",
            "bomb_site",
            "att_team",
            "vic_team",
            "att_side",
            "vic_side",
            "tick",
            "seconds",
            "source_file",
        ):
            _append_token(tokens, field, row.get(field))
        _append_position_tokens(tokens, row)
        return " ".join([lead, *tokens]).strip()

    if event_type == EVENT_TYPE_KILL:
        lead = f"Evento kill em map={map_value} file={file_value} round={round_value}."
        for field in (
            "weapon",
            "wp_type",
            "att_team",
            "vic_team",
            "att_side",
            "vic_side",
            "ct_alive",
            "t_alive",
            "is_bomb_planted",
            "tick",
            "seconds",
            "source_file",
        ):
            _append_token(tokens, field, row.get(field))
        _append_position_tokens(tokens, row)
        return " ".join([lead, *tokens]).strip()

    if event_type == EVENT_TYPE_ROUND_META:
        lead = f"Resumo round_meta em map={map_value} file={file_value} round={round_value}."
        for field in (
            "winner_team",
            "winner_side",
            "round_type",
            "ct_eq_val",
            "t_eq_val",
            "start_seconds",
            "end_seconds",
            "avg_match_rank",
            "source_file",
        ):
            _append_token(tokens, field, row.get(field))
        return " ".join([lead, *tokens]).strip()

    lead = f"Evento {event_type} em map={map_value} file={file_value} round={round_value}."
    for field, value in row.items():
        if field in {"event_type", "map", "file", "round"}:
            continue
        _append_token(tokens, field, value)
    return " ".join([lead, *tokens]).strip()


def build_document_metadata(
    row: dict[str, str | None],
    settings: DocumentSettings,
    *,
    doc_id: str,
    source_line_number: int,
) -> dict[str, Any]:
    required_metadata = {
        "doc_id": doc_id,
        "source_run_id": settings.gold_source_run_id,
        "document_run_id": settings.document_run_id,
        "dataset_prefix": settings.document_dataset_prefix,
        "event_type": clean_cell(row.get("event_type")) or "",
        "file": clean_cell(row.get("file")) or "",
        "round": _coerce_metadata_value("round", clean_cell(row.get("round")) or "0"),
        "map": clean_cell(row.get("map")) or "",
        "source_file": clean_cell(row.get("source_file")) or "",
        "source_line_number": source_line_number,
    }

    metadata = dict(required_metadata)
    for field, raw_value in row.items():
        cleaned = clean_cell(raw_value)
        if cleaned is None:
            continue
        metadata[field] = _coerce_metadata_value(field, cleaned)

    return metadata


def _load_source_response(client: Minio, settings: DocumentSettings, source_object: str) -> Any:
    try:
        return client.get_object(settings.gold_bucket, source_object)
    except KeyError as exc:
        raise FileNotFoundError(
            f"Gold object not found for run_id={settings.gold_source_run_id}: {source_object}"
        ) from exc
    except S3Error as exc:
        if exc.code in {"NoSuchKey", "NoSuchObject", "NoSuchBucket"}:
            raise FileNotFoundError(
                f"Gold object not found for run_id={settings.gold_source_run_id}: {source_object}"
            ) from exc
        raise


def run_document_build(
    settings: DocumentSettings,
    *,
    minio_factory=Minio,
) -> DocumentBuildResult:
    client = minio_factory(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )

    ensure_bucket(client, settings.document_bucket)

    source_object = build_gold_events_key(
        settings.gold_dataset_prefix,
        settings.gold_source_run_id,
    )
    artifact_prefix = build_document_object_prefix(
        settings.document_dataset_prefix,
        settings.document_run_id,
    )
    response = _load_source_response(client, settings, source_object)

    uploaded_objects: list[str] = []
    event_type_counts: Counter[str] = Counter()
    part_records: list[DocumentPart] = []
    rows_read = 0
    rows_output = 0

    with tempfile.TemporaryDirectory(prefix="document-build-") as temp_dir:
        temp_path = Path(temp_dir)
        current_part_number = 0
        current_part_rows = 0
        current_part_first_doc_id = ""
        current_part_last_doc_id = ""
        current_part_file = None
        current_part_path: Path | None = None

        def flush_part() -> None:
            nonlocal current_part_rows
            nonlocal current_part_file
            nonlocal current_part_path
            nonlocal current_part_first_doc_id
            nonlocal current_part_last_doc_id
            if current_part_file is None or current_part_path is None or current_part_rows <= 0:
                return

            current_part_file.close()
            part_key = build_document_part_key(
                settings.document_dataset_prefix,
                settings.document_run_id,
                current_part_number,
            )
            client.fput_object(
                bucket_name=settings.document_bucket,
                object_name=part_key,
                file_path=str(current_part_path),
                content_type="application/x-ndjson",
            )
            uploaded_objects.append(part_key)
            part_records.append(
                DocumentPart(
                    object_key=part_key,
                    rows=current_part_rows,
                    first_doc_id=current_part_first_doc_id,
                    last_doc_id=current_part_last_doc_id,
                )
            )
            current_part_file = None
            current_part_path = None
            current_part_rows = 0
            current_part_first_doc_id = ""
            current_part_last_doc_id = ""

        try:
            reader = csv.DictReader(_stream_text_lines(response))
            fieldnames = set(reader.fieldnames or [])
            missing_columns = sorted(REQUIRED_GOLD_COLUMNS - fieldnames)
            if missing_columns:
                joined_missing = ", ".join(missing_columns)
                raise ValueError(
                    "Gold events.csv is missing required columns: "
                    f"{joined_missing}"
                )

            for source_line_number, row in enumerate(reader, start=2):
                if (
                    settings.document_max_rows is not None
                    and rows_output >= settings.document_max_rows
                ):
                    break
                rows_read += 1
                if current_part_file is None:
                    current_part_number += 1
                    current_part_path = temp_path / f"part-{current_part_number:05d}.jsonl"
                    current_part_file = current_part_path.open("w", encoding="utf-8", newline="")

                doc_id = build_doc_id(settings.document_run_id, rows_read)
                document = {
                    "doc_id": doc_id,
                    "text": build_document_text(row),
                    "metadata": build_document_metadata(
                        row,
                        settings,
                        doc_id=doc_id,
                        source_line_number=source_line_number,
                    ),
                }
                current_part_file.write(json.dumps(document, ensure_ascii=True) + "\n")

                current_part_rows += 1
                rows_output += 1
                if not current_part_first_doc_id:
                    current_part_first_doc_id = doc_id
                current_part_last_doc_id = doc_id

                event_type = str(document["metadata"]["event_type"] or "event")
                event_type_counts[event_type] += 1

                if current_part_rows >= settings.document_part_size_rows:
                    flush_part()
        finally:
            response.close()
            response.release_conn()

        flush_part()

        if rows_output <= 0:
            raise ValueError("No documents were generated from Gold events.csv.")

        manifest = {
            "generated_at": datetime.now(UTC).isoformat(),
            "source_bucket": settings.gold_bucket,
            "source_object_key": source_object,
            "document_bucket": settings.document_bucket,
            "artifact_prefix": artifact_prefix,
            "gold_source_run_id": settings.gold_source_run_id,
            "document_run_id": settings.document_run_id,
            "dataset_prefix": settings.document_dataset_prefix,
            "part_size_rows": settings.document_part_size_rows,
            "max_rows": settings.document_max_rows,
            "total_documents": rows_output,
            "total_parts": len(part_records),
            "parts": [asdict(part) for part in part_records],
        }
        manifest_file = temp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest_key = build_document_manifest_key(
            settings.document_dataset_prefix,
            settings.document_run_id,
        )
        client.fput_object(
            bucket_name=settings.document_bucket,
            object_name=manifest_key,
            file_path=str(manifest_file),
            content_type="application/json",
        )
        uploaded_objects.append(manifest_key)

        quality_summary = {
            "files_processed": len(part_records),
            "rows_read": rows_read,
            "rows_output": rows_output,
            "documents_generated": rows_output,
            "part_size_rows": settings.document_part_size_rows,
            "max_rows": settings.document_max_rows,
            "event_type_counts": dict(sorted(event_type_counts.items())),
        }
        quality_report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "source_bucket": settings.gold_bucket,
            "source_object_key": source_object,
            "document_bucket": settings.document_bucket,
            "gold_source_run_id": settings.gold_source_run_id,
            "document_run_id": settings.document_run_id,
            "gold_dataset_prefix": settings.gold_dataset_prefix,
            "document_dataset_prefix": settings.document_dataset_prefix,
            "artifact_prefix": artifact_prefix,
            "summary": quality_summary,
            "parts": [asdict(part) for part in part_records],
        }
        quality_report_file = temp_path / "quality_report.json"
        quality_report_file.write_text(json.dumps(quality_report, indent=2), encoding="utf-8")
        quality_report_key = build_document_quality_report_key(
            settings.document_dataset_prefix,
            settings.document_run_id,
        )
        client.fput_object(
            bucket_name=settings.document_bucket,
            object_name=quality_report_key,
            file_path=str(quality_report_file),
            content_type="application/json",
        )
        uploaded_objects.append(quality_report_key)

    return DocumentBuildResult(
        uploaded_objects=uploaded_objects,
        artifact_prefix=artifact_prefix,
        manifest_key=manifest_key,
        quality_report_key=quality_report_key,
        files_processed=len(part_records),
        rows_read=rows_read,
        rows_output=rows_output,
        quality_summary=quality_summary,
    )
