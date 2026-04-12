# pyright: reportArgumentType=false

from __future__ import annotations

import csv
import json
import logging
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, cast

from minio import Minio

from rag_intelligence.config import SilverSettings
from rag_intelligence.minio_utils import clean_cell, ensure_bucket

LOGGER = logging.getLogger(__name__)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_WINNER_CT_VALUES = {"CT", "COUNTERTERRORIST", "COUNTER_TERRORIST", "COUNTER-TERRORIST"}
_WINNER_T_VALUES = {"T", "TERRORIST"}

ROUND_META_CONTEXT_COLUMNS = (
    "file",
    "round_number",
    "map",
    "round_type",
    "winner_side_current",
    "ct_eq_val",
    "t_eq_val",
)


@dataclass(frozen=True)
class SilverTransformResult:
    uploaded_objects: list[str]
    artifact_prefix: str
    quality_report_key: str
    files_processed: int
    rows_read: int
    rows_output: int
    quality_summary: dict[str, object]


def build_bronze_extracted_prefix(dataset_prefix: str, run_id: str) -> str:
    normalized_prefix = dataset_prefix.strip("/")
    normalized_run_id = run_id.strip("/")
    return f"{normalized_prefix}/{normalized_run_id}/extracted/"


def build_silver_object_key(dataset_prefix: str, run_id: str, relative_path: str) -> str:
    normalized_prefix = dataset_prefix.strip("/")
    normalized_run_id = run_id.strip("/")
    normalized_relative_path = relative_path.replace("\\", "/").lstrip("/")
    return f"{normalized_prefix}/{normalized_run_id}/cleaned/{normalized_relative_path}"


def build_round_meta_context_key(dataset_prefix: str, run_id: str) -> str:
    return build_silver_object_key(dataset_prefix, run_id, "round_meta_context.csv")


def build_quality_report_key(dataset_prefix: str, run_id: str) -> str:
    normalized_prefix = dataset_prefix.strip("/")
    normalized_run_id = run_id.strip("/")
    return f"{normalized_prefix}/{normalized_run_id}/quality_report.json"


def build_silver_artifact_prefix(dataset_prefix: str, run_id: str) -> str:
    normalized_prefix = dataset_prefix.strip("/")
    normalized_run_id = run_id.strip("/")
    return f"{normalized_prefix}/{normalized_run_id}/cleaned/"


def normalize_column_name(column_name: str) -> str:
    normalized = _NON_ALNUM.sub("_", column_name.strip().lower())
    normalized = normalized.strip("_")
    return normalized or "column"


def normalize_column_names(fieldnames: list[str]) -> list[str]:
    normalized_names: list[str] = []
    seen: dict[str, int] = {}

    for original_name in fieldnames:
        base_name = normalize_column_name(original_name)
        counter = seen.get(base_name, 0) + 1
        seen[base_name] = counter
        normalized_names.append(base_name if counter == 1 else f"{base_name}_{counter}")

    return normalized_names


def _normalize_winner_side(value: str | None) -> str | None:
    text = (value or "").strip().upper()
    if text in _WINNER_CT_VALUES:
        return "CT"
    if text in _WINNER_T_VALUES:
        return "T"
    return None


def _normalize_positive_int(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        number = int(Decimal(value))
    except (InvalidOperation, ValueError):
        return None
    if number <= 0:
        return None
    return str(number)


def _normalize_non_negative_number(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        number = Decimal(value)
    except InvalidOperation:
        return None
    if not number.is_finite() or number < 0:
        return None
    if number == number.to_integral_value():
        return str(int(number))
    return format(number.normalize(), "f").rstrip("0").rstrip(".") or "0"


def _is_round_meta_source(object_name: str) -> bool:
    normalized = object_name.lower()
    if "/round_meta/" in normalized:
        return True
    return "meta" in Path(normalized).name


def _first_available(row: dict[str, str | None], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        value = row.get(candidate)
        if value is not None:
            return value
    return None


def _normalize_round_meta_row(
    normalized_row: dict[str, str | None],
) -> tuple[dict[str, str] | None, str]:
    file_value = _first_available(normalized_row, ("file", "demo_file"))
    round_value = _first_available(
        normalized_row, ("round", "round_number", "round_num", "round_id")
    )
    map_value = _first_available(normalized_row, ("map",))
    round_type = _first_available(normalized_row, ("round_type",)) or "unknown"
    winner_side = _first_available(normalized_row, ("winner_side",))
    ct_eq_val = _first_available(normalized_row, ("ct_eq_val", "ct_eq", "ct_money"))
    t_eq_val = _first_available(normalized_row, ("t_eq_val", "t_eq", "t_money"))

    if not file_value or not round_value or not map_value:
        return None, "missing_required_fields"
    if not winner_side or ct_eq_val is None or t_eq_val is None:
        return None, "missing_required_fields"

    normalized_winner = _normalize_winner_side(winner_side)
    if normalized_winner is None:
        return None, "invalid_winner_side"

    normalized_round = _normalize_positive_int(round_value)
    if normalized_round is None:
        return None, "invalid_round"

    normalized_ct_eq = _normalize_non_negative_number(ct_eq_val)
    normalized_t_eq = _normalize_non_negative_number(t_eq_val)
    if normalized_ct_eq is None or normalized_t_eq is None:
        return None, "invalid_economy"

    return (
        {
            "file": file_value,
            "round_number": normalized_round,
            "map": map_value,
            "round_type": round_type,
            "winner_side_current": normalized_winner,
            "ct_eq_val": normalized_ct_eq,
            "t_eq_val": normalized_t_eq,
        },
        "",
    )


def run_silver_transform(
    settings: SilverSettings,
    *,
    minio_factory=Minio,
) -> SilverTransformResult:
    client = minio_factory(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )

    ensure_bucket(client, settings.silver_bucket)

    source_prefix = build_bronze_extracted_prefix(
        settings.bronze_dataset_prefix,
        settings.bronze_source_run_id,
    )

    source_objects = sorted(
        obj.object_name
        for obj in client.list_objects(
            settings.bronze_bucket, prefix=source_prefix, recursive=True
        )
        if obj.object_name and obj.object_name.lower().endswith(".csv")
    )
    if not source_objects:
        raise FileNotFoundError(
            "No CSV files were found in Bronze for "
            f"run_id={settings.bronze_source_run_id} under prefix={source_prefix}"
        )

    round_meta_sources = [
        object_name for object_name in source_objects if _is_round_meta_source(object_name)
    ]
    if not round_meta_sources:
        raise FileNotFoundError(
            "No round_meta CSV files were found in Bronze for "
            f"run_id={settings.bronze_source_run_id} under prefix={source_prefix}"
        )

    rows_read = 0
    duplicate_round_keys = 0
    invalid_rows = 0
    missing_required_rows = 0
    schema_incompatible_files = 0
    by_round_key: dict[tuple[str, str], dict[str, str]] = {}

    with tempfile.TemporaryDirectory(prefix="silver-transform-") as temp_dir:
        temp_path = Path(temp_dir)
        for source_object in round_meta_sources:
            relative_path = source_object[len(source_prefix) :]
            source_file = temp_path / "in" / relative_path
            source_file.parent.mkdir(parents=True, exist_ok=True)
            client.fget_object(settings.bronze_bucket, source_object, str(source_file))

            with source_file.open(
                "r", newline="", encoding="utf-8-sig", errors="replace"
            ) as csv_file:
                reader = csv.DictReader(csv_file)
                source_fieldnames = list(reader.fieldnames or [])
                normalized_fieldnames = normalize_column_names(source_fieldnames)
                if not source_fieldnames or len(source_fieldnames) != len(normalized_fieldnames):
                    schema_incompatible_files += 1
                    continue

                mapping = list(zip(source_fieldnames, normalized_fieldnames, strict=False))
                has_required = {
                    "file",
                    "round",
                    "map",
                    "winner_side",
                    "ct_eq_val",
                    "t_eq_val",
                }.intersection(set(normalized_fieldnames))
                if len(has_required) < 5 and (
                    "file" not in normalized_fieldnames or "map" not in normalized_fieldnames
                ):
                    # Allow round aliases, but still require minimum semantic fields.
                    schema_incompatible_files += 1
                    continue

                for source_row in reader:
                    rows_read += 1
                    normalized_row: dict[str, str | None] = {}
                    for source_col, normalized_col in mapping:
                        normalized_row[normalized_col] = clean_cell(source_row.get(source_col))

                    normalized_entry, reason = _normalize_round_meta_row(normalized_row)
                    if normalized_entry is None:
                        if reason == "missing_required_fields":
                            missing_required_rows += 1
                        else:
                            invalid_rows += 1
                        continue

                    round_key = (
                        normalized_entry["file"],
                        normalized_entry["round_number"],
                    )
                    if round_key in by_round_key:
                        duplicate_round_keys += 1
                    by_round_key[round_key] = normalized_entry

        if not by_round_key:
            raise ValueError("No valid round_meta rows were produced in Silver.")

        output_rows: list[dict[str, str]] = sorted(
            by_round_key.values(),
            key=lambda row: (row["file"], int(row["round_number"])),
        )
        rows_output = len(output_rows)

        round_meta_context_file = temp_path / "out" / "round_meta_context.csv"
        round_meta_context_file.parent.mkdir(parents=True, exist_ok=True)
        with round_meta_context_file.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(ROUND_META_CONTEXT_COLUMNS))
            writer.writeheader()
            writer.writerows(cast(list[dict[str, Any]], output_rows))

        round_meta_context_key = build_round_meta_context_key(
            settings.silver_dataset_prefix,
            settings.silver_run_id,
        )
        client.fput_object(
            bucket_name=settings.silver_bucket,
            object_name=round_meta_context_key,
            file_path=str(round_meta_context_file),
            content_type="text/csv",
        )

        artifact_prefix = build_silver_artifact_prefix(
            settings.silver_dataset_prefix,
            settings.silver_run_id,
        )
        quality_summary: dict[str, object] = {
            "files_processed": len(round_meta_sources),
            "rows_read": rows_read,
            "rows_output": rows_output,
            "rows_removed": rows_read - rows_output,
            "duplicate_round_keys": duplicate_round_keys,
            "invalid_rows": invalid_rows,
            "missing_required_rows": missing_required_rows,
            "schema_incompatible_files": schema_incompatible_files,
            "source_csv_files_total": len(source_objects),
            "source_round_meta_files": len(round_meta_sources),
        }
        quality_report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "bronze_bucket": settings.bronze_bucket,
            "silver_bucket": settings.silver_bucket,
            "bronze_source_run_id": settings.bronze_source_run_id,
            "silver_run_id": settings.silver_run_id,
            "bronze_dataset_prefix": settings.bronze_dataset_prefix,
            "silver_dataset_prefix": settings.silver_dataset_prefix,
            "artifact_prefix": artifact_prefix,
            "output_object": round_meta_context_key,
            "summary": quality_summary,
        }

        quality_report_file = temp_path / "quality_report.json"
        quality_report_file.write_text(json.dumps(quality_report, indent=2), encoding="utf-8")
        quality_report_key = build_quality_report_key(
            settings.silver_dataset_prefix,
            settings.silver_run_id,
        )
        client.fput_object(
            bucket_name=settings.silver_bucket,
            object_name=quality_report_key,
            file_path=str(quality_report_file),
            content_type="application/json",
        )

    uploaded_objects = [round_meta_context_key, quality_report_key]
    return SilverTransformResult(
        uploaded_objects=uploaded_objects,
        artifact_prefix=artifact_prefix,
        quality_report_key=quality_report_key,
        files_processed=len(round_meta_sources),
        rows_read=rows_read,
        rows_output=rows_output,
        quality_summary=quality_summary,
    )
