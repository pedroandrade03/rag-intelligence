from __future__ import annotations

import csv
import json
import logging
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from minio import Minio

from rag_intelligence.config import GoldSettings
from rag_intelligence.minio_utils import clean_cell, ensure_bucket

LOGGER = logging.getLogger(__name__)

ROUND_CONTEXT_COLUMNS = [
    "file",
    "round_number",
    "map",
    "round_type",
    "winner_side_current",
    "ct_eq_val",
    "t_eq_val",
    "eq_diff",
    "half",
    "overtime_flag",
]

_WINNER_CT_VALUES = {"CT", "COUNTERTERRORIST", "COUNTER_TERRORIST", "COUNTER-TERRORIST"}
_WINNER_T_VALUES = {"T", "TERRORIST"}


@dataclass(frozen=True)
class GoldTransformResult:
    uploaded_objects: list[str]
    artifact_prefix: str
    events_key: str
    quality_report_key: str
    files_processed: int
    rows_read: int
    rows_output: int
    quality_summary: dict[str, object]


def build_silver_cleaned_prefix(dataset_prefix: str, run_id: str) -> str:
    normalized_prefix = dataset_prefix.strip("/")
    normalized_run_id = run_id.strip("/")
    return f"{normalized_prefix}/{normalized_run_id}/cleaned/"


def build_silver_round_meta_context_key(dataset_prefix: str, run_id: str) -> str:
    normalized_prefix = dataset_prefix.strip("/")
    normalized_run_id = run_id.strip("/")
    return f"{normalized_prefix}/{normalized_run_id}/cleaned/round_meta_context.csv"


def build_gold_events_key(dataset_prefix: str, run_id: str) -> str:
    # Kept for backward naming compatibility in callers. In ML-first mode
    # this points to round-level context instead of event-level rows.
    normalized_prefix = dataset_prefix.strip("/")
    normalized_run_id = run_id.strip("/")
    return f"{normalized_prefix}/{normalized_run_id}/curated/round_context.csv"


def build_gold_quality_report_key(dataset_prefix: str, run_id: str) -> str:
    normalized_prefix = dataset_prefix.strip("/")
    normalized_run_id = run_id.strip("/")
    return f"{normalized_prefix}/{normalized_run_id}/quality_report.json"


def build_gold_artifact_prefix(dataset_prefix: str, run_id: str) -> str:
    normalized_prefix = dataset_prefix.strip("/")
    normalized_run_id = run_id.strip("/")
    return f"{normalized_prefix}/{normalized_run_id}/curated/"


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
    normalized = format(number.normalize(), "f").rstrip("0").rstrip(".")
    return normalized or "0"


def _compute_half(round_number: int) -> str:
    in_cycle = ((round_number - 1) % 30) + 1
    return "H1" if in_cycle <= 15 else "H2"


def run_gold_transform(
    settings: GoldSettings,
    *,
    minio_factory=Minio,
) -> GoldTransformResult:
    client = minio_factory(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )

    ensure_bucket(client, settings.gold_bucket)

    round_meta_context_key = build_silver_round_meta_context_key(
        settings.silver_dataset_prefix,
        settings.silver_source_run_id,
    )
    source_prefix = build_silver_cleaned_prefix(
        settings.silver_dataset_prefix,
        settings.silver_source_run_id,
    )
    source_objects = sorted(
        obj.object_name
        for obj in client.list_objects(
            settings.silver_bucket,
            prefix=source_prefix,
            recursive=True,
        )
        if obj.object_name and obj.object_name.lower().endswith(".csv")
    )
    if not source_objects:
        raise FileNotFoundError(
            "No CSV files were found in Silver for "
            f"run_id={settings.silver_source_run_id} under prefix={source_prefix}"
        )
    if round_meta_context_key not in source_objects:
        raise FileNotFoundError(
            "round_meta_context.csv was not found in Silver for "
            f"run_id={settings.silver_source_run_id}: {round_meta_context_key}"
        )

    rows_read = 0
    invalid_rows = 0
    duplicate_round_keys = 0
    by_round_key: dict[tuple[str, str], dict[str, str]] = {}

    with tempfile.TemporaryDirectory(prefix="gold-transform-") as temp_dir:
        temp_path = Path(temp_dir)
        source_file = temp_path / "in" / "round_meta_context.csv"
        source_file.parent.mkdir(parents=True, exist_ok=True)
        client.fget_object(settings.silver_bucket, round_meta_context_key, str(source_file))

        with source_file.open("r", newline="", encoding="utf-8-sig", errors="replace") as csv_file:
            reader = csv.DictReader(csv_file)
            fieldnames = set(reader.fieldnames or [])
            required_columns: set[str] = set(
                (
                    "file",
                    "round_number",
                    "map",
                    "round_type",
                    "winner_side_current",
                    "ct_eq_val",
                    "t_eq_val",
                )
            )
            missing_columns = sorted(required_columns - fieldnames)
            if missing_columns:
                joined_missing = ", ".join(missing_columns)
                raise ValueError(
                    f"Silver round_meta_context.csv is missing required columns: {joined_missing}"
                )

            for row in reader:
                rows_read += 1
                file_value = clean_cell(row.get("file"))
                round_number_value = _normalize_positive_int(clean_cell(row.get("round_number")))
                map_value = clean_cell(row.get("map"))
                round_type = clean_cell(row.get("round_type")) or "unknown"
                winner_side = _normalize_winner_side(clean_cell(row.get("winner_side_current")))
                ct_eq_val = _normalize_non_negative_number(clean_cell(row.get("ct_eq_val")))
                t_eq_val = _normalize_non_negative_number(clean_cell(row.get("t_eq_val")))

                if (
                    not file_value
                    or round_number_value is None
                    or not map_value
                    or winner_side is None
                    or ct_eq_val is None
                    or t_eq_val is None
                ):
                    invalid_rows += 1
                    continue

                round_number = int(round_number_value)
                eq_diff = Decimal(ct_eq_val) - Decimal(t_eq_val)
                if eq_diff == eq_diff.to_integral_value():
                    eq_diff_str = str(int(eq_diff))
                else:
                    eq_diff_str = format(eq_diff.normalize(), "f").rstrip("0").rstrip(".") or "0"

                output_row = {
                    "file": file_value,
                    "round_number": round_number_value,
                    "map": map_value,
                    "round_type": round_type,
                    "winner_side_current": winner_side,
                    "ct_eq_val": ct_eq_val,
                    "t_eq_val": t_eq_val,
                    "eq_diff": eq_diff_str,
                    "half": _compute_half(round_number),
                    "overtime_flag": "1" if round_number > 30 else "0",
                }

                round_key = (output_row["file"], output_row["round_number"])
                if round_key in by_round_key:
                    duplicate_round_keys += 1
                by_round_key[round_key] = output_row

        if not by_round_key:
            raise ValueError("No valid rows were produced for Gold round_context.csv.")

        output_rows = sorted(
            by_round_key.values(),
            key=lambda item: (item["file"], int(item["round_number"])),
        )
        rows_output = len(output_rows)

        round_context_file = temp_path / "out" / "round_context.csv"
        round_context_file.parent.mkdir(parents=True, exist_ok=True)
        with round_context_file.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=ROUND_CONTEXT_COLUMNS)
            writer.writeheader()
            writer.writerows(output_rows)

        events_key = build_gold_events_key(
            settings.gold_dataset_prefix,
            settings.gold_run_id,
        )
        client.fput_object(
            bucket_name=settings.gold_bucket,
            object_name=events_key,
            file_path=str(round_context_file),
            content_type="text/csv",
        )

        artifact_prefix = build_gold_artifact_prefix(
            settings.gold_dataset_prefix, settings.gold_run_id
        )
        quality_summary: dict[str, object] = {
            "files_processed": 1,
            "rows_read": rows_read,
            "rows_output": rows_output,
            "rows_removed": rows_read - rows_output,
            "duplicate_round_keys": duplicate_round_keys,
            "invalid_rows": invalid_rows,
        }
        quality_report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "silver_bucket": settings.silver_bucket,
            "gold_bucket": settings.gold_bucket,
            "silver_source_run_id": settings.silver_source_run_id,
            "gold_run_id": settings.gold_run_id,
            "silver_dataset_prefix": settings.silver_dataset_prefix,
            "gold_dataset_prefix": settings.gold_dataset_prefix,
            "artifact_prefix": artifact_prefix,
            "round_context_key": events_key,
            "summary": quality_summary,
        }

        report_file = temp_path / "quality_report.json"
        report_file.write_text(json.dumps(quality_report, indent=2), encoding="utf-8")
        quality_report_key = build_gold_quality_report_key(
            settings.gold_dataset_prefix,
            settings.gold_run_id,
        )
        client.fput_object(
            bucket_name=settings.gold_bucket,
            object_name=quality_report_key,
            file_path=str(report_file),
            content_type="application/json",
        )

    uploaded_objects = [events_key, quality_report_key]
    return GoldTransformResult(
        uploaded_objects=uploaded_objects,
        artifact_prefix=artifact_prefix,
        events_key=events_key,
        quality_report_key=quality_report_key,
        files_processed=1,
        rows_read=rows_read,
        rows_output=rows_output,
        quality_summary=quality_summary,
    )
