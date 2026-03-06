from __future__ import annotations

import csv
import json
import logging
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from minio import Minio

from rag_intelligence.config import SilverSettings

LOGGER = logging.getLogger(__name__)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")

_NUMERIC_BASE_COLUMNS = {"damage", "health", "armor", "money", "tick", "distance"}
_NUMERIC_EXACT_COLUMNS = {"round", "round_id", "round_num", "round_number", "flash_duration"}


@dataclass(frozen=True)
class FileQualityMetrics:
    rows_read: int
    rows_output: int
    duplicate_rows: int
    invalid_rows: int
    all_null_rows: int

    @property
    def rows_removed(self) -> int:
        return self.duplicate_rows + self.invalid_rows + self.all_null_rows


@dataclass(frozen=True)
class SilverTransformResult:
    uploaded_objects: list[str]
    quality_report_key: str
    files_processed: int
    rows_read: int
    rows_output: int


def build_bronze_extracted_prefix(dataset_prefix: str, run_id: str) -> str:
    normalized_prefix = dataset_prefix.strip("/")
    normalized_run_id = run_id.strip("/")
    return f"{normalized_prefix}/{normalized_run_id}/extracted/"


def build_silver_object_key(dataset_prefix: str, run_id: str, relative_path: str) -> str:
    normalized_prefix = dataset_prefix.strip("/")
    normalized_run_id = run_id.strip("/")
    normalized_relative_path = relative_path.replace("\\", "/").lstrip("/")
    return f"{normalized_prefix}/{normalized_run_id}/cleaned/{normalized_relative_path}"


def build_quality_report_key(dataset_prefix: str, run_id: str) -> str:
    normalized_prefix = dataset_prefix.strip("/")
    normalized_run_id = run_id.strip("/")
    return f"{normalized_prefix}/{normalized_run_id}/quality_report.json"


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

        if counter == 1:
            normalized_names.append(base_name)
        else:
            normalized_names.append(f"{base_name}_{counter}")

    return normalized_names


def clean_cell(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def is_numeric_metric_column(column_name: str) -> bool:
    if column_name in _NUMERIC_EXACT_COLUMNS:
        return True

    for base_column in _NUMERIC_BASE_COLUMNS:
        if (
            column_name == base_column
            or column_name.startswith(f"{base_column}_")
            or column_name.endswith(f"_{base_column}")
        ):
            return True

    return False


def normalize_numeric_value(value: str) -> str:
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid numeric value: {value}") from exc

    if not number.is_finite() or number < 0:
        raise ValueError(f"Invalid numeric value: {value}")

    if number == number.to_integral_value():
        return str(int(number))

    normalized = format(number.normalize(), "f").rstrip("0").rstrip(".")
    return normalized or "0"


def clean_csv_file(source_path: Path, target_path: Path) -> FileQualityMetrics:
    rows_read = 0
    rows_output = 0
    duplicate_rows = 0
    invalid_rows = 0
    all_null_rows = 0

    target_path.parent.mkdir(parents=True, exist_ok=True)

    with source_path.open("r", newline="", encoding="utf-8-sig", errors="replace") as source_file:
        reader = csv.DictReader(source_file)
        source_fieldnames = list(reader.fieldnames or [])
        normalized_fieldnames = normalize_column_names(source_fieldnames)
        mapping = list(zip(source_fieldnames, normalized_fieldnames, strict=False))
        numeric_columns = [col for col in normalized_fieldnames if is_numeric_metric_column(col)]

        with target_path.open("w", newline="", encoding="utf-8") as target_file:
            writer = csv.DictWriter(target_file, fieldnames=normalized_fieldnames)
            if normalized_fieldnames:
                writer.writeheader()

            seen_rows: set[tuple[str, ...]] = set()

            for source_row in reader:
                rows_read += 1
                normalized_row: dict[str, str | None] = {}

                for source_column, normalized_column in mapping:
                    normalized_row[normalized_column] = clean_cell(source_row.get(source_column))

                if normalized_row and all(value is None for value in normalized_row.values()):
                    all_null_rows += 1
                    continue

                invalid_row = False
                for numeric_column in numeric_columns:
                    current_value = normalized_row.get(numeric_column)
                    if current_value is None:
                        continue
                    try:
                        normalized_row[numeric_column] = normalize_numeric_value(current_value)
                    except ValueError:
                        invalid_row = True
                        break

                if invalid_row:
                    invalid_rows += 1
                    continue

                dedup_key = tuple((normalized_row.get(col) or "") for col in normalized_fieldnames)
                if dedup_key in seen_rows:
                    duplicate_rows += 1
                    continue

                seen_rows.add(dedup_key)
                out = {col: normalized_row.get(col) or "" for col in normalized_fieldnames}
                writer.writerow(out)
                rows_output += 1

    return FileQualityMetrics(
        rows_read=rows_read,
        rows_output=rows_output,
        duplicate_rows=duplicate_rows,
        invalid_rows=invalid_rows,
        all_null_rows=all_null_rows,
    )


def ensure_bucket(client: Minio, bucket_name: str) -> None:
    if client.bucket_exists(bucket_name):
        return
    LOGGER.info("Creating bucket %s", bucket_name)
    client.make_bucket(bucket_name)


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

    source_objects: list[str] = sorted(
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

    uploaded_objects: list[str] = []
    quality_files: list[dict[str, object]] = []
    total_rows_read = 0
    total_rows_output = 0

    with tempfile.TemporaryDirectory(prefix="silver-transform-") as temp_dir:
        temp_path = Path(temp_dir)

        for source_object in source_objects:
            relative_path = source_object[len(source_prefix) :]
            source_file = temp_path / "in" / relative_path
            target_file = temp_path / "out" / relative_path
            source_file.parent.mkdir(parents=True, exist_ok=True)

            LOGGER.info("Processing %s", source_object)
            client.fget_object(settings.bronze_bucket, source_object, str(source_file))
            metrics = clean_csv_file(source_file, target_file)

            target_object = build_silver_object_key(
                settings.silver_dataset_prefix,
                settings.silver_run_id,
                relative_path,
            )
            client.fput_object(
                bucket_name=settings.silver_bucket,
                object_name=target_object,
                file_path=str(target_file),
                content_type="text/csv",
            )
            uploaded_objects.append(target_object)
            total_rows_read += metrics.rows_read
            total_rows_output += metrics.rows_output
            quality_files.append(
                {
                    "source_object": source_object,
                    "target_object": target_object,
                    **asdict(metrics),
                    "rows_removed": metrics.rows_removed,
                }
            )

        quality_report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "bronze_bucket": settings.bronze_bucket,
            "silver_bucket": settings.silver_bucket,
            "bronze_source_run_id": settings.bronze_source_run_id,
            "silver_run_id": settings.silver_run_id,
            "bronze_dataset_prefix": settings.bronze_dataset_prefix,
            "silver_dataset_prefix": settings.silver_dataset_prefix,
            "files": quality_files,
            "summary": {
                "files_processed": len(quality_files),
                "rows_read": total_rows_read,
                "rows_output": total_rows_output,
                "rows_removed": total_rows_read - total_rows_output,
            },
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

    uploaded_objects.append(quality_report_key)
    return SilverTransformResult(
        uploaded_objects=uploaded_objects,
        quality_report_key=quality_report_key,
        files_processed=len(quality_files),
        rows_read=total_rows_read,
        rows_output=total_rows_output,
    )
