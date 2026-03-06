from __future__ import annotations

import csv
import json
import logging
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from minio import Minio

from rag_intelligence.config import GoldSettings

LOGGER = logging.getLogger(__name__)

BASE_GOLD_COLUMNS = [
    "file",
    "round",
    "map",
    "weapon",
    "hp_dmg",
    "arm_dmg",
    "att_pos_x",
    "att_pos_y",
    "vic_pos_x",
    "vic_pos_y",
]

EXTRA_GOLD_COLUMNS = [
    "event_type",
    "source_file",
    "tick",
    "seconds",
    "start_seconds",
    "end_seconds",
    "att_team",
    "vic_team",
    "att_side",
    "vic_side",
    "wp_type",
    "nade",
    "hitbox",
    "bomb_site",
    "is_bomb_planted",
    "att_id",
    "vic_id",
    "att_rank",
    "vic_rank",
    "winner_team",
    "winner_side",
    "round_type",
    "ct_eq_val",
    "t_eq_val",
    "ct_alive",
    "t_alive",
    "nade_land_x",
    "nade_land_y",
    "avg_match_rank",
]

GOLD_COLUMNS = BASE_GOLD_COLUMNS + EXTRA_GOLD_COLUMNS

FILE_REQUIRED_COLUMNS = {"file", "round"}
WEAPON_COLUMNS = ("wp", "nade")
DAMAGE_COLUMNS = ("hp_dmg", "arm_dmg")
POSITION_COLUMNS = ("att_pos_x", "att_pos_y", "vic_pos_x", "vic_pos_y")

EVENT_TYPE_DAMAGE = "damage"
EVENT_TYPE_GRENADE = "grenade"
EVENT_TYPE_KILL = "kill"
EVENT_TYPE_ROUND_META = "round_meta"
EVENT_TYPE_MAP_LAYOUT = "map_layout"
EVENT_TYPE_GENERIC = "event"

REMOVAL_REASONS = (
    "missing_weapon",
    "missing_map",
    "missing_damage",
    "invalid_position",
    "missing_required_fields",
)


@dataclass(frozen=True)
class GoldFileQualityMetrics:
    source_object: str
    event_type: str
    rows_read: int
    rows_output: int
    missing_weapon: int
    missing_map: int
    missing_damage: int
    invalid_position: int
    missing_required_fields: int
    schema_incompatible_file: int

    @property
    def rows_removed(self) -> int:
        return (
            self.missing_weapon
            + self.missing_map
            + self.missing_damage
            + self.invalid_position
            + self.missing_required_fields
        )


@dataclass(frozen=True)
class GoldTransformResult:
    uploaded_objects: list[str]
    events_key: str
    quality_report_key: str
    files_processed: int
    rows_read: int
    rows_output: int


def build_silver_cleaned_prefix(dataset_prefix: str, run_id: str) -> str:
    normalized_prefix = dataset_prefix.strip("/")
    normalized_run_id = run_id.strip("/")
    return f"{normalized_prefix}/{normalized_run_id}/cleaned/"


def build_gold_events_key(dataset_prefix: str, run_id: str) -> str:
    normalized_prefix = dataset_prefix.strip("/")
    normalized_run_id = run_id.strip("/")
    return f"{normalized_prefix}/{normalized_run_id}/curated/events.csv"


def build_gold_quality_report_key(dataset_prefix: str, run_id: str) -> str:
    normalized_prefix = dataset_prefix.strip("/")
    normalized_run_id = run_id.strip("/")
    return f"{normalized_prefix}/{normalized_run_id}/quality_report.json"


def clean_cell(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def normalize_finite_numeric(value: str) -> str:
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid numeric value: {value}") from exc

    if not number.is_finite():
        raise ValueError(f"Invalid numeric value: {value}")

    if number == number.to_integral_value():
        return str(int(number))

    normalized = format(number.normalize(), "f").rstrip("0").rstrip(".")
    return normalized or "0"


def infer_event_type(source_object: str, fieldnames: list[str]) -> str:
    name = Path(source_object).name.lower()
    available = set(fieldnames)

    if "map_data" in name:
        return EVENT_TYPE_MAP_LAYOUT
    if "meta" in name:
        return EVENT_TYPE_ROUND_META
    if "kill" in name:
        return EVENT_TYPE_KILL
    if "grenade" in name:
        return EVENT_TYPE_GRENADE
    if "dmg" in name:
        return EVENT_TYPE_DAMAGE
    if "nade" in available:
        return EVENT_TYPE_GRENADE
    if any(column in available for column in DAMAGE_COLUMNS):
        return EVENT_TYPE_DAMAGE

    return EVENT_TYPE_GENERIC


def has_minimum_projection_columns(fieldnames: list[str], event_type: str) -> bool:
    available = set(fieldnames)
    if not FILE_REQUIRED_COLUMNS.issubset(available):
        return False

    if event_type == EVENT_TYPE_MAP_LAYOUT:
        return False
    if event_type == EVENT_TYPE_ROUND_META:
        return "map" in available
    if event_type == EVENT_TYPE_KILL:
        return any(column in available for column in WEAPON_COLUMNS)
    if event_type == EVENT_TYPE_DAMAGE:
        has_weapon = any(column in available for column in WEAPON_COLUMNS)
        has_damage = any(column in available for column in DAMAGE_COLUMNS)
        return has_weapon and has_damage
    if event_type == EVENT_TYPE_GRENADE:
        return any(column in available for column in WEAPON_COLUMNS)

    return True


def build_map_lookup(source_files: list[tuple[str, Path]]) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}

    for _, source_file in source_files:
        with source_file.open("r", newline="", encoding="utf-8-sig", errors="replace") as csv_file:
            reader = csv.DictReader(csv_file)
            fieldnames = set(reader.fieldnames or [])
            if {"file", "round", "map"}.issubset(fieldnames):
                for row in reader:
                    file_value = clean_cell(row.get("file"))
                    round_value = clean_cell(row.get("round"))
                    map_value = clean_cell(row.get("map"))
                    if file_value and round_value and map_value:
                        lookup[(file_value, round_value)] = map_value

    return lookup


def project_row(
    row: dict[str, str | None],
    map_lookup: dict[tuple[str, str], str],
    *,
    event_type: str,
    source_file: str,
) -> tuple[dict[str, str], str | None]:
    file_value = clean_cell(row.get("file"))
    round_value = clean_cell(row.get("round"))
    map_value = clean_cell(row.get("map"))
    wp_value = clean_cell(row.get("wp"))
    nade_value = clean_cell(row.get("nade"))
    weapon_value = wp_value or nade_value
    hp_dmg_value = clean_cell(row.get("hp_dmg"))
    arm_dmg_value = clean_cell(row.get("arm_dmg"))

    if not map_value and file_value and round_value:
        map_value = map_lookup.get((file_value, round_value))

    if not file_value or not round_value:
        return {}, "missing_required_fields"
    if not map_value:
        return {}, "missing_map"

    if event_type in {EVENT_TYPE_DAMAGE, EVENT_TYPE_GRENADE, EVENT_TYPE_KILL} and not weapon_value:
        return {}, "missing_weapon"
    if event_type == EVENT_TYPE_DAMAGE and not hp_dmg_value and not arm_dmg_value:
        return {}, "missing_damage"

    normalized_positions = {column: "" for column in POSITION_COLUMNS}
    for column in POSITION_COLUMNS:
        value = clean_cell(row.get(column))
        if value is None:
            continue
        try:
            normalized_positions[column] = normalize_finite_numeric(value)
        except ValueError:
            return {}, "invalid_position"

    projected = {column: "" for column in GOLD_COLUMNS}
    projected.update(
        {
            "file": file_value,
            "round": round_value,
            "map": map_value,
            "weapon": weapon_value or "",
            "hp_dmg": hp_dmg_value or "",
            "arm_dmg": arm_dmg_value or "",
            "att_pos_x": normalized_positions["att_pos_x"],
            "att_pos_y": normalized_positions["att_pos_y"],
            "vic_pos_x": normalized_positions["vic_pos_x"],
            "vic_pos_y": normalized_positions["vic_pos_y"],
            "event_type": event_type,
            "source_file": source_file,
            "tick": clean_cell(row.get("tick")) or "",
            "seconds": clean_cell(row.get("seconds")) or "",
            "start_seconds": clean_cell(row.get("start_seconds")) or "",
            "end_seconds": clean_cell(row.get("end_seconds")) or "",
            "att_team": clean_cell(row.get("att_team")) or "",
            "vic_team": clean_cell(row.get("vic_team")) or "",
            "att_side": clean_cell(row.get("att_side")) or "",
            "vic_side": clean_cell(row.get("vic_side")) or "",
            "wp_type": clean_cell(row.get("wp_type")) or "",
            "nade": nade_value or "",
            "hitbox": clean_cell(row.get("hitbox")) or "",
            "bomb_site": clean_cell(row.get("bomb_site")) or "",
            "is_bomb_planted": clean_cell(row.get("is_bomb_planted")) or "",
            "att_id": clean_cell(row.get("att_id")) or "",
            "vic_id": clean_cell(row.get("vic_id")) or "",
            "att_rank": clean_cell(row.get("att_rank")) or "",
            "vic_rank": clean_cell(row.get("vic_rank")) or "",
            "winner_team": clean_cell(row.get("winner_team")) or "",
            "winner_side": clean_cell(row.get("winner_side")) or "",
            "round_type": clean_cell(row.get("round_type")) or "",
            "ct_eq_val": clean_cell(row.get("ct_eq_val")) or "",
            "t_eq_val": clean_cell(row.get("t_eq_val")) or "",
            "ct_alive": clean_cell(row.get("ct_alive")) or "",
            "t_alive": clean_cell(row.get("t_alive")) or "",
            "nade_land_x": clean_cell(row.get("nade_land_x")) or "",
            "nade_land_y": clean_cell(row.get("nade_land_y")) or "",
            "avg_match_rank": clean_cell(row.get("avg_match_rank")) or "",
        }
    )

    return projected, None


def ensure_bucket(client: Minio, bucket_name: str) -> None:
    if client.bucket_exists(bucket_name):
        return
    LOGGER.info("Creating bucket %s", bucket_name)
    client.make_bucket(bucket_name)


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
        if obj.object_name.lower().endswith(".csv")
    )
    if not source_objects:
        raise FileNotFoundError(
            "No CSV files were found in Silver for "
            f"run_id={settings.silver_source_run_id} under prefix={source_prefix}"
        )

    total_rows_read = 0
    total_rows_output = 0
    file_reports: list[GoldFileQualityMetrics] = []
    removal_totals = {reason: 0 for reason in REMOVAL_REASONS}
    schema_incompatible_total = 0
    uploaded_objects: list[str] = []

    with tempfile.TemporaryDirectory(prefix="gold-transform-") as temp_dir:
        temp_path = Path(temp_dir)
        local_sources: list[tuple[str, str, Path]] = []

        for source_object in source_objects:
            relative_path = source_object[len(source_prefix) :]
            source_file = temp_path / "in" / relative_path
            source_file.parent.mkdir(parents=True, exist_ok=True)
            client.fget_object(settings.silver_bucket, source_object, str(source_file))
            local_sources.append((source_object, relative_path, source_file))

        map_lookup = build_map_lookup(
            [(source_object, source_file) for source_object, _, source_file in local_sources]
        )

        events_file = temp_path / "out" / "events.csv"
        events_file.parent.mkdir(parents=True, exist_ok=True)

        with events_file.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=GOLD_COLUMNS)
            writer.writeheader()

            for source_object, relative_path, source_file in local_sources:
                with source_file.open(
                    "r",
                    newline="",
                    encoding="utf-8-sig",
                    errors="replace",
                ) as csv_file:
                    reader = csv.DictReader(csv_file)
                    fieldnames = list(reader.fieldnames or [])
                    event_type = infer_event_type(source_object, fieldnames)
                    if not has_minimum_projection_columns(fieldnames, event_type):
                        metrics = GoldFileQualityMetrics(
                            source_object=source_object,
                            event_type=event_type,
                            rows_read=0,
                            rows_output=0,
                            missing_weapon=0,
                            missing_map=0,
                            missing_damage=0,
                            invalid_position=0,
                            missing_required_fields=0,
                            schema_incompatible_file=1,
                        )
                        schema_incompatible_total += 1
                        file_reports.append(metrics)
                        continue

                    reason_counts = {reason: 0 for reason in REMOVAL_REASONS}
                    rows_read = 0
                    rows_output = 0

                    for row in reader:
                        rows_read += 1
                        projected, reason = project_row(
                            row,
                            map_lookup,
                            event_type=event_type,
                            source_file=relative_path,
                        )
                        if reason is not None:
                            reason_counts[reason] += 1
                            continue

                        writer.writerow(projected)
                        rows_output += 1

                    metrics = GoldFileQualityMetrics(
                        source_object=source_object,
                        event_type=event_type,
                        rows_read=rows_read,
                        rows_output=rows_output,
                        missing_weapon=reason_counts["missing_weapon"],
                        missing_map=reason_counts["missing_map"],
                        missing_damage=reason_counts["missing_damage"],
                        invalid_position=reason_counts["invalid_position"],
                        missing_required_fields=reason_counts["missing_required_fields"],
                        schema_incompatible_file=0,
                    )

                total_rows_read += metrics.rows_read
                total_rows_output += metrics.rows_output
                for reason in REMOVAL_REASONS:
                    removal_totals[reason] += getattr(metrics, reason)
                file_reports.append(metrics)

        if total_rows_output <= 0:
            raise ValueError(
                "No valid rows were produced for Gold after applying required schema rules."
            )

        events_key = build_gold_events_key(
            settings.gold_dataset_prefix,
            settings.gold_run_id,
        )
        client.fput_object(
            bucket_name=settings.gold_bucket,
            object_name=events_key,
            file_path=str(events_file),
            content_type="text/csv",
        )
        uploaded_objects.append(events_key)

        quality_report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "silver_bucket": settings.silver_bucket,
            "gold_bucket": settings.gold_bucket,
            "silver_source_run_id": settings.silver_source_run_id,
            "gold_run_id": settings.gold_run_id,
            "silver_dataset_prefix": settings.silver_dataset_prefix,
            "gold_dataset_prefix": settings.gold_dataset_prefix,
            "events_key": events_key,
            "files": [
                {
                    **asdict(metrics),
                    "rows_removed": metrics.rows_removed,
                }
                for metrics in file_reports
            ],
            "summary": {
                "files_processed": len(file_reports),
                "rows_read": total_rows_read,
                "rows_output": total_rows_output,
                "rows_removed": total_rows_read - total_rows_output,
                "removal_reasons": {
                    **removal_totals,
                    "schema_incompatible_file": schema_incompatible_total,
                },
            },
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
        uploaded_objects.append(quality_report_key)

    return GoldTransformResult(
        uploaded_objects=uploaded_objects,
        events_key=events_key,
        quality_report_key=quality_report_key,
        files_processed=len(file_reports),
        rows_read=total_rows_read,
        rows_output=total_rows_output,
    )
