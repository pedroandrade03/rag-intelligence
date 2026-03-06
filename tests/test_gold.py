from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from rag_intelligence.config import GoldSettings
from rag_intelligence.gold import (
    build_gold_events_key,
    project_row,
    run_gold_transform,
)


@dataclass(frozen=True)
class FakeObject:
    object_name: str


class FakeMinio:
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        secure: bool,
        *,
        initial_objects: dict[str, dict[str, bytes]] | None = None,
        existing_buckets: set[str] | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.secure = secure
        self.objects = initial_objects or {}
        self.buckets = set(existing_buckets or set())

    def bucket_exists(self, bucket_name: str) -> bool:
        return bucket_name in self.buckets

    def make_bucket(self, bucket_name: str) -> None:
        self.buckets.add(bucket_name)
        self.objects.setdefault(bucket_name, {})

    def list_objects(self, bucket_name: str, prefix: str, recursive: bool = True):
        del recursive
        for object_name in sorted(self.objects.get(bucket_name, {})):
            if object_name.startswith(prefix):
                yield FakeObject(object_name=object_name)

    def fget_object(self, bucket_name: str, object_name: str, file_path: str) -> None:
        data = self.objects[bucket_name][object_name]
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def fput_object(
        self,
        bucket_name: str,
        object_name: str,
        file_path: str,
        content_type: str,
    ) -> None:
        del content_type
        if bucket_name not in self.buckets:
            raise ValueError(f"Bucket not found: {bucket_name}")
        self.objects.setdefault(bucket_name, {})[object_name] = Path(file_path).read_bytes()


def build_settings() -> GoldSettings:
    return GoldSettings(
        minio_endpoint="localhost:9000",
        minio_access_key="minioadmin",
        minio_secret_key="minioadmin",
        minio_secure=False,
        silver_bucket="silver",
        gold_bucket="gold",
        silver_dataset_prefix="csgo-matchmaking-damage",
        gold_dataset_prefix="csgo-matchmaking-damage",
        silver_source_run_id="20260306T023831Z",
        gold_run_id="20260306T023831Z",
    )


def test_build_gold_events_key_uses_curated_prefix() -> None:
    assert (
        build_gold_events_key("csgo-matchmaking-damage", "20260306T023831Z")
        == "csgo-matchmaking-damage/20260306T023831Z/curated/events.csv"
    )


def test_project_row_applies_map_enrichment_and_weapon_fallback() -> None:
    row = {
        "file": "demo_1",
        "round": "1",
        "map": "",
        "wp": "",
        "nade": "hegrenade",
        "hp_dmg": "",
        "arm_dmg": "12",
        "att_pos_x": "100.0",
        "att_pos_y": "200",
        "vic_pos_x": "-50",
        "vic_pos_y": "10.500",
    }
    projected, reason = project_row(row, {("demo_1", "1"): "de_mirage"})

    assert reason is None
    assert projected == {
        "file": "demo_1",
        "round": "1",
        "map": "de_mirage",
        "weapon": "hegrenade",
        "hp_dmg": "",
        "arm_dmg": "12",
        "att_pos_x": "100",
        "att_pos_y": "200",
        "vic_pos_x": "-50",
        "vic_pos_y": "10.5",
    }


def test_project_row_requires_damage() -> None:
    row = {
        "file": "demo_1",
        "round": "1",
        "map": "de_mirage",
        "wp": "ak47",
        "nade": "",
        "hp_dmg": "",
        "arm_dmg": "",
        "att_pos_x": "100",
        "att_pos_y": "200",
        "vic_pos_x": "300",
        "vic_pos_y": "400",
    }
    projected, reason = project_row(row, {})

    assert projected == {}
    assert reason == "missing_damage"


def test_run_gold_transform_processes_csvs_and_writes_quality_report() -> None:
    silver_prefix = "csgo-matchmaking-damage/20260306T023831Z/cleaned"
    initial_objects = {
        "silver": {
            f"{silver_prefix}/metadata.csv": (
                "file,round,map\n"
                "demo_1,1,de_mirage\n"
            ).encode("utf-8"),
            f"{silver_prefix}/damage.csv": (
                "file,round,map,wp,hp_dmg,arm_dmg,att_pos_x,att_pos_y,vic_pos_x,vic_pos_y\n"
                "demo_1,1,,ak47,32,,100.0,200,-50,10\n"
                "demo_2,2,de_inferno,,10,5,1,2,3,4\n"
            ).encode("utf-8"),
            f"{silver_prefix}/grenades.csv": (
                "file,round,map,nade,hp_dmg,arm_dmg,att_pos_x,att_pos_y,vic_pos_x,vic_pos_y\n"
                "demo_3,3,de_nuke,hegrenade,,40,1,2,3,4\n"
                "demo_4,4,de_nuke,flash,,,1,2,3,4\n"
            ).encode("utf-8"),
            f"{silver_prefix}/map_data.csv": (
                "column,endx,endy\n"
                "0,1,2\n"
            ).encode("utf-8"),
        }
    }
    fake_minio = FakeMinio(
        "localhost:9000",
        "minioadmin",
        "minioadmin",
        False,
        initial_objects=initial_objects,
        existing_buckets={"silver"},
    )

    result = run_gold_transform(build_settings(), minio_factory=lambda **kwargs: fake_minio)

    assert "gold" in fake_minio.buckets
    assert result.files_processed == 4
    assert result.rows_read == 4
    assert result.rows_output == 2

    events_key = "csgo-matchmaking-damage/20260306T023831Z/curated/events.csv"
    report_key = "csgo-matchmaking-damage/20260306T023831Z/quality_report.json"
    gold_objects = fake_minio.objects["gold"]

    assert events_key in gold_objects
    assert report_key in gold_objects
    assert result.events_key == events_key
    assert result.quality_report_key == report_key

    rows = list(csv.DictReader(gold_objects[events_key].decode("utf-8").splitlines()))
    assert rows == [
        {
            "file": "demo_1",
            "round": "1",
            "map": "de_mirage",
            "weapon": "ak47",
            "hp_dmg": "32",
            "arm_dmg": "",
            "att_pos_x": "100",
            "att_pos_y": "200",
            "vic_pos_x": "-50",
            "vic_pos_y": "10",
        },
        {
            "file": "demo_3",
            "round": "3",
            "map": "de_nuke",
            "weapon": "hegrenade",
            "hp_dmg": "",
            "arm_dmg": "40",
            "att_pos_x": "1",
            "att_pos_y": "2",
            "vic_pos_x": "3",
            "vic_pos_y": "4",
        },
    ]

    report = json.loads(gold_objects[report_key].decode("utf-8"))
    assert report["summary"]["files_processed"] == 4
    assert report["summary"]["rows_read"] == 4
    assert report["summary"]["rows_output"] == 2
    assert report["summary"]["rows_removed"] == 2
    assert report["summary"]["removal_reasons"] == {
        "missing_weapon": 1,
        "missing_map": 0,
        "missing_damage": 1,
        "invalid_position": 0,
        "missing_required_fields": 0,
        "schema_incompatible_file": 2,
    }


def test_run_gold_transform_fails_when_no_csv_for_run_id() -> None:
    silver_prefix = "csgo-matchmaking-damage/20260306T023831Z/cleaned"
    fake_minio = FakeMinio(
        "localhost:9000",
        "minioadmin",
        "minioadmin",
        False,
        initial_objects={"silver": {f"{silver_prefix}/not-a-csv.png": b"png"}},
        existing_buckets={"silver"},
    )

    with pytest.raises(FileNotFoundError, match="No CSV files were found in Silver"):
        run_gold_transform(build_settings(), minio_factory=lambda **kwargs: fake_minio)


def test_run_gold_transform_fails_when_no_valid_rows() -> None:
    silver_prefix = "csgo-matchmaking-damage/20260306T023831Z/cleaned"
    fake_minio = FakeMinio(
        "localhost:9000",
        "minioadmin",
        "minioadmin",
        False,
        initial_objects={
            "silver": {
                f"{silver_prefix}/damage.csv": (
                    "file,round,map,wp,hp_dmg,arm_dmg,att_pos_x,att_pos_y,vic_pos_x,vic_pos_y\n"
                    "demo_1,1,,ak47,10,0,1,2,3,4\n"
                ).encode("utf-8"),
            }
        },
        existing_buckets={"silver"},
    )

    with pytest.raises(ValueError, match="No valid rows were produced for Gold"):
        run_gold_transform(build_settings(), minio_factory=lambda **kwargs: fake_minio)
