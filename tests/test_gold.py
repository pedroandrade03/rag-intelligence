from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from rag_intelligence.config import GoldSettings
from rag_intelligence.gold import (
    EVENT_TYPE_DAMAGE,
    EVENT_TYPE_GRENADE,
    EVENT_TYPE_KILL,
    EVENT_TYPE_ROUND_META,
    build_gold_events_key,
    infer_event_type,
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


def test_infer_event_type_uses_file_hints() -> None:
    assert infer_event_type("x/esea_master_dmg_demos.part1.csv", ["file", "round"]) == EVENT_TYPE_DAMAGE
    assert infer_event_type("x/esea_master_grenades_demos.part1.csv", ["file", "round"]) == EVENT_TYPE_GRENADE
    assert infer_event_type("x/esea_master_kills_demos.part1.csv", ["file", "round"]) == EVENT_TYPE_KILL
    assert infer_event_type("x/esea_meta_demos.part1.csv", ["file", "round"]) == EVENT_TYPE_ROUND_META


def test_project_row_applies_map_enrichment_and_allows_missing_victim_position() -> None:
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
        "vic_pos_x": "",
        "vic_pos_y": "",
    }
    projected, reason = project_row(
        row,
        {("demo_1", "1"): "de_mirage"},
        event_type=EVENT_TYPE_GRENADE,
        source_file="grenades.csv",
    )

    assert reason is None
    assert projected["map"] == "de_mirage"
    assert projected["weapon"] == "hegrenade"
    assert projected["att_pos_x"] == "100"
    assert projected["att_pos_y"] == "200"
    assert projected["vic_pos_x"] == ""
    assert projected["vic_pos_y"] == ""
    assert projected["event_type"] == EVENT_TYPE_GRENADE
    assert projected["source_file"] == "grenades.csv"


def test_project_row_rejects_invalid_position_values() -> None:
    row = {
        "file": "demo_1",
        "round": "1",
        "map": "de_mirage",
        "wp": "ak47",
        "nade": "",
        "hp_dmg": "20",
        "arm_dmg": "1",
        "att_pos_x": "abc",
        "att_pos_y": "10",
        "vic_pos_x": "20",
        "vic_pos_y": "30",
    }
    projected, reason = project_row(
        row,
        {},
        event_type=EVENT_TYPE_DAMAGE,
        source_file="damage.csv",
    )

    assert projected == {}
    assert reason == "invalid_position"


def test_run_gold_transform_processes_mixed_event_types_and_writes_quality_report() -> None:
    silver_prefix = "csgo-matchmaking-damage/20260306T023831Z/cleaned"
    initial_objects = {
        "silver": {
            f"{silver_prefix}/metadata.csv": (
                "file,round,map,start_seconds,end_seconds,winner_team,winner_side,round_type,ct_eq_val,t_eq_val\n"
                "demo_1,1,de_mirage,0,45,ct,ct,eco,2500,1200\n"
                "demo_kill,1,de_inferno,0,50,t,t,full_buy,3000,4000\n"
            ).encode("utf-8"),
            f"{silver_prefix}/damage.csv": (
                "file,round,map,wp,hp_dmg,arm_dmg,att_pos_x,att_pos_y,vic_pos_x,vic_pos_y,hitbox\n"
                "demo_1,1,,ak47,32,,100.0,200,,,head\n"
                "demo_2,2,de_inferno,,10,5,1,2,3,4,chest\n"
            ).encode("utf-8"),
            f"{silver_prefix}/grenades.csv": (
                "file,round,map,nade,hp_dmg,arm_dmg,att_pos_x,att_pos_y,vic_pos_x,vic_pos_y,nade_land_x,nade_land_y\n"
                "demo_3,3,de_nuke,hegrenade,,40,1,2,,,500,700\n"
                "demo_4,4,de_nuke,flash,,,,2,,,200,250\n"
            ).encode("utf-8"),
            f"{silver_prefix}/kills.csv": (
                "file,round,tick,seconds,att_team,vic_team,att_side,vic_side,wp,wp_type,ct_alive,t_alive,is_bomb_planted\n"
                "demo_kill,1,100,12.3,A,B,t,ct,ak47,rifle,4,2,false\n"
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
    assert result.files_processed == 5
    assert result.rows_read == 7
    assert result.rows_output == 6

    events_key = "csgo-matchmaking-damage/20260306T023831Z/curated/events.csv"
    report_key = "csgo-matchmaking-damage/20260306T023831Z/quality_report.json"
    gold_objects = fake_minio.objects["gold"]

    assert events_key in gold_objects
    assert report_key in gold_objects
    assert result.events_key == events_key
    assert result.quality_report_key == report_key

    rows = list(csv.DictReader(gold_objects[events_key].decode("utf-8").splitlines()))
    assert len(rows) == 6
    assert rows[0]["event_type"] == EVENT_TYPE_DAMAGE
    assert rows[0]["map"] == "de_mirage"
    assert rows[0]["vic_pos_x"] == ""
    assert rows[0]["hitbox"] == "head"

    event_type_counts = {EVENT_TYPE_DAMAGE: 0, EVENT_TYPE_GRENADE: 0, EVENT_TYPE_KILL: 0, EVENT_TYPE_ROUND_META: 0}
    for row in rows:
        event_type_counts[row["event_type"]] += 1

    assert event_type_counts == {
        EVENT_TYPE_DAMAGE: 1,
        EVENT_TYPE_GRENADE: 2,
        EVENT_TYPE_KILL: 1,
        EVENT_TYPE_ROUND_META: 2,
    }

    report = json.loads(gold_objects[report_key].decode("utf-8"))
    assert report["summary"]["files_processed"] == 5
    assert report["summary"]["rows_read"] == 7
    assert report["summary"]["rows_output"] == 6
    assert report["summary"]["rows_removed"] == 1
    assert report["summary"]["removal_reasons"] == {
        "missing_weapon": 1,
        "missing_map": 0,
        "missing_damage": 0,
        "invalid_position": 0,
        "missing_required_fields": 0,
        "schema_incompatible_file": 1,
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
