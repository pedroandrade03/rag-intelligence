from __future__ import annotations

import csv
import json

import pytest

from conftest import FakeMinio
from rag_intelligence.config import SilverSettings
from rag_intelligence.silver import (
    build_round_meta_context_key,
    build_silver_object_key,
    normalize_column_names,
    run_silver_transform,
)


def build_settings() -> SilverSettings:
    return SilverSettings(
        minio_endpoint="localhost:9000",
        minio_access_key="minioadmin",
        minio_secret_key="minioadmin",
        minio_secure=False,
        bronze_bucket="bronze",
        silver_bucket="silver",
        bronze_dataset_prefix="csgo-matchmaking-damage",
        silver_dataset_prefix="csgo-matchmaking-damage",
        bronze_source_run_id="20260306T023831Z",
        silver_run_id="20260306T023831Z",
    )


def test_normalize_column_names_handles_duplicates_and_empty_values() -> None:
    assert normalize_column_names([" Damage ", "Damage", "Round Number", "!!!"]) == [
        "damage",
        "damage_2",
        "round_number",
        "column",
    ]


def test_build_silver_object_key_uses_cleaned_prefix() -> None:
    key = build_silver_object_key(
        "csgo-matchmaking-damage",
        "20260306T023831Z",
        "round_meta_context.csv",
    )
    assert key == "csgo-matchmaking-damage/20260306T023831Z/cleaned/round_meta_context.csv"


def test_run_silver_transform_builds_round_meta_context_and_report() -> None:
    bronze_prefix = "csgo-matchmaking-damage/20260306T023831Z/extracted"
    initial_objects = {
        "bronze": {
            f"{bronze_prefix}/round_meta/esea_meta_demos.part1.csv": (
                b"file,round,map,winner_side,round_type,ct_eq_val,t_eq_val\n"
                b"demo_1,1,de_mirage,CT,pistol,2500,1200\n"
                b"demo_1,1,de_mirage,COUNTERTERRORIST,pistol,3000,1200\n"
                b"demo_1,2,de_mirage,T,gunround,4200,5100\n"
                b"demo_2,1,de_inferno,T,error,-10,2000\n"
                b"demo_2,2,de_inferno,T,eco,1800,2000\n"
            ),
            f"{bronze_prefix}/kills/esea_master_kills_demos.part1.csv": (
                b"file,round,tick\ndemo_1,1,100\n"
            ),
        }
    }
    fake_minio = FakeMinio(
        "localhost:9000",
        "minioadmin",
        "minioadmin",
        False,
        initial_objects=initial_objects,
        existing_buckets={"bronze"},
    )

    result = run_silver_transform(build_settings(), minio_factory=lambda **kwargs: fake_minio)

    assert "silver" in fake_minio.buckets
    assert result.files_processed == 1
    assert result.rows_read == 5
    assert result.rows_output == 3
    assert result.artifact_prefix == "csgo-matchmaking-damage/20260306T023831Z/cleaned/"

    silver_objects = fake_minio.objects["silver"]
    round_meta_context_key = build_round_meta_context_key(
        "csgo-matchmaking-damage", "20260306T023831Z"
    )
    report_key = "csgo-matchmaking-damage/20260306T023831Z/quality_report.json"
    assert round_meta_context_key in silver_objects
    assert report_key in silver_objects
    assert report_key in result.uploaded_objects

    rows = list(
        csv.DictReader(silver_objects[round_meta_context_key].decode("utf-8").splitlines())
    )
    assert len(rows) == 3
    assert rows[0]["file"] == "demo_1"
    assert rows[0]["round_number"] == "1"
    assert rows[0]["winner_side_current"] == "CT"
    assert rows[0]["ct_eq_val"] == "3000"  # duplicate round keeps latest
    assert rows[1]["winner_side_current"] == "T"
    assert rows[2]["file"] == "demo_2"

    report = json.loads(silver_objects[report_key].decode("utf-8"))
    assert report["artifact_prefix"] == result.artifact_prefix
    assert report["summary"]["files_processed"] == 1
    assert report["summary"]["rows_read"] == 5
    assert report["summary"]["rows_output"] == 3
    assert report["summary"]["rows_removed"] == 2
    assert report["summary"]["duplicate_round_keys"] == 1
    assert report["summary"]["invalid_rows"] == 1


def test_run_silver_transform_fails_when_no_round_meta_csv() -> None:
    bronze_prefix = "csgo-matchmaking-damage/20260306T023831Z/extracted"
    fake_minio = FakeMinio(
        "localhost:9000",
        "minioadmin",
        "minioadmin",
        False,
        initial_objects={
            "bronze": {
                f"{bronze_prefix}/kills/esea_master_kills_demos.part1.csv": (
                    b"file,round,tick\nx,1,1\n"
                )
            }
        },
        existing_buckets={"bronze"},
    )

    with pytest.raises(FileNotFoundError, match="No round_meta CSV files were found in Bronze"):
        run_silver_transform(build_settings(), minio_factory=lambda **kwargs: fake_minio)
