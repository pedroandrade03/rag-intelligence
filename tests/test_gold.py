from __future__ import annotations

import csv
import json

import pytest

from conftest import FakeMinio
from rag_intelligence.config import GoldSettings
from rag_intelligence.gold import (
    build_gold_events_key,
    run_gold_transform,
)


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


def test_build_gold_events_key_points_to_round_context() -> None:
    assert (
        build_gold_events_key("csgo-matchmaking-damage", "20260306T023831Z")
        == "csgo-matchmaking-damage/20260306T023831Z/curated/round_context.csv"
    )


def test_run_gold_transform_builds_round_context_and_quality_report() -> None:
    silver_prefix = "csgo-matchmaking-damage/20260306T023831Z/cleaned"
    initial_objects = {
        "silver": {
            f"{silver_prefix}/round_meta_context.csv": (
                b"file,round_number,map,round_type,winner_side_current,ct_eq_val,t_eq_val\n"
                b"demo_1,1,de_mirage,pistol,CT,3000,1200\n"
                b"demo_1,2,de_mirage,gunround,T,4200,5100\n"
                b"demo_1,2,de_mirage,gunround,T,5000,5100\n"
                b"demo_1,31,de_mirage,gunround,CT,6000,5000\n"
                b"demo_2,1,de_inferno,eco,T,1800,2000\n"
                b"demo_2,-1,de_inferno,eco,T,1800,2000\n"
            ),
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
    assert result.files_processed == 1
    assert result.rows_read == 6
    assert result.rows_output == 4
    assert result.artifact_prefix == "csgo-matchmaking-damage/20260306T023831Z/curated/"
    assert result.quality_summary["rows_removed"] == 2
    assert (
        result.events_key == "csgo-matchmaking-damage/20260306T023831Z/curated/round_context.csv"
    )

    round_context_key = result.events_key
    report_key = "csgo-matchmaking-damage/20260306T023831Z/quality_report.json"
    gold_objects = fake_minio.objects["gold"]
    assert round_context_key in gold_objects
    assert report_key in gold_objects

    rows = list(csv.DictReader(gold_objects[round_context_key].decode("utf-8").splitlines()))
    assert len(rows) == 4
    assert rows[0]["file"] == "demo_1"
    assert rows[0]["round_number"] == "1"
    assert rows[0]["eq_diff"] == "1800"
    assert rows[0]["half"] == "H1"
    assert rows[0]["overtime_flag"] == "0"

    assert rows[1]["round_number"] == "2"
    assert rows[1]["ct_eq_val"] == "5000"  # duplicate round keeps latest
    assert rows[1]["eq_diff"] == "-100"

    assert rows[2]["round_number"] == "31"
    assert rows[2]["half"] == "H1"
    assert rows[2]["overtime_flag"] == "1"

    report = json.loads(gold_objects[report_key].decode("utf-8"))
    assert report["artifact_prefix"] == result.artifact_prefix
    assert report["round_context_key"] == round_context_key
    assert report["summary"] == {
        "files_processed": 1,
        "rows_read": 6,
        "rows_output": 4,
        "rows_removed": 2,
        "duplicate_round_keys": 1,
        "invalid_rows": 1,
    }


def test_run_gold_transform_fails_when_round_meta_context_missing() -> None:
    silver_prefix = "csgo-matchmaking-damage/20260306T023831Z/cleaned"
    fake_minio = FakeMinio(
        "localhost:9000",
        "minioadmin",
        "minioadmin",
        False,
        initial_objects={"silver": {f"{silver_prefix}/other.csv": b"a,b\n1,2\n"}},
        existing_buckets={"silver"},
    )

    with pytest.raises(
        FileNotFoundError, match=r"round_meta_context\.csv was not found in Silver"
    ):
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
                f"{silver_prefix}/round_meta_context.csv": (
                    b"file,round_number,map,round_type,winner_side_current,ct_eq_val,t_eq_val\n"
                    b",1,de_mirage,pistol,CT,3000,1200\n"
                )
            }
        },
        existing_buckets={"silver"},
    )

    with pytest.raises(
        ValueError, match=r"No valid rows were produced for Gold round_context\.csv"
    ):
        run_gold_transform(build_settings(), minio_factory=lambda **kwargs: fake_minio)
