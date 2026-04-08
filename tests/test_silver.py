from __future__ import annotations

import json

import pytest

from conftest import FakeMinio
from rag_intelligence.config import SilverSettings
from rag_intelligence.silver import (
    build_silver_object_key,
    clean_csv_file,
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
    assert normalize_column_names([" Damage ", "Damage", "Flash Duration", "!!!"]) == [
        "damage",
        "damage_2",
        "flash_duration",
        "column",
    ]


def test_build_silver_object_key_uses_cleaned_prefix() -> None:
    key = build_silver_object_key(
        "csgo-matchmaking-damage",
        "20260306T023831Z",
        "maps/mm_master_demos.csv",
    )
    assert key == "csgo-matchmaking-damage/20260306T023831Z/cleaned/maps/mm_master_demos.csv"


def test_clean_csv_file_applies_schema_aware_rules(tmp_path) -> None:
    source = tmp_path / "source.csv"
    target = tmp_path / "target.csv"
    source.write_text(
        (
            "Damage,Tick,Player,Notes\n"
            "10,1, Alice , ok \n"
            "10,1,Alice,ok\n"
            "-5,2,Bob,oops\n"
            "abc,4,Eve,err\n"
            ",,,\n"
        ),
        encoding="utf-8",
    )

    metrics = clean_csv_file(source, target)
    output = target.read_text(encoding="utf-8")

    assert metrics.rows_read == 5
    assert metrics.rows_output == 1
    assert metrics.duplicate_rows == 1
    assert metrics.invalid_rows == 2
    assert metrics.all_null_rows == 1
    assert output == "damage,tick,player,notes\n10,1,Alice,ok\n"


def test_run_silver_transform_processes_all_csvs_and_writes_quality_report() -> None:
    bronze_prefix = "csgo-matchmaking-damage/20260306T023831Z/extracted"
    initial_objects = {
        "bronze": {
            f"{bronze_prefix}/mm_master_demos.csv": (
                b"Damage,Tick,Player,Notes\n"
                b"10,1, Alice , ok \n"
                b"10,1,Alice,ok\n"
                b"-5,2,Bob,oops\n"
                b"abc,4,Eve,err\n"
                b",,,\n"
            ),
            f"{bronze_prefix}/nested/mm_grenades_demos.csv": (
                b"Round Num,Flash Duration,Money\n1,1.5,800\n1,1.5,800\n2,-1,1000\n"
            ),
            f"{bronze_prefix}/maps/de_inferno.png": b"png",
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
    assert result.files_processed == 2
    assert result.rows_read == 8
    assert result.rows_output == 2
    assert result.artifact_prefix == "csgo-matchmaking-damage/20260306T023831Z/cleaned/"
    assert result.quality_summary == {
        "files_processed": 2,
        "rows_read": 8,
        "rows_output": 2,
        "rows_removed": 6,
    }
    assert result.quality_report_key in result.uploaded_objects

    silver_objects = fake_minio.objects["silver"]
    cleaned_key_1 = "csgo-matchmaking-damage/20260306T023831Z/cleaned/mm_master_demos.csv"
    cleaned_key_2 = "csgo-matchmaking-damage/20260306T023831Z/cleaned/nested/mm_grenades_demos.csv"
    report_key = "csgo-matchmaking-damage/20260306T023831Z/quality_report.json"

    assert cleaned_key_1 in silver_objects
    assert cleaned_key_2 in silver_objects
    assert report_key in silver_objects

    assert (
        silver_objects[cleaned_key_1].decode("utf-8").replace("\r\n", "\n")
        == "damage,tick,player,notes\n10,1,Alice,ok\n"
    )
    assert (
        silver_objects[cleaned_key_2].decode("utf-8").replace("\r\n", "\n")
        == "round_num,flash_duration,money\n1,1.5,800\n"
    )

    report = json.loads(silver_objects[report_key].decode("utf-8"))
    assert report["artifact_prefix"] == result.artifact_prefix
    assert report["summary"] == {
        "files_processed": 2,
        "rows_read": 8,
        "rows_output": 2,
        "rows_removed": 6,
    }


def test_run_silver_transform_fails_when_no_csv_for_run_id() -> None:
    bronze_prefix = "csgo-matchmaking-damage/20260306T023831Z/extracted"
    fake_minio = FakeMinio(
        "localhost:9000",
        "minioadmin",
        "minioadmin",
        False,
        initial_objects={"bronze": {f"{bronze_prefix}/maps/de_inferno.png": b"png"}},
        existing_buckets={"bronze"},
    )

    with pytest.raises(FileNotFoundError, match="No CSV files were found in Bronze"):
        run_silver_transform(build_settings(), minio_factory=lambda **kwargs: fake_minio)
