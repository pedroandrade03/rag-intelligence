from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from rag_intelligence.config import SilverSettings
from rag_intelligence.silver import (
    build_silver_object_key,
    clean_csv_file,
    normalize_column_names,
    run_silver_transform,
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
