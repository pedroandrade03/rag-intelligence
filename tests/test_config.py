from __future__ import annotations

import re

import pytest

from rag_intelligence.config import (
    ConfigError,
    DocumentSettings,
    GoldSettings,
    Settings,
    SilverSettings,
    default_run_id,
)
from rag_intelligence.ingest import build_object_key


def base_env() -> dict[str, str]:
    return {
        "MINIO_ENDPOINT": "localhost:9000",
        "MINIO_ACCESS_KEY": "minioadmin",
        "MINIO_SECRET_KEY": "minioadmin",
        "MINIO_BUCKET": "bronze",
        "MINIO_SECURE": "false",
        "KAGGLE_USERNAME": "user",
        "KAGGLE_KEY": "secret",
        "BRONZE_DATASET_SLUG": "skihikingkevin/csgo-matchmaking-damage",
        "BRONZE_DATASET_PREFIX": "csgo-matchmaking-damage",
    }


def test_settings_require_missing_env_var() -> None:
    env = base_env()
    env["MINIO_BUCKET"] = ""

    with pytest.raises(ConfigError, match="MINIO_BUCKET"):
        Settings.from_env(env)


def test_settings_require_kaggle_credentials(tmp_path) -> None:
    env = base_env()
    env["KAGGLE_USERNAME"] = ""
    env["KAGGLE_KEY"] = ""
    env["KAGGLE_CONFIG_DIR"] = str(tmp_path)

    with pytest.raises(ConfigError, match="Kaggle credentials are missing"):
        Settings.from_env(env)


def test_default_run_id_uses_versioned_timestamp_format() -> None:
    run_id = default_run_id()
    assert re.fullmatch(r"\d{8}T\d{6}Z", run_id)


def test_build_object_key_uses_versioned_prefix() -> None:
    assert (
        build_object_key(
            "csgo-matchmaking-damage",
            "20260306T010203Z",
            "mm_master_demos.csv",
            section="extracted",
        )
        == "csgo-matchmaking-damage/20260306T010203Z/extracted/mm_master_demos.csv"
    )


def test_silver_settings_require_bronze_source_run_id() -> None:
    env = base_env()
    env["BRONZE_SOURCE_RUN_ID"] = ""

    with pytest.raises(ConfigError, match="BRONZE_SOURCE_RUN_ID"):
        SilverSettings.from_env(env)


def test_silver_settings_use_expected_defaults() -> None:
    env = base_env()
    env["BRONZE_SOURCE_RUN_ID"] = "20260306T023831Z"

    settings = SilverSettings.from_env(env)

    assert settings.bronze_bucket == "bronze"
    assert settings.silver_bucket == "silver"
    assert settings.bronze_dataset_prefix == "csgo-matchmaking-damage"
    assert settings.silver_dataset_prefix == "csgo-matchmaking-damage"
    assert settings.bronze_source_run_id == "20260306T023831Z"
    assert settings.silver_run_id == "20260306T023831Z"


def test_gold_settings_require_silver_source_run_id() -> None:
    env = base_env()
    env["SILVER_SOURCE_RUN_ID"] = ""

    with pytest.raises(ConfigError, match="SILVER_SOURCE_RUN_ID"):
        GoldSettings.from_env(env)


def test_gold_settings_use_expected_defaults() -> None:
    env = base_env()
    env["SILVER_SOURCE_RUN_ID"] = "20260306T023831Z"

    settings = GoldSettings.from_env(env)

    assert settings.silver_bucket == "silver"
    assert settings.gold_bucket == "gold"
    assert settings.silver_dataset_prefix == "csgo-matchmaking-damage"
    assert settings.gold_dataset_prefix == "csgo-matchmaking-damage"
    assert settings.silver_source_run_id == "20260306T023831Z"
    assert settings.gold_run_id == "20260306T023831Z"


def test_gold_settings_require_dataset_prefix_when_fallbacks_are_missing() -> None:
    env = base_env()
    env["SILVER_SOURCE_RUN_ID"] = "20260306T023831Z"
    env["BRONZE_DATASET_PREFIX"] = ""
    env["SILVER_DATASET_PREFIX"] = ""

    with pytest.raises(ConfigError, match="SILVER_DATASET_PREFIX"):
        GoldSettings.from_env(env)


def test_document_settings_require_gold_source_run_id() -> None:
    env = base_env()
    env["GOLD_SOURCE_RUN_ID"] = ""

    with pytest.raises(ConfigError, match="GOLD_SOURCE_RUN_ID"):
        DocumentSettings.from_env(env)


def test_document_settings_use_expected_defaults() -> None:
    env = base_env()
    env["GOLD_SOURCE_RUN_ID"] = "20260306T025119Z"

    settings = DocumentSettings.from_env(env)

    assert settings.gold_bucket == "gold"
    assert settings.document_bucket == "gold"
    assert settings.gold_dataset_prefix == "csgo-matchmaking-damage"
    assert settings.document_dataset_prefix == "csgo-matchmaking-damage"
    assert settings.gold_source_run_id == "20260306T025119Z"
    assert settings.document_run_id == "20260306T025119Z"
    assert settings.document_part_size_rows == 100000


def test_document_settings_support_overrides() -> None:
    env = base_env()
    env["GOLD_SOURCE_RUN_ID"] = "20260306T025119Z"
    env["GOLD_BUCKET"] = "gold"
    env["GOLD_DATASET_PREFIX"] = "gold-prefix"
    env["DOCUMENT_BUCKET"] = "documents"
    env["DOCUMENT_DATASET_PREFIX"] = "doc-prefix"
    env["DOCUMENT_RUN_ID"] = "20260308T120000Z"
    env["DOCUMENT_PART_SIZE_ROWS"] = "2500"

    settings = DocumentSettings.from_env(env)

    assert settings.document_bucket == "documents"
    assert settings.document_dataset_prefix == "doc-prefix"
    assert settings.document_run_id == "20260308T120000Z"
    assert settings.document_part_size_rows == 2500


def test_document_settings_require_positive_part_size() -> None:
    env = base_env()
    env["GOLD_SOURCE_RUN_ID"] = "20260306T025119Z"
    env["DOCUMENT_PART_SIZE_ROWS"] = "0"

    with pytest.raises(ConfigError, match="greater than zero"):
        DocumentSettings.from_env(env)
