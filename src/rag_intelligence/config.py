from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


class ConfigError(ValueError):
    """Raised when required runtime configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_bucket: str
    minio_secure: bool
    dataset_slug: str
    dataset_prefix: str
    run_id: str

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Settings:
        raw_env = dict(os.environ if env is None else env)

        minio_endpoint = require_env(raw_env, "MINIO_ENDPOINT")
        minio_access_key = require_env(raw_env, "MINIO_ACCESS_KEY")
        minio_secret_key = require_env(raw_env, "MINIO_SECRET_KEY")
        minio_bucket = require_env(raw_env, "MINIO_BUCKET")
        dataset_slug = require_env(raw_env, "BRONZE_DATASET_SLUG")
        dataset_prefix = require_env(raw_env, "BRONZE_DATASET_PREFIX")

        if not kaggle_credentials_present(raw_env):
            raise ConfigError(
                "Kaggle credentials are missing. Set KAGGLE_USERNAME/KAGGLE_KEY "
                "or provide ~/.kaggle/kaggle.json."
            )

        return cls(
            minio_endpoint=minio_endpoint,
            minio_access_key=minio_access_key,
            minio_secret_key=minio_secret_key,
            minio_bucket=minio_bucket,
            minio_secure=parse_bool(raw_env.get("MINIO_SECURE", "false")),
            dataset_slug=dataset_slug,
            dataset_prefix=dataset_prefix,
            run_id=raw_env.get("BRONZE_RUN_ID") or default_run_id(),
        )


@dataclass(frozen=True)
class SilverSettings:
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_secure: bool
    bronze_bucket: str
    silver_bucket: str
    bronze_dataset_prefix: str
    silver_dataset_prefix: str
    bronze_source_run_id: str
    silver_run_id: str

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> SilverSettings:
        raw_env = dict(os.environ if env is None else env)

        minio_endpoint = require_env(raw_env, "MINIO_ENDPOINT")
        minio_access_key = require_env(raw_env, "MINIO_ACCESS_KEY")
        minio_secret_key = require_env(raw_env, "MINIO_SECRET_KEY")
        bronze_bucket = require_env(raw_env, "MINIO_BUCKET")
        bronze_dataset_prefix = require_env(raw_env, "BRONZE_DATASET_PREFIX")
        bronze_source_run_id = require_env(raw_env, "BRONZE_SOURCE_RUN_ID")

        silver_bucket = raw_env.get("SILVER_BUCKET", "silver").strip() or "silver"
        silver_dataset_prefix = (
            raw_env.get("SILVER_DATASET_PREFIX", "").strip() or bronze_dataset_prefix
        )
        silver_run_id = raw_env.get("SILVER_RUN_ID", "").strip() or bronze_source_run_id

        return cls(
            minio_endpoint=minio_endpoint,
            minio_access_key=minio_access_key,
            minio_secret_key=minio_secret_key,
            minio_secure=parse_bool(raw_env.get("MINIO_SECURE", "false")),
            bronze_bucket=bronze_bucket,
            silver_bucket=silver_bucket,
            bronze_dataset_prefix=bronze_dataset_prefix,
            silver_dataset_prefix=silver_dataset_prefix,
            bronze_source_run_id=bronze_source_run_id,
            silver_run_id=silver_run_id,
        )


def require_env(env: dict[str, str], key: str) -> str:
    value = env.get(key, "").strip()
    if not value:
        raise ConfigError(f"Missing required environment variable: {key}")
    return value


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"Invalid boolean value for MINIO_SECURE: {value}")


def default_run_id(now: datetime | None = None) -> str:
    current = now or datetime.now(UTC)
    return current.strftime("%Y%m%dT%H%M%SZ")


def kaggle_credentials_present(env: dict[str, str]) -> bool:
    username = env.get("KAGGLE_USERNAME", "").strip()
    key = env.get("KAGGLE_KEY", "").strip()
    if username and key:
        return True

    config_dir = env.get("KAGGLE_CONFIG_DIR", "").strip()
    if config_dir:
        return Path(config_dir, "kaggle.json").is_file()

    return Path.home().joinpath(".kaggle", "kaggle.json").is_file()
