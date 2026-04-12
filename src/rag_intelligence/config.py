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


@dataclass(frozen=True)
class GoldSettings:
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_secure: bool
    silver_bucket: str
    gold_bucket: str
    silver_dataset_prefix: str
    gold_dataset_prefix: str
    silver_source_run_id: str
    gold_run_id: str

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> GoldSettings:
        raw_env = dict(os.environ if env is None else env)

        minio_endpoint = require_env(raw_env, "MINIO_ENDPOINT")
        minio_access_key = require_env(raw_env, "MINIO_ACCESS_KEY")
        minio_secret_key = require_env(raw_env, "MINIO_SECRET_KEY")
        silver_source_run_id = require_env(raw_env, "SILVER_SOURCE_RUN_ID")

        bronze_dataset_prefix = raw_env.get("BRONZE_DATASET_PREFIX", "").strip()
        silver_dataset_prefix = (
            raw_env.get("SILVER_DATASET_PREFIX", "").strip() or bronze_dataset_prefix
        )
        if not silver_dataset_prefix:
            raise ConfigError(
                "Missing required environment variable: SILVER_DATASET_PREFIX "
                "(or BRONZE_DATASET_PREFIX as fallback)"
            )

        gold_dataset_prefix = (
            raw_env.get("GOLD_DATASET_PREFIX", "").strip() or silver_dataset_prefix
        )
        silver_bucket = raw_env.get("SILVER_BUCKET", "silver").strip() or "silver"
        gold_bucket = raw_env.get("GOLD_BUCKET", "gold").strip() or "gold"
        gold_run_id = raw_env.get("GOLD_RUN_ID", "").strip() or silver_source_run_id

        return cls(
            minio_endpoint=minio_endpoint,
            minio_access_key=minio_access_key,
            minio_secret_key=minio_secret_key,
            minio_secure=parse_bool(raw_env.get("MINIO_SECURE", "false")),
            silver_bucket=silver_bucket,
            gold_bucket=gold_bucket,
            silver_dataset_prefix=silver_dataset_prefix,
            gold_dataset_prefix=gold_dataset_prefix,
            silver_source_run_id=silver_source_run_id,
            gold_run_id=gold_run_id,
        )


@dataclass(frozen=True)
class DocumentSettings:
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_secure: bool
    gold_bucket: str
    document_bucket: str
    gold_dataset_prefix: str
    document_dataset_prefix: str
    gold_source_run_id: str
    document_run_id: str
    document_part_size_rows: int
    document_max_rows: int | None

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> DocumentSettings:
        raw_env = dict(os.environ if env is None else env)

        minio_endpoint = require_env(raw_env, "MINIO_ENDPOINT")
        minio_access_key = require_env(raw_env, "MINIO_ACCESS_KEY")
        minio_secret_key = require_env(raw_env, "MINIO_SECRET_KEY")
        gold_source_run_id = require_env(raw_env, "GOLD_SOURCE_RUN_ID")

        bronze_dataset_prefix = raw_env.get("BRONZE_DATASET_PREFIX", "").strip()
        silver_dataset_prefix = (
            raw_env.get("SILVER_DATASET_PREFIX", "").strip() or bronze_dataset_prefix
        )
        gold_dataset_prefix = (
            raw_env.get("GOLD_DATASET_PREFIX", "").strip() or silver_dataset_prefix
        )
        if not gold_dataset_prefix:
            raise ConfigError(
                "Missing required environment variable: GOLD_DATASET_PREFIX "
                "(or SILVER_DATASET_PREFIX / BRONZE_DATASET_PREFIX as fallback)"
            )

        document_dataset_prefix = (
            raw_env.get("DOCUMENT_DATASET_PREFIX", "").strip() or gold_dataset_prefix
        )
        gold_bucket = raw_env.get("GOLD_BUCKET", "gold").strip() or "gold"
        document_bucket = raw_env.get("DOCUMENT_BUCKET", "").strip() or gold_bucket
        document_run_id = raw_env.get("DOCUMENT_RUN_ID", "").strip() or gold_source_run_id

        part_size_raw = raw_env.get("DOCUMENT_PART_SIZE_ROWS", "100000").strip() or "100000"
        try:
            document_part_size_rows = int(part_size_raw)
        except ValueError as err:
            raise ConfigError(f"Invalid DOCUMENT_PART_SIZE_ROWS value: {part_size_raw}") from err
        if document_part_size_rows <= 0:
            raise ConfigError("DOCUMENT_PART_SIZE_ROWS must be greater than zero")
        document_max_rows = parse_optional_positive_int(raw_env, "DOCUMENT_MAX_ROWS")

        return cls(
            minio_endpoint=minio_endpoint,
            minio_access_key=minio_access_key,
            minio_secret_key=minio_secret_key,
            minio_secure=parse_bool(raw_env.get("MINIO_SECURE", "false")),
            gold_bucket=gold_bucket,
            document_bucket=document_bucket,
            gold_dataset_prefix=gold_dataset_prefix,
            document_dataset_prefix=document_dataset_prefix,
            gold_source_run_id=gold_source_run_id,
            document_run_id=document_run_id,
            document_part_size_rows=document_part_size_rows,
            document_max_rows=document_max_rows,
        )


@dataclass(frozen=True)
class EmbeddingSettings:
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_secure: bool
    document_bucket: str
    embedding_report_bucket: str
    document_dataset_prefix: str
    embedding_dataset_prefix: str
    document_source_run_id: str
    embedding_run_id: str
    embedding_batch_size: int
    embedding_num_workers: int
    embedding_parallel_batches: int
    embedding_max_documents: int | None
    embedding_resume: bool

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> EmbeddingSettings:
        raw_env = dict(os.environ if env is None else env)

        minio_endpoint = require_env(raw_env, "MINIO_ENDPOINT")
        minio_access_key = require_env(raw_env, "MINIO_ACCESS_KEY")
        minio_secret_key = require_env(raw_env, "MINIO_SECRET_KEY")
        document_source_run_id = require_env(raw_env, "DOCUMENT_SOURCE_RUN_ID")

        bronze_dataset_prefix = raw_env.get("BRONZE_DATASET_PREFIX", "").strip()
        silver_dataset_prefix = (
            raw_env.get("SILVER_DATASET_PREFIX", "").strip() or bronze_dataset_prefix
        )
        gold_dataset_prefix = (
            raw_env.get("GOLD_DATASET_PREFIX", "").strip() or silver_dataset_prefix
        )
        document_dataset_prefix = (
            raw_env.get("DOCUMENT_DATASET_PREFIX", "").strip() or gold_dataset_prefix
        )
        if not document_dataset_prefix:
            raise ConfigError(
                "Missing required environment variable: DOCUMENT_DATASET_PREFIX "
                "(or GOLD_DATASET_PREFIX / SILVER_DATASET_PREFIX / "
                "BRONZE_DATASET_PREFIX as fallback)"
            )

        gold_bucket = raw_env.get("GOLD_BUCKET", "gold").strip() or "gold"
        document_bucket = raw_env.get("DOCUMENT_BUCKET", "").strip() or gold_bucket
        embedding_report_bucket = (
            raw_env.get("EMBEDDING_REPORT_BUCKET", "").strip() or document_bucket
        )
        embedding_dataset_prefix = (
            raw_env.get("EMBEDDING_DATASET_PREFIX", "").strip() or document_dataset_prefix
        )
        embedding_run_id = raw_env.get("EMBEDDING_RUN_ID", "").strip() or document_source_run_id

        batch_size_raw = raw_env.get("EMBEDDING_BATCH_SIZE", "256").strip() or "256"
        try:
            embedding_batch_size = int(batch_size_raw)
        except ValueError as err:
            raise ConfigError(f"Invalid EMBEDDING_BATCH_SIZE value: {batch_size_raw}") from err
        if embedding_batch_size <= 0:
            raise ConfigError("EMBEDDING_BATCH_SIZE must be greater than zero")
        num_workers_raw = raw_env.get("EMBEDDING_NUM_WORKERS", "4").strip() or "4"
        try:
            embedding_num_workers = int(num_workers_raw)
        except ValueError as err:
            raise ConfigError(f"Invalid EMBEDDING_NUM_WORKERS value: {num_workers_raw}") from err
        if embedding_num_workers <= 0:
            raise ConfigError("EMBEDDING_NUM_WORKERS must be greater than zero")
        parallel_batches_raw = raw_env.get("EMBEDDING_PARALLEL_BATCHES", "4").strip() or "4"
        try:
            embedding_parallel_batches = int(parallel_batches_raw)
        except ValueError as err:
            raise ConfigError(
                f"Invalid EMBEDDING_PARALLEL_BATCHES value: {parallel_batches_raw}"
            ) from err
        if embedding_parallel_batches <= 0:
            raise ConfigError("EMBEDDING_PARALLEL_BATCHES must be greater than zero")
        embedding_max_documents = parse_optional_positive_int(
            raw_env,
            "EMBEDDING_MAX_DOCUMENTS",
        )
        try:
            embedding_resume = parse_bool(raw_env.get("EMBEDDING_RESUME", "false"))
        except ConfigError as err:
            raw_value = raw_env.get("EMBEDDING_RESUME", "false")
            raise ConfigError(f"Invalid boolean value for EMBEDDING_RESUME: {raw_value}") from err

        return cls(
            minio_endpoint=minio_endpoint,
            minio_access_key=minio_access_key,
            minio_secret_key=minio_secret_key,
            minio_secure=parse_bool(raw_env.get("MINIO_SECURE", "false")),
            document_bucket=document_bucket,
            embedding_report_bucket=embedding_report_bucket,
            document_dataset_prefix=document_dataset_prefix,
            embedding_dataset_prefix=embedding_dataset_prefix,
            document_source_run_id=document_source_run_id,
            embedding_run_id=embedding_run_id,
            embedding_batch_size=embedding_batch_size,
            embedding_num_workers=embedding_num_workers,
            embedding_parallel_batches=embedding_parallel_batches,
            embedding_max_documents=embedding_max_documents,
            embedding_resume=embedding_resume,
        )


@dataclass(frozen=True)
class TrainSettings:
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_secure: bool
    gold_bucket: str
    gold_dataset_prefix: str
    gold_source_run_id: str
    mlflow_tracking_uri: str
    experiment_name: str
    train_run_id: str
    test_size: float
    random_state: int
    model_name: str

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> TrainSettings:
        raw_env = dict(os.environ if env is None else env)

        minio_endpoint = require_env(raw_env, "MINIO_ENDPOINT")
        minio_access_key = require_env(raw_env, "MINIO_ACCESS_KEY")
        minio_secret_key = require_env(raw_env, "MINIO_SECRET_KEY")

        gold_dataset_prefix = (
            raw_env.get("GOLD_DATASET_PREFIX", "").strip()
            or raw_env.get("SILVER_DATASET_PREFIX", "").strip()
            or raw_env.get("BRONZE_DATASET_PREFIX", "").strip()
        )
        if not gold_dataset_prefix:
            raise ConfigError(
                "Missing required environment variable: GOLD_DATASET_PREFIX "
                "(or SILVER_DATASET_PREFIX / BRONZE_DATASET_PREFIX as fallback)"
            )

        gold_source_run_id = require_env(raw_env, "GOLD_SOURCE_RUN_ID")
        test_size_raw = raw_env.get("TRAIN_TEST_SIZE", "0.2").strip() or "0.2"
        random_state_raw = raw_env.get("TRAIN_RANDOM_STATE", "42").strip() or "42"
        try:
            test_size = float(test_size_raw)
        except ValueError as err:
            raise ConfigError(f"Invalid TRAIN_TEST_SIZE value: {test_size_raw}") from err
        if not 0 < test_size < 1:
            raise ConfigError("TRAIN_TEST_SIZE must be between 0 and 1")
        try:
            random_state = int(random_state_raw)
        except ValueError as err:
            raise ConfigError(f"Invalid TRAIN_RANDOM_STATE value: {random_state_raw}") from err

        return cls(
            minio_endpoint=minio_endpoint,
            minio_access_key=minio_access_key,
            minio_secret_key=minio_secret_key,
            minio_secure=parse_bool(raw_env.get("MINIO_SECURE", "false")),
            gold_bucket=raw_env.get("GOLD_BUCKET", "gold").strip() or "gold",
            gold_dataset_prefix=gold_dataset_prefix,
            gold_source_run_id=gold_source_run_id,
            mlflow_tracking_uri=raw_env.get(
                "MLFLOW_TRACKING_URI", "http://localhost:5000"
            ).strip(),
            experiment_name=raw_env.get(
                "MLFLOW_EXPERIMENT_NAME", "csgo_round_next_winner"
            ).strip(),
            train_run_id=raw_env.get("TRAIN_RUN_ID", "").strip() or default_run_id(),
            test_size=test_size,
            random_state=random_state,
            model_name=raw_env.get("TRAIN_MODEL_NAME", "").strip(),
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


def parse_optional_positive_int(
    env: dict[str, str],
    key: str,
) -> int | None:
    raw_value = env.get(key, "").strip()
    if not raw_value:
        return None
    try:
        parsed = int(raw_value)
    except ValueError as err:
        raise ConfigError(f"Invalid {key} value: {raw_value}") from err
    if parsed <= 0:
        raise ConfigError(f"{key} must be greater than zero")
    return parsed


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
