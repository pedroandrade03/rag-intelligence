"""Unified ML training CLI — one entry point per model type."""

from __future__ import annotations

import logging
from io import BytesIO

import pandas as pd
from dotenv import load_dotenv
from minio import Minio

from rag_intelligence.config import ConfigError, TrainSettings
from rag_intelligence.gold import build_gold_events_key
from rag_intelligence.logging import setup_logging
from rag_intelligence.round_winner import (
    build_round_level_frame,
    build_supervised_frame,
    extract_feature_importances,
    log_training_to_mlflow,
    run_consistency_checks,
    train_next_round_winner,
)

LOGGER = logging.getLogger(__name__)


def _load_gold_events(settings: TrainSettings) -> pd.DataFrame:
    client = Minio(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )
    events_key = build_gold_events_key(settings.gold_dataset_prefix, settings.gold_source_run_id)
    LOGGER.info("Loading events from MinIO bucket=%s key=%s", settings.gold_bucket, events_key)
    response = client.get_object(settings.gold_bucket, events_key)
    try:
        return pd.read_csv(BytesIO(response.read()), low_memory=False)
    finally:
        response.close()
        response.release_conn()


def main(model: str = "") -> int:
    load_dotenv()
    setup_logging()

    try:
        settings = TrainSettings.from_env()
        model_filter = model or settings.model_name or None
    except ConfigError as exc:
        logging.error("%s", exc)
        return 2

    try:
        events_df = _load_gold_events(settings)
        round_df = build_round_level_frame(events_df)
        supervised_df = build_supervised_frame(round_df)
        checks = run_consistency_checks(supervised_df)

        LOGGER.info("Supervised rows: %s  Checks: %s", len(supervised_df), checks)
        if checks["duplicate_round_keys"] != 0:
            raise ValueError("Consistency check failed: duplicate round keys")
        if not checks["target_shift_is_valid"]:
            raise ValueError("Consistency check failed: target shift validation")
        if not checks["lag1_is_past_only"]:
            raise ValueError("Consistency check failed: lag leakage detected")

        training_output = train_next_round_winner(
            supervised_df,
            test_size=settings.test_size,
            random_state=settings.random_state,
            model_filter=model_filter,
        )
        LOGGER.info(
            "Training complete: train=%s test=%s",
            training_output.train_rows,
            training_output.test_rows,
        )
        LOGGER.info("Metrics:\n%s", training_output.metrics.to_string(index=False))

        feature_importances = extract_feature_importances(training_output)

        log_training_to_mlflow(
            training_output=training_output,
            experiment_name=settings.experiment_name,
            tracking_uri=settings.mlflow_tracking_uri,
            target_name="winner_side_next_round",
            horizon_rounds=1,
            test_size=settings.test_size,
            random_state=settings.random_state,
        )
        LOGGER.info("MLflow logging finished. uri=%s", settings.mlflow_tracking_uri)

        try:
            from rag_intelligence.training_metadata import (
                ensure_training_schema,
                store_training_result,
            )

            ensure_training_schema()
            store_training_result(
                training_output=training_output,
                feature_importances=feature_importances,
                experiment_name=settings.experiment_name,
                run_id=settings.train_run_id,
                test_size=settings.test_size,
                model_filter=model_filter,
            )
            LOGGER.info("Training metadata stored in PostgreSQL FTS")
        except Exception as exc:
            LOGGER.warning("Failed to store training metadata (non-fatal): %s", exc)

        return 0
    except Exception as exc:
        LOGGER.exception("Training failed: %s", exc)
        return 1


def main_logreg() -> int:
    return main(model="logistic_regression")


def main_histgbt() -> int:
    return main(model="hist_gradient_boosting")


def main_baseline() -> int:
    return main(model="baseline")


if __name__ == "__main__":
    raise SystemExit(main())
