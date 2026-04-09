from __future__ import annotations

import argparse
import logging
import os
from io import BytesIO
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from minio import Minio

from rag_intelligence.gold import build_gold_events_key
from rag_intelligence.round_winner import (
    build_round_level_frame,
    build_supervised_frame,
    log_training_to_mlflow,
    run_consistency_checks,
    train_next_round_winner,
)


LOGGER = logging.getLogger(__name__)


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def load_events_from_minio() -> pd.DataFrame:
    dataset_prefix = (
        os.getenv("GOLD_DATASET_PREFIX")
        or os.getenv("SILVER_DATASET_PREFIX")
        or os.getenv("BRONZE_DATASET_PREFIX")
    )
    run_id = os.getenv("GOLD_SOURCE_RUN_ID") or os.getenv("GOLD_RUN_ID")
    if not dataset_prefix or not run_id:
        raise ValueError(
            "Missing dataset prefix or run id. Set GOLD_DATASET_PREFIX (or SILVER/BRONZE fallback) "
            "and GOLD_SOURCE_RUN_ID (or GOLD_RUN_ID)."
        )

    minio_client = Minio(
        endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        secure=parse_bool(os.getenv("MINIO_SECURE", "false")),
    )
    gold_bucket = os.getenv("GOLD_BUCKET", "gold")
    events_key = build_gold_events_key(dataset_prefix, run_id)

    LOGGER.info("Loading events from MinIO bucket=%s key=%s", gold_bucket, events_key)
    response = minio_client.get_object(gold_bucket, events_key)
    try:
        return pd.read_csv(BytesIO(response.read()), low_memory=False)
    finally:
        response.close()
        response.release_conn()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train next-round winner-side model (predict round t+1 from round t context).",
    )
    parser.add_argument(
        "--events-csv",
        help="Optional local path to events.csv. If omitted, load from MinIO using .env settings.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional limit for rows loaded from events.csv (0 = all rows).",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--segment-min-rows", type=int, default=200)
    parser.add_argument("--disable-mlflow", action="store_true")
    parser.add_argument("--tracking-uri", default="", help="MLflow tracking URI.")
    parser.add_argument("--experiment-name", default="csgo_round_next_winner")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.events_csv:
            events_path = Path(args.events_csv)
            LOGGER.info("Loading local events CSV: %s", events_path)
            events_df = pd.read_csv(events_path, low_memory=False)
        else:
            events_df = load_events_from_minio()

        if args.max_rows and args.max_rows > 0:
            events_df = events_df.head(args.max_rows).copy()

        round_df = build_round_level_frame(events_df)
        supervised_df = build_supervised_frame(round_df, lag_steps=(1, 2, 3), rolling_window=3)
        checks = run_consistency_checks(supervised_df)

        LOGGER.info("Round-level rows: %s", len(round_df))
        LOGGER.info("Supervised rows: %s", len(supervised_df))
        LOGGER.info("Checks: %s", checks)

        if checks["duplicate_round_keys"] != 0:
            raise ValueError("Consistency check failed: duplicate round keys were found.")
        if not checks["target_shift_is_valid"]:
            raise ValueError("Consistency check failed: target shift validation failed.")
        if not checks["lag1_is_past_only"]:
            raise ValueError("Consistency check failed: lag leakage was detected.")

        training_output = train_next_round_winner(
            supervised_df,
            test_size=args.test_size,
            random_state=args.random_state,
            min_segment_rows=args.segment_min_rows,
        )
        LOGGER.info("Train rows=%s Test rows=%s", training_output.train_rows, training_output.test_rows)
        LOGGER.info("Model metrics:\n%s", training_output.metrics.to_string(index=False))

        best_model = training_output.metrics.iloc[0]["model"]
        LOGGER.info("Best model by ROC-AUC: %s", best_model)

        LOGGER.info(
            "Map segment metrics (%s):\n%s",
            best_model,
            training_output.map_segment_metrics[best_model].to_string(index=False),
        )
        LOGGER.info(
            "Half segment metrics (%s):\n%s",
            best_model,
            training_output.half_segment_metrics[best_model].to_string(index=False),
        )

        if not args.disable_mlflow:
            default_tracking = f"file:{(Path.cwd() / 'mlruns').as_posix()}"
            tracking_uri = args.tracking_uri or os.getenv("MLFLOW_TRACKING_URI", default_tracking)
            log_training_to_mlflow(
                training_output=training_output,
                experiment_name=args.experiment_name,
                tracking_uri=tracking_uri,
                target_name="winner_side_next_round",
                horizon_rounds=1,
                test_size=args.test_size,
                random_state=args.random_state,
            )
            LOGGER.info("MLflow logging finished. tracking_uri=%s", tracking_uri)
        return 0
    except Exception as exc:
        LOGGER.exception("Round winner training failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
