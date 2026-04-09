from __future__ import annotations

import logging

from dotenv import load_dotenv

from rag_intelligence.config import ConfigError, GoldSettings
from rag_intelligence.gold import run_gold_transform
from rag_intelligence.logging import setup_logging


def main() -> int:
    load_dotenv()
    setup_logging()

    try:
        settings = GoldSettings.from_env()
        result = run_gold_transform(settings)
    except ConfigError as exc:
        logging.error("%s", exc)
        return 2
    except Exception as exc:
        logging.exception("Gold transform failed: %s", exc)
        return 1

    logging.info(
        (
            "Gold transform finished successfully: "
            "files=%s rows_in=%s rows_out=%s round_context=%s report=%s"
        ),
        result.files_processed,
        result.rows_read,
        result.rows_output,
        result.events_key,
        result.quality_report_key,
    )

    try:
        from rag_intelligence.metadata import (
            MetadataSettings,
            RunRecord,
            ensure_schema,
            register_run,
        )

        md_settings = MetadataSettings.from_env()
        ensure_schema(md_settings)
        register_run(
            md_settings,
            RunRecord(
                run_id=settings.gold_run_id,
                stage="gold",
                dataset_prefix=settings.gold_dataset_prefix,
                bucket=settings.gold_bucket,
                source_run_id=settings.silver_source_run_id,
                events_key=result.events_key,
                artifact_prefix=result.artifact_prefix,
                quality_report_key=result.quality_report_key,
                files_processed=result.files_processed,
                rows_read=result.rows_read,
                rows_output=result.rows_output,
                quality_summary=result.quality_summary,
            ),
        )
        logging.info("Registered gold run %s in metadata", settings.gold_run_id)
    except Exception as exc:
        logging.warning("Failed to register metadata (non-fatal): %s", exc)

    return 0
