from __future__ import annotations

import logging

from dotenv import load_dotenv

from rag_intelligence.config import ConfigError, SilverSettings
from rag_intelligence.logging import setup_logging
from rag_intelligence.silver import run_silver_transform


def main() -> int:
    load_dotenv()
    setup_logging()

    try:
        settings = SilverSettings.from_env()
        result = run_silver_transform(settings)
    except ConfigError as exc:
        logging.error("%s", exc)
        return 2
    except Exception as exc:
        logging.exception("Silver transform failed: %s", exc)
        return 1

    logging.info(
        "Silver transform finished successfully: files=%s rows_in=%s rows_out=%s report=%s",
        result.files_processed,
        result.rows_read,
        result.rows_output,
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
                run_id=settings.silver_run_id,
                stage="silver",
                dataset_prefix=settings.silver_dataset_prefix,
                bucket=settings.silver_bucket,
                source_run_id=settings.bronze_source_run_id,
                quality_report_key=result.quality_report_key,
                files_processed=result.files_processed,
                rows_read=result.rows_read,
                rows_output=result.rows_output,
            ),
        )
        logging.info("Registered silver run %s in metadata", settings.silver_run_id)
    except Exception as exc:
        logging.warning("Failed to register metadata (non-fatal): %s", exc)

    return 0
