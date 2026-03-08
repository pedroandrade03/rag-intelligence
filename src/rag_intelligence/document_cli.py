from __future__ import annotations

import logging

from dotenv import load_dotenv

from rag_intelligence.config import ConfigError, DocumentSettings
from rag_intelligence.documents import run_document_build
from rag_intelligence.logging import setup_logging


def main() -> int:
    load_dotenv()
    setup_logging()

    try:
        settings = DocumentSettings.from_env()
        result = run_document_build(settings)
    except ConfigError as exc:
        logging.error("%s", exc)
        return 2
    except Exception as exc:
        logging.exception("Document build failed: %s", exc)
        return 1

    logging.info(
        (
            "Document build finished successfully: "
            "parts=%s rows_in=%s rows_out=%s manifest=%s report=%s"
        ),
        result.files_processed,
        result.rows_read,
        result.rows_output,
        result.manifest_key,
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
                run_id=settings.document_run_id,
                stage="documents",
                dataset_prefix=settings.document_dataset_prefix,
                bucket=settings.document_bucket,
                source_run_id=settings.gold_source_run_id,
                artifact_prefix=result.artifact_prefix,
                manifest_key=result.manifest_key,
                quality_report_key=result.quality_report_key,
                files_processed=result.files_processed,
                rows_read=result.rows_read,
                rows_output=result.rows_output,
                quality_summary=result.quality_summary,
            ),
        )
        logging.info("Registered document run %s in metadata", settings.document_run_id)
    except Exception as exc:
        logging.warning("Failed to register metadata (non-fatal): %s", exc)

    return 0
