from __future__ import annotations

import logging

from dotenv import load_dotenv

from rag_intelligence.config import ConfigError, EmbeddingSettings
from rag_intelligence.embeddings import run_embedding_ingest
from rag_intelligence.logging import setup_logging


def main() -> int:
    load_dotenv()
    setup_logging()

    try:
        settings = EmbeddingSettings.from_env()
        result = run_embedding_ingest(settings)
    except ConfigError as exc:
        logging.error("%s", exc)
        return 2
    except Exception as exc:
        logging.exception("Embedding ingest failed: %s", exc)
        return 1

    logging.info(
        (
            "Embedding ingest finished successfully: "
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
                run_id=settings.embedding_run_id,
                stage="embeddings",
                dataset_prefix=settings.embedding_dataset_prefix,
                bucket=settings.embedding_report_bucket,
                source_run_id=settings.document_source_run_id,
                artifact_prefix=result.artifact_prefix,
                manifest_key=result.manifest_key,
                quality_report_key=result.quality_report_key,
                files_processed=result.files_processed,
                rows_read=result.rows_read,
                rows_output=result.rows_output,
                quality_summary=result.quality_summary,
            ),
        )
        logging.info("Registered embedding run %s in metadata", settings.embedding_run_id)
    except Exception as exc:
        logging.warning("Failed to register metadata (non-fatal): %s", exc)

    return 0
