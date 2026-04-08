from __future__ import annotations

import logging

from dotenv import load_dotenv

from rag_intelligence.config import ConfigError, Settings
from rag_intelligence.ingest import run_import


def main() -> int:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    try:
        settings = Settings.from_env()
        uploaded_objects = run_import(settings)
    except ConfigError as exc:
        logging.error("%s", exc)
        return 2
    except Exception as exc:
        logging.exception("Bronze import failed: %s", exc)
        return 1

    logging.info(
        "Bronze import finished successfully with %s uploaded objects.",
        len(uploaded_objects),
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
        run_prefix = f"{settings.dataset_prefix.strip('/')}/{settings.run_id.strip('/')}/"
        raw_count = sum(1 for object_key in uploaded_objects if "/raw/" in object_key)
        extracted_count = sum(1 for object_key in uploaded_objects if "/extracted/" in object_key)
        register_run(
            md_settings,
            RunRecord(
                run_id=settings.run_id,
                stage="bronze",
                dataset_prefix=settings.dataset_prefix,
                bucket=settings.minio_bucket,
                artifact_prefix=run_prefix,
                files_processed=len(uploaded_objects),
                quality_summary={
                    "files_processed": len(uploaded_objects),
                    "raw_objects": raw_count,
                    "extracted_objects": extracted_count,
                },
            ),
        )
        logging.info("Registered bronze run %s in metadata", settings.run_id)
    except Exception as exc:
        logging.warning("Failed to register metadata (non-fatal): %s", exc)

    return 0
