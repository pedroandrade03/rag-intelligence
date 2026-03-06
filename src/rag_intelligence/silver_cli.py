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
    return 0
