from __future__ import annotations

import logging

from dotenv import load_dotenv

from rag_intelligence.config import ConfigError, GoldSettings
from rag_intelligence.gold import run_gold_transform


def main() -> int:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

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
            "files=%s rows_in=%s rows_out=%s events=%s report=%s"
        ),
        result.files_processed,
        result.rows_read,
        result.rows_output,
        result.events_key,
        result.quality_report_key,
    )
    return 0
