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
    except Exception as exc:  # pragma: no cover - integration failure path
        logging.exception("Bronze import failed: %s", exc)
        return 1

    logging.info(
        "Bronze import finished successfully with %s uploaded objects.",
        len(uploaded_objects),
    )
    return 0
