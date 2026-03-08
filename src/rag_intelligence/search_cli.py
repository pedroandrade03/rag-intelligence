from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence

from dotenv import load_dotenv

from rag_intelligence.logging import setup_logging
from rag_intelligence.retrieval import SearchRequest, search_events


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="semantic-search")
    parser.add_argument("--query", required=True, type=_non_empty_text)
    parser.add_argument("--embedding-run-id", required=True, type=_non_empty_text)
    parser.add_argument("--top-k", default=5, type=_positive_int)
    parser.add_argument("--event-type", type=_non_empty_text)
    parser.add_argument("--map", dest="map_name", type=_non_empty_text)
    parser.add_argument("--file", dest="file_name", type=_non_empty_text)
    parser.add_argument("--round", dest="round_number", type=_integer_value)
    return parser


def build_search_request(argv: Sequence[str] | None = None) -> SearchRequest:
    args = build_parser().parse_args(argv)
    return SearchRequest(
        query=args.query,
        embedding_run_id=args.embedding_run_id,
        top_k=args.top_k,
        event_type=args.event_type,
        map_name=args.map_name,
        file_name=args.file_name,
        round_number=args.round_number,
    )


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    setup_logging()

    try:
        request = build_search_request(argv)
        response = search_events(request)
    except SystemExit as exc:
        return int(exc.code or 0)
    except ValueError as exc:
        logging.error("%s", exc)
        return 2
    except Exception as exc:
        logging.exception("Semantic search failed: %s", exc)
        return 1

    print(json.dumps(response.to_dict(), indent=2, ensure_ascii=True))
    return 0


def _non_empty_text(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise argparse.ArgumentTypeError("value cannot be empty")
    return cleaned


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _integer_value(value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
