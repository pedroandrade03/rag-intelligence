from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence

from dotenv import load_dotenv

from rag_intelligence.logging import setup_logging
from rag_intelligence.metadata import (
    LineageAuditError,
    MetadataSettings,
    RunLineageReport,
    get_run_lineage,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run-audit")
    parser.add_argument(
        "--stage", required=True, choices=("bronze", "silver", "gold", "documents", "embeddings")
    )
    parser.add_argument("--run-id", required=True, type=_non_empty_text)
    parser.add_argument("--format", dest="output_format", default="text", choices=("text", "json"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    setup_logging()

    try:
        args = build_parser().parse_args(argv)
        md_settings = MetadataSettings.from_env()
        report = get_run_lineage(md_settings, stage=args.stage, run_id=args.run_id)
    except SystemExit as exc:
        return int(exc.code or 0)
    except (LineageAuditError, ValueError) as exc:
        logging.error("%s", exc)
        return 2
    except Exception as exc:
        logging.exception("Run audit failed: %s", exc)
        return 1

    if args.output_format == "json":
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=True, sort_keys=True))
    else:
        print(_format_text_report(report))
    return 0


def _format_text_report(report: RunLineageReport) -> str:
    lines = [
        f"Run audit for {report.requested_stage}:{report.requested_run_id}",
        f"Integrity: {report.integrity_status}",
        f"Chain length: {len(report.resolved_chain)}",
        "",
    ]
    for index, node in enumerate(report.resolved_chain, start=1):
        evidence = node.evidence
        lines.extend(
            [
                f"{index}. {node.stage} run={node.run_id}",
                f"   dataset_prefix={node.dataset_prefix} bucket={evidence.bucket}",
                f"   source_run_id={evidence.source_run_id or '-'}",
                (
                    "   artifacts="
                    f"events={evidence.events_key or '-'} "
                    f"prefix={evidence.artifact_prefix or '-'} "
                    f"manifest={evidence.manifest_key or '-'} "
                    f"report={evidence.quality_report_key or '-'}"
                ),
                (
                    "   counters="
                    f"files={evidence.files_processed} "
                    f"rows_read={evidence.rows_read} "
                    f"rows_output={evidence.rows_output}"
                ),
            ]
        )
    return "\n".join(lines)


def _non_empty_text(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise argparse.ArgumentTypeError("value cannot be empty")
    return cleaned
