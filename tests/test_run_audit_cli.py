from __future__ import annotations

import json
from datetime import UTC, datetime

from rag_intelligence.metadata import LineageNode, RunEvidence, RunLineageReport
from rag_intelligence.run_audit_cli import main


def _build_report() -> RunLineageReport:
    return RunLineageReport(
        requested_stage="embeddings",
        requested_run_id="embed-run",
        integrity_status="ok",
        integrity_issues=[],
        resolved_chain=[
            LineageNode(
                stage="embeddings",
                run_id="embed-run",
                dataset_prefix="csgo",
                status="completed",
                evidence=RunEvidence(
                    bucket="reports",
                    source_run_id="docs-run",
                    events_key=None,
                    artifact_prefix="csgo/embed-run/embeddings/",
                    manifest_key="csgo/embed-run/embeddings/manifest.json",
                    quality_report_key="csgo/embed-run/embeddings/quality_report.json",
                    files_processed=1,
                    rows_read=90,
                    rows_output=90,
                    quality_summary={"documents_indexed": 90},
                    created_at=datetime(2025, 1, 15, 12, 0, tzinfo=UTC).isoformat(),
                ),
            ),
            LineageNode(
                stage="documents",
                run_id="docs-run",
                dataset_prefix="csgo",
                status="completed",
                evidence=RunEvidence(
                    bucket="gold",
                    source_run_id="gold-run",
                    events_key=None,
                    artifact_prefix="csgo/docs-run/documents/",
                    manifest_key="csgo/docs-run/documents/manifest.json",
                    quality_report_key="csgo/docs-run/documents/quality_report.json",
                    files_processed=2,
                    rows_read=100,
                    rows_output=90,
                    quality_summary={"documents_generated": 90},
                    created_at=datetime(2025, 1, 15, 12, 1, tzinfo=UTC).isoformat(),
                ),
            ),
        ],
        evidence=[],
    )


def test_main_prints_json_output(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "rag_intelligence.run_audit_cli.MetadataSettings.from_env",
        lambda: object(),
    )
    monkeypatch.setattr(
        "rag_intelligence.run_audit_cli.get_run_lineage",
        lambda _settings, *, stage, run_id: _build_report(),
    )

    exit_code = main(["--stage", "embeddings", "--run-id", "embed-run", "--format", "json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["requested_stage"] == "embeddings"
    assert payload["summary"]["chain_complete"] is False
    assert payload["resolved_chain"][0]["run_id"] == "embed-run"


def test_main_prints_text_output(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "rag_intelligence.run_audit_cli.MetadataSettings.from_env",
        lambda: object(),
    )
    monkeypatch.setattr(
        "rag_intelligence.run_audit_cli.get_run_lineage",
        lambda _settings, *, stage, run_id: _build_report(),
    )

    exit_code = main(["--stage", "embeddings", "--run-id", "embed-run"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Run audit for embeddings:embed-run" in output
    assert "1. embeddings run=embed-run" in output


def test_main_returns_validation_error_for_broken_lineage(monkeypatch) -> None:
    monkeypatch.setattr(
        "rag_intelligence.run_audit_cli.MetadataSettings.from_env",
        lambda: object(),
    )
    monkeypatch.setattr(
        "rag_intelligence.run_audit_cli.get_run_lineage",
        lambda _settings, *, stage, run_id: (_ for _ in ()).throw(
            ValueError("broken lineage")
        ),
    )

    assert main(["--stage", "embeddings", "--run-id", "embed-run"]) == 2
