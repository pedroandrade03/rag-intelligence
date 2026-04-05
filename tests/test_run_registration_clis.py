from __future__ import annotations

from rag_intelligence.cli import main as bronze_main
from rag_intelligence.config import (
    DocumentSettings,
    EmbeddingSettings,
    GoldSettings,
    Settings,
    SilverSettings,
)
from rag_intelligence.documents import DocumentBuildResult
from rag_intelligence.embedding_cli import main as embedding_main
from rag_intelligence.embeddings import EmbeddingIngestResult
from rag_intelligence.gold import GoldTransformResult
from rag_intelligence.gold_cli import main as gold_main
from rag_intelligence.document_cli import main as document_main
from rag_intelligence.silver import SilverTransformResult
from rag_intelligence.silver_cli import main as silver_main


def _capture_register_run(monkeypatch):
    captured = []
    monkeypatch.setattr("rag_intelligence.metadata.MetadataSettings.from_env", lambda: object())
    monkeypatch.setattr("rag_intelligence.metadata.ensure_schema", lambda _settings: None)
    monkeypatch.setattr(
        "rag_intelligence.metadata.register_run",
        lambda _settings, record: captured.append(record) or record,
    )
    return captured


def test_bronze_cli_registers_artifact_prefix_and_summary(monkeypatch) -> None:
    captured = _capture_register_run(monkeypatch)
    monkeypatch.setattr(
        "rag_intelligence.cli.Settings.from_env",
        lambda: Settings(
            minio_endpoint="localhost:9000",
            minio_access_key="a",
            minio_secret_key="b",
            minio_bucket="bronze",
            minio_secure=False,
            dataset_slug="owner/dataset",
            dataset_prefix="csgo",
            run_id="bronze-run",
        ),
    )
    monkeypatch.setattr(
        "rag_intelligence.cli.run_import",
        lambda _settings: [
            "csgo/bronze-run/raw/archive.zip",
            "csgo/bronze-run/extracted/file1.csv",
            "csgo/bronze-run/extracted/file2.csv",
        ],
    )

    assert bronze_main() == 0
    assert captured[0].artifact_prefix == "csgo/bronze-run/"
    assert captured[0].quality_summary == {
        "files_processed": 3,
        "raw_objects": 1,
        "extracted_objects": 2,
    }


def test_silver_cli_registers_quality_summary(monkeypatch) -> None:
    captured = _capture_register_run(monkeypatch)
    monkeypatch.setattr(
        "rag_intelligence.silver_cli.SilverSettings.from_env",
        lambda: SilverSettings(
            minio_endpoint="localhost:9000",
            minio_access_key="a",
            minio_secret_key="b",
            minio_secure=False,
            bronze_bucket="bronze",
            silver_bucket="silver",
            bronze_dataset_prefix="csgo",
            silver_dataset_prefix="csgo",
            bronze_source_run_id="bronze-run",
            silver_run_id="silver-run",
        ),
    )
    monkeypatch.setattr(
        "rag_intelligence.silver_cli.run_silver_transform",
        lambda _settings: SilverTransformResult(
            uploaded_objects=["report.json"],
            artifact_prefix="csgo/silver-run/cleaned/",
            quality_report_key="csgo/silver-run/quality_report.json",
            files_processed=2,
            rows_read=10,
            rows_output=8,
            quality_summary={"rows_removed": 2},
        ),
    )

    assert silver_main() == 0
    assert captured[0].artifact_prefix == "csgo/silver-run/cleaned/"
    assert captured[0].quality_summary == {"rows_removed": 2}


def test_gold_cli_registers_quality_summary(monkeypatch) -> None:
    captured = _capture_register_run(monkeypatch)
    monkeypatch.setattr(
        "rag_intelligence.gold_cli.GoldSettings.from_env",
        lambda: GoldSettings(
            minio_endpoint="localhost:9000",
            minio_access_key="a",
            minio_secret_key="b",
            minio_secure=False,
            silver_bucket="silver",
            gold_bucket="gold",
            silver_dataset_prefix="csgo",
            gold_dataset_prefix="csgo",
            silver_source_run_id="silver-run",
            gold_run_id="gold-run",
        ),
    )
    monkeypatch.setattr(
        "rag_intelligence.gold_cli.run_gold_transform",
        lambda _settings: GoldTransformResult(
            uploaded_objects=["events.csv", "report.json"],
            artifact_prefix="csgo/gold-run/curated/",
            events_key="csgo/gold-run/curated/events.csv",
            quality_report_key="csgo/gold-run/quality_report.json",
            files_processed=3,
            rows_read=10,
            rows_output=9,
            quality_summary={"rows_removed": 1},
        ),
    )

    assert gold_main() == 0
    assert captured[0].events_key == "csgo/gold-run/curated/events.csv"
    assert captured[0].quality_summary == {"rows_removed": 1}


def test_document_cli_keeps_registering_rich_metadata(monkeypatch) -> None:
    captured = _capture_register_run(monkeypatch)
    monkeypatch.setattr(
        "rag_intelligence.document_cli.DocumentSettings.from_env",
        lambda: DocumentSettings(
            minio_endpoint="localhost:9000",
            minio_access_key="a",
            minio_secret_key="b",
            minio_secure=False,
            gold_bucket="gold",
            document_bucket="gold",
            gold_dataset_prefix="csgo",
            document_dataset_prefix="csgo",
            gold_source_run_id="gold-run",
            document_run_id="docs-run",
            document_part_size_rows=1000,
            document_max_rows=None,
        ),
    )
    monkeypatch.setattr(
        "rag_intelligence.document_cli.run_document_build",
        lambda _settings: DocumentBuildResult(
            uploaded_objects=["manifest.json", "report.json"],
            artifact_prefix="csgo/docs-run/documents/",
            manifest_key="csgo/docs-run/documents/manifest.json",
            quality_report_key="csgo/docs-run/documents/quality_report.json",
            files_processed=2,
            rows_read=100,
            rows_output=90,
            quality_summary={"documents_generated": 90},
        ),
    )

    assert document_main() == 0
    assert captured[0].manifest_key == "csgo/docs-run/documents/manifest.json"
    assert captured[0].quality_summary == {"documents_generated": 90}


def test_embedding_cli_keeps_registering_rich_metadata(monkeypatch) -> None:
    captured = _capture_register_run(monkeypatch)
    monkeypatch.setattr(
        "rag_intelligence.embedding_cli.EmbeddingSettings.from_env",
        lambda: EmbeddingSettings(
            minio_endpoint="localhost:9000",
            minio_access_key="a",
            minio_secret_key="b",
            minio_secure=False,
            document_bucket="gold",
            embedding_report_bucket="gold",
            document_dataset_prefix="csgo",
            embedding_dataset_prefix="csgo",
            document_source_run_id="docs-run",
            embedding_run_id="embed-run",
            embedding_batch_size=32,
            embedding_num_workers=4,
            embedding_parallel_batches=2,
            embedding_max_documents=None,
            embedding_resume=False,
        ),
    )
    monkeypatch.setattr(
        "rag_intelligence.embedding_cli.run_embedding_ingest",
        lambda _settings: EmbeddingIngestResult(
            uploaded_objects=["manifest.json", "report.json"],
            artifact_prefix="csgo/embed-run/embeddings/",
            manifest_key="csgo/embed-run/embeddings/manifest.json",
            quality_report_key="csgo/embed-run/embeddings/quality_report.json",
            files_processed=2,
            rows_read=90,
            rows_output=90,
            quality_summary={"documents_indexed": 90},
        ),
    )

    assert embedding_main() == 0
    assert captured[0].artifact_prefix == "csgo/embed-run/embeddings/"
    assert captured[0].quality_summary == {"documents_indexed": 90}
