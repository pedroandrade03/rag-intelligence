"""Embed pipeline documentation (Markdown) into pgvector."""

from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv

from rag_intelligence.db import create_vector_store, ensure_pgvector_storage_contract
from rag_intelligence.embeddings_storage import delete_embeddings_for_run
from rag_intelligence.logging import setup_logging
from rag_intelligence.providers import ProviderRegistry
from rag_intelligence.settings import AppSettings

LOGGER = logging.getLogger(__name__)

PIPELINE_DOCS_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "pipeline"
EMBEDDING_RUN_ID = "pipeline-docs"

_PHASE_MAP = {
    "01-bronze": "bronze",
    "02-silver": "silver",
    "03-gold": "gold",
    "04-ml-training": "ml-training",
    "05-architecture": "architecture",
}


def main() -> int:
    load_dotenv()
    setup_logging()

    try:
        from llama_index.core.ingestion import IngestionPipeline
        from llama_index.core.node_parser import MarkdownNodeParser
        from llama_index.core.schema import Document
    except ImportError as exc:
        LOGGER.error("LlamaIndex not installed: %s", exc)
        return 2

    settings = AppSettings.from_env()
    registry = ProviderRegistry(settings)
    embed_model = registry.get_embed_model()

    docs_dir = PIPELINE_DOCS_DIR
    if not docs_dir.exists():
        LOGGER.error("Pipeline docs directory not found: %s", docs_dir)
        return 2

    md_files = sorted(docs_dir.glob("*.md"))
    if not md_files:
        LOGGER.error("No markdown files in %s", docs_dir)
        return 2

    LOGGER.info("Found %s pipeline docs in %s", len(md_files), docs_dir)

    # Build LlamaIndex documents
    documents: list[Document] = []
    for md_file in md_files:
        stem = md_file.stem
        phase = _PHASE_MAP.get(stem, stem)
        text = md_file.read_text(encoding="utf-8")
        doc = Document(
            text=text,
            metadata={
                "embedding_run_id": EMBEDDING_RUN_ID,
                "document_tier": "pipeline_doc",
                "pipeline_phase": phase,
                "source_file": md_file.name,
            },
        )
        documents.append(doc)

    # Idempotency: delete existing pipeline-docs embeddings
    deleted = delete_embeddings_for_run(settings, EMBEDDING_RUN_ID)
    if deleted:
        LOGGER.info("Deleted %s existing pipeline-docs embeddings", deleted)

    # Chunk with MarkdownNodeParser and embed via IngestionPipeline
    parser = MarkdownNodeParser()
    vector_store = create_vector_store(settings, perform_setup=True)
    ensure_pgvector_storage_contract(settings)

    pipeline = IngestionPipeline(
        transformations=[parser, embed_model],
        vector_store=vector_store,
    )
    nodes = pipeline.run(documents=documents, show_progress=True)

    LOGGER.info(
        "Embedded %s nodes from %s pipeline docs (run_id=%s)",
        len(nodes),
        len(documents),
        EMBEDDING_RUN_ID,
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
        source_files = [md_file.name for md_file in md_files]
        docs_quality_summary = {
            "source_files": source_files,
            "document_count": len(documents),
        }
        register_run(
            md_settings,
            RunRecord(
                run_id=EMBEDDING_RUN_ID,
                stage="documents",
                dataset_prefix=EMBEDDING_RUN_ID,
                bucket="repo",
                artifact_prefix=str(docs_dir),
                files_processed=len(md_files),
                rows_read=len(documents),
                rows_output=len(documents),
                quality_summary=docs_quality_summary,
            ),
        )
        register_run(
            md_settings,
            RunRecord(
                run_id=EMBEDDING_RUN_ID,
                stage="embeddings",
                dataset_prefix=EMBEDDING_RUN_ID,
                bucket=settings.pg_table_name,
                source_run_id=EMBEDDING_RUN_ID,
                artifact_prefix=str(docs_dir),
                files_processed=len(md_files),
                rows_read=len(documents),
                rows_output=len(nodes),
                quality_summary={
                    **docs_quality_summary,
                    "node_count": len(nodes),
                    "deleted_existing_rows": deleted,
                },
            ),
        )
        LOGGER.info("Registered pipeline-docs embedding lineage in metadata")
    except Exception as exc:
        LOGGER.warning("Failed to register pipeline-docs metadata (non-fatal): %s", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
