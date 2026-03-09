from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import Document

from rag_intelligence.settings import AppSettings

THREAD_LOCAL = threading.local()
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddedBatch:
    nodes: list[Any]
    rows_indexed: int


def _validate_embedding_dimensions(
    documents: Sequence[Document],
    embed_model: Any,
    *,
    expected_dim: int,
    num_workers: int,
    pipeline_factory: Any = IngestionPipeline,
) -> None:
    pipeline = pipeline_factory(transformations=[embed_model], disable_cache=True)
    nodes = pipeline.run(documents=list(documents), num_workers=num_workers)
    if not nodes:
        raise ValueError("Embedding preview batch did not produce any nodes")

    first_embedding = getattr(nodes[0], "embedding", None)
    if not isinstance(first_embedding, list) or not first_embedding:
        raise ValueError("Embedding preview batch did not produce a valid embedding")

    actual_dim = len(first_embedding)
    if actual_dim != expected_dim:
        raise ValueError(f"PG_EMBED_DIM mismatch: expected {expected_dim}, got {actual_dim}")


def _get_thread_embedding_pipeline(
    *,
    app_settings: AppSettings,
    embed_model_name: str,
    registry_factory: Any,
    pipeline_factory: Any,
) -> Any:
    cache_key = (
        app_settings.ollama_base_url,
        app_settings.ollama_embed_batch_size,
        embed_model_name,
        pipeline_factory,
    )
    cached_key = getattr(THREAD_LOCAL, "embedding_pipeline_key", None)
    cached_pipeline = getattr(THREAD_LOCAL, "embedding_pipeline", None)
    if cached_key == cache_key and cached_pipeline is not None:
        return cached_pipeline

    registry = registry_factory(app_settings)
    embed_model = registry.get_embed_model(embed_model_name)
    pipeline = pipeline_factory(transformations=[embed_model], disable_cache=True)
    THREAD_LOCAL.embedding_pipeline_key = cache_key
    THREAD_LOCAL.embedding_pipeline = pipeline
    return pipeline


def _embed_document_batch(
    documents: Sequence[Document],
    *,
    app_settings: AppSettings,
    embed_model_name: str,
    registry_factory: Any,
    pipeline_factory: Any,
    num_workers: int,
) -> EmbeddedBatch:
    pipeline = _get_thread_embedding_pipeline(
        app_settings=app_settings,
        embed_model_name=embed_model_name,
        registry_factory=registry_factory,
        pipeline_factory=pipeline_factory,
    )
    nodes = _embed_documents_with_nan_fallback(
        list(documents),
        pipeline=pipeline,
        num_workers=num_workers,
    )
    return EmbeddedBatch(nodes=list(nodes), rows_indexed=len(nodes))


def _embed_documents_with_nan_fallback(
    documents: list[Document],
    *,
    pipeline: Any,
    num_workers: int,
) -> list[Any]:
    if not documents:
        return []

    try:
        return list(pipeline.run(documents=documents, num_workers=num_workers))
    except Exception as exc:
        if not _is_ollama_nan_response_error(exc):
            raise

        if len(documents) == 1:
            doc_id = str(documents[0].doc_id)
            LOGGER.warning(
                ("Skipping document due to Ollama NaN embedding response: doc_id=%s error=%s"),
                doc_id,
                exc,
            )
            return []

        midpoint = len(documents) // 2
        left_batch = documents[:midpoint]
        right_batch = documents[midpoint:]
        LOGGER.warning(
            (
                "Embedding batch failed with NaN response; retrying in smaller "
                "batches: size=%s left=%s right=%s"
            ),
            len(documents),
            len(left_batch),
            len(right_batch),
        )
        left_nodes = _embed_documents_with_nan_fallback(
            left_batch,
            pipeline=pipeline,
            num_workers=num_workers,
        )
        right_nodes = _embed_documents_with_nan_fallback(
            right_batch,
            pipeline=pipeline,
            num_workers=num_workers,
        )
        return [*left_nodes, *right_nodes]


def _is_ollama_nan_response_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "failed to encode response" in message and "unsupported value: nan" in message


def _submit_embedding_batch(
    executor: ThreadPoolExecutor,
    documents: Sequence[Document],
    *,
    app_settings: AppSettings,
    embed_model_name: str,
    registry_factory: Any,
    pipeline_factory: Any,
    num_workers: int,
) -> Future[EmbeddedBatch]:
    submitted_documents = list(documents)
    return executor.submit(
        _embed_document_batch,
        submitted_documents,
        app_settings=app_settings,
        embed_model_name=embed_model_name,
        registry_factory=registry_factory,
        pipeline_factory=pipeline_factory,
        num_workers=num_workers,
    )


def _flush_completed_embedding_batches(
    pending_futures: list[Future[EmbeddedBatch]],
    *,
    vector_store: Any,
) -> tuple[list[Future[EmbeddedBatch]], int]:
    if not pending_futures:
        return [], 0

    done, not_done = wait(pending_futures, return_when=FIRST_COMPLETED)
    rows_indexed = 0
    for future in done:
        embedded_batch = future.result()
        if embedded_batch.nodes:
            vector_store.add(embedded_batch.nodes)
        rows_indexed += embedded_batch.rows_indexed
    return list(not_done), rows_indexed
