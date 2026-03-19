"""RAG synthesis using LlamaIndex QueryEngine.

Kept as an alternative server-side RAG endpoint. The primary frontend
uses AI SDK tool-calling with ``/search`` instead.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from rag_intelligence.db import create_vector_store
from rag_intelligence.providers import ProviderRegistry
from rag_intelligence.retrieval import SearchRequest, build_metadata_filters, build_search_result
from rag_intelligence.settings import AppSettings

LOGGER = logging.getLogger(__name__)

QA_PROMPT = PromptTemplate(
    "You are a CS:GO / Counter-Strike professional analyst. "
    "Answer the user's question using ONLY the statistical summaries provided below. "
    "The context contains pre-aggregated profiles (weapon-map profiles, map overviews, "
    "hotspot zones, round-type profiles, and global weapon profiles) derived from "
    "millions of recorded match events. "
    "Use the numbers directly — do not fabricate or extrapolate beyond what is stated. "
    "If the data is insufficient to answer, say so clearly.\n\n"
    "--- STATISTICAL SUMMARIES ---\n{context_str}\n--- END DATA ---\n\n"
    "User question: {query_str}"
)


@dataclass(frozen=True)
class RAGRequest:
    query: str
    embedding_run_id: str
    top_k: int = 10
    event_type: str | None = None
    map_name: str | None = None
    llm_key: str | None = None


@dataclass(frozen=True)
class RAGResponse:
    query: str
    answer: str
    sources: list[dict[str, Any]]
    retrieval_ms: int
    generation_ms: int


def _nodes_to_sources(nodes: list[NodeWithScore]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for i, node in enumerate(nodes, 1):
        r = build_search_result(node, rank=i)
        sources.append(
            {
                "rank": r.rank,
                "score": r.score,
                "doc_id": r.doc_id,
                "text": r.text,
                "event_type": r.event_type,
                "map": r.map,
            }
        )
    return sources


def _build_query_engine(
    request: RAGRequest,
    *,
    app_settings_factory: Any,
    registry_factory: Any,
    vector_store_factory: Any,
    index_factory: Any,
    streaming: bool = False,
) -> Any:
    settings = app_settings_factory()
    registry = registry_factory(settings)
    vector_store = vector_store_factory(settings, perform_setup=False)
    index = index_factory(vector_store, embed_model=registry.get_embed_model())

    search_req = SearchRequest(
        query=request.query,
        embedding_run_id=request.embedding_run_id,
        top_k=request.top_k,
        event_type=request.event_type,
        map_name=request.map_name,
    )
    return index.as_query_engine(
        similarity_top_k=request.top_k,
        filters=build_metadata_filters(search_req),
        text_qa_template=QA_PROMPT,
        response_mode="compact",
        llm=registry.get_llm(request.llm_key),
        streaming=streaming,
    )


def rag_query(
    request: RAGRequest,
    *,
    app_settings_factory: Any = AppSettings.from_env,
    registry_factory: Any = ProviderRegistry,
    vector_store_factory: Any = create_vector_store,
    index_factory: Any = VectorStoreIndex.from_vector_store,
) -> RAGResponse:
    """Synchronous RAG via LlamaIndex QueryEngine."""
    engine = _build_query_engine(
        request,
        app_settings_factory=app_settings_factory,
        registry_factory=registry_factory,
        vector_store_factory=vector_store_factory,
        index_factory=index_factory,
    )
    start = perf_counter()
    response = engine.query(request.query)
    total_ms = int((perf_counter() - start) * 1000)

    return RAGResponse(
        query=request.query,
        answer=str(response),
        sources=_nodes_to_sources(response.source_nodes),
        retrieval_ms=total_ms,
        generation_ms=0,
    )


async def rag_query_stream(
    request: RAGRequest,
    *,
    app_settings_factory: Any = AppSettings.from_env,
    registry_factory: Any = ProviderRegistry,
    vector_store_factory: Any = create_vector_store,
    index_factory: Any = VectorStoreIndex.from_vector_store,
) -> AsyncGenerator[str]:
    """Streaming RAG: yields SSE events (sources -> chunk* -> done)."""
    engine = _build_query_engine(
        request,
        app_settings_factory=app_settings_factory,
        registry_factory=registry_factory,
        vector_store_factory=vector_store_factory,
        index_factory=index_factory,
        streaming=True,
    )
    start = perf_counter()
    response = await engine.aquery(request.query)

    sources = _nodes_to_sources(response.source_nodes)
    retrieval_ms = int((perf_counter() - start) * 1000)
    yield _sse("sources", {"sources": sources, "retrieval_ms": retrieval_ms})

    gen_start = perf_counter()
    full_answer = ""
    async for token in response.async_response_gen():
        if token:
            full_answer += token
            yield _sse("chunk", {"token": token})

    generation_ms = int((perf_counter() - gen_start) * 1000)
    yield _sse("done", {"answer": full_answer, "generation_ms": generation_ms})


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
