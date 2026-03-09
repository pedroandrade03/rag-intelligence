"""RAG synthesis: retrieve context from pgvectorscale, format prompt, call LLM."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from rag_intelligence.providers import ProviderRegistry
from rag_intelligence.retrieval import SearchRequest, SearchResponse, SearchResult, search_events
from rag_intelligence.settings import AppSettings

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a CS:GO / Counter-Strike professional analyst. \
Answer the user's question using ONLY the match event data provided below. \
If the data is insufficient, say so — never fabricate information. \
Be precise, cite specific events when relevant, and use numbers from the data.\
"""


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


def format_context(results: list[SearchResult]) -> str:
    """Format search results as a numbered context block for the LLM prompt."""
    if not results:
        return "(no relevant events found)"
    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(f"[{i}] {r.text}")
    return "\n".join(lines)


def build_prompt(query: str, context: str) -> str:
    """Combine system prompt, retrieved context, and user query into a single prompt."""
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"--- MATCH EVENT DATA ---\n{context}\n--- END DATA ---\n\n"
        f"User question: {query}"
    )


def _search_response_to_sources(response: SearchResponse) -> list[dict[str, Any]]:
    return [
        {
            "rank": r.rank,
            "score": r.score,
            "doc_id": r.doc_id,
            "text": r.text,
            "event_type": r.event_type,
            "map": r.map,
        }
        for r in response.results
    ]


def rag_query(
    request: RAGRequest,
    *,
    search_fn: Any = search_events,
    app_settings_factory: Any = AppSettings.from_env,
    registry_factory: Any = ProviderRegistry,
) -> RAGResponse:
    """Synchronous RAG: retrieve → prompt → LLM complete."""
    settings = app_settings_factory()
    registry = registry_factory(settings)
    llm = registry.get_llm(request.llm_key)

    search_req = SearchRequest(
        query=request.query,
        embedding_run_id=request.embedding_run_id,
        top_k=request.top_k,
        event_type=request.event_type,
        map_name=request.map_name,
    )
    search_resp = search_fn(
        search_req,
        app_settings_factory=app_settings_factory,
        registry_factory=registry_factory,
    )

    context = format_context(search_resp.results)
    prompt = build_prompt(request.query, context)

    gen_start = perf_counter()
    completion = llm.complete(prompt)
    generation_ms = int((perf_counter() - gen_start) * 1000)

    return RAGResponse(
        query=request.query,
        answer=completion.text,
        sources=_search_response_to_sources(search_resp),
        retrieval_ms=search_resp.retrieval_ms,
        generation_ms=generation_ms,
    )


async def rag_query_stream(
    request: RAGRequest,
    *,
    search_fn: Any = search_events,
    app_settings_factory: Any = AppSettings.from_env,
    registry_factory: Any = ProviderRegistry,
) -> AsyncGenerator[str]:
    """Streaming RAG: yields SSE events (sources → chunk* → done)."""
    settings = app_settings_factory()
    registry = registry_factory(settings)
    llm = registry.get_llm(request.llm_key)

    search_req = SearchRequest(
        query=request.query,
        embedding_run_id=request.embedding_run_id,
        top_k=request.top_k,
        event_type=request.event_type,
        map_name=request.map_name,
    )
    search_resp = search_fn(
        search_req,
        app_settings_factory=app_settings_factory,
        registry_factory=registry_factory,
    )

    sources = _search_response_to_sources(search_resp)
    yield _sse(
        "sources",
        {
            "sources": sources,
            "retrieval_ms": search_resp.retrieval_ms,
        },
    )

    context = format_context(search_resp.results)
    prompt = build_prompt(request.query, context)

    gen_start = perf_counter()
    full_answer = ""
    async for chunk in await llm.astream_complete(prompt):
        token = chunk.delta or ""
        if token:
            full_answer += token
            yield _sse("chunk", {"token": token})

    generation_ms = int((perf_counter() - gen_start) * 1000)
    yield _sse(
        "done",
        {
            "answer": full_answer,
            "generation_ms": generation_ms,
        },
    )


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
