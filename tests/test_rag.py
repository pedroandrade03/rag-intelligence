from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
from llama_index.core.base.llms.types import CompletionResponse

from rag_intelligence.rag import (
    SYSTEM_PROMPT,
    RAGRequest,
    build_prompt,
    format_context,
    rag_query,
    rag_query_stream,
)
from rag_intelligence.retrieval import SearchRequest, SearchResponse, SearchResult
from rag_intelligence.settings import AppSettings


def _build_settings() -> AppSettings:
    return AppSettings.from_env({})


def _make_results(n: int = 2) -> list[SearchResult]:
    return [
        SearchResult(
            rank=i,
            score=0.9 - i * 0.1,
            doc_id=f"doc-{i}",
            text=f"Player killed enemy with AK-47 on round {i}",
            event_type="kill",
            map="de_dust2",
            file="match.dem",
            round=i,
            source_file="events.csv",
            metadata={"event_type": "kill", "map": "de_dust2"},
        )
        for i in range(1, n + 1)
    ]


def _make_search_response(results: list[SearchResult] | None = None) -> SearchResponse:
    results = results if results is not None else _make_results()
    return SearchResponse(
        query="AK47 kills",
        embedding_run_id="run-1",
        top_k=5,
        filters={},
        results_returned=len(results),
        retrieval_ms=42,
        results=results,
    )


class FakeSearchFn:
    def __init__(self, response: SearchResponse) -> None:
        self.response = response
        self.calls: list[SearchRequest] = []

    def __call__(self, request: SearchRequest, **kwargs: object) -> SearchResponse:
        self.calls.append(request)
        return self.response


@dataclass
class FakeLLM:
    answer: str = "The AK-47 deals 111 damage to the head."

    def complete(self, prompt: str, **kwargs: object) -> CompletionResponse:
        return CompletionResponse(text=self.answer)

    async def astream_complete(self, prompt: str, **kwargs: object):
        async def _gen():
            tokens = self.answer.split(" ")
            for i, token in enumerate(tokens):
                t = token if i == 0 else " " + token
                yield CompletionResponse(text="", delta=t)

        return _gen()


@dataclass
class FakeRegistry:
    _settings: AppSettings
    llm: FakeLLM | None = None

    def get_llm(self, key: str | None = None) -> FakeLLM:
        return self.llm or FakeLLM()

    def get_embed_model(self) -> object:
        return object()


# --- format_context ---


def test_format_context_numbered_list():
    results = _make_results(3)
    ctx = format_context(results)
    lines = ctx.strip().split("\n")
    assert len(lines) == 3
    assert lines[0].startswith("[1]")
    assert lines[2].startswith("[3]")
    assert "AK-47" in lines[0]


def test_format_context_empty():
    assert format_context([]) == "(no relevant events found)"


# --- build_prompt ---


def test_build_prompt_includes_all_parts():
    prompt = build_prompt("What weapons?", "Some context here")
    assert SYSTEM_PROMPT in prompt
    assert "Some context here" in prompt
    assert "What weapons?" in prompt
    assert "--- MATCH EVENT DATA ---" in prompt


# --- rag_query (sync) ---


def test_rag_query_returns_answer():
    fake_search = FakeSearchFn(_make_search_response())
    fake_llm = FakeLLM(answer="AK-47 is the best.")

    response = rag_query(
        RAGRequest(query="best weapon", embedding_run_id="run-1", top_k=5),
        search_fn=fake_search,
        app_settings_factory=_build_settings,
        registry_factory=lambda s: FakeRegistry(s, llm=fake_llm),
    )

    assert response.answer == "AK-47 is the best."
    assert response.query == "best weapon"
    assert len(response.sources) == 2
    assert response.retrieval_ms == 42
    assert response.generation_ms >= 0
    assert fake_search.calls[0].query == "best weapon"


def test_rag_query_empty_results():
    empty_resp = _make_search_response(results=[])
    fake_search = FakeSearchFn(empty_resp)

    response = rag_query(
        RAGRequest(query="nothing", embedding_run_id="run-1"),
        search_fn=fake_search,
        app_settings_factory=_build_settings,
        registry_factory=lambda s: FakeRegistry(s),
    )

    assert response.sources == []
    assert response.answer  # LLM still returns something


# --- rag_query_stream (async) ---


@pytest.mark.asyncio
async def test_rag_query_stream_yields_sse_events():
    fake_search = FakeSearchFn(_make_search_response())
    fake_llm = FakeLLM(answer="headshot damage")

    events: list[str] = []
    async for event in rag_query_stream(
        RAGRequest(query="headshot", embedding_run_id="run-1"),
        search_fn=fake_search,
        app_settings_factory=_build_settings,
        registry_factory=lambda s: FakeRegistry(s, llm=fake_llm),
    ):
        events.append(event)

    # First event: sources
    assert events[0].startswith("event: sources\n")
    sources_data = json.loads(events[0].split("data: ", 1)[1].strip())
    assert "sources" in sources_data
    assert sources_data["retrieval_ms"] == 42

    # Middle events: chunks
    chunk_events = [e for e in events if e.startswith("event: chunk\n")]
    assert len(chunk_events) >= 1
    tokens = []
    for ce in chunk_events:
        d = json.loads(ce.split("data: ", 1)[1].strip())
        tokens.append(d["token"])
    assert "".join(tokens) == "headshot damage"

    # Last event: done
    assert events[-1].startswith("event: done\n")
    done_data = json.loads(events[-1].split("data: ", 1)[1].strip())
    assert done_data["answer"] == "headshot damage"
    assert "generation_ms" in done_data


@pytest.mark.asyncio
async def test_rag_query_stream_empty_results():
    empty_resp = _make_search_response(results=[])
    fake_search = FakeSearchFn(empty_resp)

    events: list[str] = []
    async for event in rag_query_stream(
        RAGRequest(query="nothing", embedding_run_id="run-1"),
        search_fn=fake_search,
        app_settings_factory=_build_settings,
        registry_factory=lambda s: FakeRegistry(s),
    ):
        events.append(event)

    sources_data = json.loads(events[0].split("data: ", 1)[1].strip())
    assert sources_data["sources"] == []
