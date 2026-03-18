from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from rag_intelligence.rag import (
    QA_PROMPT,
    RAGRequest,
    _nodes_to_sources,
    rag_query,
    rag_query_stream,
)
from rag_intelligence.settings import AppSettings


def _build_settings() -> AppSettings:
    return AppSettings.from_env({})


def _make_nodes(n: int = 2) -> list[NodeWithScore]:
    return [
        NodeWithScore(
            node=TextNode(
                text=f"Player killed enemy with AK-47 on round {i}",
                id_=f"doc-{i}",
                metadata={
                    "doc_id": f"doc-{i}",
                    "event_type": "kill",
                    "map": "de_dust2",
                    "file": "match.dem",
                    "round": i,
                    "source_file": "events.csv",
                },
            ),
            score=0.9 - (i - 1) * 0.1,
        )
        for i in range(1, n + 1)
    ]


# --- Test fakes matching LlamaIndex QueryEngine interface ---


@dataclass
class FakeResponse:
    response: str
    source_nodes: list = field(default_factory=list)

    def __str__(self) -> str:
        return self.response


class FakeAsyncStreamingResponse:
    def __init__(self, answer: str, source_nodes: list) -> None:
        self.source_nodes = source_nodes
        self._tokens = answer.split(" ")

    async def async_response_gen(self):
        for i, token in enumerate(self._tokens):
            yield token if i == 0 else " " + token


class FakeQueryEngine:
    def __init__(self, answer: str, source_nodes: list) -> None:
        self.answer = answer
        self.source_nodes = source_nodes

    def query(self, query_str: str) -> FakeResponse:
        return FakeResponse(response=self.answer, source_nodes=self.source_nodes)

    async def aquery(self, query_str: str) -> FakeAsyncStreamingResponse:
        return FakeAsyncStreamingResponse(self.answer, self.source_nodes)


class FakeIndex:
    def __init__(self, engine: FakeQueryEngine) -> None:
        self._engine = engine

    def as_query_engine(self, **kwargs: object) -> FakeQueryEngine:
        return self._engine


@dataclass
class FakeRegistry:
    _settings: AppSettings

    def get_llm(self, key: str | None = None) -> object:
        return object()

    def get_embed_model(self) -> object:
        return object()


def _engine_deps(answer: str = "AK-47 is the best.", nodes: list | None = None) -> dict:
    source_nodes = nodes if nodes is not None else _make_nodes()
    engine = FakeQueryEngine(answer=answer, source_nodes=source_nodes)
    return {
        "app_settings_factory": _build_settings,
        "registry_factory": lambda s: FakeRegistry(s),
        "vector_store_factory": lambda _s, perform_setup=True: object(),
        "index_factory": lambda _vs, embed_model=None: FakeIndex(engine),
    }


# --- QA_PROMPT ---


def test_qa_prompt_has_required_template_variables():
    assert "{context_str}" in QA_PROMPT.template
    assert "{query_str}" in QA_PROMPT.template


# --- _nodes_to_sources ---


def test_nodes_to_sources_extracts_fields():
    sources = _nodes_to_sources(_make_nodes(2))
    assert len(sources) == 2
    assert sources[0]["rank"] == 1
    assert sources[0]["doc_id"] == "doc-1"
    assert sources[0]["event_type"] == "kill"
    assert "AK-47" in sources[0]["text"]


def test_nodes_to_sources_empty():
    assert _nodes_to_sources([]) == []


# --- rag_query (sync) ---


def test_rag_query_returns_answer():
    response = rag_query(
        RAGRequest(query="best weapon", embedding_run_id="run-1", top_k=5),
        **_engine_deps(answer="AK-47 is the best."),
    )
    assert response.answer == "AK-47 is the best."
    assert response.query == "best weapon"
    assert len(response.sources) == 2
    assert response.retrieval_ms >= 0


def test_rag_query_empty_results():
    response = rag_query(
        RAGRequest(query="nothing", embedding_run_id="run-1"),
        **_engine_deps(answer="No data available.", nodes=[]),
    )
    assert response.sources == []
    assert response.answer == "No data available."


# --- rag_query_stream (async) ---


@pytest.mark.asyncio
async def test_rag_query_stream_yields_sse_events():
    events: list[str] = []
    async for event in rag_query_stream(
        RAGRequest(query="headshot", embedding_run_id="run-1"),
        **_engine_deps(answer="headshot damage"),
    ):
        events.append(event)

    assert events[0].startswith("event: sources\n")
    sources_data = json.loads(events[0].split("data: ", 1)[1].strip())
    assert "sources" in sources_data
    assert "retrieval_ms" in sources_data

    chunk_events = [e for e in events if e.startswith("event: chunk\n")]
    assert len(chunk_events) >= 1
    tokens = [json.loads(ce.split("data: ", 1)[1].strip())["token"] for ce in chunk_events]
    assert "".join(tokens) == "headshot damage"

    assert events[-1].startswith("event: done\n")
    done_data = json.loads(events[-1].split("data: ", 1)[1].strip())
    assert done_data["answer"] == "headshot damage"
    assert "generation_ms" in done_data


@pytest.mark.asyncio
async def test_rag_query_stream_empty_results():
    events: list[str] = []
    async for event in rag_query_stream(
        RAGRequest(query="nothing", embedding_run_id="run-1"),
        **_engine_deps(answer="no data", nodes=[]),
    ):
        events.append(event)

    sources_data = json.loads(events[0].split("data: ", 1)[1].strip())
    assert sources_data["sources"] == []
