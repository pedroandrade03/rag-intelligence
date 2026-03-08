from __future__ import annotations

from dataclasses import dataclass

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from rag_intelligence.retrieval import (
    SearchRequest,
    build_metadata_filters,
    build_search_result,
    sanitize_metadata,
    search_events,
)
from rag_intelligence.settings import AppSettings


def build_app_settings() -> AppSettings:
    return AppSettings.from_env(
        {
            "PG_HOST": "localhost",
            "PG_PORT": "5432",
            "PG_USER": "raguser",
            "PG_PASSWORD": "ragpassword",
            "PG_DATABASE": "ragdb",
            "PG_TABLE_NAME": "rag_embeddings",
            "PG_EMBED_DIM": "768",
            "DEFAULT_EMBED_MODEL": "ollama/nomic-embed-text",
            "DEFAULT_LLM": "ollama/qwen2.5",
            "OLLAMA_BASE_URL": "http://localhost:11434",
        }
    )


def test_build_metadata_filters_always_includes_embedding_run_id() -> None:
    request = SearchRequest(
        query="dano ak47 inferno",
        embedding_run_id="embeddings-smoke",
        top_k=3,
        event_type="damage",
        map_name="de_inferno",
        file_name="match_01.dem",
        round_number=12,
    )

    filters = build_metadata_filters(request)

    assert [(item.key, item.value) for item in filters.filters] == [
        ("embedding_run_id", "embeddings-smoke"),
        ("event_type", "damage"),
        ("map", "de_inferno"),
        ("file", "match_01.dem"),
        ("round", 12),
    ]


def test_sanitize_metadata_removes_internal_keys_and_duplicate_ids() -> None:
    sanitized = sanitize_metadata(
        {
            "_node_content": "internal",
            "doc_id": "doc-1",
            "document_id": "doc-1",
            "ref_doc_id": "doc-1",
            "event_type": "damage",
            "map": "de_inferno",
        }
    )

    assert sanitized == {
        "doc_id": "doc-1",
        "event_type": "damage",
        "map": "de_inferno",
    }


def test_build_search_result_serializes_core_fields() -> None:
    item = NodeWithScore(
        node=TextNode(
            text="Evento damage em map=de_inferno file=match.dem round=12.",
            metadata={
                "doc_id": "doc-1",
                "document_id": "doc-1",
                "_node_content": "internal",
                "event_type": "damage",
                "map": "de_inferno",
                "file": "match.dem",
                "round": 12,
                "source_file": "events.csv",
            },
        ),
        score=0.91,
    )

    result = build_search_result(item, rank=1)

    assert result.rank == 1
    assert result.score == 0.91
    assert result.doc_id == "doc-1"
    assert result.event_type == "damage"
    assert result.map == "de_inferno"
    assert result.file == "match.dem"
    assert result.round == 12
    assert result.source_file == "events.csv"
    assert "_node_content" not in result.metadata
    assert "document_id" not in result.metadata


@dataclass
class FakeRegistry:
    _settings: AppSettings

    def get_embed_model(self) -> object:
        return object()


class FakeRetriever:
    def __init__(self, returned: list[NodeWithScore]) -> None:
        self.returned = returned
        self.queries: list[str] = []

    def retrieve(self, query: str) -> list[NodeWithScore]:
        self.queries.append(query)
        return self.returned


class FakeIndex:
    def __init__(self, returned: list[NodeWithScore]) -> None:
        self.returned = returned
        self.calls: list[dict[str, object]] = []
        self.retriever = FakeRetriever(returned)

    def as_retriever(self, **kwargs: object) -> FakeRetriever:
        self.calls.append(kwargs)
        return self.retriever


def test_search_events_uses_vector_store_filters_and_returns_results() -> None:
    captured: dict[str, object] = {}
    node = NodeWithScore(
        node=TextNode(
            text="Evento damage em map=de_inferno file=match.dem round=12.",
            metadata={
                "doc_id": "doc-1",
                "_node_type": "text",
                "event_type": "damage",
                "map": "de_inferno",
                "file": "match.dem",
                "round": 12,
                "source_file": "events.csv",
            },
        ),
        score=0.88,
    )
    fake_index = FakeIndex([node])

    def fake_vector_store_factory(
        _settings: AppSettings,
        *,
        perform_setup: bool = True,
    ) -> object:
        captured["perform_setup"] = perform_setup
        return object()

    def fake_index_factory(vector_store: object, embed_model: object | None = None) -> FakeIndex:
        captured["vector_store"] = vector_store
        captured["embed_model"] = embed_model
        return fake_index

    response = search_events(
        SearchRequest(
            query="dano ak47 inferno",
            embedding_run_id="embeddings-smoke",
            top_k=4,
            event_type="damage",
            map_name="de_inferno",
        ),
        app_settings_factory=build_app_settings,
        registry_factory=FakeRegistry,
        vector_store_factory=fake_vector_store_factory,
        vector_table_exists_fn=lambda _settings: True,
        index_factory=fake_index_factory,
    )

    assert captured["perform_setup"] is False
    assert fake_index.retriever.queries == ["dano ak47 inferno"]
    retriever_call = fake_index.calls[0]
    assert retriever_call["similarity_top_k"] == 4
    filters = retriever_call["filters"]
    assert [(item.key, item.value) for item in filters.filters] == [
        ("embedding_run_id", "embeddings-smoke"),
        ("event_type", "damage"),
        ("map", "de_inferno"),
    ]
    assert response.embedding_run_id == "embeddings-smoke"
    assert response.results_returned == 1
    assert response.results[0].doc_id == "doc-1"
    assert response.results[0].metadata["event_type"] == "damage"


def test_search_events_returns_empty_results_without_error() -> None:
    fake_index = FakeIndex([])

    response = search_events(
        SearchRequest(query="nada", embedding_run_id="embeddings-smoke"),
        app_settings_factory=build_app_settings,
        registry_factory=FakeRegistry,
        vector_store_factory=lambda _settings, perform_setup=True: object(),
        vector_table_exists_fn=lambda _settings: True,
        index_factory=lambda _vector_store, embed_model=None: fake_index,
    )

    assert response.results_returned == 0
    assert response.results == []


def test_search_events_fails_when_vector_table_is_missing() -> None:
    with pytest.raises(RuntimeError, match="Vector table not found"):
        search_events(
            SearchRequest(query="ak47", embedding_run_id="embeddings-smoke"),
            app_settings_factory=build_app_settings,
            registry_factory=FakeRegistry,
            vector_store_factory=lambda _settings, perform_setup=True: object(),
            vector_table_exists_fn=lambda _settings: False,
            index_factory=lambda _vector_store, embed_model=None: FakeIndex([]),
        )
