from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient
from llama_index.core.schema import NodeWithScore, TextNode

from rag_intelligence.api.main import create_app
from rag_intelligence.settings import AppSettings


def _settings() -> AppSettings:
    return AppSettings.from_env({"DEFAULT_EMBEDDING_RUN_ID": "run-test"})


def _make_node(doc_id: str = "doc-1", score: float = 0.9) -> NodeWithScore:
    return NodeWithScore(
        node=TextNode(
            text="Player killed enemy with AK-47",
            metadata={
                "doc_id": doc_id,
                "event_type": "kill",
                "map": "de_dust2",
                "file": "match.dem",
                "round": 5,
                "embedding_run_id": "run-test",
            },
        ),
        score=score,
    )


def _patch_search(nodes: list[NodeWithScore]):
    """Patch search_events to avoid DB/vector store access."""
    from rag_intelligence.retrieval import SearchRequest, SearchResponse, build_search_result

    def fake_search(request: SearchRequest, **_kwargs):
        results = [build_search_result(n, rank=i) for i, n in enumerate(nodes, 1)]
        return SearchResponse(
            query=request.query,
            embedding_run_id=request.embedding_run_id,
            top_k=request.top_k,
            filters=request.filters_payload,
            results_returned=len(results),
            retrieval_ms=10,
            results=results,
        )

    return patch("rag_intelligence.api.routes.search.search_events", side_effect=fake_search)


def test_search_returns_results():
    app = create_app(settings=_settings())
    with _patch_search([_make_node()]), TestClient(app) as client:
        resp = client.post("/search", json={"query": "AK47 kills"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["results_returned"] == 1
    assert data["results"][0]["doc_id"] == "doc-1"


def test_search_uses_default_embedding_run_id():
    app = create_app(settings=_settings())
    with _patch_search([]) as mock_fn, TestClient(app) as client:
        resp = client.post("/search", json={"query": "test"})

    assert resp.status_code == 200
    call_args = mock_fn.call_args
    request = call_args[0][0]
    assert request.embedding_run_id == "run-test"


def test_search_422_when_no_embedding_run_id():
    settings = AppSettings.from_env({})  # no DEFAULT_EMBEDDING_RUN_ID
    app = create_app(settings=settings)
    with TestClient(app) as client:
        resp = client.post("/search", json={"query": "test"})
    assert resp.status_code == 422


def test_search_422_for_empty_query():
    app = create_app(settings=_settings())
    with TestClient(app) as client:
        resp = client.post("/search", json={"embedding_run_id": "run-1"})
    assert resp.status_code == 422


def test_search_passes_filters():
    app = create_app(settings=_settings())
    with _patch_search([_make_node()]) as mock_fn, TestClient(app) as client:
        resp = client.post(
            "/search",
            json={
                "query": "headshot",
                "embedding_run_id": "run-1",
                "event_type": "kill",
                "map_name": "de_dust2",
                "top_k": 3,
            },
        )

    assert resp.status_code == 200
    request = mock_fn.call_args[0][0]
    assert request.event_type == "kill"
    assert request.map_name == "de_dust2"
    assert request.top_k == 3
