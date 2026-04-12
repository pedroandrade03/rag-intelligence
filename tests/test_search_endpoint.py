from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class _FakeLexicalHit:
    run_id: str
    created_at: str
    model_name: str

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": 1,
            "score": 0.42,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "model_name": self.model_name,
            "roc_auc": 0.9,
            "f1": 0.8,
            "balanced_accuracy": 0.85,
            "log_loss_val": 0.2,
            "brier": 0.1,
            "feature_importances": {"eq_diff": 0.5},
            "text_summary": "summary",
        }


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


def test_search_does_not_require_fastapi_threadpool_for_state_dependencies():
    app = create_app(settings=_settings())
    with (
        _patch_search([_make_node()]),
        TestClient(app) as client,
        patch(
            "fastapi.dependencies.utils.run_in_threadpool",
            side_effect=AssertionError("run_in_threadpool should not be used for /search"),
        ),
    ):
        resp = client.post("/search", json={"query": "AK47 kills"})

    assert resp.status_code == 200


def test_hybrid_search_returns_semantic_and_lexical_results():
    app = create_app(settings=_settings())
    with (
        _patch_search([_make_node(doc_id="doc-sem")]),
        patch(
            "rag_intelligence.lexical_retrieval.lexical_search",
            return_value=[
                _FakeLexicalHit(
                    "train-run-1",
                    "2026-04-11T12:00:00+00:00",
                    "logistic_regression",
                )
            ],
        ),
        TestClient(app) as client,
    ):
        resp = client.post("/search/hybrid", json={"query": "gold phase roc auc", "top_k": 3})

    assert resp.status_code == 200
    data = resp.json()
    assert data["semantic_results"][0]["doc_id"] == "doc-sem"
    assert data["lexical_results"][0]["run_id"] == "train-run-1"
    assert data["lexical_results"][0]["model_name"] == "logistic_regression"
    assert "retrieval_ms" in data


def test_hybrid_search_uses_pipeline_docs_default_run_id():
    app = create_app(settings=_settings())
    with _patch_search([]) as mock_fn, TestClient(app) as client:
        resp = client.post(
            "/search/hybrid",
            json={"query": "silver phase", "include_lexical": False},
        )

    assert resp.status_code == 200
    request = mock_fn.call_args[0][0]
    assert request.embedding_run_id == "pipeline-docs"


def test_hybrid_search_passes_model_filter_to_lexical_search():
    app = create_app(settings=_settings())
    with (
        _patch_search([]),
        patch(
            "rag_intelligence.lexical_retrieval.lexical_search",
            return_value=[],
        ) as lexical_mock,
        TestClient(app) as client,
    ):
        resp = client.post(
            "/search/hybrid",
            json={
                "query": "best logistic regression run",
                "include_semantic": False,
                "model_filter": "logistic_regression",
            },
        )

    assert resp.status_code == 200
    assert lexical_mock.call_args.kwargs["model_filter"] == "logistic_regression"
