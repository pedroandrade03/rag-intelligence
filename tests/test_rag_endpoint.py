from __future__ import annotations

import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from rag_intelligence.api.main import create_app
from rag_intelligence.rag import RAGRequest, RAGResponse
from rag_intelligence.settings import AppSettings


def _settings() -> AppSettings:
    return AppSettings.from_env({"DEFAULT_EMBEDDING_RUN_ID": "run-test"})


def _fake_rag_response(request: RAGRequest, **_kwargs) -> RAGResponse:
    return RAGResponse(
        query=request.query,
        answer="The AK-47 is the most popular weapon.",
        sources=[
            {
                "rank": 1,
                "score": 0.9,
                "doc_id": "doc-1",
                "text": "AK-47 kill",
                "event_type": "kill",
                "map": "de_dust2",
            },
        ],
        retrieval_ms=10,
        generation_ms=50,
    )


async def _fake_rag_stream(request: RAGRequest, **_kwargs):
    yield f"event: sources\ndata: {json.dumps({'sources': [{'rank': 1}], 'retrieval_ms': 10})}\n\n"
    yield f"event: chunk\ndata: {json.dumps({'token': 'hello'})}\n\n"
    yield f"event: chunk\ndata: {json.dumps({'token': ' world'})}\n\n"
    yield f"event: done\ndata: {json.dumps({'answer': 'hello world', 'generation_ms': 50})}\n\n"


def test_rag_query_non_streaming():
    app = create_app(settings=_settings())
    with (
        patch("rag_intelligence.api.routes.rag.rag_query", side_effect=_fake_rag_response),
        TestClient(app) as client,
    ):
        resp = client.post(
            "/rag/query",
            json={"query": "best weapon", "stream": False},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "The AK-47 is the most popular weapon."
    assert data["query"] == "best weapon"
    assert len(data["sources"]) == 1
    assert data["retrieval_ms"] == 10
    assert data["generation_ms"] == 50


def test_rag_query_streaming():
    app = create_app(settings=_settings())
    with (
        patch("rag_intelligence.api.routes.rag.rag_query_stream", side_effect=_fake_rag_stream),
        TestClient(app) as client,
    ):
        resp = client.post(
            "/rag/query",
            json={"query": "best weapon", "stream": True},
        )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert "event: sources" in body
    assert "event: chunk" in body
    assert "event: done" in body
    assert '"hello"' in body


def test_rag_query_422_when_no_embedding_run_id():
    settings = AppSettings.from_env({})
    app = create_app(settings=settings)
    with TestClient(app) as client:
        resp = client.post("/rag/query", json={"query": "test"})
    assert resp.status_code == 422


def test_rag_query_422_for_missing_query():
    app = create_app(settings=_settings())
    with TestClient(app) as client:
        resp = client.post("/rag/query", json={"stream": False})
    assert resp.status_code == 422


def test_rag_query_uses_default_embedding_run_id():
    app = create_app(settings=_settings())
    with (
        patch(
            "rag_intelligence.api.routes.rag.rag_query",
            side_effect=_fake_rag_response,
        ) as mock_fn,
        TestClient(app) as client,
    ):
        resp = client.post(
            "/rag/query",
            json={"query": "test", "stream": False},
        )

    assert resp.status_code == 200
    request = mock_fn.call_args[0][0]
    assert request.embedding_run_id == "run-test"
