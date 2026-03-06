from __future__ import annotations

from fastapi.testclient import TestClient

from rag_intelligence.api.main import create_app
from rag_intelligence.settings import AppSettings


def test_health_returns_ok():
    settings = AppSettings.from_env({})
    app = create_app(settings=settings)

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_response_includes_request_id():
    settings = AppSettings.from_env({})
    app = create_app(settings=settings)

    client = TestClient(app)
    response = client.get("/health")

    assert "x-request-id" in response.headers
    # UUID4 format: 8-4-4-4-12 hex chars
    assert len(response.headers["x-request-id"]) == 36


def test_request_id_is_echoed_back():
    settings = AppSettings.from_env({})
    app = create_app(settings=settings)

    client = TestClient(app)
    custom_id = "test-correlation-id-123"
    response = client.get("/health", headers={"X-Request-ID": custom_id})

    assert response.headers["x-request-id"] == custom_id
