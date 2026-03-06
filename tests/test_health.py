from __future__ import annotations

from fastapi.testclient import TestClient

from rag_intelligence.api.main import create_app
from rag_intelligence.providers import ProviderRegistry
from rag_intelligence.settings import AppSettings


def test_health_returns_ok():
    app = create_app()

    # Override lifespan by setting state directly
    settings = AppSettings.from_env({})
    app.state.settings = settings
    app.state.registry = ProviderRegistry(settings)

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
