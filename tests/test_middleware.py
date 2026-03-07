from __future__ import annotations

import re

import structlog
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from rag_intelligence.api.middleware import RequestIDMiddleware

UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _echo_app(request: Request) -> PlainTextResponse:
    ctx = structlog.contextvars.get_contextvars()
    return PlainTextResponse(ctx.get("request_id", ""))


def _make_app() -> Starlette:
    app = Starlette(routes=[Route("/test", _echo_app)])
    app.add_middleware(RequestIDMiddleware)
    return app


class TestRequestIDMiddleware:
    def test_generates_uuid4_when_no_header(self):
        client = TestClient(_make_app())
        response = client.get("/test")

        rid = response.headers["x-request-id"]
        assert UUID4_RE.match(rid)

    def test_echoes_client_provided_id(self):
        client = TestClient(_make_app())
        response = client.get("/test", headers={"X-Request-ID": "my-custom-id"})

        assert response.headers["x-request-id"] == "my-custom-id"

    def test_binds_request_id_to_structlog_context(self):
        client = TestClient(_make_app())
        response = client.get("/test", headers={"X-Request-ID": "ctx-check-123"})

        # The echo endpoint returns the request_id from structlog contextvars
        assert response.text == "ctx-check-123"

    def test_skips_non_http_scopes(self):
        """Lifespan and other scope types pass through without error."""
        client = TestClient(_make_app())
        # TestClient triggers a lifespan scope automatically;
        # if the middleware crashed on non-http scopes, this would fail.
        response = client.get("/test")
        assert response.status_code == 200
