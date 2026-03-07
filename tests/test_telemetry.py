from __future__ import annotations

from unittest.mock import MagicMock, patch

from rag_intelligence.logging import _add_otel_context


class TestAddOtelContext:
    def test_adds_trace_and_span_ids_when_span_is_valid(self):
        mock_ctx = MagicMock()
        mock_ctx.is_valid = True
        mock_ctx.trace_id = 0xABCDEF1234567890ABCDEF1234567890
        mock_ctx.span_id = 0x1234567890ABCDEF

        mock_span = MagicMock()
        mock_span.get_span_context.return_value = mock_ctx

        with patch("opentelemetry.trace.get_current_span", return_value=mock_span):
            event_dict: dict[str, object] = {"event": "test"}
            result = _add_otel_context(None, "info", event_dict)

        assert result["trace_id"] == format(mock_ctx.trace_id, "032x")
        assert result["span_id"] == format(mock_ctx.span_id, "016x")

    def test_skips_when_span_is_invalid(self):
        mock_ctx = MagicMock()
        mock_ctx.is_valid = False

        mock_span = MagicMock()
        mock_span.get_span_context.return_value = mock_ctx

        with patch("opentelemetry.trace.get_current_span", return_value=mock_span):
            event_dict: dict[str, object] = {"event": "test"}
            result = _add_otel_context(None, "info", event_dict)

        assert "trace_id" not in result
        assert "span_id" not in result

    def test_no_crash_when_otel_not_available(self):
        with patch.dict("sys.modules", {"opentelemetry": None, "opentelemetry.trace": None}):
            event_dict: dict[str, object] = {"event": "test"}
            result = _add_otel_context(None, "info", event_dict)

        assert result["event"] == "test"
        assert "trace_id" not in result


class TestSetupTelemetry:
    def test_setup_creates_tracer_provider(self):
        from opentelemetry import trace

        from rag_intelligence.telemetry import setup_telemetry

        setup_telemetry(
            service_name="test-service",
            otel_endpoint="http://localhost:4318",
        )

        provider = trace.get_tracer_provider()
        assert provider is not None

    def test_instrument_fastapi_does_not_crash(self):
        from fastapi import FastAPI

        from rag_intelligence.telemetry import instrument_fastapi

        app = FastAPI()
        instrument_fastapi(app)
