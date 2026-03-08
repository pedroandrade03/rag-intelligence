from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

LOGGER = logging.getLogger(__name__)


def setup_telemetry(
    *,
    service_name: str = "rag-intelligence",
    otel_endpoint: str = "http://localhost:4318",
) -> None:
    """Initialise OpenTelemetry tracing with OTLP export + LlamaIndex instrumentation.

    Lets LlamaIndex own the TracerProvider (it calls ``set_tracer_provider``
    internally), so both LlamaIndex spans and FastAPI spans share the same
    provider without the "Overriding TracerProvider" warning.
    """
    from llama_index.observability.otel import LlamaIndexOpenTelemetry
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    exporter = OTLPSpanExporter(endpoint=f"{otel_endpoint}/v1/traces")

    instrumentor = LlamaIndexOpenTelemetry(
        span_exporter=exporter,
        service_name_or_resource=service_name,
    )
    instrumentor.start_registering()

    LOGGER.info("OpenTelemetry + LlamaIndex tracing enabled (endpoint=%s)", otel_endpoint)


def instrument_fastapi(app: FastAPI) -> None:
    """Auto-instrument a FastAPI app for OpenTelemetry tracing.

    Must be called *after* ``setup_telemetry`` so the global TracerProvider
    is already configured.
    """
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app)
    LOGGER.info("FastAPI OpenTelemetry instrumentation enabled")
