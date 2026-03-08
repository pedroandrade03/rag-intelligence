from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

from rag_intelligence.db import (
    build_pgvector_data_table_name,
    create_vector_store,
    pgvector_data_table_exists,
)
from rag_intelligence.providers import ProviderRegistry
from rag_intelligence.settings import AppSettings


@dataclass(frozen=True)
class SearchRequest:
    query: str
    embedding_run_id: str
    top_k: int = 5
    event_type: str | None = None
    map_name: str | None = None
    file_name: str | None = None
    round_number: int | None = None

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query cannot be empty")
        if not self.embedding_run_id.strip():
            raise ValueError("embedding_run_id cannot be empty")
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")

    @property
    def filters_payload(self) -> dict[str, str | int]:
        filters: dict[str, str | int] = {}
        if self.event_type is not None:
            filters["event_type"] = self.event_type
        if self.map_name is not None:
            filters["map"] = self.map_name
        if self.file_name is not None:
            filters["file"] = self.file_name
        if self.round_number is not None:
            filters["round"] = self.round_number
        return filters


@dataclass(frozen=True)
class SearchResult:
    rank: int
    score: float | None
    doc_id: str
    text: str
    event_type: str | None
    map: str | None
    file: str | None
    round: int | str | None
    source_file: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SearchResponse:
    query: str
    embedding_run_id: str
    top_k: int
    filters: dict[str, str | int]
    results_returned: int
    retrieval_ms: int
    results: list[SearchResult]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_metadata_filters(request: SearchRequest) -> MetadataFilters:
    filters: list[MetadataFilter | MetadataFilters] = [
        MetadataFilter(key="embedding_run_id", value=request.embedding_run_id),
    ]
    if request.event_type is not None:
        filters.append(MetadataFilter(key="event_type", value=request.event_type))
    if request.map_name is not None:
        filters.append(MetadataFilter(key="map", value=request.map_name))
    if request.file_name is not None:
        filters.append(MetadataFilter(key="file", value=request.file_name))
    if request.round_number is not None:
        filters.append(MetadataFilter(key="round", value=request.round_number))
    return MetadataFilters(filters=filters)


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    has_doc_id = "doc_id" in metadata and metadata["doc_id"] not in {None, ""}
    for key, value in metadata.items():
        if key.startswith("_"):
            continue
        if has_doc_id and key in {"document_id", "ref_doc_id"}:
            continue
        sanitized[key] = value
    return sanitized


def build_search_result(item: NodeWithScore, *, rank: int) -> SearchResult:
    node = item.node
    metadata = sanitize_metadata(dict(getattr(node, "metadata", {}) or {}))
    doc_id = str(
        metadata.get("doc_id")
        or getattr(node, "doc_id", "")
        or getattr(node, "node_id", "")
        or getattr(node, "id_", "")
    )
    text = str(getattr(node, "text", "") or node.get_content())
    return SearchResult(
        rank=rank,
        score=float(item.score) if item.score is not None else None,
        doc_id=doc_id,
        text=text,
        event_type=_optional_str(metadata.get("event_type")),
        map=_optional_str(metadata.get("map")),
        file=_optional_str(metadata.get("file")),
        round=_round_value(metadata.get("round")),
        source_file=_optional_str(metadata.get("source_file")),
        metadata=metadata,
    )


def search_events(
    request: SearchRequest,
    *,
    app_settings_factory=AppSettings.from_env,
    registry_factory=ProviderRegistry,
    vector_store_factory=create_vector_store,
    vector_table_exists_fn=pgvector_data_table_exists,
    index_factory=VectorStoreIndex.from_vector_store,
) -> SearchResponse:
    app_settings = app_settings_factory()
    if not vector_table_exists_fn(app_settings):
        raise RuntimeError(
            "Vector table not found: "
            f"public.{build_pgvector_data_table_name(app_settings.pg_table_name)}"
        )

    registry = registry_factory(app_settings)
    embed_model = registry.get_embed_model()
    vector_store = vector_store_factory(app_settings, perform_setup=False)
    index = index_factory(vector_store, embed_model=embed_model)
    retriever = index.as_retriever(
        similarity_top_k=request.top_k,
        filters=build_metadata_filters(request),
    )

    started_at = perf_counter()
    retrieved_nodes = retriever.retrieve(request.query)
    retrieval_ms = int((perf_counter() - started_at) * 1000)

    results = [
        build_search_result(item, rank=rank)
        for rank, item in enumerate(retrieved_nodes, start=1)
    ]
    return SearchResponse(
        query=request.query,
        embedding_run_id=request.embedding_run_id,
        top_k=request.top_k,
        filters=request.filters_payload,
        results_returned=len(results),
        retrieval_ms=retrieval_ms,
        results=results,
    )


def _optional_str(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    return str(value)


def _round_value(value: Any) -> int | str | None:
    if value in {None, ""}:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else str(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped.lstrip("-").isdigit():
            return int(stripped)
        return stripped or None
    return str(value)
