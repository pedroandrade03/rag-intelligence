from __future__ import annotations

import json

from rag_intelligence.retrieval import SearchResponse, SearchResult
from rag_intelligence.search_cli import build_search_request, main


def test_build_search_request_from_cli_args() -> None:
    request = build_search_request(
        [
            "--query",
            "dano de ak47",
            "--embedding-run-id",
            "embeddings-smoke",
            "--top-k",
            "3",
            "--event-type",
            "damage",
            "--map",
            "de_inferno",
            "--file",
            "match.dem",
            "--round",
            "12",
        ]
    )

    assert request.query == "dano de ak47"
    assert request.embedding_run_id == "embeddings-smoke"
    assert request.top_k == 3
    assert request.event_type == "damage"
    assert request.map_name == "de_inferno"
    assert request.file_name == "match.dem"
    assert request.round_number == 12


def test_build_search_request_rejects_invalid_round() -> None:
    exit_code = main(
        [
            "--query",
            "dano",
            "--embedding-run-id",
            "embeddings-smoke",
            "--round",
            "abc",
        ]
    )

    assert exit_code == 2


def test_main_prints_json_response(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "rag_intelligence.search_cli.search_events",
        lambda request: SearchResponse(
            query=request.query,
            embedding_run_id=request.embedding_run_id,
            top_k=request.top_k,
            filters=request.filters_payload,
            results_returned=1,
            retrieval_ms=17,
            results=[
                SearchResult(
                    rank=1,
                    score=0.87,
                    doc_id="doc-1",
                    text="Evento damage em map=de_inferno file=match.dem round=12.",
                    event_type="damage",
                    map="de_inferno",
                    file="match.dem",
                    round=12,
                    source_file="events.csv",
                    metadata={"doc_id": "doc-1", "event_type": "damage"},
                )
            ],
        ),
    )

    exit_code = main(
        [
            "--query",
            "dano de ak47",
            "--embedding-run-id",
            "embeddings-smoke",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["query"] == "dano de ak47"
    assert output["embedding_run_id"] == "embeddings-smoke"
    assert output["results_returned"] == 1
    assert output["results"][0]["doc_id"] == "doc-1"
