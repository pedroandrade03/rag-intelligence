from __future__ import annotations

import json

import pytest

from conftest import FakeMinio
from rag_intelligence.config import DocumentSettings
from rag_intelligence.documents import (
    build_doc_id,
    build_document_metadata,
    build_document_object_prefix,
    build_document_part_key,
    build_document_text,
    run_document_build,
)


def build_settings(part_size: int = 100000) -> DocumentSettings:
    return DocumentSettings(
        minio_endpoint="localhost:9000",
        minio_access_key="minioadmin",
        minio_secret_key="minioadmin",
        minio_secure=False,
        gold_bucket="gold",
        document_bucket="gold",
        gold_dataset_prefix="csgo-matchmaking-damage",
        document_dataset_prefix="csgo-matchmaking-damage",
        gold_source_run_id="20260306T025119Z",
        document_run_id="20260306T025119Z",
        document_part_size_rows=part_size,
    )


def _events_key() -> str:
    return "csgo-matchmaking-damage/20260306T025119Z/curated/events.csv"


def test_build_document_keys_use_documents_prefix() -> None:
    assert (
        build_document_object_prefix("csgo-matchmaking-damage", "20260306T025119Z")
        == "csgo-matchmaking-damage/20260306T025119Z/documents/"
    )
    assert (
        build_document_part_key("csgo-matchmaking-damage", "20260306T025119Z", 1)
        == "csgo-matchmaking-damage/20260306T025119Z/documents/part-00001.jsonl"
    )


def test_build_doc_id_uses_stable_line_number() -> None:
    assert build_doc_id("20260306T025119Z", 42) == "20260306T025119Z:42"


@pytest.mark.parametrize(
    ("row", "expected_tokens"),
    [
        (
            {
                "event_type": "damage",
                "map": "de_mirage",
                "file": "demo_1",
                "round": "1",
                "weapon": "ak47",
                "hp_dmg": "32",
                "source_file": "damage.csv",
            },
            ("Evento damage", "weapon=ak47", "hp_dmg=32"),
        ),
        (
            {
                "event_type": "grenade",
                "map": "de_nuke",
                "file": "demo_2",
                "round": "3",
                "weapon": "hegrenade",
                "nade_land_x": "500",
                "source_file": "grenades.csv",
            },
            ("Evento grenade", "weapon=hegrenade", "nade_land_x=500"),
        ),
        (
            {
                "event_type": "kill",
                "map": "de_inferno",
                "file": "demo_3",
                "round": "4",
                "weapon": "m4a1",
                "wp_type": "rifle",
                "source_file": "kills.csv",
            },
            ("Evento kill", "weapon=m4a1", "wp_type=rifle"),
        ),
        (
            {
                "event_type": "round_meta",
                "map": "de_ancient",
                "file": "demo_4",
                "round": "5",
                "winner_team": "ct",
                "round_type": "eco",
                "source_file": "metadata.csv",
            },
            ("Resumo round_meta", "winner_team=ct", "round_type=eco"),
        ),
    ],
)
def test_build_document_text_uses_event_templates(
    row: dict[str, str],
    expected_tokens: tuple[str, ...],
) -> None:
    text = build_document_text(row)
    for token in expected_tokens:
        assert token in text


def test_build_document_metadata_coerces_numeric_and_boolean_fields() -> None:
    settings = build_settings()
    metadata = build_document_metadata(
        {
            "event_type": "kill",
            "file": "demo_1",
            "round": "3",
            "map": "de_mirage",
            "source_file": "kills.csv",
            "tick": "123",
            "seconds": "12.5",
            "is_bomb_planted": "false",
            "weapon": "ak47",
        },
        settings,
        doc_id="20260306T025119Z:1",
        source_line_number=2,
    )

    assert metadata["round"] == 3
    assert metadata["tick"] == 123
    assert metadata["seconds"] == 12.5
    assert metadata["is_bomb_planted"] is False
    assert metadata["source_line_number"] == 2
    assert metadata["doc_id"] == "20260306T025119Z:1"


def test_run_document_build_streams_gold_csv_and_partitions_output() -> None:
    fake_minio = FakeMinio(
        "localhost:9000",
        "minioadmin",
        "minioadmin",
        False,
        initial_objects={
            "gold": {
                _events_key(): (
                    b"event_type,file,round,map,source_file,weapon,hp_dmg,arm_dmg,att_pos_x,"
                    b"att_pos_y,vic_pos_x,vic_pos_y,hitbox,nade,wp_type,att_team,vic_team,"
                    b"att_side,vic_side,tick,seconds,start_seconds,end_seconds,winner_team,"
                    b"winner_side,round_type,ct_eq_val,t_eq_val,nade_land_x,nade_land_y,"
                    b"is_bomb_planted\n"
                    b"damage,demo_1,1,de_mirage,damage.csv,ak47,32,,100,200,300,400,head,,,"
                    b"team_a,team_b,t,ct,100,12.5,,,,,,,,,\n"
                    b"grenade,demo_2,2,de_nuke,grenades.csv,hegrenade,,20,10,20,,,," 
                    b"hegrenade,,team_c,team_d,t,ct,200,14.0,,,,,,500,700,\n"
                    b"kill,demo_3,3,de_inferno,kills.csv,m4a1,,,,,,,,,rifle,team_e,team_f,"
                    b"ct,t,300,25.0,,,,,,,true\n"
                    b"round_meta,demo_4,4,de_ancient,metadata.csv,,,,,,,,,,,,,,,,0,45,ct,ct,"
                    b"eco,2500,1200,,,\n"
                    b"event,demo_5,5,de_train,events.csv,famas,10,2,1,2,3,4,,,,team_x,"
                    b"team_y,ct,t,400,30.0,,,,,,,,,\n"
                ),
            }
        },
        existing_buckets={"gold"},
    )

    result = run_document_build(
        build_settings(part_size=2),
        minio_factory=lambda **kwargs: fake_minio,
    )

    assert result.files_processed == 3
    assert result.rows_read == 5
    assert result.rows_output == 5
    assert result.artifact_prefix == "csgo-matchmaking-damage/20260306T025119Z/documents/"

    gold_objects = fake_minio.objects["gold"]
    part_keys = [
        "csgo-matchmaking-damage/20260306T025119Z/documents/part-00001.jsonl",
        "csgo-matchmaking-damage/20260306T025119Z/documents/part-00002.jsonl",
        "csgo-matchmaking-damage/20260306T025119Z/documents/part-00003.jsonl",
    ]
    for part_key in part_keys:
        assert part_key in gold_objects

    manifest_key = "csgo-matchmaking-damage/20260306T025119Z/documents/manifest.json"
    report_key = "csgo-matchmaking-damage/20260306T025119Z/documents/quality_report.json"
    assert manifest_key in gold_objects
    assert report_key in gold_objects
    assert result.manifest_key == manifest_key
    assert result.quality_report_key == report_key

    first_part = [
        json.loads(line)
        for line in gold_objects[part_keys[0]].decode("utf-8").splitlines()
    ]
    assert len(first_part) == 2
    assert first_part[0]["doc_id"] == "20260306T025119Z:1"
    assert "Evento damage" in first_part[0]["text"]
    assert first_part[0]["metadata"]["round"] == 1
    assert first_part[0]["metadata"]["seconds"] == 12.5

    manifest = json.loads(gold_objects[manifest_key].decode("utf-8"))
    assert manifest["total_documents"] == 5
    assert manifest["total_parts"] == 3
    assert manifest["parts"][0]["rows"] == 2

    report = json.loads(gold_objects[report_key].decode("utf-8"))
    assert report["summary"]["files_processed"] == 3
    assert report["summary"]["rows_read"] == 5
    assert report["summary"]["rows_output"] == 5
    assert report["summary"]["documents_generated"] == 5
    assert report["summary"]["event_type_counts"] == {
        "damage": 1,
        "event": 1,
        "grenade": 1,
        "kill": 1,
        "round_meta": 1,
    }


def test_run_document_build_fails_when_gold_object_is_missing() -> None:
    fake_minio = FakeMinio(
        "localhost:9000",
        "minioadmin",
        "minioadmin",
        False,
        initial_objects={"gold": {}},
        existing_buckets={"gold"},
    )

    with pytest.raises(FileNotFoundError, match="Gold object not found"):
        run_document_build(build_settings(), minio_factory=lambda **kwargs: fake_minio)


def test_run_document_build_fails_when_required_columns_are_missing() -> None:
    fake_minio = FakeMinio(
        "localhost:9000",
        "minioadmin",
        "minioadmin",
        False,
        initial_objects={
            "gold": {
                _events_key(): (
                    b"event_type,file,round,map,weapon\n"
                    b"damage,demo_1,1,de_mirage,ak47\n"
                ),
            }
        },
        existing_buckets={"gold"},
    )

    with pytest.raises(ValueError, match="missing required columns: source_file"):
        run_document_build(build_settings(), minio_factory=lambda **kwargs: fake_minio)
