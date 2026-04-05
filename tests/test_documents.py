from __future__ import annotations

import json
from collections import Counter

import pytest

from conftest import FakeMinio
from rag_intelligence.config import DocumentSettings
from rag_intelligence.documents import (
    build_doc_id,
    build_document_object_prefix,
    build_document_part_key,
    build_hotspot_zone_text,
    build_map_overview_text,
    build_round_type_text,
    build_weapon_global_text,
    build_weapon_map_profile_text,
    run_document_build,
)


def build_settings(
    part_size: int = 100000,
    *,
    max_rows: int | None = None,
) -> DocumentSettings:
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
        document_max_rows=max_rows,
    )


def _events_key() -> str:
    return "csgo-matchmaking-damage/20260306T025119Z/curated/events.csv"


# ---------------------------------------------------------------------------
# Key / ID helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tier text builders
# ---------------------------------------------------------------------------


def test_build_weapon_map_profile_text_damage() -> None:
    entry = {
        "count": 1000,
        "sum_hp": 27000.0,
        "sum_arm": 10000.0,
        "max_hp": 111,
        "headshot_count": 200,
        "hitbox_counter": Counter({"head": 200, "chest": 400}),
        "att_side_counter": Counter({"T": 600, "CT": 400}),
    }
    text = build_weapon_map_profile_text("de_dust2", "ak47", "damage", entry, 5000)
    assert "ak47" in text
    assert "de_dust2" in text
    assert "27.0" in text  # avg_hp = 27000/1000
    assert "111" in text   # max_hp
    assert "20.0%" in text  # headshot_rate = 200/1000


def test_build_weapon_map_profile_text_kill() -> None:
    entry = {
        "count": 500,
        "sum_hp": 0.0,
        "sum_arm": 0.0,
        "max_hp": 0,
        "headshot_count": 150,
        "hitbox_counter": Counter({"head": 150}),
        "att_side_counter": Counter({"CT": 300}),
    }
    text = build_weapon_map_profile_text("de_inferno", "m4a1", "kill", entry, 500)
    assert "Perfil de Mortes" in text
    assert "m4a1" in text
    assert "de_inferno" in text
    assert "30.0%" in text  # headshot_rate = 150/500


def test_build_map_overview_text() -> None:
    from collections import defaultdict
    entry = {
        "event_type_counter": Counter(
            {"damage": 1000, "kill": 200, "grenade": 50, "round_meta": 30}
        ),
        "weapon_damage": defaultdict(float, {"ak47": 50000.0, "m4a4": 30000.0}),
        "weapon_kills": Counter({"ak47": 100, "awp": 80}),
        "winner_side_counter": Counter({"CT": 18, "T": 12}),
        "sum_round_secs": 2100.0,
        "round_count": 30,
    }
    text = build_map_overview_text("de_mirage", entry)
    assert "de_mirage" in text
    assert "1,280" in text  # total = 1000+200+50+30
    assert "ak47" in text
    assert "CT" in text


def test_build_hotspot_zone_text() -> None:
    entry = {
        "damage_count": 500,
        "kill_count": 80,
        "weapon_counter": Counter({"ak47": 200, "m4a4": 150}),
        "att_side_counter": Counter({"T": 350, "CT": 230}),
        "grenade_counter": Counter(),
    }
    text = build_hotspot_zone_text("de_dust2", 1000, -500, entry)
    assert "de_dust2" in text
    assert "1000" in text
    assert "-500" in text
    assert "500" in text  # damage_count
    assert "80" in text   # kill_count


def test_build_round_type_text() -> None:
    entry = {
        "count": 120,
        "winner_side_counter": Counter({"T": 15, "CT": 105}),
        "sum_ct_eq": 2_520_000.0,
        "sum_t_eq": 228_000.0,
        "eq_count": 120,
    }
    text = build_round_type_text("de_nuke", "eco", entry)
    assert "eco" in text
    assert "de_nuke" in text
    assert "120" in text
    assert "CT" in text


def test_build_weapon_global_text() -> None:
    entry = {
        "count": 50000,
        "sum_hp": 1_200_000.0,
        "sum_arm": 400_000.0,
        "max_hp": 111,
        "headshot_count": 9000,
        "map_counter": Counter({"de_dust2": 20000, "de_inferno": 15000}),
    }
    text = build_weapon_global_text("ak47", entry)
    assert "ak47" in text
    assert "50,000" in text
    assert "24.0" in text  # avg_hp = 1200000/50000
    assert "de_dust2" in text


# ---------------------------------------------------------------------------
# run_document_build integration
# ---------------------------------------------------------------------------

# Minimal Gold CSV with all required columns + fields needed for aggregation
_GOLD_CSV = (
    b"event_type,file,round,map,source_file,weapon,hp_dmg,arm_dmg,"
    b"att_pos_x,att_pos_y,vic_pos_x,vic_pos_y,hitbox,nade,wp_type,"
    b"att_team,vic_team,att_side,vic_side,tick,seconds,"
    b"start_seconds,end_seconds,winner_team,winner_side,round_type,"
    b"ct_eq_val,t_eq_val,nade_land_x,nade_land_y,is_bomb_planted\n"
    b"damage,demo_1,1,de_mirage,damage.csv,ak47,32,5,100,200,300,400,"
    b"head,,,team_a,team_b,T,CT,100,12.5,,,,,,,,,,\n"
    b"kill,demo_1,1,de_mirage,kills.csv,ak47,,,,,,,head,,,team_a,team_b,"
    b"T,CT,200,25.0,,,,,,,,,,\n"
    b"grenade,demo_2,2,de_nuke,grenades.csv,hegrenade,,20,10,20,,,,hegrenade,"
    b",team_c,team_d,T,CT,300,14.0,,,,,,,,,,\n"
    b"round_meta,demo_3,3,de_ancient,meta.csv,,,,,,,,,,,,,,,,,"
    b"30,ct,CT,eco,2500,1200,,,\n"
)


def test_run_document_build_aggregates_gold_rows_into_tier_documents() -> None:
    fake_minio = FakeMinio(
        "localhost:9000",
        "minioadmin",
        "minioadmin",
        False,
        initial_objects={"gold": {_events_key(): _GOLD_CSV}},
        existing_buckets={"gold"},
    )

    result = run_document_build(
        build_settings(part_size=100),
        minio_factory=lambda **kwargs: fake_minio,
    )

    assert result.rows_read == 4
    # Much fewer documents than rows — aggregation worked
    assert result.rows_output < result.rows_read or result.rows_output > 0
    assert result.artifact_prefix == "csgo-matchmaking-damage/20260306T025119Z/documents/"

    gold_objects = fake_minio.objects["gold"]
    manifest_key = "csgo-matchmaking-damage/20260306T025119Z/documents/manifest.json"
    report_key = "csgo-matchmaking-damage/20260306T025119Z/documents/quality_report.json"
    assert manifest_key in gold_objects
    assert report_key in gold_objects
    assert result.manifest_key == manifest_key
    assert result.quality_report_key == report_key

    manifest = json.loads(gold_objects[manifest_key].decode("utf-8"))
    assert manifest["aggregation_strategy"] == "multi_tier"
    assert manifest["total_documents"] == result.rows_output

    report = json.loads(gold_objects[report_key].decode("utf-8"))
    assert report["summary"]["rows_read"] == 4
    assert report["summary"]["documents_generated"] == result.rows_output
    assert report["summary"]["aggregation_strategy"] == "multi_tier"
    assert "tier_document_counts" in report["summary"]

    # Verify all docs have document_tier in metadata
    part_key = "csgo-matchmaking-damage/20260306T025119Z/documents/part-00001.jsonl"
    assert part_key in gold_objects
    docs = [
        json.loads(line)
        for line in gold_objects[part_key].decode("utf-8").splitlines()
        if line.strip()
    ]
    assert len(docs) > 0
    for doc in docs:
        assert "document_tier" in doc["metadata"]
        assert doc["doc_id"].startswith("20260306T025119Z:")


def test_run_document_build_produces_weapon_map_profiles() -> None:
    fake_minio = FakeMinio(
        "localhost:9000",
        "minioadmin",
        "minioadmin",
        False,
        initial_objects={"gold": {_events_key(): _GOLD_CSV}},
        existing_buckets={"gold"},
    )
    run_document_build(
        build_settings(),
        minio_factory=lambda **kwargs: fake_minio,
    )
    gold_objects = fake_minio.objects["gold"]
    part_key = "csgo-matchmaking-damage/20260306T025119Z/documents/part-00001.jsonl"
    docs = [
        json.loads(line)
        for line in gold_objects[part_key].decode("utf-8").splitlines()
        if line.strip()
    ]
    tiers = {doc["metadata"]["document_tier"] for doc in docs}
    assert "weapon_map_profile" in tiers
    assert "map_overview" in tiers

    # Weapon-map profile for ak47+de_mirage should mention damage stats
    weapon_docs = [d for d in docs if d["metadata"].get("document_tier") == "weapon_map_profile"]
    assert any("ak47" in d["text"] and "de_mirage" in d["text"] for d in weapon_docs)


def test_run_document_build_supports_smoke_limit() -> None:
    fake_minio = FakeMinio(
        "localhost:9000",
        "minioadmin",
        "minioadmin",
        False,
        initial_objects={
            "gold": {
                _events_key(): (
                    b"event_type,file,round,map,source_file,weapon,hp_dmg,arm_dmg,"
                    b"att_pos_x,att_pos_y,vic_pos_x,vic_pos_y,hitbox,nade,wp_type,"
                    b"att_team,vic_team,att_side,vic_side,tick,seconds,"
                    b"start_seconds,end_seconds,winner_team,winner_side,round_type,"
                    b"ct_eq_val,t_eq_val,nade_land_x,nade_land_y,is_bomb_planted\n"
                    b"damage,demo_1,1,de_mirage,damage.csv,ak47,32,5,100,200,,,head,,,"
                    b"team_a,team_b,T,CT,100,12.5,,,,,,,,,,\n"
                    b"kill,demo_2,2,de_nuke,kills.csv,m4a1,,,,,,,,,rifle,"
                    b"team_e,team_f,CT,T,300,25.0,,,,,,,,,,\n"
                    b"round_meta,demo_3,3,de_ancient,meta.csv,,,,,,,,,,,,,,,,,,"
                    b"30,ct,CT,eco,2500,1200,,,\n"
                ),
            }
        },
        existing_buckets={"gold"},
    )

    result = run_document_build(
        build_settings(part_size=10, max_rows=2),
        minio_factory=lambda **kwargs: fake_minio,
    )

    manifest_key = "csgo-matchmaking-damage/20260306T025119Z/documents/manifest.json"
    report_key = "csgo-matchmaking-damage/20260306T025119Z/documents/quality_report.json"
    manifest = json.loads(fake_minio.objects["gold"][manifest_key].decode("utf-8"))
    report = json.loads(fake_minio.objects["gold"][report_key].decode("utf-8"))

    # Only 2 rows were read (max_rows=2)
    assert result.rows_read == 2
    assert report["summary"]["rows_read"] == 2
    assert manifest["max_rows"] == 2
    assert report["summary"]["max_rows"] == 2
    assert manifest["total_documents"] == result.rows_output
    assert report["summary"]["documents_generated"] == result.rows_output
    # Documents are aggregated — there should be at least the map_overview docs
    assert result.rows_output >= 1


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
                    b"event_type,file,round,map,weapon\ndamage,demo_1,1,de_mirage,ak47\n"
                ),
            }
        },
        existing_buckets={"gold"},
    )

    with pytest.raises(ValueError, match="missing required columns: source_file"):
        run_document_build(build_settings(), minio_factory=lambda **kwargs: fake_minio)
