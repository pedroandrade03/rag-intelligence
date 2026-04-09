from __future__ import annotations

import csv
import json
import logging
import math
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from minio import Minio

from rag_intelligence.config import DocumentSettings
from rag_intelligence.gold import build_gold_events_key
from rag_intelligence.minio_utils import (
    clean_cell,
    ensure_bucket,
    load_minio_object,
    stream_text_lines,
)

LOGGER = logging.getLogger(__name__)

REQUIRED_GOLD_COLUMNS = {"event_type", "file", "round", "map", "source_file"}
EVENT_TYPE_DAMAGE = "damage"
EVENT_TYPE_GRENADE = "grenade"
EVENT_TYPE_KILL = "kill"
EVENT_TYPE_ROUND_META = "round_meta"

# Coordinate bucket size for hotspot zone grid (CS:GO map units)
HOTSPOT_GRID_SIZE = 500


@dataclass(frozen=True)
class DocumentPart:
    object_key: str
    rows: int
    first_doc_id: str
    last_doc_id: str


@dataclass(frozen=True)
class DocumentBuildResult:
    uploaded_objects: list[str]
    artifact_prefix: str
    manifest_key: str
    quality_report_key: str
    files_processed: int
    rows_read: int
    rows_output: int
    quality_summary: dict[str, Any]


def build_document_object_prefix(dataset_prefix: str, run_id: str) -> str:
    normalized_prefix = dataset_prefix.strip("/")
    normalized_run_id = run_id.strip("/")
    return f"{normalized_prefix}/{normalized_run_id}/documents/"


def build_document_part_key(dataset_prefix: str, run_id: str, part_number: int) -> str:
    return f"{build_document_object_prefix(dataset_prefix, run_id)}part-{part_number:05d}.jsonl"


def build_document_manifest_key(dataset_prefix: str, run_id: str) -> str:
    return f"{build_document_object_prefix(dataset_prefix, run_id)}manifest.json"


def build_document_quality_report_key(dataset_prefix: str, run_id: str) -> str:
    return f"{build_document_object_prefix(dataset_prefix, run_id)}quality_report.json"


def build_doc_id(document_run_id: str, doc_index: int) -> str:
    return f"{document_run_id}:{doc_index}"


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _safe_float(value: str | None) -> float | None:
    if not value:
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def _grid_bucket(coord: float | None) -> int | None:
    if coord is None:
        return None
    return math.floor(coord / HOTSPOT_GRID_SIZE) * HOTSPOT_GRID_SIZE


def _top_n(counter: Counter, n: int = 5) -> list[tuple[str, int]]:
    return counter.most_common(n)


def _pct(part: int | float, total: int | float) -> float:
    return round(part / total * 100, 1) if total else 0.0


# ---------------------------------------------------------------------------
# Tier 1 — Weapon-Map Profiles: (map, weapon, event_type in {damage, kill})
# ---------------------------------------------------------------------------


def _new_tier1_acc() -> dict[str, Any]:
    return {
        "count": 0,
        "sum_hp": 0.0,
        "sum_arm": 0.0,
        "max_hp": 0,
        "headshot_count": 0,
        "hitbox_counter": Counter(),
        "att_side_counter": Counter(),
    }


def _update_tier1(
    acc: dict,
    row: dict[str, str | None],
    event_type: str,
) -> None:
    map_val = clean_cell(row.get("map"))
    weapon = clean_cell(row.get("weapon"))
    if not map_val or not weapon:
        return
    key = (map_val, weapon, event_type)
    entry = acc[key]
    entry["count"] += 1
    hp = _safe_float(row.get("hp_dmg"))
    arm = _safe_float(row.get("arm_dmg"))
    if hp is not None:
        entry["sum_hp"] += hp
        if hp > entry["max_hp"]:
            entry["max_hp"] = int(hp)
    if arm is not None:
        entry["sum_arm"] += arm
    hitbox = clean_cell(row.get("hitbox"))
    if hitbox:
        entry["hitbox_counter"][hitbox] += 1
        if hitbox.lower() == "head":
            entry["headshot_count"] += 1
    att_side = clean_cell(row.get("att_side"))
    if att_side:
        entry["att_side_counter"][att_side] += 1


def build_weapon_map_profile_text(
    map_val: str,
    weapon: str,
    event_type: str,
    entry: dict[str, Any],
    total_events_on_map: int,
) -> str:
    count = entry["count"]
    avg_hp = round(entry["sum_hp"] / count, 1) if count else 0.0
    avg_arm = round(entry["sum_arm"] / count, 1) if count else 0.0
    max_hp = entry["max_hp"]
    headshot_rate = _pct(entry["headshot_count"], count)
    top_hitboxes = _top_n(entry["hitbox_counter"], 4)
    hitbox_str = (
        ", ".join(f"{hb} ({_pct(c, count)}%)" for hb, c in top_hitboxes) if top_hitboxes else "N/A"
    )
    side_dist = _top_n(entry["att_side_counter"], 2)
    side_str = ", ".join(f"{s} {_pct(c, count)}%" for s, c in side_dist) if side_dist else "N/A"
    pct_of_map = _pct(count, total_events_on_map)
    if event_type == EVENT_TYPE_DAMAGE:
        return (
            f"Perfil de Arma: {weapon} em {map_val} (eventos de dano). "
            f"Com base em {count:,} instâncias de dano registradas: "
            f"Dano HP médio por acerto: {avg_hp} | Máximo: {max_hp}. "
            f"Dano de armadura médio: {avg_arm}. "
            f"Taxa de headshot: {headshot_rate}% dos acertos. "
            f"Hitboxes mais atingidas: {hitbox_str}. "
            f"Distribuição por lado atacante: {side_str}. "
            f"Esta arma representa {pct_of_map}% dos eventos de dano em {map_val}."
        )
    return (
        f"Perfil de Mortes: {weapon} em {map_val} (eventos de kill). "
        f"Total de mortes registradas: {count:,}. "
        f"Taxa de headshot fatal: {headshot_rate}%. "
        f"Hitboxes fatais mais comuns: {hitbox_str}. "
        f"Distribuição por lado atacante: {side_str}. "
        f"Esta arma representa {pct_of_map}% dos eventos de kill em {map_val}."
    )


# ---------------------------------------------------------------------------
# Tier 2 — Map Overview: (map)
# ---------------------------------------------------------------------------


def _new_tier2_acc() -> dict[str, Any]:
    return {
        "event_type_counter": Counter(),
        "weapon_damage": defaultdict(float),
        "weapon_kills": Counter(),
        "winner_side_counter": Counter(),
        "sum_round_secs": 0.0,
        "round_count": 0,
    }


def _update_tier2(
    acc: dict,
    row: dict[str, str | None],
    event_type: str,
) -> None:
    map_val = clean_cell(row.get("map"))
    if not map_val:
        return
    entry = acc[map_val]
    entry["event_type_counter"][event_type] += 1
    if event_type == EVENT_TYPE_DAMAGE:
        weapon = clean_cell(row.get("weapon"))
        hp = _safe_float(row.get("hp_dmg"))
        if weapon and hp is not None:
            entry["weapon_damage"][weapon] += hp
    elif event_type == EVENT_TYPE_KILL:
        weapon = clean_cell(row.get("weapon"))
        if weapon:
            entry["weapon_kills"][weapon] += 1
    elif event_type == EVENT_TYPE_ROUND_META:
        winner_side = clean_cell(row.get("winner_side"))
        if winner_side:
            entry["winner_side_counter"][winner_side] += 1
        start_s = _safe_float(row.get("start_seconds"))
        end_s = _safe_float(row.get("end_seconds"))
        if start_s is not None and end_s is not None and end_s > start_s:
            entry["sum_round_secs"] += end_s - start_s
            entry["round_count"] += 1


def build_map_overview_text(map_val: str, entry: dict[str, Any]) -> str:
    total = sum(entry["event_type_counter"].values())
    damage_count = entry["event_type_counter"].get(EVENT_TYPE_DAMAGE, 0)
    kill_count = entry["event_type_counter"].get(EVENT_TYPE_KILL, 0)
    grenade_count = entry["event_type_counter"].get(EVENT_TYPE_GRENADE, 0)
    round_meta_count = entry["event_type_counter"].get(EVENT_TYPE_ROUND_META, 0)

    total_weapon_dmg = sum(entry["weapon_damage"].values())
    top_damage_weapons = sorted(entry["weapon_damage"].items(), key=lambda x: -x[1])[:5]
    top_dmg_str = (
        ", ".join(f"{w} ({_pct(v, total_weapon_dmg)}%)" for w, v in top_damage_weapons)
        if top_damage_weapons
        else "N/A"
    )

    total_kills = sum(entry["weapon_kills"].values())
    top_kill_weapons = entry["weapon_kills"].most_common(5)
    top_kill_str = (
        ", ".join(f"{w} ({_pct(c, total_kills)}%)" for w, c in top_kill_weapons)
        if top_kill_weapons
        else "N/A"
    )

    side_totals = sum(entry["winner_side_counter"].values())
    ct_wins = entry["winner_side_counter"].get("CT", 0)
    t_wins = entry["winner_side_counter"].get("T", 0)
    ct_rate = _pct(ct_wins, side_totals)
    t_rate = _pct(t_wins, side_totals)

    avg_round_dur = (
        round(entry["sum_round_secs"] / entry["round_count"], 1) if entry["round_count"] else 0.0
    )
    return (
        f"Visão Geral do Mapa: {map_val}. "
        f"Total de eventos registrados: {total:,} "
        f"(dano: {damage_count:,}, kills: {kill_count:,}, "
        f"granadas: {grenade_count:,}, rounds: {round_meta_count:,}). "
        f"Top 5 armas por dano total causado: {top_dmg_str}. "
        f"Top 5 armas por número de kills: {top_kill_str}. "
        f"Taxa de vitória: CT {ct_rate}%, T {t_rate}%. "
        f"Duração média de round: {avg_round_dur}s. "
        f"Total de rounds registrados: {round_meta_count:,}."
    )


# ---------------------------------------------------------------------------
# Tier 3 — Hotspot Zones: (map, grid_x_bucket, grid_y_bucket)
# ---------------------------------------------------------------------------


def _new_tier3_acc() -> dict[str, Any]:
    return {
        "damage_count": 0,
        "kill_count": 0,
        "weapon_counter": Counter(),
        "att_side_counter": Counter(),
        "grenade_counter": Counter(),
    }


def _update_tier3(
    acc: dict,
    row: dict[str, str | None],
    event_type: str,
) -> None:
    map_val = clean_cell(row.get("map"))
    if not map_val:
        return
    x = _safe_float(row.get("att_pos_x"))
    y = _safe_float(row.get("att_pos_y"))
    if x is None or y is None:
        return
    key = (map_val, _grid_bucket(x), _grid_bucket(y))
    entry = acc[key]
    weapon = clean_cell(row.get("weapon"))
    att_side = clean_cell(row.get("att_side"))
    if event_type == EVENT_TYPE_DAMAGE:
        entry["damage_count"] += 1
        if weapon:
            entry["weapon_counter"][weapon] += 1
        if att_side:
            entry["att_side_counter"][att_side] += 1
    elif event_type == EVENT_TYPE_KILL:
        entry["kill_count"] += 1
        if weapon:
            entry["weapon_counter"][weapon] += 1
        if att_side:
            entry["att_side_counter"][att_side] += 1
    elif event_type == EVENT_TYPE_GRENADE:
        entry["damage_count"] += 1
        nade = clean_cell(row.get("nade"))
        if nade:
            entry["grenade_counter"][nade] += 1


def build_hotspot_zone_text(
    map_val: str,
    gx: int,
    gy: int,
    entry: dict[str, Any],
) -> str:
    total = entry["damage_count"] + entry["kill_count"]
    total_weapons = sum(entry["weapon_counter"].values())
    top_weapons = _top_n(entry["weapon_counter"], 3)
    weapon_str = (
        ", ".join(f"{w} ({_pct(c, total_weapons)}%)" for w, c in top_weapons)
        if top_weapons
        else "N/A"
    )
    side_totals = sum(entry["att_side_counter"].values())
    side_str = (
        ", ".join(
            f"{s} {_pct(c, side_totals)}%" for s, c in entry["att_side_counter"].most_common()
        )
        if side_totals
        else "N/A"
    )
    grenade_str = (
        ", ".join(f"{g}: {c}" for g, c in entry["grenade_counter"].most_common(3))
        if entry["grenade_counter"]
        else "sem granadas"
    )
    gx_end = gx + HOTSPOT_GRID_SIZE
    gy_end = gy + HOTSPOT_GRID_SIZE
    return (
        f"Zona de Combate: {map_val}, setor x:[{gx},{gx_end}] y:[{gy},{gy_end}]. "
        f"Eventos de dano: {entry['damage_count']:,}. "
        f"Eventos de kill: {entry['kill_count']:,}. "
        f"Total de ações registradas: {total:,}. "
        f"Armas mais usadas nesta zona: {weapon_str}. "
        f"Distribuição por lado atacante: {side_str}. "
        f"Granadas usadas: {grenade_str}."
    )


# ---------------------------------------------------------------------------
# Tier 4 — Round-Type Profiles: (map, round_type)  [from round_meta rows]
# ---------------------------------------------------------------------------


def _new_tier4_acc() -> dict[str, Any]:
    return {
        "count": 0,
        "winner_side_counter": Counter(),
        "sum_ct_eq": 0.0,
        "sum_t_eq": 0.0,
        "eq_count": 0,
    }


def _update_tier4(
    acc: dict,
    row: dict[str, str | None],
) -> None:
    map_val = clean_cell(row.get("map"))
    round_type = clean_cell(row.get("round_type"))
    if not map_val or not round_type:
        return
    key = (map_val, round_type)
    entry = acc[key]
    entry["count"] += 1
    winner_side = clean_cell(row.get("winner_side"))
    if winner_side:
        entry["winner_side_counter"][winner_side] += 1
    ct_eq = _safe_float(row.get("ct_eq_val"))
    t_eq = _safe_float(row.get("t_eq_val"))
    if ct_eq is not None and t_eq is not None:
        entry["sum_ct_eq"] += ct_eq
        entry["sum_t_eq"] += t_eq
        entry["eq_count"] += 1


def build_round_type_text(map_val: str, round_type: str, entry: dict[str, Any]) -> str:
    count = entry["count"]
    side_totals = sum(entry["winner_side_counter"].values())
    ct_wins = entry["winner_side_counter"].get("CT", 0)
    t_wins = entry["winner_side_counter"].get("T", 0)
    ct_rate = _pct(ct_wins, side_totals)
    t_rate = _pct(t_wins, side_totals)
    avg_ct_eq = round(entry["sum_ct_eq"] / entry["eq_count"]) if entry["eq_count"] else 0
    avg_t_eq = round(entry["sum_t_eq"] / entry["eq_count"]) if entry["eq_count"] else 0
    return (
        f"Perfil de Round: tipo '{round_type}' em {map_val}. "
        f"Total de rounds deste tipo: {count:,}. "
        f"Taxa de vitória: CT {ct_rate}%, T {t_rate}%. "
        f"Equipamento médio: CT R${avg_ct_eq:,}, T R${avg_t_eq:,}."
    )


# ---------------------------------------------------------------------------
# Tier 5 — Weapon Global Profiles: (weapon) across all maps
# ---------------------------------------------------------------------------


def _new_tier5_acc() -> dict[str, Any]:
    return {
        "count": 0,
        "sum_hp": 0.0,
        "sum_arm": 0.0,
        "max_hp": 0,
        "headshot_count": 0,
        "map_counter": Counter(),
    }


def _update_tier5(
    acc: dict,
    row: dict[str, str | None],
) -> None:
    weapon = clean_cell(row.get("weapon"))
    if not weapon:
        return
    entry = acc[weapon]
    entry["count"] += 1
    hp = _safe_float(row.get("hp_dmg"))
    arm = _safe_float(row.get("arm_dmg"))
    if hp is not None:
        entry["sum_hp"] += hp
        if hp > entry["max_hp"]:
            entry["max_hp"] = int(hp)
    if arm is not None:
        entry["sum_arm"] += arm
    hitbox = clean_cell(row.get("hitbox"))
    if hitbox and hitbox.lower() == "head":
        entry["headshot_count"] += 1
    map_val = clean_cell(row.get("map"))
    if map_val:
        entry["map_counter"][map_val] += 1


def build_weapon_global_text(weapon: str, entry: dict[str, Any]) -> str:
    count = entry["count"]
    avg_hp = round(entry["sum_hp"] / count, 1) if count else 0.0
    avg_arm = round(entry["sum_arm"] / count, 1) if count else 0.0
    max_hp = entry["max_hp"]
    headshot_rate = _pct(entry["headshot_count"], count)
    top_maps = _top_n(entry["map_counter"], 3)
    maps_str = ", ".join(f"{m} ({c:,})" for m, c in top_maps) if top_maps else "N/A"
    return (
        f"Perfil Global de Arma: {weapon} (todos os mapas, eventos de dano). "
        f"Total de acertos registrados: {count:,}. "
        f"Dano HP médio por acerto: {avg_hp} | Máximo: {max_hp}. "
        f"Dano de armadura médio: {avg_arm}. "
        f"Taxa de headshot: {headshot_rate}%. "
        f"Mapas com mais eventos: {maps_str}."
    )


# ---------------------------------------------------------------------------
# Document generation from all accumulators
# ---------------------------------------------------------------------------


def _build_doc(
    document_run_id: str,
    doc_index: int,
    text: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    doc_id = build_doc_id(document_run_id, doc_index)
    return {"doc_id": doc_id, "text": text, "metadata": {"doc_id": doc_id, **metadata}}


def _generate_all_documents(
    tier1_acc: dict,
    tier2_acc: dict,
    tier3_acc: dict,
    tier4_acc: dict,
    tier5_acc: dict,
    document_run_id: str,
    settings: DocumentSettings,
) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    doc_index = 0

    base_meta = {
        "document_run_id": document_run_id,
        "dataset_prefix": settings.document_dataset_prefix,
        "source_run_id": settings.gold_source_run_id,
    }

    # Compute per-map totals for Tier 1 percentage calculation
    map_damage_totals: Counter = Counter()
    map_kill_totals: Counter = Counter()
    for (map_val, _weapon, event_type), entry in tier1_acc.items():
        if event_type == EVENT_TYPE_DAMAGE:
            map_damage_totals[map_val] += entry["count"]
        elif event_type == EVENT_TYPE_KILL:
            map_kill_totals[map_val] += entry["count"]

    # Tier 1
    for (map_val, weapon, event_type), entry in sorted(tier1_acc.items()):
        doc_index += 1
        total_on_map = (
            map_damage_totals[map_val]
            if event_type == EVENT_TYPE_DAMAGE
            else map_kill_totals[map_val]
        )
        text = build_weapon_map_profile_text(map_val, weapon, event_type, entry, total_on_map)
        docs.append(
            _build_doc(
                document_run_id,
                doc_index,
                text,
                {
                    **base_meta,
                    "document_tier": "weapon_map_profile",
                    "map": map_val,
                    "weapon": weapon,
                    "event_type": event_type,
                    "count": entry["count"],
                },
            )
        )

    # Tier 2
    for map_val, entry in sorted(tier2_acc.items()):
        doc_index += 1
        text = build_map_overview_text(map_val, entry)
        docs.append(
            _build_doc(
                document_run_id,
                doc_index,
                text,
                {
                    **base_meta,
                    "document_tier": "map_overview",
                    "map": map_val,
                    "event_type": "overview",
                },
            )
        )

    # Tier 3
    for (map_val, gx, gy), entry in sorted(tier3_acc.items()):
        doc_index += 1
        text = build_hotspot_zone_text(map_val, gx, gy, entry)
        docs.append(
            _build_doc(
                document_run_id,
                doc_index,
                text,
                {
                    **base_meta,
                    "document_tier": "hotspot_zone",
                    "map": map_val,
                    "grid_x": gx,
                    "grid_y": gy,
                    "event_type": "spatial",
                },
            )
        )

    # Tier 4
    for (map_val, round_type), entry in sorted(tier4_acc.items()):
        doc_index += 1
        text = build_round_type_text(map_val, round_type, entry)
        docs.append(
            _build_doc(
                document_run_id,
                doc_index,
                text,
                {
                    **base_meta,
                    "document_tier": "round_type_profile",
                    "map": map_val,
                    "round_type": round_type,
                    "event_type": EVENT_TYPE_ROUND_META,
                    "count": entry["count"],
                },
            )
        )

    # Tier 5
    for weapon, entry in sorted(tier5_acc.items()):
        doc_index += 1
        text = build_weapon_global_text(weapon, entry)
        docs.append(
            _build_doc(
                document_run_id,
                doc_index,
                text,
                {
                    **base_meta,
                    "document_tier": "weapon_global_profile",
                    "weapon": weapon,
                    "event_type": EVENT_TYPE_DAMAGE,
                    "count": entry["count"],
                },
            )
        )

    return docs


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_document_build(
    settings: DocumentSettings,
    *,
    minio_factory=Minio,
) -> DocumentBuildResult:
    client = minio_factory(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )

    ensure_bucket(client, settings.document_bucket)

    source_object = build_gold_events_key(
        settings.gold_dataset_prefix,
        settings.gold_source_run_id,
    )
    artifact_prefix = build_document_object_prefix(
        settings.document_dataset_prefix,
        settings.document_run_id,
    )
    response = load_minio_object(
        client,
        settings.gold_bucket,
        source_object,
        label="Gold object",
    )

    # Accumulators (keyed dicts, stay small regardless of input size)
    tier1_acc: dict = defaultdict(_new_tier1_acc)
    tier2_acc: dict = defaultdict(_new_tier2_acc)
    tier3_acc: dict = defaultdict(_new_tier3_acc)
    tier4_acc: dict = defaultdict(_new_tier4_acc)
    tier5_acc: dict = defaultdict(_new_tier5_acc)

    rows_read = 0
    event_type_counts: Counter = Counter()

    try:
        reader = csv.DictReader(stream_text_lines(response))
        fieldnames = set(reader.fieldnames or [])
        missing_columns = sorted(REQUIRED_GOLD_COLUMNS - fieldnames)
        if missing_columns:
            joined_missing = ", ".join(missing_columns)
            if {
                "file",
                "round_number",
                "winner_side_current",
                "ct_eq_val",
                "t_eq_val",
            }.issubset(fieldnames):
                raise ValueError(
                    "Gold is in ML-first round_context mode. "
                    "Document generation from event-level rows is disabled in this mode."
                )
            raise ValueError(f"Gold events.csv is missing required columns: {joined_missing}")

        for row in reader:
            if settings.document_max_rows is not None and rows_read >= settings.document_max_rows:
                break
            rows_read += 1
            event_type = clean_cell(row.get("event_type")) or "event"
            event_type_counts[event_type] += 1

            _update_tier2(tier2_acc, row, event_type)

            if event_type in (EVENT_TYPE_DAMAGE, EVENT_TYPE_KILL):
                _update_tier1(tier1_acc, row, event_type)
                _update_tier5(tier5_acc, row)
                _update_tier3(tier3_acc, row, event_type)
            elif event_type == EVENT_TYPE_GRENADE:
                _update_tier3(tier3_acc, row, event_type)
            elif event_type == EVENT_TYPE_ROUND_META:
                _update_tier4(tier4_acc, row)
    finally:
        response.close()
        response.release_conn()

    all_docs = _generate_all_documents(
        tier1_acc,
        tier2_acc,
        tier3_acc,
        tier4_acc,
        tier5_acc,
        settings.document_run_id,
        settings,
    )
    rows_output = len(all_docs)

    if rows_output <= 0:
        raise ValueError("No documents were generated from Gold events.csv.")

    LOGGER.info(
        "Aggregated %d Gold rows into %d documents across 5 tiers",
        rows_read,
        rows_output,
    )

    uploaded_objects: list[str] = []
    part_records: list[DocumentPart] = []

    with tempfile.TemporaryDirectory(prefix="document-build-") as temp_dir:
        temp_path = Path(temp_dir)
        part_number = 0
        part_start = 0

        while part_start < len(all_docs):
            part_number += 1
            part_docs = all_docs[part_start : part_start + settings.document_part_size_rows]
            part_start += settings.document_part_size_rows

            part_path = temp_path / f"part-{part_number:05d}.jsonl"
            with part_path.open("w", encoding="utf-8", newline="") as f:
                for doc in part_docs:
                    f.write(json.dumps(doc, ensure_ascii=True) + "\n")

            part_key = build_document_part_key(
                settings.document_dataset_prefix,
                settings.document_run_id,
                part_number,
            )
            client.fput_object(
                bucket_name=settings.document_bucket,
                object_name=part_key,
                file_path=str(part_path),
                content_type="application/x-ndjson",
            )
            uploaded_objects.append(part_key)
            part_records.append(
                DocumentPart(
                    object_key=part_key,
                    rows=len(part_docs),
                    first_doc_id=part_docs[0]["doc_id"],
                    last_doc_id=part_docs[-1]["doc_id"],
                )
            )

        tier_counts = Counter(doc["metadata"]["document_tier"] for doc in all_docs)
        manifest = {
            "generated_at": datetime.now(UTC).isoformat(),
            "source_bucket": settings.gold_bucket,
            "source_object_key": source_object,
            "document_bucket": settings.document_bucket,
            "artifact_prefix": artifact_prefix,
            "gold_source_run_id": settings.gold_source_run_id,
            "document_run_id": settings.document_run_id,
            "dataset_prefix": settings.document_dataset_prefix,
            "part_size_rows": settings.document_part_size_rows,
            "max_rows": settings.document_max_rows,
            "total_documents": rows_output,
            "total_parts": len(part_records),
            "aggregation_strategy": "multi_tier",
            "parts": [asdict(part) for part in part_records],
        }
        manifest_file = temp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest_key = build_document_manifest_key(
            settings.document_dataset_prefix,
            settings.document_run_id,
        )
        client.fput_object(
            bucket_name=settings.document_bucket,
            object_name=manifest_key,
            file_path=str(manifest_file),
            content_type="application/json",
        )
        uploaded_objects.append(manifest_key)

        quality_summary = {
            "rows_read": rows_read,
            "rows_output": rows_output,
            "documents_generated": rows_output,
            "aggregation_strategy": "multi_tier",
            "tier_document_counts": dict(sorted(tier_counts.items())),
            "event_type_counts": dict(sorted(event_type_counts.items())),
            "part_size_rows": settings.document_part_size_rows,
            "max_rows": settings.document_max_rows,
        }
        quality_report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "source_bucket": settings.gold_bucket,
            "source_object_key": source_object,
            "document_bucket": settings.document_bucket,
            "gold_source_run_id": settings.gold_source_run_id,
            "document_run_id": settings.document_run_id,
            "gold_dataset_prefix": settings.gold_dataset_prefix,
            "document_dataset_prefix": settings.document_dataset_prefix,
            "artifact_prefix": artifact_prefix,
            "summary": quality_summary,
            "parts": [asdict(part) for part in part_records],
        }
        quality_report_file = temp_path / "quality_report.json"
        quality_report_file.write_text(json.dumps(quality_report, indent=2), encoding="utf-8")
        quality_report_key = build_document_quality_report_key(
            settings.document_dataset_prefix,
            settings.document_run_id,
        )
        client.fput_object(
            bucket_name=settings.document_bucket,
            object_name=quality_report_key,
            file_path=str(quality_report_file),
            content_type="application/json",
        )
        uploaded_objects.append(quality_report_key)

    return DocumentBuildResult(
        uploaded_objects=uploaded_objects,
        artifact_prefix=artifact_prefix,
        manifest_key=manifest_key,
        quality_report_key=quality_report_key,
        files_processed=len(part_records),
        rows_read=rows_read,
        rows_output=rows_output,
        quality_summary=quality_summary,
    )
