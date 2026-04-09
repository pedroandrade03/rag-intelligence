from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from rag_intelligence.round_winner import (
    build_round_level_frame,
    build_supervised_frame,
    normalize_winner_side,
    run_consistency_checks,
    train_next_round_winner,
)


def _synthetic_events_df():
    rows: list[dict[str, object]] = []
    files = [f"demo_{i}" for i in range(1, 9)]
    rounds_per_file = 10

    for file_idx, file_name in enumerate(files, start=1):
        map_name = "de_mirage" if file_idx % 2 == 0 else "de_inferno"
        for round_number in range(1, rounds_per_file + 1):
            eq_diff = 2500 if (round_number + file_idx) % 2 == 0 else -2500
            ct_eq = 18000 + max(eq_diff, 0)
            t_eq = 18000 + max(-eq_diff, 0)
            winner_side = "CT" if eq_diff > 0 else "T"
            bomb_planted = round_number % 3 == 0

            rows.append(
                {
                    "file": file_name,
                    "round": round_number,
                    "event_type": "round_meta",
                    "map": map_name,
                    "round_type": "pistol" if round_number <= 2 else "gunround",
                    "winner_side": winner_side,
                    "ct_eq_val": ct_eq,
                    "t_eq_val": t_eq,
                    "hp_dmg": 0.0,
                    "arm_dmg": 0.0,
                    "tick": round_number * 1000 + 10,
                    "ct_alive": None,
                    "t_alive": None,
                    "is_bomb_planted": False,
                }
            )
            rows.append(
                {
                    "file": file_name,
                    "round": round_number,
                    "event_type": "damage",
                    "map": map_name,
                    "round_type": "",
                    "winner_side": "",
                    "ct_eq_val": "",
                    "t_eq_val": "",
                    "hp_dmg": 20 + round_number,
                    "arm_dmg": 5 + file_idx,
                    "tick": round_number * 1000 + 100,
                    "ct_alive": 5,
                    "t_alive": 5,
                    "is_bomb_planted": bomb_planted,
                }
            )
            rows.append(
                {
                    "file": file_name,
                    "round": round_number,
                    "event_type": "kill",
                    "map": map_name,
                    "round_type": "",
                    "winner_side": "",
                    "ct_eq_val": "",
                    "t_eq_val": "",
                    "hp_dmg": 100.0,
                    "arm_dmg": 0.0,
                    "tick": round_number * 1000 + 200,
                    "ct_alive": 5 if winner_side == "CT" else 3,
                    "t_alive": 3 if winner_side == "CT" else 5,
                    "is_bomb_planted": bomb_planted,
                }
            )

    return pd.DataFrame(rows), len(files), rounds_per_file


def test_normalize_winner_side_maps_values() -> None:
    assert normalize_winner_side("CT") == "CT"
    assert normalize_winner_side("counterterrorist") == "CT"
    assert normalize_winner_side("T") == "T"
    assert normalize_winner_side("terrorist") == "T"
    assert normalize_winner_side("unknown") is None


def test_build_round_level_frame_aggregates_by_file_round() -> None:
    events_df, file_count, rounds_per_file = _synthetic_events_df()
    round_df = build_round_level_frame(events_df)

    assert len(round_df) == file_count * rounds_per_file
    assert round_df.duplicated(subset=["file", "round_number"]).sum() == 0
    assert {"eq_diff", "alive_diff_end", "half", "overtime_flag"}.issubset(round_df.columns)
    assert round_df["winner_side_norm"].isin({"CT", "T"}).all()


def test_build_supervised_frame_creates_next_round_target() -> None:
    events_df, file_count, rounds_per_file = _synthetic_events_df()
    round_df = build_round_level_frame(events_df)
    supervised_df = build_supervised_frame(round_df)

    assert len(supervised_df) == file_count * (rounds_per_file - 1)
    assert {
        "target_winner_side_next",
        "winner_side_code_lag1",
        "eq_diff_lag1",
        "total_hp_dmg_rollmean3",
    }.issubset(supervised_df.columns)

    demo_1 = supervised_df[supervised_df["file"] == "demo_1"].sort_values("round_number")
    expected = (
        round_df[round_df["file"] == "demo_1"]
        .sort_values("round_number")["winner_side_norm"]
        .shift(-1)
        .dropna()
        .tolist()
    )
    assert demo_1["target_winner_side_next"].tolist() == expected


def test_run_consistency_checks_reports_valid_state() -> None:
    events_df, _, _ = _synthetic_events_df()
    round_df = build_round_level_frame(events_df)
    supervised_df = build_supervised_frame(round_df)
    checks = run_consistency_checks(supervised_df)

    assert checks["duplicate_round_keys"] == 0
    assert checks["target_shift_is_valid"] is True
    assert checks["lag1_is_past_only"] is True


def test_train_next_round_winner_group_split_and_metrics() -> None:
    events_df, _, _ = _synthetic_events_df()
    round_df = build_round_level_frame(events_df)
    supervised_df = build_supervised_frame(round_df)

    output = train_next_round_winner(
        supervised_df,
        test_size=0.25,
        random_state=42,
        min_segment_rows=1,
    )

    models = set(output.metrics["model"].tolist())
    assert "baseline_repeat_current_winner" in models
    assert "logistic_regression" in models
    assert "hist_gradient_boosting" in models

    assert output.train_rows > 0
    assert output.test_rows > 0
    assert output.train_files.isdisjoint(output.test_files)

    for model_name in models:
        assert model_name in output.map_segment_metrics
        assert model_name in output.half_segment_metrics
