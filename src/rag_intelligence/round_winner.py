from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EVENT_TYPE_ROUND_META = "round_meta"
EVENT_TYPE_DAMAGE = "damage"
EVENT_TYPE_KILL = "kill"

_TRUE_VALUES = {"1", "true", "yes", "on"}
_WINNER_CT_VALUES = {"CT", "COUNTERTERRORIST", "COUNTER_TERRORIST", "COUNTER-TERRORIST"}
_WINNER_T_VALUES = {"T", "TERRORIST"}


@dataclass(frozen=True)
class TrainingOutput:
    metrics: pd.DataFrame
    map_segment_metrics: dict[str, pd.DataFrame]
    half_segment_metrics: dict[str, pd.DataFrame]
    X_test: pd.DataFrame
    y_test: pd.Series
    y_probabilities: dict[str, np.ndarray]
    baseline_probabilities: np.ndarray
    trained_models: dict[str, Pipeline]
    feature_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    train_rows: int
    test_rows: int
    train_files: set[str]
    test_files: set[str]


def normalize_winner_side(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in _WINNER_CT_VALUES:
        return "CT"
    if text in _WINNER_T_VALUES:
        return "T"
    return None


def _series_to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False).astype(bool)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin(_TRUE_VALUES)


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_round_level_frame(events_df: pd.DataFrame) -> pd.DataFrame:
    round_context_schema = {
        "file",
        "round_number",
        "map",
        "round_type",
        "winner_side_current",
        "ct_eq_val",
        "t_eq_val",
        "eq_diff",
        "half",
        "overtime_flag",
    }
    if round_context_schema.issubset(events_df.columns):
        round_df = events_df.copy()
        round_df["round_number"] = _to_numeric(round_df["round_number"]).astype("Int64")
        round_df = round_df.dropna(subset=["file", "round_number", "map", "winner_side_current"]).copy()
        round_df["round_number"] = round_df["round_number"].astype(int)
        round_df["winner_side_norm"] = round_df["winner_side_current"].apply(normalize_winner_side)
        round_df = round_df.dropna(subset=["winner_side_norm"]).copy()
        round_df["winner_side_code"] = (round_df["winner_side_norm"] == "CT").astype(int)
        round_df["ct_eq_val"] = _to_numeric(round_df["ct_eq_val"])
        round_df["t_eq_val"] = _to_numeric(round_df["t_eq_val"])
        round_df["eq_diff"] = _to_numeric(round_df["eq_diff"]).fillna(
            round_df["ct_eq_val"] - round_df["t_eq_val"]
        )
        round_df["overtime_flag"] = _to_numeric(round_df["overtime_flag"]).fillna(0).astype(int)
        round_df["half"] = round_df["half"].fillna("H1").astype(str)
        round_df["round_type"] = round_df["round_type"].fillna("unknown").astype(str)
        round_df["map"] = round_df["map"].fillna("unknown").astype(str)
        return round_df.sort_values(["file", "round_number"]).reset_index(drop=True)

    required_columns = {
        "file",
        "round",
        "event_type",
        "map",
        "round_type",
        "winner_side",
        "ct_eq_val",
        "t_eq_val",
        "hp_dmg",
        "arm_dmg",
        "tick",
        "ct_alive",
        "t_alive",
        "is_bomb_planted",
    }
    missing_columns = sorted(required_columns.difference(events_df.columns))
    if missing_columns:
        joined = ", ".join(missing_columns)
        raise ValueError(f"events_df is missing required columns: {joined}")

    working = events_df.copy()
    working["round"] = _to_numeric(working["round"])
    working["tick"] = _to_numeric(working["tick"])
    working["ct_eq_val"] = _to_numeric(working["ct_eq_val"])
    working["t_eq_val"] = _to_numeric(working["t_eq_val"])
    working["ct_alive"] = _to_numeric(working["ct_alive"])
    working["t_alive"] = _to_numeric(working["t_alive"])
    working["hp_dmg"] = _to_numeric(working["hp_dmg"]).fillna(0.0)
    working["arm_dmg"] = _to_numeric(working["arm_dmg"]).fillna(0.0)
    working["is_bomb_planted_bool"] = _series_to_bool(working["is_bomb_planted"])

    meta = working[working["event_type"] == EVENT_TYPE_ROUND_META].copy()
    if meta.empty:
        raise ValueError("No round_meta rows found. Cannot build round-level dataset.")

    meta["winner_side_norm"] = meta["winner_side"].apply(normalize_winner_side)
    meta = meta.dropna(subset=["file", "round", "map", "winner_side_norm"])
    meta["round"] = meta["round"].astype(int)

    meta = (
        meta.sort_values(["file", "round", "tick"], na_position="last")
        .drop_duplicates(subset=["file", "round"], keep="last")
        .copy()
    )
    meta["round_type"] = meta["round_type"].fillna("unknown").astype(str)
    meta["map"] = meta["map"].fillna("unknown").astype(str)

    combat = working[working["event_type"].isin({EVENT_TYPE_DAMAGE, EVENT_TYPE_KILL})].copy()
    combat["round"] = combat["round"].astype("Int64")
    combat = combat.dropna(subset=["file", "round"])
    combat["round"] = combat["round"].astype(int)
    combat["is_kill"] = combat["event_type"].eq(EVENT_TYPE_KILL).astype(int)

    combat_agg = (
        combat.groupby(["file", "round"], as_index=False)
        .agg(
            total_hp_dmg=("hp_dmg", "sum"),
            total_arm_dmg=("arm_dmg", "sum"),
            kills_round=("is_kill", "sum"),
            bomb_planted_any=("is_bomb_planted_bool", "max"),
        )
        .copy()
    )

    end_state = (
        combat.sort_values(["file", "round", "tick"], na_position="last")
        .dropna(subset=["tick"])
        .groupby(["file", "round"], as_index=False)
        .tail(1)[["file", "round", "ct_alive", "t_alive"]]
        .rename(columns={"ct_alive": "ct_alive_end", "t_alive": "t_alive_end"})
    )

    round_df = meta[
        [
            "file",
            "round",
            "map",
            "round_type",
            "winner_side_norm",
            "ct_eq_val",
            "t_eq_val",
        ]
    ].merge(combat_agg, on=["file", "round"], how="left")
    round_df = round_df.merge(end_state, on=["file", "round"], how="left")

    for col in ("total_hp_dmg", "total_arm_dmg", "kills_round", "bomb_planted_any"):
        round_df[col] = round_df[col].fillna(0)

    round_df["bomb_planted_any"] = round_df["bomb_planted_any"].astype(int)
    round_df["eq_diff"] = round_df["ct_eq_val"] - round_df["t_eq_val"]
    round_df["alive_diff_end"] = round_df["ct_alive_end"] - round_df["t_alive_end"]
    round_df["round_number"] = round_df["round"].astype(int)
    round_df["overtime_flag"] = (round_df["round_number"] > 30).astype(int)

    # Half label cycles every 30 rounds (1-15 and 16-30), including overtime cycles.
    half_idx = ((round_df["round_number"] - 1) % 30) < 15
    round_df["half"] = np.where(half_idx, "H1", "H2")

    round_df["winner_side_code"] = (round_df["winner_side_norm"] == "CT").astype(int)
    round_df = round_df.sort_values(["file", "round_number"]).reset_index(drop=True)
    return round_df


def build_supervised_frame(
    round_df: pd.DataFrame,
    *,
    lag_steps: tuple[int, ...] = (1, 2, 3),
    rolling_window: int = 3,
) -> pd.DataFrame:
    if "file" not in round_df.columns or "round_number" not in round_df.columns:
        raise ValueError("round_df must contain file and round_number columns.")

    df = round_df.sort_values(["file", "round_number"]).copy()
    grouped = df.groupby("file", sort=False)

    df["target_winner_side_next"] = grouped["winner_side_norm"].shift(-1)
    df["target_winner_side_next_code"] = grouped["winner_side_code"].shift(-1)

    lag_source_columns: list[str] = []
    for candidate in ("eq_diff", "winner_side_code", "ct_eq_val", "t_eq_val", "kills_round", "total_hp_dmg"):
        if candidate in df.columns:
            lag_source_columns.append(candidate)

    for column in lag_source_columns:
        for lag in lag_steps:
            df[f"{column}_lag{lag}"] = grouped[column].shift(lag)

    rolling_source_columns = [column for column in ("eq_diff", "total_hp_dmg", "kills_round") if column in df.columns]
    for column in rolling_source_columns:
        roll_name = f"{column}_rollmean{rolling_window}"
        rolled = grouped[column].shift(1).rolling(window=rolling_window, min_periods=1).mean()
        df[roll_name] = rolled.reset_index(level=0, drop=True)

    df = df.dropna(subset=["target_winner_side_next", "target_winner_side_next_code"]).copy()
    df["target_winner_side_next_code"] = df["target_winner_side_next_code"].astype(int)
    return df


def run_consistency_checks(supervised_df: pd.DataFrame) -> dict[str, int | bool]:
    sorted_df = supervised_df.sort_values(["file", "round_number"]).copy()
    duplicate_round_keys = int(sorted_df.duplicated(subset=["file", "round_number"]).sum())

    expected_target = sorted_df.groupby("file", sort=False)["winner_side_norm"].shift(-1)
    expected_target = expected_target.dropna()
    actual_target = sorted_df.loc[expected_target.index, "target_winner_side_next"]
    shift_matches = bool((actual_target == expected_target).all())

    expected_lag1 = sorted_df.groupby("file", sort=False)["winner_side_code"].shift(1)
    lag_mask = expected_lag1.notna()
    lag_matches = bool(
        (
            sorted_df.loc[lag_mask, "winner_side_code_lag1"]
            == expected_lag1.loc[lag_mask]
        ).all()
    )

    return {
        "duplicate_round_keys": duplicate_round_keys,
        "target_shift_is_valid": shift_matches,
        "lag1_is_past_only": lag_matches,
    }


def _safe_roc_auc(y_true: pd.Series, y_prob: np.ndarray) -> float:
    if y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _classification_metrics(y_true: pd.Series, y_prob: np.ndarray) -> dict[str, float]:
    clipped = np.clip(y_prob.astype(float), 1e-6, 1 - 1e-6)
    y_pred = (clipped >= 0.5).astype(int)
    return {
        "roc_auc": _safe_roc_auc(y_true, clipped),
        "f1": float(f1_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, clipped)),
    }


def _build_feature_spec(supervised_df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    lag_features = [column for column in supervised_df.columns if "_lag" in column]
    rolling_features = [column for column in supervised_df.columns if "_rollmean" in column]

    numeric_base_candidates = [
        "ct_eq_val",
        "t_eq_val",
        "eq_diff",
        "round_number",
        "overtime_flag",
        "total_hp_dmg",
        "total_arm_dmg",
        "kills_round",
        "bomb_planted_any",
        "ct_alive_end",
        "t_alive_end",
        "alive_diff_end",
    ]
    numeric_features = [
        column for column in numeric_base_candidates if column in supervised_df.columns
    ] + sorted(set(lag_features + rolling_features))

    categorical_base_candidates = ["map", "round_type", "half", "winner_side_norm"]
    categorical_features = [
        column for column in categorical_base_candidates if column in supervised_df.columns
    ]

    feature_columns = numeric_features + categorical_features
    return feature_columns, numeric_features, categorical_features


def _segment_metrics(
    y_true: pd.Series,
    y_prob: np.ndarray,
    segment_values: pd.Series,
    *,
    min_rows: int,
) -> pd.DataFrame:
    segment_df = pd.DataFrame(
        {
            "segment": segment_values.astype(str).to_numpy(),
            "y_true": y_true.to_numpy(),
            "y_prob": y_prob,
        }
    )
    rows: list[dict[str, Any]] = []
    for segment, group in segment_df.groupby("segment"):
        if len(group) < min_rows:
            continue
        y_group = group["y_true"]
        p_group = group["y_prob"].to_numpy()
        metrics = _classification_metrics(y_group, p_group)
        rows.append({"segment": segment, "rows": int(len(group)), **metrics})
    if not rows:
        return pd.DataFrame(columns=["segment", "rows", "roc_auc", "f1", "balanced_accuracy", "log_loss", "brier"])
    return pd.DataFrame(rows).sort_values("rows", ascending=False).reset_index(drop=True)


def train_next_round_winner(
    supervised_df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    min_segment_rows: int = 200,
) -> TrainingOutput:
    feature_columns, numeric_features, categorical_features = _build_feature_spec(supervised_df)

    missing_columns = [column for column in feature_columns if column not in supervised_df.columns]
    if missing_columns:
        joined = ", ".join(missing_columns)
        raise ValueError(f"supervised_df is missing required feature columns: {joined}")

    X = supervised_df[feature_columns].copy()
    y = supervised_df["target_winner_side_next_code"].astype(int)
    groups = supervised_df["file"].astype(str)

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model_builders: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=6,
            max_iter=350,
            random_state=random_state,
        ),
    }

    trained_models: dict[str, Pipeline] = {}
    y_probabilities: dict[str, np.ndarray] = {}
    metric_rows: list[dict[str, Any]] = []
    map_segment_metrics: dict[str, pd.DataFrame] = {}
    half_segment_metrics: dict[str, pd.DataFrame] = {}

    baseline_probabilities = X_test["winner_side_norm"].map({"CT": 1.0, "T": 0.0}).fillna(0.5).to_numpy()
    baseline_metrics = _classification_metrics(y_test, baseline_probabilities)
    metric_rows.append({"model": "baseline_repeat_current_winner", **baseline_metrics})

    for model_name, estimator in model_builders.items():
        pipeline = Pipeline([("prep", preprocess), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_probabilities[model_name] = y_prob
        trained_models[model_name] = pipeline

        metrics = _classification_metrics(y_test, y_prob)
        metric_rows.append({"model": model_name, **metrics})

        map_segment_metrics[model_name] = _segment_metrics(
            y_test,
            y_prob,
            X_test["map"],
            min_rows=min_segment_rows,
        )
        half_segment_metrics[model_name] = _segment_metrics(
            y_test,
            y_prob,
            X_test["half"],
            min_rows=min_segment_rows,
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    map_segment_metrics["baseline_repeat_current_winner"] = _segment_metrics(
        y_test,
        baseline_probabilities,
        X_test["map"],
        min_rows=min_segment_rows,
    )
    half_segment_metrics["baseline_repeat_current_winner"] = _segment_metrics(
        y_test,
        baseline_probabilities,
        X_test["half"],
        min_rows=min_segment_rows,
    )

    return TrainingOutput(
        metrics=metrics_df,
        map_segment_metrics=map_segment_metrics,
        half_segment_metrics=half_segment_metrics,
        X_test=X_test,
        y_test=y_test,
        y_probabilities=y_probabilities,
        baseline_probabilities=baseline_probabilities,
        trained_models=trained_models,
        feature_columns=feature_columns,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        train_rows=len(X_train),
        test_rows=len(X_test),
        train_files=set(groups.iloc[train_idx].astype(str)),
        test_files=set(groups.iloc[test_idx].astype(str)),
    )


def log_training_to_mlflow(
    *,
    training_output: TrainingOutput,
    experiment_name: str,
    tracking_uri: str,
    target_name: str,
    horizon_rounds: int,
    test_size: float,
    random_state: int,
) -> None:
    try:
        import mlflow
        import mlflow.sklearn
    except Exception as exc:
        raise RuntimeError(
            "MLflow is not available. Install optional dependencies with: pip install -e \".[mlops]\""
        ) from exc

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    shared_params = {
        "target": target_name,
        "horizon_rounds": horizon_rounds,
        "split_strategy": "group_shuffle_split_by_file",
        "test_size": test_size,
        "random_state": random_state,
        "feature_count": len(training_output.feature_columns),
        "numeric_feature_count": len(training_output.numeric_features),
        "categorical_feature_count": len(training_output.categorical_features),
        "train_rows": training_output.train_rows,
        "test_rows": training_output.test_rows,
    }

    metrics_lookup = {
        row["model"]: {
            "roc_auc": float(row["roc_auc"]),
            "f1": float(row["f1"]),
            "balanced_accuracy": float(row["balanced_accuracy"]),
            "log_loss": float(row["log_loss"]),
            "brier": float(row["brier"]),
        }
        for _, row in training_output.metrics.iterrows()
    }

    for model_name, metrics in metrics_lookup.items():
        with mlflow.start_run(run_name=model_name):
            for key, value in shared_params.items():
                mlflow.log_param(key, value)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("feature_columns", ",".join(training_output.feature_columns))
            for metric_name, metric_value in metrics.items():
                if np.isfinite(metric_value):
                    mlflow.log_metric(metric_name, metric_value)

            if model_name in training_output.trained_models:
                mlflow.sklearn.log_model(training_output.trained_models[model_name], artifact_path="model")
