from __future__ import annotations

from datetime import UTC, datetime

import pytest

pd = pytest.importorskip("pandas")

lexical_retrieval = pytest.importorskip("rag_intelligence.lexical_retrieval")
settings_module = pytest.importorskip("rag_intelligence.settings")
training_metadata = pytest.importorskip("rag_intelligence.training_metadata")

lexical_search = lexical_retrieval.lexical_search
AppSettings = settings_module.AppSettings
store_training_result = training_metadata.store_training_result


class FakeCursor:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.executed: list[tuple[str, tuple[object, ...] | None]] = []

    def execute(self, sql: str, params=None) -> None:
        self.executed.append((sql, params))

    def fetchall(self):
        return self.rows

    def close(self) -> None:
        return None


class FakeConnection:
    def __init__(self, cursor: FakeCursor):
        self._cursor = cursor
        self.committed = False

    def cursor(self) -> FakeCursor:
        return self._cursor

    def commit(self) -> None:
        self.committed = True

    def close(self) -> None:
        return None


class FakeTrainingOutput:
    def __init__(self) -> None:
        self.metrics = pd.DataFrame(
            [
                {
                    "model": "baseline_repeat_current_winner",
                    "roc_auc": 0.55,
                    "f1": 0.60,
                    "balanced_accuracy": 0.58,
                    "log_loss": 0.69,
                    "brier": 0.24,
                },
                {
                    "model": "logistic_regression",
                    "roc_auc": 0.73,
                    "f1": 0.70,
                    "balanced_accuracy": 0.71,
                    "log_loss": 0.51,
                    "brier": 0.18,
                },
            ]
        )
        self.map_segment_metrics = {
            "baseline_repeat_current_winner": pd.DataFrame(columns=["segment", "rows"]),
            "logistic_regression": pd.DataFrame([{"segment": "de_inferno", "rows": 10}]),
        }
        self.half_segment_metrics = {
            "baseline_repeat_current_winner": pd.DataFrame(columns=["segment", "rows"]),
            "logistic_regression": pd.DataFrame([{"segment": "H1", "rows": 10}]),
        }
        self.feature_columns = ["eq_diff", "winner_side_norm"]
        self.train_rows = 100
        self.test_rows = 20


def _settings() -> AppSettings:
    return AppSettings.from_env({})


def test_store_training_result_uses_supplied_run_id_and_test_size() -> None:
    cursor = FakeCursor()
    conn = FakeConnection(cursor)

    store_training_result(
        training_output=FakeTrainingOutput(),
        feature_importances={
            "logistic_regression": {"eq_diff": 0.3},
            "baseline_repeat_current_winner": {},
        },
        experiment_name="exp-1",
        run_id="train-run-123",
        test_size=0.25,
        settings=_settings(),
        conn_factory=lambda _settings: conn,
    )

    insert_params = [
        params for sql, params in cursor.executed if "INSERT INTO training_runs" in sql
    ]
    assert insert_params
    assert {params[0] for params in insert_params} == {"train-run-123"}
    assert {params[11] for params in insert_params} == {0.25}
    assert conn.committed is True


def test_lexical_search_returns_run_metadata_and_decodes_feature_importances() -> None:
    cursor = FakeCursor(
        rows=[
            (
                "train-run-123",
                "logistic_regression",
                0.73,
                0.70,
                0.71,
                0.51,
                0.18,
                {"eq_diff": 0.3},
                "logistic regression roc_auc 0.73 de_inferno",
                datetime(2026, 4, 11, 12, 0, tzinfo=UTC),
                0.42,
            )
        ]
    )
    conn = FakeConnection(cursor)

    results = lexical_search(
        "logistic regression roc auc",
        settings=_settings(),
        conn_factory=lambda _settings: conn,
    )

    assert len(results) == 1
    assert results[0].run_id == "train-run-123"
    assert results[0].model_name == "logistic_regression"
    assert results[0].feature_importances == {"eq_diff": 0.3}
    assert results[0].created_at == "2026-04-11T12:00:00+00:00"
    assert "created_at DESC" in cursor.executed[0][0]
