"""Tests for model run tracking: ModelRun, Prediction, and RunStore."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest
from pydantic import ValidationError

from ncaa_eval.model.tracking import ModelRun, Prediction, RunStore

# ---------------------------------------------------------------------------
# Task 5.1: ModelRun creation with all fields
# ---------------------------------------------------------------------------


class TestModelRun:
    def test_create_with_all_fields(self) -> None:
        run = ModelRun(
            run_id="abc-123",
            model_type="elo",
            hyperparameters={"k": 32},
            git_hash="abc1234",
            start_year=2015,
            end_year=2025,
            prediction_count=100,
        )
        assert run.run_id == "abc-123"
        assert run.model_type == "elo"
        assert run.hyperparameters == {"k": 32}
        assert run.git_hash == "abc1234"
        assert run.start_year == 2015
        assert run.end_year == 2025
        assert run.prediction_count == 100
        assert run.timestamp is not None

    def test_json_round_trip(self) -> None:
        run = ModelRun(
            run_id="abc-123",
            model_type="xgboost",
            hyperparameters={"n_estimators": 500, "lr": 0.05},
            git_hash="def5678",
            start_year=2016,
            end_year=2024,
            prediction_count=50,
        )
        json_str = run.model_dump_json()
        restored = ModelRun.model_validate_json(json_str)
        assert restored.run_id == run.run_id
        assert restored.model_type == run.model_type
        assert restored.hyperparameters == run.hyperparameters
        assert restored.git_hash == run.git_hash
        assert restored.start_year == run.start_year
        assert restored.end_year == run.end_year
        assert restored.prediction_count == run.prediction_count
        assert restored.timestamp == run.timestamp


# ---------------------------------------------------------------------------
# Task 5.3: Prediction creation and pred_win_prob constraint
# ---------------------------------------------------------------------------


class TestPrediction:
    def test_create_valid(self) -> None:
        pred = Prediction(
            run_id="abc-123",
            game_id="game_1",
            season=2023,
            team_a_id=101,
            team_b_id=202,
            pred_win_prob=0.75,
        )
        assert pred.run_id == "abc-123"
        assert pred.game_id == "game_1"
        assert pred.season == 2023
        assert pred.team_a_id == 101
        assert pred.team_b_id == 202
        assert pred.pred_win_prob == 0.75

    def test_pred_win_prob_lower_bound(self) -> None:
        pred = Prediction(
            run_id="r1",
            game_id="g1",
            season=2020,
            team_a_id=1,
            team_b_id=2,
            pred_win_prob=0.0,
        )
        assert pred.pred_win_prob == 0.0

    def test_pred_win_prob_upper_bound(self) -> None:
        pred = Prediction(
            run_id="r1",
            game_id="g1",
            season=2020,
            team_a_id=1,
            team_b_id=2,
            pred_win_prob=1.0,
        )
        assert pred.pred_win_prob == 1.0

    def test_pred_win_prob_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError, match="pred_win_prob"):
            Prediction(
                run_id="r1",
                game_id="g1",
                season=2020,
                team_a_id=1,
                team_b_id=2,
                pred_win_prob=-0.1,
            )

    def test_pred_win_prob_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError, match="pred_win_prob"):
            Prediction(
                run_id="r1",
                game_id="g1",
                season=2020,
                team_a_id=1,
                team_b_id=2,
                pred_win_prob=1.1,
            )


# ---------------------------------------------------------------------------
# Task 5.4–5.7: RunStore persistence
# ---------------------------------------------------------------------------


def _make_predictions(run_id: str, n: int = 3) -> list[Prediction]:
    return [
        Prediction(
            run_id=run_id,
            game_id=f"game_{i}",
            season=2023,
            team_a_id=100 + i,
            team_b_id=200 + i,
            pred_win_prob=0.5 + i * 0.1,
        )
        for i in range(n)
    ]


class TestRunStore:
    def test_save_and_load_run(self, tmp_path: Path) -> None:
        store = RunStore(base_path=tmp_path)
        run = ModelRun(
            run_id="test-run-1",
            model_type="elo",
            hyperparameters={"k": 32},
            git_hash="abc1234",
            start_year=2015,
            end_year=2025,
            prediction_count=3,
        )
        preds = _make_predictions("test-run-1")
        store.save_run(run, preds)

        loaded = store.load_run("test-run-1")
        assert loaded.run_id == run.run_id
        assert loaded.model_type == run.model_type
        assert loaded.hyperparameters == run.hyperparameters

    def test_load_predictions_returns_dataframe(self, tmp_path: Path) -> None:
        store = RunStore(base_path=tmp_path)
        run = ModelRun(
            run_id="test-run-2",
            model_type="xgboost",
            hyperparameters={},
            git_hash="xyz",
            start_year=2020,
            end_year=2023,
            prediction_count=3,
        )
        preds = _make_predictions("test-run-2")
        store.save_run(run, preds)

        df = store.load_predictions("test-run-2")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "run_id" in df.columns
        assert "game_id" in df.columns
        assert "pred_win_prob" in df.columns

    def test_list_runs_discovers_saved(self, tmp_path: Path) -> None:
        store = RunStore(base_path=tmp_path)
        for i in range(3):
            run = ModelRun(
                run_id=f"run-{i}",
                model_type="elo",
                hyperparameters={},
                git_hash="abc",
                start_year=2015,
                end_year=2025,
                prediction_count=0,
            )
            store.save_run(run, [])

        runs = store.list_runs()
        assert len(runs) == 3
        run_ids = {r.run_id for r in runs}
        assert run_ids == {"run-0", "run-1", "run-2"}

    def test_load_run_missing_raises(self, tmp_path: Path) -> None:
        store = RunStore(base_path=tmp_path)
        with pytest.raises(FileNotFoundError):
            store.load_run("nonexistent-run")

    def test_load_predictions_missing_raises(self, tmp_path: Path) -> None:
        """load_predictions raises FileNotFoundError for unknown run IDs."""
        store = RunStore(base_path=tmp_path)
        with pytest.raises(FileNotFoundError):
            store.load_predictions("nonexistent-run")

    def test_save_creates_run_json(self, tmp_path: Path) -> None:
        store = RunStore(base_path=tmp_path)
        run = ModelRun(
            run_id="file-check",
            model_type="elo",
            hyperparameters={},
            git_hash="abc",
            start_year=2015,
            end_year=2025,
            prediction_count=0,
        )
        store.save_run(run, [])

        run_dir = tmp_path / "runs" / "file-check"
        assert (run_dir / "run.json").exists()
        data = json.loads((run_dir / "run.json").read_text())
        assert data["run_id"] == "file-check"

    def test_save_creates_predictions_parquet(self, tmp_path: Path) -> None:
        store = RunStore(base_path=tmp_path)
        run = ModelRun(
            run_id="parquet-check",
            model_type="elo",
            hyperparameters={},
            git_hash="abc",
            start_year=2015,
            end_year=2025,
            prediction_count=2,
        )
        preds = _make_predictions("parquet-check", n=2)
        store.save_run(run, preds)

        run_dir = tmp_path / "runs" / "parquet-check"
        assert (run_dir / "predictions.parquet").exists()
