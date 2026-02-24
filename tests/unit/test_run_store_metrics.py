"""Tests for RunStore metric persistence: save_metrics, load_metrics, load_all_summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest

from ncaa_eval.model.tracking import ModelRun, RunStore


def _make_run(run_id: str = "run-1", start_year: int = 2015, end_year: int = 2025) -> ModelRun:
    """Create a minimal ModelRun for testing."""
    return ModelRun(
        run_id=run_id,
        model_type="elo",
        hyperparameters={"k": 32},
        git_hash="abc1234",
        start_year=start_year,
        end_year=end_year,
        prediction_count=0,
    )


def _make_summary() -> pd.DataFrame:
    """Create a summary DataFrame mimicking BacktestResult.summary."""
    return pd.DataFrame(
        {
            "log_loss": [0.55, 0.52],
            "brier_score": [0.20, 0.19],
            "roc_auc": [0.73, 0.76],
            "ece": [0.035, 0.028],
            "elapsed_seconds": [1.2, 1.1],
        },
        index=pd.Index([2023, 2024], name="year"),
    )


class TestSaveMetrics:
    def test_creates_summary_parquet(self, tmp_path: Path) -> None:
        store = RunStore(tmp_path)
        run = _make_run()
        store.save_run(run, [])
        summary = _make_summary()

        store.save_metrics(run.run_id, summary)

        assert (tmp_path / "runs" / run.run_id / "summary.parquet").exists()

    def test_raises_for_missing_run(self, tmp_path: Path) -> None:
        store = RunStore(tmp_path)

        with pytest.raises(FileNotFoundError, match="Run directory not found"):
            store.save_metrics("nonexistent-run", _make_summary())


class TestLoadMetrics:
    def test_round_trip(self, tmp_path: Path) -> None:
        store = RunStore(tmp_path)
        run = _make_run()
        store.save_run(run, [])
        original = _make_summary()

        store.save_metrics(run.run_id, original)
        loaded = store.load_metrics(run.run_id)

        assert loaded is not None
        pd.testing.assert_frame_equal(loaded, original)

    def test_returns_none_for_legacy_run(self, tmp_path: Path) -> None:
        store = RunStore(tmp_path)
        run = _make_run()
        store.save_run(run, [])

        result = store.load_metrics(run.run_id)
        assert result is None


class TestLoadAllSummaries:
    def test_concatenates_multiple_runs(self, tmp_path: Path) -> None:
        store = RunStore(tmp_path)
        run1 = _make_run("run-1")
        run2 = _make_run("run-2")
        store.save_run(run1, [])
        store.save_run(run2, [])

        summary1 = _make_summary()
        summary2 = pd.DataFrame(
            {
                "log_loss": [0.60],
                "brier_score": [0.22],
                "roc_auc": [0.70],
                "ece": [0.04],
                "elapsed_seconds": [1.5],
            },
            index=pd.Index([2024], name="year"),
        )
        store.save_metrics("run-1", summary1)
        store.save_metrics("run-2", summary2)

        result = store.load_all_summaries()

        assert "run_id" in result.columns
        assert set(result["run_id"].unique()) == {"run-1", "run-2"}
        assert len(result) == 3  # 2 rows from run-1 + 1 from run-2

    def test_skips_runs_without_summaries(self, tmp_path: Path) -> None:
        store = RunStore(tmp_path)
        run1 = _make_run("run-1")
        run2 = _make_run("run-2")
        store.save_run(run1, [])
        store.save_run(run2, [])
        store.save_metrics("run-1", _make_summary())

        result = store.load_all_summaries()

        assert set(result["run_id"].unique()) == {"run-1"}

    def test_returns_empty_dataframe_when_no_runs(self, tmp_path: Path) -> None:
        store = RunStore(tmp_path)
        result = store.load_all_summaries()

        assert isinstance(result, pd.DataFrame)
        assert result.empty
        expected_cols = {"run_id", "year", "log_loss", "brier_score", "roc_auc", "ece", "elapsed_seconds"}
        assert set(result.columns) == expected_cols

    def test_returns_empty_dataframe_when_all_legacy(self, tmp_path: Path) -> None:
        store = RunStore(tmp_path)
        store.save_run(_make_run("run-1"), [])
        store.save_run(_make_run("run-2"), [])

        result = store.load_all_summaries()

        assert isinstance(result, pd.DataFrame)
        assert result.empty
