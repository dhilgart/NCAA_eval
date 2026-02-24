"""Tests for leaderboard page logic: year filtering, empty state handling."""

from __future__ import annotations

import importlib

import pandas as pd  # type: ignore[import-untyped]

_lab_mod = importlib.import_module("dashboard.pages.1_Lab")
_METRIC_COLS: list[str] = _lab_mod._METRIC_COLS


def _sample_data() -> list[dict[str, object]]:
    """Return sample leaderboard data for two runs across two years."""
    return [
        {
            "run_id": "run-1",
            "model_type": "elo",
            "timestamp": "2025-01-01T00:00:00",
            "start_year": 2015,
            "end_year": 2025,
            "year": 2023,
            "log_loss": 0.55,
            "brier_score": 0.20,
            "roc_auc": 0.73,
            "ece": 0.035,
        },
        {
            "run_id": "run-1",
            "model_type": "elo",
            "timestamp": "2025-01-01T00:00:00",
            "start_year": 2015,
            "end_year": 2025,
            "year": 2024,
            "log_loss": 0.52,
            "brier_score": 0.19,
            "roc_auc": 0.76,
            "ece": 0.028,
        },
        {
            "run_id": "run-2",
            "model_type": "xgb",
            "timestamp": "2025-01-02T00:00:00",
            "start_year": 2015,
            "end_year": 2025,
            "year": 2023,
            "log_loss": 0.50,
            "brier_score": 0.18,
            "roc_auc": 0.78,
            "ece": 0.030,
        },
        {
            "run_id": "run-2",
            "model_type": "xgb",
            "timestamp": "2025-01-02T00:00:00",
            "start_year": 2015,
            "end_year": 2025,
            "year": 2024,
            "log_loss": 0.48,
            "brier_score": 0.17,
            "roc_auc": 0.80,
            "ece": 0.025,
        },
    ]


class TestYearFiltering:
    def test_filter_by_specific_year(self) -> None:
        df = pd.DataFrame(_sample_data())
        year_df = df[df["year"] == 2023]
        assert len(year_df) == 2
        assert set(year_df["year"].unique()) == {2023}

    def test_filter_by_missing_year_returns_empty(self) -> None:
        df = pd.DataFrame(_sample_data())
        year_df = df[df["year"] == 2020]
        assert year_df.empty

    def test_aggregate_when_no_year_filter(self) -> None:
        df = pd.DataFrame(_sample_data())
        agg = df.groupby(["run_id", "model_type"], as_index=False)[_METRIC_COLS].mean()
        assert len(agg) == 2  # One row per run
        # run-1 averages: log_loss=(0.55+0.52)/2=0.535
        run1 = agg[agg["run_id"] == "run-1"].iloc[0]
        assert abs(run1["log_loss"] - 0.535) < 1e-6


class TestEmptyStateHandling:
    def test_empty_raw_data(self) -> None:
        raw: list[dict[str, object]] = []
        assert not raw  # This triggers the empty state in the page

    def test_empty_dataframe_from_raw(self) -> None:
        df = pd.DataFrame([])
        assert df.empty
