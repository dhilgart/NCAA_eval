"""Tests for leaderboard page logic: year filtering, empty state handling."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pandas as pd  # type: ignore[import-untyped]

_lab_mod = importlib.import_module("dashboard.pages.1_Lab")
_METRIC_COLS: list[str] = _lab_mod._METRIC_COLS
_DISPLAY_COLS: list[str] = _lab_mod._DISPLAY_COLS


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

    def test_year_filter_applied_through_render_function(self) -> None:
        """_render_leaderboard with selected_year set renders only the filtered year rows."""
        mock_st = MagicMock()
        mock_st.session_state = {"selected_year": 2023}
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        mock_st.dataframe.return_value = MagicMock(selection=MagicMock(rows=[]))

        with (
            patch.object(_lab_mod, "load_leaderboard_data", return_value=_sample_data()),
            patch.object(_lab_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_lab_mod, "st", mock_st),
        ):
            _lab_mod._render_leaderboard()

        # st.dataframe must have been called (renders the leaderboard)
        mock_st.dataframe.assert_called_once()
        rendered_df = mock_st.dataframe.call_args[0][0].data
        assert set(rendered_df["year"].unique()) == {2023}

    def test_year_filtered_display_uses_display_cols_only(self) -> None:
        """Year-filtered view must not expose internal columns (timestamp, start_year, etc.)."""
        mock_st = MagicMock()
        mock_st.session_state = {"selected_year": 2023}
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        mock_st.dataframe.return_value = MagicMock(selection=MagicMock(rows=[]))

        with (
            patch.object(_lab_mod, "load_leaderboard_data", return_value=_sample_data()),
            patch.object(_lab_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_lab_mod, "st", mock_st),
        ):
            _lab_mod._render_leaderboard()

        rendered_df = mock_st.dataframe.call_args[0][0].data
        assert list(rendered_df.columns) == _DISPLAY_COLS


class TestEmptyStateHandling:
    def test_no_runs_triggers_info_message(self) -> None:
        """When no runs AND no leaderboard data, st.info should be called."""
        from unittest.mock import MagicMock, patch

        mock_st = MagicMock()
        with (
            patch.object(_lab_mod, "load_leaderboard_data", return_value=[]),
            patch.object(_lab_mod, "load_available_runs", return_value=[]),
            patch.object(_lab_mod, "st", mock_st),
        ):
            _lab_mod._render_leaderboard()

        mock_st.info.assert_called_once()
        call_args = mock_st.info.call_args[0][0]
        assert "Train a model" in call_args

    def test_legacy_runs_triggers_warning_message(self) -> None:
        """When runs exist but no summaries, st.warning should be called."""
        from unittest.mock import MagicMock, patch

        mock_st = MagicMock()
        with (
            patch.object(_lab_mod, "load_leaderboard_data", return_value=[]),
            patch.object(_lab_mod, "load_available_runs", return_value=[{"run_id": "old-run"}]),
            patch.object(_lab_mod, "st", mock_st),
        ):
            _lab_mod._render_leaderboard()

        mock_st.warning.assert_called_once()
        call_args = mock_st.warning.call_args[0][0]
        assert "Re-run training" in call_args
