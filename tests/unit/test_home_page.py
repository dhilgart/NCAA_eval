"""Tests for home page rendering logic: empty states and happy path."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

_home_mod = importlib.import_module("dashboard.pages.home")


class TestHomePageEmptyState:
    """Verify home page empty-state banners introduced in Story 8.8."""

    def test_no_data_triggers_error_message(self) -> None:
        """When years is empty, st.error must be called with sync guidance."""
        mock_st = MagicMock()

        with (
            patch.object(_home_mod, "load_available_years", return_value=[]),
            patch.object(_home_mod, "load_available_runs", return_value=[]),
            patch.object(_home_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_home_mod, "st", mock_st),
        ):
            _home_mod._render_home()

        mock_st.error.assert_called_once()
        call_text = mock_st.error.call_args[0][0]
        assert "Setup needed" in call_text
        assert "ncaa-eval sync" in call_text

    def test_no_runs_triggers_info_message(self) -> None:
        """When years exist but runs is empty, st.info must be called with train guidance."""
        mock_st = MagicMock()

        with (
            patch.object(_home_mod, "load_available_years", return_value=[2025, 2024]),
            patch.object(_home_mod, "load_available_runs", return_value=[]),
            patch.object(_home_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_home_mod, "st", mock_st),
        ):
            _home_mod._render_home()

        mock_st.info.assert_called_once()
        call_text = mock_st.info.call_args[0][0]
        assert "no models trained" in call_text
        assert "ncaa-eval train" in call_text

    def test_no_data_does_not_call_st_columns(self) -> None:
        """Empty-data path must not attempt to render metrics columns."""
        mock_st = MagicMock()

        with (
            patch.object(_home_mod, "load_available_years", return_value=[]),
            patch.object(_home_mod, "load_available_runs", return_value=[]),
            patch.object(_home_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_home_mod, "st", mock_st),
        ):
            _home_mod._render_home()

        mock_st.columns.assert_not_called()

    def test_happy_path_renders_metrics(self) -> None:
        """When data and runs both exist, st.columns and st.metric must be called."""
        mock_st = MagicMock()
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_st.columns.return_value = [mock_col1, mock_col2]

        with (
            patch.object(_home_mod, "load_available_years", return_value=[2025, 2024, 2023]),
            patch.object(_home_mod, "load_available_runs", return_value=[{"run_id": "run-1"}]),
            patch.object(_home_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_home_mod, "st", mock_st),
        ):
            _home_mod._render_home()

        mock_st.columns.assert_called_once_with(2)
        mock_col1.metric.assert_called_once_with("Available Seasons", 3)
        mock_col2.metric.assert_called_once_with("Model Runs", 1)
