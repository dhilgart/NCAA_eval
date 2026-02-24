"""Tests for dashboard data-loading functions with mocked dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from ncaa_eval.ingest.schema import Season
from ncaa_eval.model.tracking import ModelRun


def _unwrap(fn: Any) -> Any:
    """Return the original function behind ``@st.cache_data``."""
    return fn.__wrapped__


class TestGetDataDir:
    def test_returns_path(self) -> None:
        from dashboard.lib.filters import get_data_dir

        result = get_data_dir()
        assert isinstance(result, Path)
        assert result.name == "data"

    def test_path_is_absolute(self) -> None:
        from dashboard.lib.filters import get_data_dir

        result = get_data_dir()
        assert result.is_absolute()


class TestLoadAvailableYears:
    @patch("dashboard.lib.filters.ParquetRepository")
    def test_returns_sorted_years_descending(self, mock_repo_cls: MagicMock) -> None:
        mock_repo = MagicMock()
        mock_repo.get_seasons.return_value = [
            Season(year=2020),
            Season(year=2023),
            Season(year=2018),
        ]
        mock_repo_cls.return_value = mock_repo

        from dashboard.lib.filters import load_available_years

        result: list[int] = _unwrap(load_available_years)("/fake/data")
        assert result == [2023, 2020, 2018]

    @patch("dashboard.lib.filters.ParquetRepository")
    def test_returns_empty_when_no_seasons(self, mock_repo_cls: MagicMock) -> None:
        mock_repo = MagicMock()
        mock_repo.get_seasons.return_value = []
        mock_repo_cls.return_value = mock_repo

        from dashboard.lib.filters import load_available_years

        result: list[int] = _unwrap(load_available_years)("/fake/data")
        assert result == []

    @patch("dashboard.lib.filters.ParquetRepository")
    def test_passes_data_dir_as_path(self, mock_repo_cls: MagicMock) -> None:
        mock_repo = MagicMock()
        mock_repo.get_seasons.return_value = []
        mock_repo_cls.return_value = mock_repo

        from dashboard.lib.filters import load_available_years

        _unwrap(load_available_years)("/some/path")
        mock_repo_cls.assert_called_once_with(Path("/some/path"))


class TestLoadAvailableRuns:
    @patch("dashboard.lib.filters.RunStore")
    def test_returns_serialised_runs(self, mock_store_cls: MagicMock) -> None:
        mock_store = MagicMock()
        run = ModelRun(
            run_id="run-1",
            model_type="elo",
            hyperparameters={"k": 32},
            git_hash="abc1234",
            start_year=2015,
            end_year=2025,
            prediction_count=100,
        )
        mock_store.list_runs.return_value = [run]
        mock_store_cls.return_value = mock_store

        from dashboard.lib.filters import load_available_runs

        result: list[dict[str, object]] = _unwrap(load_available_runs)("/fake/data")
        assert len(result) == 1
        assert result[0]["run_id"] == "run-1"
        assert result[0]["model_type"] == "elo"
        assert isinstance(result[0], dict)

    @patch("dashboard.lib.filters.RunStore")
    def test_returns_empty_when_no_runs(self, mock_store_cls: MagicMock) -> None:
        mock_store = MagicMock()
        mock_store.list_runs.return_value = []
        mock_store_cls.return_value = mock_store

        from dashboard.lib.filters import load_available_runs

        result: list[dict[str, object]] = _unwrap(load_available_runs)("/fake/data")
        assert result == []


class TestLoadAvailableScorings:
    @patch("dashboard.lib.filters.list_scorings")
    def test_returns_scoring_names(self, mock_list: MagicMock) -> None:
        mock_list.return_value = ["fibonacci", "standard", "seed_diff_bonus"]

        from dashboard.lib.filters import load_available_scorings

        result: list[str] = _unwrap(load_available_scorings)()
        assert result == ["fibonacci", "standard", "seed_diff_bonus"]

    @patch("dashboard.lib.filters.list_scorings")
    def test_return_type_is_list_of_str(self, mock_list: MagicMock) -> None:
        mock_list.return_value = ["standard"]

        from dashboard.lib.filters import load_available_scorings

        result: list[str] = _unwrap(load_available_scorings)()
        assert all(isinstance(s, str) for s in result)
