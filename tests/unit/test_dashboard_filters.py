"""Tests for dashboard data-loading functions with mocked dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ncaa_eval.ingest.schema import Season, Team
from ncaa_eval.model.tracking import ModelRun
from ncaa_eval.transform.normalization import TourneySeed


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
    @patch("dashboard.lib.filters.Path.exists", return_value=True)
    @patch("dashboard.lib.filters.ParquetRepository")
    def test_returns_sorted_years_descending(self, mock_repo_cls: MagicMock, mock_exists: MagicMock) -> None:
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

    @patch("dashboard.lib.filters.Path.exists", return_value=True)
    @patch("dashboard.lib.filters.ParquetRepository")
    def test_returns_empty_when_no_seasons(self, mock_repo_cls: MagicMock, mock_exists: MagicMock) -> None:
        mock_repo = MagicMock()
        mock_repo.get_seasons.return_value = []
        mock_repo_cls.return_value = mock_repo

        from dashboard.lib.filters import load_available_years

        result: list[int] = _unwrap(load_available_years)("/fake/data")
        assert result == []

    @patch("dashboard.lib.filters.Path.exists", return_value=True)
    @patch("dashboard.lib.filters.ParquetRepository")
    def test_passes_data_dir_as_path(self, mock_repo_cls: MagicMock, mock_exists: MagicMock) -> None:
        mock_repo = MagicMock()
        mock_repo.get_seasons.return_value = []
        mock_repo_cls.return_value = mock_repo

        from dashboard.lib.filters import load_available_years

        _unwrap(load_available_years)("/some/path")
        mock_repo_cls.assert_called_once_with(Path("/some/path"))

    def test_returns_empty_when_data_dir_missing(self) -> None:
        from dashboard.lib.filters import load_available_years

        result: list[int] = _unwrap(load_available_years)("/nonexistent/path/that/cannot/exist")
        assert result == []

    @patch("dashboard.lib.filters.Path.exists", return_value=True)
    @patch("dashboard.lib.filters.ParquetRepository")
    def test_returns_empty_on_repository_exception(
        self, mock_repo_cls: MagicMock, mock_exists: MagicMock
    ) -> None:
        mock_repo_cls.side_effect = OSError("disk error")

        from dashboard.lib.filters import load_available_years

        result: list[int] = _unwrap(load_available_years)("/fake/data")
        assert result == []


class TestLoadAvailableRuns:
    @patch("dashboard.lib.filters.Path.exists", return_value=True)
    @patch("dashboard.lib.filters.RunStore")
    def test_returns_serialised_runs(self, mock_store_cls: MagicMock, mock_exists: MagicMock) -> None:
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

    @patch("dashboard.lib.filters.Path.exists", return_value=True)
    @patch("dashboard.lib.filters.RunStore")
    def test_returns_empty_when_no_runs(self, mock_store_cls: MagicMock, mock_exists: MagicMock) -> None:
        mock_store = MagicMock()
        mock_store.list_runs.return_value = []
        mock_store_cls.return_value = mock_store

        from dashboard.lib.filters import load_available_runs

        result: list[dict[str, object]] = _unwrap(load_available_runs)("/fake/data")
        assert result == []

    def test_returns_empty_when_data_dir_missing(self) -> None:
        from dashboard.lib.filters import load_available_runs

        result: list[dict[str, object]] = _unwrap(load_available_runs)("/nonexistent/path/that/cannot/exist")
        assert result == []

    @patch("dashboard.lib.filters.Path.exists", return_value=True)
    @patch("dashboard.lib.filters.RunStore")
    def test_returns_empty_on_store_exception(
        self, mock_store_cls: MagicMock, mock_exists: MagicMock
    ) -> None:
        mock_store_cls.side_effect = OSError("disk error")

        from dashboard.lib.filters import load_available_runs

        result: list[dict[str, object]] = _unwrap(load_available_runs)("/fake/data")
        assert result == []


class TestLoadLeaderboardData:
    @patch("dashboard.lib.filters.Path.exists", return_value=True)
    @patch("dashboard.lib.filters.RunStore")
    def test_returns_joined_data(self, mock_store_cls: MagicMock, mock_exists: MagicMock) -> None:
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
        summary = pd.DataFrame(
            {
                "run_id": ["run-1", "run-1"],
                "year": [2023, 2024],
                "log_loss": [0.55, 0.52],
                "brier_score": [0.20, 0.19],
                "roc_auc": [0.73, 0.76],
                "ece": [0.035, 0.028],
                "elapsed_seconds": [1.2, 1.1],
            }
        )
        mock_store.load_all_summaries.return_value = summary
        mock_store_cls.return_value = mock_store

        from dashboard.lib.filters import load_leaderboard_data

        result: list[dict[str, object]] = _unwrap(load_leaderboard_data)("/fake/data")
        assert len(result) == 2
        assert result[0]["run_id"] == "run-1"
        assert result[0]["model_type"] == "elo"
        assert result[0]["year"] == 2023
        assert result[0]["log_loss"] == 0.55

    @patch("dashboard.lib.filters.Path.exists", return_value=True)
    @patch("dashboard.lib.filters.RunStore")
    def test_returns_empty_when_no_summaries(self, mock_store_cls: MagicMock, mock_exists: MagicMock) -> None:
        mock_store = MagicMock()
        mock_store.load_all_summaries.return_value = pd.DataFrame(
            columns=["run_id", "year", "log_loss", "brier_score", "roc_auc", "ece", "elapsed_seconds"]
        )
        mock_store_cls.return_value = mock_store

        from dashboard.lib.filters import load_leaderboard_data

        result: list[dict[str, object]] = _unwrap(load_leaderboard_data)("/fake/data")
        assert result == []

    def test_returns_empty_when_data_dir_missing(self) -> None:
        from dashboard.lib.filters import load_leaderboard_data

        result: list[dict[str, object]] = _unwrap(load_leaderboard_data)(
            "/nonexistent/path/that/cannot/exist"
        )
        assert result == []

    @patch("dashboard.lib.filters.Path.exists", return_value=True)
    @patch("dashboard.lib.filters.RunStore")
    def test_returns_empty_on_store_exception(
        self, mock_store_cls: MagicMock, mock_exists: MagicMock
    ) -> None:
        mock_store_cls.side_effect = OSError("disk error")

        from dashboard.lib.filters import load_leaderboard_data

        result: list[dict[str, object]] = _unwrap(load_leaderboard_data)("/fake/data")
        assert result == []

    @patch("dashboard.lib.filters.Path.exists", return_value=True)
    @patch("dashboard.lib.filters.RunStore")
    def test_returns_empty_when_summaries_present_but_runs_missing(
        self, mock_store_cls: MagicMock, mock_exists: MagicMock
    ) -> None:
        """Summaries loaded but list_runs returns empty → returns empty (no orphan rows)."""
        mock_store = MagicMock()
        mock_store.list_runs.return_value = []
        mock_store.load_all_summaries.return_value = pd.DataFrame(
            {
                "run_id": ["ghost-run"],
                "year": [2024],
                "log_loss": [0.55],
                "brier_score": [0.20],
                "roc_auc": [0.73],
                "ece": [0.035],
                "elapsed_seconds": [1.2],
            }
        )
        mock_store_cls.return_value = mock_store

        from dashboard.lib.filters import load_leaderboard_data

        result: list[dict[str, object]] = _unwrap(load_leaderboard_data)("/fake/data")
        assert result == []


class TestLoadFoldPredictions:
    @patch("dashboard.lib.filters.Path.exists", return_value=True)
    @patch("dashboard.lib.filters.RunStore")
    def test_returns_list_of_dicts(self, mock_store_cls: MagicMock, mock_exists: MagicMock) -> None:
        mock_store = MagicMock()
        mock_store.load_fold_predictions.return_value = pd.DataFrame(
            {
                "year": [2023, 2024],
                "game_id": ["g1", "g2"],
                "team_a_id": [101, 102],
                "team_b_id": [201, 202],
                "pred_win_prob": [0.7, 0.6],
                "team_a_won": [1.0, 0.0],
            }
        )
        mock_store_cls.return_value = mock_store

        from dashboard.lib.filters import load_fold_predictions

        result: list[dict[str, object]] = _unwrap(load_fold_predictions)("/fake/data", "run-1")
        assert len(result) == 2
        assert result[0]["year"] == 2023
        assert result[0]["pred_win_prob"] == 0.7

    @patch("dashboard.lib.filters.Path.exists", return_value=True)
    @patch("dashboard.lib.filters.RunStore")
    def test_returns_empty_for_legacy_run(self, mock_store_cls: MagicMock, mock_exists: MagicMock) -> None:
        mock_store = MagicMock()
        mock_store.load_fold_predictions.return_value = None
        mock_store_cls.return_value = mock_store

        from dashboard.lib.filters import load_fold_predictions

        result: list[dict[str, object]] = _unwrap(load_fold_predictions)("/fake/data", "run-1")
        assert result == []

    def test_returns_empty_on_missing_dir(self) -> None:
        from dashboard.lib.filters import load_fold_predictions

        result: list[dict[str, object]] = _unwrap(load_fold_predictions)("/nonexistent", "run-1")
        assert result == []

    @patch("dashboard.lib.filters.Path.exists", return_value=True)
    @patch("dashboard.lib.filters.RunStore")
    def test_returns_empty_on_oserror(self, mock_store_cls: MagicMock, mock_exists: MagicMock) -> None:
        mock_store_cls.side_effect = OSError("disk error")

        from dashboard.lib.filters import load_fold_predictions

        result: list[dict[str, object]] = _unwrap(load_fold_predictions)("/fake/data", "run-1")
        assert result == []


class TestLoadFeatureImportances:
    @patch("dashboard.lib.filters.Path.exists", return_value=True)
    @patch("dashboard.lib.filters.RunStore")
    def test_returns_sorted_importances(self, mock_store_cls: MagicMock, mock_exists: MagicMock) -> None:
        mock_store = MagicMock()
        mock_store.load_feature_names.return_value = ["elo_delta", "seed_diff"]
        mock_model = MagicMock()
        mock_model._clf = MagicMock()
        import numpy as np

        mock_model._clf.feature_importances_ = np.array([0.3, 0.7])
        mock_store.load_model.return_value = mock_model
        mock_run = MagicMock()
        mock_run.model_type = "xgboost"
        mock_store.load_run.return_value = mock_run
        mock_store_cls.return_value = mock_store

        from dashboard.lib.filters import load_feature_importances

        result: list[dict[str, object]] = _unwrap(load_feature_importances)("/fake/data", "run-1")
        assert len(result) == 2
        assert result[0]["feature"] == "seed_diff"
        assert result[0]["importance"] == 0.7

    @patch("dashboard.lib.filters.Path.exists", return_value=True)
    @patch("dashboard.lib.filters.RunStore")
    def test_returns_empty_for_elo_model(self, mock_store_cls: MagicMock, mock_exists: MagicMock) -> None:
        mock_store = MagicMock()
        mock_run = MagicMock()
        mock_run.model_type = "elo"
        mock_store.load_run.return_value = mock_run
        mock_store_cls.return_value = mock_store

        from dashboard.lib.filters import load_feature_importances

        result: list[dict[str, object]] = _unwrap(load_feature_importances)("/fake/data", "run-1")
        assert result == []

    @patch("dashboard.lib.filters.Path.exists", return_value=True)
    @patch("dashboard.lib.filters.RunStore")
    def test_returns_empty_for_legacy_run(self, mock_store_cls: MagicMock, mock_exists: MagicMock) -> None:
        mock_store = MagicMock()
        mock_run = MagicMock()
        mock_run.model_type = "xgboost"
        mock_store.load_run.return_value = mock_run
        mock_store.load_model.return_value = None
        mock_store_cls.return_value = mock_store

        from dashboard.lib.filters import load_feature_importances

        result: list[dict[str, object]] = _unwrap(load_feature_importances)("/fake/data", "run-1")
        assert result == []

    def test_returns_empty_on_missing_dir(self) -> None:
        from dashboard.lib.filters import load_feature_importances

        result: list[dict[str, object]] = _unwrap(load_feature_importances)("/nonexistent", "run-1")
        assert result == []

    @patch("dashboard.lib.filters.Path.exists", return_value=True)
    @patch("dashboard.lib.filters.RunStore")
    def test_returns_empty_on_oserror_from_load_run(
        self, mock_store_cls: MagicMock, mock_exists: MagicMock
    ) -> None:
        mock_store = MagicMock()
        mock_store.load_run.side_effect = FileNotFoundError("no run.json")
        mock_store_cls.return_value = mock_store

        from dashboard.lib.filters import load_feature_importances

        result: list[dict[str, object]] = _unwrap(load_feature_importances)("/fake/data", "missing-run")
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


# ---------------------------------------------------------------------------
# Story 7.5: Bracket Visualizer loader tests
# ---------------------------------------------------------------------------


class TestLoadTourneySeeds:
    @patch("dashboard.lib.filters.TourneySeedTable")
    def test_returns_serialised_seeds(self, mock_table_cls: MagicMock) -> None:
        mock_table = MagicMock()
        mock_table.all_seeds.return_value = [
            TourneySeed(season=2023, team_id=100, seed_str="W01", region="W", seed_num=1, is_play_in=False),
            TourneySeed(season=2023, team_id=200, seed_str="W16", region="W", seed_num=16, is_play_in=False),
        ]
        mock_table_cls.from_csv.return_value = mock_table

        from dashboard.lib.filters import load_tourney_seeds

        # Need to create the CSV path to exist for the function to proceed
        with patch("dashboard.lib.filters.Path.exists", return_value=True):
            result = _unwrap(load_tourney_seeds)("/fake/data", 2023)

        assert len(result) == 2
        assert result[0]["team_id"] == 100
        assert result[0]["seed_num"] == 1
        assert result[0]["region"] == "W"

    def test_returns_empty_when_csv_missing(self) -> None:
        from dashboard.lib.filters import load_tourney_seeds

        result = _unwrap(load_tourney_seeds)("/nonexistent", 2023)
        assert result == []

    @patch("dashboard.lib.filters.TourneySeedTable")
    def test_returns_empty_for_season_with_no_seeds(self, mock_table_cls: MagicMock) -> None:
        mock_table = MagicMock()
        mock_table.all_seeds.return_value = []
        mock_table_cls.from_csv.return_value = mock_table

        from dashboard.lib.filters import load_tourney_seeds

        with patch("dashboard.lib.filters.Path.exists", return_value=True):
            result = _unwrap(load_tourney_seeds)("/fake/data", 1900)

        assert result == []


class TestLoadTeamNames:
    @patch("dashboard.lib.filters.Path.exists", return_value=True)
    @patch("dashboard.lib.filters.ParquetRepository")
    def test_returns_id_to_name_mapping(self, mock_repo_cls: MagicMock, mock_exists: MagicMock) -> None:
        mock_repo = MagicMock()
        mock_repo.get_teams.return_value = [
            Team(team_id=100, team_name="Duke", canonical_name="Duke"),
            Team(team_id=200, team_name="UNC", canonical_name="UNC"),
        ]
        mock_repo_cls.return_value = mock_repo

        from dashboard.lib.filters import load_team_names

        result = _unwrap(load_team_names)("/fake/data")
        assert result == {100: "Duke", 200: "UNC"}

    def test_returns_empty_when_data_dir_missing(self) -> None:
        from dashboard.lib.filters import load_team_names

        result = _unwrap(load_team_names)("/nonexistent")
        assert result == {}

    @patch("dashboard.lib.filters.Path.exists", return_value=True)
    @patch("dashboard.lib.filters.ParquetRepository")
    def test_returns_empty_on_exception(self, mock_repo_cls: MagicMock, mock_exists: MagicMock) -> None:
        mock_repo_cls.side_effect = OSError("disk error")

        from dashboard.lib.filters import load_team_names

        result = _unwrap(load_team_names)("/fake/data")
        assert result == {}


# ---------------------------------------------------------------------------
# Story 7.5: _build_provider_from_folds and _build_team_labels tests
# ---------------------------------------------------------------------------


def _make_mock_bracket(
    team_ids: tuple[int, ...],
    seed_map: dict[int, int] | None = None,
) -> MagicMock:
    """Return a MagicMock that mimics BracketStructure."""
    bracket = MagicMock()
    bracket.team_ids = team_ids
    bracket.team_index_map = {tid: i for i, tid in enumerate(team_ids)}
    bracket.seed_map = seed_map or {tid: (i % 16) + 1 for i, tid in enumerate(team_ids)}
    return bracket


class TestBuildProviderFromFolds:
    def test_returns_matrix_provider_for_valid_preds(self) -> None:
        from dashboard.lib.filters import _build_provider_from_folds

        team_ids = (100, 200, 300, 400)
        bracket = _make_mock_bracket(team_ids)
        fold_df = pd.DataFrame(
            {
                "year": [2023, 2023],
                "team_a_id": [100, 300],
                "team_b_id": [200, 400],
                "pred_win_prob": [0.7, 0.6],
            }
        )
        mock_store = MagicMock()
        mock_store.load_fold_predictions.return_value = fold_df

        result = _build_provider_from_folds(mock_store, "run-1", 2023, bracket)

        assert result is not None
        # Verify the probability matrix was filled correctly
        # P[0,1] should be 0.7 (team_a=100→idx=0 beats team_b=200→idx=1)
        assert abs(result._P[0, 1] - 0.7) < 1e-6
        assert abs(result._P[1, 0] - 0.3) < 1e-6

    def test_returns_none_when_no_fold_predictions(self) -> None:
        from dashboard.lib.filters import _build_provider_from_folds

        bracket = _make_mock_bracket((100, 200))
        mock_store = MagicMock()
        mock_store.load_fold_predictions.return_value = None

        result = _build_provider_from_folds(mock_store, "run-1", 2023, bracket)

        assert result is None

    def test_returns_none_when_no_preds_for_season(self) -> None:
        from dashboard.lib.filters import _build_provider_from_folds

        bracket = _make_mock_bracket((100, 200))
        fold_df = pd.DataFrame(
            {"year": [2022], "team_a_id": [100], "team_b_id": [200], "pred_win_prob": [0.7]}
        )
        mock_store = MagicMock()
        mock_store.load_fold_predictions.return_value = fold_df

        result = _build_provider_from_folds(mock_store, "run-1", 2023, bracket)

        assert result is None

    def test_ignores_teams_not_in_bracket(self) -> None:
        from dashboard.lib.filters import _build_provider_from_folds

        team_ids = (100, 200)
        bracket = _make_mock_bracket(team_ids)
        fold_df = pd.DataFrame(
            {
                "year": [2023, 2023],
                "team_a_id": [100, 999],  # 999 is NOT in bracket
                "team_b_id": [200, 888],  # 888 is NOT in bracket
                "pred_win_prob": [0.7, 0.5],
            }
        )
        mock_store = MagicMock()
        mock_store.load_fold_predictions.return_value = fold_df

        result = _build_provider_from_folds(mock_store, "run-1", 2023, bracket)

        assert result is not None
        # Only the valid row (100 vs 200) should be filled; 999/888 row ignored
        assert abs(result._P[0, 1] - 0.7) < 1e-6


class TestBuildTeamLabels:
    @patch("dashboard.lib.filters.Path.exists", return_value=True)
    @patch("dashboard.lib.filters.ParquetRepository")
    def test_builds_seed_name_labels(self, mock_repo_cls: MagicMock, mock_exists: MagicMock) -> None:
        from dashboard.lib.filters import _build_team_labels

        mock_repo = MagicMock()
        mock_repo.get_teams.return_value = [
            Team(team_id=100, team_name="Duke"),
            Team(team_id=200, team_name="Norfolk St"),
        ]
        mock_repo_cls.return_value = mock_repo

        bracket = _make_mock_bracket((100, 200), seed_map={100: 1, 200: 16})
        result = _build_team_labels("/fake/data", bracket)

        assert result[0] == "[1] Duke"
        assert result[1] == "[16] Norfolk St"

    @patch("dashboard.lib.filters.Path.exists", return_value=True)
    @patch("dashboard.lib.filters.ParquetRepository")
    def test_falls_back_to_team_id_when_name_missing(
        self, mock_repo_cls: MagicMock, mock_exists: MagicMock
    ) -> None:
        from dashboard.lib.filters import _build_team_labels

        mock_repo = MagicMock()
        mock_repo.get_teams.return_value = []  # no team names available
        mock_repo_cls.return_value = mock_repo

        bracket = _make_mock_bracket((100,), seed_map={100: 5})
        result = _build_team_labels("/fake/data", bracket)

        assert result[0] == "[5] 100"


class TestRunBracketSimulation:
    @patch("dashboard.lib.filters.Path.exists", return_value=True)
    @patch("dashboard.lib.filters.TourneySeedTable")
    @patch("dashboard.lib.filters.RunStore")
    @patch("dashboard.lib.filters.build_bracket")
    @patch("dashboard.lib.filters.build_probability_matrix")
    @patch("dashboard.lib.filters.compute_most_likely_bracket")
    @patch("dashboard.lib.filters.simulate_tournament")
    @patch("dashboard.lib.filters.get_scoring")
    @patch("dashboard.lib.filters._build_team_labels")
    def test_returns_result_for_elo_model(  # noqa: PLR0913
        self,
        mock_labels: MagicMock,
        mock_get_scoring: MagicMock,
        mock_simulate: MagicMock,
        mock_most_likely: MagicMock,
        mock_prob_matrix: MagicMock,
        mock_build_bracket: MagicMock,
        mock_run_store: MagicMock,
        mock_seed_table: MagicMock,
        mock_exists: MagicMock,
    ) -> None:
        from dashboard.lib.filters import run_bracket_simulation

        bracket = _make_mock_bracket((100, 200))
        mock_build_bracket.return_value = bracket
        mock_prob_matrix.return_value = np.full((2, 2), 0.5)
        mock_most_likely.return_value = MagicMock()
        mock_simulate.return_value = MagicMock()
        mock_labels.return_value = {0: "[1] Duke", 1: "[16] Norfolk St"}
        mock_get_scoring.return_value = MagicMock(return_value=MagicMock())

        mock_store = MagicMock()
        mock_run = MagicMock()
        mock_run.model_type = "elo"
        mock_store.load_run.return_value = mock_run
        mock_store.load_model.return_value = MagicMock()
        mock_run_store.return_value = mock_store

        mock_table = MagicMock()
        mock_table.all_seeds.return_value = [MagicMock()]
        mock_seed_table.from_csv.return_value = mock_table

        result = _unwrap(run_bracket_simulation)("/fake/data", "run-elo", 2023, "standard")

        assert result is not None
        assert result.team_labels == {0: "[1] Duke", 1: "[16] Norfolk St"}

    @patch("dashboard.lib.filters.Path.exists", return_value=False)
    def test_returns_none_when_seed_csv_missing(self, mock_exists: MagicMock) -> None:
        from dashboard.lib.filters import run_bracket_simulation

        result = _unwrap(run_bracket_simulation)("/fake/data", "run-1", 2023, "standard")

        assert result is None

    @patch("dashboard.lib.filters.Path.exists", return_value=True)
    @patch("dashboard.lib.filters.TourneySeedTable")
    @patch("dashboard.lib.filters.RunStore")
    @patch("dashboard.lib.filters.build_bracket")
    def test_returns_none_when_model_missing(
        self,
        mock_build_bracket: MagicMock,
        mock_run_store: MagicMock,
        mock_seed_table: MagicMock,
        mock_exists: MagicMock,
    ) -> None:
        from dashboard.lib.filters import run_bracket_simulation

        bracket = _make_mock_bracket((100, 200))
        mock_build_bracket.return_value = bracket

        mock_table = MagicMock()
        mock_table.all_seeds.return_value = [MagicMock()]
        mock_seed_table.from_csv.return_value = mock_table

        mock_store = MagicMock()
        mock_store.load_run.return_value = MagicMock(model_type="elo")
        mock_store.load_model.return_value = None  # model not found
        mock_run_store.return_value = mock_store

        result = _unwrap(run_bracket_simulation)("/fake/data", "run-1", 2023, "standard")

        assert result is None

    @patch("dashboard.lib.filters.Path.exists", return_value=True)
    @patch("dashboard.lib.filters.TourneySeedTable")
    @patch("dashboard.lib.filters.RunStore")
    def test_returns_none_when_no_seeds_for_season(
        self,
        mock_run_store: MagicMock,
        mock_seed_table: MagicMock,
        mock_exists: MagicMock,
    ) -> None:
        from dashboard.lib.filters import run_bracket_simulation

        mock_table = MagicMock()
        mock_table.all_seeds.return_value = []  # empty seeds
        mock_seed_table.from_csv.return_value = mock_table

        result = _unwrap(run_bracket_simulation)("/fake/data", "run-1", 2023, "standard")

        assert result is None

    @patch("dashboard.lib.filters.Path.exists", return_value=True)
    @patch("dashboard.lib.filters.TourneySeedTable")
    @patch("dashboard.lib.filters.RunStore")
    @patch("dashboard.lib.filters.build_bracket")
    @patch("dashboard.lib.filters._build_provider_from_folds")
    def test_returns_none_when_xgboost_fold_provider_fails(
        self,
        mock_build_provider: MagicMock,
        mock_build_bracket: MagicMock,
        mock_run_store: MagicMock,
        mock_seed_table: MagicMock,
        mock_exists: MagicMock,
    ) -> None:
        from dashboard.lib.filters import run_bracket_simulation

        bracket = _make_mock_bracket((100, 200))
        mock_build_bracket.return_value = bracket
        mock_build_provider.return_value = None  # no fold predictions

        mock_table = MagicMock()
        mock_table.all_seeds.return_value = [MagicMock()]
        mock_seed_table.from_csv.return_value = mock_table

        mock_store = MagicMock()
        mock_store.load_run.return_value = MagicMock(model_type="xgboost")
        mock_store.load_model.return_value = MagicMock()
        mock_run_store.return_value = mock_store

        result = _unwrap(run_bracket_simulation)("/fake/data", "run-xgb", 2023, "standard")

        assert result is None


# ---------------------------------------------------------------------------
# Story 7.6: Pool Scorer helper tests
# ---------------------------------------------------------------------------


class TestScoreChosenBracket:
    def test_returns_bracket_distributions(self) -> None:
        from dashboard.lib.filters import score_chosen_bracket

        n_sims = 100
        sim_winners = np.random.default_rng(42).integers(0, 4, size=(n_sims, 3)).astype(np.int32)
        mock_sim_result = MagicMock()
        mock_sim_result.sim_winners = sim_winners

        mock_most_likely = MagicMock()
        mock_most_likely.winners = (0, 2, 0)

        sim_data = MagicMock()
        sim_data.sim_result = mock_sim_result
        sim_data.most_likely = mock_most_likely

        mock_rule = MagicMock()
        mock_rule.name = "standard"
        mock_rule.points_per_round.side_effect = lambda r: [1, 2, 4, 8, 16, 32][r]

        result = score_chosen_bracket(sim_data, [mock_rule], "standard")

        assert "standard" in result
        dist = result["standard"]
        assert hasattr(dist, "mean")
        assert hasattr(dist, "percentiles")
        assert 50 in dist.percentiles

    def test_raises_when_no_sim_winners(self) -> None:
        import pytest

        from dashboard.lib.filters import score_chosen_bracket

        mock_sim_result = MagicMock()
        mock_sim_result.sim_winners = None

        sim_data = MagicMock()
        sim_data.sim_result = mock_sim_result

        with pytest.raises(ValueError, match="MC sim_winners required"):
            score_chosen_bracket(sim_data, [MagicMock()], "standard")


class TestBuildCustomScoring:
    def test_wraps_six_element_tuple(self) -> None:
        from dashboard.lib.filters import build_custom_scoring

        scoring = build_custom_scoring((1.0, 2.0, 4.0, 8.0, 16.0, 32.0))

        assert scoring.name == "custom"
        assert scoring.points_per_round(0) == 1.0
        assert scoring.points_per_round(5) == 32.0

    def test_raises_on_wrong_length(self) -> None:
        import pytest

        from dashboard.lib.filters import build_custom_scoring

        with pytest.raises(ValueError, match="exactly 6 entries"):
            build_custom_scoring((1.0, 2.0, 4.0))  # only 3 elements


class TestExportBracketCsv:
    def test_csv_has_correct_structure(self) -> None:
        from dashboard.lib.filters import export_bracket_csv

        # 4-team bracket: 3 games (2 R64 + 1 Championship equivalent)
        team_ids = (100, 200, 300, 400)
        bracket = MagicMock()
        bracket.team_ids = team_ids
        bracket.team_index_map = {100: 0, 200: 1, 300: 2, 400: 3}
        bracket.seed_map = {100: 1, 200: 16, 300: 2, 400: 15}

        most_likely = MagicMock()
        most_likely.winners = (0, 2, 0)  # 3 games in a 4-team bracket

        labels = {0: "[1] Duke", 1: "[16] Norfolk St", 2: "[2] UConn", 3: "[15] Wagner"}
        P = np.full((4, 4), 0.5)
        P[0, 1] = 0.9
        P[1, 0] = 0.1
        P[2, 3] = 0.8
        P[3, 2] = 0.2
        P[0, 2] = 0.6
        P[2, 0] = 0.4

        csv_str = export_bracket_csv(bracket, most_likely, labels, P)

        lines = csv_str.strip().split("\n")
        assert lines[0] == "game_number,round,team_id,team_name,seed,win_probability"
        assert len(lines) == 4  # header + 3 games

    def test_csv_columns_match_spec(self) -> None:
        from dashboard.lib.filters import export_bracket_csv

        team_ids = (100, 200, 300, 400)
        bracket = MagicMock()
        bracket.team_ids = team_ids
        bracket.team_index_map = {100: 0, 200: 1, 300: 2, 400: 3}
        bracket.seed_map = {100: 1, 200: 16, 300: 2, 400: 15}

        most_likely = MagicMock()
        most_likely.winners = (0, 2, 0)

        labels = {0: "[1] Duke", 1: "[16] Norfolk St", 2: "[2] UConn", 3: "[15] Wagner"}
        P = np.full((4, 4), 0.5)

        csv_str = export_bracket_csv(bracket, most_likely, labels, P)

        import csv as csv_mod

        reader = csv_mod.reader(csv_str.strip().split("\n"))
        header = next(reader)
        assert header == ["game_number", "round", "team_id", "team_name", "seed", "win_probability"]

    def test_round_labels_present(self) -> None:
        from dashboard.lib.filters import export_bracket_csv

        team_ids = (100, 200, 300, 400)
        bracket = MagicMock()
        bracket.team_ids = team_ids
        bracket.team_index_map = {100: 0, 200: 1, 300: 2, 400: 3}
        bracket.seed_map = {100: 1, 200: 16, 300: 2, 400: 15}

        most_likely = MagicMock()
        most_likely.winners = (0, 2, 0)

        labels = {0: "[1] Duke", 1: "[16] Norfolk St", 2: "[2] UConn", 3: "[15] Wagner"}
        P = np.full((4, 4), 0.5)

        csv_str = export_bracket_csv(bracket, most_likely, labels, P)

        # 4-team bracket has 2 rounds: R64 (2 games) and R32 (1 game)
        assert "R64" in csv_str
        assert "R32" in csv_str
