"""Tests for Model Deep Dive page logic."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

_dd_mod = importlib.import_module("dashboard.pages.3_Model_Deep_Dive")


def _sample_runs() -> list[dict[str, object]]:
    """Return sample run metadata."""
    return [
        {
            "run_id": "abc12345-6789",
            "model_type": "elo",
            "hyperparameters": {"k": 32},
            "timestamp": "2025-01-01T00:00:00",
            "git_hash": "abc1234",
            "start_year": 2015,
            "end_year": 2025,
            "prediction_count": 100,
        },
    ]


def _sample_fold_predictions() -> list[dict[str, object]]:
    """Return sample fold predictions."""
    return [
        {
            "year": 2023,
            "game_id": "g1",
            "team_a_id": 101,
            "team_b_id": 201,
            "pred_win_prob": 0.7,
            "team_a_won": 1.0,
        },
        {
            "year": 2023,
            "game_id": "g2",
            "team_a_id": 102,
            "team_b_id": 202,
            "pred_win_prob": 0.4,
            "team_a_won": 0.0,
        },
        {
            "year": 2024,
            "game_id": "g3",
            "team_a_id": 103,
            "team_b_id": 203,
            "pred_win_prob": 0.6,
            "team_a_won": 1.0,
        },
        {
            "year": 2024,
            "game_id": "g4",
            "team_a_id": 104,
            "team_b_id": 204,
            "pred_win_prob": 0.55,
            "team_a_won": 0.0,
        },
    ]


class TestNoRunSelected:
    def test_shows_info_when_no_run_selected(self) -> None:
        mock_st = MagicMock()
        mock_st.session_state = {}

        with patch.object(_dd_mod, "st", mock_st):
            _dd_mod._render_deep_dive()

        mock_st.info.assert_called_once()
        assert "Select a model run" in mock_st.info.call_args[0][0]

    def test_shows_page_link_to_leaderboard(self) -> None:
        mock_st = MagicMock()
        mock_st.session_state = {}

        with patch.object(_dd_mod, "st", mock_st):
            _dd_mod._render_deep_dive()

        mock_st.page_link.assert_called_once()
        assert "1_Lab.py" in mock_st.page_link.call_args[0][0]


class TestWithFoldPredictions:
    def test_renders_plotly_chart(self) -> None:
        mock_st = MagicMock()
        mock_st.session_state = {"selected_run_id": "abc12345-6789"}
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.selectbox.return_value = "All Years (Aggregate)"

        with (
            patch.object(_dd_mod, "st", mock_st),
            patch.object(_dd_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_dd_mod, "load_available_runs", return_value=_sample_runs()),
            patch.object(_dd_mod, "load_fold_predictions", return_value=_sample_fold_predictions()),
            patch.object(_dd_mod, "load_leaderboard_data", return_value=[]),
            patch.object(_dd_mod, "load_feature_importances", return_value=[]),
        ):
            _dd_mod._render_deep_dive()

        mock_st.plotly_chart.assert_called()


class TestLegacyRun:
    def test_shows_warning_for_missing_fold_predictions(self) -> None:
        mock_st = MagicMock()
        mock_st.session_state = {"selected_run_id": "abc12345-6789"}
        mock_st.columns.return_value = [MagicMock(), MagicMock()]

        with (
            patch.object(_dd_mod, "st", mock_st),
            patch.object(_dd_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_dd_mod, "load_available_runs", return_value=_sample_runs()),
            patch.object(_dd_mod, "load_fold_predictions", return_value=[]),
            patch.object(_dd_mod, "load_leaderboard_data", return_value=[]),
            patch.object(_dd_mod, "load_feature_importances", return_value=[]),
        ):
            _dd_mod._render_deep_dive()

        mock_st.warning.assert_called()
        assert "fold predictions" in mock_st.warning.call_args[0][0].lower()


class TestFeatureImportance:
    def test_renders_chart_for_xgboost(self) -> None:
        mock_st = MagicMock()
        mock_st.session_state = {"selected_run_id": "abc12345-6789"}
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.selectbox.return_value = "All Years (Aggregate)"

        xgb_runs = [
            {
                "run_id": "abc12345-6789",
                "model_type": "xgboost",
                "hyperparameters": {"n_estimators": 100},
                "timestamp": "2025-01-01T00:00:00",
                "git_hash": "abc1234",
                "start_year": 2015,
                "end_year": 2025,
                "prediction_count": 100,
            },
        ]
        importances = [
            {"feature": "elo_delta", "importance": 0.3},
            {"feature": "seed_diff", "importance": 0.7},
        ]

        with (
            patch.object(_dd_mod, "st", mock_st),
            patch.object(_dd_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_dd_mod, "load_available_runs", return_value=xgb_runs),
            patch.object(_dd_mod, "load_fold_predictions", return_value=_sample_fold_predictions()),
            patch.object(_dd_mod, "load_leaderboard_data", return_value=[]),
            patch.object(_dd_mod, "load_feature_importances", return_value=importances),
        ):
            _dd_mod._render_deep_dive()

        # plotly_chart called at least twice: reliability diagram + feature importance
        assert mock_st.plotly_chart.call_count >= 2

    def test_shows_info_for_elo_model(self) -> None:
        mock_st = MagicMock()
        mock_st.session_state = {"selected_run_id": "abc12345-6789"}
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.selectbox.return_value = "All Years (Aggregate)"

        with (
            patch.object(_dd_mod, "st", mock_st),
            patch.object(_dd_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_dd_mod, "load_available_runs", return_value=_sample_runs()),
            patch.object(_dd_mod, "load_fold_predictions", return_value=_sample_fold_predictions()),
            patch.object(_dd_mod, "load_leaderboard_data", return_value=[]),
            patch.object(_dd_mod, "load_feature_importances", return_value=[]),
        ):
            _dd_mod._render_deep_dive()

        # st.info should be called for feature importance section
        info_calls = [call[0][0] for call in mock_st.info.call_args_list]
        assert any("stateful" in msg.lower() for msg in info_calls)
