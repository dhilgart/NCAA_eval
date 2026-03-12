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


class TestRunNotFound:
    def test_shows_warning_when_run_id_not_in_available_runs(self) -> None:
        mock_st = MagicMock()
        mock_st.session_state = {"selected_run_id": "nonexistent-run-id"}

        with (
            patch.object(_dd_mod, "st", mock_st),
            patch.object(_dd_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_dd_mod, "load_available_runs", return_value=[]),
        ):
            _dd_mod._render_deep_dive()

        mock_st.warning.assert_called_once()
        assert "not found" in mock_st.warning.call_args[0][0].lower()


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

    def test_renders_chart_for_elo_model(self) -> None:
        """Story 9.3 AC#3: Elo model shows 'Team Elo Ratings (Top 50)' chart title."""
        mock_st = MagicMock()
        mock_st.session_state = {"selected_run_id": "abc12345-6789"}
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.selectbox.return_value = "All Years (Aggregate)"

        elo_importances = [
            {"feature": "team_1181", "importance": 1650.0},
            {"feature": "team_1242", "importance": 1600.0},
        ]

        with (
            patch.object(_dd_mod, "st", mock_st),
            patch.object(_dd_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_dd_mod, "load_available_runs", return_value=_sample_runs()),
            patch.object(_dd_mod, "load_fold_predictions", return_value=_sample_fold_predictions()),
            patch.object(_dd_mod, "load_leaderboard_data", return_value=[]),
            patch.object(_dd_mod, "load_feature_importances", return_value=elo_importances),
        ):
            _dd_mod._render_deep_dive()

        # Verify plotly_chart was called and figure layout has Elo-specific title and axis
        assert mock_st.plotly_chart.call_count >= 2
        fig_calls = [call[0][0] for call in mock_st.plotly_chart.call_args_list]
        titles = [fig.layout.title.text for fig in fig_calls if hasattr(fig, "layout")]
        assert any("Team Elo Ratings" in (t or "") for t in titles)
        xaxis_titles = [fig.layout.xaxis.title.text for fig in fig_calls if hasattr(fig, "layout")]
        assert any("Rating" in (t or "") for t in xaxis_titles)

    def test_renders_chart_for_logistic_regression(self) -> None:
        """Story 9.3 AC#2: LogReg model shows '|Coefficient|' chart title."""
        mock_st = MagicMock()
        mock_st.session_state = {"selected_run_id": "abc12345-6789"}
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.selectbox.return_value = "All Years (Aggregate)"

        lr_runs = [
            {
                "run_id": "abc12345-6789",
                "model_type": "logistic_regression",
                "hyperparameters": {"C": 1.0},
                "timestamp": "2025-01-01T00:00:00",
                "git_hash": "abc1234",
                "start_year": 2015,
                "end_year": 2025,
                "prediction_count": 100,
            },
        ]
        lr_importances = [
            {"feature": "elo_delta", "importance": 0.8},
            {"feature": "seed_diff", "importance": 0.3},
        ]

        with (
            patch.object(_dd_mod, "st", mock_st),
            patch.object(_dd_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_dd_mod, "load_available_runs", return_value=lr_runs),
            patch.object(_dd_mod, "load_fold_predictions", return_value=_sample_fold_predictions()),
            patch.object(_dd_mod, "load_leaderboard_data", return_value=[]),
            patch.object(_dd_mod, "load_feature_importances", return_value=lr_importances),
        ):
            _dd_mod._render_deep_dive()

        # Verify figure layout has LogReg-specific title and axis label
        assert mock_st.plotly_chart.call_count >= 2
        fig_calls = [call[0][0] for call in mock_st.plotly_chart.call_args_list]
        titles = [fig.layout.title.text for fig in fig_calls if hasattr(fig, "layout")]
        assert any("Coefficient" in (t or "") for t in titles)
        xaxis_titles = [fig.layout.xaxis.title.text for fig in fig_calls if hasattr(fig, "layout")]
        assert any("Absolute Coefficient" in (t or "") for t in xaxis_titles)

    def test_shows_info_when_no_importances(self) -> None:
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

        # st.info should show generic unavailability message
        info_calls = [call[0][0] for call in mock_st.info.call_args_list]
        assert any("not available" in msg.lower() for msg in info_calls)


class TestEnsembleDeepDive:
    """Story 10.3 AC#2: Ensemble components section rendered for ensemble model_type."""

    def test_renders_ensemble_components_expander(self) -> None:
        mock_st = MagicMock()
        mock_st.session_state = {"selected_run_id": "ens12345-6789"}
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.selectbox.return_value = "All Years (Aggregate)"

        ensemble_runs = [
            {
                "run_id": "ens12345-6789",
                "model_type": "ensemble",
                "hyperparameters": {},
                "timestamp": "2025-01-01T00:00:00",
                "git_hash": "abc1234",
                "start_year": 2015,
                "end_year": 2025,
                "prediction_count": 100,
            },
        ]
        importances = [
            {"feature": "Xgboost Prediction", "importance": 0.6},
            {"feature": "seed_diff", "importance": 0.1},
        ]
        manifest = {
            "base_model_types": ["xgboost", "elo"],
            "base_model_count": 2,
            "contextual_features": ["seed_diff", "is_tournament"],
            "meta_learner_type": "logistic_regression",
            "oof_backtest_run_ids": ["oof-run-1", "oof-run-2"],
        }

        with (
            patch.object(_dd_mod, "st", mock_st),
            patch.object(_dd_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_dd_mod, "load_available_runs", return_value=ensemble_runs),
            patch.object(_dd_mod, "load_fold_predictions", return_value=[]),
            patch.object(_dd_mod, "load_leaderboard_data", return_value=[]),
            patch.object(_dd_mod, "load_feature_importances", return_value=importances),
            patch.object(_dd_mod, "load_ensemble_manifest", return_value=manifest),
        ):
            _dd_mod._render_deep_dive()

        # Verify expander was called with "Ensemble Components"
        mock_st.expander.assert_called_once_with("Ensemble Components", expanded=True)

    def test_ensemble_feature_importance_chart_title(self) -> None:
        """Story 10.3 AC#3: Ensemble shows 'Meta-Learner Feature Importance' chart title."""
        mock_st = MagicMock()
        mock_st.session_state = {"selected_run_id": "ens12345-6789"}
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.selectbox.return_value = "All Years (Aggregate)"

        ensemble_runs = [
            {
                "run_id": "ens12345-6789",
                "model_type": "ensemble",
                "hyperparameters": {},
                "timestamp": "2025-01-01T00:00:00",
                "git_hash": "abc1234",
                "start_year": 2015,
                "end_year": 2025,
                "prediction_count": 100,
            },
        ]
        importances = [
            {"feature": "Xgboost Prediction", "importance": 0.6},
            {"feature": "seed_diff", "importance": 0.1},
        ]

        with (
            patch.object(_dd_mod, "st", mock_st),
            patch.object(_dd_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_dd_mod, "load_available_runs", return_value=ensemble_runs),
            patch.object(_dd_mod, "load_fold_predictions", return_value=_sample_fold_predictions()),
            patch.object(_dd_mod, "load_leaderboard_data", return_value=[]),
            patch.object(_dd_mod, "load_feature_importances", return_value=importances),
            patch.object(_dd_mod, "load_ensemble_manifest", return_value={}),
        ):
            _dd_mod._render_deep_dive()

        # Verify feature importance chart has ensemble-specific title
        fig_calls = [call[0][0] for call in mock_st.plotly_chart.call_args_list]
        titles = [fig.layout.title.text for fig in fig_calls if hasattr(fig, "layout")]
        assert any("Meta-Learner" in (t or "") for t in titles)

    def test_no_ensemble_components_for_non_ensemble(self) -> None:
        """Ensemble Components section is NOT rendered for non-ensemble model types."""
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

        # expander should NOT be called for non-ensemble models
        mock_st.expander.assert_not_called()
