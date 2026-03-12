"""Tests for the Pool Scorer page rendering."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt

_page_mod = importlib.import_module("dashboard.pages.4_Pool_Scorer")


@dataclass(frozen=True)
class _FakeDistribution:
    """Minimal fake for BracketDistribution."""

    scores: npt.NDArray[np.float64]
    percentiles: dict[int, float]
    mean: float
    std: float
    histogram_bins: npt.NDArray[np.float64]
    histogram_counts: npt.NDArray[np.int64]


@dataclass(frozen=True)
class _FakeSimResult:
    """Minimal fake for SimulationResult with sim_winners."""

    method: str
    sim_winners: npt.NDArray[np.int32] | None = None


@dataclass(frozen=True)
class _FakeBracket:
    """Minimal fake for BracketStructure."""

    team_ids: tuple[int, ...]
    team_index_map: dict[int, int]
    seed_map: dict[int, int]


@dataclass(frozen=True)
class _FakeMostLikely:
    """Minimal fake for MostLikelyBracket."""

    winners: tuple[int, ...]
    champion_team_id: int
    log_likelihood: float


@dataclass(frozen=True)
class _FakeBracketSimResult:
    """Minimal fake for BracketSimulationResult."""

    sim_result: _FakeSimResult
    bracket: _FakeBracket
    most_likely: _FakeMostLikely
    prob_matrix: npt.NDArray[np.float64]
    team_labels: dict[int, str]


def _make_sim_data() -> _FakeBracketSimResult:
    """Create a minimal MC simulation result fixture."""
    n = 4
    n_sims = 100
    P = np.full((n, n), 0.5)
    np.fill_diagonal(P, 0.0)
    sim_winners = np.random.default_rng(42).integers(0, n, size=(n_sims, 3)).astype(np.int32)
    return _FakeBracketSimResult(
        sim_result=_FakeSimResult(
            method="monte_carlo",
            sim_winners=sim_winners,
        ),
        bracket=_FakeBracket(
            team_ids=(100, 200, 300, 400),
            team_index_map={100: 0, 200: 1, 300: 2, 400: 3},
            seed_map={100: 1, 200: 16, 300: 2, 400: 15},
        ),
        most_likely=_FakeMostLikely(
            winners=(0, 2, 0),
            champion_team_id=100,
            log_likelihood=-1.5,
        ),
        prob_matrix=P,
        team_labels={0: "[1] Duke", 1: "[16] Norfolk St", 2: "[2] UConn", 3: "[15] Wagner"},
    )


def _make_distribution() -> _FakeDistribution:
    """Create a minimal BracketDistribution fixture."""
    scores = np.array([50.0, 60.0, 70.0, 80.0, 90.0] * 20)
    return _FakeDistribution(
        scores=scores,
        percentiles={5: 52.0, 25: 60.0, 50: 70.0, 75: 80.0, 95: 88.0},
        mean=70.0,
        std=12.5,
        histogram_bins=np.linspace(50, 90, 11),
        histogram_counts=np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype=np.int64),
    )


class TestNoRunSelected:
    def test_shows_info_when_no_run(self) -> None:
        mock_st = MagicMock()
        mock_st.session_state = {}
        mock_st.columns.side_effect = lambda n: [
            MagicMock() for _ in range(n if isinstance(n, int) else len(n))
        ]

        with patch.object(_page_mod, "st", mock_st):
            _page_mod._render_pool_scorer_page()

        mock_st.info.assert_called()
        info_msgs = [call[0][0] for call in mock_st.info.call_args_list]
        assert any("model run" in msg.lower() for msg in info_msgs)


class TestNoYearSelected:
    def test_shows_info_when_no_year(self) -> None:
        mock_st = MagicMock()
        mock_st.session_state = {"selected_run_id": "abc123"}
        mock_st.columns.side_effect = lambda n: [
            MagicMock() for _ in range(n if isinstance(n, int) else len(n))
        ]

        with patch.object(_page_mod, "st", mock_st):
            _page_mod._render_pool_scorer_page()

        mock_st.info.assert_called()
        info_msgs = [call[0][0] for call in mock_st.info.call_args_list]
        assert any("year" in msg.lower() for msg in info_msgs)


class TestNoSeedsAvailable:
    def test_shows_warning_when_no_seeds(self) -> None:
        mock_st = MagicMock()
        mock_st.session_state = {
            "selected_run_id": "abc123",
            "selected_year": 2023,
            "selected_scoring": "standard",
        }
        mock_st.columns.side_effect = lambda n: [
            MagicMock() for _ in range(n if isinstance(n, int) else len(n))
        ]

        with (
            patch.object(_page_mod, "st", mock_st),
            patch.object(_page_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_page_mod, "load_tourney_seeds", return_value=[]),
        ):
            _page_mod._render_pool_scorer_page()

        mock_st.warning.assert_called()
        assert "seeds" in mock_st.warning.call_args[0][0].lower()


class TestSuccessfulRender:
    def test_renders_metrics_and_chart_when_sim_cached(self) -> None:
        sim_data = _make_sim_data()
        dist = _make_distribution()
        mock_dist_fig = MagicMock()
        mock_scoring_rule = MagicMock()
        mock_scoring_rule.name = "standard"

        mock_st = MagicMock()
        mock_st.session_state = {
            "selected_run_id": "abc123",
            "selected_year": 2023,
            "selected_scoring": "standard",
            "pool_sim_data": sim_data,
            "pool_sim_key": ("abc123", 2023, 10_000),
            "pool_use_custom": False,
        }
        # columns() may be called with varying sizes (2, 3, 4, 6)
        mock_st.columns.side_effect = lambda n: [
            MagicMock() for _ in range(n if isinstance(n, int) else len(n))
        ]
        mock_st.checkbox.return_value = False
        mock_st.slider.return_value = 10_000
        mock_st.button.return_value = False

        with (
            patch.object(_page_mod, "st", mock_st),
            patch.object(_page_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_page_mod, "load_tourney_seeds", return_value=[{"seed": "W01"}]),
            patch.object(
                _page_mod,
                "score_chosen_bracket",
                return_value={"standard": dist},
            ),
            patch.object(_page_mod, "get_scoring") as mock_get_scoring,
            patch.object(_page_mod, "plot_score_distribution", return_value=mock_dist_fig),
            patch.object(
                _page_mod,
                "export_bracket_csv",
                return_value="game_number,round\n1,R64\n",
            ),
            patch.object(_page_mod, "get_overrides", return_value={}),
            patch.object(_page_mod, "apply_overrides", return_value=sim_data.most_likely),
        ):
            mock_get_scoring.return_value = MagicMock(return_value=mock_scoring_rule)
            _page_mod._render_pool_scorer_page()

        # Outcome summary metrics should be rendered
        mock_st.metric.assert_called()
        metric_labels = [call[0][0] for call in mock_st.metric.call_args_list]
        assert "Median" in metric_labels
        assert "Mean" in metric_labels

        # Score distribution chart should be rendered
        mock_st.plotly_chart.assert_called()

        # CSV download button should be rendered
        mock_st.download_button.assert_called_once()
        dl_kwargs = mock_st.download_button.call_args
        assert dl_kwargs[1]["file_name"] == "bracket_submission.csv" or (
            len(dl_kwargs[0]) > 0 and "bracket" in str(dl_kwargs)
        )


class TestCustomScoringPath:
    def test_renders_with_custom_scoring_when_checked(self) -> None:
        """Custom scoring path: build_custom_scoring called, get_scoring NOT called."""
        sim_data = _make_sim_data()
        dist = _make_distribution()
        mock_dist_fig = MagicMock()
        mock_custom_rule = MagicMock()
        mock_custom_rule.name = "custom"

        mock_st = MagicMock()
        mock_st.session_state = {
            "selected_run_id": "abc123",
            "selected_year": 2023,
            "selected_scoring": "standard",
            "pool_sim_data": sim_data,
            "pool_sim_key": ("abc123", 2023, 10_000),
            "pool_use_custom": True,
        }
        mock_st.columns.side_effect = lambda n: [
            MagicMock() for _ in range(n if isinstance(n, int) else len(n))
        ]
        mock_st.checkbox.return_value = True
        mock_st.number_input.return_value = 4
        mock_st.slider.return_value = 10_000
        mock_st.button.return_value = False

        with (
            patch.object(_page_mod, "st", mock_st),
            patch.object(_page_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_page_mod, "load_tourney_seeds", return_value=[{"seed": "W01"}]),
            patch.object(
                _page_mod,
                "score_chosen_bracket",
                return_value={"custom": dist},
            ),
            patch.object(_page_mod, "build_custom_scoring", return_value=mock_custom_rule) as mock_build,
            patch.object(_page_mod, "get_scoring") as mock_get_scoring,
            patch.object(_page_mod, "plot_score_distribution", return_value=mock_dist_fig),
            patch.object(
                _page_mod,
                "export_bracket_csv",
                return_value="game_number,round\n1,R64\n",
            ),
            patch.object(_page_mod, "get_overrides", return_value={}),
            patch.object(_page_mod, "apply_overrides", return_value=sim_data.most_likely),
        ):
            _page_mod._render_pool_scorer_page()

        # build_custom_scoring should be called (not get_scoring)
        mock_build.assert_called_once()
        mock_get_scoring.assert_not_called()

        # Metrics and chart should still render
        mock_st.metric.assert_called()
        mock_st.plotly_chart.assert_called()


class TestAnalyzeOutcomesButton:
    def test_button_triggers_simulation_with_progress(self) -> None:
        mock_st = MagicMock()
        mock_st.session_state = {
            "selected_run_id": "abc123",
            "selected_year": 2023,
            "selected_scoring": "standard",
        }
        mock_st.columns.side_effect = lambda n: [
            MagicMock() for _ in range(n if isinstance(n, int) else len(n))
        ]
        mock_st.checkbox.return_value = False
        mock_st.slider.return_value = 10_000
        mock_st.button.return_value = True  # Button clicked

        sim_data = _make_sim_data()

        with (
            patch.object(_page_mod, "st", mock_st),
            patch.object(_page_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_page_mod, "load_tourney_seeds", return_value=[{"seed": "W01"}]),
            patch.object(
                _page_mod, "run_bracket_simulation_with_progress", return_value=sim_data
            ) as mock_sim,
        ):
            _page_mod._render_pool_scorer_page()

        # run_bracket_simulation_with_progress should have been called
        mock_sim.assert_called_once()
        call_kwargs = mock_sim.call_args[1]
        assert call_kwargs["n_simulations"] == 10_000

    def test_progress_bar_used_instead_of_spinner(self) -> None:
        """st.progress should be called for MC, not st.spinner."""
        mock_st = MagicMock()
        mock_progress_bar = MagicMock()
        mock_st.progress.return_value = mock_progress_bar
        mock_st.session_state = {
            "selected_run_id": "abc123",
            "selected_year": 2023,
            "selected_scoring": "standard",
        }
        mock_st.columns.side_effect = lambda n: [
            MagicMock() for _ in range(n if isinstance(n, int) else len(n))
        ]
        mock_st.checkbox.return_value = False
        mock_st.slider.return_value = 10_000
        mock_st.button.return_value = True

        sim_data = _make_sim_data()

        with (
            patch.object(_page_mod, "st", mock_st),
            patch.object(_page_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_page_mod, "load_tourney_seeds", return_value=[{"seed": "W01"}]),
            patch.object(_page_mod, "run_bracket_simulation_with_progress", return_value=sim_data),
        ):
            _page_mod._render_pool_scorer_page()

        # st.progress should be called (not st.spinner for MC path)
        mock_st.progress.assert_called()
        # progress bar should be cleared after simulation
        mock_progress_bar.empty.assert_called_once()


class TestSimulationFailure:
    def test_shows_warning_when_simulation_returns_none(self) -> None:
        mock_st = MagicMock()
        mock_st.session_state = {
            "selected_run_id": "abc123",
            "selected_year": 2023,
            "selected_scoring": "standard",
        }
        mock_st.columns.side_effect = lambda n: [
            MagicMock() for _ in range(n if isinstance(n, int) else len(n))
        ]
        mock_st.checkbox.return_value = False
        mock_st.slider.return_value = 10_000
        mock_st.button.return_value = True  # Button clicked

        with (
            patch.object(_page_mod, "st", mock_st),
            patch.object(_page_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_page_mod, "load_tourney_seeds", return_value=[{"seed": "W01"}]),
            patch.object(_page_mod, "run_bracket_simulation_with_progress", return_value=None),
        ):
            _page_mod._render_pool_scorer_page()

        mock_st.warning.assert_called()
        warning_msgs = [call[0][0] for call in mock_st.warning.call_args_list]
        assert any("simulate" in msg.lower() for msg in warning_msgs)


class TestNoSimWinners:
    def test_shows_error_when_sim_winners_is_none(self) -> None:
        """Analytical-only sim has no sim_winners — should show error."""
        mock_st = MagicMock()
        mock_st.session_state = {
            "selected_run_id": "abc123",
            "selected_year": 2023,
            "selected_scoring": "standard",
        }
        mock_st.columns.side_effect = lambda n: [
            MagicMock() for _ in range(n if isinstance(n, int) else len(n))
        ]
        mock_st.checkbox.return_value = False
        mock_st.slider.return_value = 10_000
        mock_st.button.return_value = True

        # Simulation result without sim_winners (analytical mode)
        no_mc_sim = _FakeBracketSimResult(
            sim_result=_FakeSimResult(method="analytical", sim_winners=None),
            bracket=_FakeBracket(
                team_ids=(100, 200),
                team_index_map={100: 0, 200: 1},
                seed_map={100: 1, 200: 16},
            ),
            most_likely=_FakeMostLikely(winners=(0,), champion_team_id=100, log_likelihood=-0.5),
            prob_matrix=np.full((2, 2), 0.5),
            team_labels={0: "[1] Duke", 1: "[16] Norfolk St"},
        )

        with (
            patch.object(_page_mod, "st", mock_st),
            patch.object(_page_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_page_mod, "load_tourney_seeds", return_value=[{"seed": "W01"}]),
            patch.object(_page_mod, "run_bracket_simulation_with_progress", return_value=no_mc_sim),
        ):
            _page_mod._render_pool_scorer_page()

        mock_st.error.assert_called()
        error_msgs = [call[0][0] for call in mock_st.error.call_args_list]
        assert any("sim_winners" in msg.lower() for msg in error_msgs)


class TestOverrideIntegration:
    """Pool Scorer applies bracket overrides from session state (Story 9.8)."""

    def test_override_info_displayed_when_overrides_exist(self) -> None:
        sim_data = _make_sim_data()
        dist = _make_distribution()
        mock_dist_fig = MagicMock()
        mock_scoring_rule = MagicMock()
        mock_scoring_rule.name = "standard"

        # Edited bracket with game 0 overridden: team 1 wins instead of team 0
        edited_ml = _FakeMostLikely(winners=(1, 2, 2), champion_team_id=300, log_likelihood=-2.0)

        mock_st = MagicMock()
        mock_st.session_state = {
            "selected_run_id": "abc123",
            "selected_year": 2023,
            "selected_scoring": "standard",
            "pool_sim_data": sim_data,
            "pool_sim_key": ("abc123", 2023, 10_000),
            "pool_use_custom": False,
        }
        mock_st.columns.side_effect = lambda n: [
            MagicMock() for _ in range(n if isinstance(n, int) else len(n))
        ]
        mock_st.checkbox.return_value = False
        mock_st.slider.return_value = 10_000
        mock_st.button.return_value = False

        with (
            patch.object(_page_mod, "st", mock_st),
            patch.object(_page_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_page_mod, "load_tourney_seeds", return_value=[{"seed": "W01"}]),
            patch.object(_page_mod, "score_chosen_bracket", return_value={"standard": dist}),
            patch.object(_page_mod, "get_scoring") as mock_get_scoring,
            patch.object(_page_mod, "plot_score_distribution", return_value=mock_dist_fig),
            patch.object(_page_mod, "export_bracket_csv", return_value="game_number,round\n1,R64\n"),
            patch.object(_page_mod, "get_overrides", return_value={0: 1}),
            patch.object(_page_mod, "apply_overrides", return_value=edited_ml),
        ):
            mock_get_scoring.return_value = MagicMock(return_value=mock_scoring_rule)
            _page_mod._render_pool_scorer_page()

        # Override count info should be displayed
        info_msgs = [call[0][0] for call in mock_st.info.call_args_list]
        assert any("1 of 3 picks overridden" in msg for msg in info_msgs)

    def test_chosen_winners_passed_to_score_chosen_bracket(self) -> None:
        sim_data = _make_sim_data()
        dist = _make_distribution()
        mock_dist_fig = MagicMock()
        mock_scoring_rule = MagicMock()
        mock_scoring_rule.name = "standard"

        edited_ml = _FakeMostLikely(winners=(1, 2, 2), champion_team_id=300, log_likelihood=-2.0)

        mock_st = MagicMock()
        mock_st.session_state = {
            "selected_run_id": "abc123",
            "selected_year": 2023,
            "selected_scoring": "standard",
            "pool_sim_data": sim_data,
            "pool_sim_key": ("abc123", 2023, 10_000),
            "pool_use_custom": False,
        }
        mock_st.columns.side_effect = lambda n: [
            MagicMock() for _ in range(n if isinstance(n, int) else len(n))
        ]
        mock_st.checkbox.return_value = False
        mock_st.slider.return_value = 10_000
        mock_st.button.return_value = False

        with (
            patch.object(_page_mod, "st", mock_st),
            patch.object(_page_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_page_mod, "load_tourney_seeds", return_value=[{"seed": "W01"}]),
            patch.object(_page_mod, "score_chosen_bracket", return_value={"standard": dist}) as mock_score,
            patch.object(_page_mod, "get_scoring") as mock_get_scoring,
            patch.object(_page_mod, "plot_score_distribution", return_value=mock_dist_fig),
            patch.object(_page_mod, "export_bracket_csv", return_value="game_number,round\n1,R64\n"),
            patch.object(_page_mod, "get_overrides", return_value={0: 1}),
            patch.object(_page_mod, "apply_overrides", return_value=edited_ml),
        ):
            mock_get_scoring.return_value = MagicMock(return_value=mock_scoring_rule)
            _page_mod._render_pool_scorer_page()

        # score_chosen_bracket should receive chosen_winners from edited bracket
        mock_score.assert_called_once()
        call_kwargs = mock_score.call_args[1]
        assert call_kwargs["chosen_winners"] == (1, 2, 2)

    def test_export_uses_edited_bracket(self) -> None:
        sim_data = _make_sim_data()
        dist = _make_distribution()
        mock_dist_fig = MagicMock()
        mock_scoring_rule = MagicMock()
        mock_scoring_rule.name = "standard"

        edited_ml = _FakeMostLikely(winners=(1, 2, 2), champion_team_id=300, log_likelihood=-2.0)

        mock_st = MagicMock()
        mock_st.session_state = {
            "selected_run_id": "abc123",
            "selected_year": 2023,
            "selected_scoring": "standard",
            "pool_sim_data": sim_data,
            "pool_sim_key": ("abc123", 2023, 10_000),
            "pool_use_custom": False,
        }
        mock_st.columns.side_effect = lambda n: [
            MagicMock() for _ in range(n if isinstance(n, int) else len(n))
        ]
        mock_st.checkbox.return_value = False
        mock_st.slider.return_value = 10_000
        mock_st.button.return_value = False

        with (
            patch.object(_page_mod, "st", mock_st),
            patch.object(_page_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_page_mod, "load_tourney_seeds", return_value=[{"seed": "W01"}]),
            patch.object(_page_mod, "score_chosen_bracket", return_value={"standard": dist}),
            patch.object(_page_mod, "get_scoring") as mock_get_scoring,
            patch.object(_page_mod, "plot_score_distribution", return_value=mock_dist_fig),
            patch.object(
                _page_mod, "export_bracket_csv", return_value="game_number,round\n1,R64\n"
            ) as mock_export,
            patch.object(_page_mod, "get_overrides", return_value={0: 1}),
            patch.object(_page_mod, "apply_overrides", return_value=edited_ml),
        ):
            mock_get_scoring.return_value = MagicMock(return_value=mock_scoring_rule)
            _page_mod._render_pool_scorer_page()

        # export_bracket_csv should receive the edited bracket, not sim_data.most_likely
        mock_export.assert_called_once()
        export_args = mock_export.call_args[0]
        assert export_args[1] == edited_ml

    def test_invalidation_check_called_on_every_render(self) -> None:
        """Pool Scorer must call check_invalidation() so stale overrides are cleared (AC #4)."""
        mock_st = MagicMock()
        mock_st.session_state = {
            "selected_run_id": "abc123",
            "selected_year": 2023,
            "selected_scoring": "standard",
        }
        mock_st.columns.side_effect = lambda n: [
            MagicMock() for _ in range(n if isinstance(n, int) else len(n))
        ]
        mock_st.checkbox.return_value = False
        mock_st.slider.return_value = 10_000
        mock_st.button.return_value = False

        with (
            patch.object(_page_mod, "st", mock_st),
            patch.object(_page_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_page_mod, "load_tourney_seeds", return_value=[{"seed": "W01"}]),
            patch.object(_page_mod, "check_invalidation", return_value=False) as mock_invalidation,
        ):
            _page_mod._render_pool_scorer_page()

        # check_invalidation must have been called with the current parameters
        mock_invalidation.assert_called_once_with("abc123", 2023, "standard", 0, 0)

    def test_invalidation_shows_info_message(self) -> None:
        """When overrides are invalidated, Pool Scorer shows an info message."""
        mock_st = MagicMock()
        mock_st.session_state = {
            "selected_run_id": "abc123",
            "selected_year": 2023,
            "selected_scoring": "standard",
        }
        mock_st.columns.side_effect = lambda n: [
            MagicMock() for _ in range(n if isinstance(n, int) else len(n))
        ]
        mock_st.checkbox.return_value = False
        mock_st.slider.return_value = 10_000
        mock_st.button.return_value = False

        with (
            patch.object(_page_mod, "st", mock_st),
            patch.object(_page_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_page_mod, "load_tourney_seeds", return_value=[{"seed": "W01"}]),
            patch.object(_page_mod, "check_invalidation", return_value=True),
        ):
            _page_mod._render_pool_scorer_page()

        info_msgs = [call[0][0] for call in mock_st.info.call_args_list]
        assert any("overrides" in msg.lower() and "reset" in msg.lower() for msg in info_msgs)
