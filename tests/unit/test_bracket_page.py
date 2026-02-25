"""Tests for the Bracket Visualizer page rendering."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np

_page_mod = importlib.import_module("dashboard.pages.2_Presentation")


@dataclass(frozen=True)
class _FakeSimResult:
    """Minimal fake for SimulationResult."""

    advancement_probs: np.ndarray
    expected_points: dict[str, np.ndarray]
    method: str
    bracket_distributions: dict[str, object] | None = None


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
    prob_matrix: np.ndarray
    team_labels: dict[int, str]


def _make_sim_data() -> _FakeBracketSimResult:
    """Create a minimal simulation result fixture."""
    n = 4
    adv = np.ones((n, 2), dtype=np.float64) * 0.5
    ep = np.array([10.0, 5.0, 8.0, 3.0])
    P = np.full((n, n), 0.5)
    np.fill_diagonal(P, 0.0)
    return _FakeBracketSimResult(
        sim_result=_FakeSimResult(
            advancement_probs=adv,
            expected_points={"standard": ep},
            method="analytical",
        ),
        bracket=_FakeBracket(
            team_ids=(100, 200, 300, 400),
            team_index_map={100: 0, 200: 1, 300: 2, 400: 3},
            seed_map={100: 1, 200: 16, 300: 1, 400: 16},
        ),
        most_likely=_FakeMostLikely(
            winners=(0, 2, 0),
            champion_team_id=100,
            log_likelihood=-1.5,
        ),
        prob_matrix=P,
        team_labels={0: "[1] Duke", 1: "[16] Norfolk St", 2: "[1] UConn", 3: "[16] Wagner"},
    )


class TestNoRunSelected:
    def test_shows_info_when_no_run(self) -> None:
        mock_st = MagicMock()
        mock_st.session_state = {}
        mock_st.columns.return_value = [MagicMock(), MagicMock()]

        with patch.object(_page_mod, "st", mock_st):
            _page_mod._render_bracket_page()

        mock_st.info.assert_called()
        info_msgs = [call[0][0] for call in mock_st.info.call_args_list]
        assert any("model run" in msg.lower() for msg in info_msgs)


class TestNoYearSelected:
    def test_shows_info_when_no_year(self) -> None:
        mock_st = MagicMock()
        mock_st.session_state = {"selected_run_id": "abc123"}
        mock_st.columns.return_value = [MagicMock(), MagicMock()]

        with patch.object(_page_mod, "st", mock_st):
            _page_mod._render_bracket_page()

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
        mock_st.columns.return_value = [MagicMock(), MagicMock()]

        with (
            patch.object(_page_mod, "st", mock_st),
            patch.object(_page_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_page_mod, "load_tourney_seeds", return_value=[]),
        ):
            _page_mod._render_bracket_page()

        mock_st.warning.assert_called()
        assert "seeds" in mock_st.warning.call_args[0][0].lower()


class TestSimulationFails:
    def test_shows_warning_when_simulation_returns_none(self) -> None:
        mock_st = MagicMock()
        mock_st.session_state = {
            "selected_run_id": "abc123",
            "selected_year": 2023,
            "selected_scoring": "standard",
            "bracket_sim_method": "analytical",
        }
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.selectbox.return_value = "analytical"

        with (
            patch.object(_page_mod, "st", mock_st),
            patch.object(_page_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_page_mod, "load_tourney_seeds", return_value=[{"seed": "W01"}]),
            patch.object(_page_mod, "run_bracket_simulation", return_value=None),
        ):
            _page_mod._render_bracket_page()

        mock_st.warning.assert_called()
        assert "simulate" in mock_st.warning.call_args[0][0].lower()


class TestSuccessfulRender:
    def test_renders_champion_and_charts(self) -> None:
        mock_st = MagicMock()
        mock_st.session_state = {
            "selected_run_id": "abc123",
            "selected_year": 2023,
            "selected_scoring": "standard",
            "bracket_sim_method": "analytical",
        }
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        # First selectbox call = simulation method, subsequent = pairwise team selectors
        mock_st.selectbox.side_effect = ["analytical", "[1] Duke", "[16] Norfolk St"]

        sim_data = _make_sim_data()
        mock_heatmap = MagicMock()
        mock_components = MagicMock()

        with (
            patch.object(_page_mod, "st", mock_st),
            patch.object(_page_mod, "components", mock_components),
            patch.object(_page_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_page_mod, "load_tourney_seeds", return_value=[{"seed": "W01"}]),
            patch.object(_page_mod, "run_bracket_simulation", return_value=sim_data),
            patch.object(_page_mod, "render_bracket_html", return_value="<html></html>"),
            patch.object(_page_mod, "plot_advancement_heatmap", return_value=mock_heatmap),
        ):
            _page_mod._render_bracket_page()

        # Should show champion success message
        mock_st.success.assert_called_once()
        assert "Duke" in mock_st.success.call_args[0][0]

        # Should render bracket HTML
        mock_components.html.assert_called_once()

        # Should render heatmap
        mock_st.plotly_chart.assert_called()

        # Should render EP dataframe
        mock_st.dataframe.assert_called()

        # Should render pairwise win probability expander (AC #5)
        mock_st.expander.assert_called()
        expander_title = mock_st.expander.call_args[0][0]
        assert "Pairwise" in expander_title


class TestMCModeRender:
    """Verify AC #4: Monte Carlo score distribution renders correctly."""

    def _make_mc_sim_data(self) -> _FakeBracketSimResult:
        """Create a simulation result fixture with MC mode outputs."""
        n = 4
        adv = np.ones((n, 2), dtype=np.float64) * 0.5
        ep = np.array([10.0, 5.0, 8.0, 3.0])
        P = np.full((n, n), 0.5)
        np.fill_diagonal(P, 0.0)
        mock_dist = MagicMock()
        return _FakeBracketSimResult(
            sim_result=_FakeSimResult(
                advancement_probs=adv,
                expected_points={"standard": ep},
                method="monte_carlo",
                bracket_distributions={"standard": mock_dist},
            ),
            bracket=_FakeBracket(
                team_ids=(100, 200, 300, 400),
                team_index_map={100: 0, 200: 1, 300: 2, 400: 3},
                seed_map={100: 1, 200: 16, 300: 1, 400: 16},
            ),
            most_likely=_FakeMostLikely(
                winners=(0, 2, 0),
                champion_team_id=100,
                log_likelihood=-1.5,
            ),
            prob_matrix=P,
            team_labels={0: "[1] Duke", 1: "[16] Norfolk St", 2: "[1] UConn", 3: "[16] Wagner"},
        )

    def test_mc_renders_score_distribution(self) -> None:
        mock_st = MagicMock()
        mock_st.session_state = {
            "selected_run_id": "abc123",
            "selected_year": 2023,
            "selected_scoring": "standard",
        }
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.selectbox.side_effect = ["monte_carlo", "[1] Duke", "[16] Norfolk St"]

        sim_data = self._make_mc_sim_data()
        mock_dist_fig = MagicMock()
        mock_components = MagicMock()
        mock_plot_score = MagicMock(return_value=mock_dist_fig)

        with (
            patch.object(_page_mod, "st", mock_st),
            patch.object(_page_mod, "components", mock_components),
            patch.object(_page_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_page_mod, "load_tourney_seeds", return_value=[{"seed": "W01"}]),
            patch.object(_page_mod, "run_bracket_simulation", return_value=sim_data),
            patch.object(_page_mod, "render_bracket_html", return_value="<html></html>"),
            patch.object(_page_mod, "plot_advancement_heatmap", return_value=MagicMock()),
            patch.object(_page_mod, "plot_score_distribution", mock_plot_score),
        ):
            _page_mod._render_bracket_page()

        # Score distribution chart must be rendered via plotly_chart
        mock_st.plotly_chart.assert_called()
        # plot_score_distribution must have been called (AC #4)
        mock_plot_score.assert_called_once()

    def test_mc_missing_scoring_key_shows_info(self) -> None:
        """When bracket_distributions lacks the scoring key, st.info is shown (L2 fix)."""
        mock_st = MagicMock()
        mock_st.session_state = {
            "selected_run_id": "abc123",
            "selected_year": 2023,
            "selected_scoring": "fibonacci",  # not in bracket_distributions
        }
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.selectbox.side_effect = ["monte_carlo", "[1] Duke", "[16] Norfolk St"]

        # MC result but bracket_distributions only has "standard", not "fibonacci"
        n = 4
        adv = np.ones((n, 2), dtype=np.float64) * 0.5
        ep = np.array([10.0, 5.0, 8.0, 3.0])
        P = np.full((n, n), 0.5)
        np.fill_diagonal(P, 0.0)
        sim_data = _FakeBracketSimResult(
            sim_result=_FakeSimResult(
                advancement_probs=adv,
                expected_points={"fibonacci": ep},
                method="monte_carlo",
                bracket_distributions={"standard": MagicMock()},  # fibonacci missing
            ),
            bracket=_FakeBracket(
                team_ids=(100, 200, 300, 400),
                team_index_map={100: 0, 200: 1, 300: 2, 400: 3},
                seed_map={100: 1, 200: 16, 300: 1, 400: 16},
            ),
            most_likely=_FakeMostLikely(
                winners=(0, 2, 0),
                champion_team_id=100,
                log_likelihood=-1.5,
            ),
            prob_matrix=P,
            team_labels={0: "[1] Duke", 1: "[16] Norfolk St", 2: "[1] UConn", 3: "[16] Wagner"},
        )

        with (
            patch.object(_page_mod, "st", mock_st),
            patch.object(_page_mod, "components", MagicMock()),
            patch.object(_page_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_page_mod, "load_tourney_seeds", return_value=[{"seed": "W01"}]),
            patch.object(_page_mod, "run_bracket_simulation", return_value=sim_data),
            patch.object(_page_mod, "render_bracket_html", return_value="<html></html>"),
            patch.object(_page_mod, "plot_advancement_heatmap", return_value=MagicMock()),
        ):
            _page_mod._render_bracket_page()

        # st.info must have been called for the missing scoring key in MC mode
        mock_st.info.assert_called()
        info_msgs = [call[0][0] for call in mock_st.info.call_args_list]
        assert any("fibonacci" in msg for msg in info_msgs)
