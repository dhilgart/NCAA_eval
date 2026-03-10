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


def _override_patches() -> list[tuple[object, str, object]]:
    """Return list of (target, attr, mock) tuples for override function mocks."""
    return [
        (_page_mod, "check_invalidation", MagicMock(return_value=False)),
        (_page_mod, "get_overrides", MagicMock(return_value={})),
        (_page_mod, "apply_overrides", MagicMock(side_effect=lambda ml, ov, br, pm: ml)),
        (_page_mod, "clear_overrides", MagicMock()),
        (_page_mod, "set_override", MagicMock()),
    ]


class TestNoRunSelected:
    def test_shows_info_when_no_run(self) -> None:
        mock_st = MagicMock()
        mock_st.session_state = {}
        mock_st.columns.side_effect = lambda *a, **kw: [
            MagicMock() for _ in range(a[0] if isinstance(a[0], int) else len(a[0]))
        ]

        with patch.object(_page_mod, "st", mock_st):
            _page_mod._render_bracket_page()

        mock_st.info.assert_called()
        info_msgs = [call[0][0] for call in mock_st.info.call_args_list]
        assert any("model run" in msg.lower() for msg in info_msgs)


class TestNoYearSelected:
    def test_shows_info_when_no_year(self) -> None:
        mock_st = MagicMock()
        mock_st.session_state = {"selected_run_id": "abc123"}
        mock_st.columns.side_effect = lambda *a, **kw: [
            MagicMock() for _ in range(a[0] if isinstance(a[0], int) else len(a[0]))
        ]

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
        mock_st.columns.side_effect = lambda *a, **kw: [
            MagicMock() for _ in range(a[0] if isinstance(a[0], int) else len(a[0]))
        ]

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
        mock_st.columns.side_effect = lambda *a, **kw: [
            MagicMock() for _ in range(a[0] if isinstance(a[0], int) else len(a[0]))
        ]
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
        mock_st.columns.side_effect = lambda *a, **kw: [
            MagicMock() for _ in range(a[0] if isinstance(a[0], int) else len(a[0]))
        ]
        # selectbox: sim method, pairwise team A, pairwise team B, + 3 edit picks (4-team bracket)
        mock_st.selectbox.side_effect = [
            "analytical",
            "[1] Duke",
            "[16] Norfolk St",
            "[1] Duke",  # edit pick game 1
            "[1] UConn",  # edit pick game 2
            "[1] Duke",  # edit pick game 3 (championship)
        ]

        sim_data = _make_sim_data()
        mock_heatmap = MagicMock()
        mock_components = MagicMock()

        patches = _override_patches()
        with (
            patch.object(_page_mod, "st", mock_st),
            patch.object(_page_mod, "components", mock_components),
            patch.object(_page_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_page_mod, "load_tourney_seeds", return_value=[{"seed": "W01"}]),
            patch.object(_page_mod, "run_bracket_simulation", return_value=sim_data),
            patch.object(_page_mod, "render_bracket_html", return_value="<html></html>"),
            patch.object(_page_mod, "plot_advancement_heatmap", return_value=mock_heatmap),
            patch.object(*patches[0]),
            patch.object(*patches[1]),
            patch.object(*patches[2]),
            patch.object(*patches[3]),
            patch.object(*patches[4]),
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
        expander_calls = [call[0][0] for call in mock_st.expander.call_args_list]
        assert any("Pairwise" in title for title in expander_calls)


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
        mock_st.columns.side_effect = lambda *a, **kw: [
            MagicMock() for _ in range(a[0] if isinstance(a[0], int) else len(a[0]))
        ]
        mock_st.selectbox.side_effect = [
            "monte_carlo",
            "[1] Duke",
            "[16] Norfolk St",
            "[1] Duke",
            "[1] UConn",
            "[1] Duke",
        ]

        sim_data = self._make_mc_sim_data()
        mock_dist_fig = MagicMock()
        mock_components = MagicMock()
        mock_plot_score = MagicMock(return_value=mock_dist_fig)

        patches = _override_patches()
        with (
            patch.object(_page_mod, "st", mock_st),
            patch.object(_page_mod, "components", mock_components),
            patch.object(_page_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_page_mod, "load_tourney_seeds", return_value=[{"seed": "W01"}]),
            patch.object(_page_mod, "run_bracket_simulation", return_value=sim_data),
            patch.object(_page_mod, "render_bracket_html", return_value="<html></html>"),
            patch.object(_page_mod, "plot_advancement_heatmap", return_value=MagicMock()),
            patch.object(_page_mod, "plot_score_distribution", mock_plot_score),
            patch.object(*patches[0]),
            patch.object(*patches[1]),
            patch.object(*patches[2]),
            patch.object(*patches[3]),
            patch.object(*patches[4]),
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
        mock_st.columns.side_effect = lambda *a, **kw: [
            MagicMock() for _ in range(a[0] if isinstance(a[0], int) else len(a[0]))
        ]
        mock_st.selectbox.side_effect = [
            "monte_carlo",
            "[1] Duke",
            "[16] Norfolk St",
            "[1] Duke",
            "[1] UConn",
            "[1] Duke",
        ]

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

        patches = _override_patches()
        with (
            patch.object(_page_mod, "st", mock_st),
            patch.object(_page_mod, "components", MagicMock()),
            patch.object(_page_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_page_mod, "load_tourney_seeds", return_value=[{"seed": "W01"}]),
            patch.object(_page_mod, "run_bracket_simulation", return_value=sim_data),
            patch.object(_page_mod, "render_bracket_html", return_value="<html></html>"),
            patch.object(_page_mod, "plot_advancement_heatmap", return_value=MagicMock()),
            patch.object(*patches[0]),
            patch.object(*patches[1]),
            patch.object(*patches[2]),
            patch.object(*patches[3]),
            patch.object(*patches[4]),
        ):
            _page_mod._render_bracket_page()

        # st.info must have been called for the missing scoring key in MC mode
        mock_st.info.assert_called()
        info_msgs = [call[0][0] for call in mock_st.info.call_args_list]
        assert any("fibonacci" in msg for msg in info_msgs)


class TestSliderValuesPassThrough:
    """Verify slider values from session state flow to run_bracket_simulation."""

    def test_slider_values_passed_to_simulation(self) -> None:
        mock_st = MagicMock()
        mock_st.session_state = {
            "selected_run_id": "abc123",
            "selected_year": 2023,
            "selected_scoring": "standard",
        }
        mock_st.columns.side_effect = lambda *a, **kw: [
            MagicMock() for _ in range(a[0] if isinstance(a[0], int) else len(a[0]))
        ]
        # Simulate non-neutral slider values
        mock_st.selectbox.side_effect = ["analytical"]
        mock_st.slider.side_effect = [3, 25]  # upset_aggression=3, seed_weight_pct=25

        mock_run_sim = MagicMock(return_value=None)

        with (
            patch.object(_page_mod, "st", mock_st),
            patch.object(_page_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_page_mod, "load_tourney_seeds", return_value=[{"seed": "W01"}]),
            patch.object(_page_mod, "run_bracket_simulation", mock_run_sim),
        ):
            _page_mod._render_bracket_page()

        mock_run_sim.assert_called_once()
        call_kwargs = mock_run_sim.call_args[1]
        assert call_kwargs["upset_aggression"] == 3
        assert call_kwargs["seed_weight_pct"] == 25

    def test_reset_button_resets_session_state(self) -> None:
        mock_st = MagicMock()
        mock_st.session_state = {
            "selected_run_id": "abc123",
            "selected_year": 2023,
            "selected_scoring": "standard",
            "bracket_upset_aggression": 3,
            "bracket_seed_weight": 50,
        }
        mock_st.columns.side_effect = lambda *a, **kw: [
            MagicMock() for _ in range(a[0] if isinstance(a[0], int) else len(a[0]))
        ]
        mock_st.selectbox.return_value = "analytical"
        mock_st.slider.side_effect = [3, 50]  # current non-neutral values
        mock_st.button.return_value = True  # simulate button click

        with (
            patch.object(_page_mod, "st", mock_st),
            patch.object(_page_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_page_mod, "load_tourney_seeds", return_value=[{"seed": "W01"}]),
        ):
            _page_mod._render_bracket_page()

        assert mock_st.session_state["bracket_upset_aggression"] == 0
        assert mock_st.session_state["bracket_seed_weight"] == 0
        mock_st.rerun.assert_called_once()


class TestOverrideInvalidation:
    """Verify AC #4: Overrides are cleared when bracket parameters change."""

    def test_override_invalidation_shows_info(self) -> None:
        mock_st = MagicMock()
        mock_st.session_state = {
            "selected_run_id": "abc123",
            "selected_year": 2023,
            "selected_scoring": "standard",
        }
        mock_st.columns.side_effect = lambda *a, **kw: [
            MagicMock() for _ in range(a[0] if isinstance(a[0], int) else len(a[0]))
        ]
        mock_st.selectbox.side_effect = [
            "analytical",
            "[1] Duke",
            "[16] Norfolk St",
            "[1] Duke",
            "[1] UConn",
            "[1] Duke",
        ]

        sim_data = _make_sim_data()

        with (
            patch.object(_page_mod, "st", mock_st),
            patch.object(_page_mod, "components", MagicMock()),
            patch.object(_page_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_page_mod, "load_tourney_seeds", return_value=[{"seed": "W01"}]),
            patch.object(_page_mod, "run_bracket_simulation", return_value=sim_data),
            patch.object(_page_mod, "render_bracket_html", return_value="<html></html>"),
            patch.object(_page_mod, "plot_advancement_heatmap", return_value=MagicMock()),
            patch.object(_page_mod, "check_invalidation", return_value=True),
            patch.object(_page_mod, "get_overrides", return_value={}),
            patch.object(_page_mod, "apply_overrides", side_effect=lambda ml, ov, br, pm: ml),
            patch.object(_page_mod, "clear_overrides"),
            patch.object(_page_mod, "set_override"),
        ):
            _page_mod._render_bracket_page()

        mock_st.info.assert_called()
        info_msgs = [call[0][0] for call in mock_st.info.call_args_list]
        assert any("overrides" in msg.lower() and "reset" in msg.lower() for msg in info_msgs)

    def test_override_count_displayed(self) -> None:
        mock_st = MagicMock()
        mock_st.session_state = {
            "selected_run_id": "abc123",
            "selected_year": 2023,
            "selected_scoring": "standard",
        }
        mock_st.columns.side_effect = lambda *a, **kw: [
            MagicMock() for _ in range(a[0] if isinstance(a[0], int) else len(a[0]))
        ]
        # edited bracket: winners=(1, 2, 1) means:
        # game 0 (0 vs 1): winner=1 "[16] Norfolk St"
        # game 1 (2 vs 3): winner=2 "[1] UConn"
        # game 2 (1 vs 2): winner=1 "[16] Norfolk St"
        mock_st.selectbox.side_effect = [
            "analytical",
            "[1] Duke",
            "[16] Norfolk St",
            "[16] Norfolk St",  # edit pick game 1: team 0 vs 1
            "[1] UConn",  # edit pick game 2: team 2 vs 3
            "[16] Norfolk St",  # edit pick game 3 (champ): team 1 vs 2
        ]

        sim_data = _make_sim_data()
        overrides = {0: 1}  # One override
        edited_ml = _FakeMostLikely(winners=(1, 2, 1), champion_team_id=200, log_likelihood=-2.0)

        with (
            patch.object(_page_mod, "st", mock_st),
            patch.object(_page_mod, "components", MagicMock()),
            patch.object(_page_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_page_mod, "load_tourney_seeds", return_value=[{"seed": "W01"}]),
            patch.object(_page_mod, "run_bracket_simulation", return_value=sim_data),
            patch.object(_page_mod, "render_bracket_html", return_value="<html></html>"),
            patch.object(_page_mod, "plot_advancement_heatmap", return_value=MagicMock()),
            patch.object(_page_mod, "check_invalidation", return_value=False),
            patch.object(_page_mod, "get_overrides", return_value=overrides),
            patch.object(_page_mod, "apply_overrides", return_value=edited_ml),
            patch.object(_page_mod, "clear_overrides"),
            patch.object(_page_mod, "set_override"),
        ):
            _page_mod._render_bracket_page()

        info_msgs = [call[0][0] for call in mock_st.info.call_args_list]
        assert any("1 of 3" in msg for msg in info_msgs)

    def test_reset_button_shown_when_overrides_exist(self) -> None:
        mock_st = MagicMock()
        mock_st.session_state = {
            "selected_run_id": "abc123",
            "selected_year": 2023,
            "selected_scoring": "standard",
        }
        mock_st.columns.side_effect = lambda *a, **kw: [
            MagicMock() for _ in range(a[0] if isinstance(a[0], int) else len(a[0]))
        ]
        mock_st.selectbox.side_effect = [
            "analytical",
            "[1] Duke",
            "[16] Norfolk St",
            "[16] Norfolk St",
            "[1] UConn",
            "[1] Duke",
        ]
        mock_st.button.return_value = False  # button not clicked

        sim_data = _make_sim_data()
        overrides = {0: 1}

        with (
            patch.object(_page_mod, "st", mock_st),
            patch.object(_page_mod, "components", MagicMock()),
            patch.object(_page_mod, "get_data_dir", return_value="/fake/data"),
            patch.object(_page_mod, "load_tourney_seeds", return_value=[{"seed": "W01"}]),
            patch.object(_page_mod, "run_bracket_simulation", return_value=sim_data),
            patch.object(_page_mod, "render_bracket_html", return_value="<html></html>"),
            patch.object(_page_mod, "plot_advancement_heatmap", return_value=MagicMock()),
            patch.object(_page_mod, "check_invalidation", return_value=False),
            patch.object(_page_mod, "get_overrides", return_value=overrides),
            patch.object(_page_mod, "apply_overrides", return_value=sim_data.most_likely),
            patch.object(_page_mod, "clear_overrides"),
            patch.object(_page_mod, "set_override"),
        ):
            _page_mod._render_bracket_page()

        # Reset button should have been rendered
        button_calls = [call for call in mock_st.button.call_args_list]
        button_labels = [call[0][0] if call[0] else call[1].get("label", "") for call in button_calls]
        assert any("Reset" in label for label in button_labels)
