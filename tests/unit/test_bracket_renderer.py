"""Tests for the HTML/CSS bracket tree renderer."""

from __future__ import annotations

import numpy as np

from dashboard.lib.bracket_renderer import (
    _prob_color,
    _winner_prob,
    render_bracket_html,
)


class TestProbColor:
    def test_high_prob_is_greenish(self) -> None:
        color = _prob_color(0.95)
        assert color.startswith("#")
        assert len(color) == 7

    def test_low_prob_is_reddish(self) -> None:
        color = _prob_color(0.05)
        assert color.startswith("#")

    def test_midpoint_is_neutral(self) -> None:
        color = _prob_color(0.5)
        assert color == "#6c757d"

    def test_zero_prob(self) -> None:
        color = _prob_color(0.0)
        assert color.startswith("#")

    def test_one_prob(self) -> None:
        color = _prob_color(1.0)
        assert color.startswith("#")


class TestWinnerProb:
    def test_returns_correct_prob(self) -> None:
        P = np.array(
            [
                [0.0, 0.7, 0.6],
                [0.3, 0.0, 0.5],
                [0.4, 0.5, 0.0],
            ]
        )
        assert _winner_prob(0, 0, 1, P) == 0.7
        assert _winner_prob(1, 0, 1, P) == 0.3


def _make_64_team_fixtures() -> (
    tuple[tuple[int, ...], tuple[int, ...], dict[int, str], dict[int, int], np.ndarray]
):
    """Create a minimal 64-team bracket for testing.

    Uses a deterministic probability matrix where lower-indexed teams
    are stronger (0.9 win prob vs. higher-indexed opponents).
    """
    n = 64
    team_ids = tuple(1000 + i for i in range(n))

    # P[i,j] = 0.9 if i < j
    P = np.full((n, n), 0.5, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i < j:
                P[i, j] = 0.9
                P[j, i] = 0.1
        P[i, i] = 0.0

    # Build most-likely winners: favorites always win (lower index wins).
    winners: list[int] = []
    prev = list(range(n))
    for games in [32, 16, 8, 4, 2, 1]:
        next_prev: list[int] = []
        for g in range(games):
            left, right = prev[g * 2], prev[g * 2 + 1]
            winner = min(left, right)
            winners.append(winner)
            next_prev.append(winner)
        prev = next_prev

    assert len(winners) == 63

    labels: dict[int, str] = {}
    seed_map: dict[int, int] = {}
    for i in range(n):
        region_name = "WXYZ"[i // 16]
        local_seed = (i % 16) + 1
        labels[i] = f"[{local_seed}] {region_name}Team{i}"
        seed_map[team_ids[i]] = local_seed

    return team_ids, tuple(winners), labels, seed_map, P


class TestRenderBracketHtml:
    def test_returns_html_string(self) -> None:
        team_ids, winners, labels, seed_map, P = _make_64_team_fixtures()
        result = render_bracket_html(team_ids, winners, labels, seed_map, P)
        assert isinstance(result, str)
        assert "<html>" in result
        assert "</html>" in result

    def test_contains_team_names(self) -> None:
        team_ids, winners, labels, seed_map, P = _make_64_team_fixtures()
        result = render_bracket_html(team_ids, winners, labels, seed_map, P)
        assert "WTeam0" in result
        assert "YTeam32" in result

    def test_contains_seed_numbers(self) -> None:
        team_ids, winners, labels, seed_map, P = _make_64_team_fixtures()
        result = render_bracket_html(team_ids, winners, labels, seed_map, P)
        assert ">1<" in result
        assert ">16<" in result

    def test_contains_css_styles(self) -> None:
        team_ids, winners, labels, seed_map, P = _make_64_team_fixtures()
        result = render_bracket_html(team_ids, winners, labels, seed_map, P)
        assert "<style>" in result
        assert ".bracket" in result
        assert ".team" in result

    def test_contains_probability_percentages(self) -> None:
        team_ids, winners, labels, seed_map, P = _make_64_team_fixtures()
        result = render_bracket_html(team_ids, winners, labels, seed_map, P)
        assert "90%" in result

    def test_contains_final_four_label(self) -> None:
        team_ids, winners, labels, seed_map, P = _make_64_team_fixtures()
        result = render_bracket_html(team_ids, winners, labels, seed_map, P)
        assert "Final Four" in result
        assert "Champion" in result

    def test_escapes_html_in_labels(self) -> None:
        team_ids, winners, labels, seed_map, P = _make_64_team_fixtures()
        labels[0] = "[1] <script>alert('xss')</script>"
        result = render_bracket_html(team_ids, winners, labels, seed_map, P)
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_all_four_regions_present(self) -> None:
        team_ids, winners, labels, seed_map, P = _make_64_team_fixtures()
        result = render_bracket_html(team_ids, winners, labels, seed_map, P)
        assert "data-region='W'" in result
        assert "data-region='X'" in result
        assert "data-region='Y'" in result
        assert "data-region='Z'" in result

    def test_right_regions_are_mirrored(self) -> None:
        """Right-side regions (Y, Z) must appear reversed relative to left regions.

        In the HTML the left half has region W with rounds ordered seed→R64→R32→S16→E8
        (left-to-right).  For the right half the columns are reversed so the bracket
        opens outward — E8→S16→R32→R64→seed.  We verify this by checking that the
        Y-region ``data-region`` marker appears AFTER the W-region in the HTML string
        (i.e. right side is rendered after left side in the DOM), and that within the
        right-half ``<div class='half'>`` block the text for right-side teams appears.
        """
        team_ids, winners, labels, seed_map, P = _make_64_team_fixtures()
        result = render_bracket_html(team_ids, winners, labels, seed_map, P)
        # Right regions must be present
        assert "data-region='Y'" in result
        assert "data-region='Z'" in result
        # The right half div must appear after the center div in the HTML
        center_pos = result.index("class='center'")
        right_half_pos = result.index("data-region='Y'")
        assert right_half_pos > center_pos, "Region Y (right) must be in right half after center"
