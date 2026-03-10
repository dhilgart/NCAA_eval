"""Tests for bracket override state management and cascade logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from dashboard.lib.bracket_overrides import (
    _SESSION_KEY,
    _SESSION_OVERRIDE_PARAMS_KEY,
    apply_overrides,
    check_invalidation,
    clear_overrides,
    get_overrides,
    set_override,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prob_matrix_4() -> np.ndarray:
    """4-team probability matrix where team 0 beats everyone."""
    P = np.array(
        [
            [0.0, 0.7, 0.8, 0.9],
            [0.3, 0.0, 0.6, 0.7],
            [0.2, 0.4, 0.0, 0.6],
            [0.1, 0.3, 0.4, 0.0],
        ]
    )
    return P


def _make_most_likely_4() -> MagicMock:
    """4-team bracket: game0(0v1)=0, game1(2v3)=2, game2(0v2)=0."""
    ml = MagicMock()
    ml.winners = (0, 2, 0)
    ml.champion_team_id = 100
    ml.log_likelihood = -1.5
    return ml


def _make_bracket_4() -> MagicMock:
    """4-team bracket structure."""
    b = MagicMock()
    b.team_ids = (100, 200, 300, 400)
    b.team_index_map = {100: 0, 200: 1, 300: 2, 400: 3}
    b.seed_map = {100: 1, 200: 4, 300: 2, 400: 3}
    return b


def _make_prob_matrix_8() -> np.ndarray:
    """8-team probability matrix."""
    P = np.full((8, 8), 0.5)
    np.fill_diagonal(P, 0.0)
    # Make team 0 dominant
    for i in range(1, 8):
        P[0, i] = 0.8
        P[i, 0] = 0.2
    # Ensure P[i, j] + P[j, i] == 1 for others too
    for i in range(8):
        for j in range(i + 1, 8):
            if i == 0:
                continue
            P[i, j] = 0.5
            P[j, i] = 0.5
    return P


def _make_most_likely_8() -> MagicMock:
    """8-team bracket most likely: all lower-indexed teams win."""
    # R64: g0(0v1)=0, g1(2v3)=2, g2(4v5)=4, g3(6v7)=6
    # R32: g4(0v2)=0, g5(4v6)=4
    # Championship: g6(0v4)=0
    ml = MagicMock()
    ml.winners = (0, 2, 4, 6, 0, 4, 0)
    ml.champion_team_id = 1000
    ml.log_likelihood = -3.0
    return ml


# ---------------------------------------------------------------------------
# Tests: get_overrides
# ---------------------------------------------------------------------------


class TestGetOverrides:
    def test_returns_empty_dict_when_no_overrides(self) -> None:
        mock_st = MagicMock()
        mock_st.session_state = {}
        with patch("dashboard.lib.bracket_overrides.st", mock_st):
            result = get_overrides()
        assert result == {}

    def test_returns_stored_overrides(self) -> None:
        mock_st = MagicMock()
        mock_st.session_state = {_SESSION_KEY: {5: 11, 35: 3}}
        with patch("dashboard.lib.bracket_overrides.st", mock_st):
            result = get_overrides()
        assert result == {5: 11, 35: 3}


# ---------------------------------------------------------------------------
# Tests: set_override
# ---------------------------------------------------------------------------


class TestSetOverride:
    def test_stores_override_in_session_state(self) -> None:
        mock_st = MagicMock()
        mock_st.session_state = {}
        with patch("dashboard.lib.bracket_overrides.st", mock_st):
            set_override(5, 11)
        assert mock_st.session_state[_SESSION_KEY] == {5: 11}

    def test_adds_to_existing_overrides(self) -> None:
        mock_st = MagicMock()
        mock_st.session_state = {_SESSION_KEY: {5: 11}}
        with patch("dashboard.lib.bracket_overrides.st", mock_st):
            set_override(10, 20)
        assert mock_st.session_state[_SESSION_KEY] == {5: 11, 10: 20}

    def test_overwrites_existing_override(self) -> None:
        mock_st = MagicMock()
        mock_st.session_state = {_SESSION_KEY: {5: 11}}
        with patch("dashboard.lib.bracket_overrides.st", mock_st):
            set_override(5, 10)
        assert mock_st.session_state[_SESSION_KEY] == {5: 10}


# ---------------------------------------------------------------------------
# Tests: clear_overrides
# ---------------------------------------------------------------------------


class TestClearOverrides:
    def test_removes_all_overrides(self) -> None:
        mock_st = MagicMock()
        mock_st.session_state = {_SESSION_KEY: {5: 11, 10: 20}}
        with patch("dashboard.lib.bracket_overrides.st", mock_st):
            clear_overrides()
        assert mock_st.session_state[_SESSION_KEY] == {}

    def test_noop_when_no_overrides(self) -> None:
        mock_st = MagicMock()
        mock_st.session_state = {}
        with patch("dashboard.lib.bracket_overrides.st", mock_st):
            clear_overrides()
        assert mock_st.session_state.get(_SESSION_KEY) == {}


# ---------------------------------------------------------------------------
# Tests: apply_overrides — identity
# ---------------------------------------------------------------------------


class TestApplyOverridesIdentity:
    def test_no_overrides_returns_original(self) -> None:
        ml = _make_most_likely_4()
        P = _make_prob_matrix_4()
        bracket = _make_bracket_4()

        result = apply_overrides(ml, {}, bracket, P)

        assert result.winners == ml.winners
        assert result.champion_team_id == ml.champion_team_id


# ---------------------------------------------------------------------------
# Tests: apply_overrides — single override
# ---------------------------------------------------------------------------


class TestApplyOverridesSingle:
    def test_single_r64_override(self) -> None:
        """Override game 0: pick team 1 instead of team 0."""
        ml = _make_most_likely_4()
        P = _make_prob_matrix_4()
        bracket = _make_bracket_4()

        result = apply_overrides(ml, {0: 1}, bracket, P)

        # Game 0 winner should be 1
        assert result.winners[0] == 1
        # Game 2 (championship) had 0 vs 2, now should be 1 vs 2
        # Model picks: P[1,2]=0.6 > P[2,1]=0.4, so model picks 1
        assert result.winners[2] == 1
        # Champion changed
        assert result.champion_team_id == bracket.team_ids[1]


# ---------------------------------------------------------------------------
# Tests: apply_overrides — cascade
# ---------------------------------------------------------------------------


class TestApplyOverridesCascade:
    def test_cascade_through_rounds(self) -> None:
        """Override in R64 cascades to championship via model picks."""
        ml = _make_most_likely_8()
        P = _make_prob_matrix_8()

        bracket = MagicMock()
        bracket.team_ids = tuple(range(1000, 1008))

        # Override game 0 (R64): pick team 1 instead of 0
        result = apply_overrides(ml, {0: 1}, bracket, P)

        # Game 0 now picks team 1
        assert result.winners[0] == 1
        # Game 4 (R32) was 0 vs 2, now 1 vs 2. P[1,2]=0.5, P[2,1]=0.5.
        # argmax picks lower index → 1 (or either, depends on tie-breaking)
        assert result.winners[4] in (1, 2)
        # Championship game should also be re-resolved

    def test_override_championship_directly(self) -> None:
        """Override the championship game directly."""
        ml = _make_most_likely_4()
        P = _make_prob_matrix_4()
        bracket = _make_bracket_4()

        # Override championship (game 2): pick team 2 instead of 0
        result = apply_overrides(ml, {2: 2}, bracket, P)

        assert result.winners[2] == 2
        assert result.champion_team_id == bracket.team_ids[2]

    def test_stale_downstream_override_falls_back_to_model(self) -> None:
        """When upstream override invalidates a downstream override's winner."""
        ml = _make_most_likely_4()
        P = _make_prob_matrix_4()
        bracket = _make_bracket_4()

        # Override game 0: pick 1 (instead of 0)
        # Override game 2 (championship): pick 0
        # But game 0 changes R64 winner from 0 to 1, so participant is now 1, not 0
        # Championship participants become 1 and 2 — override pick 0 is invalid
        result = apply_overrides(ml, {0: 1, 2: 0}, bracket, P)

        # Game 0: team 1 wins
        assert result.winners[0] == 1
        # Championship: override tried team 0, but participants are 1 and 2
        # Falls back to model: P[1,2]=0.6 > P[2,1]=0.4 → picks team 1
        assert result.winners[2] == 1


# ---------------------------------------------------------------------------
# Tests: apply_overrides — log_likelihood
# ---------------------------------------------------------------------------


class TestApplyOverridesLogLikelihood:
    def test_log_likelihood_recomputed(self) -> None:
        ml = _make_most_likely_4()
        P = _make_prob_matrix_4()
        bracket = _make_bracket_4()

        result = apply_overrides(ml, {}, bracket, P)
        # With no overrides, log-likelihood should match original winners
        expected = float(np.log(P[0, 1]) + np.log(P[2, 3]) + np.log(P[0, 2]))
        assert abs(result.log_likelihood - expected) < 1e-10

    def test_override_changes_log_likelihood(self) -> None:
        ml = _make_most_likely_4()
        P = _make_prob_matrix_4()
        bracket = _make_bracket_4()

        result_no_override = apply_overrides(ml, {}, bracket, P)
        result_with_override = apply_overrides(ml, {0: 1}, bracket, P)

        assert result_no_override.log_likelihood != result_with_override.log_likelihood


# ---------------------------------------------------------------------------
# Tests: check_invalidation
# ---------------------------------------------------------------------------


class TestCheckInvalidation:
    def test_returns_false_when_params_match(self) -> None:
        mock_st = MagicMock()
        params = ("run1", 2023, "standard", 0, 0)
        mock_st.session_state = {
            _SESSION_KEY: {5: 11},
            _SESSION_OVERRIDE_PARAMS_KEY: params,
        }
        with patch("dashboard.lib.bracket_overrides.st", mock_st):
            result = check_invalidation(*params)
        assert result is False

    def test_returns_true_and_clears_when_params_differ(self) -> None:
        mock_st = MagicMock()
        mock_st.session_state = {
            _SESSION_KEY: {5: 11},
            _SESSION_OVERRIDE_PARAMS_KEY: ("run1", 2023, "standard", 0, 0),
        }
        new_params = ("run2", 2023, "standard", 0, 0)
        with patch("dashboard.lib.bracket_overrides.st", mock_st):
            result = check_invalidation(*new_params)
        assert result is True
        assert mock_st.session_state[_SESSION_KEY] == {}

    def test_returns_false_when_no_overrides_exist(self) -> None:
        mock_st = MagicMock()
        mock_st.session_state = {}
        with patch("dashboard.lib.bracket_overrides.st", mock_st):
            result = check_invalidation("run1", 2023, "standard", 0, 0)
        assert result is False

    def test_stores_new_params(self) -> None:
        mock_st = MagicMock()
        mock_st.session_state = {}
        params = ("run1", 2023, "standard", 3, 25)
        with patch("dashboard.lib.bracket_overrides.st", mock_st):
            check_invalidation(*params)
        assert mock_st.session_state[_SESSION_OVERRIDE_PARAMS_KEY] == params
