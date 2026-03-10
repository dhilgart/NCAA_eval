"""Bracket override state management and downstream cascade logic.

Provides functions to store, retrieve, and apply user bracket overrides
in ``st.session_state``.  The ``apply_overrides`` function produces a new
``MostLikelyBracket`` that incorporates user picks and cascades downstream
effects through later tournament rounds.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import streamlit as st

from ncaa_eval.evaluation.simulation import MostLikelyBracket

if TYPE_CHECKING:
    import numpy.typing as npt

    from ncaa_eval.evaluation.simulation import BracketStructure

_SESSION_KEY = "bracket_overrides"
_SESSION_OVERRIDE_PARAMS_KEY = "bracket_override_params"


def get_overrides() -> dict[int, int]:
    """Return current overrides from session state (empty dict if none)."""
    return dict(st.session_state.get(_SESSION_KEY, {}))


def set_override(game_index: int, winner_index: int) -> None:
    """Add or update a single override."""
    overrides: dict[int, int] = st.session_state.get(_SESSION_KEY, {})
    overrides[game_index] = winner_index
    st.session_state[_SESSION_KEY] = overrides


def clear_overrides() -> None:
    """Remove all overrides from session state."""
    st.session_state[_SESSION_KEY] = {}


def check_invalidation(
    run_id: str,
    year: int,
    scoring: str,
    upset_aggression: int,
    seed_weight_pct: int,
) -> bool:
    """Check if overrides should be invalidated due to parameter change.

    Compares current parameters against stored invalidation key.  If they
    differ and overrides exist, clears overrides and returns ``True``.
    Always stores the current parameters for future comparison.

    Returns:
        ``True`` if overrides were cleared, ``False`` otherwise.
    """
    current_params = (run_id, year, scoring, upset_aggression, seed_weight_pct)
    stored_params = st.session_state.get(_SESSION_OVERRIDE_PARAMS_KEY)
    overrides = st.session_state.get(_SESSION_KEY, {})

    invalidated = False
    if stored_params is not None and stored_params != current_params and overrides:
        clear_overrides()
        invalidated = True

    st.session_state[_SESSION_OVERRIDE_PARAMS_KEY] = current_params
    return invalidated


def _get_participants(
    game_index: int,
    winners: Sequence[int],
    n_teams: int,
) -> tuple[int, int]:
    """Determine the two participants in a game.

    For Round of 64 (first ``n_teams // 2`` games), participants are
    determined by bracket position.  For later rounds, participants are
    the winners of the two feeder games.
    """
    n_r64 = n_teams // 2  # 32 for 64-team bracket
    if game_index < n_r64:
        return game_index * 2, game_index * 2 + 1

    # Determine round structure
    offset = 0
    games_in_round = n_r64
    while offset + games_in_round <= game_index:
        offset += games_in_round
        games_in_round //= 2

    game_in_round = game_index - offset
    prev_round_games = games_in_round * 2
    prev_offset = offset - prev_round_games
    feeder_a = prev_offset + game_in_round * 2
    feeder_b = prev_offset + game_in_round * 2 + 1
    return winners[feeder_a], winners[feeder_b]


def apply_overrides(
    most_likely: MostLikelyBracket,
    overrides: dict[int, int],
    bracket: BracketStructure,
    prob_matrix: npt.NDArray[np.float64],
) -> MostLikelyBracket:
    """Produce a new ``MostLikelyBracket`` with user overrides applied.

    Starts from the model's ``most_likely.winners``, applies overrides in
    round order, and cascades downstream effects.  When an upstream
    override changes a participant in a downstream game, the downstream
    game is re-resolved using either its own override (if valid) or the
    model's argmax from ``prob_matrix``.

    Args:
        most_likely: The model's greedy bracket picks.
        overrides: Mapping of ``game_index → winner_team_index``.
        bracket: Bracket structure (for team_ids lookup).
        prob_matrix: Pairwise win probability matrix, shape ``(N, N)``.

    Returns:
        New ``MostLikelyBracket`` with overrides applied.
    """
    if not overrides:
        # Recompute log-likelihood for consistency
        winners = list(most_likely.winners)
        ll = _compute_log_likelihood(winners, len(bracket.team_ids), prob_matrix)
        return MostLikelyBracket(
            winners=most_likely.winners,
            champion_team_id=most_likely.champion_team_id,
            log_likelihood=ll,
        )

    n_games = len(most_likely.winners)
    n_teams = n_games + 1
    winners = list(most_likely.winners)

    # Process games in round-major order (ascending index).
    # For each game, determine participants, then decide winner.
    # Track which team indices changed from most_likely to drive cascade correctly:
    # only re-resolve a downstream game when its participants actually changed.
    n_r64 = n_teams // 2
    offset = 0
    games_in_round = n_r64

    for _ in range(int(np.log2(n_teams))):
        for g in range(games_in_round):
            game_idx = offset + g
            participant_a, participant_b = _get_participants(game_idx, winners, n_teams)

            if game_idx in overrides:
                override_winner = overrides[game_idx]
                if override_winner in (participant_a, participant_b):
                    winners[game_idx] = override_winner
                else:
                    # Stale override — fall back to model
                    winners[game_idx] = _pick_model_winner(participant_a, participant_b, prob_matrix)
            elif game_idx >= n_r64:
                # No override: only re-resolve if participants changed from most_likely.
                # Compute the participants the model originally faced for this game.
                orig_a, orig_b = _get_participants(game_idx, list(most_likely.winners), n_teams)
                if participant_a != orig_a or participant_b != orig_b:
                    # Upstream override altered participants — re-resolve via model argmax.
                    winners[game_idx] = _pick_model_winner(participant_a, participant_b, prob_matrix)
                # else: participants unchanged → keep most_likely.winners[game_idx]
            # R64 games without overrides keep their original winner

        offset += games_in_round
        games_in_round //= 2

    champion_team_id = bracket.team_ids[winners[-1]]
    ll = _compute_log_likelihood(winners, n_teams, prob_matrix)

    return MostLikelyBracket(
        winners=tuple(winners),
        champion_team_id=champion_team_id,
        log_likelihood=ll,
    )


def _pick_model_winner(a: int, b: int, prob_matrix: npt.NDArray[np.float64]) -> int:
    """Pick the winner of a matchup using argmax of probability matrix."""
    if prob_matrix[a, b] >= prob_matrix[b, a]:
        return a
    return b


def _compute_log_likelihood(
    winners: list[int],
    n_teams: int,
    prob_matrix: npt.NDArray[np.float64],
) -> float:
    """Compute log-likelihood of a bracket as sum of log win probabilities."""
    ll = 0.0
    n_games = n_teams - 1

    for game_idx in range(n_games):
        participant_a, participant_b = _get_participants(game_idx, winners, n_teams)
        winner = winners[game_idx]
        opponent = participant_b if winner == participant_a else participant_a
        prob = float(prob_matrix[winner, opponent])
        if prob > 0:
            ll += float(np.log(prob))
        else:
            ll += float(-np.inf)

    return ll
