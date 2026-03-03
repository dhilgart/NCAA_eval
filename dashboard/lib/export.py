"""Bracket CSV export helpers for the dashboard."""

from __future__ import annotations

import io

import numpy as np
import numpy.typing as npt

from ncaa_eval.evaluation.simulation import BracketStructure, MostLikelyBracket

_ROUND_LABELS: tuple[str, ...] = ("R64", "R32", "S16", "E8", "F4", "Championship")


def export_bracket_csv(
    bracket: BracketStructure,
    most_likely: MostLikelyBracket,
    team_labels: dict[int, str],
    prob_matrix: npt.NDArray[np.float64],
) -> str:
    """Build a CSV string of the most-likely bracket picks for download.

    Derives the number of rounds from ``log2(n_games + 1)`` (63 games →
    6 rounds).  Iterates rounds outer-to-inner: each round halves
    ``games_in_round`` and advances ``game_offset`` by the previous
    round's game count.  Within each round, looks up the winner index
    from ``most_likely.winners``, resolves team ID and seed from
    ``bracket``, strips the ``"[seed] "`` prefix from the label, then
    delegates the per-game win probability to ``_game_win_probability``.

    Returns one row per game (63 rows) in round-major order with columns:
    ``game_number``, ``round``, ``team_id``, ``team_name``, ``seed``,
    ``win_probability``.

    Args:
        bracket: Bracket structure with team ordering and seed map.
        most_likely: Greedy most-likely bracket picks.
        team_labels: Bracket index → display label mapping.
        prob_matrix: Pairwise win probability matrix.

    Returns:
        CSV string suitable for ``st.download_button(data=…)``.
    """
    buf = io.StringIO()
    buf.write("game_number,round,team_id,team_name,seed,win_probability\n")

    n_games = len(most_likely.winners)
    # A single-elimination bracket with 2^n teams has 2^n - 1 games and n rounds.
    # Inverting: n_rounds = log2(n_games + 1).  For 63 games → 6 rounds.
    n_rounds = int(np.log2(n_games + 1))
    game_offset = 0
    games_in_round = n_games + 1  # 64 teams → 32 games first round

    for r in range(n_rounds):
        games_in_round = games_in_round // 2
        round_label = _ROUND_LABELS[r] if r < len(_ROUND_LABELS) else f"R{r}"
        for g in range(games_in_round):
            game_idx = game_offset + g
            winner_idx = most_likely.winners[game_idx]
            team_id = bracket.team_ids[winner_idx]
            seed = bracket.seed_map.get(team_id, 0)
            label = team_labels.get(winner_idx, str(team_id))
            # Strip seed prefix from label if present (e.g., "[1] Duke" → "Duke")
            name = label.split("] ", 1)[1] if "] " in label else label

            # Win probability: pairwise head-to-head probability between the
            # two most-likely participants in this game (R64 uses direct
            # pairwise probs; later rounds trace back to actual advancing teams).
            win_prob = _game_win_probability(
                r,
                g,
                game_offset,
                most_likely.winners,
                prob_matrix,
                bracket,
            )

            buf.write(f"{game_idx + 1},{round_label},{team_id},{name},{seed},{win_prob:.3f}\n")
        game_offset += games_in_round

    return buf.getvalue()


def _game_win_probability(  # noqa: PLR0913
    round_idx: int,
    game_in_round: int,
    game_offset: int,
    winners: tuple[int, ...],
    prob_matrix: npt.NDArray[np.float64],
    bracket: BracketStructure,
) -> float:
    """Compute the win probability for a specific game in the bracket.

    For Round of 64, this is the direct pairwise probability.
    For later rounds, it uses the pairwise probability between the two
    teams that advanced from the previous round.
    """
    if round_idx == 0:
        # R64: teams are seeded in bracket order, game g pits team 2g vs 2g+1
        team_a_idx = game_in_round * 2
        team_b_idx = game_in_round * 2 + 1
    else:
        # Later rounds: the two participants are the winners of the two
        # feeder games from the previous round (games 2g and 2g+1 in round r-1)
        prev_games_in_round = (len(winners) + 1) // (2**round_idx)
        prev_offset = game_offset - prev_games_in_round
        feeder_a = prev_offset + game_in_round * 2
        feeder_b = prev_offset + game_in_round * 2 + 1
        team_a_idx = winners[feeder_a]
        team_b_idx = winners[feeder_b]

    winner_idx = winners[game_offset + game_in_round]
    if winner_idx == team_a_idx:
        return float(prob_matrix[team_a_idx, team_b_idx])
    return float(prob_matrix[team_b_idx, team_a_idx])
