"""HTML/CSS bracket tree renderer for 64-team NCAA tournament brackets.

Pure function ``render_bracket_html`` returns a self-contained HTML string
with embedded CSS that renders a bracket layout using CSS flexbox.  No
Streamlit imports — the output is intended for ``st.components.html()``.

Layout:
    Left side:  Region W (top) + Region X (bottom)  → rounds R64..E8 (left-to-right)
    Center:     Final Four + Championship
    Right side: Region Y (top) + Region Z (bottom)  → rounds R64..E8 (right-to-left, mirrored)
"""

from __future__ import annotations

import html
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

# Number of teams per region and region bracket-index offsets.
_REGION_STARTS: tuple[int, ...] = (0, 16, 32, 48)

_BRACKET_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    background: #0E1117;
    color: #FAFAFA;
    font-family: 'Source Code Pro', 'Courier New', monospace;
    font-size: 11px;
}
.bracket {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    width: 100%;
    min-height: 600px;
}
.half {
    display: flex;
    flex-direction: column;
    gap: 8px;
    flex: 1;
}
.region {
    display: flex;
    align-items: center;
    gap: 2px;
}
.round {
    display: flex;
    flex-direction: column;
    justify-content: space-around;
    gap: 2px;
    min-height: 100%;
    flex: 1;
}
.team {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 2px 6px;
    border-radius: 3px;
    white-space: nowrap;
    min-height: 18px;
    border: 1px solid #2a2e3d;
}
.seed {
    color: #888;
    font-size: 10px;
    min-width: 14px;
    text-align: right;
}
.name {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    font-size: 10px;
}
.prob {
    color: #ccc;
    font-size: 9px;
    min-width: 28px;
    text-align: right;
}
.center {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 6px;
    padding: 0 12px;
    min-width: 140px;
}
.center .team { width: 130px; }
.f4-label, .champ-label {
    color: #888;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 8px;
}
.champ-label { margin-top: 16px; }
"""


def _prob_color(prob: float) -> str:
    """Return a CSS color interpolating green→neutral→red by win probability."""
    if prob >= 0.5:
        t = (prob - 0.5) * 2.0
        r = int(108 + (40 - 108) * t)
        g = int(117 + (167 - 117) * t)
        b = int(125 + (69 - 125) * t)
    else:
        t = prob * 2.0
        r = int(220 + (108 - 220) * t)
        g = int(53 + (117 - 53) * t)
        b = int(69 + (125 - 69) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _esc(text: str) -> str:
    """HTML-escape a string."""
    return html.escape(text, quote=True)


def _winner_prob(
    winner: int,
    left: int,
    right: int,
    prob_matrix: npt.NDArray[np.float64],
) -> float:
    """Return the win probability for *winner* beating the opponent."""
    opponent = right if winner == left else left
    return float(prob_matrix[winner, opponent])


def _resolve_round_winners(
    prev_teams: list[int],
    round_winners: list[int],
    prob_matrix: npt.NDArray[np.float64],
) -> list[tuple[int, float]]:
    """Pair previous-round survivors, look up winner probabilities.

    Args:
        prev_teams: Team indices from the previous round (or seeds).
        round_winners: Winner team indices for this round's games.
        prob_matrix: Pairwise probability matrix.

    Returns:
        List of ``(winner_index, win_prob)`` tuples.
    """
    result: list[tuple[int, float]] = []
    for g_idx, winner in enumerate(round_winners):
        left = prev_teams[g_idx * 2]
        right = prev_teams[g_idx * 2 + 1]
        result.append((winner, _winner_prob(winner, left, right, prob_matrix)))
    return result


def _build_region_rounds(
    region_idx: int,
    rounds: list[list[int]],
    prob_matrix: npt.NDArray[np.float64],
) -> list[list[tuple[int, float]]]:
    """Build rounds for one region: seeds + R64 winners through E8 winner.

    Returns:
        List of 5 round entries.  Index 0 is the seed row (16 teams,
        probability=1.0); indices 1-4 are R64, R32, S16, E8 winners.
    """
    start = _REGION_STARTS[region_idx]
    seed_teams = list(range(start, start + 16))
    region_rounds: list[list[tuple[int, float]]] = [[(t, 1.0) for t in seed_teams]]

    # Track survivors through 4 intra-region rounds
    prev: list[int] = seed_teams
    games_per_round = [8, 4, 2, 1]
    round_offsets = [region_idx * g for g in games_per_round]
    for r_idx in range(4):
        g = games_per_round[r_idx]
        off = round_offsets[r_idx]
        winners_list = rounds[r_idx][off : off + g]
        rnd = _resolve_round_winners(prev, winners_list, prob_matrix)
        region_rounds.append(rnd)
        prev = winners_list

    return region_rounds


def _team_cell(  # noqa: PLR0913
    team_idx: int,
    prob: float,
    bracket_team_ids: tuple[int, ...],
    team_labels: Mapping[int, str],
    seed_map: Mapping[int, int],
    *,
    is_seed: bool = False,
) -> str:
    """Render one team cell as HTML."""
    label = _esc(team_labels.get(team_idx, str(team_idx)))
    tid = bracket_team_ids[team_idx]
    seed = seed_map.get(tid, 0)
    bg = _prob_color(prob) if not is_seed else "#1e2130"
    prob_str = "" if is_seed else f"<span class='prob'>{prob:.0%}</span>"
    return (
        f"<div class='team' style='background:{bg};'>"
        f"<span class='seed'>{seed}</span>"
        f"<span class='name'>{label}</span>"
        f"{prob_str}"
        f"</div>"
    )


def _render_region_html(  # noqa: PLR0913
    region_rounds: list[list[tuple[int, float]]],
    region_name: str,
    bracket_team_ids: tuple[int, ...],
    team_labels: Mapping[int, str],
    seed_map: Mapping[int, int],
    *,
    mirror: bool = False,
) -> str:
    """Render a region bracket (5 columns: seeds + 4 rounds)."""
    cols: list[str] = []
    for i, rd in enumerate(region_rounds):
        cells = "".join(
            _team_cell(t, p, bracket_team_ids, team_labels, seed_map, is_seed=(i == 0)) for t, p in rd
        )
        cols.append(f"<div class='round'>{cells}</div>")

    if mirror:
        cols = list(reversed(cols))

    inner = "".join(cols)
    return f"<div class='region' data-region='{region_name}'>{inner}</div>"


def render_bracket_html(
    bracket_team_ids: tuple[int, ...],
    most_likely_winners: tuple[int, ...],
    team_labels: Mapping[int, str],
    seed_map: Mapping[int, int],
    prob_matrix: npt.NDArray[np.float64],
) -> str:
    """Render a 64-team bracket as a self-contained HTML/CSS string.

    The bracket shows the most-likely path through all 6 rounds with
    win probabilities color-coded green (favorites) to red (upsets).

    Args:
        bracket_team_ids: 64 team IDs in bracket-position order (leaf order).
        most_likely_winners: 63 team **indices** (into bracket_team_ids) in
            round-major order: games 0-31 = R64, 32-47 = R32, 48-55 = S16,
            56-59 = E8, 60-61 = F4, 62 = Championship.
        team_labels: Mapping of bracket index → display label (e.g. "[1] Duke").
        seed_map: Mapping of team_id → seed number.
        prob_matrix: Pairwise win probability matrix, shape (64, 64).

    Returns:
        Self-contained HTML string with embedded CSS.
    """
    n_teams = len(bracket_team_ids)
    # Parse winners into per-round lists
    rounds: list[list[int]] = []
    offset = 0
    games_in_round = n_teams // 2
    for _ in range(6):
        rounds.append(list(most_likely_winners[offset : offset + games_in_round]))
        offset += games_in_round
        games_in_round //= 2

    # Build per-region data
    regions_data = [_build_region_rounds(i, rounds, prob_matrix) for i in range(4)]

    # Final Four and Championship
    f4_winners = rounds[4]
    champ_winner = rounds[5][0]

    e8_winners = [regions_data[i][4][0][0] for i in range(4)]
    f4_prob_0 = _winner_prob(f4_winners[0], e8_winners[0], e8_winners[1], prob_matrix)
    f4_prob_1 = _winner_prob(f4_winners[1], e8_winners[2], e8_winners[3], prob_matrix)
    champ_prob = _winner_prob(champ_winner, f4_winners[0], f4_winners[1], prob_matrix)

    # Render regions
    cell_args = (bracket_team_ids, team_labels, seed_map)
    region_labels = ("W", "X", "Y", "Z")
    left_top = _render_region_html(regions_data[0], region_labels[0], *cell_args)
    left_bot = _render_region_html(regions_data[1], region_labels[1], *cell_args)
    right_top = _render_region_html(regions_data[2], region_labels[2], *cell_args, mirror=True)
    right_bot = _render_region_html(regions_data[3], region_labels[3], *cell_args, mirror=True)

    # Center column
    center_html = (
        "<div class='center'>"
        "<div class='f4-label'>Final Four</div>"
        f"{_team_cell(f4_winners[0], f4_prob_0, *cell_args)}"
        f"{_team_cell(f4_winners[1], f4_prob_1, *cell_args)}"
        "<div class='champ-label'>Champion</div>"
        f"{_team_cell(champ_winner, champ_prob, *cell_args)}"
        "</div>"
    )

    return (
        f"<html><head><style>{_BRACKET_CSS}</style></head><body>"
        f"<div class='bracket'>"
        f"<div class='half'>{left_top}{left_bot}</div>"
        f"{center_html}"
        f"<div class='half'>{right_top}{right_bot}</div>"
        f"</div>"
        f"</body></html>"
    )
