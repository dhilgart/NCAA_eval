"""Bracket data structures and construction for NCAA tournament simulation.

Provides the immutable bracket tree used by the simulation engine:

* :class:`MatchupContext` — context for hypothetical matchup queries.
* :class:`BracketNode` — node in a tournament bracket tree.
* :class:`BracketStructure` — immutable tournament bracket.
* :func:`build_bracket` — constructs a 64-team tree from :class:`TourneySeed`.
* :func:`_build_subtree` — recursive balanced binary tree builder.

Constants:

* :data:`N_ROUNDS` — number of rounds in a 64-team bracket (6).
* :data:`N_GAMES` — total games in a 64-team bracket (63).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ncaa_eval.transform.normalization import TourneySeed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: NCAA bracket matchup order per region (seed pairings).
#: Position in this list determines bracket-tree leaf order.
_REGION_SEED_ORDER: tuple[tuple[int, int], ...] = (
    (1, 16),
    (8, 9),
    (5, 12),
    (4, 13),
    (6, 11),
    (3, 14),
    (7, 10),
    (2, 15),
)

#: Region codes in bracket-position order.  W vs X in one semi, Y vs Z
#: in the other, winners play in the championship.
_REGION_ORDER: tuple[str, ...] = ("W", "X", "Y", "Z")

#: Number of rounds in a 64-team single-elimination bracket.
N_ROUNDS: int = 6

#: Total number of games in a 64-team bracket (63).
N_GAMES: int = 63

# ---------------------------------------------------------------------------
# Bracket data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchupContext:
    """Context for a hypothetical matchup probability query.

    Passed to :class:`ProbabilityProvider` so that stateless models can
    construct the correct feature row for a hypothetical pairing.  Stateful
    models (Elo) typically ignore context and use internal ratings.

    Attributes:
        season: Tournament season year (e.g. 2024).
        day_num: Tournament day number (e.g. 136 for Round of 64).
        is_neutral: ``True`` for all tournament games (neutral site).
    """

    season: int
    day_num: int
    is_neutral: bool


@dataclass(frozen=True)
class BracketNode:
    """Node in a tournament bracket tree.

    A leaf node represents a single team; an internal node represents a
    game whose winner advances.

    Attributes:
        round_index: Round number (0-indexed).  Leaves have ``round_index=-1``.
        team_index: Index into the bracket's ``team_ids`` tuple for leaf
            nodes.  ``-1`` for internal nodes.
        left: Left child (``None`` for leaves).
        right: Right child (``None`` for leaves).
    """

    round_index: int
    team_index: int = -1
    left: BracketNode | None = None
    right: BracketNode | None = None

    @property
    def is_leaf(self) -> bool:
        """Return ``True`` if this is a leaf (team) node."""
        return self.left is None and self.right is None


@dataclass(frozen=True)
class BracketStructure:
    """Immutable tournament bracket.

    Attributes:
        root: Root :class:`BracketNode` of the bracket tree.
        team_ids: Tuple of team IDs in bracket-position order (leaf order).
        team_index_map: Mapping of ``team_id → index`` into ``team_ids``.
        seed_map: Mapping of ``team_id → seed_num`` for seed-aware scoring.
    """

    root: BracketNode
    team_ids: tuple[int, ...]
    team_index_map: dict[int, int]
    seed_map: dict[int, int] = field(default_factory=dict)


def _build_subtree(
    team_indices: list[int],
    round_offset: int,
) -> BracketNode:
    """Recursively build a balanced binary bracket subtree.

    Args:
        team_indices: List of team indices for this sub-bracket (must be
            power-of-2 length).
        round_offset: Round index for games at this level.

    Returns:
        Root :class:`BracketNode` of the subtree.
    """
    if len(team_indices) == 1:
        return BracketNode(round_index=-1, team_index=team_indices[0])

    mid = len(team_indices) // 2
    left = _build_subtree(team_indices[:mid], round_offset - 1)
    right = _build_subtree(team_indices[mid:], round_offset - 1)
    return BracketNode(round_index=round_offset, left=left, right=right)


def build_bracket(seeds: list[TourneySeed], season: int) -> BracketStructure:
    """Construct a 64-team bracket tree from tournament seeds.

    Play-in teams (``is_play_in=True``) are excluded.  Exactly 64 non-play-in
    seeds are required.

    Args:
        seeds: List of :class:`TourneySeed` objects for the given season.
        season: Season year to filter seeds.

    Returns:
        Fully constructed :class:`BracketStructure`.

    Raises:
        ValueError: If the number of non-play-in seeds for *season* is not 64.
    """
    season_seeds = [s for s in seeds if s.season == season and not s.is_play_in]

    # Build lookup: (region, seed_num) → team_id
    seed_lookup: dict[tuple[str, int], int] = {}
    seed_num_map: dict[int, int] = {}
    for s in season_seeds:
        seed_lookup[(s.region, s.seed_num)] = s.team_id
        seed_num_map[s.team_id] = s.seed_num

    # Determine team ordering following bracket structure
    team_ids_ordered: list[int] = []
    for region in _REGION_ORDER:
        for seed_a, seed_b in _REGION_SEED_ORDER:
            team_a = seed_lookup.get((region, seed_a))
            team_b = seed_lookup.get((region, seed_b))
            if team_a is None or team_b is None:
                msg = f"Missing seed for region={region}: seed {seed_a} → {team_a}, seed {seed_b} → {team_b}"
                raise ValueError(msg)
            team_ids_ordered.append(team_a)
            team_ids_ordered.append(team_b)

    if len(team_ids_ordered) != 64:
        msg = f"Expected 64 teams, got {len(team_ids_ordered)}"
        raise ValueError(msg)

    team_ids_tuple = tuple(team_ids_ordered)
    team_index_map = {tid: i for i, tid in enumerate(team_ids_tuple)}

    # Build bracket tree
    # 64 leaves → 6 rounds.  Root is round 5 (championship).
    all_indices = list(range(64))
    root = _build_subtree(all_indices, round_offset=N_ROUNDS - 1)

    return BracketStructure(
        root=root,
        team_ids=team_ids_tuple,
        team_index_map=team_index_map,
        seed_map=seed_num_map,
    )
