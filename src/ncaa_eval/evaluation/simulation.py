"""Monte Carlo and analytical tournament simulation engine.

Implements the Phylourny algorithm (Bettisworth & Jordan 2023) for exact
advancement probability computation, plus a vectorized Monte Carlo fallback
for score-distribution analysis.  Provides bracket data structures,
probability provider protocols, scoring rule plugins, and a high-level
:func:`simulate_tournament` orchestrator.

Key components:

* :class:`BracketNode` / :class:`BracketStructure` — immutable bracket tree.
* :func:`build_bracket` — constructs a 64-team tree from :class:`TourneySeed`.
* :class:`ProbabilityProvider` — protocol for pairwise win probabilities.
* :func:`compute_advancement_probs` — Phylourny analytical computation.
* :func:`compute_expected_points` — ``adv_probs @ points_vector``.
* :func:`simulate_tournament_mc` — vectorized MC simulation engine.
* :func:`simulate_tournament` — high-level orchestrator.

References:
    Bettisworth et al. (2023), "Phylourny: efficiently calculating
    elimination tournament win probabilities via phylogenetic methods,"
    *Statistics and Computing* 33(4):80.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from ncaa_eval.transform.normalization import TourneySeed

logger = logging.getLogger(__name__)

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
# Bracket data structures (Task 1)
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
                msg = (
                    f"Missing seed for region={region}: "
                    f"seed {seed_a} → {team_a}, seed {seed_b} → {team_b}"
                )
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


# ---------------------------------------------------------------------------
# Probability provider protocol (Task 2)
# ---------------------------------------------------------------------------


@runtime_checkable
class ProbabilityProvider(Protocol):
    """Protocol for pairwise win probability computation.

    All implementations must satisfy the complementarity contract:
    ``P(A beats B) + P(B beats A) = 1`` for every ``(A, B)`` pair.
    """

    def matchup_probability(
        self,
        team_a_id: int,
        team_b_id: int,
        context: MatchupContext,
    ) -> float:
        """Return P(team_a beats team_b).

        Args:
            team_a_id: First team's canonical ID.
            team_b_id: Second team's canonical ID.
            context: Matchup context (season, day_num, neutral).

        Returns:
            Probability in ``[0, 1]``.
        """
        ...

    def batch_matchup_probabilities(
        self,
        team_a_ids: Sequence[int],
        team_b_ids: Sequence[int],
        context: MatchupContext,
    ) -> npt.NDArray[np.float64]:
        """Return P(a_i beats b_i) for all pairs.

        Args:
            team_a_ids: Sequence of first-team IDs.
            team_b_ids: Sequence of second-team IDs (same length).
            context: Matchup context.

        Returns:
            1-D float64 array of shape ``(len(team_a_ids),)``.
        """
        ...


class MatrixProvider:
    """Wraps a pre-computed probability matrix as a :class:`ProbabilityProvider`.

    Args:
        prob_matrix: n×n pairwise probability matrix.
        team_ids: Sequence of team IDs matching matrix indices.
    """

    def __init__(
        self,
        prob_matrix: npt.NDArray[np.float64],
        team_ids: Sequence[int],
    ) -> None:
        self._P = prob_matrix
        self._index = {tid: i for i, tid in enumerate(team_ids)}

    def matchup_probability(
        self,
        team_a_id: int,
        team_b_id: int,
        context: MatchupContext,
    ) -> float:
        """Return P(team_a beats team_b) from the stored matrix."""
        i = self._index[team_a_id]
        j = self._index[team_b_id]
        return float(self._P[i, j])

    def batch_matchup_probabilities(
        self,
        team_a_ids: Sequence[int],
        team_b_ids: Sequence[int],
        context: MatchupContext,
    ) -> npt.NDArray[np.float64]:
        """Return batch probabilities from the stored matrix."""
        rows = np.array([self._index[a] for a in team_a_ids])
        cols = np.array([self._index[b] for b in team_b_ids])
        result: npt.NDArray[np.float64] = self._P[rows, cols].astype(np.float64)
        return result


class EloProvider:
    """Wraps a :class:`StatefulModel` as a :class:`ProbabilityProvider`.

    Uses the model's ``_predict_one`` method for probability computation.

    Args:
        model: Any :class:`StatefulModel` instance with ``_predict_one``.
    """

    def __init__(self, model: Any) -> None:
        if not hasattr(model, "_predict_one"):
            msg = "model must have a _predict_one(team_a_id, team_b_id) method"
            raise TypeError(msg)
        self._model: Any = model

    def matchup_probability(
        self,
        team_a_id: int,
        team_b_id: int,
        context: MatchupContext,
    ) -> float:
        """Return P(team_a beats team_b) via the model's ``_predict_one``."""
        result: float = self._model._predict_one(team_a_id, team_b_id)
        return result

    def batch_matchup_probabilities(
        self,
        team_a_ids: Sequence[int],
        team_b_ids: Sequence[int],
        context: MatchupContext,
    ) -> npt.NDArray[np.float64]:
        """Return batch probabilities by looping ``_predict_one``.

        Elo is O(1) per pair so looping is acceptable.
        """
        return np.array(
            [self._model._predict_one(a, b) for a, b in zip(team_a_ids, team_b_ids)],
            dtype=np.float64,
        )


def build_probability_matrix(
    provider: ProbabilityProvider,
    team_ids: Sequence[int],
    context: MatchupContext,
) -> npt.NDArray[np.float64]:
    """Build n×n pairwise win probability matrix.

    Uses upper-triangle batch call, then fills ``P[j,i] = 1 - P[i,j]``
    via the complementarity contract.

    Args:
        provider: Probability provider implementing the protocol.
        team_ids: Team IDs in bracket order.
        context: Matchup context.

    Returns:
        Float64 array of shape ``(n, n)``.  Diagonal is zero.
    """
    n = len(team_ids)
    rows, cols = np.triu_indices(n, k=1)
    a_ids = [team_ids[int(i)] for i in rows]
    b_ids = [team_ids[int(j)] for j in cols]

    probs = provider.batch_matchup_probabilities(a_ids, b_ids, context)

    P = np.zeros((n, n), dtype=np.float64)
    P[rows, cols] = probs
    P[cols, rows] = 1.0 - probs
    return P


# ---------------------------------------------------------------------------
# Scoring rules (Task 4)
# ---------------------------------------------------------------------------


@runtime_checkable
class ScoringRule(Protocol):
    """Protocol for tournament bracket scoring rules."""

    @property
    def name(self) -> str:
        """Human-readable name of the scoring rule."""
        ...

    def points_per_round(self, round_idx: int) -> float:
        """Return points awarded for a correct pick in round *round_idx*.

        Args:
            round_idx: Zero-indexed round number (0=R64 through 5=NCG).

        Returns:
            Points as a float.
        """
        ...


class StandardScoring:
    """ESPN-style scoring: 1-2-4-8-16-32 (192 total for perfect bracket)."""

    _POINTS: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)

    @property
    def name(self) -> str:
        """Return ``'standard'``."""
        return "standard"

    def points_per_round(self, round_idx: int) -> float:
        """Return standard scoring points for *round_idx*."""
        return self._POINTS[round_idx]


class FibonacciScoring:
    """Fibonacci-style scoring: 2-3-5-8-13-21 (231 total for perfect bracket)."""

    _POINTS: tuple[float, ...] = (2.0, 3.0, 5.0, 8.0, 13.0, 21.0)

    @property
    def name(self) -> str:
        """Return ``'fibonacci'``."""
        return "fibonacci"

    def points_per_round(self, round_idx: int) -> float:
        """Return Fibonacci scoring points for *round_idx*."""
        return self._POINTS[round_idx]


class SeedDiffBonusScoring:
    """Base points + seed-difference bonus when lower seed wins.

    Uses same base as StandardScoring (1-2-4-8-16-32).  When the lower
    seed (higher seed number) wins, adds ``|seed_a - seed_b|`` bonus.

    Note: This scoring rule's ``points_per_round`` returns only the base
    points.  Full EP computation for seed-diff scoring (which requires
    per-matchup seed information) is deferred to Story 6.6, which will add
    a dedicated ``compute_expected_points_seed_diff`` function.

    Args:
        seed_map: Mapping of ``team_id → seed_num``.
    """

    _BASE_POINTS: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)

    def __init__(self, seed_map: dict[int, int]) -> None:
        self._seed_map = seed_map

    @property
    def name(self) -> str:
        """Return ``'seed_diff_bonus'``."""
        return "seed_diff_bonus"

    def points_per_round(self, round_idx: int) -> float:
        """Return base points (excludes seed-diff bonus)."""
        return self._BASE_POINTS[round_idx]

    def seed_diff_bonus(self, seed_a: int, seed_b: int) -> float:
        """Return bonus points when the lower seed wins.

        Args:
            seed_a: Winner's seed number.
            seed_b: Loser's seed number.

        Returns:
            ``|seed_a - seed_b|`` if winner has higher seed number
            (lower seed = upset), else 0.
        """
        if seed_a > seed_b:
            return float(abs(seed_a - seed_b))
        return 0.0

    @property
    def seed_map(self) -> dict[int, int]:
        """Return the seed lookup map."""
        return self._seed_map


class CustomScoring:
    """User-defined scoring rule wrapping a callable.

    Args:
        scoring_fn: Callable mapping ``round_idx`` → points.
        scoring_name: Name for this custom rule.
    """

    def __init__(self, scoring_fn: Callable[[int], float], scoring_name: str) -> None:
        self._fn = scoring_fn
        self._name = scoring_name

    @property
    def name(self) -> str:
        """Return the custom rule name."""
        return self._name

    def points_per_round(self, round_idx: int) -> float:
        """Return points from the wrapped callable."""
        return self._fn(round_idx)


#: Registry of built-in scoring rules (analogous to model registry).
SCORING_REGISTRY: dict[str, type[StandardScoring] | type[FibonacciScoring]] = {
    "standard": StandardScoring,
    "fibonacci": FibonacciScoring,
}


# ---------------------------------------------------------------------------
# SimulationResult (Task 5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulationResult:
    """Result of tournament simulation for one season.

    Both the analytical path and MC path produce a ``SimulationResult``.

    Attributes:
        season: Tournament season year.
        advancement_probs: Per-team advancement probabilities,
            shape ``(n_teams, n_rounds)``.
        expected_points: Mapping of ``scoring_rule_name → per-team EP``,
            each shape ``(n_teams,)``.
        method: ``"analytical"`` or ``"monte_carlo"``.
        n_simulations: ``None`` for analytical; N for MC.
        confidence_intervals: Optional mapping of
            ``rule_name → (lower, upper)`` arrays.
        score_distribution: Optional mapping of
            ``rule_name → per-sim scores`` array, shape ``(n_simulations,)``.
    """

    season: int
    advancement_probs: npt.NDArray[np.float64]
    expected_points: dict[str, npt.NDArray[np.float64]]
    method: str
    n_simulations: int | None
    confidence_intervals: dict[str, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] | None
    score_distribution: dict[str, npt.NDArray[np.float64]] | None


# ---------------------------------------------------------------------------
# Analytical computation — Phylourny algorithm (Task 3)
# ---------------------------------------------------------------------------


def compute_advancement_probs(
    bracket: BracketStructure,
    P: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute exact advancement probabilities via the Phylourny algorithm.

    Post-order traversal of the bracket tree computing Win Probability
    Vectors (WPVs) at each internal node using the formula:

        ``R = V ⊙ (P^T · W) + W ⊙ (P^T · V)``

    Args:
        bracket: Tournament bracket structure.
        P: Pairwise win probability matrix, shape ``(n, n)``.

    Returns:
        Advancement probabilities, shape ``(n, n_rounds)``.
        ``adv_probs[i, r]`` = P(team i wins their game in round r).

    Raises:
        ValueError: If ``n`` is not a power of 2 or does not match
            the bracket's team count.
    """
    n = P.shape[0]
    if n == 0 or (n & (n - 1)) != 0:
        msg = f"n must be a positive power of 2, got {n}"
        raise ValueError(msg)

    expected_teams = len(bracket.team_ids)
    if n != expected_teams:
        msg = f"P has {n} teams but bracket has {expected_teams}"
        raise ValueError(msg)

    n_rounds = int(np.log2(n))
    adv_probs = np.zeros((n, n_rounds), dtype=np.float64)

    def _traverse(node: BracketNode) -> npt.NDArray[np.float64]:
        """Post-order traversal returning WPV at this node."""
        if node.is_leaf:
            wpv = np.zeros(n, dtype=np.float64)
            wpv[node.team_index] = 1.0
            return wpv

        if node.left is None or node.right is None:
            msg = "Internal bracket node missing child — tree is malformed"
            raise RuntimeError(msg)
        left_wpv = _traverse(node.left)
        right_wpv = _traverse(node.right)

        # Phylourny core formula (adapted for P[i,j] = P(team_i beats team_j))
        # R[i] = V[i] * sum_j(P[i,j] * W[j]) + W[i] * sum_j(P[i,j] * V[j])
        wpv = left_wpv * (P @ right_wpv) + right_wpv * (P @ left_wpv)

        # Accumulate — safe because each round has disjoint game slots
        adv_probs[:, node.round_index] += wpv
        return wpv

    _traverse(bracket.root)
    return adv_probs


def compute_expected_points(
    adv_probs: npt.NDArray[np.float64],
    scoring_rule: ScoringRule,
) -> npt.NDArray[np.float64]:
    """Compute Expected Points per team via matrix-vector multiply.

    Args:
        adv_probs: Advancement probabilities, shape ``(n, n_rounds)``.
        scoring_rule: Scoring rule providing per-round point values.

    Returns:
        Expected Points per team, shape ``(n,)``.
    """
    n_rounds = adv_probs.shape[1]
    points = np.array(
        [scoring_rule.points_per_round(r) for r in range(n_rounds)],
        dtype=np.float64,
    )
    result: npt.NDArray[np.float64] = adv_probs @ points
    return result


# ---------------------------------------------------------------------------
# Monte Carlo simulation engine (Task 6)
# ---------------------------------------------------------------------------


def simulate_tournament_mc(  # noqa: PLR0913
    bracket: BracketStructure,
    P: npt.NDArray[np.float64],
    scoring_rules: Sequence[ScoringRule],
    season: int,
    n_simulations: int = 10_000,
    rng: np.random.Generator | None = None,
) -> SimulationResult:
    """Vectorized Monte Carlo tournament simulation.

    All N simulations run in parallel per round (no per-sim Python loops).
    Pre-generates random numbers and uses fancy indexing for batch outcome
    determination.

    Args:
        bracket: Tournament bracket structure (64 teams).
        P: Pairwise win probability matrix, shape ``(n, n)``.
        scoring_rules: Scoring rules to compute scores for.
        season: Tournament season year.
        n_simulations: Number of simulations (default 10,000).
        rng: NumPy random generator for reproducibility.

    Returns:
        :class:`SimulationResult` with MC-derived advancement probs,
        expected points, and score distributions.

    Raises:
        ValueError: If ``n_simulations < 100``.
    """
    if n_simulations < 100:
        msg = f"n_simulations must be >= 100, got {n_simulations}"
        raise ValueError(msg)

    if rng is None:
        rng = np.random.default_rng()

    n = P.shape[0]
    n_rounds = int(np.log2(n))

    # Flatten bracket leaves into an ordered array of team indices.
    # This is the initial survivor array: shape (n,) where n=64.
    leaf_order = _collect_leaves(bracket.root)

    # Pre-generate all random numbers: shape (n_simulations, n_games)
    randoms = rng.random((n_simulations, N_GAMES))

    # Survivor array: shape (n_simulations, n_teams_current_round)
    # Start with all 64 teams for all sims
    survivors = np.tile(np.array(leaf_order, dtype=np.int32), (n_simulations, 1))

    # Track advancement counts: shape (n, n_rounds)
    advancement_counts = np.zeros((n, n_rounds), dtype=np.int64)

    # Per-round chalk results: list of (n_simulations, n_games_in_round) bool arrays.
    # For each game, True if the pre-game favorite (P >= 0.5) actually won.
    # Used to compute a meaningful per-sim score distribution that varies across
    # simulations as upsets occur.
    chalk_results: list[npt.NDArray[np.bool_]] = []

    game_offset = 0
    for r in range(n_rounds):
        n_games_in_round = survivors.shape[1] // 2

        if n_simulations >= 10_000 and r == 0:
            logger.info(
                "MC simulation: %d sims, round %d/%d (%d games)",
                n_simulations,
                r + 1,
                n_rounds,
                n_games_in_round,
            )

        # Pair adjacent survivors: left vs right
        left_teams = survivors[:, 0::2]  # shape (N, n_games_in_round)
        right_teams = survivors[:, 1::2]  # shape (N, n_games_in_round)

        # Look up P(left beats right) from probability matrix
        probs = P[left_teams, right_teams]  # shape (N, n_games_in_round)

        # Determine winners using pre-generated randoms
        round_randoms = randoms[:, game_offset : game_offset + n_games_in_round]
        left_wins = round_randoms < probs  # shape (N, n_games_in_round)

        winners = np.where(left_wins, left_teams, right_teams)

        # Chalk-bracket tracking: for each game, did the pre-game favorite win?
        # This gives genuine per-sim variation: sims with many upsets score less.
        left_favored = probs >= 0.5  # shape (N, n_games_in_round)
        chalk_won: npt.NDArray[np.bool_] = np.where(left_favored, left_wins, ~left_wins)
        chalk_results.append(chalk_won)

        # Accumulate advancement counts (vectorized via np.bincount — no Python loop)
        advancement_counts[:, r] = np.bincount(winners.ravel().astype(np.intp), minlength=n)

        # Update survivors for next round
        survivors = winners
        game_offset += n_games_in_round

    if n_simulations >= 10_000:
        logger.info("MC simulation complete: %d sims", n_simulations)

    # Compute advancement probs
    adv_probs = advancement_counts.astype(np.float64) / n_simulations

    # Compute expected points and score distributions per scoring rule
    ep_dict: dict[str, npt.NDArray[np.float64]] = {}
    score_dist_dict: dict[str, npt.NDArray[np.float64]] = {}

    for rule in scoring_rules:
        # Per-team EP from advancement probs
        ep_dict[rule.name] = compute_expected_points(adv_probs, rule)

        # Score distribution: per-sim chalk bracket score.
        # For each simulation, sum points for games where the pre-game favorite won.
        # Upsets reduce the chalk score, producing a genuine distribution.
        total_scores = np.zeros(n_simulations, dtype=np.float64)
        for r_idx, chalk_won_r in enumerate(chalk_results):
            total_scores += rule.points_per_round(r_idx) * chalk_won_r.sum(axis=1)
        score_dist_dict[rule.name] = total_scores

    return SimulationResult(
        season=season,
        advancement_probs=adv_probs,
        expected_points=ep_dict,
        method="monte_carlo",
        n_simulations=n_simulations,
        confidence_intervals=None,
        score_distribution=score_dist_dict,
    )


def _collect_leaves(node: BracketNode) -> list[int]:
    """Collect leaf team indices in left-to-right order.

    Args:
        node: Root of the subtree.

    Returns:
        List of ``team_index`` values from leaf nodes.
    """
    if node.is_leaf:
        return [node.team_index]
    if node.left is None or node.right is None:
        msg = "Internal bracket node missing child — tree is malformed"
        raise RuntimeError(msg)
    return _collect_leaves(node.left) + _collect_leaves(node.right)


# ---------------------------------------------------------------------------
# High-level orchestrator (Task 7)
# ---------------------------------------------------------------------------


def simulate_tournament(  # noqa: PLR0913
    bracket: BracketStructure,
    probability_provider: ProbabilityProvider,
    context: MatchupContext,
    scoring_rules: Sequence[ScoringRule] | None = None,
    method: str = "analytical",
    n_simulations: int = 10_000,
    rng: np.random.Generator | None = None,
) -> SimulationResult:
    """High-level tournament simulation orchestrator.

    Dispatches to analytical (Phylourny) or Monte Carlo path based on
    *method*.

    Args:
        bracket: Tournament bracket structure.
        probability_provider: Provider for pairwise win probabilities.
        context: Matchup context (season, day_num, neutral).
        scoring_rules: Scoring rules for EP computation.  Defaults to
            :class:`StandardScoring` only.
        method: ``"analytical"`` (default) or ``"monte_carlo"``.
        n_simulations: Number of MC simulations (ignored for analytical).
        rng: NumPy random generator (MC only).

    Returns:
        :class:`SimulationResult`.

    Raises:
        ValueError: If *method* is not ``"analytical"`` or ``"monte_carlo"``,
            or if MC is requested with ``n_simulations < 100``.
    """
    valid_methods = {"analytical", "monte_carlo"}
    if method not in valid_methods:
        msg = f"method must be one of {valid_methods}, got {method!r}"
        raise ValueError(msg)

    if scoring_rules is None:
        scoring_rules = [StandardScoring()]

    # Build probability matrix
    P = build_probability_matrix(probability_provider, bracket.team_ids, context)

    if method == "analytical":
        adv_probs = compute_advancement_probs(bracket, P)
        ep_dict = {rule.name: compute_expected_points(adv_probs, rule) for rule in scoring_rules}
        return SimulationResult(
            season=context.season,
            advancement_probs=adv_probs,
            expected_points=ep_dict,
            method="analytical",
            n_simulations=None,
            confidence_intervals=None,
            score_distribution=None,
        )

    # Monte Carlo path
    return simulate_tournament_mc(
        bracket=bracket,
        P=P,
        scoring_rules=scoring_rules,
        season=context.season,
        n_simulations=n_simulations,
        rng=rng,
    )
