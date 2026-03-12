"""Monte Carlo and analytical tournament simulation engine.

Implements the Phylourny algorithm (Bettisworth & Jordan 2023) for exact
advancement probability computation, plus a vectorized Monte Carlo fallback
for score-distribution analysis.  Provides a high-level
:func:`simulate_tournament` orchestrator.

Key components:

* :func:`compute_advancement_probs` — Phylourny analytical computation.
* :func:`compute_expected_points` — ``adv_probs @ points_vector``.
* :func:`simulate_tournament_mc` — vectorized MC simulation engine.
* :func:`simulate_tournament` — high-level orchestrator.

Bracket data structures, probability providers, and scoring rules are in
their respective submodules (:mod:`bracket`, :mod:`providers`, :mod:`scoring`).

References:
    Bettisworth et al. (2023), "Phylourny: efficiently calculating
    elimination tournament win probabilities via phylogenetic methods,"
    *Statistics and Computing* 33(4):80.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ncaa_eval.evaluation.bracket import (
    N_GAMES,
    N_ROUNDS,
    BracketNode,
    BracketStructure,
    MatchupContext,
    _build_subtree,
    build_bracket,
)
from ncaa_eval.evaluation.providers import (
    EloProvider,
    MatrixProvider,
    ProbabilityProvider,
    build_probability_matrix,
)
from ncaa_eval.evaluation.scoring import (
    _SCORING_REGISTRY,
    CustomScoring,
    DictScoring,
    FibonacciScoring,
    ScoringNotFoundError,
    ScoringRule,
    SeedDiffBonusScoring,
    StandardScoring,
    get_scoring,
    list_scorings,
    register_scoring,
    scoring_from_config,
)

# Re-export all symbols for backward compatibility
__all__ = [
    "BracketDistribution",
    "BracketNode",
    "BracketStructure",
    "CustomScoring",
    "DictScoring",
    "EloProvider",
    "FibonacciScoring",
    "MatchupContext",
    "MatrixProvider",
    "MostLikelyBracket",
    "N_GAMES",
    "N_ROUNDS",
    "ProbabilityProvider",
    "ScoringNotFoundError",
    "ScoringRule",
    "SeedDiffBonusScoring",
    "SimulationResult",
    "StandardScoring",
    "_SCORING_REGISTRY",
    "_build_subtree",
    "_collect_leaves",
    "build_bracket",
    "build_probability_matrix",
    "compute_advancement_probs",
    "compute_bracket_distribution",
    "compute_expected_points",
    "compute_expected_points_seed_diff",
    "compute_most_likely_bracket",
    "get_scoring",
    "list_scorings",
    "register_scoring",
    "score_bracket_against_sims",
    "scoring_from_config",
    "simulate_tournament",
    "simulate_tournament_mc",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SimulationResult dataclasses
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
        bracket_distributions: Optional mapping of
            ``rule_name → BracketDistribution`` (MC only; ``None`` for analytical).
            Note: distributions are computed from the chalk-bracket score (how many
            pre-game favorites won).  For pool scoring analysis ("how would *my*
            chosen bracket score across all simulations?"), use ``sim_winners`` with
            :func:`score_bracket_against_sims`.
        sim_winners: Optional array of per-simulation game winners,
            shape ``(n_simulations, n_games)`` (MC only; ``None`` for analytical).
    """

    season: int
    advancement_probs: npt.NDArray[np.float64]
    expected_points: dict[str, npt.NDArray[np.float64]]
    method: str
    n_simulations: int | None
    confidence_intervals: dict[str, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] | None
    score_distribution: dict[str, npt.NDArray[np.float64]] | None
    bracket_distributions: dict[str, BracketDistribution] | None = None
    sim_winners: npt.NDArray[np.int32] | None = None


@dataclass(frozen=True)
class BracketDistribution:
    """Score distribution statistics from Monte Carlo simulation.

    Attributes:
        scores: Raw per-simulation scores, shape ``(n_simulations,)``.
        percentiles: Mapping of percentile → value for keys 5, 25, 50, 75, 95.
        mean: Mean score across simulations.
        std: Standard deviation of scores.
        histogram_bins: Histogram bin edges, shape ``(n_bins + 1,)``.
        histogram_counts: Histogram counts, shape ``(n_bins,)``.
    """

    scores: npt.NDArray[np.float64]
    percentiles: dict[int, float]
    mean: float
    std: float
    histogram_bins: npt.NDArray[np.float64]
    histogram_counts: npt.NDArray[np.int64]


@dataclass(frozen=True)
class MostLikelyBracket:
    """Maximum-likelihood bracket from greedy traversal.

    Attributes:
        winners: Tuple of team indices for each game's predicted winner,
            in **round-major order** matching ``SimulationResult.sim_winners``
            rows — all Round-of-64 games first (indices 0–31 for 64 teams),
            then Round-of-32 (32–47), through to the championship (index 62).
            63 entries for a 64-team bracket.  Pass directly to
            :func:`score_bracket_against_sims` as ``chosen_bracket``.
        champion_team_id: Canonical team ID of the predicted champion
            (from BracketStructure.team_ids[champion_index]).
        log_likelihood: Sum of ``log(max(P[left, right], P[right, left]))``
            across all games.
    """

    winners: tuple[int, ...]
    champion_team_id: int
    log_likelihood: float


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
        """Post-order traversal returning WPV at this node.

        Performs post-order traversal of the bracket tree, computing Win
        Probability Vectors (WPVs) at each internal node using the formula
        R = V * (P^T * W) + W * (P^T * V), and accumulates per-round
        advancement probabilities into the outer array.
        """
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


def compute_expected_points_seed_diff(
    adv_probs: npt.NDArray[np.float64],
    bracket: BracketStructure,
    P: npt.NDArray[np.float64],
    seed_map: dict[int, int],
) -> npt.NDArray[np.float64]:
    """Compute Expected Points with seed-difference upset bonus.

    Extends standard EP by adding per-matchup seed-diff bonus.  For each
    internal bracket node at round *r*, the bonus contribution for team *i*
    beating opponent *j* is::

        P(i reaches node) * P(i beats j) * P(j reaches node) * bonus(seed_i, seed_j)

    where ``bonus = |seed_i - seed_j|`` when ``seed_i > seed_j`` (upset), else 0.

    Uses ``SeedDiffBonusScoring`` base points for standard round points and
    a post-order traversal of the bracket tree (reusing WPVs from
    :func:`compute_advancement_probs` logic) for bonus computation.

    Args:
        adv_probs: Advancement probabilities, shape ``(n, n_rounds)``.
        bracket: Tournament bracket structure (for tree traversal).
        P: Pairwise win probability matrix, shape ``(n, n)``.
        seed_map: Mapping of ``team_id → seed_num``.

    Returns:
        Expected Points per team, shape ``(n,)``, including base + bonus.
    """
    n = P.shape[0]

    # Base EP from standard round points (same as StandardScoring)
    base_rule = SeedDiffBonusScoring(seed_map)
    base_ep = compute_expected_points(adv_probs, base_rule)

    # Build seed vector indexed by team_index (bracket position)
    seed_vec = np.zeros(n, dtype=np.float64)
    for team_id, idx in bracket.team_index_map.items():
        seed_vec[idx] = float(seed_map.get(team_id, 0))

    # Precompute bonus matrix: bonus[i,j] = |seed_i - seed_j| if seed_i > seed_j else 0
    seed_diff = seed_vec[:, None] - seed_vec[None, :]  # (n, n)
    bonus_matrix = np.where(seed_diff > 0, seed_diff, 0.0)

    # Bonus EP via bracket tree traversal
    bonus_ep = np.zeros(n, dtype=np.float64)

    def _traverse_bonus(node: BracketNode) -> npt.NDArray[np.float64]:
        """Post-order traversal returning WPV and accumulating bonus EP.

        Reuses the Phylourny WPV traversal while also accumulating
        seed-difference upset bonuses via element-wise multiplication of the
        probability matrix by a pre-built bonus matrix, with results
        accumulated into the outer bonus_ep array.
        """
        if node.is_leaf:
            wpv = np.zeros(n, dtype=np.float64)
            wpv[node.team_index] = 1.0
            return wpv

        if node.left is None or node.right is None:
            msg = "Internal bracket node missing child — tree is malformed"
            raise RuntimeError(msg)
        left_wpv = _traverse_bonus(node.left)
        right_wpv = _traverse_bonus(node.right)

        # WPV at this node (same as Phylourny)
        wpv = left_wpv * (P @ right_wpv) + right_wpv * (P @ left_wpv)

        # Bonus: for each team i on the left beating opponent j on the right
        # bonus_ep[i] += left_wpv[i] * P[i,j] * right_wpv[j] * bonus[i,j]
        # Vectorized: left_wpv * ((P * bonus_matrix) @ right_wpv)
        #           + right_wpv * ((P * bonus_matrix) @ left_wpv)
        P_bonus = P * bonus_matrix  # element-wise: P[i,j] * bonus[i,j]
        bonus_ep[:] += left_wpv * (P_bonus @ right_wpv)
        bonus_ep[:] += right_wpv * (P_bonus @ left_wpv)

        return wpv

    _traverse_bonus(bracket.root)

    result: npt.NDArray[np.float64] = base_ep + bonus_ep
    return result


def compute_most_likely_bracket(
    bracket: BracketStructure,
    P: npt.NDArray[np.float64],
) -> MostLikelyBracket:
    """Compute the maximum-likelihood bracket via greedy traversal.

    At each internal node, picks the team with the higher win probability
    (``argmax(P[left, right])``).  Returns the full bracket of winners and
    the log-likelihood of the chosen bracket.

    The ``winners`` array is in **round-major order** — the same order as
    ``SimulationResult.sim_winners`` rows — so it can be passed directly to
    :func:`score_bracket_against_sims`:
    all Round-of-64 games first (indices 0–31), then Round-of-32 (32–47),
    through to the championship game (index 62).

    Args:
        bracket: Tournament bracket structure.
        P: Pairwise win probability matrix, shape ``(n, n)``.

    Returns:
        :class:`MostLikelyBracket` with winners, champion, and log-likelihood.
    """
    log_likelihood = 0.0
    # Collect (round_index, game_order_within_round, winner) tuples so we can
    # sort into round-major order matching sim_winners layout.
    games: list[tuple[int, int, int]] = []
    round_counters: dict[int, int] = {}

    def _traverse(node: BracketNode) -> int:
        """Return team index of the predicted winner at this node.

        Greedy post-order traversal that picks the most likely winner at each
        bracket node via argmax of the probability submatrix, recording game
        results and log-likelihoods into outer-scope lists for round-major
        ordering.
        """
        nonlocal log_likelihood
        if node.is_leaf:
            return node.team_index

        if node.left is None or node.right is None:
            msg = "Internal bracket node missing child — tree is malformed"
            raise RuntimeError(msg)

        left_winner = _traverse(node.left)
        right_winner = _traverse(node.right)

        p_left = float(P[left_winner, right_winner])
        if p_left >= 0.5:
            winner = left_winner
            log_likelihood += float(np.log(max(p_left, 1e-300)))
        else:
            winner = right_winner
            log_likelihood += float(np.log(max(1.0 - p_left, 1e-300)))

        r = node.round_index
        game_idx = round_counters.get(r, 0)
        round_counters[r] = game_idx + 1
        games.append((r, game_idx, winner))
        return winner

    champion_index = _traverse(bracket.root)

    # Sort by (round_index, game_index) to produce round-major order
    games.sort(key=lambda x: (x[0], x[1]))
    winners_ordered = tuple(w for _, _, w in games)

    return MostLikelyBracket(
        winners=winners_ordered,
        champion_team_id=bracket.team_ids[champion_index],
        log_likelihood=log_likelihood,
    )


def compute_bracket_distribution(
    scores: npt.NDArray[np.float64],
    n_bins: int = 50,
) -> BracketDistribution:
    """Compute score distribution statistics from raw MC scores.

    Computes the 5th/25th/50th/75th/95th percentiles via ``np.percentile``,
    builds a ``n_bins``-bucket histogram via ``np.histogram``, and wraps all
    statistics into a :class:`BracketDistribution`.

    Args:
        scores: Raw per-simulation scores, shape ``(n_simulations,)``.
        n_bins: Number of histogram bins (default 50).

    Returns:
        :class:`BracketDistribution` with percentiles, mean, std, and histogram.
    """
    percentile_keys = (5, 25, 50, 75, 95)
    pct_values = np.percentile(scores, percentile_keys)
    percentiles = {k: float(v) for k, v in zip(percentile_keys, pct_values)}

    counts_arr, bins_arr = np.histogram(scores, bins=n_bins)

    return BracketDistribution(
        scores=scores,
        percentiles=percentiles,
        mean=float(np.mean(scores)),
        std=float(np.std(scores)),
        histogram_bins=bins_arr.astype(np.float64),
        histogram_counts=counts_arr.astype(np.int64),
    )


def score_bracket_against_sims(
    chosen_bracket: npt.NDArray[np.int32],
    sim_winners: npt.NDArray[np.int32],
    scoring_rules: Sequence[ScoringRule],
) -> dict[str, npt.NDArray[np.float64]]:
    """Score a chosen bracket against each simulated tournament outcome.

    Broadcasts ``chosen_bracket`` across all simulations to build a boolean
    match matrix (``sim_winners == chosen_bracket[None, :]``).  For each
    scoring rule, constructs a per-game point vector by iterating rounds with
    a running ``game_offset``, then computes per-sim scores as
    ``(matches * game_points).sum(axis=1)`` — one vectorized dot product per
    rule, no Python loop over simulations.

    Args:
        chosen_bracket: Game winners for the chosen bracket, shape ``(n_games,)``.
        sim_winners: Per-simulation game winners, shape ``(n_simulations, n_games)``.
        scoring_rules: Scoring rules to score against.

    Returns:
        Mapping of ``rule_name → per-sim scores``, each shape ``(n_simulations,)``.
    """
    n_games = chosen_bracket.shape[0]
    n_rounds = int(np.log2(n_games + 1))

    # Boolean match array: (n_simulations, n_games)
    matches = sim_winners == chosen_bracket[None, :]

    result: dict[str, npt.NDArray[np.float64]] = {}
    for rule in scoring_rules:
        # Build per-game point values based on which round each game belongs to
        game_points = np.zeros(n_games, dtype=np.float64)
        game_offset = 0
        games_in_round = n_games + 1  # n teams for first round → n/2 games
        for r in range(n_rounds):
            games_in_round = games_in_round // 2
            game_points[game_offset : game_offset + games_in_round] = rule.points_per_round(r)
            game_offset += games_in_round

        # Per-sim score: sum of points for matching picks
        scores: npt.NDArray[np.float64] = (matches * game_points[None, :]).sum(axis=1)
        result[rule.name] = scores

    return result


# ---------------------------------------------------------------------------
# Monte Carlo simulation engine (Task 6)
# ---------------------------------------------------------------------------


def simulate_tournament_mc(  # noqa: PLR0913 — REFACTOR Story 8.1
    bracket: BracketStructure,
    P: npt.NDArray[np.float64],
    scoring_rules: Sequence[ScoringRule],
    season: int,
    n_simulations: int = 10_000,
    rng: np.random.Generator | None = None,
    progress: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
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
        progress: Display a tqdm progress bar for simulation rounds.
        progress_callback: Optional callback invoked after each round
            with ``(round_completed, total_rounds)``.  UI-agnostic hook
            for external progress reporting (e.g. Streamlit ``st.progress``).

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

    # Total games in single-elimination bracket: n-1
    total_games = n - 1

    # Pre-generate all random numbers: shape (n_simulations, total_games)
    randoms = rng.random((n_simulations, total_games))

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

    # Track all game winners: shape (n_simulations, total_games)
    all_winners = np.zeros((n_simulations, total_games), dtype=np.int32)

    game_offset = 0
    round_iter: Iterable[int] = range(n_rounds)
    if progress:
        from tqdm.auto import tqdm  # type: ignore[import-untyped]

        round_iter = tqdm(round_iter, desc="MC rounds", total=n_rounds)
    for r in round_iter:
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

        # Store per-game winners
        all_winners[:, game_offset : game_offset + n_games_in_round] = winners

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

        if progress_callback is not None:
            progress_callback(r + 1, n_rounds)

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

    # Compute bracket distributions from score distributions
    bracket_dist_dict: dict[str, BracketDistribution] = {
        name: compute_bracket_distribution(scores) for name, scores in score_dist_dict.items()
    }

    return SimulationResult(
        season=season,
        advancement_probs=adv_probs,
        expected_points=ep_dict,
        method="monte_carlo",
        n_simulations=n_simulations,
        confidence_intervals=None,
        score_distribution=score_dist_dict,
        bracket_distributions=bracket_dist_dict,
        sim_winners=all_winners,
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


def simulate_tournament(  # noqa: PLR0913 — REFACTOR Story 8.1
    bracket: BracketStructure,
    probability_provider: ProbabilityProvider,
    context: MatchupContext,
    scoring_rules: Sequence[ScoringRule] | None = None,
    method: str = "analytical",
    n_simulations: int = 10_000,
    rng: np.random.Generator | None = None,
    progress: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
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
        progress: Display a tqdm progress bar for MC simulation rounds.
            Ignored when ``method="analytical"``.
        progress_callback: Optional callback invoked after each MC round
            with ``(round_completed, total_rounds)``.  Ignored when
            ``method="analytical"``.

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
        ep_dict: dict[str, npt.NDArray[np.float64]] = {}
        for rule in scoring_rules:
            if isinstance(rule, SeedDiffBonusScoring):
                ep_dict[rule.name] = compute_expected_points_seed_diff(adv_probs, bracket, P, rule.seed_map)
            else:
                ep_dict[rule.name] = compute_expected_points(adv_probs, rule)
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
        progress=progress,
        progress_callback=progress_callback,
    )
