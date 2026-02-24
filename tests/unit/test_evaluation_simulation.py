"""Unit tests for the tournament simulation engine.

Tests cover:
- Bracket data structures (Task 1)
- Probability provider protocol (Task 2)
- Phylourny analytical computation (Task 3)
- Scoring rules (Task 4)
- SimulationResult data model (Task 5)
- Monte Carlo simulation engine (Task 6)
- High-level orchestrator (Task 7)
"""

from __future__ import annotations

import numpy as np
import pytest

from ncaa_eval.evaluation.simulation import (
    SCORING_REGISTRY,
    BracketNode,
    BracketStructure,
    CustomScoring,
    EloProvider,
    FibonacciScoring,
    MatchupContext,
    MatrixProvider,
    ScoringRule,
    SeedDiffBonusScoring,
    SimulationResult,
    StandardScoring,
    build_bracket,
    build_probability_matrix,
    compute_advancement_probs,
    compute_expected_points,
    simulate_tournament,
    simulate_tournament_mc,
)
from ncaa_eval.transform.normalization import TourneySeed

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_seeds(season: int = 2024) -> list[TourneySeed]:
    """Create 64 non-play-in seeds for testing (4 regions × 16 seeds).

    Team IDs are: region_offset * 100 + seed_num.
    W → 100s, X → 200s, Y → 300s, Z → 400s.
    """
    seeds: list[TourneySeed] = []
    for region_idx, region in enumerate("WXYZ"):
        base = (region_idx + 1) * 100
        for seed_num in range(1, 17):
            seeds.append(
                TourneySeed(
                    season=season,
                    team_id=base + seed_num,
                    seed_str=f"{region}{seed_num:02d}",
                    region=region,
                    seed_num=seed_num,
                    is_play_in=False,
                )
            )
    return seeds


def _make_bracket(season: int = 2024) -> BracketStructure:
    """Build a standard 64-team bracket for testing."""
    return build_bracket(_make_seeds(season), season)


def _make_context(season: int = 2024) -> MatchupContext:
    """Create a default matchup context for testing."""
    return MatchupContext(season=season, day_num=136, is_neutral=True)


def _make_uniform_matrix(n: int) -> np.ndarray:
    """Create an n×n probability matrix where all matchups are 50/50."""
    P = np.full((n, n), 0.5, dtype=np.float64)
    np.fill_diagonal(P, 0.0)
    return P


def _make_deterministic_matrix(n: int) -> np.ndarray:
    """Create an n×n matrix where team i always beats team j when i < j."""
    P = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i < j:
                P[i, j] = 1.0
            elif i > j:
                P[i, j] = 0.0
    return P


def _make_small_bracket(n: int) -> BracketStructure:
    """Build a small bracket for analytical verification.

    Creates a bracket with n teams (must be power of 2, <= 8).
    """
    team_ids = tuple(range(n))
    team_index_map = {i: i for i in range(n)}

    def _build(indices: list[int], depth: int) -> BracketNode:
        if len(indices) == 1:
            return BracketNode(round_index=-1, team_index=indices[0])
        mid = len(indices) // 2
        left = _build(indices[:mid], depth - 1)
        right = _build(indices[mid:], depth - 1)
        return BracketNode(round_index=depth, left=left, right=right)

    n_rounds = int(np.log2(n))
    root = _build(list(range(n)), n_rounds - 1)
    return BracketStructure(
        root=root,
        team_ids=team_ids,
        team_index_map=team_index_map,
    )


# ---------------------------------------------------------------------------
# Task 1: Bracket data structures
# ---------------------------------------------------------------------------


class TestBracketNode:
    """Tests for BracketNode."""

    def test_leaf_node(self) -> None:
        node = BracketNode(round_index=-1, team_index=0)
        assert node.is_leaf
        assert node.team_index == 0
        assert node.left is None
        assert node.right is None

    def test_internal_node(self) -> None:
        left = BracketNode(round_index=-1, team_index=0)
        right = BracketNode(round_index=-1, team_index=1)
        parent = BracketNode(round_index=0, left=left, right=right)
        assert not parent.is_leaf
        assert parent.round_index == 0
        assert parent.left is left
        assert parent.right is right

    def test_frozen(self) -> None:
        node = BracketNode(round_index=-1, team_index=0)
        with pytest.raises(AttributeError):
            node.team_index = 5  # type: ignore[misc]


class TestBracketStructure:
    """Tests for BracketStructure and build_bracket."""

    def test_correct_number_of_leaves(self) -> None:
        bracket = _make_bracket()
        assert len(bracket.team_ids) == 64

    def test_correct_depth(self) -> None:
        bracket = _make_bracket()
        # Count depth by traversing down the left side
        depth = 0
        node = bracket.root
        while not node.is_leaf:
            assert node.left is not None
            node = node.left
            depth += 1
        assert depth == 6  # 6 rounds for 64 teams

    def test_correct_matchup_pairing_by_seed(self) -> None:
        """Verify that 1-seed is paired against 16-seed in R64, etc."""
        bracket = _make_bracket()
        # In the W region (team IDs 101-116), the first two leaves should be
        # 1-seed (101) and 16-seed (116), since matchup order is (1,16) first.
        assert bracket.team_ids[0] == 101  # W 1-seed
        assert bracket.team_ids[1] == 116  # W 16-seed
        assert bracket.team_ids[2] == 108  # W 8-seed
        assert bracket.team_ids[3] == 109  # W 9-seed

    def test_team_index_map_consistent(self) -> None:
        bracket = _make_bracket()
        for team_id, idx in bracket.team_index_map.items():
            assert bracket.team_ids[idx] == team_id

    def test_all_64_unique_teams(self) -> None:
        bracket = _make_bracket()
        assert len(set(bracket.team_ids)) == 64

    def test_seed_map_populated(self) -> None:
        bracket = _make_bracket()
        assert len(bracket.seed_map) == 64
        # W 1-seed (team 101) should map to seed_num 1
        assert bracket.seed_map[101] == 1
        assert bracket.seed_map[116] == 16

    def test_regions_ordered_correctly(self) -> None:
        """First 16 leaves are W region, next 16 are X, etc."""
        bracket = _make_bracket()
        # W region teams (101-116)
        w_teams = set(range(101, 117))
        first_16 = set(bracket.team_ids[:16])
        assert first_16 == w_teams

    def test_play_in_seeds_excluded(self) -> None:
        seeds = _make_seeds()
        # Add a play-in seed
        seeds.append(
            TourneySeed(
                season=2024,
                team_id=999,
                seed_str="W16a",
                region="W",
                seed_num=16,
                is_play_in=True,
            )
        )
        bracket = build_bracket(seeds, 2024)
        assert 999 not in bracket.team_ids

    def test_wrong_season_raises(self) -> None:
        seeds = _make_seeds(2024)
        with pytest.raises(ValueError, match="Missing seed"):
            build_bracket(seeds, 2023)  # No seeds for 2023


class TestMatchupContext:
    """Tests for MatchupContext."""

    def test_frozen(self) -> None:
        ctx = MatchupContext(season=2024, day_num=136, is_neutral=True)
        with pytest.raises(AttributeError):
            ctx.season = 2025  # type: ignore[misc]

    def test_attributes(self) -> None:
        ctx = MatchupContext(season=2024, day_num=136, is_neutral=True)
        assert ctx.season == 2024
        assert ctx.day_num == 136
        assert ctx.is_neutral is True


# ---------------------------------------------------------------------------
# Task 2: Probability provider protocol
# ---------------------------------------------------------------------------


class TestMatrixProvider:
    """Tests for MatrixProvider."""

    def test_matchup_probability(self) -> None:
        P = np.array([[0.0, 0.6], [0.4, 0.0]])
        provider = MatrixProvider(P, [10, 20])
        ctx = _make_context()
        assert provider.matchup_probability(10, 20, ctx) == pytest.approx(0.6)
        assert provider.matchup_probability(20, 10, ctx) == pytest.approx(0.4)

    def test_batch_matchup_probabilities(self) -> None:
        P = np.array([[0.0, 0.7, 0.3], [0.3, 0.0, 0.5], [0.7, 0.5, 0.0]])
        provider = MatrixProvider(P, [1, 2, 3])
        ctx = _make_context()
        result = provider.batch_matchup_probabilities([1, 2], [2, 3], ctx)
        np.testing.assert_allclose(result, [0.7, 0.5])

    def test_complementarity(self) -> None:
        """P(A,B) + P(B,A) = 1."""
        P = np.array([[0.0, 0.65], [0.35, 0.0]])
        provider = MatrixProvider(P, [1, 2])
        ctx = _make_context()
        p_ab = provider.matchup_probability(1, 2, ctx)
        p_ba = provider.matchup_probability(2, 1, ctx)
        assert p_ab + p_ba == pytest.approx(1.0)


class TestEloProvider:
    """Tests for EloProvider."""

    def test_wraps_predict_one(self) -> None:
        class FakeElo:
            def _predict_one(self, team_a_id: int, team_b_id: int) -> float:
                return 0.75 if team_a_id < team_b_id else 0.25

        provider = EloProvider(FakeElo())
        ctx = _make_context()
        assert provider.matchup_probability(1, 2, ctx) == pytest.approx(0.75)
        assert provider.matchup_probability(2, 1, ctx) == pytest.approx(0.25)

    def test_batch_consistency(self) -> None:
        class FakeElo:
            def _predict_one(self, team_a_id: int, team_b_id: int) -> float:
                return 0.6

        provider = EloProvider(FakeElo())
        ctx = _make_context()
        batch = provider.batch_matchup_probabilities([1, 2], [3, 4], ctx)
        assert batch.shape == (2,)
        np.testing.assert_allclose(batch, [0.6, 0.6])

    def test_requires_predict_one(self) -> None:
        with pytest.raises(TypeError, match="_predict_one"):
            EloProvider(object())

    def test_complementarity(self) -> None:
        class FakeElo:
            def _predict_one(self, team_a_id: int, team_b_id: int) -> float:
                return 0.7 if team_a_id < team_b_id else 0.3

        provider = EloProvider(FakeElo())
        ctx = _make_context()
        p_ab = provider.matchup_probability(1, 2, ctx)
        p_ba = provider.matchup_probability(2, 1, ctx)
        assert p_ab + p_ba == pytest.approx(1.0)


class TestBuildProbabilityMatrix:
    """Tests for build_probability_matrix."""

    def test_matrix_symmetry(self) -> None:
        """P[i,j] + P[j,i] = 1 for all i != j."""

        class ConstProvider:
            def matchup_probability(self, a: int, b: int, ctx: MatchupContext) -> float:
                return 0.6

            def batch_matchup_probabilities(
                self, a_ids: list[int], b_ids: list[int], ctx: MatchupContext
            ) -> np.ndarray:
                return np.full(len(a_ids), 0.6, dtype=np.float64)

        ctx = _make_context()
        P = build_probability_matrix(ConstProvider(), [1, 2, 3], ctx)  # type: ignore[arg-type]
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert P[i, j] + P[j, i] == pytest.approx(1.0)

    def test_diagonal_zero(self) -> None:
        P = _make_uniform_matrix(4)
        provider = MatrixProvider(P, list(range(4)))
        ctx = _make_context()
        result = build_probability_matrix(provider, list(range(4)), ctx)
        np.testing.assert_allclose(np.diag(result), 0.0)

    def test_batch_vs_scalar_consistency(self) -> None:
        """Batch call matches individual scalar calls."""

        class TrackingProvider:
            def matchup_probability(self, a: int, b: int, ctx: MatchupContext) -> float:
                return 0.5 + 0.01 * (a - b)

            def batch_matchup_probabilities(
                self,
                a_ids: list[int],
                b_ids: list[int],
                ctx: MatchupContext,
            ) -> np.ndarray:
                return np.array(
                    [self.matchup_probability(a, b, ctx) for a, b in zip(a_ids, b_ids)],
                    dtype=np.float64,
                )

        ctx = _make_context()
        team_ids = [10, 20, 30, 40]
        P = build_probability_matrix(TrackingProvider(), team_ids, ctx)  # type: ignore[arg-type]

        # Verify against individual calls
        provider = TrackingProvider()
        for i, a in enumerate(team_ids):
            for j, b in enumerate(team_ids):
                if i < j:
                    expected = provider.matchup_probability(a, b, ctx)
                    assert P[i, j] == pytest.approx(expected)
                    assert P[j, i] == pytest.approx(1.0 - expected)


# ---------------------------------------------------------------------------
# Task 3: Analytical computation
# ---------------------------------------------------------------------------


class TestComputeAdvancementProbs:
    """Tests for compute_advancement_probs."""

    def test_4_team_bracket_uniform(self) -> None:
        """4-team bracket with 50/50 matchups: all teams equal."""
        bracket = _make_small_bracket(4)
        P = _make_uniform_matrix(4)
        adv = compute_advancement_probs(bracket, P)
        assert adv.shape == (4, 2)

        # Round 0 (semi): each team has 50% chance of winning
        np.testing.assert_allclose(adv[:, 0], 0.5, atol=1e-10)
        assert adv[:, 0].sum() == pytest.approx(2.0)

        # Round 1 (final): each team has 25% chance of winning championship
        np.testing.assert_allclose(adv[:, 1], 0.25, atol=1e-10)
        assert adv[:, 1].sum() == pytest.approx(1.0)

    def test_4_team_bracket_deterministic(self) -> None:
        """4-team bracket where lower-index always wins."""
        bracket = _make_small_bracket(4)
        P = _make_deterministic_matrix(4)
        adv = compute_advancement_probs(bracket, P)

        # Team 0 beats team 1 in R0 (prob=1), then beats winner of 2v3 in R1
        assert adv[0, 0] == pytest.approx(1.0)  # Beats team 1
        assert adv[0, 1] == pytest.approx(1.0)  # Wins championship
        assert adv[1, 0] == pytest.approx(0.0)  # Loses to team 0
        assert adv[2, 0] == pytest.approx(1.0)  # Beats team 3
        assert adv[3, 0] == pytest.approx(0.0)  # Loses to team 2
        assert adv[2, 1] == pytest.approx(0.0)  # Loses to team 0 in final

    def test_8_team_bracket_column_sums(self) -> None:
        """Verify column sums: round 0 → 4, round 1 → 2, round 2 → 1."""
        bracket = _make_small_bracket(8)
        P = _make_uniform_matrix(8)
        adv = compute_advancement_probs(bracket, P)
        assert adv.shape == (8, 3)

        assert adv[:, 0].sum() == pytest.approx(4.0, abs=1e-10)
        assert adv[:, 1].sum() == pytest.approx(2.0, abs=1e-10)
        assert adv[:, 2].sum() == pytest.approx(1.0, abs=1e-10)

    def test_64_team_bracket_column_sums(self) -> None:
        """For 64 teams: round 0 → 32, ..., round 5 → 1."""
        bracket = _make_bracket()
        P = _make_uniform_matrix(64)
        adv = compute_advancement_probs(bracket, P)
        assert adv.shape == (64, 6)

        expected_sums = [32, 16, 8, 4, 2, 1]
        for r, expected in enumerate(expected_sums):
            assert adv[:, r].sum() == pytest.approx(
                expected, abs=1e-8
            ), f"Round {r} sum: {adv[:, r].sum()} != {expected}"

    def test_all_probs_non_negative(self) -> None:
        bracket = _make_bracket()
        rng = np.random.default_rng(42)
        # Random probability matrix
        raw = rng.random((64, 64))
        P = raw / (raw + raw.T)
        np.fill_diagonal(P, 0.0)
        adv = compute_advancement_probs(bracket, P)
        assert np.all(adv >= -1e-15)

    def test_mismatched_size_raises(self) -> None:
        bracket = _make_small_bracket(4)
        P = _make_uniform_matrix(8)
        with pytest.raises(ValueError, match="P has 8 teams but bracket has 4"):
            compute_advancement_probs(bracket, P)

    def test_non_power_of_two_raises(self) -> None:
        bracket = _make_small_bracket(4)
        P = np.zeros((3, 3))
        with pytest.raises(ValueError, match="positive power of 2"):
            compute_advancement_probs(bracket, P)


class TestComputeExpectedPoints:
    """Tests for compute_expected_points."""

    def test_standard_scoring_uniform(self) -> None:
        """Uniform 4-team bracket: EP should be equal for all teams."""
        bracket = _make_small_bracket(4)
        P = _make_uniform_matrix(4)
        adv = compute_advancement_probs(bracket, P)
        # Use a simple scoring: 1 point per round
        rule = CustomScoring(lambda r: 1.0, "flat")
        ep = compute_expected_points(adv, rule)
        # Each team: 0.5 * 1 + 0.25 * 1 = 0.75
        np.testing.assert_allclose(ep, 0.75, atol=1e-10)

    def test_standard_scoring_deterministic(self) -> None:
        """Deterministic 4-team: team 0 wins everything."""
        bracket = _make_small_bracket(4)
        P = _make_deterministic_matrix(4)
        adv = compute_advancement_probs(bracket, P)
        rule = StandardScoring()
        ep = compute_expected_points(adv, rule)
        # Team 0: 1*1 + 1*2 = 3 (wins R0 at 1pt, R1 at 2pt)
        assert ep[0] == pytest.approx(3.0)
        # Team 2: 1*1 + 0*2 = 1 (wins R0, loses R1)
        assert ep[2] == pytest.approx(1.0)
        # Team 1, 3: lose R0 → 0 EP
        assert ep[1] == pytest.approx(0.0)
        assert ep[3] == pytest.approx(0.0)

    def test_perfect_bracket_standard_scoring(self) -> None:
        """Perfect bracket total for StandardScoring with 64 teams = 192."""
        rule = StandardScoring()
        # Perfect bracket: all correct = sum over rounds of games * points
        # R0: 32*1=32, R1: 16*2=32, R2: 8*4=32, R3: 4*8=32, R4: 2*16=32, R5: 1*32=32
        total = sum((32 // (2**r)) * rule.points_per_round(r) for r in range(6))
        assert total == 192


class TestAnalyticalMatchesMC:
    """Verify analytical EP matches MC EP within statistical tolerance."""

    def test_analytical_vs_mc_convergence(self) -> None:
        """At large N, MC advancement probs converge to analytical."""
        bracket = _make_small_bracket(8)
        n = 8
        rng_matrix = np.random.default_rng(123)
        raw = rng_matrix.random((n, n))
        P = raw / (raw + raw.T)
        np.fill_diagonal(P, 0.0)

        # Analytical
        adv_analytical = compute_advancement_probs(bracket, P)

        # MC with large N
        rules = [StandardScoring()]
        mc_result = simulate_tournament_mc(
            bracket, P, rules, season=2024, n_simulations=50_000, rng=np.random.default_rng(42)
        )

        # MC advancement probs should be close to analytical
        np.testing.assert_allclose(
            mc_result.advancement_probs,
            adv_analytical,
            atol=0.02,
        )


# ---------------------------------------------------------------------------
# Task 4: Scoring rules
# ---------------------------------------------------------------------------


class TestScoringRules:
    """Tests for scoring rule implementations."""

    def test_standard_scoring_values(self) -> None:
        rule = StandardScoring()
        assert rule.name == "standard"
        assert rule.points_per_round(0) == 1.0
        assert rule.points_per_round(1) == 2.0
        assert rule.points_per_round(2) == 4.0
        assert rule.points_per_round(3) == 8.0
        assert rule.points_per_round(4) == 16.0
        assert rule.points_per_round(5) == 32.0

    def test_fibonacci_scoring_values(self) -> None:
        rule = FibonacciScoring()
        assert rule.name == "fibonacci"
        assert rule.points_per_round(0) == 2.0
        assert rule.points_per_round(1) == 3.0
        assert rule.points_per_round(2) == 5.0
        assert rule.points_per_round(3) == 8.0
        assert rule.points_per_round(4) == 13.0
        assert rule.points_per_round(5) == 21.0

    def test_fibonacci_perfect_bracket(self) -> None:
        rule = FibonacciScoring()
        # R0: 32×2=64, R1: 16×3=48, R2: 8×5=40, R3: 4×8=32, R4: 2×13=26, R5: 1×21=21
        total = sum((32 // (2**r)) * rule.points_per_round(r) for r in range(6))
        assert total == 231

    def test_seed_diff_bonus(self) -> None:
        seed_map = {1: 1, 2: 16}
        rule = SeedDiffBonusScoring(seed_map)
        assert rule.name == "seed_diff_bonus"
        # Lower seed wins (upset): bonus = |16 - 1| = 15
        assert rule.seed_diff_bonus(16, 1) == 15.0
        # Higher seed wins (expected): no bonus
        assert rule.seed_diff_bonus(1, 16) == 0.0
        # Equal seeds: no bonus
        assert rule.seed_diff_bonus(5, 5) == 0.0

    def test_custom_scoring(self) -> None:
        rule = CustomScoring(lambda r: float(r + 1), "linear")
        assert rule.name == "linear"
        assert rule.points_per_round(0) == 1.0
        assert rule.points_per_round(5) == 6.0

    def test_scoring_registry(self) -> None:
        assert "standard" in SCORING_REGISTRY
        assert "fibonacci" in SCORING_REGISTRY
        assert SCORING_REGISTRY["standard"] is StandardScoring
        assert SCORING_REGISTRY["fibonacci"] is FibonacciScoring


# ---------------------------------------------------------------------------
# Task 5: SimulationResult
# ---------------------------------------------------------------------------


class TestSimulationResult:
    """Tests for SimulationResult dataclass."""

    def test_construction(self) -> None:
        adv = np.zeros((4, 2))
        ep = {"standard": np.zeros(4)}
        result = SimulationResult(
            season=2024,
            advancement_probs=adv,
            expected_points=ep,
            method="analytical",
            n_simulations=None,
            confidence_intervals=None,
            score_distribution=None,
        )
        assert result.season == 2024
        assert result.method == "analytical"
        assert result.n_simulations is None
        assert result.score_distribution is None

    def test_immutability(self) -> None:
        result = SimulationResult(
            season=2024,
            advancement_probs=np.zeros((4, 2)),
            expected_points={},
            method="analytical",
            n_simulations=None,
            confidence_intervals=None,
            score_distribution=None,
        )
        with pytest.raises(AttributeError):
            result.season = 2025  # type: ignore[misc]

    def test_mc_result(self) -> None:
        result = SimulationResult(
            season=2024,
            advancement_probs=np.zeros((4, 2)),
            expected_points={"standard": np.zeros(4)},
            method="monte_carlo",
            n_simulations=10_000,
            confidence_intervals=None,
            score_distribution={"standard": np.zeros(10_000)},
        )
        assert result.method == "monte_carlo"
        assert result.n_simulations == 10_000
        assert result.score_distribution is not None


# ---------------------------------------------------------------------------
# Task 6: Monte Carlo simulation
# ---------------------------------------------------------------------------


class TestMonteCarlo:
    """Tests for simulate_tournament_mc."""

    def test_bracket_integrity_one_champion(self) -> None:
        """Exactly 1 champion per simulation (round 5 sums to ~1)."""
        bracket = _make_small_bracket(8)
        P = _make_uniform_matrix(8)
        rules = [StandardScoring()]
        result = simulate_tournament_mc(
            bracket, P, rules, season=2024, n_simulations=1000, rng=np.random.default_rng(0)
        )
        # Advancement probs for last round should sum to 1
        assert result.advancement_probs[:, -1].sum() == pytest.approx(1.0, abs=0.01)

    def test_advancement_monotonically_decreasing(self) -> None:
        """Total advancement counts decrease each round."""
        bracket = _make_small_bracket(8)
        P = _make_uniform_matrix(8)
        rules = [StandardScoring()]
        result = simulate_tournament_mc(
            bracket, P, rules, season=2024, n_simulations=5000, rng=np.random.default_rng(1)
        )
        round_sums = result.advancement_probs.sum(axis=0)
        for r in range(len(round_sums) - 1):
            assert round_sums[r] > round_sums[r + 1]

    def test_mc_convergence_to_analytical(self) -> None:
        """MC probs converge to analytical at large N."""
        bracket = _make_small_bracket(4)
        P = _make_uniform_matrix(4)
        adv_exact = compute_advancement_probs(bracket, P)

        rules = [StandardScoring()]
        result = simulate_tournament_mc(
            bracket, P, rules, season=2024, n_simulations=50_000, rng=np.random.default_rng(99)
        )
        np.testing.assert_allclose(result.advancement_probs, adv_exact, atol=0.015)

    def test_score_distribution_populated(self) -> None:
        bracket = _make_small_bracket(4)
        P = _make_uniform_matrix(4)
        rules = [StandardScoring()]
        result = simulate_tournament_mc(
            bracket, P, rules, season=2024, n_simulations=500, rng=np.random.default_rng(2)
        )
        assert result.score_distribution is not None
        assert "standard" in result.score_distribution
        assert result.score_distribution["standard"].shape == (500,)

    def test_minimum_simulations(self) -> None:
        bracket = _make_small_bracket(4)
        P = _make_uniform_matrix(4)
        with pytest.raises(ValueError, match="n_simulations must be >= 100"):
            simulate_tournament_mc(bracket, P, [StandardScoring()], season=2024, n_simulations=50)

    def test_deterministic_mc(self) -> None:
        """With deterministic probs, MC should give exact results."""
        bracket = _make_small_bracket(4)
        P = _make_deterministic_matrix(4)
        rules = [StandardScoring()]
        result = simulate_tournament_mc(
            bracket, P, rules, season=2024, n_simulations=1000, rng=np.random.default_rng(3)
        )
        # Team 0 always wins everything
        assert result.advancement_probs[0, 0] == pytest.approx(1.0)
        assert result.advancement_probs[0, 1] == pytest.approx(1.0)
        assert result.advancement_probs[1, 0] == pytest.approx(0.0)

    def test_reproducibility(self) -> None:
        bracket = _make_small_bracket(4)
        P = _make_uniform_matrix(4)
        rules = [StandardScoring()]
        r1 = simulate_tournament_mc(
            bracket, P, rules, season=2024, n_simulations=200, rng=np.random.default_rng(42)
        )
        r2 = simulate_tournament_mc(
            bracket, P, rules, season=2024, n_simulations=200, rng=np.random.default_rng(42)
        )
        np.testing.assert_array_equal(r1.advancement_probs, r2.advancement_probs)


# ---------------------------------------------------------------------------
# Task 7: Orchestrator
# ---------------------------------------------------------------------------


class TestSimulateTournament:
    """Tests for the high-level simulate_tournament orchestrator."""

    def test_analytical_path(self) -> None:
        bracket = _make_small_bracket(4)
        P = _make_uniform_matrix(4)
        provider = MatrixProvider(P, list(range(4)))
        ctx = _make_context()

        result = simulate_tournament(bracket, provider, ctx, method="analytical")
        assert result.method == "analytical"
        assert result.n_simulations is None
        assert result.score_distribution is None
        assert "standard" in result.expected_points

    def test_mc_path(self) -> None:
        bracket = _make_small_bracket(4)
        P = _make_uniform_matrix(4)
        provider = MatrixProvider(P, list(range(4)))
        ctx = _make_context()

        result = simulate_tournament(
            bracket,
            provider,
            ctx,
            method="monte_carlo",
            n_simulations=500,
            rng=np.random.default_rng(0),
        )
        assert result.method == "monte_carlo"
        assert result.n_simulations == 500

    def test_invalid_method_raises(self) -> None:
        bracket = _make_small_bracket(4)
        P = _make_uniform_matrix(4)
        provider = MatrixProvider(P, list(range(4)))
        ctx = _make_context()

        with pytest.raises(ValueError, match="method must be one of"):
            simulate_tournament(bracket, provider, ctx, method="invalid")

    def test_custom_scoring_rules(self) -> None:
        bracket = _make_small_bracket(4)
        P = _make_uniform_matrix(4)
        provider = MatrixProvider(P, list(range(4)))
        ctx = _make_context()

        rules: list[ScoringRule] = [StandardScoring(), FibonacciScoring()]
        result = simulate_tournament(bracket, provider, ctx, scoring_rules=rules)
        assert "standard" in result.expected_points
        assert "fibonacci" in result.expected_points

    def test_end_to_end_from_seeds(self) -> None:
        """Integration: TourneySeed → BracketStructure → Provider → Result."""
        seeds = _make_seeds(2024)
        bracket = build_bracket(seeds, 2024)
        P = _make_uniform_matrix(64)
        provider = MatrixProvider(P, list(bracket.team_ids))
        ctx = MatchupContext(season=2024, day_num=136, is_neutral=True)

        result = simulate_tournament(bracket, provider, ctx)
        assert result.season == 2024
        assert result.advancement_probs.shape == (64, 6)
        assert result.advancement_probs[:, 5].sum() == pytest.approx(1.0, abs=1e-8)

    def test_mc_minimum_simulations(self) -> None:
        bracket = _make_small_bracket(4)
        P = _make_uniform_matrix(4)
        provider = MatrixProvider(P, list(range(4)))
        ctx = _make_context()

        with pytest.raises(ValueError, match="n_simulations must be >= 100"):
            simulate_tournament(bracket, provider, ctx, method="monte_carlo", n_simulations=50)
