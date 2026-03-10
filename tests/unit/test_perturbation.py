"""Unit tests for the game-theory slider perturbation module.

Tests cover:
- slider_to_temperature() — boundary values, neutral, out-of-range
- power_transform() — identity, compression, sharpening, edge cases
- build_seed_prior_matrix() — lookups, interpolation, same-seed
- perturb_probability_matrix() — identity, combined transform, complementarity
- Parametrized worked examples from spike research (Section 4)
"""

from __future__ import annotations

import numpy as np
import pytest

from ncaa_eval.evaluation.perturbation import (
    FIRST_ROUND_SEED_PRIORS,
    _lookup_seed_prior,
    build_seed_prior_matrix,
    perturb_probability_matrix,
    power_transform,
    slider_to_temperature,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prob_matrix(probs: list[float]) -> np.ndarray:
    """Build a small probability matrix from upper-triangle probabilities.

    For a 2×2 matrix, pass one probability [p]; for 4×4 pass 6 probs, etc.
    """
    # Determine n from len(probs) = n*(n-1)/2
    n = 2
    while n * (n - 1) // 2 < len(probs):
        n += 1
    assert n * (n - 1) // 2 == len(probs)

    P = np.zeros((n, n), dtype=np.float64)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            P[i, j] = probs[idx]
            P[j, i] = 1.0 - probs[idx]
            idx += 1
    return P


# ---------------------------------------------------------------------------
# slider_to_temperature
# ---------------------------------------------------------------------------


class TestSliderToTemperature:
    def test_neutral_returns_one(self) -> None:
        assert slider_to_temperature(0) == 1.0

    def test_negative_five(self) -> None:
        t = slider_to_temperature(-5)
        assert pytest.approx(t, abs=0.001) == 2 ** (-5 / 3)

    def test_positive_five(self) -> None:
        t = slider_to_temperature(5)
        assert pytest.approx(t, abs=0.001) == 2 ** (5 / 3)

    def test_negative_three_gives_half(self) -> None:
        assert pytest.approx(slider_to_temperature(-3), abs=1e-10) == 0.5

    def test_positive_three_gives_two(self) -> None:
        assert pytest.approx(slider_to_temperature(3), abs=1e-10) == 2.0

    def test_all_integer_values_positive(self) -> None:
        for v in range(-5, 6):
            t = slider_to_temperature(v)
            assert t > 0

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="slider_value must be in"):
            slider_to_temperature(-6)
        with pytest.raises(ValueError, match="slider_value must be in"):
            slider_to_temperature(6)


# ---------------------------------------------------------------------------
# power_transform
# ---------------------------------------------------------------------------


class TestPowerTransform:
    def test_identity_at_t1(self) -> None:
        P = _make_prob_matrix([0.7, 0.9, 0.55, 0.3, 0.8, 0.6])
        result = power_transform(P, temperature=1.0)
        np.testing.assert_array_almost_equal(result, P)

    def test_t_greater_than_1_compresses(self) -> None:
        P = _make_prob_matrix([0.9])
        result = power_transform(P, temperature=2.0)
        # Should compress 0.9 toward 0.5
        assert 0.5 < float(result[0, 1]) < 0.9

    def test_t_less_than_1_sharpens(self) -> None:
        P = _make_prob_matrix([0.7])
        result = power_transform(P, temperature=0.5)
        # Should sharpen 0.7 away from 0.5 → closer to 1.0
        assert 0.7 < float(result[0, 1]) < 1.0

    def test_preserves_complementarity(self) -> None:
        P = _make_prob_matrix([0.65, 0.99, 0.52, 0.4, 0.8, 0.3])
        for t in [0.5, 1.0, 1.5, 2.0, 3.0]:
            result = power_transform(P, temperature=t)
            n = P.shape[0]
            for i in range(n):
                for j in range(i + 1, n):
                    assert pytest.approx(float(result[i, j] + result[j, i]), abs=1e-12) == 1.0

    def test_preserves_diagonal_zeros(self) -> None:
        P = _make_prob_matrix([0.7, 0.8, 0.6])
        result = power_transform(P, temperature=2.0)
        np.testing.assert_array_equal(np.diag(result), np.zeros(3))

    def test_p_zero_stays_zero(self) -> None:
        P = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        result = power_transform(P, temperature=2.0)
        assert float(result[0, 1]) == 0.0
        assert float(result[1, 0]) == 1.0

    def test_p_one_stays_one(self) -> None:
        P = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
        result = power_transform(P, temperature=2.0)
        assert float(result[0, 1]) == 1.0
        assert float(result[1, 0]) == 0.0

    def test_p_half_fixed_point(self) -> None:
        P = _make_prob_matrix([0.5])
        for t in [0.3, 0.5, 1.0, 2.0, 3.0]:
            result = power_transform(P, temperature=t)
            assert pytest.approx(float(result[0, 1]), abs=1e-12) == 0.5

    def test_negative_temperature_raises(self) -> None:
        P = _make_prob_matrix([0.7])
        with pytest.raises(ValueError, match="temperature must be positive"):
            power_transform(P, temperature=0.0)
        with pytest.raises(ValueError, match="temperature must be positive"):
            power_transform(P, temperature=-1.0)


# ---------------------------------------------------------------------------
# build_seed_prior_matrix
# ---------------------------------------------------------------------------


class TestBuildSeedPriorMatrix:
    def test_same_seed_returns_half(self) -> None:
        seed_map = {100: 8, 200: 8}
        team_ids = [100, 200]
        S = build_seed_prior_matrix(seed_map, team_ids)
        assert float(S[0, 1]) == 0.5
        assert float(S[1, 0]) == 0.5

    def test_1v16_lookup(self) -> None:
        seed_map = {100: 1, 200: 16}
        team_ids = [100, 200]
        S = build_seed_prior_matrix(seed_map, team_ids)
        # seed_diff=15, team 100 is higher seed (seed=1)
        assert pytest.approx(float(S[0, 1]), abs=1e-6) == 0.993
        assert pytest.approx(float(S[1, 0]), abs=1e-6) == 0.007

    def test_8v9_lookup(self) -> None:
        seed_map = {100: 8, 200: 9}
        team_ids = [100, 200]
        S = build_seed_prior_matrix(seed_map, team_ids)
        assert pytest.approx(float(S[0, 1]), abs=1e-6) == 0.521

    def test_even_seed_diff_interpolated(self) -> None:
        # seed_diff=2: between d_low=1 (0.521) and d_high=3 (0.604)
        # interpolation: 0.521 + 1/2 * (0.604 - 0.521) = 0.5625
        seed_map = {100: 6, 200: 8}
        team_ids = [100, 200]
        S = build_seed_prior_matrix(seed_map, team_ids)
        # team 100 is higher seed (seed=6 < seed=8)
        assert pytest.approx(float(S[0, 1]), abs=1e-6) == 0.5625

    def test_seed_diff_above_15_clamps(self) -> None:
        # This shouldn't happen in real brackets, but test the guard.
        seed_map = {100: 1, 200: 20}
        team_ids = [100, 200]
        S = build_seed_prior_matrix(seed_map, team_ids)
        assert pytest.approx(float(S[0, 1]), abs=1e-6) == 0.993

    def test_complementarity(self) -> None:
        seed_map = {100: 1, 200: 16, 300: 5, 400: 12}
        team_ids = [100, 200, 300, 400]
        S = build_seed_prior_matrix(seed_map, team_ids)
        n = S.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                assert pytest.approx(float(S[i, j] + S[j, i]), abs=1e-12) == 1.0

    def test_diagonal_zeros(self) -> None:
        seed_map = {100: 1, 200: 16, 300: 8}
        team_ids = [100, 200, 300]
        S = build_seed_prior_matrix(seed_map, team_ids)
        np.testing.assert_array_equal(np.diag(S), np.zeros(3))

    def test_underdog_gets_lower_probability(self) -> None:
        # 11-seed (lower seed = higher seed number) should have < 0.5 against 6-seed
        seed_map = {100: 11, 200: 6}
        team_ids = [100, 200]
        S = build_seed_prior_matrix(seed_map, team_ids)
        # team 100 is seed 11 (underdog), team 200 is seed 6 (higher seed)
        assert float(S[0, 1]) < 0.5
        assert float(S[1, 0]) > 0.5


# ---------------------------------------------------------------------------
# perturb_probability_matrix
# ---------------------------------------------------------------------------


class TestPerturbProbabilityMatrix:
    def test_identity_at_neutral(self) -> None:
        P = _make_prob_matrix([0.7, 0.9, 0.55, 0.3, 0.8, 0.6])
        seed_map = {1: 1, 2: 16, 3: 8, 4: 9}
        team_ids = [1, 2, 3, 4]
        result = perturb_probability_matrix(P, seed_map, team_ids, temperature=1.0, seed_weight=0.0)
        np.testing.assert_array_almost_equal(result, P)

    def test_complementarity_preserved(self) -> None:
        P = _make_prob_matrix([0.65, 0.99, 0.52, 0.4, 0.8, 0.3])
        seed_map = {1: 1, 2: 16, 3: 5, 4: 12}
        team_ids = [1, 2, 3, 4]
        for t, w in [(0.5, 0.0), (2.0, 0.0), (1.0, 0.5), (1.5, 0.3), (3.0, 1.0)]:
            result = perturb_probability_matrix(P, seed_map, team_ids, temperature=t, seed_weight=w)
            n = result.shape[0]
            for i in range(n):
                for j in range(i + 1, n):
                    assert pytest.approx(float(result[i, j] + result[j, i]), abs=1e-10) == 1.0

    def test_diagonal_remains_zero(self) -> None:
        P = _make_prob_matrix([0.7, 0.8, 0.6])
        seed_map = {1: 1, 2: 16, 3: 8}
        team_ids = [1, 2, 3]
        result = perturb_probability_matrix(P, seed_map, team_ids, temperature=2.0, seed_weight=0.3)
        np.testing.assert_array_equal(np.diag(result), np.zeros(3))

    def test_invalid_temperature_raises(self) -> None:
        P = _make_prob_matrix([0.7])
        with pytest.raises(ValueError, match="temperature must be positive"):
            perturb_probability_matrix(P, {}, [], temperature=0.0, seed_weight=0.0)

    def test_invalid_seed_weight_raises(self) -> None:
        P = _make_prob_matrix([0.7])
        with pytest.raises(ValueError, match="seed_weight must be in"):
            perturb_probability_matrix(P, {}, [], temperature=1.0, seed_weight=-0.1)
        with pytest.raises(ValueError, match="seed_weight must be in"):
            perturb_probability_matrix(P, {}, [], temperature=1.0, seed_weight=1.1)

    def test_pure_seed_weight_ignores_model(self) -> None:
        # With w=1.0, result should equal the seed prior matrix regardless of P
        P = _make_prob_matrix([0.99])
        seed_map = {100: 1, 200: 16}
        team_ids = [100, 200]
        result = perturb_probability_matrix(P, seed_map, team_ids, temperature=1.0, seed_weight=1.0)
        # Should be pure seed prior: 0.993 for 1v16
        assert pytest.approx(float(result[0, 1]), abs=1e-6) == 0.993

    def test_temperature_only(self) -> None:
        P = _make_prob_matrix([0.65])
        seed_map = {100: 5, 200: 12}
        team_ids = [100, 200]
        result = perturb_probability_matrix(P, seed_map, team_ids, temperature=2.0, seed_weight=0.0)
        # Should compress 0.65 toward 0.5 — same as power_transform alone
        expected = power_transform(P, temperature=2.0)
        np.testing.assert_array_almost_equal(result, expected)


# ---------------------------------------------------------------------------
# Parametrized worked examples from spike research (Section 4)
# ---------------------------------------------------------------------------

# Temperature-only examples (w=0): Section 4.2
# Matchup: (p_model, seed_a, seed_b)
_TEMP_EXAMPLES: list[tuple[float, dict[str, float]]] = [
    # (p_model, {T_value: expected_p_prime})
    (0.990, {"0.5": 1.000, "1.0": 0.990, "1.5": 0.955, "2.0": 0.909, "3.0": 0.822}),
    (0.650, {"0.5": 0.775, "1.0": 0.650, "1.5": 0.602, "2.0": 0.577, "3.0": 0.551}),
    (0.520, {"0.5": 0.540, "1.0": 0.520, "1.5": 0.513, "2.0": 0.510, "3.0": 0.507}),
    (0.400, {"0.5": 0.308, "1.0": 0.400, "1.5": 0.433, "2.0": 0.449, "3.0": 0.466}),
]


@pytest.mark.parametrize(
    ("p_model", "expected_by_t"),
    _TEMP_EXAMPLES,
    ids=["1v16_p=0.99", "5v12_p=0.65", "8v9_p=0.52", "6v11_p=0.40"],
)
def test_temperature_worked_examples(p_model: float, expected_by_t: dict[str, float]) -> None:
    """Verify computed values match spike research Section 4.2."""
    P = _make_prob_matrix([p_model])
    for t_str, expected in expected_by_t.items():
        t = float(t_str)
        result = power_transform(P, temperature=t)
        assert pytest.approx(float(result[0, 1]), abs=0.002) == expected, (
            f"p={p_model}, T={t}: got {float(result[0, 1]):.4f}, expected {expected:.4f}"
        )


# Seed-weight-only examples (T=1): Section 4.3
_SEED_WEIGHT_EXAMPLES: list[tuple[float, float, int, int, dict[str, float]]] = [
    # (p_model, p_seed, seed_a, seed_b, {w_str: expected_p_prime})
    (0.990, 0.993, 1, 16, {"0.0": 0.990, "0.25": 0.991, "0.5": 0.992, "0.75": 0.992, "1.0": 0.993}),
    (0.650, 0.646, 5, 12, {"0.0": 0.650, "0.25": 0.649, "0.5": 0.648, "0.75": 0.647, "1.0": 0.646}),
    (0.520, 0.521, 8, 9, {"0.0": 0.520, "0.25": 0.520, "0.5": 0.521, "0.75": 0.521, "1.0": 0.521}),
    (0.400, 0.375, 11, 6, {"0.0": 0.400, "0.25": 0.394, "0.5": 0.388, "0.75": 0.381, "1.0": 0.375}),
]


@pytest.mark.parametrize(
    ("p_model", "p_seed", "seed_a", "seed_b", "expected_by_w"),
    _SEED_WEIGHT_EXAMPLES,
    ids=["1v16_seed", "5v12_seed", "8v9_seed", "6v11_seed"],
)
def test_seed_weight_worked_examples(
    p_model: float,
    p_seed: float,  # noqa: ARG001 — kept for documentation
    seed_a: int,
    seed_b: int,
    expected_by_w: dict[str, float],
) -> None:
    """Verify computed values match spike research Section 4.3."""
    P = _make_prob_matrix([p_model])
    seed_map = {100: seed_a, 200: seed_b}
    team_ids = [100, 200]
    for w_str, expected in expected_by_w.items():
        w = float(w_str)
        result = perturb_probability_matrix(P, seed_map, team_ids, temperature=1.0, seed_weight=w)
        assert pytest.approx(float(result[0, 1]), abs=0.002) == expected, (
            f"p={p_model}, w={w}: got {float(result[0, 1]):.4f}, expected {expected:.4f}"
        )


# Combined examples (T=1.5, w=0.3): Section 4.4
_COMBINED_EXAMPLES: list[tuple[float, int, int, float]] = [
    # (p_model, seed_a, seed_b, expected_combined)
    (0.990, 1, 16, 0.967),
    (0.650, 5, 12, 0.615),
    (0.520, 8, 9, 0.516),
    (0.400, 11, 6, 0.415),
]


@pytest.mark.parametrize(
    ("p_model", "seed_a", "seed_b", "expected"),
    _COMBINED_EXAMPLES,
    ids=["1v16_combined", "5v12_combined", "8v9_combined", "6v11_combined"],
)
def test_combined_worked_examples(
    p_model: float,
    seed_a: int,
    seed_b: int,
    expected: float,
) -> None:
    """Verify combined T=1.5 + w=0.3 from spike research Section 4.4."""
    P = _make_prob_matrix([p_model])
    seed_map = {100: seed_a, 200: seed_b}
    team_ids = [100, 200]
    result = perturb_probability_matrix(P, seed_map, team_ids, temperature=1.5, seed_weight=0.3)
    assert pytest.approx(float(result[0, 1]), abs=0.002) == expected, (
        f"p={p_model}, T=1.5, w=0.3: got {float(result[0, 1]):.4f}, expected {expected:.4f}"
    )


# ---------------------------------------------------------------------------
# _lookup_seed_prior
# ---------------------------------------------------------------------------


class TestLookupSeedPrior:
    def test_all_tabulated_values(self) -> None:
        for diff, expected in FIRST_ROUND_SEED_PRIORS.items():
            assert _lookup_seed_prior(diff) == expected

    def test_zero_returns_half(self) -> None:
        assert _lookup_seed_prior(0) == 0.5

    @pytest.mark.parametrize(
        ("seed_diff", "d_low", "p_low", "d_high", "p_high"),
        [
            (2, 1, 0.521, 3, 0.604),
            (4, 3, 0.604, 5, 0.625),
            (6, 5, 0.625, 7, 0.646),
            (8, 7, 0.646, 9, 0.792),
            (10, 9, 0.792, 11, 0.854),
            (12, 11, 0.854, 13, 0.938),
            (14, 13, 0.938, 15, 0.993),
        ],
        ids=["diff=2", "diff=4", "diff=6", "diff=8", "diff=10", "diff=12", "diff=14"],
    )
    def test_interpolation_all_even_diffs(
        self,
        seed_diff: int,
        d_low: int,
        p_low: float,
        d_high: int,
        p_high: float,
    ) -> None:
        frac = (seed_diff - d_low) / (d_high - d_low)
        expected = p_low + frac * (p_high - p_low)
        assert pytest.approx(_lookup_seed_prior(seed_diff), abs=1e-6) == expected

    def test_above_15_clamps(self) -> None:
        assert _lookup_seed_prior(16) == 0.993
        assert _lookup_seed_prior(20) == 0.993


# ---------------------------------------------------------------------------
# Slider → temperature → perturb_probability_matrix integration
# ---------------------------------------------------------------------------


class TestSliderToPerturbIntegration:
    """End-to-end: slider values wire through temperature into a valid perturbed matrix."""

    _SEED_MAP = {100: 1, 200: 16, 300: 5, 400: 12}
    _TEAM_IDS = [100, 200, 300, 400]

    def _assert_valid_perturbed_matrix(self, result: np.ndarray, P: np.ndarray, msg: str) -> None:
        n = result.shape[0]
        assert not np.any(np.isnan(result)), f"{msg}: NaN values found"
        assert not np.any(np.isinf(result)), f"{msg}: Inf values found"
        assert np.all(result >= 0.0), f"{msg}: values below 0"
        assert np.all(result <= 1.0), f"{msg}: values above 1"
        np.testing.assert_array_equal(np.diag(result), np.zeros(n), err_msg=f"{msg}: diagonal non-zero")
        for i in range(n):
            for j in range(i + 1, n):
                assert pytest.approx(float(result[i, j] + result[j, i]), abs=1e-10) == 1.0, (
                    f"{msg}: complementarity violated at ({i},{j})"
                )

    @pytest.mark.parametrize("slider_val", [-5, -3, 0, 3, 5])
    def test_temperature_boundary_values_produce_valid_matrix(self, slider_val: int) -> None:
        """Boundary slider values must produce valid, complementary, bounded matrices."""
        P = _make_prob_matrix([0.99, 0.65, 0.52, 0.40, 0.75, 0.60])
        temperature = slider_to_temperature(slider_val)
        result = perturb_probability_matrix(
            P, self._SEED_MAP, self._TEAM_IDS, temperature=temperature, seed_weight=0.0
        )
        self._assert_valid_perturbed_matrix(result, P, f"slider={slider_val}, T={temperature:.4f}")

    @pytest.mark.parametrize(("slider_val", "seed_weight"), [(-5, 1.0), (5, 1.0), (-5, 0.5), (5, 0.5)])
    def test_combined_boundary_slider_values(self, slider_val: int, seed_weight: float) -> None:
        """Extreme slider combinations must not produce NaN/Inf or violate constraints."""
        P = _make_prob_matrix([0.99, 0.65, 0.52, 0.40, 0.75, 0.60])
        temperature = slider_to_temperature(slider_val)
        result = perturb_probability_matrix(
            P, self._SEED_MAP, self._TEAM_IDS, temperature=temperature, seed_weight=seed_weight
        )
        self._assert_valid_perturbed_matrix(result, P, f"slider={slider_val}, w={seed_weight}")
