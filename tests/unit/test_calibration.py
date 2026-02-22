"""Unit tests for ncaa_eval.transform.calibration."""

from __future__ import annotations

import numpy as np
import pytest

from ncaa_eval.transform.calibration import (
    IsotonicCalibrator,
    SigmoidCalibrator,
)

# ── IsotonicCalibrator tests ────────────────────────────────────────────────


class TestIsotonicCalibrator:
    """Tests for IsotonicCalibrator wrapping sklearn.isotonic.IsotonicRegression."""

    def test_fit_transform_produces_probabilities(self) -> None:
        """Output probabilities must be in [0, 1]."""
        cal = IsotonicCalibrator()
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.9, 0.6, 0.85, 0.15, 0.25])
        cal.fit(y_true, y_prob)

        test_probs = np.array([0.0, 0.5, 1.0, 0.3, 0.7])
        result = cal.transform(test_probs)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_monotonicity(self) -> None:
        """Isotonic calibration must preserve monotonicity."""
        cal = IsotonicCalibrator()
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.15, 0.25, 0.6, 0.65, 0.7, 0.8, 0.9])
        cal.fit(y_true, y_prob)

        sorted_probs = np.linspace(0.0, 1.0, 20)
        result = cal.transform(sorted_probs)
        # Each output should be >= the previous one (monotonic non-decreasing)
        diffs = np.diff(result)
        assert np.all(diffs >= -1e-10), f"Non-monotonic: {result}"

    def test_leakage_prevention_disjoint_data(self) -> None:
        """Fit and transform on disjoint data (no leakage)."""
        cal = IsotonicCalibrator()
        rng = np.random.default_rng(42)

        # Training fold: 100 samples
        y_true_train = rng.integers(0, 2, size=100).astype(float)
        y_prob_train = rng.uniform(0, 1, size=100)
        cal.fit(y_true_train, y_prob_train)

        # Test fold: 50 completely new samples
        y_prob_test = rng.uniform(0, 1, size=50)
        result = cal.transform(y_prob_test)

        # Output should still be valid probabilities
        assert len(result) == 50
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_not_fitted_raises(self) -> None:
        """Transform before fit should raise."""
        cal = IsotonicCalibrator()
        with pytest.raises(RuntimeError, match="not fitted"):
            cal.transform(np.array([0.5]))

    def test_empty_input_returns_empty(self) -> None:
        """Transform on empty input returns empty array."""
        cal = IsotonicCalibrator()
        cal.fit(np.array([0, 1]), np.array([0.3, 0.7]))
        result = cal.transform(np.array([]))
        assert len(result) == 0

    def test_perfect_calibration_passthrough(self) -> None:
        """Well-calibrated predictions should pass through ~unchanged."""
        cal = IsotonicCalibrator()
        y_true = np.array([0.0] * 50 + [1.0] * 50)
        y_prob = np.concatenate([np.linspace(0, 0.4, 50), np.linspace(0.6, 1.0, 50)])
        cal.fit(y_true, y_prob)

        test_probs = np.array([0.1, 0.9])
        result = cal.transform(test_probs)
        # Should be close to input (well-calibrated → near identity)
        assert result[0] < 0.5  # low prediction stays low
        assert result[1] > 0.5  # high prediction stays high


# ── SigmoidCalibrator tests ─────────────────────────────────────────────────


class TestSigmoidCalibrator:
    """Tests for SigmoidCalibrator (Platt scaling) for small-sample fallback."""

    def test_fit_transform_produces_probabilities(self) -> None:
        """Output probabilities must be in [0, 1]."""
        cal = SigmoidCalibrator()
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.9, 0.6, 0.85, 0.15, 0.25])
        cal.fit(y_true, y_prob)

        test_probs = np.array([0.0, 0.5, 1.0, 0.3, 0.7])
        result = cal.transform(test_probs)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_not_fitted_raises(self) -> None:
        """Transform before fit should raise."""
        cal = SigmoidCalibrator()
        with pytest.raises(RuntimeError, match="not fitted"):
            cal.transform(np.array([0.5]))

    def test_empty_input_returns_empty(self) -> None:
        """Transform on empty input returns empty array."""
        cal = SigmoidCalibrator()
        cal.fit(np.array([0, 1]), np.array([0.3, 0.7]))
        result = cal.transform(np.array([]))
        assert len(result) == 0

    def test_leakage_prevention_disjoint_data(self) -> None:
        """Fit and transform on disjoint data (no leakage)."""
        cal = SigmoidCalibrator()
        rng = np.random.default_rng(42)

        y_true_train = rng.integers(0, 2, size=100).astype(float)
        y_prob_train = rng.uniform(0, 1, size=100)
        cal.fit(y_true_train, y_prob_train)

        y_prob_test = rng.uniform(0, 1, size=50)
        result = cal.transform(y_prob_test)
        assert len(result) == 50
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_monotonicity(self) -> None:
        """Sigmoid (logistic) calibration must preserve monotone ordering."""
        cal = SigmoidCalibrator()
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.15, 0.25, 0.6, 0.65, 0.7, 0.8, 0.9])
        cal.fit(y_true, y_prob)

        sorted_probs = np.linspace(0.05, 0.95, 20)
        result = cal.transform(sorted_probs)
        diffs = np.diff(result)
        assert np.all(diffs >= -1e-10), f"Non-monotonic sigmoid output: {result}"

    def test_boundary_inputs_clipped(self) -> None:
        """Exact 0.0 and 1.0 inputs are clipped to avoid log(0) errors."""
        cal = SigmoidCalibrator()
        cal.fit(np.array([0, 1]), np.array([0.3, 0.7]))

        # Should not raise; clipping prevents log(0)
        result = cal.transform(np.array([0.0, 0.5, 1.0]))
        assert len(result) == 3
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
