"""Unit tests for ncaa_eval.evaluation.metrics."""

from __future__ import annotations

import numpy as np
import pytest

from ncaa_eval.evaluation.metrics import (
    ReliabilityData,
    brier_score,
    expected_calibration_error,
    log_loss,
    reliability_diagram_data,
    roc_auc,
)

# ── LogLoss tests ─────────────────────────────────────────────────────────────


class TestLogLoss:
    """Tests for log_loss wrapping sklearn.metrics.log_loss."""

    def test_known_values(self) -> None:
        """Verify against hand-computed / sklearn reference values."""
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2])
        result = log_loss(y_true, y_prob)
        # sklearn.metrics.log_loss([1,0,1,0], [0.9,0.1,0.8,0.2])
        expected = -np.mean([np.log(0.9), np.log(1 - 0.1), np.log(0.8), np.log(1 - 0.2)])
        assert result == pytest.approx(expected, rel=1e-10)

    def test_perfect_predictions(self) -> None:
        """Perfect predictions should yield log loss near 0."""
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_prob = np.array([1.0, 0.0, 1.0, 0.0])
        result = log_loss(y_true, y_prob)
        # sklearn clips to avoid log(0), so result is very small but not exactly 0
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_single_prediction(self) -> None:
        """Single prediction should compute without error."""
        y_true = np.array([1.0])
        y_prob = np.array([0.7])
        result = log_loss(y_true, y_prob)
        expected = -np.log(0.7)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_all_same_class(self) -> None:
        """All-same-class: log loss should still compute (not undefined)."""
        y_true = np.array([1.0, 1.0, 1.0])
        y_prob = np.array([0.9, 0.8, 0.7])
        result = log_loss(y_true, y_prob)
        expected = -np.mean([np.log(0.9), np.log(0.8), np.log(0.7)])
        assert result == pytest.approx(expected, rel=1e-10)


# ── BrierScore tests ──────────────────────────────────────────────────────────


class TestBrierScore:
    """Tests for brier_score wrapping sklearn.metrics.brier_score_loss."""

    def test_known_values(self) -> None:
        """Verify against hand-computed values."""
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2])
        result = brier_score(y_true, y_prob)
        # Brier = mean((y_prob - y_true)^2)
        expected = np.mean((y_prob - y_true) ** 2)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_perfect_predictions(self) -> None:
        """Perfect predictions should yield Brier score of 0."""
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_prob = np.array([1.0, 0.0, 1.0, 0.0])
        result = brier_score(y_true, y_prob)
        assert result == pytest.approx(0.0, abs=1e-15)

    def test_worst_predictions(self) -> None:
        """Worst predictions should yield Brier score of 1."""
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_prob = np.array([0.0, 1.0, 0.0, 1.0])
        result = brier_score(y_true, y_prob)
        assert result == pytest.approx(1.0, rel=1e-10)

    def test_single_prediction(self) -> None:
        """Single prediction should compute without error."""
        y_true = np.array([1.0])
        y_prob = np.array([0.6])
        result = brier_score(y_true, y_prob)
        expected = (0.6 - 1.0) ** 2
        assert result == pytest.approx(expected, rel=1e-10)

    def test_all_same_class(self) -> None:
        """All-same-class: Brier score should still compute."""
        y_true = np.array([0.0, 0.0, 0.0])
        y_prob = np.array([0.1, 0.2, 0.3])
        result = brier_score(y_true, y_prob)
        expected = np.mean(np.array([0.1, 0.2, 0.3]) ** 2)
        assert result == pytest.approx(expected, rel=1e-10)


# ── RocAuc tests ───────────────────────────────────────────────────────────────


class TestRocAuc:
    """Tests for roc_auc wrapping sklearn.metrics.roc_auc_score."""

    def test_known_values(self) -> None:
        """Verify against sklearn reference values."""
        y_true = np.array([0.0, 0.0, 1.0, 1.0])
        y_prob = np.array([0.1, 0.4, 0.35, 0.8])
        result = roc_auc(y_true, y_prob)
        # sklearn.metrics.roc_auc_score([0,0,1,1], [0.1,0.4,0.35,0.8]) = 0.75
        assert result == pytest.approx(0.75, rel=1e-10)

    def test_perfect_predictions(self) -> None:
        """Perfect separation should yield AUC of 1.0."""
        y_true = np.array([0.0, 0.0, 1.0, 1.0])
        y_prob = np.array([0.1, 0.2, 0.9, 0.95])
        result = roc_auc(y_true, y_prob)
        assert result == pytest.approx(1.0, rel=1e-10)

    def test_all_same_class_raises(self) -> None:
        """All-same-class should raise ValueError (AUC is undefined)."""
        y_true = np.array([1.0, 1.0, 1.0])
        y_prob = np.array([0.5, 0.6, 0.7])
        with pytest.raises(ValueError, match="both positive and negative"):
            roc_auc(y_true, y_prob)

    def test_all_same_class_zero_raises(self) -> None:
        """All-negative should also raise ValueError."""
        y_true = np.array([0.0, 0.0, 0.0])
        y_prob = np.array([0.5, 0.6, 0.7])
        with pytest.raises(ValueError, match="both positive and negative"):
            roc_auc(y_true, y_prob)


# ── ECE tests ──────────────────────────────────────────────────────────────────


class TestExpectedCalibrationError:
    """Tests for expected_calibration_error (custom numpy vectorized)."""

    def test_perfect_calibration(self) -> None:
        """Perfectly calibrated predictions should yield ECE near 0."""
        rng = np.random.default_rng(42)
        n = 10000
        y_prob = rng.uniform(0, 1, size=n)
        y_true = (rng.uniform(0, 1, size=n) < y_prob).astype(np.float64)
        result = expected_calibration_error(y_true, y_prob, n_bins=10)
        # With large N, ECE should be small for calibrated data
        assert result < 0.05

    def test_perfect_predictions(self) -> None:
        """Perfect 0/1 predictions should yield ECE of 0."""
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_prob = np.array([1.0, 0.0, 1.0, 0.0])
        result = expected_calibration_error(y_true, y_prob, n_bins=10)
        assert result == pytest.approx(0.0, abs=1e-15)

    def test_hand_computed_values(self) -> None:
        """Verify ECE with hand-computed example.

        2 bins on [0, 1]: [0, 0.5), [0.5, 1.0]
        Samples: y_true=[1, 0, 1, 1], y_prob=[0.2, 0.3, 0.7, 0.9]
        Bin 0 ([0, 0.5)): y_true=[1, 0], y_prob=[0.2, 0.3]
            acc=0.5, conf=0.25, count=2
        Bin 1 ([0.5, 1.0]): y_true=[1, 1], y_prob=[0.7, 0.9]
            acc=1.0, conf=0.8, count=2
        ECE = (2/4)*|0.5-0.25| + (2/4)*|1.0-0.8| = 0.5*0.25 + 0.5*0.2 = 0.225
        """
        y_true = np.array([1.0, 0.0, 1.0, 1.0])
        y_prob = np.array([0.2, 0.3, 0.7, 0.9])
        result = expected_calibration_error(y_true, y_prob, n_bins=2)
        assert result == pytest.approx(0.225, rel=1e-10)

    def test_single_prediction(self) -> None:
        """Single prediction should compute without error."""
        y_true = np.array([1.0])
        y_prob = np.array([0.7])
        result = expected_calibration_error(y_true, y_prob, n_bins=10)
        # Single sample: acc=1.0, conf=0.7, weight=1.0, ECE=0.3
        assert result == pytest.approx(0.3, rel=1e-10)

    def test_ece_bounded(self) -> None:
        """ECE should always be in [0, 1]."""
        rng = np.random.default_rng(123)
        y_true = rng.integers(0, 2, size=100).astype(np.float64)
        y_prob = rng.uniform(0, 1, size=100)
        result = expected_calibration_error(y_true, y_prob, n_bins=10)
        assert 0.0 <= result <= 1.0

    def test_all_same_class(self) -> None:
        """All-same-class: ECE should still compute (not raise)."""
        y_true = np.array([1.0, 1.0, 1.0, 1.0])
        y_prob = np.array([0.8, 0.85, 0.9, 0.95])
        result = expected_calibration_error(y_true, y_prob, n_bins=10)
        # acc=1.0, conf=mean(y_prob), ECE = |1 - mean(y_prob)|
        assert result == pytest.approx(1.0 - np.mean(y_prob), rel=1e-10)


# ── ReliabilityDiagramData tests ───────────────────────────────────────────────


class TestReliabilityDiagramData:
    """Tests for reliability_diagram_data wrapping sklearn.calibration.calibration_curve."""

    def test_output_structure(self) -> None:
        """Verify output is a ReliabilityData with correct types."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=100).astype(np.float64)
        y_prob = rng.uniform(0, 1, size=100)
        result = reliability_diagram_data(y_true, y_prob, n_bins=5)
        assert isinstance(result, ReliabilityData)
        assert result.n_bins == 5
        assert isinstance(result.fraction_of_positives, np.ndarray)
        assert isinstance(result.mean_predicted_value, np.ndarray)
        assert isinstance(result.bin_counts, np.ndarray)

    def test_bin_counts_sum(self) -> None:
        """Bin counts from non-empty bins should sum to total samples."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=200).astype(np.float64)
        y_prob = rng.uniform(0, 1, size=200)
        result = reliability_diagram_data(y_true, y_prob, n_bins=5)
        assert int(np.sum(result.bin_counts)) == 200

    def test_arrays_same_length(self) -> None:
        """fraction_of_positives, mean_predicted_value, and bin_counts
        should all have the same length (number of non-empty bins)."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=100).astype(np.float64)
        y_prob = rng.uniform(0, 1, size=100)
        result = reliability_diagram_data(y_true, y_prob, n_bins=5)
        assert len(result.fraction_of_positives) == len(result.mean_predicted_value)
        assert len(result.fraction_of_positives) == len(result.bin_counts)

    def test_fraction_of_positives_bounded(self) -> None:
        """Fraction of positives should be in [0, 1]."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=100).astype(np.float64)
        y_prob = rng.uniform(0, 1, size=100)
        result = reliability_diagram_data(y_true, y_prob, n_bins=5)
        assert np.all(result.fraction_of_positives >= 0.0)
        assert np.all(result.fraction_of_positives <= 1.0)

    def test_mean_predicted_value_bounded(self) -> None:
        """Mean predicted values should be in [0, 1]."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=100).astype(np.float64)
        y_prob = rng.uniform(0, 1, size=100)
        result = reliability_diagram_data(y_true, y_prob, n_bins=5)
        assert np.all(result.mean_predicted_value >= 0.0)
        assert np.all(result.mean_predicted_value <= 1.0)

    def test_single_prediction(self) -> None:
        """Single prediction should yield one bin."""
        y_true = np.array([1.0])
        y_prob = np.array([0.7])
        result = reliability_diagram_data(y_true, y_prob, n_bins=10)
        assert len(result.bin_counts) >= 1
        assert int(np.sum(result.bin_counts)) == 1

    def test_frozen_dataclass(self) -> None:
        """ReliabilityData should be frozen (immutable)."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=50).astype(np.float64)
        y_prob = rng.uniform(0, 1, size=50)
        result = reliability_diagram_data(y_true, y_prob, n_bins=5)
        with pytest.raises(AttributeError):
            result.n_bins = 20  # type: ignore[misc]

    def test_all_same_class(self) -> None:
        """All-same-class: reliability data should still compute."""
        y_true = np.array([1.0, 1.0, 1.0, 1.0])
        y_prob = np.array([0.8, 0.85, 0.9, 0.95])
        result = reliability_diagram_data(y_true, y_prob, n_bins=10)
        assert isinstance(result, ReliabilityData)
        assert int(np.sum(result.bin_counts)) == 4


# ── Edge case tests (cross-cutting) ───────────────────────────────────────────


class TestEdgeCases:
    """Cross-cutting edge case tests for all metric functions."""

    def test_empty_arrays_raise_log_loss(self) -> None:
        """Empty arrays should raise ValueError for log_loss."""
        with pytest.raises(ValueError, match="non-empty"):
            log_loss(np.array([]), np.array([]))

    def test_empty_arrays_raise_brier(self) -> None:
        """Empty arrays should raise ValueError for brier_score."""
        with pytest.raises(ValueError, match="non-empty"):
            brier_score(np.array([]), np.array([]))

    def test_empty_arrays_raise_roc_auc(self) -> None:
        """Empty arrays should raise ValueError for roc_auc."""
        with pytest.raises(ValueError, match="non-empty"):
            roc_auc(np.array([]), np.array([]))

    def test_empty_arrays_raise_ece(self) -> None:
        """Empty arrays should raise ValueError for ECE."""
        with pytest.raises(ValueError, match="non-empty"):
            expected_calibration_error(np.array([]), np.array([]))

    def test_empty_arrays_raise_reliability(self) -> None:
        """Empty arrays should raise ValueError for reliability_diagram_data."""
        with pytest.raises(ValueError, match="non-empty"):
            reliability_diagram_data(np.array([]), np.array([]))

    def test_mismatched_lengths_log_loss(self) -> None:
        """Mismatched array lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            log_loss(np.array([1.0, 0.0]), np.array([0.5]))

    def test_mismatched_lengths_brier(self) -> None:
        """Mismatched array lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            brier_score(np.array([1.0, 0.0]), np.array([0.5]))

    def test_mismatched_lengths_roc_auc(self) -> None:
        """Mismatched array lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            roc_auc(np.array([1.0, 0.0]), np.array([0.5]))

    def test_mismatched_lengths_ece(self) -> None:
        """Mismatched array lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            expected_calibration_error(np.array([1.0, 0.0]), np.array([0.5]))

    def test_mismatched_lengths_reliability(self) -> None:
        """Mismatched array lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            reliability_diagram_data(np.array([1.0, 0.0]), np.array([0.5]))

    def test_probs_below_zero_log_loss(self) -> None:
        """Probabilities below 0 should raise ValueError."""
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            log_loss(np.array([1.0]), np.array([-0.1]))

    def test_probs_above_one_log_loss(self) -> None:
        """Probabilities above 1 should raise ValueError."""
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            log_loss(np.array([1.0]), np.array([1.1]))

    def test_probs_below_zero_brier(self) -> None:
        """Probabilities below 0 should raise ValueError."""
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            brier_score(np.array([1.0]), np.array([-0.1]))

    def test_probs_above_one_brier(self) -> None:
        """Probabilities above 1 should raise ValueError."""
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            brier_score(np.array([1.0]), np.array([1.1]))

    def test_probs_below_zero_roc_auc(self) -> None:
        """Probabilities below 0 should raise ValueError."""
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            roc_auc(np.array([1.0, 0.0]), np.array([-0.1, 0.5]))

    def test_probs_above_one_roc_auc(self) -> None:
        """Probabilities above 1 should raise ValueError."""
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            roc_auc(np.array([1.0, 0.0]), np.array([0.5, 1.1]))

    def test_probs_below_zero_ece(self) -> None:
        """Probabilities below 0 should raise ValueError."""
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            expected_calibration_error(np.array([1.0]), np.array([-0.1]))

    def test_probs_above_one_ece(self) -> None:
        """Probabilities above 1 should raise ValueError."""
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            expected_calibration_error(np.array([1.0]), np.array([1.1]))

    def test_probs_below_zero_reliability(self) -> None:
        """Probabilities below 0 should raise ValueError."""
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            reliability_diagram_data(np.array([1.0]), np.array([-0.1]))

    def test_probs_above_one_reliability(self) -> None:
        """Probabilities above 1 should raise ValueError."""
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            reliability_diagram_data(np.array([1.0]), np.array([1.1]))

    def test_single_prediction_roc_auc_raises(self) -> None:
        """Single prediction should raise ValueError for roc_auc (needs 2+ classes)."""
        with pytest.raises(ValueError, match="both positive and negative"):
            roc_auc(np.array([1.0]), np.array([0.8]))
