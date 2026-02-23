"""Evaluation metrics for NCAA basketball model predictions.

Provides metric functions for evaluating probabilistic predictions:

* :func:`log_loss` — Log Loss via ``sklearn.metrics.log_loss``
* :func:`brier_score` — Brier Score via ``sklearn.metrics.brier_score_loss``
* :func:`roc_auc` — ROC-AUC via ``sklearn.metrics.roc_auc_score``
* :func:`expected_calibration_error` — ECE via vectorized numpy binning
* :func:`reliability_diagram_data` — Reliability diagram bin data via
  ``sklearn.calibration.calibration_curve``

All functions accept ``npt.NDArray[np.float64]`` inputs and return ``float``
scalars or structured data (:class:`ReliabilityData`).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import numpy.typing as npt


@dataclasses.dataclass(frozen=True)
class ReliabilityData:
    """Structured return type for reliability diagram data.

    Attributes
    ----------
    fraction_of_positives
        Observed fraction of positives per bin (from calibration_curve).
    mean_predicted_value
        Mean predicted probability per bin (from calibration_curve).
    bin_counts
        Number of samples in each bin.
    n_bins
        Requested number of bins.
    """

    fraction_of_positives: npt.NDArray[np.float64]
    mean_predicted_value: npt.NDArray[np.float64]
    bin_counts: npt.NDArray[np.int64]
    n_bins: int


def _validate_inputs(
    y_true: npt.NDArray[np.float64],
    y_prob: npt.NDArray[np.float64],
) -> None:
    """Validate metric inputs: non-empty, matching lengths, probs in [0, 1]."""
    if len(y_true) == 0 or len(y_prob) == 0:
        msg = "y_true and y_prob must be non-empty arrays."
        raise ValueError(msg)
    if len(y_true) != len(y_prob):
        msg = f"y_true and y_prob must have the same length, " f"got {len(y_true)} and {len(y_prob)}."
        raise ValueError(msg)
    if np.any(y_prob < 0.0) or np.any(y_prob > 1.0):
        msg = "y_prob values must be in [0, 1]."
        raise ValueError(msg)


def log_loss(
    y_true: npt.NDArray[np.float64],
    y_prob: npt.NDArray[np.float64],
) -> float:
    """Compute Log Loss (cross-entropy loss) for binary predictions.

    Parameters
    ----------
    y_true
        Binary labels (0 or 1).
    y_prob
        Predicted probabilities for the positive class.

    Returns
    -------
    float
        Log Loss value.

    Raises
    ------
    ValueError
        If inputs are empty, mismatched, or probabilities are outside [0, 1].
    """
    from sklearn.metrics import log_loss as sklearn_log_loss  # type: ignore[import-untyped]

    _validate_inputs(y_true, y_prob)
    result: float = float(sklearn_log_loss(y_true, y_prob, labels=[0, 1]))
    return result


def brier_score(
    y_true: npt.NDArray[np.float64],
    y_prob: npt.NDArray[np.float64],
) -> float:
    """Compute Brier Score for binary predictions.

    Parameters
    ----------
    y_true
        Binary labels (0 or 1).
    y_prob
        Predicted probabilities for the positive class.

    Returns
    -------
    float
        Brier Score value (lower is better).

    Raises
    ------
    ValueError
        If inputs are empty, mismatched, or probabilities are outside [0, 1].
    """
    from sklearn.metrics import brier_score_loss

    _validate_inputs(y_true, y_prob)
    result: float = float(brier_score_loss(y_true, y_prob))
    return result


def roc_auc(
    y_true: npt.NDArray[np.float64],
    y_prob: npt.NDArray[np.float64],
) -> float:
    """Compute ROC-AUC for binary predictions.

    Parameters
    ----------
    y_true
        Binary labels (0 or 1).
    y_prob
        Predicted probabilities for the positive class.

    Returns
    -------
    float
        ROC-AUC value.

    Raises
    ------
    ValueError
        If inputs are empty, mismatched, probabilities are outside [0, 1],
        or ``y_true`` contains only one class (AUC is undefined).
    """
    from sklearn.metrics import roc_auc_score

    _validate_inputs(y_true, y_prob)
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        msg = "roc_auc requires both positive and negative samples in y_true."
        raise ValueError(msg)
    result: float = float(roc_auc_score(y_true, y_prob))
    return result


def expected_calibration_error(
    y_true: npt.NDArray[np.float64],
    y_prob: npt.NDArray[np.float64],
    *,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) using vectorized numpy.

    ECE measures how well predicted probabilities match observed frequencies.
    Predictions are binned into ``n_bins`` equal-width bins on [0, 1], and
    ECE is the weighted average of per-bin |accuracy - confidence| gaps.

    Parameters
    ----------
    y_true
        Binary labels (0 or 1).
    y_prob
        Predicted probabilities for the positive class.
    n_bins
        Number of equal-width bins (default 10).

    Returns
    -------
    float
        ECE value in [0, 1] (lower is better).

    Raises
    ------
    ValueError
        If inputs are empty, mismatched, or probabilities are outside [0, 1].
    """
    _validate_inputs(y_true, y_prob)

    # Bin edges: [0, 1/n_bins, 2/n_bins, ..., 1]
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize assigns bin index starting at 1; clip to [1, n_bins]
    bin_indices = np.clip(np.digitize(y_prob, bin_edges[1:-1]), 0, n_bins - 1)

    # Vectorized per-bin statistics using np.bincount
    bin_counts = np.bincount(bin_indices, minlength=n_bins).astype(np.float64)
    bin_sums_true = np.bincount(bin_indices, weights=y_true, minlength=n_bins)
    bin_sums_prob = np.bincount(bin_indices, weights=y_prob, minlength=n_bins)

    # Mask for non-empty bins (avoid division by zero)
    non_empty = bin_counts > 0

    acc = np.zeros(n_bins, dtype=np.float64)
    conf = np.zeros(n_bins, dtype=np.float64)
    acc[non_empty] = bin_sums_true[non_empty] / bin_counts[non_empty]
    conf[non_empty] = bin_sums_prob[non_empty] / bin_counts[non_empty]

    weights = bin_counts / float(len(y_true))
    ece: float = float(np.sum(weights * np.abs(acc - conf)))
    return ece


def reliability_diagram_data(
    y_true: npt.NDArray[np.float64],
    y_prob: npt.NDArray[np.float64],
    *,
    n_bins: int = 10,
) -> ReliabilityData:
    """Generate reliability diagram data for calibration visualization.

    Uses ``sklearn.calibration.calibration_curve`` for bin statistics and
    augments with per-bin sample counts.

    Parameters
    ----------
    y_true
        Binary labels (0 or 1).
    y_prob
        Predicted probabilities for the positive class.
    n_bins
        Number of bins (default 10).

    Returns
    -------
    ReliabilityData
        Structured data containing fraction of positives, mean predicted
        values, bin counts, and requested number of bins.

    Raises
    ------
    ValueError
        If inputs are empty, mismatched, or probabilities are outside [0, 1].
    """
    from sklearn.calibration import calibration_curve  # type: ignore[import-untyped]

    _validate_inputs(y_true, y_prob)

    fraction_of_positives: npt.NDArray[np.float64]
    mean_predicted_value: npt.NDArray[np.float64]
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    # Compute bin counts using same binning as calibration_curve (uniform)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(y_prob, bin_edges[1:-1]), 0, n_bins - 1)
    all_bin_counts = np.bincount(bin_indices, minlength=n_bins)

    # calibration_curve only returns non-empty bins — filter to match
    non_empty_mask = all_bin_counts > 0
    bin_counts: npt.NDArray[np.int64] = all_bin_counts[non_empty_mask].astype(np.int64)

    return ReliabilityData(
        fraction_of_positives=fraction_of_positives,
        mean_predicted_value=mean_predicted_value,
        bin_counts=bin_counts,
        n_bins=n_bins,
    )
