"""Probability calibration for NCAA basketball model predictions.

Provides calibration wrappers for adjusting model-output probabilities
so they are well-calibrated (when the model says 70%, the event happens
~70% of the time).

* :class:`IsotonicCalibrator` — non-parametric monotonic calibration via
  ``sklearn.isotonic.IsotonicRegression``.  Best with >=1000 calibration
  samples.
* :class:`SigmoidCalibrator` — parametric Platt scaling via logistic
  regression on log-odds.  Better for small folds.

Design invariants:

- **In-fold only**: ``fit()`` on training fold predictions, ``transform()``
  on test fold predictions.  Never fit on the data being calibrated.
- ``goto_conversion`` was assessed and found **not applicable** — it removes
  bookmaker overround from betting odds, which is a fundamentally different
  problem from calibrating model-predicted probabilities.  See Story 4.7
  Dev Notes for the full assessment.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class IsotonicCalibrator:
    """Non-parametric monotonic probability calibration.

    Wraps ``sklearn.isotonic.IsotonicRegression`` with ``y_min=0.0``,
    ``y_max=1.0``, and ``out_of_bounds="clip"`` for probability bounds.

    Example::

        cal = IsotonicCalibrator()
        cal.fit(y_true_train, y_prob_train)
        calibrated = cal.transform(y_prob_test)
    """

    def __init__(self) -> None:
        self._fitted = False
        self._model: Any = None

    def fit(
        self,
        y_true: npt.NDArray[np.float64],
        y_prob: npt.NDArray[np.float64],
    ) -> None:
        """Fit the isotonic regression on training fold predictions.

        Parameters
        ----------
        y_true
            Binary labels (0 or 1) from the training fold.
        y_prob
            Model-predicted probabilities from the training fold.
        """
        from sklearn.isotonic import IsotonicRegression  # type: ignore[import-untyped]

        ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        ir.fit(y_prob, y_true)
        self._model = ir
        self._fitted = True

    def transform(
        self,
        y_prob: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Apply calibration to test fold predictions.

        Parameters
        ----------
        y_prob
            Model-predicted probabilities to calibrate.

        Returns
        -------
        npt.NDArray[np.float64]
            Calibrated probabilities in [0, 1].

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called.
        """
        if not self._fitted:
            msg = "IsotonicCalibrator is not fitted. Call fit() first."
            raise RuntimeError(msg)

        if len(y_prob) == 0:
            return np.array([], dtype=np.float64)

        result: npt.NDArray[np.float64] = self._model.transform(y_prob)
        return result


class SigmoidCalibrator:
    """Parametric Platt scaling for probability calibration.

    Uses logistic regression to fit a sigmoid function mapping raw
    probabilities to calibrated probabilities.  More robust than isotonic
    regression for small samples (<1000).

    Example::

        cal = SigmoidCalibrator()
        cal.fit(y_true_train, y_prob_train)
        calibrated = cal.transform(y_prob_test)
    """

    def __init__(self) -> None:
        self._fitted = False
        self._a: float = 0.0
        self._b: float = 0.0

    def fit(
        self,
        y_true: npt.NDArray[np.float64],
        y_prob: npt.NDArray[np.float64],
    ) -> None:
        """Fit Platt scaling parameters on training fold predictions.

        Parameters
        ----------
        y_true
            Binary labels (0 or 1) from the training fold.
        y_prob
            Model-predicted probabilities from the training fold.
        """
        from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

        eps = 1e-15
        clipped = np.clip(y_prob, eps, 1.0 - eps)
        log_odds = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)

        lr = LogisticRegression(solver="lbfgs", max_iter=1000)
        lr.fit(log_odds, y_true)

        self._a = float(lr.coef_[0, 0])
        self._b = float(lr.intercept_[0])
        self._fitted = True

    def transform(
        self,
        y_prob: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Apply sigmoid calibration to test fold predictions.

        Parameters
        ----------
        y_prob
            Model-predicted probabilities to calibrate.

        Returns
        -------
        npt.NDArray[np.float64]
            Calibrated probabilities in [0, 1].

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called.
        """
        if not self._fitted:
            msg = "SigmoidCalibrator is not fitted. Call fit() first."
            raise RuntimeError(msg)

        if len(y_prob) == 0:
            return np.array([], dtype=np.float64)

        eps = 1e-15
        clipped = np.clip(y_prob, eps, 1.0 - eps)
        log_odds = np.log(clipped / (1.0 - clipped))

        result: npt.NDArray[np.float64] = 1.0 / (1.0 + np.exp(-(self._a * log_odds + self._b)))
        return result
