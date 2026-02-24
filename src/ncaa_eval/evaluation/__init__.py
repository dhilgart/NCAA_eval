"""Evaluation metrics and cross-validation module."""

from __future__ import annotations

from ncaa_eval.evaluation.backtest import BacktestResult, FoldResult, run_backtest
from ncaa_eval.evaluation.metrics import (
    ReliabilityData,
    brier_score,
    expected_calibration_error,
    log_loss,
    reliability_diagram_data,
    roc_auc,
)
from ncaa_eval.evaluation.splitter import CVFold, walk_forward_splits

__all__ = [
    "BacktestResult",
    "CVFold",
    "FoldResult",
    "ReliabilityData",
    "brier_score",
    "expected_calibration_error",
    "log_loss",
    "reliability_diagram_data",
    "roc_auc",
    "run_backtest",
    "walk_forward_splits",
]
