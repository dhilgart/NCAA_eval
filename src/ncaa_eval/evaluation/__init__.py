"""Evaluation metrics, cross-validation, and tournament simulation module."""

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
from ncaa_eval.evaluation.simulation import (
    SCORING_REGISTRY,
    BracketNode,
    BracketStructure,
    CustomScoring,
    EloProvider,
    FibonacciScoring,
    MatchupContext,
    MatrixProvider,
    ProbabilityProvider,
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
from ncaa_eval.evaluation.splitter import CVFold, walk_forward_splits

__all__ = [
    "BacktestResult",
    "BracketNode",
    "BracketStructure",
    "CVFold",
    "CustomScoring",
    "EloProvider",
    "FibonacciScoring",
    "FoldResult",
    "MatchupContext",
    "MatrixProvider",
    "ProbabilityProvider",
    "ReliabilityData",
    "SCORING_REGISTRY",
    "ScoringRule",
    "SeedDiffBonusScoring",
    "SimulationResult",
    "StandardScoring",
    "brier_score",
    "build_bracket",
    "build_probability_matrix",
    "compute_advancement_probs",
    "compute_expected_points",
    "expected_calibration_error",
    "log_loss",
    "reliability_diagram_data",
    "roc_auc",
    "run_backtest",
    "simulate_tournament",
    "simulate_tournament_mc",
    "walk_forward_splits",
]
