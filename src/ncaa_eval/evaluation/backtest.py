"""Parallel cross-validation backtest orchestrator.

Provides :func:`run_backtest`, which executes walk-forward cross-validation
folds in parallel using ``joblib.Parallel``.  Each fold trains an independent
deep-copied model instance, generates predictions on tournament games, and
computes evaluation metrics.  Results are aggregated into a
:class:`BacktestResult` containing per-fold details and a summary DataFrame.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import time
import types
from collections.abc import Callable, Mapping, Sequence

import joblib  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]
from rich.console import Console
from rich.table import Table

from ncaa_eval.evaluation.metrics import (
    brier_score,
    expected_calibration_error,
    log_loss,
    roc_auc,
)
from ncaa_eval.evaluation.splitter import CVFold, walk_forward_splits
from ncaa_eval.model.base import Model, StatefulModel
from ncaa_eval.transform.feature_serving import StatefulFeatureServer

# Metadata columns that must be stripped before feeding stateless models.
METADATA_COLS: frozenset[str] = frozenset(
    {
        "game_id",
        "season",
        "day_num",
        "date",
        "team_a_id",
        "team_b_id",
        "is_tournament",
        "loc_encoding",
        "team_a_won",
        "w_score",
        "l_score",
        "num_ot",
    }
)

_VALID_MODES: frozenset[str] = frozenset({"batch", "stateful"})

DEFAULT_METRICS: Mapping[
    str,
    Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], float],
] = types.MappingProxyType(
    {
        "log_loss": log_loss,
        "brier_score": brier_score,
        "roc_auc": roc_auc,
        "ece": expected_calibration_error,
    }
)


def _feature_cols(df: pd.DataFrame) -> list[str]:
    """Return feature column names (everything not in METADATA_COLS).

    Args:
        df: DataFrame whose columns are inspected.

    Returns:
        List of column names that are not metadata.
    """
    return [c for c in df.columns if c not in METADATA_COLS]


@dataclasses.dataclass(frozen=True)
class FoldResult:
    """Result of evaluating a single cross-validation fold.

    Attributes:
        year: The test season year for this fold.
        predictions: Predicted probabilities for tournament games.
        actuals: Actual binary outcomes for tournament games.
        metrics: Mapping of metric name to computed value.
        elapsed_seconds: Wall-clock time for the fold evaluation.
    """

    year: int
    predictions: pd.Series
    actuals: pd.Series
    metrics: Mapping[str, float]
    elapsed_seconds: float


@dataclasses.dataclass(frozen=True)
class BacktestResult:
    """Aggregated result of a full backtest across all folds.

    Attributes:
        fold_results: Per-fold evaluation results, sorted by year.
        summary: DataFrame with year as index, metric columns + elapsed_seconds.
        elapsed_seconds: Total wall-clock time for the entire backtest.
    """

    fold_results: tuple[FoldResult, ...]
    summary: pd.DataFrame
    elapsed_seconds: float


def _evaluate_fold(
    fold: CVFold,
    model: Model,
    metric_fns: Mapping[
        str,
        Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], float],
    ],
) -> FoldResult:
    """Train model on fold.train, predict on fold.test, compute metrics.

    Args:
        fold: A single CV fold with train/test DataFrames.
        model: A deep-copied model instance (caller is responsible for copying).
        metric_fns: Mapping of metric name to callable(y_true, y_prob) returning float.

    Returns:
        FoldResult with predictions, actuals, computed metrics, and timing.
    """
    start = time.perf_counter()

    if fold.test.empty:
        elapsed = time.perf_counter() - start
        return FoldResult(
            year=fold.year,
            predictions=pd.Series(dtype=np.float64),
            actuals=pd.Series(dtype=np.float64),
            metrics={name: float("nan") for name in metric_fns},
            elapsed_seconds=elapsed,
        )

    y_train = fold.train["team_a_won"].astype(np.float64)
    y_test = fold.test["team_a_won"].astype(np.float64)

    is_stateful = isinstance(model, StatefulModel)
    feat_cols = _feature_cols(fold.train)

    if is_stateful:
        model.fit(fold.train, y_train)
    else:
        model.fit(fold.train[feat_cols], y_train)

    if is_stateful:
        preds = model.predict_proba(fold.test)
    else:
        preds = model.predict_proba(fold.test[feat_cols])

    y_true_np = y_test.to_numpy()
    y_prob_np = preds.to_numpy().astype(np.float64)

    metrics: dict[str, float] = {}
    for name, fn in metric_fns.items():
        try:
            metrics[name] = fn(y_true_np, y_prob_np)
        except Exception:  # noqa: BLE001
            metrics[name] = float("nan")

    elapsed = time.perf_counter() - start
    return FoldResult(
        year=fold.year,
        predictions=preds,
        actuals=y_test,
        metrics=metrics,
        elapsed_seconds=elapsed,
    )


def run_backtest(  # noqa: PLR0913
    model: Model,
    feature_server: StatefulFeatureServer,
    *,
    seasons: Sequence[int],
    mode: str = "batch",
    n_jobs: int = -1,
    metric_fns: Mapping[
        str,
        Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], float],
    ]
    | None = None,
    console: Console | None = None,
) -> BacktestResult:
    """Run parallelized walk-forward cross-validation backtest.

    Args:
        model: Model instance to evaluate (will be deep-copied per fold).
        feature_server: Configured feature server for building CV folds.
        seasons: Season years to include (passed to walk_forward_splits).
        mode: Feature serving mode (``"batch"`` or ``"stateful"``).
        n_jobs: Number of parallel workers. -1 = all cores, 1 = sequential.
        metric_fns: Metric functions to compute per fold. Defaults to
            {log_loss, brier_score, roc_auc, expected_calibration_error}.
        console: Rich Console for progress output.

    Returns:
        BacktestResult with per-fold results and summary DataFrame.

    Raises:
        ValueError: If ``mode`` is not ``"batch"`` or ``"stateful"``, or if
            ``seasons`` contains fewer than 2 elements (propagated from
            :func:`walk_forward_splits`).
    """
    if mode not in _VALID_MODES:
        msg = f"mode must be 'batch' or 'stateful', got {mode!r}"
        raise ValueError(msg)

    resolved_metrics = dict(DEFAULT_METRICS) if metric_fns is None else dict(metric_fns)

    total_start = time.perf_counter()

    # Materialize folds eagerly (generators can't be pickled for joblib)
    folds = list(walk_forward_splits(seasons, feature_server, mode=mode))

    # Deep-copy model per fold to avoid shared-state corruption
    models = [copy.deepcopy(model) for _ in folds]

    _console = console or Console()
    _console.print(f"Running backtest: {len(folds)} folds, n_jobs={n_jobs}")

    # Dispatch parallel fold evaluation
    results: list[FoldResult] = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_evaluate_fold)(fold, m, resolved_metrics) for fold, m in zip(folds, models)
    )

    # Sort by year ascending (joblib may return out of order)
    results.sort(key=lambda r: r.year)

    total_elapsed = time.perf_counter() - total_start

    # Build summary DataFrame
    summary_rows: list[dict[str, object]] = []
    for r in results:
        row: dict[str, object] = {"year": r.year}
        for metric_name in resolved_metrics:
            row[metric_name] = r.metrics.get(metric_name, float("nan"))
        row["elapsed_seconds"] = r.elapsed_seconds
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).set_index("year")

    # Progress report via Rich table
    table = Table(title="Backtest Results")
    table.add_column("Year", style="cyan")
    for metric_name in resolved_metrics:
        table.add_column(metric_name, style="green")
    table.add_column("Time (s)", style="yellow")

    for r in results:
        row_values = [str(r.year)]
        for metric_name in resolved_metrics:
            val = r.metrics.get(metric_name, float("nan"))
            row_values.append(f"{val:.4f}" if not math.isnan(val) else "NaN")
        row_values.append(f"{r.elapsed_seconds:.2f}")
        table.add_row(*row_values)

    _console.print(table)
    _console.print(f"Total backtest time: {total_elapsed:.2f}s")

    return BacktestResult(
        fold_results=tuple(results),
        summary=summary,
        elapsed_seconds=total_elapsed,
    )
