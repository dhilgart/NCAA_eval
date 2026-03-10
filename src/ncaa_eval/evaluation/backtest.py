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
import logging
import math
import time
import types
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import joblib  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]
from rich.console import Console
from rich.table import Table

from ncaa_eval.evaluation.metrics import (
    MetricFn,
    brier_score,
    expected_calibration_error,
    get_metric,
    list_metrics,
    log_loss,
    roc_auc,
)
from ncaa_eval.evaluation.splitter import CVFold, walk_forward_splits
from ncaa_eval.model.base import Model, StatefulModel
from ncaa_eval.transform.feature_serving import StatefulFeatureServer

logger = logging.getLogger(__name__)

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


def default_metrics() -> dict[str, MetricFn]:
    """Return all registered metric functions (built-in + user-registered)."""
    return {name: get_metric(name) for name in list_metrics()}


def feature_cols(df: pd.DataFrame) -> list[str]:
    """Return feature column names (everything not in METADATA_COLS).

    Args:
        df: DataFrame whose columns are inspected.

    Returns:
        List of column names that are not metadata.
    """
    return [c for c in df.columns if c not in METADATA_COLS]


def _randomize_team_assignment(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Randomly swap team_a/team_b for ~50% of rows to balance binary labels.

    The feature server assigns ``team_a = winner`` for every game, so
    ``team_a_won`` is always ``True``.  Stateless classifiers require at least
    two classes in the training labels; this function randomly re-assigns which
    team is "a" so roughly half the rows become ``team_a_won = False``.

    Paired ``_a`` / ``_b`` feature columns are swapped; ``delta_*`` and
    ``seed_diff`` columns are negated; ``loc_encoding`` is negated.

    Args:
        df: Feature DataFrame from ``StatefulFeatureServer.serve_season_features``.
        seed: RNG seed for reproducibility.

    Returns:
        Copy of ``df`` with team assignment randomly swapped for ~50% of rows.
    """
    rng = np.random.default_rng(seed)
    mask = pd.Series(rng.random(len(df)) < 0.5, index=df.index)
    result = df.copy()

    # Swap team IDs
    tmp_id = result.loc[mask, "team_a_id"].copy()
    result.loc[mask, "team_a_id"] = result.loc[mask, "team_b_id"]
    result.loc[mask, "team_b_id"] = tmp_id

    # Flip label
    result.loc[mask, "team_a_won"] = ~result.loc[mask, "team_a_won"].astype(bool)

    # Negate direction-dependent scalar columns
    for col in result.columns:
        if col.startswith("delta_") or col in ("seed_diff", "loc_encoding"):
            result.loc[mask, col] = -result.loc[mask, col]

    # Swap paired _a / _b feature columns (e.g. srs_a ↔ srs_b)
    checked: set[str] = set()
    for col in list(result.columns):
        if col.endswith("_a") and col not in ("team_a_id", "team_a_won"):
            base = col[:-2]
            col_b = f"{base}_b"
            if col_b in result.columns and base not in checked:
                checked.add(base)
                tmp_a = result.loc[mask, col].copy()
                result.loc[mask, col] = result.loc[mask, col_b]
                result.loc[mask, col_b] = tmp_a

    return result


@dataclasses.dataclass(frozen=True)
class FoldResult:
    """Result of evaluating a single cross-validation fold.

    Attributes:
        year: The test season year for this fold.
        predictions: Predicted probabilities for tournament games.
        actuals: Actual binary outcomes for tournament games.
        metrics: Mapping of metric name to computed value.
        elapsed_seconds: Wall-clock time for the fold evaluation.
        test_game_ids: Game IDs from the test fold (aligned to predictions).
        test_team_a_ids: team_a IDs from the test fold.
        test_team_b_ids: team_b IDs from the test fold.
    """

    year: int
    predictions: pd.Series
    actuals: pd.Series
    metrics: Mapping[str, float]
    elapsed_seconds: float
    test_game_ids: pd.Series = dataclasses.field(
        default_factory=lambda: pd.Series(dtype=object),
    )
    test_team_a_ids: pd.Series = dataclasses.field(
        default_factory=lambda: pd.Series(dtype="int64"),
    )
    test_team_b_ids: pd.Series = dataclasses.field(
        default_factory=lambda: pd.Series(dtype="int64"),
    )


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
            test_game_ids=pd.Series(dtype=object),
            test_team_a_ids=pd.Series(dtype="int64"),
            test_team_b_ids=pd.Series(dtype="int64"),
        )

    # Randomize test-fold team assignment: feature server always assigns
    # team_a = winner, so y_test would be all 1s without this, making
    # roc_auc undefined.  Use a distinct seed from the train randomisation.
    test_data = _randomize_team_assignment(fold.test, seed=43)
    y_test = test_data["team_a_won"].astype(np.float64)

    is_stateful = isinstance(model, StatefulModel)
    feat_cols = feature_cols(fold.train)

    if is_stateful:
        y_train = fold.train["team_a_won"].astype(np.float64)
        model.fit(fold.train, y_train)
    else:
        # Balance labels: feature server assigns team_a = winner always,
        # so team_a_won is always True.  Randomise assignment before fitting.
        train_data = _randomize_team_assignment(fold.train)
        y_train = train_data["team_a_won"].astype(np.float64)
        # Drop all-NaN columns so sklearn estimators that reject NaN can fit.
        feat_cols = [c for c in feat_cols if not fold.train[c].isna().all()]
        model.fit(train_data[feat_cols], y_train)

    if is_stateful:
        preds = model.predict_proba(test_data)
    else:
        preds = model.predict_proba(test_data[feat_cols])

    y_true_np = y_test.to_numpy()
    y_prob_np = preds.to_numpy().astype(np.float64)

    metrics: dict[str, float] = {name: fn(y_true_np, y_prob_np) for name, fn in metric_fns.items()}

    elapsed = time.perf_counter() - start
    return FoldResult(
        year=fold.year,
        predictions=preds,
        actuals=y_test,
        metrics=metrics,
        elapsed_seconds=elapsed,
        test_game_ids=test_data["game_id"].reset_index(drop=True),
        test_team_a_ids=test_data["team_a_id"].reset_index(drop=True),
        test_team_b_ids=test_data["team_b_id"].reset_index(drop=True),
    )


def run_backtest(  # noqa: PLR0913 — REFACTOR Story 8.1
    model: Model,
    feature_server: StatefulFeatureServer,
    *,
    seasons: Sequence[int],
    mode: Literal["batch", "stateful"] = "batch",
    n_jobs: int = -1,
    metric_fns: Mapping[
        str,
        Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], float],
    ]
    | None = None,
    console: Console | None = None,
    progress: bool = False,
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
        progress: Display a tqdm progress bar for fold evaluation.
            Most useful with ``n_jobs=1`` (sequential execution).

    Returns:
        BacktestResult with per-fold results and summary DataFrame.

    Raises:
        ValueError: If ``mode`` is not ``"batch"`` or ``"stateful"``, or if
            ``seasons`` contains fewer than 2 elements (propagated from
            :func:`walk_forward_splits`).
    """
    # Runtime guard: Literal["batch","stateful"] enforces at static-analysis
    # time; this check also protects callers who bypass mypy (e.g. YAML config).
    if mode not in ("batch", "stateful"):
        msg = f"mode must be 'batch' or 'stateful', got {mode!r}"
        raise ValueError(msg)

    resolved_metrics = default_metrics() if metric_fns is None else dict(metric_fns)

    total_start = time.perf_counter()

    # Materialize folds eagerly (generators can't be pickled for joblib)
    folds = list(walk_forward_splits(seasons, feature_server, mode=mode))

    # Deep-copy model per fold to avoid shared-state corruption
    models = [copy.deepcopy(model) for _ in folds]

    _console = console or Console()
    _console.print(f"Running backtest: {len(folds)} folds, n_jobs={n_jobs}")

    # Dispatch fold evaluation (with optional tqdm progress bar)
    if progress and n_jobs != 1:
        import warnings

        warnings.warn(
            "progress=True is only supported with n_jobs=1; progress bar skipped for parallel execution.",
            UserWarning,
            stacklevel=2,
        )
    if progress and n_jobs == 1:
        from tqdm.auto import tqdm  # type: ignore[import-untyped]

        results: list[FoldResult] = [
            _evaluate_fold(fold, m, resolved_metrics)
            for fold, m in tqdm(
                zip(folds, models),
                total=len(folds),
                desc="Backtest folds",
            )
        ]
    else:
        results = joblib.Parallel(n_jobs=n_jobs)(
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
