"""Training pipeline orchestration.

Assembles feature serving, model training, prediction generation, and
run tracking into a single ``run_training()`` function consumed by the
Typer CLI entry point.
"""

from __future__ import annotations

import copy
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd  # type: ignore[import-untyped]
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from ncaa_eval.evaluation import BacktestResult, feature_cols as _feature_cols, run_backtest
from ncaa_eval.evaluation.backtest import _randomize_team_assignment
from ncaa_eval.ingest import ParquetRepository
from ncaa_eval.model.base import Model, StatefulModel
from ncaa_eval.model.tracking import ModelRun, Prediction, RunStore
from ncaa_eval.transform.feature_serving import FeatureConfig, StatefulFeatureServer
from ncaa_eval.transform.serving import ChronologicalDataServer


@dataclass
class _TrainingContext:
    """Internal context passed between pipeline stages."""

    model: Model
    model_name: str
    start_year: int
    end_year: int
    is_stateful: bool
    console: Console
    store: RunStore
    server: StatefulFeatureServer


def _get_git_hash() -> str:
    """Return the short git hash of HEAD, or ``"unknown"`` on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _build_fold_predictions(result: BacktestResult) -> pd.DataFrame | None:
    """Build a fold predictions DataFrame from backtest results.

    Args:
        result: Backtest result containing fold results with game metadata.

    Returns:
        DataFrame with columns [year, game_id, team_a_id, team_b_id,
        pred_win_prob, team_a_won], or None if no fold predictions exist.
    """
    fold_frames: list[pd.DataFrame] = []
    for fr in result.fold_results:
        if fr.predictions.empty:
            continue
        fold_frames.append(
            pd.DataFrame(
                {
                    "year": fr.year,
                    "game_id": fr.test_game_ids.values,
                    "team_a_id": fr.test_team_a_ids.values,
                    "team_b_id": fr.test_team_b_ids.values,
                    "pred_win_prob": fr.predictions.values,
                    "team_a_won": fr.actuals.values,
                }
            )
        )
    if not fold_frames:
        return None
    return pd.concat(fold_frames, ignore_index=True)


def _setup_feature_server(data_dir: Path) -> StatefulFeatureServer:
    """Initialize the repository, data server, and feature server."""
    repo = ParquetRepository(base_path=data_dir)
    data_server = ChronologicalDataServer(repo)
    feature_config = FeatureConfig(
        graph_features_enabled=False,
        batch_rating_types=("srs",),
        ordinal_composite=None,
        calibration_method=None,
    )
    return StatefulFeatureServer(config=feature_config, data_server=data_server)


def _build_season_features(ctx: _TrainingContext) -> list[pd.DataFrame]:
    """Build feature matrices per season with a progress display."""
    season_frames: list[pd.DataFrame] = []
    with Progress() as progress:
        task = progress.add_task(
            "Building features...",
            total=ctx.end_year - ctx.start_year + 1,
        )
        for year in range(ctx.start_year, ctx.end_year + 1):
            mode: Literal["batch", "stateful"] = "stateful" if ctx.is_stateful else "batch"
            df = ctx.server.serve_season_features(year, mode=mode)
            if not df.empty:
                season_frames.append(df)
            progress.advance(task)
    return season_frames


def _prepare_and_train(ctx: _TrainingContext, combined: pd.DataFrame) -> list[str]:
    """Extract labels, check balance, compute feature columns, and train.

    Extracts ``team_a_won`` as integer labels and warns if the label mean
    is outside ``[0.05, 0.95]`` (heavy imbalance).  Computes ``feat_cols``
    via ``_feature_cols(combined)``.  For stateful models, passes the full
    ``combined`` DataFrame (model uses internal state for features); for
    stateless models, slices to ``combined[feat_cols]`` before calling
    ``model.fit``.

    Returns:
        List of feature column names used for training.
    """
    # Stateless classifiers require balanced labels; the feature server
    # assigns team_a = winner for every game, making team_a_won always True.
    if not ctx.is_stateful:
        combined = _randomize_team_assignment(combined)

    y = combined["team_a_won"].astype(int)

    label_mean = y.mean()
    if label_mean > 0.95 or label_mean < 0.05:
        ctx.console.print(
            f"[yellow]Warning: labels are heavily imbalanced "
            f"(mean={label_mean:.3f}). Consider randomising team assignment "
            f"or adjusting scale_pos_weight.[/yellow]"
        )

    feat_cols = _feature_cols(combined)
    if not ctx.is_stateful:
        # Drop columns that are entirely NaN (e.g. seed features without a seed
        # table) so sklearn classifiers that reject NaN inputs can still fit.
        feat_cols = [c for c in feat_cols if not combined[c].isna().all()]

    ctx.console.print(f"Training [bold]{ctx.model_name}[/bold] on seasons {ctx.start_year}–{ctx.end_year}...")
    if ctx.is_stateful:
        ctx.model.fit(combined, y)
    else:
        ctx.model.fit(combined[feat_cols], y)

    return feat_cols


def _generate_tournament_predictions(
    ctx: _TrainingContext,
    combined: pd.DataFrame,
    feat_cols: list[str],
    run_id: str,
) -> list[Prediction]:
    """Generate predictions on tournament games."""
    tourney = combined[combined["is_tournament"] == True].copy()  # noqa: E712
    predictions: list[Prediction] = []

    if not tourney.empty:
        if ctx.is_stateful:
            probs = ctx.model.predict_proba(tourney)
        else:
            probs = ctx.model.predict_proba(tourney[feat_cols])

        for idx, prob in probs.items():
            row = tourney.loc[idx]
            predictions.append(
                Prediction(
                    run_id=run_id,
                    game_id=str(row["game_id"]),
                    season=int(row["season"]),
                    team_a_id=int(row["team_a_id"]),
                    team_b_id=int(row["team_b_id"]),
                    pred_win_prob=float(min(max(prob, 0.0), 1.0)),
                )
            )

    return predictions


def _run_backtest_and_persist(ctx: _TrainingContext, run_id: str) -> None:
    """Run walk-forward backtest and persist metrics and fold predictions.

    Guards on ``len(seasons) >= 2`` — a single season cannot produce
    walk-forward folds.  Deep-copies the model before passing it to
    ``run_backtest`` to prevent the backtest's sequential ``fit`` calls
    from mutating the already-trained model held in ``ctx``.  Saves the
    summary metrics and, if fold-level predictions exist, the per-game
    prediction DataFrame via ``RunStore``.
    """
    seasons = list(range(ctx.start_year, ctx.end_year + 1))
    if len(seasons) >= 2:
        ctx.console.print("Running walk-forward backtest...")
        # Deep-copy to avoid mutating the trained model: run_backtest
        # calls model.fit() on each fold, which would overwrite ctx.model.
        backtest_model = copy.deepcopy(ctx.model)
        mode: Literal["batch", "stateful"] = "stateful" if ctx.is_stateful else "batch"
        result = run_backtest(
            backtest_model,
            ctx.server,
            seasons=seasons,
            mode=mode,
            n_jobs=1,
            console=ctx.console,
        )
        ctx.store.save_metrics(run_id, result.summary)

        fold_preds = _build_fold_predictions(result)
        if fold_preds is not None:
            ctx.store.save_fold_predictions(run_id, fold_preds)

        ctx.console.print("[green]Backtest metrics persisted.[/green]")
    else:
        ctx.console.print("[yellow]Skipping backtest: need ≥ 2 seasons.[/yellow]")


def _persist_artifacts_and_summarize(
    ctx: _TrainingContext,
    run: ModelRun,
    feat_cols: list[str],
    combined: pd.DataFrame,
    predictions: list[Prediction],
) -> None:
    """Save the trained model and print a summary table."""
    ctx.store.save_model(run.run_id, ctx.model, feature_names=feat_cols)
    ctx.console.print("[green]Model artifacts persisted.[/green]")

    table = Table(title="Training Results")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Run ID", run.run_id)
    table.add_row("Model", ctx.model_name)
    table.add_row("Seasons", f"{ctx.start_year}–{ctx.end_year}")
    table.add_row("Games trained", str(len(combined)))
    table.add_row("Tournament predictions", str(len(predictions)))
    table.add_row("Git hash", run.git_hash)
    ctx.console.print(table)


def run_training(  # noqa: PLR0913
    model: Model,
    *,
    start_year: int,
    end_year: int,
    data_dir: Path,
    output_dir: Path,
    model_name: str,
    console: Console | None = None,
) -> ModelRun:
    """Execute the full train → predict → persist pipeline.

    Initialises a ``_TrainingContext`` by calling ``_setup_feature_server``
    and wrapping all run-scoped state.  Calls ``_build_season_features`` to
    produce per-season DataFrames; short-circuits with an empty ``ModelRun``
    if no game data is found.  Concatenates frames, then calls
    ``_prepare_and_train`` (label extraction, balance check, fit) to get
    ``feat_cols``.  Generates tournament predictions via
    ``_generate_tournament_predictions``, persists the ``ModelRun`` record,
    runs the walk-forward backtest via ``_run_backtest_and_persist``
    (skipped if fewer than 2 seasons), and finally saves model artifacts and
    prints the summary table via ``_persist_artifacts_and_summarize``.

    Args:
        model: An instantiated model (stateful or stateless).
        start_year: First season year (inclusive) for training.
        end_year: Last season year (inclusive) for training.
        data_dir: Path to the local Parquet data store.
        output_dir: Path where run artifacts are persisted.
        model_name: Registered plugin name (used in the ModelRun record).
        console: Rich Console instance for terminal output. Defaults to a
            fresh ``Console()`` so callers (e.g. tests) can suppress output
            by passing ``Console(quiet=True)``.

    Returns:
        The persisted run metadata record.
    """
    _console = console or Console()
    server = _setup_feature_server(data_dir)
    store = RunStore(base_path=output_dir)
    ctx = _TrainingContext(
        model=model,
        model_name=model_name,
        start_year=start_year,
        end_year=end_year,
        is_stateful=isinstance(model, StatefulModel),
        console=_console,
        store=store,
        server=server,
    )

    # Build feature matrices per season
    season_frames = _build_season_features(ctx)

    if not season_frames:
        _console.print("[yellow]No game data found for the specified year range.[/yellow]")
        run_id = str(uuid.uuid4())
        run = ModelRun(
            run_id=run_id,
            model_type=model_name,
            hyperparameters=model.get_config().model_dump(),
            git_hash=_get_git_hash(),
            start_year=start_year,
            end_year=end_year,
            prediction_count=0,
        )
        store.save_run(run, [])
        return run

    combined = pd.concat(season_frames, ignore_index=True)

    # Train model
    feat_cols = _prepare_and_train(ctx, combined)

    # Generate predictions
    run_id = str(uuid.uuid4())
    predictions = _generate_tournament_predictions(ctx, combined, feat_cols, run_id)

    # Persist run
    run = ModelRun(
        run_id=run_id,
        model_type=model_name,
        hyperparameters=model.get_config().model_dump(),
        git_hash=_get_git_hash(),
        start_year=start_year,
        end_year=end_year,
        prediction_count=len(predictions),
    )
    store.save_run(run, predictions)

    # Backtest
    _run_backtest_and_persist(ctx, run.run_id)

    # Save model and summarize
    _persist_artifacts_and_summarize(ctx, run, feat_cols, combined, predictions)

    return run
