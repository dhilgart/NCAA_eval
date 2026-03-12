"""Training pipeline orchestration.

Assembles feature serving, model training, prediction generation, and
run tracking into a single ``run_training()`` function consumed by the
Typer CLI entry point.
"""

from __future__ import annotations

import copy
import json
import logging
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
from ncaa_eval.model.ensemble import StackedEnsemble
from ncaa_eval.model.tracking import ModelRun, Prediction, RunStore
from ncaa_eval.transform.feature_serving import FeatureConfig, StatefulFeatureServer
from ncaa_eval.transform.serving import ChronologicalDataServer

_logger = logging.getLogger(__name__)


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


def _setup_feature_server(data_dir: Path, feature_config: FeatureConfig) -> StatefulFeatureServer:
    """Initialize the repository, data server, and feature server."""
    repo = ParquetRepository(base_path=data_dir)
    data_server = ChronologicalDataServer(repo)
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
    model: Model | StackedEnsemble,
    *,
    start_year: int,
    end_year: int,
    data_dir: Path,
    output_dir: Path,
    model_name: str,
    console: Console | None = None,
) -> ModelRun:
    """Execute the full train → predict → persist pipeline.

    Dispatches to ``_run_ensemble_training`` when *model* is a
    ``StackedEnsemble``; otherwise runs the single-model pipeline.

    Args:
        model: An instantiated model (stateful, stateless, or ensemble).
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

    if isinstance(model, StackedEnsemble):
        return _run_ensemble_training(
            model,
            start_year=start_year,
            end_year=end_year,
            data_dir=data_dir,
            output_dir=output_dir,
            model_name=model_name,
            console=_console,
        )

    server = _setup_feature_server(data_dir, model.feature_config)
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


# ── Ensemble training pipeline ─────────────────────────────────────────────


def _collect_oof_predictions(  # noqa: PLR0913
    base_model: Model,
    base_idx: int,
    data_dir: Path,
    start_year: int,
    end_year: int,
    console: Console,
) -> pd.DataFrame:
    """Run a walk-forward backtest for one base model and return OOF predictions.

    Returns a DataFrame with columns
    ``[year, game_id, team_a_id, team_b_id, pred_base_<idx>, team_a_won]``.
    Contextual features (``seed_diff``, ``is_tournament``, ``loc_encoding``)
    are NOT included here; they are joined in Step 3 of
    ``_run_ensemble_training`` from the union feature server.
    """
    server = _setup_feature_server(data_dir, base_model.feature_config)
    seasons = list(range(start_year, end_year + 1))
    mode: Literal["batch", "stateful"] = "stateful" if isinstance(base_model, StatefulModel) else "batch"

    console.print(f"  OOF backtest for base model {base_idx} ({type(base_model).__name__})...")
    result = run_backtest(
        copy.deepcopy(base_model),
        server,
        seasons=seasons,
        mode=mode,
        n_jobs=1,
        console=console,
    )

    fold_preds = _build_fold_predictions(result)
    if fold_preds is None:
        return pd.DataFrame()

    fold_preds = fold_preds.rename(columns={"pred_win_prob": f"pred_base_{base_idx}"})
    return fold_preds


def _align_oof_predictions(
    oof_frames: list[pd.DataFrame],
) -> pd.DataFrame:
    """Inner-join OOF predictions from all base models on ``game_id``.

    Logs a warning if >5% of games are dropped by the join.
    """
    if not oof_frames:
        return pd.DataFrame()

    aligned = oof_frames[0]
    for frame in oof_frames[1:]:
        pred_cols = [c for c in frame.columns if c.startswith("pred_base_")]
        aligned = aligned.merge(
            frame[["game_id", *pred_cols]],
            on="game_id",
            how="inner",
        )

    max_len = max(len(f) for f in oof_frames)
    if max_len > 0:
        drop_pct = 1.0 - len(aligned) / max_len
        if drop_pct > 0.05:
            _logger.warning(
                "OOF alignment dropped %.1f%% of games (%d → %d)",
                drop_pct * 100,
                max_len,
                len(aligned),
            )

    return aligned


def _build_meta_training_set(
    aligned: pd.DataFrame,
    contextual_features: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Build meta-learner X and y from aligned OOF predictions.

    Returns (meta_X, meta_y) where meta_X has columns
    ``[pred_base_0, ..., <contextual_features>]`` and meta_y is ``team_a_won``.
    """
    pred_cols = sorted(c for c in aligned.columns if c.startswith("pred_base_"))
    context_cols = [c for c in contextual_features if c in aligned.columns]
    meta_X = aligned[pred_cols + context_cols].copy()
    meta_y = aligned["team_a_won"].astype(int)
    return meta_X, meta_y


def _retrain_base_models(
    ensemble: StackedEnsemble,
    data_dir: Path,
    start_year: int,
    end_year: int,
    console: Console,
) -> None:
    """Retrain each base model on the full dataset."""
    for i, base_model in enumerate(ensemble.base_models):
        console.print(f"  Retraining base model {i} on full dataset...")
        server = _setup_feature_server(data_dir, base_model.feature_config)
        is_stateful = isinstance(base_model, StatefulModel)

        season_frames: list[pd.DataFrame] = []
        for year in range(start_year, end_year + 1):
            mode: Literal["batch", "stateful"] = "stateful" if is_stateful else "batch"
            df = server.serve_season_features(year, mode=mode)
            if not df.empty:
                season_frames.append(df)

        if not season_frames:
            continue

        combined = pd.concat(season_frames, ignore_index=True)
        if not is_stateful:
            combined = _randomize_team_assignment(combined)

        y = combined["team_a_won"].astype(int)
        if is_stateful:
            base_model.fit(combined, y)
        else:
            feat_cols = _feature_cols(combined)
            feat_cols = [c for c in feat_cols if not combined[c].isna().all()]
            base_model.fit(combined[feat_cols], y)


def _run_ensemble_training(  # noqa: PLR0913
    ensemble: StackedEnsemble,
    *,
    start_year: int,
    end_year: int,
    data_dir: Path,
    output_dir: Path,
    model_name: str,
    console: Console,
) -> ModelRun:
    """Execute the stacked ensemble training pipeline.

    Steps:
      1. OOF generation — walk-forward backtest per base model
      2. OOF alignment — inner join on game_id
      3. Meta-training set construction
      4. Meta-learner training
      5. Full-dataset base model retraining
      6. Artifact persistence
    """
    store = RunStore(base_path=output_dir)

    # Step 1: OOF generation
    console.print("[bold]Step 1/6: Generating OOF predictions...[/bold]")
    oof_frames: list[pd.DataFrame] = []
    oof_run_ids: list[str] = []
    for i, base_model in enumerate(ensemble.base_models):
        oof_df = _collect_oof_predictions(
            base_model,
            base_idx=i,
            data_dir=data_dir,
            start_year=start_year,
            end_year=end_year,
            console=console,
        )
        oof_frames.append(oof_df)
        oof_run_ids.append(str(uuid.uuid4()))

    # Step 2: OOF alignment
    console.print("[bold]Step 2/6: Aligning OOF predictions...[/bold]")
    aligned = _align_oof_predictions(oof_frames)
    if aligned.empty:
        console.print("[yellow]No aligned OOF predictions — aborting ensemble training.[/yellow]")
        run_id = str(uuid.uuid4())
        run = ModelRun(
            run_id=run_id,
            model_type="ensemble",
            hyperparameters=ensemble.get_config().model_dump(),
            git_hash=_get_git_hash(),
            start_year=start_year,
            end_year=end_year,
            prediction_count=0,
        )
        store.save_run(run, [])
        return run

    console.print(f"  Aligned OOF games: {len(aligned)}")

    # Step 3: Meta-training set
    console.print("[bold]Step 3/6: Building meta-training set...[/bold]")
    # Contextual features (seed_diff, is_tournament, loc_encoding) are
    # present in the OOF fold predictions from _build_fold_predictions
    # via _evaluate_fold's test data, but _build_fold_predictions only
    # returns game_id/pred/actuals. We need to get them from a feature
    # server. Build one with the ensemble's union feature_config.
    union_server = _setup_feature_server(data_dir, ensemble.feature_config)
    context_frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        df = union_server.serve_season_features(year, mode="batch")
        if not df.empty:
            context_frames.append(df[["game_id", *ensemble.contextual_features]])

    if context_frames:
        context_df = pd.concat(context_frames, ignore_index=True)
        # The feature server assigns team_a = winner always, so game_ids
        # are unique per game. Inner join: if a game_id is missing from the
        # context server it likely indicates a data inconsistency; NaN
        # contextual features would silently poison the meta-learner.
        before_merge = len(aligned)
        aligned = aligned.merge(context_df, on="game_id", how="inner")
        if len(aligned) < before_merge:
            _logger.warning(
                "Context feature join dropped %d OOF games (%d → %d); "
                "possible game_id mismatch between base model OOF and union feature server.",
                before_merge - len(aligned),
                before_merge,
                len(aligned),
            )

    meta_X, meta_y = _build_meta_training_set(aligned, ensemble.contextual_features)
    meta_column_order = list(meta_X.columns)
    ensemble.meta_column_order = meta_column_order
    console.print(f"  Meta-training shape: {meta_X.shape}")

    # Step 4: Meta-learner training
    console.print("[bold]Step 4/6: Training meta-learner...[/bold]")
    ensemble.meta_learner.fit(meta_X, meta_y)

    # Step 5: Retrain base models on full dataset
    console.print("[bold]Step 5/6: Retraining base models on full dataset...[/bold]")
    _retrain_base_models(ensemble, data_dir, start_year, end_year, console)

    # Step 6: Persist
    console.print("[bold]Step 6/6: Persisting ensemble artifacts...[/bold]")
    run_id = str(uuid.uuid4())
    run = ModelRun(
        run_id=run_id,
        model_type="ensemble",
        hyperparameters=ensemble.get_config().model_dump(),
        git_hash=_get_git_hash(),
        start_year=start_year,
        end_year=end_year,
        prediction_count=0,
    )
    store.save_run(run, [])
    store.save_model(run_id, ensemble)

    # Augment manifest with OOF metadata
    model_dir = store.model_dir(run_id)
    manifest_path = model_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["meta_column_order"] = meta_column_order
    manifest["oof_backtest_run_ids"] = oof_run_ids
    manifest["oof_game_count"] = len(aligned)
    max_len = max(len(f) for f in oof_frames) if oof_frames else 0
    manifest["oof_drop_pct"] = round(1.0 - len(aligned) / max_len, 4) if max_len > 0 else 0.0
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Summary table
    table = Table(title="Ensemble Training Results")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Run ID", run.run_id)
    table.add_row("Model", model_name)
    table.add_row("Base models", str(len(ensemble.base_models)))
    table.add_row("Seasons", f"{start_year}–{end_year}")
    table.add_row("OOF games", str(len(aligned)))
    table.add_row("Meta features", str(len(meta_column_order)))
    table.add_row("Git hash", run.git_hash)
    console.print(table)

    return run
