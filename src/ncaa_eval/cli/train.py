"""Training pipeline orchestration.

Assembles feature serving, model training, prediction generation, and
run tracking into a single ``run_training()`` function consumed by the
Typer CLI entry point.
"""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from ncaa_eval.evaluation.backtest import _feature_cols
from ncaa_eval.ingest import ParquetRepository
from ncaa_eval.model.base import Model, StatefulModel
from ncaa_eval.model.tracking import ModelRun, Prediction, RunStore
from ncaa_eval.transform.feature_serving import FeatureConfig, StatefulFeatureServer
from ncaa_eval.transform.serving import ChronologicalDataServer


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

    repo = ParquetRepository(base_path=data_dir)
    data_server = ChronologicalDataServer(repo)
    feature_config = FeatureConfig(
        graph_features_enabled=False,
        batch_rating_types=(),
        ordinal_composite=None,
        calibration_method=None,
    )
    server = StatefulFeatureServer(config=feature_config, data_server=data_server)

    is_stateful = isinstance(model, StatefulModel)

    # -- Build feature matrices per season with progress display --
    season_frames: list[pd.DataFrame] = []
    with Progress() as progress:
        task = progress.add_task(
            "Building features...",
            total=end_year - start_year + 1,
        )
        for year in range(start_year, end_year + 1):
            mode = "stateful" if is_stateful else "batch"
            df = server.serve_season_features(year, mode=mode)
            if not df.empty:
                season_frames.append(df)
            progress.advance(task)

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
        store = RunStore(base_path=output_dir)
        store.save_run(run, [])
        return run

    combined = pd.concat(season_frames, ignore_index=True)

    # -- Label vector --
    y = combined["team_a_won"].astype(int)

    # -- Label balance warning --
    label_mean = y.mean()
    if label_mean > 0.95 or label_mean < 0.05:
        _console.print(
            f"[yellow]Warning: labels are heavily imbalanced "
            f"(mean={label_mean:.3f}). Consider randomising team assignment "
            f"or adjusting scale_pos_weight.[/yellow]"
        )

    # -- Compute feature columns once (reused for both training and prediction) --
    feat_cols = _feature_cols(combined)

    # -- Train --
    _console.print(f"Training [bold]{model_name}[/bold] on seasons {start_year}–{end_year}...")
    if is_stateful:
        # Stateful models need full DataFrame (metadata + features)
        model.fit(combined, y)
    else:
        # Stateless models need only feature columns
        model.fit(combined[feat_cols], y)

    # -- Generate predictions on tournament games --
    tourney = combined[combined["is_tournament"] == True].copy()  # noqa: E712
    predictions: list[Prediction] = []
    run_id = str(uuid.uuid4())

    if not tourney.empty:
        if is_stateful:
            probs = model.predict_proba(tourney)
        else:
            probs = model.predict_proba(tourney[feat_cols])

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

    # -- Persist --
    run = ModelRun(
        run_id=run_id,
        model_type=model_name,
        hyperparameters=model.get_config().model_dump(),
        git_hash=_get_git_hash(),
        start_year=start_year,
        end_year=end_year,
        prediction_count=len(predictions),
    )
    store = RunStore(base_path=output_dir)
    store.save_run(run, predictions)

    # -- Results summary --
    table = Table(title="Training Results")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Run ID", run.run_id)
    table.add_row("Model", model_name)
    table.add_row("Seasons", f"{start_year}–{end_year}")
    table.add_row("Games trained", str(len(combined)))
    table.add_row("Tournament predictions", str(len(predictions)))
    table.add_row("Git hash", run.git_hash)
    _console.print(table)

    return run
