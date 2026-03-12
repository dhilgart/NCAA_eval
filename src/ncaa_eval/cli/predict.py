"""Prediction orchestration for CLI predict command.

Loads a trained model, generates win-probability predictions for a target
season, and formats output as CSV. Supports both stateful (Elo) and
stateless (XGBoost, LogisticRegression) model types.
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
from rich.console import Console

from ncaa_eval.cli.train import _setup_feature_server
from ncaa_eval.evaluation.bracket import MatchupContext
from ncaa_eval.evaluation.kaggle_export import KAGGLE_NEUTRAL_DAY_NUM
from ncaa_eval.evaluation.providers import EloProvider, build_probability_matrix
from ncaa_eval.ingest import ParquetRepository
from ncaa_eval.model.base import Model, StatefulModel
from ncaa_eval.model.ensemble import StackedEnsemble
from ncaa_eval.model.tracking import RunStore


def _build_stateful_predictions(
    *,
    model: StatefulModel,
    season: int,
    repo: ParquetRepository,
) -> list[tuple[int, int, int, float]]:
    """Build pairwise predictions for a stateful (Elo) model.

    Returns:
        List of ``(season, team_a_id, team_b_id, pred_win_prob)`` tuples
        for all C(n,2) team pairs where ``team_a_id < team_b_id``.
    """
    games = repo.get_games(season)
    if not games:
        msg = f"No games found for season {season}"
        raise FileNotFoundError(msg)

    team_id_set: set[int] = set()
    for g in games:
        team_id_set.add(g.w_team_id)
        team_id_set.add(g.l_team_id)
    team_ids = sorted(team_id_set)

    provider = EloProvider(model)
    context = MatchupContext(season=season, day_num=KAGGLE_NEUTRAL_DAY_NUM, is_neutral=True)
    prob_matrix = build_probability_matrix(provider, team_ids, context)

    rows: list[tuple[int, int, int, float]] = []
    n = len(team_ids)
    for i in range(n):
        for j in range(i + 1, n):
            rows.append((season, team_ids[i], team_ids[j], float(prob_matrix[i, j])))
    return rows


def _build_stateless_predictions(
    *,
    model: Model,
    run_id: str,
    season: int,
    data_dir: Path,
    store: RunStore,
) -> list[tuple[int, int, int, float]]:
    """Build game-level predictions for a stateless model.

    Returns:
        List of ``(season, team_a_id, team_b_id, pred_win_prob)`` tuples
        for all games in the season dataset.
    """
    feat_names = store.load_feature_names(run_id)
    if feat_names is None:
        msg = f"No feature names saved for run {run_id!r}"
        raise FileNotFoundError(msg)

    feature_config = model.feature_config
    server = _setup_feature_server(data_dir, feature_config)

    df = server.serve_season_features(season, mode="batch")
    if df.empty:
        msg = f"No game data found for season {season}"
        raise FileNotFoundError(msg)

    probs: pd.Series[float] = model.predict_proba(df[feat_names])

    rows: list[tuple[int, int, int, float]] = []
    for idx, prob in probs.items():
        row = df.loc[idx]
        rows.append(
            (
                season,
                int(row["team_a_id"]),
                int(row["team_b_id"]),
                float(min(max(prob, 0.0), 1.0)),
            )
        )
    return rows


def format_predictions_csv(rows: list[tuple[int, int, int, float]]) -> str:
    """Format prediction rows as a CSV string.

    Args:
        rows: List of ``(season, team_a_id, team_b_id, pred_win_prob)`` tuples.

    Returns:
        CSV string with header ``season,team_a_id,team_b_id,pred_win_prob``.
    """
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["season", "team_a_id", "team_b_id", "pred_win_prob"])
    for season, a, b, prob in rows:
        writer.writerow([season, a, b, f"{prob:.4f}"])
    return buf.getvalue()


def build_predictions(*, run_id: str, season: int, data_dir: Path) -> str:
    """Load a model and return a predictions CSV string.

    Orchestration layer: loads the model and season data from disk, routes
    to the appropriate prediction path (stateful or stateless), and formats
    the result as a CSV string.  Callers decide where to write the output.

    Args:
        run_id: Model run identifier.
        season: Target season year (e.g. 2025).
        data_dir: Path to the local data directory.

    Returns:
        CSV string with ``season,team_a_id,team_b_id,pred_win_prob`` header.

    Raises:
        FileNotFoundError: If the run, model, or season data cannot be loaded.
    """
    store = RunStore(base_path=data_dir)
    model = store.load_model(run_id)
    if model is None:
        msg = f"No model found for run {run_id!r}"
        raise FileNotFoundError(msg)

    if isinstance(model, StackedEnsemble):
        msg = "Ensemble prediction is not yet supported (Story 10.2)"
        raise NotImplementedError(msg)

    repo = ParquetRepository(base_path=data_dir)

    if isinstance(model, StatefulModel):
        rows = _build_stateful_predictions(model=model, season=season, repo=repo)
    else:
        rows = _build_stateless_predictions(
            model=model,
            run_id=run_id,
            season=season,
            data_dir=data_dir,
            store=store,
        )

    return format_predictions_csv(rows)


def run_predict(
    *,
    run_id: str,
    season: int,
    data_dir: Path,
    output: Path | None,
    console: Console | None = None,
) -> str:
    """Load a model and produce a predictions CSV.

    Thin CLI wrapper around ``build_predictions`` that handles
    progress output and writing to a file or stdout.

    Args:
        run_id: Model run identifier.
        season: Target season year (e.g. 2025).
        data_dir: Path to the local data directory.
        output: File path to write the CSV. ``None`` means stdout.
        console: Rich Console instance for status output.

    Returns:
        The CSV string.

    Raises:
        FileNotFoundError: If the run or model cannot be loaded.
        TypeError: If the model lacks a ``feature_config`` attribute
            (e.g. a malformed plugin).
        AttributeError: If the model subclass did not set ``feature_config``.
    """
    # When writing CSV to stdout, route status messages to stderr so the
    # output stream stays pipe-safe (no non-CSV lines mixed in).
    con = console or Console(stderr=output is None)
    con.print(f"Generating predictions for season {season}...")

    csv_str = build_predictions(run_id=run_id, season=season, data_dir=data_dir)

    if output is not None:
        output.write_text(csv_str)
        con.print(f"[green]Predictions written to {output}[/green]")
    else:
        sys.stdout.write(csv_str)

    return csv_str
