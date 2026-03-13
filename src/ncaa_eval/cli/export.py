"""Kaggle submission export orchestration.

Loads a trained model, builds a pairwise probability matrix for all
teams in a season, and writes a Kaggle-format CSV.
"""

from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console

from ncaa_eval.evaluation.bracket import MatchupContext
from ncaa_eval.evaluation.kaggle_export import KAGGLE_NEUTRAL_DAY_NUM, format_kaggle_submission
from ncaa_eval.evaluation.providers import EloProvider, build_probability_matrix
from ncaa_eval.ingest import ParquetRepository
from ncaa_eval.model.base import StatefulModel
from ncaa_eval.model.tracking import RunStore


def build_kaggle_submission(*, run_id: str, season: int, data_dir: Path) -> str:
    """Load a model and return a Kaggle submission CSV string.

    Pure orchestration: loads the model, collects all team IDs for the
    season, builds the all-pairs probability matrix, and formats the CSV.
    No I/O side-effects — callers decide where to write the result.

    Args:
        run_id: Model run identifier.
        season: Tournament season year (e.g. 2026).
        data_dir: Path to the local data directory.

    Returns:
        CSV string with ``ID,Pred`` header and C(n,2) data rows.

    Raises:
        FileNotFoundError: If the run, model, or season games cannot be loaded.
        TypeError: If the model type is not supported for Kaggle export.
    """
    store = RunStore(base_path=data_dir)
    model = store.load_model(run_id)
    if model is None:
        msg = f"No model found for run {run_id!r}"
        raise FileNotFoundError(msg)

    if not isinstance(model, StatefulModel):
        msg = (
            "Kaggle export currently supports stateful (Elo) models only. "
            "Stateless model export requires a feature server and is not yet implemented."
        )
        raise TypeError(msg)

    repo = ParquetRepository(base_path=data_dir)
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
    return format_kaggle_submission(season, team_ids, prob_matrix)


def run_export(
    *,
    run_id: str,
    season: int,
    data_dir: Path,
    output: Path | None,
    console: Console | None = None,
) -> str:
    """Load a model and produce a Kaggle submission CSV.

    Thin CLI wrapper around ``build_kaggle_submission`` that handles
    progress output and writing to a file or stdout.

    Args:
        run_id: Model run identifier.
        season: Tournament season year (e.g. 2026).
        data_dir: Path to the local data directory.
        output: File path to write the CSV. ``None`` means stdout.
        console: Rich Console instance for status output.

    Returns:
        The CSV string.

    Raises:
        FileNotFoundError: If the run or model cannot be loaded.
        TypeError: If the model type is not supported for export.
    """
    con = console or Console()
    con.print(f"Building Kaggle submission for season {season}...")

    csv_str = build_kaggle_submission(run_id=run_id, season=season, data_dir=data_dir)

    if output is not None:
        output.write_text(csv_str)
        con.print(f"[green]Kaggle submission written to {output}[/green]")
    else:
        # Write raw CSV to stdout (no Rich formatting) so the output is
        # pipe-safe: `ncaa_eval export ... | gzip > submission.csv.gz`
        sys.stdout.write(csv_str)

    return csv_str
