"""NCAA_eval data sync CLI.

Fetches NCAA basketball data from external sources (Kaggle, ESPN) and
persists it to a local Parquet store with smart caching.

Usage:
    python sync.py --source kaggle --dest data/
    python sync.py --source espn   --dest data/
    python sync.py --source all    --dest data/
    python sync.py --source all    --dest data/ --force-refresh
"""

from __future__ import annotations

import time
from pathlib import Path

import typer

from ncaa_eval.ingest import ParquetRepository, SyncEngine
from ncaa_eval.ingest.connectors import ConnectorError

app = typer.Typer(help="NCAA_eval data sync command")

VALID_SOURCES = ("kaggle", "espn", "all")


@app.command()
def main(
    source: str = typer.Option("all", help="Source to sync: kaggle | espn | all"),
    dest: Path = typer.Option(Path("data/"), help="Local data directory path"),
    force_refresh: bool = typer.Option(
        False,
        "--force-refresh",
        help="Bypass cache and re-fetch all data",
    ),
) -> None:
    """Fetch NCAA data from external sources and persist to local store."""
    if source not in VALID_SOURCES:
        typer.echo(f"Error: --source must be one of: {', '.join(VALID_SOURCES)}", err=True)
        raise typer.Exit(code=1)

    dest.mkdir(parents=True, exist_ok=True)
    repo = ParquetRepository(base_path=dest)
    engine = SyncEngine(repository=repo, data_dir=dest)

    start = time.monotonic()
    try:
        if source == "kaggle":
            results = [engine.sync_kaggle(force_refresh=force_refresh)]
        elif source == "espn":
            results = [engine.sync_espn(force_refresh=force_refresh)]
        else:
            results = engine.sync_all(force_refresh=force_refresh)
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    except ConnectorError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    elapsed = time.monotonic() - start
    total_teams = sum(r.teams_written for r in results)
    total_seasons = sum(r.seasons_written for r in results)
    total_games = sum(r.games_written for r in results)
    total_cached = sum(r.seasons_cached for r in results)

    typer.echo(
        f"\nSync complete in {elapsed:.1f}s — "
        f"teams: {total_teams}, seasons: {total_seasons}, "
        f"games: {total_games}, cache hits: {total_cached}"
    )


if __name__ == "__main__":
    app()
