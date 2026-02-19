"""Sync engine for fetching NCAA data and persisting it with smart caching.

`SyncEngine` orchestrates data retrieval from configured connectors (Kaggle,
ESPN) and stores results via a `Repository`. Parquet-level caching prevents
redundant fetches on subsequent runs.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import typer

from ncaa_eval.ingest.connectors.espn import EspnConnector
from ncaa_eval.ingest.connectors.kaggle import KaggleConnector
from ncaa_eval.ingest.repository import Repository


@dataclasses.dataclass
class SyncResult:
    """Summary of a single source sync operation."""

    source: str
    teams_written: int = 0
    seasons_written: int = 0
    games_written: int = 0
    seasons_cached: int = 0


class SyncEngine:
    """Orchestrates data sync from external sources into the local repository.

    Args:
        repository: Repository instance used for reading and writing data.
        data_dir: Root directory for local Parquet files and cached CSVs.
    """

    def __init__(self, repository: Repository, data_dir: Path) -> None:
        self._repo = repository
        self._data_dir = data_dir

    def _espn_marker(self, year: int) -> Path:
        """Return the path of the ESPN sync marker file for *year*."""
        return self._data_dir / f".espn_synced_{year}"

    def sync_kaggle(self, force_refresh: bool = False) -> SyncResult:
        """Sync NCAA data from Kaggle with Parquet-level caching.

        Downloads CSVs (if not cached) and converts them to Parquet.
        Skips individual entities whose Parquet files already exist,
        unless *force_refresh* is ``True``.

        Args:
            force_refresh: Bypass all caches and re-fetch everything.

        Returns:
            SyncResult summarising teams/seasons/games written and cached.
        """
        result = SyncResult(source="kaggle")
        connector = KaggleConnector(extract_dir=self._data_dir / "kaggle")
        connector.download(force=force_refresh)  # CSV-level cache

        # Teams: Parquet-level cache
        teams_path = self._data_dir / "teams.parquet"
        if force_refresh or not teams_path.exists():
            teams = connector.fetch_teams()
            self._repo.save_teams(teams)
            result.teams_written = len(teams)
            typer.echo(f"[kaggle] teams: {len(teams)} written")
        else:
            typer.echo("[kaggle] teams: cache hit, skipped")

        # Seasons: Parquet-level cache
        seasons_path = self._data_dir / "seasons.parquet"
        if force_refresh or not seasons_path.exists():
            seasons = connector.fetch_seasons()
            self._repo.save_seasons(seasons)
            result.seasons_written = len(seasons)
            typer.echo(f"[kaggle] seasons: {len(seasons)} written")
        else:
            seasons = self._repo.get_seasons()
            typer.echo("[kaggle] seasons: cache hit, skipped")

        # Games: per-season Parquet-level cache
        for season in seasons:
            game_path = self._data_dir / "games" / f"season={season.year}" / "data.parquet"
            if not force_refresh and game_path.exists():
                result.seasons_cached += 1
                typer.echo(f"[kaggle] season {season.year}: cache hit, skipped")
                continue
            games = connector.fetch_games(season.year)
            self._repo.save_games(games)
            result.games_written += len(games)
            typer.echo(f"[kaggle] season {season.year}: {len(games)} games written")

        return result

    def sync_espn(self, force_refresh: bool = False) -> SyncResult:
        """Sync the most recent season's games from ESPN.

        Requires Kaggle data to be synced first (needs team and season
        mappings).  Uses a marker-file cache: if ``.espn_synced_{year}``
        exists the season is considered up-to-date unless *force_refresh*.

        ESPN games are merged with existing Kaggle games for the same
        season partition before saving (because ``save_games`` overwrites).

        Args:
            force_refresh: Bypass marker-file cache and re-fetch from ESPN.

        Returns:
            SyncResult summarising games written and seasons cached.

        Raises:
            RuntimeError: Kaggle data has not been synced yet.
        """
        result = SyncResult(source="espn")
        teams = self._repo.get_teams()
        seasons = self._repo.get_seasons()
        if not teams or not seasons:
            raise RuntimeError(
                "ESPN sync requires Kaggle data to be synced first. "
                "Run: python sync.py --source kaggle --dest <path>"
            )

        team_name_to_id = {t.team_name: t.team_id for t in teams}

        # Load DayZero mapping from already-downloaded Kaggle CSVs (no network call).
        kaggle_connector = KaggleConnector(extract_dir=self._data_dir / "kaggle")
        season_day_zeros = kaggle_connector.load_day_zeros()

        # ESPN scope: most recent season only
        year = max(s.year for s in seasons)

        # Cache check via marker file
        marker = self._espn_marker(year)
        if not force_refresh and marker.exists():
            result.seasons_cached += 1
            typer.echo(f"[espn] season {year}: cache hit, skipped")
            return result

        if force_refresh and marker.exists():
            marker.unlink()

        connector = EspnConnector(
            team_name_to_id=team_name_to_id,
            season_day_zeros=season_day_zeros,
        )
        espn_games = connector.fetch_games(year)

        # Merge with existing Kaggle games: save_games() overwrites the partition.
        existing_games = self._repo.get_games(year)
        all_games = existing_games + espn_games
        self._repo.save_games(all_games)
        result.games_written = len(espn_games)
        typer.echo(f"[espn] season {year}: {len(espn_games)} games written")

        # Mark season as synced
        self._data_dir.mkdir(parents=True, exist_ok=True)
        marker.touch()
        return result

    def sync_all(self, force_refresh: bool = False) -> list[SyncResult]:
        """Sync all configured sources: Kaggle first, then ESPN.

        Args:
            force_refresh: Bypass caches for all sources.

        Returns:
            List of SyncResult, one per source (kaggle, espn).
        """
        kaggle_result = self.sync_kaggle(force_refresh)
        espn_result = self.sync_espn(force_refresh)
        return [kaggle_result, espn_result]
