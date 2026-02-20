"""Sync engine for fetching NCAA data and persisting it with smart caching.

`SyncEngine` orchestrates data retrieval from configured connectors (Kaggle,
ESPN) and stores results via a `Repository`. Parquet-level caching prevents
redundant fetches on subsequent runs.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import typer
from rapidfuzz import fuzz

from ncaa_eval.ingest.connectors.espn import EspnConnector
from ncaa_eval.ingest.connectors.kaggle import KaggleConnector
from ncaa_eval.ingest.repository import Repository

logger = logging.getLogger(__name__)

# Minimum fuzzy-match score when resolving ESPN locations not found in spellings.
_FUZZY_THRESHOLD = 80

# ESPN location names that aren't covered by MTeamSpellings.csv.  These are
# persistent ESPN abbreviations/nicknames that no generic fuzzy algorithm
# can safely resolve without false positives (e.g. "App State" would match
# "Iowa State" at 88 via partial_ratio; "minnesota" would match at 100 for
# "St. Thomas-Minnesota").  Add entries here when the sync log reports
# unmatched ESPN locations.
_ESPN_LOCATION_OVERRIDES: dict[str, int] = {
    "App State": 1111,  # ESPN abbrev.; Kaggle: "Appalachian St"
    "UL Monroe": 1419,  # ESPN abbrev.; Kaggle: "ULM"
    "St. Thomas-Minnesota": 1472,  # ESPN hyphenated; Kaggle: "St Thomas MN"
}


def _build_espn_team_map(year: int, spellings: dict[str, int]) -> dict[str, int]:
    """Build ESPN location-name → Kaggle TeamID mapping via cbbpy's bundled team map.

    cbbpy ships a `mens_team_map.csv` that lists every D-I team per season
    with the `location` name ESPN uses internally (e.g. `"UC Santa Barbara"`,
    `"Florida Gulf Coast"`).  Using these location names as the keys in
    `team_name_to_id` means:

    * `_fetch_per_team` queries cbbpy with *exact* ESPN names → no wrong
      fuzzy match (avoids `"california-santa-barbara"` → `"California"`).
    * The schedule DataFrame's `team`/`opponent` columns also use these
      location names → `_resolve_team_id` can do direct dict lookups.

    Each ESPN location is resolved to a Kaggle ID by exact lookup in the
    Kaggle spellings dict (lowercased).  A token-set-ratio fuzzy fallback
    handles any locations not covered by the spellings.

    Falls back to the latest available season in the map if *year* is absent
    (e.g. cbbpy hasn't published the current season's map yet).
    """
    import cbbpy  # type: ignore[import-untyped]  # local import; no stubs

    map_path = Path(cbbpy.__file__).parent / "utils" / "mens_team_map.csv"
    df: pd.DataFrame = pd.read_csv(map_path)

    available: set[int] = set(int(s) for s in df["season"].unique())
    if year not in available:
        fallback = max(available)
        logger.info("espn: season %d not in cbbpy team map; using %d", year, fallback)
        year = fallback

    season_df: pd.DataFrame = df[df["season"] == year]
    locations: list[str] = season_df["location"].astype(str).tolist()
    result: dict[str, int] = {}
    unmatched: list[str] = []

    for location in locations:
        # Explicit overrides take priority (handles ESPN abbreviations that
        # can't be safely resolved by generic fuzzy matching).
        if location in _ESPN_LOCATION_OVERRIDES:
            result[location] = _ESPN_LOCATION_OVERRIDES[location]
            continue
        kaggle_id: int | None = spellings.get(location.lower())
        if kaggle_id is not None:
            result[location] = kaggle_id
            continue
        # Fuzzy fallback for locations not covered by the spellings file.
        best_score = 0.0
        best_id: int | None = None
        for spelling, tid in spellings.items():
            score = float(fuzz.token_set_ratio(location.lower(), spelling))
            if score > best_score:
                best_score = score
                best_id = tid
        if best_score >= _FUZZY_THRESHOLD and best_id is not None:
            result[location] = best_id
        else:
            unmatched.append(location)

    if unmatched:
        logger.warning(
            "espn: %d ESPN locations could not be matched to a Kaggle team: %s%s",
            len(unmatched),
            unmatched[:5],
            " ..." if len(unmatched) > 5 else "",
        )

    return result


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

        # Load DayZero mapping and alternate spellings from already-downloaded Kaggle CSVs.
        kaggle_connector = KaggleConnector(extract_dir=self._data_dir / "kaggle")
        season_day_zeros = kaggle_connector.load_day_zeros()
        spellings = kaggle_connector.fetch_team_spellings()

        # ESPN scope: most recent season only
        year = max(s.year for s in seasons)

        # Build ESPN location → Kaggle ID mapping using cbbpy's authoritative
        # team list.  This ensures _fetch_per_team passes exact ESPN location
        # names to cbbpy (no wrong internal fuzzy matches).
        team_name_to_id = _build_espn_team_map(year, spellings)

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
