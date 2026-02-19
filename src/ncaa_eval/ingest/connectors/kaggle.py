"""Kaggle data source connector for NCAA March Madness competition data.

Downloads and parses CSV files from the Kaggle March Machine Learning Mania
competition.  The ``download()`` method handles the network-dependent download
step while the ``fetch_*()`` methods perform pure CSV parsing, making it
straightforward to test without network access.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Literal, cast

import pandas as pd  # type: ignore[import-untyped]

from ncaa_eval.ingest.connectors.base import (
    AuthenticationError,
    Connector,
    ConnectorError,
    DataFormatError,
    NetworkError,
)
from ncaa_eval.ingest.schema import Game, Season, Team

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expected CSV columns (used for schema validation)
# ---------------------------------------------------------------------------

_TEAMS_COLUMNS = {"TeamID", "TeamName"}
_REGULAR_SEASON_COLUMNS = {
    "Season",
    "DayNum",
    "WTeamID",
    "LTeamID",
    "WScore",
    "LScore",
    "WLoc",
    "NumOT",
}
_TOURNEY_COLUMNS = _REGULAR_SEASON_COLUMNS
_SEASONS_COLUMNS = {"Season", "DayZero"}


def _validate_columns(df: pd.DataFrame, expected: set[str], filename: str) -> None:
    """Raise :class:`DataFormatError` if *df* is missing required columns."""
    missing = expected - set(df.columns)
    if missing:
        msg = f"kaggle: {filename} missing columns: {sorted(missing)}"
        raise DataFormatError(msg)


# ---------------------------------------------------------------------------
# KaggleConnector
# ---------------------------------------------------------------------------


class KaggleConnector(Connector):
    """Connector for Kaggle March Machine Learning Mania competition data.

    Args:
        extract_dir: Local directory where CSV files are downloaded/extracted.
        competition: Kaggle competition slug.
    """

    def __init__(
        self,
        extract_dir: Path,
        competition: str = "march-machine-learning-mania-2025",
    ) -> None:
        self._extract_dir = extract_dir
        self._competition = competition
        # Cache DayZero mapping {season_year: date} once loaded.
        self._day_zeros: dict[int, datetime.date] | None = None

    # -- network step -------------------------------------------------------

    def download(self, *, force: bool = False) -> None:
        """Download and extract competition CSV files via the Kaggle API.

        Args:
            force: Re-download even if files already exist.

        Raises:
            AuthenticationError: Credentials missing or invalid.
            NetworkError: Download failed due to connection issues.
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore[import-untyped]
        except ImportError as exc:
            msg = "kaggle: the 'kaggle' package is required. Install it with: pip install kaggle"
            raise ConnectorError(msg) from exc

        api = KaggleApi()
        try:
            api.authenticate()
        except Exception as exc:
            msg = (
                "kaggle: credentials not found. "
                "Create ~/.kaggle/kaggle.json or set KAGGLE_USERNAME/KAGGLE_KEY environment variables."
            )
            raise AuthenticationError(msg) from exc

        self._extract_dir.mkdir(parents=True, exist_ok=True)
        try:
            api.competition_download_files(
                self._competition,
                path=str(self._extract_dir),
                force=force,
            )
        except Exception as exc:
            msg = f"kaggle: failed to download competition '{self._competition}': {exc}"
            raise NetworkError(msg) from exc

    # -- CSV loading helpers ------------------------------------------------

    def _read_csv(self, filename: str) -> pd.DataFrame:
        """Read a CSV file from the extract directory.

        Raises:
            DataFormatError: File not found or unreadable.
        """
        path = self._extract_dir / filename
        if not path.exists():
            msg = f"kaggle: file not found: {path}"
            raise DataFormatError(msg)
        try:
            df: pd.DataFrame = pd.read_csv(path)
        except Exception as exc:
            msg = f"kaggle: failed to parse {filename}: {exc}"
            raise DataFormatError(msg) from exc
        return df

    def load_day_zeros(self) -> dict[int, datetime.date]:
        """Load and cache the season → DayZero mapping.

        Returns:
            Mapping of season year to the date of Day 0 for that season.
        """
        if self._day_zeros is not None:
            return self._day_zeros
        df = self._read_csv("MSeasons.csv")
        _validate_columns(df, _SEASONS_COLUMNS, "MSeasons.csv")
        mapping: dict[int, datetime.date] = {}
        for _, row in df.iterrows():
            mapping[int(row["Season"])] = datetime.date.fromisoformat(str(row["DayZero"]))
        self._day_zeros = mapping
        return mapping

    # -- Connector interface ------------------------------------------------

    def fetch_teams(self) -> list[Team]:
        """Parse ``MTeams.csv`` into Team models."""
        df = self._read_csv("MTeams.csv")
        _validate_columns(df, _TEAMS_COLUMNS, "MTeams.csv")
        return [Team(team_id=int(row["TeamID"]), team_name=str(row["TeamName"])) for _, row in df.iterrows()]

    def fetch_games(self, season: int) -> list[Game]:
        """Parse regular-season and tournament CSVs into Game models.

        Games from ``MRegularSeasonCompactResults.csv`` have
        ``is_tournament=False``; games from ``MNCAATourneyCompactResults.csv``
        have ``is_tournament=True``.
        """
        day_zeros = self.load_day_zeros()
        games: list[Game] = []
        games.extend(
            self._parse_games_csv("MRegularSeasonCompactResults.csv", season, day_zeros, is_tournament=False)
        )
        games.extend(
            self._parse_games_csv("MNCAATourneyCompactResults.csv", season, day_zeros, is_tournament=True)
        )
        return games

    def fetch_seasons(self) -> list[Season]:
        """Parse ``MSeasons.csv`` into Season models."""
        df = self._read_csv("MSeasons.csv")
        _validate_columns(df, _SEASONS_COLUMNS, "MSeasons.csv")
        return [Season(year=int(row["Season"])) for _, row in df.iterrows()]

    # -- internal parsing ---------------------------------------------------

    def _parse_games_csv(
        self,
        filename: str,
        season: int,
        day_zeros: dict[int, datetime.date],
        *,
        is_tournament: bool,
    ) -> list[Game]:
        """Parse a single games CSV, filtering to *season*."""
        df = self._read_csv(filename)
        _validate_columns(df, _REGULAR_SEASON_COLUMNS, filename)
        df = df[df["Season"] == season]
        games: list[Game] = []
        for _, row in df.iterrows():
            s = int(row["Season"])
            day_num = int(row["DayNum"])
            w_team_id = int(row["WTeamID"])
            l_team_id = int(row["LTeamID"])

            game_date: datetime.date | None = None
            dz = day_zeros.get(s)
            if dz is not None:
                game_date = dz + datetime.timedelta(days=day_num)

            wloc = str(row["WLoc"])
            if wloc not in ("H", "A", "N"):
                msg = f"kaggle: {filename} has unexpected WLoc value: {wloc!r}"
                raise DataFormatError(msg)

            games.append(
                Game(
                    game_id=f"{s}_{day_num}_{w_team_id}_{l_team_id}",
                    season=s,
                    day_num=day_num,
                    date=game_date,
                    w_team_id=w_team_id,
                    l_team_id=l_team_id,
                    w_score=int(row["WScore"]),
                    l_score=int(row["LScore"]),
                    loc=cast("Literal['H', 'A', 'N']", wloc),
                    num_ot=int(row["NumOT"]),
                    is_tournament=is_tournament,
                ),
            )
        return games
