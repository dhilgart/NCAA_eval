"""Kaggle data source connector for NCAA March Madness competition data.

Downloads and parses CSV files from the Kaggle March Machine Learning Mania
competition.  The ``download()`` method handles the network-dependent download
step while the ``fetch_*()`` methods perform pure CSV parsing, making it
straightforward to test without network access.
"""

from __future__ import annotations

import datetime
import logging
import zipfile
from pathlib import Path
from typing import Literal, cast

import pandas as pd  # type: ignore[import-untyped]
import pandera.errors
import pandera.pandas as pa

# pandera.pandas (not bare pandera) avoids the FutureWarning in v0.29+.
# pandera.errors is imported separately because pandera.pandas does not
# re-export the errors sub-module.
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
# Pandera schemas for CSV validation
# ---------------------------------------------------------------------------

_TEAMS_SCHEMA = pa.DataFrameSchema(
    {
        "TeamID": pa.Column(int, pa.Check.ge(1)),
        "TeamName": pa.Column(str, nullable=False),
    },
)

_SPELLINGS_SCHEMA = pa.DataFrameSchema(
    {
        "TeamNameSpelling": pa.Column(str, nullable=False),
        "TeamID": pa.Column(int, pa.Check.ge(1)),
    },
)

_GAMES_SCHEMA = pa.DataFrameSchema(
    {
        "Season": pa.Column(int, pa.Check.ge(1985)),
        "DayNum": pa.Column(int, pa.Check.ge(0)),
        "WTeamID": pa.Column(int, pa.Check.ge(1)),
        "LTeamID": pa.Column(int, pa.Check.ge(1)),
        "WScore": pa.Column(int, pa.Check.ge(0)),
        "LScore": pa.Column(int, pa.Check.ge(0)),
        "WLoc": pa.Column(str, pa.Check.isin(["H", "A", "N"])),
        "NumOT": pa.Column(int, pa.Check.ge(0)),
    },
)

_SEASONS_SCHEMA = pa.DataFrameSchema(
    {
        "Season": pa.Column(int, pa.Check.ge(1985)),
        "DayZero": pa.Column(str, nullable=False),
    },
)


def _validate_schema(
    df: pd.DataFrame,
    schema: pa.DataFrameSchema,
    filename: str,
) -> None:
    """Validate *df* against a Pandera schema, wrapping errors in DataFormatError."""
    try:
        schema.validate(df)
    except pandera.errors.SchemaError as exc:
        raise DataFormatError(
            f"kaggle: {filename} schema validation failed: {exc}",
        ) from exc


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
                "Save your API token to ~/.kaggle/access_token (see README for setup instructions)."
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

        # kaggle 2.0.0 no longer auto-extracts the zip — unzip manually
        zip_path = self._extract_dir / f"{self._competition}.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(self._extract_dir)
            zip_path.unlink()

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
        _validate_schema(df, _SEASONS_SCHEMA, "MSeasons.csv")
        mapping: dict[int, datetime.date] = {}
        for _, row in df.iterrows():
            mapping[int(row["Season"])] = datetime.datetime.strptime(str(row["DayZero"]), "%m/%d/%Y").date()
        self._day_zeros = mapping
        return mapping

    # -- Connector interface ------------------------------------------------

    def fetch_teams(self) -> list[Team]:
        """Parse ``MTeams.csv`` into Team models.

        Reads MTeams.csv, validates required columns, then constructs Team
        models from each row's TeamID and TeamName.
        """
        df = self._read_csv("MTeams.csv")
        _validate_schema(df, _TEAMS_SCHEMA, "MTeams.csv")
        return [Team(team_id=int(row["TeamID"]), team_name=str(row["TeamName"])) for _, row in df.iterrows()]

    def fetch_team_spellings(self) -> dict[str, int]:
        """Parse ``MTeamSpellings.csv`` into a spelling → TeamID mapping.

        Returns every alternate spelling (lower-cased) for each team, which
        provides much wider coverage than the canonical names in MTeams.csv
        when resolving ESPN team name strings to Kaggle IDs.
        """
        df = self._read_csv("MTeamSpellings.csv")
        _validate_schema(df, _SPELLINGS_SCHEMA, "MTeamSpellings.csv")
        # Pandera already enforced int dtype; no .astype() needed here.
        return dict(zip(df["TeamNameSpelling"].str.lower(), df["TeamID"]))

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
        """Parse ``MSeasons.csv`` into Season models.

        Delegates to :meth:`load_day_zeros` (which already reads and validates
        MSeasons.csv) to avoid a second disk read and Pandera validation pass.
        """
        day_zeros = self.load_day_zeros()
        return [Season(year=year) for year in day_zeros]

    # -- internal parsing ---------------------------------------------------

    def _parse_games_csv(
        self,
        filename: str,
        season: int,
        day_zeros: dict[int, datetime.date],
        *,
        is_tournament: bool,
    ) -> list[Game]:
        """Parse a single games CSV, filtering to *season*.

        Reads the CSV file (regular-season or tournament), filters by season,
        iterates rows to extract team IDs and scores, computes game_date from
        day_num and the season's day-zero, validates WLoc, then builds Game
        models.
        """
        df = self._read_csv(filename)
        _validate_schema(df, _GAMES_SCHEMA, filename)
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
