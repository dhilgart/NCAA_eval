"""Repository pattern for NCAA basketball data storage.

Defines an abstract ``Repository`` interface and a concrete
``ParquetRepository`` implementation backed by Apache Parquet files.
The abstraction lets downstream code remain storage-agnostic — a SQLite
implementation can be plugged in later (Story 5.5) without changing any
business logic.
"""

from __future__ import annotations

import abc
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.dataset as ds  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from ncaa_eval.ingest.schema import Game, Season, Team

# ---------------------------------------------------------------------------
# Abstract Repository
# ---------------------------------------------------------------------------


class Repository(abc.ABC):
    """Abstract base class for NCAA data persistence."""

    @abc.abstractmethod
    def get_teams(self) -> list[Team]:
        """Return all stored teams."""

    @abc.abstractmethod
    def get_games(self, season: int) -> list[Game]:
        """Return all games for a given *season* year."""

    @abc.abstractmethod
    def get_seasons(self) -> list[Season]:
        """Return all stored seasons."""

    @abc.abstractmethod
    def save_teams(self, teams: list[Team]) -> None:
        """Persist a collection of teams (overwrite)."""

    @abc.abstractmethod
    def save_games(self, games: list[Game]) -> None:
        """Persist a collection of games (overwrite per season partition)."""

    @abc.abstractmethod
    def save_seasons(self, seasons: list[Season]) -> None:
        """Persist a collection of seasons (overwrite)."""


# ---------------------------------------------------------------------------
# Parquet Repository
# ---------------------------------------------------------------------------

# Explicit PyArrow schemas for deterministic column types across reads/writes.

_TEAM_SCHEMA = pa.schema([
    ("team_id", pa.int64()),
    ("team_name", pa.string()),
    ("canonical_name", pa.string()),
])

_SEASON_SCHEMA = pa.schema([
    ("year", pa.int64()),
])

_GAME_SCHEMA = pa.schema([
    ("game_id", pa.string()),
    ("season", pa.int64()),
    ("day_num", pa.int64()),
    ("date", pa.date32()),
    ("w_team_id", pa.int64()),
    ("l_team_id", pa.int64()),
    ("w_score", pa.int64()),
    ("l_score", pa.int64()),
    ("loc", pa.string()),
    ("num_ot", pa.int64()),
    ("is_tournament", pa.bool_()),
])


class ParquetRepository(Repository):
    """Repository implementation backed by Parquet files.

    Directory layout::

        {base_path}/
            teams.parquet
            seasons.parquet
            games/
                season={year}/
                    data.parquet
    """

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    # -- reads ---------------------------------------------------------------

    def get_teams(self) -> list[Team]:
        path = self._base_path / "teams.parquet"
        if not path.exists():
            return []
        df = pd.read_parquet(path, engine="pyarrow")
        return [Team(**row) for row in df.to_dict(orient="records")]

    def get_games(self, season: int) -> list[Game]:
        games_dir = self._base_path / "games"
        if not games_dir.exists():
            return []

        dataset = ds.dataset(
            games_dir,
            format="parquet",
            partitioning=ds.partitioning(pa.schema([("season", pa.int64())]), flavor="hive"),
        )
        table = dataset.to_table(filter=ds.field("season") == season)
        if table.num_rows == 0:
            return []

        df = table.to_pandas()
        # Schema evolution: when the dataset spans partitions with different schemas
        # (e.g., older season files lack columns added later), pyarrow fills missing
        # cells with null after unifying schemas.  Re-apply Pydantic defaults for
        # non-nullable Game fields so model construction doesn't fail on null input.
        if "num_ot" in df.columns:
            df["num_ot"] = df["num_ot"].fillna(0)
        if "is_tournament" in df.columns:
            df["is_tournament"] = df["is_tournament"].fillna(value=False)
        return [Game(**row) for row in df.to_dict(orient="records")]

    def get_seasons(self) -> list[Season]:
        path = self._base_path / "seasons.parquet"
        if not path.exists():
            return []
        df = pd.read_parquet(path, engine="pyarrow")
        return [Season(**row) for row in df.to_dict(orient="records")]

    # -- writes --------------------------------------------------------------

    def save_teams(self, teams: list[Team]) -> None:
        if not teams:
            return
        self._base_path.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pydict(
            {field: [getattr(t, field) for t in teams] for field in _TEAM_SCHEMA.names},
            schema=_TEAM_SCHEMA,
        )
        pq.write_table(table, self._base_path / "teams.parquet")

    def save_games(self, games: list[Game]) -> None:
        if not games:
            return

        games_dir = self._base_path / "games"

        # Group games by season for partitioned writes.
        seasons: dict[int, list[Game]] = {}
        for g in games:
            seasons.setdefault(g.season, []).append(g)

        for season_year, season_games in seasons.items():
            partition_dir = games_dir / f"season={season_year}"
            partition_dir.mkdir(parents=True, exist_ok=True)

            # Build a schema without the partition column (pyarrow hive
            # partitioning stores it in the directory name).
            write_schema = pa.schema([f for f in _GAME_SCHEMA if f.name != "season"])
            data = {
                field.name: [getattr(g, field.name) for g in season_games]
                for field in write_schema
            }
            table = pa.Table.from_pydict(data, schema=write_schema)
            pq.write_table(table, partition_dir / "data.parquet")

    def save_seasons(self, seasons: list[Season]) -> None:
        if not seasons:
            return
        self._base_path.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pydict(
            {field: [getattr(s, field) for s in seasons] for field in _SEASON_SCHEMA.names},
            schema=_SEASON_SCHEMA,
        )
        pq.write_table(table, self._base_path / "seasons.parquet")
