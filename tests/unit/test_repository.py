"""Unit tests for ncaa_eval.ingest.repository (ParquetRepository)."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from ncaa_eval.ingest.repository import ParquetRepository
from ncaa_eval.ingest.schema import Game, Season, Team

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path: Path) -> ParquetRepository:
    """Return a ParquetRepository rooted at a temporary directory."""
    return ParquetRepository(tmp_path)


def _make_team(team_id: int = 1101, team_name: str = "Abilene Chr") -> Team:
    return Team(team_id=team_id, team_name=team_name)


_GAME_DEFAULTS: dict[str, Any] = {
    "season": 2024,
    "day_num": 10,
    "w_team_id": 1101,
    "l_team_id": 1102,
    "w_score": 75,
    "l_score": 60,
    "loc": "H",
}


def _make_game(**overrides: Any) -> Game:
    kw: dict[str, Any] = {**_GAME_DEFAULTS, **overrides}
    kw.setdefault("game_id", f"{kw['season']}_{kw['day_num']}_{kw['w_team_id']}_{kw['l_team_id']}")
    return Game(**kw)


# ---------------------------------------------------------------------------
# Teams round-trip  (subtask 6.1)
# ---------------------------------------------------------------------------


class TestTeamRoundTrip:
    """Tests for save_teams -> get_teams."""

    @pytest.mark.smoke
    def test_round_trip_single(self, repo: ParquetRepository) -> None:
        teams = [_make_team()]
        repo.save_teams(teams)
        result = repo.get_teams()
        assert len(result) == 1
        assert result[0] == teams[0]

    def test_round_trip_multiple(self, repo: ParquetRepository) -> None:
        teams = [_make_team(1101, "Abilene Chr"), _make_team(1102, "Air Force")]
        repo.save_teams(teams)
        result = repo.get_teams()
        assert len(result) == 2
        assert {t.team_id for t in result} == {1101, 1102}

    def test_get_teams_empty(self, repo: ParquetRepository) -> None:
        assert repo.get_teams() == []

    def test_overwrite(self, repo: ParquetRepository) -> None:
        repo.save_teams([_make_team(1101, "Old Name")])
        repo.save_teams([_make_team(1101, "New Name")])
        result = repo.get_teams()
        assert len(result) == 1
        assert result[0].team_name == "New Name"

    def test_save_empty_is_noop(self, repo: ParquetRepository) -> None:
        repo.save_teams([_make_team()])
        repo.save_teams([])
        result = repo.get_teams()
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Games round-trip — single season  (subtask 6.2)
# ---------------------------------------------------------------------------


class TestGameRoundTripSingle:
    """Tests for save_games -> get_games with a single season."""

    @pytest.mark.smoke
    def test_round_trip(self, repo: ParquetRepository) -> None:
        games = [_make_game()]
        repo.save_games(games)
        result = repo.get_games(2024)
        assert len(result) == 1
        assert result[0].game_id == games[0].game_id
        assert result[0].w_score == 75
        assert result[0].l_score == 60
        assert result[0].date is None

    def test_round_trip_with_date(self, repo: ParquetRepository) -> None:
        game = Game(
            game_id="2024_132_1101_1102",
            season=2024,
            day_num=132,
            date=datetime.date(2024, 3, 21),
            w_team_id=1101,
            l_team_id=1102,
            w_score=85,
            l_score=70,
            loc="N",
            num_ot=2,
            is_tournament=True,
        )
        repo.save_games([game])
        result = repo.get_games(2024)
        assert len(result) == 1
        assert result[0].date == datetime.date(2024, 3, 21)
        assert result[0].num_ot == 2
        assert result[0].is_tournament is True


# ---------------------------------------------------------------------------
# Games round-trip — multiple seasons  (subtask 6.3)
# ---------------------------------------------------------------------------


class TestGameRoundTripMulti:
    """Tests for save_games -> get_games with multiple seasons."""

    def test_partition_isolation(self, repo: ParquetRepository) -> None:
        games = [
            _make_game(season=2023, w_team_id=1101, l_team_id=1102),
            _make_game(season=2024, w_team_id=1103, l_team_id=1104),
        ]
        repo.save_games(games)

        result_2023 = repo.get_games(2023)
        result_2024 = repo.get_games(2024)

        assert len(result_2023) == 1
        assert result_2023[0].season == 2023
        assert len(result_2024) == 1
        assert result_2024[0].season == 2024

    def test_multiple_games_per_season(self, repo: ParquetRepository) -> None:
        games = [
            _make_game(season=2024, day_num=10, w_team_id=1101, l_team_id=1102),
            _make_game(season=2024, day_num=11, w_team_id=1103, l_team_id=1104),
        ]
        repo.save_games(games)
        result = repo.get_games(2024)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Nonexistent season returns empty  (subtask 6.4)
# ---------------------------------------------------------------------------


class TestGameNonexistent:
    """Tests for get_games when no data exists."""

    def test_nonexistent_season_empty(self, repo: ParquetRepository) -> None:
        repo.save_games([_make_game(season=2024)])
        assert repo.get_games(1999) == []

    def test_no_games_dir_empty(self, repo: ParquetRepository) -> None:
        assert repo.get_games(2024) == []


# ---------------------------------------------------------------------------
# Overwrite semantics  (subtask 6.5)
# ---------------------------------------------------------------------------


class TestGameOverwrite:
    """Tests for save_games overwrite behaviour."""

    def test_overwrite_season(self, repo: ParquetRepository) -> None:
        repo.save_games([_make_game(season=2024, w_score=75)])
        repo.save_games([_make_game(season=2024, w_score=80)])
        result = repo.get_games(2024)
        assert len(result) == 1
        assert result[0].w_score == 80

    def test_save_empty_is_noop(self, repo: ParquetRepository) -> None:
        repo.save_games([_make_game(season=2024)])
        repo.save_games([])
        result = repo.get_games(2024)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Directory auto-creation  (subtask 6.6)
# ---------------------------------------------------------------------------


class TestDirectoryCreation:
    """Tests that the repository creates directories automatically."""

    def test_teams_creates_dir(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        repo = ParquetRepository(nested)
        repo.save_teams([_make_team()])
        assert (nested / "teams.parquet").exists()

    def test_games_creates_dir(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        repo = ParquetRepository(nested)
        repo.save_games([_make_game()])
        assert (nested / "games" / "season=2024" / "data.parquet").exists()

    def test_seasons_creates_dir(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        repo = ParquetRepository(nested)
        repo.save_seasons([Season(year=2024)])
        assert (nested / "seasons.parquet").exists()


# ---------------------------------------------------------------------------
# Seasons round-trip
# ---------------------------------------------------------------------------


class TestSeasonRoundTrip:
    """Tests for save_seasons -> get_seasons."""

    @pytest.mark.smoke
    def test_round_trip(self, repo: ParquetRepository) -> None:
        seasons = [Season(year=2023), Season(year=2024)]
        repo.save_seasons(seasons)
        result = repo.get_seasons()
        assert len(result) == 2
        assert {s.year for s in result} == {2023, 2024}

    def test_get_seasons_empty(self, repo: ParquetRepository) -> None:
        assert repo.get_seasons() == []

    def test_overwrite(self, repo: ParquetRepository) -> None:
        repo.save_seasons([Season(year=2023)])
        repo.save_seasons([Season(year=2024)])
        result = repo.get_seasons()
        assert len(result) == 1
        assert result[0].year == 2024

    def test_save_empty_is_noop(self, repo: ParquetRepository) -> None:
        repo.save_seasons([Season(year=2023)])
        repo.save_seasons([])
        result = repo.get_seasons()
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Schema evolution  (Task 3.6)
# ---------------------------------------------------------------------------


class TestSchemaEvolution:
    """Tests that older Parquet files missing new columns are still readable.

    Scenario: after a schema update adds ``num_ot`` and ``is_tournament``,
    season partitions written *before* the update lack those columns.  When
    pyarrow reads the dataset it unifies schemas across all partitions and fills
    the missing cells with null.  The repository must convert those nulls back
    to the Pydantic model defaults rather than passing null into a non-nullable
    field (which would raise a ValidationError).
    """

    def test_old_partition_readable_in_mixed_schema_dataset(self, tmp_path: Path) -> None:
        """Old-schema partition is readable when a newer partition exists alongside it."""
        repo = ParquetRepository(tmp_path)
        games_dir = tmp_path / "games"

        # Write a current-schema partition so pyarrow has both schemas in the dataset.
        repo.save_games([_make_game(season=2024)])

        # Simulate an old-format partition (season=2022) written before num_ot and
        # is_tournament were added to the schema.
        old_schema = pa.schema(
            [
                ("game_id", pa.string()),
                ("day_num", pa.int64()),
                ("date", pa.date32()),
                ("w_team_id", pa.int64()),
                ("l_team_id", pa.int64()),
                ("w_score", pa.int64()),
                ("l_score", pa.int64()),
                ("loc", pa.string()),
            ]
        )
        partition_dir = games_dir / "season=2022"
        partition_dir.mkdir(parents=True, exist_ok=True)
        old_table = pa.Table.from_pydict(
            {
                "game_id": ["2022_10_1101_1102"],
                "day_num": [10],
                "date": [None],
                "w_team_id": [1101],
                "l_team_id": [1102],
                "w_score": [70],
                "l_score": [65],
                "loc": ["H"],
            },
            schema=old_schema,
        )
        pq.write_table(old_table, partition_dir / "data.parquet")

        # Reading the old partition via the dataset API (which unifies schemas across
        # all partitions, filling missing cells with null) must succeed and apply the
        # Pydantic model defaults for the missing columns.
        games = repo.get_games(2022)
        assert len(games) == 1
        assert games[0].season == 2022
        assert games[0].num_ot == 0  # default applied, not null
        assert games[0].is_tournament is False  # default applied, not null
