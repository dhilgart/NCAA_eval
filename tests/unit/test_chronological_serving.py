"""Unit tests for ncaa_eval.transform.serving (ChronologicalDataServer)."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import pytest

from ncaa_eval.ingest.repository import ParquetRepository
from ncaa_eval.ingest.schema import Game
from ncaa_eval.transform.serving import (
    ChronologicalDataServer,
    SeasonGames,
    rescale_overtime,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

_GAME_DEFAULTS: dict[str, Any] = {
    "season": 2024,
    "day_num": 10,
    "date": datetime.date(2024, 1, 15),
    "w_team_id": 1101,
    "l_team_id": 1102,
    "w_score": 75,
    "l_score": 60,
    "loc": "H",
}


def _make_game(**overrides: Any) -> Game:
    kw: dict[str, Any] = {**_GAME_DEFAULTS, **overrides}
    kw.setdefault(
        "game_id",
        f"{kw['season']}_{kw['day_num']}_{kw['w_team_id']}_{kw['l_team_id']}",
    )
    return Game(**kw)


@pytest.fixture
def repo(tmp_path: Path) -> ParquetRepository:
    """Return a ParquetRepository rooted at a temporary directory."""
    return ParquetRepository(tmp_path)


@pytest.fixture
def server(repo: ParquetRepository) -> ChronologicalDataServer:
    """Return a ChronologicalDataServer backed by the temp repo."""
    return ChronologicalDataServer(repo)


# ---------------------------------------------------------------------------
# OT rescaling (subtasks 6.9, 6.10)
# ---------------------------------------------------------------------------


class TestRescaleOvertime:
    """Tests for the rescale_overtime module-level function."""

    @pytest.mark.smoke
    @pytest.mark.unit
    def test_regulation_no_change(self) -> None:
        """rescale_overtime with num_ot=0 returns score unchanged as float (6.9)."""
        assert rescale_overtime(75, 0) == 75.0

    @pytest.mark.smoke
    @pytest.mark.unit
    def test_one_ot_penalty(self) -> None:
        """rescale_overtime with 1 OT returns 80 × 40/45 (6.10)."""
        result = rescale_overtime(80, 1)
        assert abs(result - (80 * 40.0 / 45)) < 1e-9

    @pytest.mark.unit
    def test_two_ot(self) -> None:
        """rescale_overtime with 2 OT returns score × 40/50."""
        result = rescale_overtime(100, 2)
        assert abs(result - (100 * 40.0 / 50)) < 1e-9

    @pytest.mark.unit
    def test_returns_float(self) -> None:
        """rescale_overtime always returns a float, even for regulation."""
        assert isinstance(rescale_overtime(60, 0), float)


# ---------------------------------------------------------------------------
# Chronological ordering (subtask 6.1)
# ---------------------------------------------------------------------------


class TestChronologicalOrdering:
    """Tests that games are returned in ascending date order."""

    @pytest.mark.smoke
    @pytest.mark.unit
    def test_games_sorted_ascending_by_date(
        self, repo: ParquetRepository, server: ChronologicalDataServer
    ) -> None:
        """Games are returned strictly ordered by date ascending (AC 1, 6.1)."""
        games = [
            _make_game(
                day_num=20,
                date=datetime.date(2024, 2, 1),
                w_team_id=1103,
                l_team_id=1104,
            ),
            _make_game(
                day_num=10,
                date=datetime.date(2024, 1, 15),
                w_team_id=1101,
                l_team_id=1102,
            ),
        ]
        repo.save_games(games)
        result = server.get_chronological_season(2024)
        assert len(result.games) == 2
        assert result.games[0].date == datetime.date(2024, 1, 15)
        assert result.games[1].date == datetime.date(2024, 2, 1)

    @pytest.mark.unit
    def test_deterministic_order_same_date(
        self, repo: ParquetRepository, server: ChronologicalDataServer
    ) -> None:
        """Same-day games are ordered deterministically by game_id."""
        games = [
            _make_game(
                game_id="2024_10_1103_1104",
                day_num=10,
                date=datetime.date(2024, 1, 15),
                w_team_id=1103,
                l_team_id=1104,
            ),
            _make_game(
                game_id="2024_10_1101_1102",
                day_num=10,
                date=datetime.date(2024, 1, 15),
                w_team_id=1101,
                l_team_id=1102,
            ),
        ]
        repo.save_games(games)
        result = server.get_chronological_season(2024)
        assert result.games[0].game_id < result.games[1].game_id


# ---------------------------------------------------------------------------
# Temporal boundary enforcement (subtasks 6.2, 6.3)
# ---------------------------------------------------------------------------


class TestCutoffDate:
    """Tests for cutoff_date temporal boundary enforcement (AC 2, 3)."""

    @pytest.mark.smoke
    @pytest.mark.unit
    def test_cutoff_excludes_later_games(
        self, repo: ParquetRepository, server: ChronologicalDataServer
    ) -> None:
        """Games after cutoff_date are excluded; games on cutoff_date included (6.2)."""
        games = [
            _make_game(
                day_num=10,
                date=datetime.date(2024, 1, 15),
                w_team_id=1101,
                l_team_id=1102,
            ),
            _make_game(
                day_num=20,
                date=datetime.date(2024, 2, 1),
                w_team_id=1103,
                l_team_id=1104,
            ),
            _make_game(
                day_num=30,
                date=datetime.date(2024, 3, 1),
                w_team_id=1105,
                l_team_id=1106,
            ),
        ]
        repo.save_games(games)
        result = server.get_chronological_season(2024, cutoff_date=datetime.date(2024, 2, 1))
        assert len(result.games) == 2
        assert all(g.date is not None and g.date <= datetime.date(2024, 2, 1) for g in result.games)

    @pytest.mark.unit
    def test_cutoff_includes_same_day_games(
        self, repo: ParquetRepository, server: ChronologicalDataServer
    ) -> None:
        """Games exactly on the cutoff date are included."""
        game = _make_game(date=datetime.date(2024, 2, 1))
        repo.save_games([game])
        result = server.get_chronological_season(2024, cutoff_date=datetime.date(2024, 2, 1))
        assert len(result.games) == 1

    @pytest.mark.unit
    def test_cutoff_none_returns_all_games(
        self, repo: ParquetRepository, server: ChronologicalDataServer
    ) -> None:
        """cutoff_date=None returns all games for the season."""
        games = [
            _make_game(day_num=10, date=datetime.date(2024, 1, 15), w_team_id=1101, l_team_id=1102),
            _make_game(day_num=20, date=datetime.date(2024, 2, 1), w_team_id=1103, l_team_id=1104),
        ]
        repo.save_games(games)
        result = server.get_chronological_season(2024)
        assert len(result.games) == 2

    @pytest.mark.smoke
    @pytest.mark.unit
    def test_future_cutoff_raises_value_error(self, server: ChronologicalDataServer) -> None:
        """Future cutoff_date raises ValueError with descriptive message (AC 3, 6.3)."""
        tomorrow = datetime.date.today() + datetime.timedelta(days=1)
        with pytest.raises(ValueError, match="Cannot request future game data"):
            server.get_chronological_season(2024, cutoff_date=tomorrow)

    @pytest.mark.unit
    def test_future_cutoff_message_includes_date(self, server: ChronologicalDataServer) -> None:
        """ValueError message includes the requested cutoff_date."""
        future = datetime.date.today() + datetime.timedelta(days=30)
        with pytest.raises(ValueError, match=str(future)):
            server.get_chronological_season(2024, cutoff_date=future)


# ---------------------------------------------------------------------------
# 2020 COVID year handling (subtask 6.4)
# ---------------------------------------------------------------------------


class TestCovidYear:
    """Tests for 2020 COVID year has_tournament flag (AC 4)."""

    @pytest.mark.smoke
    @pytest.mark.unit
    def test_2020_has_tournament_false(
        self, repo: ParquetRepository, server: ChronologicalDataServer
    ) -> None:
        """2020 season returns has_tournament=False (AC 4, 6.4)."""
        repo.save_games([_make_game(season=2020, day_num=10, date=datetime.date(2020, 1, 15))])
        result = server.get_chronological_season(2020)
        assert result.has_tournament is False

    @pytest.mark.smoke
    @pytest.mark.unit
    def test_non_2020_has_tournament_true(
        self, repo: ParquetRepository, server: ChronologicalDataServer
    ) -> None:
        """Non-2020 years return has_tournament=True (6.4)."""
        repo.save_games([_make_game(season=2024)])
        result = server.get_chronological_season(2024)
        assert result.has_tournament is True

    @pytest.mark.unit
    def test_2020_regular_season_games_returned(
        self, repo: ParquetRepository, server: ChronologicalDataServer
    ) -> None:
        """2020 regular-season games are returned normally; only the flag differs."""
        game = _make_game(season=2020, day_num=10, date=datetime.date(2020, 1, 15))
        repo.save_games([game])
        result = server.get_chronological_season(2020)
        assert len(result.games) == 1
        assert result.games[0].game_id == game.game_id


# ---------------------------------------------------------------------------
# Iterator / streaming (subtasks 6.5, 6.6)
# ---------------------------------------------------------------------------


class TestIterGamesByDate:
    """Tests for iter_games_by_date streaming interface (AC 5)."""

    @pytest.mark.unit
    def test_yields_batches_in_date_order(
        self, repo: ParquetRepository, server: ChronologicalDataServer
    ) -> None:
        """iter_games_by_date yields one batch per date in ascending order (6.5)."""
        games = [
            _make_game(
                day_num=10,
                date=datetime.date(2024, 1, 15),
                w_team_id=1101,
                l_team_id=1102,
            ),
            _make_game(
                day_num=20,
                date=datetime.date(2024, 2, 1),
                w_team_id=1103,
                l_team_id=1104,
            ),
        ]
        repo.save_games(games)
        batches = list(server.iter_games_by_date(2024))
        assert len(batches) == 2
        assert len(batches[0]) == 1
        assert batches[0][0].date == datetime.date(2024, 1, 15)
        assert len(batches[1]) == 1
        assert batches[1][0].date == datetime.date(2024, 2, 1)

    @pytest.mark.smoke
    @pytest.mark.unit
    def test_same_day_games_in_same_batch(
        self, repo: ParquetRepository, server: ChronologicalDataServer
    ) -> None:
        """Multiple games on the same date appear in the same yielded batch (6.6)."""
        games = [
            _make_game(
                day_num=10,
                date=datetime.date(2024, 1, 15),
                w_team_id=1101,
                l_team_id=1102,
            ),
            _make_game(
                day_num=10,
                date=datetime.date(2024, 1, 15),
                w_team_id=1103,
                l_team_id=1104,
            ),
        ]
        repo.save_games(games)
        batches = list(server.iter_games_by_date(2024))
        assert len(batches) == 1
        assert len(batches[0]) == 2

    @pytest.mark.unit
    def test_iter_respects_cutoff_date(
        self, repo: ParquetRepository, server: ChronologicalDataServer
    ) -> None:
        """iter_games_by_date respects the same cutoff_date semantics."""
        games = [
            _make_game(day_num=10, date=datetime.date(2024, 1, 15), w_team_id=1101, l_team_id=1102),
            _make_game(day_num=20, date=datetime.date(2024, 2, 1), w_team_id=1103, l_team_id=1104),
        ]
        repo.save_games(games)
        batches = list(server.iter_games_by_date(2024, cutoff_date=datetime.date(2024, 1, 15)))
        assert len(batches) == 1
        assert batches[0][0].date == datetime.date(2024, 1, 15)

    @pytest.mark.unit
    def test_iter_empty_season_yields_nothing(self, server: ChronologicalDataServer) -> None:
        """iter_games_by_date yields nothing for an empty season."""
        batches = list(server.iter_games_by_date(2024))
        assert batches == []


# ---------------------------------------------------------------------------
# 2025 deduplication (subtasks 6.7, 6.8)
# ---------------------------------------------------------------------------


class TestDeduplication2025:
    """Tests for 2025 duplicate game elimination (AC 1, 6.7–6.8)."""

    @pytest.mark.unit
    def test_deduplication_reduces_to_single_game(
        self, repo: ParquetRepository, server: ChronologicalDataServer
    ) -> None:
        """Two records for same (w_team_id, l_team_id, day_num) → one game (6.7)."""
        kaggle_game = _make_game(
            game_id="123456",
            season=2025,
            day_num=50,
            date=datetime.date(2025, 2, 15),
            w_team_id=1101,
            l_team_id=1102,
            loc="N",
            num_ot=0,
        )
        espn_game = _make_game(
            game_id="espn_abc123",
            season=2025,
            day_num=50,
            date=datetime.date(2025, 2, 15),
            w_team_id=1101,
            l_team_id=1102,
            loc="H",
            num_ot=0,
        )
        repo.save_games([kaggle_game, espn_game])
        result = server.get_chronological_season(2025)
        assert len(result.games) == 1

    @pytest.mark.unit
    def test_deduplication_prefers_espn_loc(
        self, repo: ParquetRepository, server: ChronologicalDataServer
    ) -> None:
        """After deduplication the ESPN loc value is used (6.8)."""
        kaggle_game = _make_game(
            game_id="123456",
            season=2025,
            day_num=50,
            date=datetime.date(2025, 2, 15),
            w_team_id=1101,
            l_team_id=1102,
            loc="N",
            num_ot=0,
        )
        espn_game = _make_game(
            game_id="espn_abc123",
            season=2025,
            day_num=50,
            date=datetime.date(2025, 2, 15),
            w_team_id=1101,
            l_team_id=1102,
            loc="H",
            num_ot=1,
        )
        repo.save_games([kaggle_game, espn_game])
        result = server.get_chronological_season(2025)
        assert result.games[0].loc == "H"

    @pytest.mark.unit
    def test_deduplication_prefers_espn_num_ot(
        self, repo: ParquetRepository, server: ChronologicalDataServer
    ) -> None:
        """After deduplication the ESPN num_ot value is used (6.8)."""
        kaggle_game = _make_game(
            game_id="123456",
            season=2025,
            day_num=50,
            date=datetime.date(2025, 2, 15),
            w_team_id=1101,
            l_team_id=1102,
            loc="N",
            num_ot=0,
        )
        espn_game = _make_game(
            game_id="espn_abc123",
            season=2025,
            day_num=50,
            date=datetime.date(2025, 2, 15),
            w_team_id=1101,
            l_team_id=1102,
            loc="N",
            num_ot=2,
        )
        repo.save_games([kaggle_game, espn_game])
        result = server.get_chronological_season(2025)
        assert result.games[0].num_ot == 2

    @pytest.mark.unit
    def test_deduplication_not_applied_to_non_2025(
        self, repo: ParquetRepository, server: ChronologicalDataServer
    ) -> None:
        """Deduplication is only applied for year=2025; other years untouched."""
        # Two different games in 2024 with same team pair on different days — no dedup
        games = [
            _make_game(
                season=2024,
                day_num=10,
                date=datetime.date(2024, 1, 15),
                w_team_id=1101,
                l_team_id=1102,
            ),
            _make_game(
                season=2024,
                day_num=20,
                date=datetime.date(2024, 2, 1),
                w_team_id=1101,
                l_team_id=1102,
            ),
        ]
        repo.save_games(games)
        result = server.get_chronological_season(2024)
        assert len(result.games) == 2

    @pytest.mark.unit
    def test_deduplication_preserves_unique_games(
        self, repo: ParquetRepository, server: ChronologicalDataServer
    ) -> None:
        """Non-duplicate 2025 games are all preserved after deduplication."""
        games = [
            _make_game(
                game_id="espn_game1",
                season=2025,
                day_num=50,
                date=datetime.date(2025, 2, 15),
                w_team_id=1101,
                l_team_id=1102,
            ),
            _make_game(
                game_id="espn_game2",
                season=2025,
                day_num=51,
                date=datetime.date(2025, 2, 16),
                w_team_id=1103,
                l_team_id=1104,
            ),
        ]
        repo.save_games(games)
        result = server.get_chronological_season(2025)
        assert len(result.games) == 2


# ---------------------------------------------------------------------------
# Empty season (subtask 6.11)
# ---------------------------------------------------------------------------


class TestEmptySeason:
    """Tests for seasons with no games in the repository."""

    @pytest.mark.unit
    def test_empty_season_returns_season_games(self, server: ChronologicalDataServer) -> None:
        """Empty 2020 season returns SeasonGames with games=[] and has_tournament=False (6.11)."""
        result = server.get_chronological_season(2020)
        assert isinstance(result, SeasonGames)
        assert result.year == 2020
        assert result.games == []
        assert result.has_tournament is False

    @pytest.mark.unit
    def test_empty_non_covid_season(self, server: ChronologicalDataServer) -> None:
        """Empty non-2020 season returns games=[] with has_tournament=True."""
        result = server.get_chronological_season(2024)
        assert result.games == []
        assert result.has_tournament is True
