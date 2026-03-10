"""Unit tests for ncaa_eval.ingest.validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ncaa_eval.ingest.repository import ParquetRepository
from ncaa_eval.ingest.schema import Game, Season, Team
from ncaa_eval.ingest.validation import (
    ValidationReport,
    ValidationResult,
    _check_duplicate_games,
    _check_game_counts,
    _check_team_references,
    validate_sync,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEAMS = [
    Team(team_id=1101, team_name="Abilene Chr"),
    Team(team_id=1102, team_name="Air Force"),
    Team(team_id=1103, team_name="Akron"),
    Team(team_id=1104, team_name="Alabama"),
]

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
    kw.setdefault(
        "game_id",
        f"{kw['season']}_{kw['day_num']}_{kw['w_team_id']}_{kw['l_team_id']}",
    )
    return Game(**kw)


def _make_n_games(season: int, n: int) -> list[Game]:
    """Create *n* unique games for a season."""
    games: list[Game] = []
    for i in range(n):
        w_id = 1101 + (i % 4)
        l_id = 1101 + ((i + 1) % 4)
        if w_id == l_id:
            l_id = 1101 + ((i + 2) % 4)
        games.append(
            _make_game(
                season=season,
                day_num=10 + i,
                w_team_id=w_id,
                l_team_id=l_id,
            )
        )
    return games


@pytest.fixture
def repo(tmp_path: Path) -> ParquetRepository:
    """Return a ParquetRepository rooted at a temporary directory."""
    return ParquetRepository(tmp_path)


# ---------------------------------------------------------------------------
# ValidationResult / ValidationReport model tests
# ---------------------------------------------------------------------------


class TestValidationModels:
    """Tests for ValidationResult and ValidationReport Pydantic models."""

    def test_result_fields(self) -> None:
        r = ValidationResult(check_name="test", passed=True, message="ok", details={"k": 1})
        assert r.check_name == "test"
        assert r.passed is True
        assert r.message == "ok"
        assert r.details == {"k": 1}

    def test_result_details_default_empty(self) -> None:
        r = ValidationResult(check_name="test", passed=True, message="ok")
        assert r.details == {}

    def test_report_all_passed_true(self) -> None:
        report = ValidationReport(
            results=[
                ValidationResult(check_name="a", passed=True, message="ok"),
                ValidationResult(check_name="b", passed=True, message="ok"),
            ]
        )
        assert report.all_passed is True

    def test_report_all_passed_false(self) -> None:
        report = ValidationReport(
            results=[
                ValidationResult(check_name="a", passed=True, message="ok"),
                ValidationResult(check_name="b", passed=False, message="bad"),
            ]
        )
        assert report.all_passed is False

    def test_report_empty_results_all_passed(self) -> None:
        report = ValidationReport(results=[])
        assert report.all_passed is True


# ---------------------------------------------------------------------------
# _check_game_counts
# ---------------------------------------------------------------------------


class TestCheckGameCounts:
    """Tests for the game count validation check."""

    def test_empty_repo(self, repo: ParquetRepository) -> None:
        results = _check_game_counts(repo)
        assert len(results) == 1
        assert results[0].passed is True
        assert "No seasons" in results[0].message

    def test_all_seasons_within_range(self, repo: ParquetRepository) -> None:
        """Multiple seasons with similar game counts should all pass."""
        repo.save_teams(_TEAMS)
        for year in (2022, 2023, 2024):
            repo.save_seasons([Season(year=y) for y in (2022, 2023, 2024)])
            repo.save_games(_make_n_games(year, 100))

        results = _check_game_counts(repo)
        assert len(results) == 1
        assert results[0].passed is True

    def test_anomalous_season_flagged(self, repo: ParquetRepository) -> None:
        """A season with far fewer games should be flagged."""
        seasons = [Season(year=y) for y in (2022, 2023, 2024)]
        repo.save_seasons(seasons)
        repo.save_teams(_TEAMS)

        # Two normal seasons, one anomalous
        repo.save_games(_make_n_games(2022, 100))
        repo.save_games(_make_n_games(2023, 100))
        repo.save_games(_make_n_games(2024, 10))  # way below median

        results = _check_game_counts(repo)
        assert len(results) == 1
        assert results[0].passed is False
        assert "2024" in str(results[0].details.get("anomalies", {}))

    def test_covid_2020_excluded_from_median(self, repo: ParquetRepository) -> None:
        """Season 2020 is excluded from median but still validated."""
        seasons = [Season(year=y) for y in (2019, 2020, 2021)]
        repo.save_seasons(seasons)
        repo.save_teams(_TEAMS)

        repo.save_games(_make_n_games(2019, 100))
        repo.save_games(_make_n_games(2020, 20))  # COVID — excluded from median
        repo.save_games(_make_n_games(2021, 100))

        results = _check_game_counts(repo)
        assert len(results) == 1
        # COVID season should be flagged as anomalous
        assert results[0].passed is False
        assert "2020" in str(results[0].details.get("anomalies", {}))
        # But 2019/2021 should NOT be flagged (they match the median)
        assert "2019" not in str(results[0].details.get("anomalies", {}))

    def test_only_covid_season(self, repo: ParquetRepository) -> None:
        """If only the COVID season exists, skip gracefully."""
        repo.save_seasons([Season(year=2020)])
        repo.save_teams(_TEAMS)
        repo.save_games(_make_n_games(2020, 50))

        results = _check_game_counts(repo)
        assert len(results) == 1
        assert results[0].passed is True
        assert "COVID" in results[0].message or "Only" in results[0].message

    def test_single_non_covid_season(self, repo: ParquetRepository) -> None:
        """Single season is its own median — always passes."""
        repo.save_seasons([Season(year=2024)])
        repo.save_teams(_TEAMS)
        repo.save_games(_make_n_games(2024, 100))

        results = _check_game_counts(repo)
        assert len(results) == 1
        assert results[0].passed is True


# ---------------------------------------------------------------------------
# _check_duplicate_games
# ---------------------------------------------------------------------------


class TestCheckDuplicateGames:
    """Tests for duplicate game detection."""

    def test_empty_repo(self, repo: ParquetRepository) -> None:
        results = _check_duplicate_games(repo)
        assert len(results) == 1
        assert results[0].passed is True

    def test_no_duplicates(self, repo: ParquetRepository) -> None:
        repo.save_seasons([Season(year=2024)])
        repo.save_games(_make_n_games(2024, 50))

        results = _check_duplicate_games(repo)
        assert len(results) == 1
        assert results[0].passed is True

    def test_duplicates_detected(self, repo: ParquetRepository) -> None:
        """Duplicate games (same season/day/teams) should be detected."""
        repo.save_seasons([Season(year=2024)])
        games = _make_n_games(2024, 5)
        # Add a duplicate of the first game with a different game_id
        dup = _make_game(
            game_id="dup_2024_10_1101_1102",
            season=2024,
            day_num=games[0].day_num,
            w_team_id=games[0].w_team_id,
            l_team_id=games[0].l_team_id,
        )
        games.append(dup)
        repo.save_games(games)

        results = _check_duplicate_games(repo)
        assert len(results) == 1
        assert results[0].passed is False
        assert results[0].details["duplicates_per_season"]["2024"] == 1

    def test_multiple_duplicates_across_seasons(self, repo: ParquetRepository) -> None:
        """Duplicates in multiple seasons are reported per season."""
        repo.save_seasons([Season(year=2023), Season(year=2024)])
        repo.save_teams(_TEAMS)

        # 2023: one duplicate
        games_2023 = _make_n_games(2023, 3)
        games_2023.append(
            _make_game(
                game_id="dup_2023",
                season=2023,
                day_num=games_2023[0].day_num,
                w_team_id=games_2023[0].w_team_id,
                l_team_id=games_2023[0].l_team_id,
            )
        )
        repo.save_games(games_2023)

        # 2024: two duplicates of same game (triple = 2 extra)
        games_2024 = _make_n_games(2024, 3)
        for i in range(2):
            games_2024.append(
                _make_game(
                    game_id=f"dup_2024_{i}",
                    season=2024,
                    day_num=games_2024[0].day_num,
                    w_team_id=games_2024[0].w_team_id,
                    l_team_id=games_2024[0].l_team_id,
                )
            )
        repo.save_games(games_2024)

        results = _check_duplicate_games(repo)
        assert results[0].passed is False
        per_season = results[0].details["duplicates_per_season"]
        assert per_season["2023"] == 1
        assert per_season["2024"] == 2


# ---------------------------------------------------------------------------
# _check_team_references
# ---------------------------------------------------------------------------


class TestCheckTeamReferences:
    """Tests for team reference integrity check."""

    def test_empty_repo(self, repo: ParquetRepository) -> None:
        results = _check_team_references(repo)
        assert len(results) == 1
        assert results[0].passed is True

    def test_all_references_valid(self, repo: ParquetRepository) -> None:
        repo.save_teams(_TEAMS)
        repo.save_seasons([Season(year=2024)])
        repo.save_games(_make_n_games(2024, 10))

        results = _check_team_references(repo)
        assert len(results) == 1
        assert results[0].passed is True

    def test_orphan_team_id_detected(self, repo: ParquetRepository) -> None:
        """A game referencing a non-existent team ID should be flagged."""
        repo.save_teams(_TEAMS)  # team IDs 1101-1104
        repo.save_seasons([Season(year=2024)])

        games = _make_n_games(2024, 3)
        # Add a game with an unknown team ID
        games.append(
            _make_game(
                game_id="orphan_game",
                season=2024,
                day_num=99,
                w_team_id=9999,  # unknown
                l_team_id=1101,
            )
        )
        repo.save_games(games)

        results = _check_team_references(repo)
        assert len(results) == 1
        assert results[0].passed is False
        orphans = results[0].details["orphans_per_season"]["2024"]
        assert 9999 in orphans

    def test_orphan_in_losing_team(self, repo: ParquetRepository) -> None:
        """Orphan in l_team_id is also detected."""
        repo.save_teams(_TEAMS)
        repo.save_seasons([Season(year=2024)])

        games = [
            _make_game(
                game_id="orphan_l",
                season=2024,
                day_num=99,
                w_team_id=1101,
                l_team_id=8888,  # unknown
            )
        ]
        repo.save_games(games)

        results = _check_team_references(repo)
        assert results[0].passed is False
        assert 8888 in results[0].details["orphans_per_season"]["2024"]


# ---------------------------------------------------------------------------
# validate_sync (top-level function)
# ---------------------------------------------------------------------------


class TestValidateSync:
    """Tests for the top-level validate_sync function."""

    def test_clean_data_all_pass(self, repo: ParquetRepository) -> None:
        """All checks pass with clean, consistent data."""
        repo.save_teams(_TEAMS)
        seasons = [Season(year=y) for y in (2022, 2023, 2024)]
        repo.save_seasons(seasons)
        for s in seasons:
            repo.save_games(_make_n_games(s.year, 100))

        report = validate_sync(repo)
        assert report.all_passed is True
        assert len(report.results) == 3
        check_names = {r.check_name for r in report.results}
        assert check_names == {"game_counts", "duplicate_games", "team_references"}

    def test_multiple_failures(self, repo: ParquetRepository) -> None:
        """Multiple checks can fail simultaneously."""
        repo.save_teams(_TEAMS)
        seasons = [Season(year=y) for y in (2022, 2023, 2024)]
        repo.save_seasons(seasons)

        # Normal counts for 2022/2023, anomalous for 2024
        repo.save_games(_make_n_games(2022, 100))
        repo.save_games(_make_n_games(2023, 100))

        # 2024: few games + a duplicate + an orphan team
        games_2024 = _make_n_games(2024, 5)
        games_2024.append(
            _make_game(
                game_id="dup",
                season=2024,
                day_num=games_2024[0].day_num,
                w_team_id=games_2024[0].w_team_id,
                l_team_id=games_2024[0].l_team_id,
            )
        )
        games_2024.append(
            _make_game(
                game_id="orphan",
                season=2024,
                day_num=99,
                w_team_id=9999,
                l_team_id=1101,
            )
        )
        repo.save_games(games_2024)

        report = validate_sync(repo)
        assert report.all_passed is False
        failed = [r for r in report.results if not r.passed]
        assert len(failed) >= 2  # game_counts + duplicate_games at least

    def test_empty_repo_all_pass(self, repo: ParquetRepository) -> None:
        """Empty repository passes all checks (nothing to validate)."""
        report = validate_sync(repo)
        assert report.all_passed is True

    def test_does_not_raise_on_failures(self, repo: ParquetRepository) -> None:
        """validate_sync never raises on validation failures."""
        repo.save_teams([Team(team_id=1101, team_name="Only Team")])
        repo.save_seasons([Season(year=2024)])
        # Game with orphan team ID
        repo.save_games(
            [
                _make_game(
                    game_id="orphan",
                    season=2024,
                    day_num=10,
                    w_team_id=9999,
                    l_team_id=8888,
                )
            ]
        )

        # Should not raise — just return report with failures
        report = validate_sync(repo)
        assert report.all_passed is False
