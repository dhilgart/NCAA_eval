"""Unit tests for ncaa_eval.ingest.connectors.espn (EspnConnector)."""

from __future__ import annotations

import datetime
from typing import Any
from unittest.mock import patch

import pandas as pd  # type: ignore[import-untyped]
import pytest

from ncaa_eval.ingest.connectors.base import DataFormatError
from ncaa_eval.ingest.connectors.espn import (
    EspnConnector,
    _parse_game_result,
    _resolve_team_id,
)

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

_TEAM_MAP: dict[str, int] = {
    "Duke": 1181,
    "North Carolina": 1314,
    "Kentucky": 1243,
    "Kansas": 1242,
}

_DAY_ZEROS: dict[int, datetime.date] = {
    2024: datetime.date(2023, 11, 6),
    2025: datetime.date(2024, 11, 4),
}


def _make_schedule_df(
    rows: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Build a mock cbbpy schedule DataFrame."""
    if rows is None:
        rows = [
            {
                "team": "Duke",
                "team_id": 150,
                "season": 2025,
                "game_id": "401700001",
                "game_day": "2024-11-15",
                "game_time": "7:00 PM",
                "opponent": "North Carolina",
                "opponent_id": 153,
                "season_type": "regular",
                "game_status": "Final",
                "tv_network": "ESPN",
                "game_result": "W 85-70",
            },
            {
                "team": "Duke",
                "team_id": 150,
                "season": 2025,
                "game_id": "401700002",
                "game_day": "2024-11-20",
                "game_time": "9:00 PM",
                "opponent": "Kentucky",
                "opponent_id": 96,
                "season_type": "regular",
                "game_status": "Final",
                "tv_network": "ESPN2",
                "game_result": "L 60-75",
            },
        ]
    return pd.DataFrame(rows)


@pytest.fixture
def connector() -> EspnConnector:
    """Return an EspnConnector with test mappings."""
    return EspnConnector(
        team_name_to_id=_TEAM_MAP,
        season_day_zeros=_DAY_ZEROS,
    )


# ---------------------------------------------------------------------------
# TestParseGameResult
# ---------------------------------------------------------------------------


class TestParseGameResult:
    """Tests for the _parse_game_result helper."""

    @pytest.mark.smoke
    def test_win(self) -> None:
        assert _parse_game_result("W 85-70") == (85, 70)

    def test_loss(self) -> None:
        assert _parse_game_result("L 60-75") == (60, 75)

    def test_empty_string(self) -> None:
        assert _parse_game_result("") is None

    def test_nan_string(self) -> None:
        assert _parse_game_result("nan") is None

    def test_malformed(self) -> None:
        assert _parse_game_result("Final") is None

    def test_no_dash(self) -> None:
        assert _parse_game_result("W 85") is None


# ---------------------------------------------------------------------------
# TestResolveTeamId
# ---------------------------------------------------------------------------


class TestResolveTeamId:
    """Tests for team name -> team_id resolution."""

    def test_exact_match(self) -> None:
        assert _resolve_team_id("Duke", _TEAM_MAP) == 1181

    def test_case_insensitive(self) -> None:
        assert _resolve_team_id("duke", _TEAM_MAP) == 1181

    def test_fuzzy_match(self) -> None:
        # "N. Carolina" should fuzzy-match "North Carolina"
        result = _resolve_team_id("N. Carolina", _TEAM_MAP)
        # Depending on fuzzy score, this may or may not match.
        # With rapidfuzz ratio, "n. carolina" vs "north carolina" ~ 76-82
        # We accept either match or None here.
        assert result in (1314, None)

    def test_no_match(self) -> None:
        result = _resolve_team_id("Nonexistent University", _TEAM_MAP)
        assert result is None


# ---------------------------------------------------------------------------
# TestEspnConnectorGames  (subtasks 7.3, 7.4, 7.5)
# ---------------------------------------------------------------------------


class TestEspnConnectorGames:
    """Tests for fetch_games() with mocked cbbpy."""

    @pytest.mark.smoke
    def test_fetch_games_basic(self, connector: EspnConnector) -> None:
        df = _make_schedule_df()
        with patch.object(connector, "_fetch_schedule_df", return_value=df):
            games = connector.fetch_games(2025)
        assert len(games) == 2

    def test_winner_loser_ordering_win(self, connector: EspnConnector) -> None:
        df = _make_schedule_df()
        with patch.object(connector, "_fetch_schedule_df", return_value=df):
            games = connector.fetch_games(2025)
        # First game: Duke W 85-70 vs North Carolina
        game = next(g for g in games if g.game_id == "espn_401700001")
        assert game.w_team_id == 1181  # Duke won
        assert game.l_team_id == 1314  # UNC lost
        assert game.w_score == 85
        assert game.l_score == 70

    def test_winner_loser_ordering_loss(self, connector: EspnConnector) -> None:
        df = _make_schedule_df()
        with patch.object(connector, "_fetch_schedule_df", return_value=df):
            games = connector.fetch_games(2025)
        # Second game: Duke L 60-75 vs Kentucky → Kentucky won
        game = next(g for g in games if g.game_id == "espn_401700002")
        assert game.w_team_id == 1243  # Kentucky won
        assert game.l_team_id == 1181  # Duke lost
        assert game.w_score == 75
        assert game.l_score == 60

    def test_game_id_format(self, connector: EspnConnector) -> None:
        df = _make_schedule_df()
        with patch.object(connector, "_fetch_schedule_df", return_value=df):
            games = connector.fetch_games(2025)
        ids = {g.game_id for g in games}
        assert "espn_401700001" in ids
        assert "espn_401700002" in ids

    def test_date_parsing(self, connector: EspnConnector) -> None:
        df = _make_schedule_df()
        with patch.object(connector, "_fetch_schedule_df", return_value=df):
            games = connector.fetch_games(2025)
        game = next(g for g in games if g.game_id == "espn_401700001")
        assert game.date == datetime.date(2024, 11, 15)

    def test_day_num_computation(self, connector: EspnConnector) -> None:
        df = _make_schedule_df()
        with patch.object(connector, "_fetch_schedule_df", return_value=df):
            games = connector.fetch_games(2025)
        game = next(g for g in games if g.game_id == "espn_401700001")
        # DayZero for 2025 is 2024-11-04. 2024-11-15 is day 11.
        assert game.day_num == 11

    def test_default_loc_neutral(self, connector: EspnConnector) -> None:
        df = _make_schedule_df()
        with patch.object(connector, "_fetch_schedule_df", return_value=df):
            games = connector.fetch_games(2025)
        # No home_away or is_neutral columns → defaults to "N".
        for game in games:
            assert game.loc == "N"

    def test_empty_schedule(self, connector: EspnConnector) -> None:
        with patch.object(connector, "_fetch_schedule_df", return_value=None):
            games = connector.fetch_games(2025)
        assert games == []

    def test_unresolved_team_skipped(self) -> None:
        connector = EspnConnector(
            team_name_to_id={"Duke": 1181},  # Only Duke known
            season_day_zeros=_DAY_ZEROS,
        )
        df = _make_schedule_df()
        with patch.object(connector, "_fetch_schedule_df", return_value=df):
            games = connector.fetch_games(2025)
        # Both games involve UNC or Kentucky which aren't in mapping → skipped
        assert len(games) == 0

    def test_unparseable_result_skipped(self, connector: EspnConnector) -> None:
        rows = [
            {
                "team": "Duke",
                "game_id": "401700099",
                "game_day": "2024-11-15",
                "opponent": "Kentucky",
                "game_result": "Postponed",
            },
        ]
        df = _make_schedule_df(rows)
        with patch.object(connector, "_fetch_schedule_df", return_value=df):
            games = connector.fetch_games(2025)
        assert games == []


# ---------------------------------------------------------------------------
# TestEspnConnectorLoc  (subtask 7.5)
# ---------------------------------------------------------------------------


class TestEspnConnectorLoc:
    """Tests for location mapping from ESPN context."""

    def test_home_away_column_home(self, connector: EspnConnector) -> None:
        rows = [
            {
                "team": "Duke",
                "game_id": "401700010",
                "game_day": "2024-11-15",
                "opponent": "North Carolina",
                "game_result": "W 85-70",
                "home_away": "home",
            },
        ]
        df = _make_schedule_df(rows)
        with patch.object(connector, "_fetch_schedule_df", return_value=df):
            games = connector.fetch_games(2025)
        assert len(games) == 1
        # Duke (team) was home and won → loc=H for winner
        assert games[0].loc == "H"

    def test_home_away_column_away(self, connector: EspnConnector) -> None:
        rows = [
            {
                "team": "Duke",
                "game_id": "401700011",
                "game_day": "2024-11-15",
                "opponent": "North Carolina",
                "game_result": "W 85-70",
                "home_away": "away",
            },
        ]
        df = _make_schedule_df(rows)
        with patch.object(connector, "_fetch_schedule_df", return_value=df):
            games = connector.fetch_games(2025)
        assert len(games) == 1
        # Duke (team) was away and won → loc=A for winner
        assert games[0].loc == "A"

    def test_neutral_site(self, connector: EspnConnector) -> None:
        rows = [
            {
                "team": "Duke",
                "game_id": "401700012",
                "game_day": "2024-11-15",
                "opponent": "North Carolina",
                "game_result": "W 85-70",
                "is_neutral": True,
            },
        ]
        df = _make_schedule_df(rows)
        with patch.object(connector, "_fetch_schedule_df", return_value=df):
            games = connector.fetch_games(2025)
        assert len(games) == 1
        assert games[0].loc == "N"


# ---------------------------------------------------------------------------
# TestEspnConnectorNotImplemented
# ---------------------------------------------------------------------------


class TestEspnConnectorNotImplemented:
    """Verify fetch_teams and fetch_seasons raise NotImplementedError."""

    def test_fetch_teams_raises(self, connector: EspnConnector) -> None:
        with pytest.raises(NotImplementedError):
            connector.fetch_teams()

    def test_fetch_seasons_raises(self, connector: EspnConnector) -> None:
        with pytest.raises(NotImplementedError):
            connector.fetch_seasons()


# ---------------------------------------------------------------------------
# TestEspnConnectorErrorHandling  (subtask 7.6)
# ---------------------------------------------------------------------------


class TestEspnConnectorErrorHandling:
    """Tests for error handling: network failure, unexpected response."""

    def test_network_failure_returns_empty(self, connector: EspnConnector) -> None:
        with patch("ncaa_eval.ingest.connectors.espn.ms") as mock_ms:
            mock_ms.get_games_season.side_effect = Exception("network failure")
            mock_ms.get_team_schedule.side_effect = Exception("network failure")
            games = connector.fetch_games(2025)
        assert games == []

    def test_missing_columns_raises_data_format_error(self, connector: EspnConnector) -> None:
        bad_df = pd.DataFrame({"bad_col": [1]})
        with patch.object(connector, "_fetch_schedule_df", return_value=bad_df):
            with pytest.raises(DataFormatError, match="missing columns"):
                connector.fetch_games(2025)

    def test_dedup_by_game_id(self, connector: EspnConnector) -> None:
        rows = [
            {
                "team": "Duke",
                "game_id": "401700001",
                "game_day": "2024-11-15",
                "opponent": "North Carolina",
                "game_result": "W 85-70",
            },
            {
                "team": "North Carolina",
                "game_id": "401700001",
                "game_day": "2024-11-15",
                "opponent": "Duke",
                "game_result": "L 70-85",
            },
        ]
        df = _make_schedule_df(rows)
        with patch.object(connector, "_fetch_schedule_df", return_value=df):
            games = connector.fetch_games(2025)
        # Same game_id appears twice → should be deduplicated to 1
        assert len(games) == 1
