"""Unit tests for ncaa_eval.ingest.connectors.kaggle (KaggleConnector)."""

from __future__ import annotations

import datetime
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ncaa_eval.ingest.connectors.base import AuthenticationError, DataFormatError
from ncaa_eval.ingest.connectors.kaggle import KaggleConnector

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "kaggle"


@pytest.fixture
def kaggle_dir(tmp_path: Path) -> Path:
    """Copy fixture CSVs to a temp directory and return its path."""
    dest = tmp_path / "kaggle"
    shutil.copytree(_FIXTURES_DIR, dest)
    return dest


@pytest.fixture
def connector(kaggle_dir: Path) -> KaggleConnector:
    """Return a KaggleConnector pointed at fixture CSVs."""
    return KaggleConnector(extract_dir=kaggle_dir)


# ---------------------------------------------------------------------------
# TestKaggleConnectorTeams  (subtask 6.3)
# ---------------------------------------------------------------------------


class TestKaggleConnectorTeams:
    """Tests for fetch_teams() parsing MTeams.csv."""

    @pytest.mark.smoke
    def test_fetch_teams_count(self, connector: KaggleConnector) -> None:
        teams = connector.fetch_teams()
        assert len(teams) == 4

    def test_fetch_teams_ids(self, connector: KaggleConnector) -> None:
        teams = connector.fetch_teams()
        ids = {t.team_id for t in teams}
        assert ids == {1101, 1102, 1103, 1104}

    def test_fetch_teams_names(self, connector: KaggleConnector) -> None:
        teams = connector.fetch_teams()
        name_map = {t.team_id: t.team_name for t in teams}
        assert name_map[1101] == "Abilene Chr"
        assert name_map[1102] == "Air Force"

    def test_fetch_teams_missing_file(self, tmp_path: Path) -> None:
        connector = KaggleConnector(extract_dir=tmp_path)
        with pytest.raises(DataFormatError, match="file not found"):
            connector.fetch_teams()

    def test_fetch_teams_missing_columns(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "MTeams.csv"
        csv_path.write_text("ID,Name\n1,Foo\n")
        connector = KaggleConnector(extract_dir=tmp_path)
        with pytest.raises(DataFormatError, match="missing columns"):
            connector.fetch_teams()


# ---------------------------------------------------------------------------
# TestKaggleConnectorGames  (subtask 6.4, 6.5)
# ---------------------------------------------------------------------------


class TestKaggleConnectorGames:
    """Tests for fetch_games(season) parsing game CSVs."""

    @pytest.mark.smoke
    def test_fetch_games_regular_season(self, connector: KaggleConnector) -> None:
        games = connector.fetch_games(2024)
        regular = [g for g in games if not g.is_tournament]
        assert len(regular) == 2

    def test_fetch_games_tournament(self, connector: KaggleConnector) -> None:
        games = connector.fetch_games(2024)
        tourney = [g for g in games if g.is_tournament]
        assert len(tourney) == 2

    def test_tournament_flag(self, connector: KaggleConnector) -> None:
        games = connector.fetch_games(2024)
        tourney = [g for g in games if g.is_tournament]
        assert all(g.is_tournament for g in tourney)

    def test_field_mapping(self, connector: KaggleConnector) -> None:
        games = connector.fetch_games(2024)
        regular = [g for g in games if not g.is_tournament]
        game = next(g for g in regular if g.day_num == 11)
        assert game.season == 2024
        assert game.w_team_id == 1101
        assert game.l_team_id == 1102
        assert game.w_score == 75
        assert game.l_score == 60
        assert game.loc == "H"
        assert game.num_ot == 0

    def test_overtime_games(self, connector: KaggleConnector) -> None:
        games = connector.fetch_games(2024)
        ot_games = [g for g in games if g.num_ot > 0]
        assert len(ot_games) == 2  # one regular, one tourney
        regular_ot = next(g for g in ot_games if not g.is_tournament)
        assert regular_ot.num_ot == 1

    def test_season_filter(self, connector: KaggleConnector) -> None:
        games_2023 = connector.fetch_games(2023)
        assert len(games_2023) == 1
        assert games_2023[0].season == 2023

    def test_empty_season_returns_empty(self, connector: KaggleConnector) -> None:
        games = connector.fetch_games(1999)
        assert games == []

    def test_fetch_games_missing_csv(self, tmp_path: Path) -> None:
        connector = KaggleConnector(extract_dir=tmp_path)
        with pytest.raises(DataFormatError, match="file not found"):
            connector.fetch_games(2024)


# ---------------------------------------------------------------------------
# TestKaggleConnectorGameId  (subtask 6.8)
# ---------------------------------------------------------------------------


class TestKaggleConnectorGameId:
    """Tests for game_id format: "{season}_{day_num}_{w_team_id}_{l_team_id}"."""

    def test_game_id_format(self, connector: KaggleConnector) -> None:
        games = connector.fetch_games(2024)
        regular = [g for g in games if not g.is_tournament]
        game = next(g for g in regular if g.day_num == 11)
        assert game.game_id == "2024_11_1101_1102"

    def test_tournament_game_id_format(self, connector: KaggleConnector) -> None:
        games = connector.fetch_games(2024)
        tourney = [g for g in games if g.is_tournament]
        game = next(g for g in tourney if g.day_num == 134)
        assert game.game_id == "2024_134_1101_1104"


# ---------------------------------------------------------------------------
# TestKaggleConnectorDateComputation  (subtask 6.7)
# ---------------------------------------------------------------------------


class TestKaggleConnectorDateComputation:
    """Tests for calendar date computation from DayNum + DayZero."""

    def test_date_from_day_zero(self, connector: KaggleConnector) -> None:
        games = connector.fetch_games(2024)
        game = next(g for g in games if g.day_num == 11 and not g.is_tournament)
        # DayZero for 2024 is 2023-11-06, day_num=11 → 2023-11-17
        expected = datetime.date(2023, 11, 6) + datetime.timedelta(days=11)
        assert game.date == expected

    def test_tournament_date(self, connector: KaggleConnector) -> None:
        games = connector.fetch_games(2024)
        game = next(g for g in games if g.day_num == 134 and g.is_tournament)
        expected = datetime.date(2023, 11, 6) + datetime.timedelta(days=134)
        assert game.date == expected


# ---------------------------------------------------------------------------
# TestKaggleConnectorSeasons  (subtask 6.6)
# ---------------------------------------------------------------------------


class TestKaggleConnectorSeasons:
    """Tests for fetch_seasons() parsing MSeasons.csv."""

    @pytest.mark.smoke
    def test_fetch_seasons(self, connector: KaggleConnector) -> None:
        seasons = connector.fetch_seasons()
        assert len(seasons) == 2
        years = {s.year for s in seasons}
        assert years == {2023, 2024}

    def test_fetch_seasons_missing_file(self, tmp_path: Path) -> None:
        connector = KaggleConnector(extract_dir=tmp_path)
        with pytest.raises(DataFormatError, match="file not found"):
            connector.fetch_seasons()


# ---------------------------------------------------------------------------
# TestKaggleConnectorDownload  (subtask 6.9 — error handling)
# ---------------------------------------------------------------------------


def _mock_kaggle_modules() -> MagicMock:
    """Pre-populate sys.modules with a fake kaggle package so that the
    ``from kaggle.api.kaggle_api_extended import KaggleApi`` inside
    ``KaggleConnector.download()`` resolves to our mock instead of importing
    the real kaggle package (which calls ``api.authenticate()`` at import
    time and fails without credentials).
    """
    mock_api_cls = MagicMock()
    mock_extended = MagicMock()
    mock_extended.KaggleApi = mock_api_cls
    mock_api_mod = MagicMock()
    mock_kaggle = MagicMock()

    modules = {
        "kaggle": mock_kaggle,
        "kaggle.api": mock_api_mod,
        "kaggle.api.kaggle_api_extended": mock_extended,
    }
    return mock_api_cls, modules  # type: ignore[return-value]


class TestKaggleConnectorDownload:
    """Tests for download() with mocked KaggleApi."""

    def test_auth_failure(self, tmp_path: Path) -> None:
        mock_api_cls, modules = _mock_kaggle_modules()
        mock_instance = MagicMock()
        mock_instance.authenticate.side_effect = Exception("bad creds")
        mock_api_cls.return_value = mock_instance

        with patch.dict(sys.modules, modules):
            connector = KaggleConnector(extract_dir=tmp_path)
            with pytest.raises(AuthenticationError, match="credentials not found"):
                connector.download()

    def test_download_failure(self, tmp_path: Path) -> None:
        mock_api_cls, modules = _mock_kaggle_modules()
        mock_instance = MagicMock()
        mock_instance.authenticate.return_value = None
        mock_instance.competition_download_files.side_effect = Exception("network error")
        mock_api_cls.return_value = mock_instance

        with patch.dict(sys.modules, modules):
            connector = KaggleConnector(extract_dir=tmp_path)
            from ncaa_eval.ingest.connectors.base import NetworkError

            with pytest.raises(NetworkError, match="failed to download"):
                connector.download()

    def test_successful_download(self, tmp_path: Path) -> None:
        mock_api_cls, modules = _mock_kaggle_modules()
        mock_instance = MagicMock()
        mock_instance.authenticate.return_value = None
        mock_instance.competition_download_files.return_value = None
        mock_api_cls.return_value = mock_instance

        with patch.dict(sys.modules, modules):
            connector = KaggleConnector(extract_dir=tmp_path)
            connector.download()

            mock_instance.authenticate.assert_called_once()
            mock_instance.competition_download_files.assert_called_once()
