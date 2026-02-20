"""Integration tests for SyncEngine and sync CLI.

These tests exercise the full fetch-store-cache cycle using mocked
connectors and the real ParquetRepository to verify end-to-end behaviour.
"""

from __future__ import annotations

import datetime
import shutil
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ncaa_eval.ingest.repository import ParquetRepository
from ncaa_eval.ingest.schema import Game, Season, Team
from ncaa_eval.ingest.sync import SyncEngine, SyncResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIXTURE_KAGGLE = Path(__file__).parent.parent / "fixtures" / "kaggle"

_TEAMS = [
    Team(team_id=1101, team_name="Abilene Chr"),
    Team(team_id=1102, team_name="Air Force"),
    Team(team_id=1103, team_name="Akron"),
    Team(team_id=1104, team_name="Alabama"),
]
_SEASONS = [Season(year=2023), Season(year=2024)]

_GAMES_2024: list[dict[str, Any]] = [
    {
        "game_id": "2024_11_1101_1102",
        "season": 2024,
        "day_num": 11,
        "w_team_id": 1101,
        "l_team_id": 1102,
        "w_score": 75,
        "l_score": 60,
        "loc": "H",
    },
    {
        "game_id": "2024_15_1103_1104",
        "season": 2024,
        "day_num": 15,
        "w_team_id": 1103,
        "l_team_id": 1104,
        "w_score": 80,
        "l_score": 70,
        "loc": "A",
    },
]
_GAMES_2023: list[dict[str, Any]] = [
    {
        "game_id": "2023_20_1101_1103",
        "season": 2023,
        "day_num": 20,
        "w_team_id": 1101,
        "l_team_id": 1103,
        "w_score": 65,
        "l_score": 55,
        "loc": "N",
    }
]

# ESPN games for season 2024 (most recent season = max(2023, 2024) = 2024)
_ESPN_GAMES_2024: list[dict[str, Any]] = [
    {
        "game_id": "espn_99001",
        "season": 2024,
        "day_num": 90,
        "w_team_id": 1101,
        "l_team_id": 1104,
        "w_score": 85,
        "l_score": 72,
        "loc": "H",
    },
]

_DAY_ZEROS: dict[int, datetime.date] = {
    2023: datetime.date(2022, 11, 7),
    2024: datetime.date(2023, 11, 6),
}

# Spellings dict returned by KaggleConnector.fetch_team_spellings() in ESPN tests.
_SPELLINGS: dict[str, int] = {
    "abilene chr": 1101,
    "air force": 1102,
    "akron": 1103,
    "alabama": 1104,
}

# Team map returned by _build_espn_team_map() in ESPN tests.
_ESPN_TEAM_MAP: dict[str, int] = {
    "Abilene Christian": 1101,
    "Air Force": 1102,
    "Akron": 1103,
    "Alabama": 1104,
}


def _make_games(records: list[dict[str, Any]]) -> list[Game]:
    return [Game(**r) for r in records]


def _setup_kaggle_fixture_dir(dest: Path) -> None:
    """Copy fixture Kaggle CSVs to dest/kaggle/ (simulating a download)."""
    kaggle_dir = dest / "kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    for csv in _FIXTURE_KAGGLE.glob("*.csv"):
        shutil.copy(csv, kaggle_dir / csv.name)


# ---------------------------------------------------------------------------
# Test 5.2: Kaggle full cycle
# ---------------------------------------------------------------------------


@pytest.mark.integration
@patch("ncaa_eval.ingest.sync.KaggleConnector")
def test_sync_kaggle_full_cycle(mock_cls: MagicMock, tmp_path: Path) -> None:
    """Full Kaggle sync: fetch teams/seasons/games and persist to Parquet."""
    instance = mock_cls.return_value
    instance.download.return_value = None
    instance.fetch_teams.return_value = _TEAMS
    instance.fetch_seasons.return_value = _SEASONS
    instance.fetch_games.side_effect = lambda year: _make_games(_GAMES_2024 if year == 2024 else _GAMES_2023)

    repo = ParquetRepository(tmp_path)
    engine = SyncEngine(repository=repo, data_dir=tmp_path)
    result = engine.sync_kaggle()

    assert result.source == "kaggle"
    assert result.teams_written == len(_TEAMS)
    assert result.seasons_written == len(_SEASONS)
    assert result.games_written == len(_GAMES_2024) + len(_GAMES_2023)
    assert result.seasons_cached == 0

    assert repo.get_teams() == _TEAMS
    games_2024 = repo.get_games(2024)
    assert len(games_2024) == len(_GAMES_2024)
    game_ids = {g.game_id for g in games_2024}
    assert "2024_11_1101_1102" in game_ids


# ---------------------------------------------------------------------------
# Test 5.3: Kaggle cache hit (second run)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@patch("ncaa_eval.ingest.sync.KaggleConnector")
def test_sync_kaggle_cache_hit(mock_cls: MagicMock, tmp_path: Path) -> None:
    """Second sync call skips fetch_games when Parquet already exists."""
    instance = mock_cls.return_value
    instance.download.return_value = None
    instance.fetch_teams.return_value = _TEAMS
    instance.fetch_seasons.return_value = _SEASONS
    instance.fetch_games.side_effect = lambda year: _make_games(_GAMES_2024 if year == 2024 else _GAMES_2023)

    repo = ParquetRepository(tmp_path)
    engine = SyncEngine(repository=repo, data_dir=tmp_path)

    # First run: writes Parquet
    engine.sync_kaggle()
    fetch_games_count_after_first = instance.fetch_games.call_count
    fetch_teams_count_after_first = instance.fetch_teams.call_count
    fetch_seasons_count_after_first = instance.fetch_seasons.call_count

    # Second run: should hit cache for all seasons
    result2 = engine.sync_kaggle(force_refresh=False)

    assert instance.fetch_games.call_count == fetch_games_count_after_first
    assert instance.fetch_teams.call_count == fetch_teams_count_after_first
    assert instance.fetch_seasons.call_count == fetch_seasons_count_after_first
    assert result2.seasons_cached == len(_SEASONS)
    assert result2.games_written == 0
    assert result2.teams_written == 0
    assert result2.seasons_written == 0


# ---------------------------------------------------------------------------
# Test 5.4: --force-refresh bypasses Parquet cache
# ---------------------------------------------------------------------------


@pytest.mark.integration
@patch("ncaa_eval.ingest.sync.KaggleConnector")
def test_sync_kaggle_force_refresh(mock_cls: MagicMock, tmp_path: Path) -> None:
    """force_refresh=True re-fetches even when Parquet files exist."""
    instance = mock_cls.return_value
    instance.download.return_value = None
    instance.fetch_teams.return_value = _TEAMS
    instance.fetch_seasons.return_value = _SEASONS
    instance.fetch_games.side_effect = lambda year: _make_games(_GAMES_2024 if year == 2024 else _GAMES_2023)

    repo = ParquetRepository(tmp_path)
    engine = SyncEngine(repository=repo, data_dir=tmp_path)

    # First run to create Parquet files
    engine.sync_kaggle()

    # Force refresh run
    instance.fetch_teams.reset_mock()
    instance.fetch_seasons.reset_mock()
    instance.fetch_games.reset_mock()

    result = engine.sync_kaggle(force_refresh=True)

    instance.download.assert_called_with(force=True)
    instance.fetch_teams.assert_called_once()
    instance.fetch_seasons.assert_called_once()
    assert instance.fetch_games.call_count == len(_SEASONS)
    assert result.seasons_cached == 0


# ---------------------------------------------------------------------------
# Test 5.5: ESPN dependency guard
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_sync_espn_requires_kaggle(tmp_path: Path) -> None:
    """sync_espn raises RuntimeError when repo has no teams/seasons."""
    repo = ParquetRepository(tmp_path)
    engine = SyncEngine(repository=repo, data_dir=tmp_path)

    with pytest.raises(RuntimeError, match="kaggle"):
        engine.sync_espn()


# ---------------------------------------------------------------------------
# ESPN full cycle, cache hit, and force-refresh
# ---------------------------------------------------------------------------


@pytest.mark.integration
@patch("ncaa_eval.ingest.sync._build_espn_team_map", return_value=_ESPN_TEAM_MAP)
@patch("ncaa_eval.ingest.sync.EspnConnector")
@patch("ncaa_eval.ingest.sync.KaggleConnector")
def test_sync_espn_full_cycle(
    mock_kaggle_cls: MagicMock,
    mock_espn_cls: MagicMock,
    mock_build_map: MagicMock,
    tmp_path: Path,
) -> None:
    """ESPN sync fetches games, merges with existing Kaggle games, and creates marker."""
    # Pre-populate repo with Kaggle data (required ESPN pre-condition)
    repo = ParquetRepository(tmp_path)
    repo.save_teams(_TEAMS)
    repo.save_seasons(_SEASONS)
    repo.save_games(_make_games(_GAMES_2024))  # existing Kaggle games

    # Mock KaggleConnector used to load day_zeros + spellings (no download)
    kaggle_instance = mock_kaggle_cls.return_value
    kaggle_instance.load_day_zeros.return_value = _DAY_ZEROS
    kaggle_instance.fetch_team_spellings.return_value = _SPELLINGS

    # Mock EspnConnector
    espn_instance = mock_espn_cls.return_value
    espn_instance.fetch_games.return_value = _make_games(_ESPN_GAMES_2024)

    engine = SyncEngine(repository=repo, data_dir=tmp_path)
    result = engine.sync_espn()

    assert result.source == "espn"
    assert result.games_written == len(_ESPN_GAMES_2024)
    assert result.seasons_cached == 0

    # Verify marker file was created for year 2024
    assert (tmp_path / ".espn_synced_2024").exists()

    # Verify games were merged: original Kaggle games + ESPN games
    all_games_2024 = repo.get_games(2024)
    game_ids = {g.game_id for g in all_games_2024}
    assert "2024_11_1101_1102" in game_ids  # Kaggle game preserved
    assert "espn_99001" in game_ids  # ESPN game added

    # Verify _build_espn_team_map was called with the spellings dict
    mock_build_map.assert_called_once_with(2024, _SPELLINGS)


@pytest.mark.integration
@patch("ncaa_eval.ingest.sync._build_espn_team_map", return_value=_ESPN_TEAM_MAP)
@patch("ncaa_eval.ingest.sync.EspnConnector")
@patch("ncaa_eval.ingest.sync.KaggleConnector")
def test_sync_espn_cache_hit(
    mock_kaggle_cls: MagicMock,
    mock_espn_cls: MagicMock,
    mock_build_map: MagicMock,
    tmp_path: Path,
) -> None:
    """sync_espn skips fetch when marker file exists and force_refresh is False."""
    repo = ParquetRepository(tmp_path)
    repo.save_teams(_TEAMS)
    repo.save_seasons(_SEASONS)

    kaggle_instance = mock_kaggle_cls.return_value
    kaggle_instance.load_day_zeros.return_value = _DAY_ZEROS
    kaggle_instance.fetch_team_spellings.return_value = _SPELLINGS

    espn_instance = mock_espn_cls.return_value

    # Pre-create marker for max year (2024)
    (tmp_path / ".espn_synced_2024").touch()

    engine = SyncEngine(repository=repo, data_dir=tmp_path)
    result = engine.sync_espn()

    assert result.seasons_cached == 1
    assert result.games_written == 0
    espn_instance.fetch_games.assert_not_called()


@pytest.mark.integration
@patch("ncaa_eval.ingest.sync._build_espn_team_map", return_value=_ESPN_TEAM_MAP)
@patch("ncaa_eval.ingest.sync.EspnConnector")
@patch("ncaa_eval.ingest.sync.KaggleConnector")
def test_sync_espn_force_refresh(
    mock_kaggle_cls: MagicMock,
    mock_espn_cls: MagicMock,
    mock_build_map: MagicMock,
    tmp_path: Path,
) -> None:
    """force_refresh=True deletes marker and re-fetches ESPN games."""
    repo = ParquetRepository(tmp_path)
    repo.save_teams(_TEAMS)
    repo.save_seasons(_SEASONS)

    kaggle_instance = mock_kaggle_cls.return_value
    kaggle_instance.load_day_zeros.return_value = _DAY_ZEROS
    kaggle_instance.fetch_team_spellings.return_value = _SPELLINGS

    espn_instance = mock_espn_cls.return_value
    espn_instance.fetch_games.return_value = _make_games(_ESPN_GAMES_2024)

    # Pre-create marker (simulating previous sync)
    marker = tmp_path / ".espn_synced_2024"
    marker.touch()

    engine = SyncEngine(repository=repo, data_dir=tmp_path)
    result = engine.sync_espn(force_refresh=True)

    # Force refresh should re-fetch games (marker deleted then recreated)
    espn_instance.fetch_games.assert_called_once_with(2024)
    assert result.games_written == len(_ESPN_GAMES_2024)
    assert result.seasons_cached == 0
    # Marker re-created after successful sync
    assert marker.exists()


# ---------------------------------------------------------------------------
# Test 5.6: sync_all invokes Kaggle before ESPN
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_sync_all_order(tmp_path: Path) -> None:
    """sync_all calls sync_kaggle before sync_espn."""
    repo = ParquetRepository(tmp_path)
    engine = SyncEngine(repository=repo, data_dir=tmp_path)

    call_order: list[str] = []

    kaggle_result = SyncResult(source="kaggle")
    espn_result = SyncResult(source="espn")

    def _kaggle(force_refresh: bool = False) -> SyncResult:
        call_order.append("kaggle")
        return kaggle_result

    def _espn(force_refresh: bool = False) -> SyncResult:
        call_order.append("espn")
        return espn_result

    engine.sync_kaggle = _kaggle  # type: ignore[method-assign]
    engine.sync_espn = _espn  # type: ignore[method-assign]

    results = engine.sync_all()

    assert call_order == ["kaggle", "espn"]
    assert results == [kaggle_result, espn_result]


# ---------------------------------------------------------------------------
# Test 5.7: CLI via typer.testing.CliRunner
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.no_mutation
@patch("ncaa_eval.ingest.sync.KaggleConnector")
def test_cli_sync_kaggle(mock_cls: MagicMock, tmp_path: Path) -> None:
    """CLI --source kaggle exits 0 and creates Parquet files."""
    instance = mock_cls.return_value
    instance.download.return_value = None
    instance.fetch_teams.return_value = _TEAMS
    instance.fetch_seasons.return_value = [Season(year=2024)]
    instance.fetch_games.return_value = _make_games(_GAMES_2024)

    # Add project root to sys.path so 'import sync' works
    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from typer.testing import CliRunner

    import sync as sync_module

    runner = CliRunner()
    result = runner.invoke(sync_module.app, ["--source", "kaggle", "--dest", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "teams.parquet").exists()
    assert (tmp_path / "seasons.parquet").exists()
    assert (tmp_path / "games" / "season=2024" / "data.parquet").exists()


@pytest.mark.integration
@pytest.mark.no_mutation
def test_cli_invalid_source(tmp_path: Path) -> None:
    """CLI exits 1 for unknown --source values."""
    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from typer.testing import CliRunner

    import sync as sync_module

    runner = CliRunner()
    result = runner.invoke(sync_module.app, ["--source", "invalid", "--dest", str(tmp_path)])
    assert result.exit_code == 1


@pytest.mark.integration
@pytest.mark.no_mutation
def test_cli_espn_dependency_guard(tmp_path: Path) -> None:
    """CLI exits 1 when syncing ESPN without prior Kaggle data."""
    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from typer.testing import CliRunner

    import sync as sync_module

    runner = CliRunner()
    result = runner.invoke(sync_module.app, ["--source", "espn", "--dest", str(tmp_path)])
    assert result.exit_code == 1
    assert "kaggle" in result.output.lower() or "kaggle" in (result.stderr or "").lower()
