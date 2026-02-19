"""Unit tests for ncaa_eval.ingest.schema Pydantic models."""

from __future__ import annotations

import datetime
from typing import Any

import pytest
from pydantic import ValidationError

from ncaa_eval.ingest.schema import Game, Season, Team

# ---------------------------------------------------------------------------
# Team model
# ---------------------------------------------------------------------------


class TestTeam:
    """Tests for the Team model."""

    def test_valid_construction(self) -> None:
        team = Team(team_id=1101, team_name="Abilene Chr", canonical_name="abilene-christian")
        assert team.team_id == 1101
        assert team.team_name == "Abilene Chr"
        assert team.canonical_name == "abilene-christian"

    def test_canonical_name_defaults_empty(self) -> None:
        team = Team(team_id=1101, team_name="Abilene Chr")
        assert team.canonical_name == ""

    def test_alias_construction(self) -> None:
        team = Team.model_validate({"TeamID": 1101, "TeamName": "Abilene Chr"})
        assert team.team_id == 1101

    def test_invalid_team_id_zero(self) -> None:
        with pytest.raises(ValidationError):
            Team(team_id=0, team_name="Bad")

    def test_invalid_team_id_negative(self) -> None:
        with pytest.raises(ValidationError):
            Team(team_id=-1, team_name="Bad")

    def test_empty_team_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Team(team_id=1, team_name="")

    def test_serialization_round_trip(self) -> None:
        team = Team(team_id=1101, team_name="Abilene Chr", canonical_name="abilene-christian")
        data = team.model_dump()
        restored = Team(**data)
        assert restored == team

    def test_serialization_by_alias(self) -> None:
        team = Team(team_id=1101, team_name="Abilene Chr")
        data = team.model_dump(by_alias=True)
        assert "TeamID" in data
        assert "TeamName" in data


# ---------------------------------------------------------------------------
# Season model
# ---------------------------------------------------------------------------


class TestSeason:
    """Tests for the Season model."""

    def test_valid_construction(self) -> None:
        season = Season(year=2024)
        assert season.year == 2024

    def test_alias_construction(self) -> None:
        season = Season.model_validate({"Year": 2024})
        assert season.year == 2024

    def test_year_too_low(self) -> None:
        with pytest.raises(ValidationError):
            Season(year=1984)

    def test_year_boundary(self) -> None:
        season = Season(year=1985)
        assert season.year == 1985

    def test_serialization_round_trip(self) -> None:
        season = Season(year=2024)
        data = season.model_dump()
        restored = Season(**data)
        assert restored == season


# ---------------------------------------------------------------------------
# Game model
# ---------------------------------------------------------------------------


class TestGame:
    """Tests for the Game model."""

    @pytest.fixture
    def valid_game_kwargs(self) -> dict[str, Any]:
        return {
            "game_id": "2024_10_1101_1102",
            "season": 2024,
            "day_num": 10,
            "w_team_id": 1101,
            "l_team_id": 1102,
            "w_score": 75,
            "l_score": 60,
            "loc": "H",
        }

    def test_valid_construction(self, valid_game_kwargs: dict[str, Any]) -> None:
        game = Game(**valid_game_kwargs)
        assert game.game_id == "2024_10_1101_1102"
        assert game.season == 2024
        assert game.day_num == 10
        assert game.w_team_id == 1101
        assert game.l_team_id == 1102
        assert game.w_score == 75
        assert game.l_score == 60
        assert game.loc == "H"
        assert game.num_ot == 0
        assert game.is_tournament is False
        assert game.date is None

    def test_all_fields(self) -> None:
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
        assert game.date == datetime.date(2024, 3, 21)
        assert game.num_ot == 2
        assert game.is_tournament is True

    def test_alias_construction(self) -> None:
        game = Game.model_validate({
            "GameID": "2024_10_1101_1102",
            "Season": 2024,
            "DayNum": 10,
            "WTeamID": 1101,
            "LTeamID": 1102,
            "WScore": 75,
            "LScore": 60,
            "Loc": "H",
        })
        assert game.game_id == "2024_10_1101_1102"

    def test_loc_values(self, valid_game_kwargs: dict[str, Any]) -> None:
        for loc_val in ("H", "A", "N"):
            game = Game(**{**valid_game_kwargs, "loc": loc_val})
            assert game.loc == loc_val

    def test_invalid_loc(self, valid_game_kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            Game(**{**valid_game_kwargs, "loc": "X"})

    def test_negative_score_rejected(self, valid_game_kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            Game(**{**valid_game_kwargs, "w_score": -1})

    def test_negative_l_score_rejected(self, valid_game_kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            Game(**{**valid_game_kwargs, "l_score": -1})

    def test_season_too_low(self, valid_game_kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            Game(**{**valid_game_kwargs, "season": 1984})

    def test_invalid_team_id(self, valid_game_kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            Game(**{**valid_game_kwargs, "w_team_id": 0})

    def test_negative_day_num(self, valid_game_kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            Game(**{**valid_game_kwargs, "day_num": -1})

    def test_negative_num_ot(self, valid_game_kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            Game(**{**valid_game_kwargs, "num_ot": -1})

    def test_empty_game_id(self, valid_game_kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            Game(**{**valid_game_kwargs, "game_id": ""})

    def test_serialization_round_trip(self, valid_game_kwargs: dict[str, Any]) -> None:
        game = Game(**valid_game_kwargs)
        data = game.model_dump()
        restored = Game(**data)
        assert restored == game

    def test_serialization_round_trip_with_date(self) -> None:
        game = Game(
            game_id="2024_10_1101_1102",
            season=2024,
            day_num=10,
            date=datetime.date(2024, 3, 21),
            w_team_id=1101,
            l_team_id=1102,
            w_score=75,
            l_score=60,
            loc="N",
        )
        data = game.model_dump()
        restored = Game(**data)
        assert restored == game
        assert restored.date == datetime.date(2024, 3, 21)
