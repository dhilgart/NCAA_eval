"""Pydantic v2 schema models for NCAA basketball data entities.

Defines the core data structures — Team, Game, and Season — that form
the internal representation layer.  All downstream code operates on
these models regardless of the upstream data source (Kaggle, BartTorvik,
ESPN, etc.) or the storage backend (Parquet, SQLite, …).
"""

from __future__ import annotations

import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Team(BaseModel):
    """A college basketball team."""

    model_config = ConfigDict(populate_by_name=True)

    team_id: int = Field(..., ge=1, alias="TeamID")
    team_name: str = Field(..., min_length=1, alias="TeamName")
    canonical_name: str = Field(default="", alias="CanonicalName")


class Season(BaseModel):
    """A single NCAA basketball season (identified by calendar year)."""

    model_config = ConfigDict(populate_by_name=True)

    year: int = Field(..., ge=1985, alias="Year")


class Game(BaseModel):
    """A single NCAA basketball game result."""

    model_config = ConfigDict(populate_by_name=True)

    game_id: str = Field(..., min_length=1, alias="GameID")
    season: int = Field(..., ge=1985, alias="Season")
    day_num: int = Field(..., ge=0, alias="DayNum")
    date: datetime.date | None = Field(default=None, alias="Date")
    w_team_id: int = Field(..., ge=1, alias="WTeamID")
    l_team_id: int = Field(..., ge=1, alias="LTeamID")
    w_score: int = Field(..., ge=0, alias="WScore")
    l_score: int = Field(..., ge=0, alias="LScore")
    loc: Literal["H", "A", "N"] = Field(..., alias="Loc")
    num_ot: int = Field(default=0, ge=0, alias="NumOT")
    is_tournament: bool = Field(default=False, alias="IsTournament")
