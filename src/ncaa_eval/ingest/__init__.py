"""Data ingestion module."""

from __future__ import annotations

from ncaa_eval.ingest.repository import ParquetRepository, Repository
from ncaa_eval.ingest.schema import Game, Season, Team

__all__ = [
    "Game",
    "ParquetRepository",
    "Repository",
    "Season",
    "Team",
]
