"""Data ingestion module."""

from __future__ import annotations

from ncaa_eval.ingest.connectors import (
    AuthenticationError,
    Connector,
    ConnectorError,
    DataFormatError,
    EspnConnector,
    KaggleConnector,
    NetworkError,
)
from ncaa_eval.ingest.repository import ParquetRepository, Repository
from ncaa_eval.ingest.schema import Game, Season, Team
from ncaa_eval.ingest.sync import SyncEngine, SyncResult

__all__ = [
    "AuthenticationError",
    "Connector",
    "ConnectorError",
    "DataFormatError",
    "EspnConnector",
    "Game",
    "KaggleConnector",
    "NetworkError",
    "ParquetRepository",
    "Repository",
    "Season",
    "SyncEngine",
    "SyncResult",
    "Team",
]
