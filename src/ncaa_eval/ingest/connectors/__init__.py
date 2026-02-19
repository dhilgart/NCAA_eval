"""Data source connectors for NCAA basketball data ingestion."""

from __future__ import annotations

from ncaa_eval.ingest.connectors.base import (
    AuthenticationError,
    Connector,
    ConnectorError,
    DataFormatError,
    NetworkError,
)
from ncaa_eval.ingest.connectors.espn import EspnConnector
from ncaa_eval.ingest.connectors.kaggle import KaggleConnector

__all__ = [
    "AuthenticationError",
    "Connector",
    "ConnectorError",
    "DataFormatError",
    "EspnConnector",
    "KaggleConnector",
    "NetworkError",
]
