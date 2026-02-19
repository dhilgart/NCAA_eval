"""Unit tests for the Connector ABC and exception hierarchy."""

from __future__ import annotations

import pytest

from ncaa_eval.ingest.connectors.base import (
    AuthenticationError,
    Connector,
    ConnectorError,
    DataFormatError,
    NetworkError,
)
from ncaa_eval.ingest.schema import Game, Season, Team

# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    """Verify the exception inheritance chain."""

    @pytest.mark.smoke
    def test_authentication_error_is_connector_error(self) -> None:
        assert issubclass(AuthenticationError, ConnectorError)

    def test_data_format_error_is_connector_error(self) -> None:
        assert issubclass(DataFormatError, ConnectorError)

    def test_network_error_is_connector_error(self) -> None:
        assert issubclass(NetworkError, ConnectorError)

    def test_connector_error_is_exception(self) -> None:
        assert issubclass(ConnectorError, Exception)

    def test_raise_authentication_error(self) -> None:
        with pytest.raises(ConnectorError):
            raise AuthenticationError("kaggle: credentials not found")

    def test_raise_data_format_error(self) -> None:
        with pytest.raises(ConnectorError):
            raise DataFormatError("CSV columns mismatch")

    def test_raise_network_error(self) -> None:
        with pytest.raises(ConnectorError):
            raise NetworkError("connection timeout")


# ---------------------------------------------------------------------------
# ABC contract
# ---------------------------------------------------------------------------


class _StubConnector(Connector):
    """Minimal concrete connector for testing the ABC contract."""

    def fetch_teams(self) -> list[Team]:
        return []

    def fetch_games(self, season: int) -> list[Game]:
        return []

    def fetch_seasons(self) -> list[Season]:
        return []


class TestConnectorABC:
    """Verify the Connector ABC enforces abstract methods."""

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            Connector()  # type: ignore[abstract]

    def test_stub_connector_instantiates(self) -> None:
        connector = _StubConnector()
        assert connector.fetch_teams() == []
        assert connector.fetch_games(2024) == []
        assert connector.fetch_seasons() == []
