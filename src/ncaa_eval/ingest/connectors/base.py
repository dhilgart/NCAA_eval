"""Abstract base class for data source connectors and shared exception hierarchy.

All concrete connectors (Kaggle, ESPN, etc.) inherit from :class:`Connector`
and implement the ``fetch_*`` methods relevant to their data source.  The
exception hierarchy provides a uniform error contract across connectors so that
callers can handle failures without coupling to a specific source.
"""

from __future__ import annotations

import abc

from ncaa_eval.ingest.schema import Game, Season, Team

# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class ConnectorError(Exception):
    """Base exception for all connector errors."""


class AuthenticationError(ConnectorError):
    """Credentials missing, invalid, or expired."""


class DataFormatError(ConnectorError):
    """Raw data (CSV / API response) does not match the expected schema."""


class NetworkError(ConnectorError):
    """Connection failure, timeout, or HTTP error."""


# ---------------------------------------------------------------------------
# Abstract Connector
# ---------------------------------------------------------------------------


class Connector(abc.ABC):
    """Abstract base class for NCAA data source connectors.

    All connectors must implement :meth:`fetch_games`, which is the universal
    capability.  :meth:`fetch_teams` and :meth:`fetch_seasons` are optional
    capabilities — subclasses that do not support them inherit the default
    implementation, which raises ``NotImplementedError``.  Callers should use
    :func:`isinstance` checks or ``try``/``except NotImplementedError`` to
    probe optional capabilities before calling them.
    """

    @abc.abstractmethod
    def fetch_games(self, season: int) -> list[Game]:
        """Fetch game results for a given *season* year."""

    def fetch_teams(self) -> list[Team]:
        """Fetch team data from the source.

        Optional capability — not all connectors provide team master data.
        Raises:
            NotImplementedError: If this connector does not support fetching teams.
        """
        raise NotImplementedError(f"{type(self).__name__} does not provide team data")

    def fetch_seasons(self) -> list[Season]:
        """Fetch available seasons from the source.

        Optional capability — not all connectors provide season master data.
        Raises:
            NotImplementedError: If this connector does not support fetching seasons.
        """
        raise NotImplementedError(f"{type(self).__name__} does not provide season data")
