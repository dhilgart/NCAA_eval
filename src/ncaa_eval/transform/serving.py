"""Chronological data serving layer for walk-forward model training.

Provides ``ChronologicalDataServer``, which wraps a ``Repository`` and
streams game data in strict date order with temporal boundary enforcement.
Downstream consumers (walk-forward splitters, feature pipelines) use this
layer to ensure no data from future games leaks into model training.
"""

from __future__ import annotations

import datetime
import logging
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd  # type: ignore[import-untyped]

from ncaa_eval.ingest.repository import Repository
from ncaa_eval.ingest.schema import Game

logger = logging.getLogger(__name__)

# Seasons in which the NCAA tournament was cancelled and therefore contain no
# is_tournament=True games.  The flag is derived from this constant rather than
# inferred from the data so that downstream consumers can distinguish "no
# tournament data loaded yet" from "tournament was cancelled this year".
_NO_TOURNAMENT_SEASONS: frozenset[int] = frozenset({2020})


@dataclass(frozen=True)
class SeasonGames:
    """Result of a chronological season query.

    Attributes:
        year: Season year (e.g., 2023 for the 2022-23 season).
        games: All qualifying games sorted ascending by (date, game_id).
        has_tournament: False only for known no-tournament years (e.g., 2020
            COVID cancellation).  Signals to downstream walk-forward splitters
            that tournament evaluation should be skipped for this season.
    """

    year: int
    games: list[Game]
    has_tournament: bool


def rescale_overtime(score: int, num_ot: int) -> float:
    """Rescale a game score to a 40-minute equivalent for OT normalization.

    Overtime games inflate per-game scoring statistics because they involve
    more than 40 minutes of play.  The standard correction (Edwards 2021)
    normalises every game to a 40-minute basis:

        adjusted = raw_score × 40 / (40 + 5 × num_ot)

    Args:
        score: Raw final score (not adjusted).
        num_ot: Number of overtime periods played (0 for regulation).

    Returns:
        Score normalised to a 40-minute equivalent.

    Examples:
        >>> rescale_overtime(75, 0)   # Regulation: no change
        75.0
        >>> rescale_overtime(80, 1)   # 1 OT: 80 × 40 / 45 ≈ 71.11
        71.11111111111111
    """
    return score * 40.0 / (40 + 5 * num_ot)


def _effective_date(game: Game, year: int) -> datetime.date:
    """Return the calendar date for *game*, with a day_num fallback if date is None.

    In practice ``game.date`` is always set — Kaggle games derive it from
    ``DayZero + timedelta(days=day_num)`` and ESPN games carry the actual API
    date.  The fallback handles the theoretical ``None`` case introduced by the
    optional ``date`` field for schema-evolution resilience.

    Args:
        game: The game whose date is needed.
        year: Season year used to compute the fallback DayZero approximation.

    Returns:
        The game's calendar date.
    """
    if game.date is not None:
        return game.date
    # DayZero ≈ November 1 of the year before the season year.
    fallback = datetime.date(year - 1, 11, 1) + datetime.timedelta(days=game.day_num)
    logger.warning(
        "Game %s (season=%d) has no date; deriving ordering date from day_num=%d → %s",
        game.game_id,
        year,
        game.day_num,
        fallback,
    )
    return fallback


def _deduplicate_2025(games: list[Game]) -> list[Game]:
    """Remove duplicate 2025 games, preferring ESPN records for loc and num_ot.

    The 2025 season ingests 4,545 games twice — once from Kaggle (numeric
    ``game_id``) and once from ESPN (``game_id`` prefixed with ``"espn_"``).
    The canonical deduplication key is ``(w_team_id, l_team_id, day_num)``.
    When both records exist for the same physical game the ESPN record is kept
    because it provides more accurate ``loc`` (H/A/N) and ``num_ot`` values.

    Args:
        games: Raw game list for the 2025 season (may contain duplicates).

    Returns:
        Deduplicated list with at most one record per canonical game triplet.
    """
    if not games:
        return games
    df = pd.DataFrame([g.model_dump() for g in games])
    # Sort so Kaggle records (no "espn_" prefix, _is_espn=False) come first
    # and ESPN records (_is_espn=True) come last.  drop_duplicates(keep="last")
    # then retains the ESPN record when both exist for the same triplet.
    df["_is_espn"] = df["game_id"].str.startswith("espn_")
    df = df.sort_values("_is_espn")
    df = df.drop_duplicates(subset=["w_team_id", "l_team_id", "day_num"], keep="last")
    df = df.drop(columns=["_is_espn"]).reset_index(drop=True)
    return [Game(**row) for row in df.to_dict(orient="records")]


class ChronologicalDataServer:
    """Serves game data in strict chronological order for walk-forward modeling.

    Wraps a ``Repository`` and enforces temporal boundaries so that callers
    cannot accidentally access future game data during walk-forward validation.

    Args:
        repository: The data store from which games are retrieved.

    Example::

        from ncaa_eval.ingest.repository import ParquetRepository
        from ncaa_eval.transform.serving import ChronologicalDataServer

        repo = ParquetRepository(Path("data/"))
        server = ChronologicalDataServer(repo)
        season = server.get_chronological_season(2023)
        for daily_batch in server.iter_games_by_date(2023):
            process(daily_batch)
    """

    def __init__(self, repository: Repository) -> None:
        self._repo = repository

    def get_chronological_season(
        self,
        year: int,
        cutoff_date: datetime.date | None = None,
    ) -> SeasonGames:
        """Return all games for *year* sorted ascending by (date, game_id).

        Applies optional temporal cutoff so callers cannot retrieve games that
        had not yet been played as of a given date.  This is the primary
        leakage-prevention mechanism for walk-forward model training.

        Args:
            year: Season year (e.g., 2023 for the 2022-23 season).
            cutoff_date: If provided, only games on or before this date are
                returned.  Must not be in the future.

        Returns:
            ``SeasonGames`` with games sorted by ``(date, game_id)`` and the
            ``has_tournament`` flag reflecting known tournament cancellations.

        Raises:
            ValueError: If ``cutoff_date`` is strictly after today's date.
        """
        if cutoff_date is not None and cutoff_date > datetime.date.today():
            today = datetime.date.today()
            msg = f"Cannot request future game data: cutoff_date {cutoff_date} " f"is after today ({today})"
            raise ValueError(msg)

        games = self._repo.get_games(year)

        if year == 2025:
            games = _deduplicate_2025(games)

        if cutoff_date is not None:
            games = [g for g in games if _effective_date(g, year) <= cutoff_date]

        games = sorted(
            games,
            key=lambda g: (_effective_date(g, year), g.game_id),
        )

        has_tournament = year not in _NO_TOURNAMENT_SEASONS
        return SeasonGames(year=year, games=games, has_tournament=has_tournament)

    def iter_games_by_date(
        self,
        year: int,
        cutoff_date: datetime.date | None = None,
    ) -> Iterator[list[Game]]:
        """Yield batches of games grouped by calendar date, in chronological order.

        Each yielded list contains all games played on a single calendar date.
        Dates with no games are skipped.  Applies the same ``cutoff_date``
        semantics as :meth:`get_chronological_season`.

        Args:
            year: Season year.
            cutoff_date: Optional temporal cutoff (must not be in the future).

        Yields:
            Non-empty ``list[Game]`` for each calendar date, in ascending order.
        """
        season_games = self.get_chronological_season(year, cutoff_date)
        by_date: dict[datetime.date, list[Game]] = defaultdict(list)
        for game in season_games.games:
            by_date[_effective_date(game, year)].append(game)
        for date in sorted(by_date.keys()):
            yield by_date[date]
