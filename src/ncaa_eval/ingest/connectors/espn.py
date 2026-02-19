"""ESPN data source connector backed by the cbbpy scraper library.

The :class:`EspnConnector` fetches current/recent season game data from ESPN
via cbbpy.  It does **not** provide team or season master data — those come
exclusively from the Kaggle connector.  A ``team_name_to_id`` mapping (built
from Kaggle's MTeams.csv) is required for translating ESPN team names into
Kaggle integer IDs.
"""

from __future__ import annotations

import datetime
import logging
from typing import Literal, cast

import cbbpy.mens_scraper as ms  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]
from rapidfuzz import fuzz

from ncaa_eval.ingest.connectors.base import (
    Connector,
    DataFormatError,
)
from ncaa_eval.ingest.schema import Game, Season, Team

logger = logging.getLogger(__name__)

# Minimum fuzzy-match score (0–100) for team name resolution.
_FUZZY_THRESHOLD = 80

# Expected columns from cbbpy get_team_schedule() output.
_SCHEDULE_COLUMNS = {"game_id", "game_day", "game_result", "team", "opponent"}


def _parse_game_result(result_str: str) -> tuple[int, int] | None:
    """Parse a cbbpy ``game_result`` string like ``'W 75-60'``.

    Returns ``(team_score, opponent_score)`` or ``None`` if unparseable.
    """
    if not isinstance(result_str, str) or not result_str.strip():
        return None
    parts = result_str.strip().split()
    if len(parts) != 2:
        return None
    scores = parts[1].split("-")
    if len(scores) != 2:
        return None
    try:
        return int(scores[0]), int(scores[1])
    except ValueError:
        return None


def _resolve_team_id(
    name: str,
    mapping: dict[str, int],
) -> int | None:
    """Resolve an ESPN team name to a Kaggle team ID.

    Tries exact match first, then falls back to fuzzy matching via rapidfuzz.
    """
    # Exact match (case-insensitive).
    lower_map = {k.lower(): v for k, v in mapping.items()}
    exact = lower_map.get(name.lower())
    if exact is not None:
        return exact

    # Fuzzy match.
    best_score = 0.0
    best_id: int | None = None
    for known_name, tid in mapping.items():
        score = fuzz.ratio(name.lower(), known_name.lower())
        if score > best_score:
            best_score = score
            best_id = tid
    if best_score >= _FUZZY_THRESHOLD and best_id is not None:
        return best_id

    logger.warning("espn: no team ID match for '%s' (best score: %.0f)", name, best_score)
    return None


class EspnConnector(Connector):
    """Connector for ESPN game data via the cbbpy scraper.

    Args:
        team_name_to_id: Mapping from team name strings to Kaggle TeamIDs.
        season_day_zeros: Mapping from season year to DayZero date.
    """

    def __init__(
        self,
        team_name_to_id: dict[str, int],
        season_day_zeros: dict[int, datetime.date],
    ) -> None:
        self._team_name_to_id = team_name_to_id
        self._season_day_zeros = season_day_zeros

    # -- Not supported: Teams and Seasons come from Kaggle only -------------

    def fetch_teams(self) -> list[Team]:
        """Not supported — teams come from Kaggle exclusively."""
        raise NotImplementedError("EspnConnector does not provide team data")

    def fetch_seasons(self) -> list[Season]:
        """Not supported — seasons come from Kaggle exclusively."""
        raise NotImplementedError("EspnConnector does not provide season data")

    # -- Games --------------------------------------------------------------

    def fetch_games(self, season: int) -> list[Game]:
        """Fetch game results for *season* from ESPN via cbbpy.

        Attempts ``get_games_season()`` first; falls back to per-team
        ``get_team_schedule()`` if the season-wide call fails.
        """
        df = self._fetch_schedule_df(season)
        if df is None or df.empty:
            return []
        return self._parse_schedule_df(df, season)

    # -- internal -----------------------------------------------------------

    def _fetch_schedule_df(self, season: int) -> pd.DataFrame | None:
        """Try to load a season schedule DataFrame from cbbpy."""
        # Attempt 1: season-wide call.
        try:
            result = ms.get_games_season(season)
            if isinstance(result, tuple):
                df = result[0]  # game info is the first element
            else:
                df = result
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception:
            logger.info("espn: get_games_season(%d) failed, trying per-team fallback", season)

        # Attempt 2: per-team schedule fallback.
        return self._fetch_per_team(season)

    def _fetch_per_team(self, season: int) -> pd.DataFrame | None:
        """Fetch schedules for each team in the mapping and concatenate."""
        frames: list[pd.DataFrame] = []
        for team_name in self._team_name_to_id:
            try:
                df = ms.get_team_schedule(team_name, season)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    frames.append(df)
            except Exception:
                logger.debug("espn: get_team_schedule('%s', %d) failed", team_name, season)
                continue
        if not frames:
            return None
        combined = pd.concat(frames, ignore_index=True)
        # Deduplicate by ESPN game_id (each game appears in both teams' schedules).
        if "game_id" in combined.columns:
            combined = combined.drop_duplicates(subset=["game_id"])
        return combined

    def _parse_schedule_df(self, df: pd.DataFrame, season: int) -> list[Game]:
        """Convert a cbbpy schedule DataFrame into Game models."""
        missing = _SCHEDULE_COLUMNS - set(df.columns)
        if missing:
            msg = f"espn: schedule DataFrame missing columns: {sorted(missing)}"
            raise DataFormatError(msg)

        day_zero = self._season_day_zeros.get(season)
        games: list[Game] = []
        seen_ids: set[str] = set()

        for _, row in df.iterrows():
            espn_game_id = str(row["game_id"])
            game_id = f"espn_{espn_game_id}"
            if game_id in seen_ids:
                continue
            seen_ids.add(game_id)

            # Parse scores from game_result.
            parsed = _parse_game_result(str(row.get("game_result", "")))
            if parsed is None:
                logger.debug("espn: skipping game %s — unparseable result", espn_game_id)
                continue
            team_score, opp_score = parsed

            # Resolve team IDs.
            team_name = str(row["team"])
            opp_name = str(row["opponent"])
            team_tid = _resolve_team_id(team_name, self._team_name_to_id)
            opp_tid = _resolve_team_id(opp_name, self._team_name_to_id)
            if team_tid is None or opp_tid is None:
                logger.warning("espn: skipping game %s — unresolved team(s)", espn_game_id)
                continue
            if team_tid == opp_tid:
                logger.warning("espn: skipping game %s — same team ID for both sides", espn_game_id)
                continue

            # Determine winner/loser ordering.
            if team_score > opp_score:
                w_team_id, l_team_id = team_tid, opp_tid
                w_score, l_score = team_score, opp_score
            elif opp_score > team_score:
                w_team_id, l_team_id = opp_tid, team_tid
                w_score, l_score = opp_score, team_score
            else:
                # Tie — shouldn't happen in basketball but skip gracefully.
                logger.warning("espn: skipping game %s — tied scores", espn_game_id)
                continue

            # Parse date and compute day_num.
            game_date = self._parse_date(row.get("game_day"))
            day_num = 0
            if game_date is not None and day_zero is not None:
                day_num = (game_date - day_zero).days

            # Determine location.
            loc = self._infer_loc(row, team_tid, w_team_id)

            games.append(
                Game(
                    game_id=game_id,
                    season=season,
                    day_num=day_num,
                    date=game_date,
                    w_team_id=w_team_id,
                    l_team_id=l_team_id,
                    w_score=w_score,
                    l_score=l_score,
                    loc=loc,
                ),
            )
        return games

    @staticmethod
    def _parse_date(value: object) -> datetime.date | None:
        """Best-effort date parsing from cbbpy game_day values."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        try:
            ts = pd.Timestamp(value)
            if pd.isna(ts):
                return None
            return cast("datetime.date", ts.date())
        except Exception:
            return None

    @staticmethod
    def _infer_loc(
        row: pd.Series,
        team_tid: int,
        w_team_id: int,
    ) -> Literal["H", "A", "N"]:
        """Infer game location from available ESPN context.

        Falls back to ``"N"`` (neutral) when location cannot be determined.
        """
        # Some DataFrames include a 'home_away' or 'is_neutral' column.
        if "is_neutral" in row.index:
            val = row["is_neutral"]
            if val is True or str(val).lower() in ("true", "1", "yes"):
                return "N"

        if "home_away" in row.index:
            ha = str(row["home_away"]).lower()
            if ha == "home":
                # The row's team was home.
                return "H" if team_tid == w_team_id else "A"
            if ha == "away":
                return "A" if team_tid == w_team_id else "H"

        # Default to neutral when ambiguous.
        return "N"
