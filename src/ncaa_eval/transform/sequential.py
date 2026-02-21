"""Sequential feature transformations for NCAA basketball game data.

Provides rolling windows, EWMA, momentum, streak, per-possession, and
Four Factor features computed from chronologically ordered game data.

* :class:`DetailedResultsLoader` — loads box-score CSVs and provides
  per-team, per-season game views in long format.
* :class:`SequentialTransformer` — orchestrates all sequential feature
  computation steps in temporal order without data leakage.

Design invariants:
- No imports from ``ncaa_eval.ingest`` — pure CSV-loading transform layer.
- No ``df.iterrows()`` — vectorized pandas operations throughout.
- ``mypy --strict`` compliant: all types fully annotated.
- No hardcoded data paths — accept Path parameters.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_COUNTING_STATS: tuple[str, ...] = (
    "fgm",
    "fga",
    "fgm3",
    "fga3",
    "ftm",
    "fta",
    "oreb",
    "dreb",
    "ast",
    "to",
    "stl",
    "blk",
    "pf",
    "score",
    "opp_score",
)

_LONG_COLS: tuple[str, ...] = (
    "season",
    "day_num",
    "team_id",
    "opp_id",
    "won",
    "loc_encoded",
    "num_ot",
    "is_tournament",
    "score",
    "opp_score",
    "fgm",
    "fga",
    "fgm3",
    "fga3",
    "ftm",
    "fta",
    "oreb",
    "dreb",
    "ast",
    "to",
    "stl",
    "blk",
    "pf",
    "opp_oreb",
    "opp_dreb",
)


# ---------------------------------------------------------------------------
# Wide-to-long reshape helper
# ---------------------------------------------------------------------------


def _reshape_to_long(df: pd.DataFrame, is_tournament: bool) -> pd.DataFrame:
    """Reshape a W/L-columnar game DataFrame to long (per-team) format.

    Each input row (one game) becomes two output rows — one per team.
    Uses vectorized rename + concat; no iterrows.

    Args:
        df: Raw game DataFrame with W/L prefixed columns.
        is_tournament: True for tournament games, False for regular season.

    Returns:
        Long-format DataFrame with one row per (team, game).
    """
    common_cols = {"Season": "season", "DayNum": "day_num", "NumOT": "num_ot"}

    w_rename = {
        "WTeamID": "team_id",
        "LTeamID": "opp_id",
        "WScore": "score",
        "LScore": "opp_score",
        "WFGM": "fgm",
        "WFGA": "fga",
        "WFGM3": "fgm3",
        "WFGA3": "fga3",
        "WFTM": "ftm",
        "WFTA": "fta",
        "WOR": "oreb",
        "WDR": "dreb",
        "WAst": "ast",
        "WTO": "to",
        "WStl": "stl",
        "WBlk": "blk",
        "WPF": "pf",
        "LOR": "opp_oreb",
        "LDR": "opp_dreb",
    }
    l_rename = {
        "LTeamID": "team_id",
        "WTeamID": "opp_id",
        "LScore": "score",
        "WScore": "opp_score",
        "LFGM": "fgm",
        "LFGA": "fga",
        "LFGM3": "fgm3",
        "LFGA3": "fga3",
        "LFTM": "ftm",
        "LFTA": "fta",
        "LOR": "oreb",
        "LDR": "dreb",
        "LAst": "ast",
        "LTO": "to",
        "LStl": "stl",
        "LBlk": "blk",
        "LPF": "pf",
        "WOR": "opp_oreb",
        "WDR": "opp_dreb",
    }

    w_df = df.rename(columns={**common_cols, **w_rename})
    w_df["won"] = True
    # WLoc from winner's perspective: H=winner at home, A=winner away, N=neutral
    w_df["loc_encoded"] = df["WLoc"].map({"H": 1, "A": -1, "N": 0})

    l_df = df.rename(columns={**common_cols, **l_rename})
    l_df["won"] = False
    # For loser: invert H/A (winner home → loser away)
    l_df["loc_encoded"] = df["WLoc"].map({"H": -1, "A": 1, "N": 0})

    for side in (w_df, l_df):
        side["is_tournament"] = is_tournament

    return pd.concat(
        [w_df[list(_LONG_COLS)], l_df[list(_LONG_COLS)]],
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# DetailedResultsLoader
# ---------------------------------------------------------------------------


class DetailedResultsLoader:
    """Loads detailed box-score results and provides per-team game views.

    Reads ``MRegularSeasonDetailedResults.csv`` and
    ``MNCAATourneyDetailedResults.csv`` into a combined long-format DataFrame
    with one row per (team, game).

    Box-score stats are only available from the 2003 season onwards.
    Pre-2003 seasons return empty DataFrames from :meth:`get_team_season`.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df  # Long-format, all seasons

    @classmethod
    def from_csvs(cls, regular_path: Path, tourney_path: Path) -> DetailedResultsLoader:
        """Construct a loader from the two Kaggle detailed-results CSV paths.

        Args:
            regular_path: Path to ``MRegularSeasonDetailedResults.csv``.
            tourney_path: Path to ``MNCAATourneyDetailedResults.csv``.

        Returns:
            :class:`DetailedResultsLoader` instance with combined data.
        """
        reg_df = pd.read_csv(regular_path)
        tur_df = pd.read_csv(tourney_path)
        long = pd.concat(
            [
                _reshape_to_long(reg_df, is_tournament=False),
                _reshape_to_long(tur_df, is_tournament=True),
            ],
            ignore_index=True,
        )
        return cls(long)

    def get_season_long_format(self, season: int) -> pd.DataFrame:
        """Return all games for a season in long format.

        Args:
            season: Season year (e.g., 2023).

        Returns:
            DataFrame sorted by ``(day_num, team_id)``, reset index.
        """
        mask = self._df["season"] == season
        return self._df[mask].sort_values(["day_num", "team_id"]).reset_index(drop=True)

    def get_team_season(self, team_id: int, season: int) -> pd.DataFrame:
        """Return all games for one team in one season, sorted by day_num.

        Args:
            team_id: Canonical Kaggle TeamID integer.
            season: Season year (e.g., 2023).

        Returns:
            DataFrame sorted by ``day_num`` ascending, reset index.
            Returns empty DataFrame if team or season not found.
        """
        mask = (self._df["team_id"] == team_id) & (self._df["season"] == season)
        return self._df[mask].sort_values("day_num").reset_index(drop=True)


# ---------------------------------------------------------------------------
# OT rescaling helper
# ---------------------------------------------------------------------------


def apply_ot_rescaling(
    team_games: pd.DataFrame,
    stats: tuple[str, ...] = _COUNTING_STATS,
) -> pd.DataFrame:
    """Rescale all counting stats to 40-minute equivalent for OT games.

    Applies: ``stat_adj = stat × 40 / (40 + 5 × num_ot)``
    Regulation games (num_ot=0) are unchanged (multiplier = 1.0).

    Returns a copy; does not modify the input DataFrame in-place.

    Args:
        team_games: Per-team game DataFrame containing a ``num_ot`` column.
        stats: Tuple of stat column names to rescale.

    Returns:
        Copy of ``team_games`` with rescaled stat columns.
    """
    df = team_games.copy()
    multiplier = 40.0 / (40.0 + 5.0 * df["num_ot"])
    available = [s for s in stats if s in df.columns]
    df[available] = df[available].mul(multiplier, axis=0)
    return df


# ---------------------------------------------------------------------------
# Time-decay weighting helper
# ---------------------------------------------------------------------------


def compute_game_weights(
    day_nums: pd.Series,
    reference_day_num: int | None = None,
) -> pd.Series:
    """BartTorvik time-decay weights: 1% per day after 40 days old; floor 60%.

    Formula: ``weight = max(0.6, 1 − 0.01 × max(0, days_ago − 40))``

    Args:
        day_nums: Series of game day numbers (ascending order).
        reference_day_num: Reference point for ``days_ago``.
            Defaults to ``max(day_nums)``.

    Returns:
        Series of weights in [0.6, 1.0] for each game.
    """
    if day_nums.empty:
        return pd.Series([], dtype=float)
    ref = int(day_nums.max()) if reference_day_num is None else reference_day_num
    days_ago = ref - day_nums
    weight = (1.0 - 0.01 * (days_ago - 40).clip(lower=0)).clip(lower=0.6)
    return weight


# ---------------------------------------------------------------------------
# Rolling window features
# ---------------------------------------------------------------------------


def compute_rolling_stats(
    team_games: pd.DataFrame,
    windows: list[int],
    stats: tuple[str, ...],
    weights: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute rolling mean features for all specified windows and stats.

    No future data leakage: rolling window at position i only uses
    rows at positions ≤ i (pandas ``rolling`` default closed='right').

    Args:
        team_games: Per-team game DataFrame (sorted by day_num ascending).
        windows: List of window sizes (e.g., [5, 10, 20]).
        stats: Tuple of stat column names.
        weights: Optional per-game weights for weighted rolling mean.

    Returns:
        DataFrame with columns ``rolling_{w}_{stat}`` and
        ``rolling_full_{stat}`` (expanding mean).
    """
    result = {}

    for w in windows:
        for stat in stats:
            if stat not in team_games.columns:
                continue
            col = team_games[stat]
            if weights is not None:
                # Weighted rolling mean: sum(stat*w)/sum(w) over window
                num = (col * weights).rolling(w, min_periods=1).sum()
                den = weights.rolling(w, min_periods=1).sum()
                result[f"rolling_{w}_{stat}"] = num / den
            else:
                result[f"rolling_{w}_{stat}"] = col.rolling(w, min_periods=1).mean()

    # Full-season aggregate (expanding mean)
    for stat in stats:
        if stat in team_games.columns:
            result[f"rolling_full_{stat}"] = team_games[stat].expanding().mean()

    return pd.DataFrame(result, index=team_games.index)


# ---------------------------------------------------------------------------
# EWMA features
# ---------------------------------------------------------------------------


def compute_ewma_stats(
    team_games: pd.DataFrame,
    alphas: list[float],
    stats: tuple[str, ...],
) -> pd.DataFrame:
    """Compute EWMA features for all specified alphas and stats.

    Uses ``adjust=False`` for standard exponential smoothing:
    ``value_t = α × obs_t + (1−α) × value_{t−1}``

    Args:
        team_games: Per-team game DataFrame (sorted by day_num ascending).
        alphas: List of smoothing factors (e.g., [0.15, 0.20]).
        stats: Tuple of stat column names.

    Returns:
        DataFrame with columns ``ewma_{alpha_str}_{stat}`` where
        ``alpha_str`` replaces the decimal point with 'p'
        (e.g., ``ewma_0p15_score``).
    """
    result = {}
    for alpha in alphas:
        alpha_str = f"{alpha:.2f}".replace(".", "p")
        for stat in stats:
            if stat in team_games.columns:
                result[f"ewma_{alpha_str}_{stat}"] = team_games[stat].ewm(alpha=alpha, adjust=False).mean()
    return pd.DataFrame(result, index=team_games.index)


# ---------------------------------------------------------------------------
# Momentum feature
# ---------------------------------------------------------------------------


def compute_momentum(
    team_games: pd.DataFrame,
    alpha_fast: float,
    alpha_slow: float,
    stats: tuple[str, ...],
) -> pd.DataFrame:
    """Compute ewma_fast − ewma_slow momentum for each stat.

    Positive momentum means recent performance is above the longer-term
    trend (improving form into tournament).

    Args:
        team_games: Per-team game DataFrame (sorted by day_num ascending).
        alpha_fast: Fast EWMA smoothing factor (larger → more reactive).
        alpha_slow: Slow EWMA smoothing factor (smaller → smoother baseline).
        stats: Tuple of stat column names.

    Returns:
        DataFrame with columns ``momentum_{stat}``.
    """
    available = [s for s in stats if s in team_games.columns]
    fast = team_games[available].ewm(alpha=alpha_fast, adjust=False).mean()
    slow = team_games[available].ewm(alpha=alpha_slow, adjust=False).mean()
    result = fast - slow
    result.columns = [f"momentum_{s}" for s in result.columns]
    return result


# ---------------------------------------------------------------------------
# Streak feature
# ---------------------------------------------------------------------------


def compute_streak(won: pd.Series) -> pd.Series:
    """Compute signed win/loss streak.

    Returns +N for a winning streak of N games, −N for a losing streak.
    Vectorized using cumsum-based grouping; no iterrows.

    Args:
        won: Boolean Series of game outcomes (True = win), sorted by day_num.

    Returns:
        Integer Series named ``"streak"``.
    """
    if won.empty:
        return pd.Series([], dtype=int, name="streak")

    # Force group break at position 0 by filling NaN shift with the opposite
    group = (won != won.shift(fill_value=not bool(won.iloc[0]))).cumsum()
    streak_len = won.groupby(group).cumcount() + 1
    return streak_len.where(won, -streak_len).rename("streak")


# ---------------------------------------------------------------------------
# Per-possession normalization
# ---------------------------------------------------------------------------


def compute_possessions(team_games: pd.DataFrame) -> pd.Series:
    """Compute possession count: FGA − OR + TO + 0.44 × FTA.

    Zero or negative possession counts (rare in short fixtures) are
    replaced with NaN to prevent division-by-zero downstream.

    Args:
        team_games: Per-team game DataFrame with box-score columns.

    Returns:
        Series named ``"possessions"``.
    """
    poss = team_games["fga"] - team_games["oreb"] + team_games["to"] + 0.44 * team_games["fta"]
    # Guard: 0 or negative possessions → NaN
    return poss.where(poss > 0, other=float("nan")).rename("possessions")


def compute_per_possession_stats(
    team_games: pd.DataFrame,
    stats: tuple[str, ...],
    possessions: pd.Series,
) -> pd.DataFrame:
    """Normalize counting stats by possessions (per-100 possessions).

    Args:
        team_games: Per-team game DataFrame.
        stats: Tuple of stat column names to normalize.
        possessions: Series of possession counts (NaN for guard rows).

    Returns:
        DataFrame with columns ``{stat}_per100``.
    """
    result = {}
    for stat in stats:
        if stat in team_games.columns:
            result[f"{stat}_per100"] = team_games[stat] * 100.0 / possessions
    return pd.DataFrame(result, index=team_games.index)


# ---------------------------------------------------------------------------
# Four Factors
# ---------------------------------------------------------------------------


def compute_four_factors(
    team_games: pd.DataFrame,
    possessions: pd.Series,
) -> pd.DataFrame:
    """Compute Dean Oliver's Four Factors efficiency ratios.

    - ``efg_pct``: Effective field goal % = (FGM + 0.5 × FGM3) / FGA
    - ``orb_pct``: Offensive rebound % = OR / (OR + opp_DR)
    - ``ftr``: Free throw rate = FTA / FGA
    - ``to_pct``: Turnover % = TO / possessions

    All denominators are guarded against zero (returns NaN when zero).

    Args:
        team_games: Per-team game DataFrame with box-score columns.
        possessions: Series of possession counts (used for TO%).

    Returns:
        DataFrame with columns ``["efg_pct", "orb_pct", "ftr", "to_pct"]``.
    """
    fga = team_games["fga"].replace(0, float("nan"))
    orb_den = (team_games["oreb"] + team_games["opp_dreb"]).replace(0, float("nan"))
    return pd.DataFrame(
        {
            "efg_pct": (team_games["fgm"] + 0.5 * team_games["fgm3"]) / fga,
            "orb_pct": team_games["oreb"] / orb_den,
            "ftr": team_games["fta"] / fga,
            "to_pct": team_games["to"] / possessions,
        },
        index=team_games.index,
    )


# ---------------------------------------------------------------------------
# SequentialTransformer
# ---------------------------------------------------------------------------


class SequentialTransformer:
    """Orchestrates all sequential feature computation steps.

    Applies OT rescaling, time-decay weighting, rolling windows, EWMA,
    momentum, streak, per-possession normalization, and Four Factors to
    a per-team game history in chronological order.

    All features respect temporal ordering — no feature for game N uses
    data from games N+1 or later.
    """

    def __init__(
        self,
        windows: list[int] | None = None,
        alphas: list[float] | None = None,
        alpha_fast: float = 0.20,
        alpha_slow: float = 0.10,
        stats: tuple[str, ...] | None = None,
    ) -> None:
        """Initialise with optional custom parameters.

        Args:
            windows: Rolling window sizes. Defaults to [5, 10, 20].
            alphas: EWMA smoothing factors. Defaults to [0.15, 0.20].
            alpha_fast: Fast EWMA alpha for momentum. Defaults to 0.20.
            alpha_slow: Slow EWMA alpha for momentum. Defaults to 0.10.
            stats: Counting stat columns. Defaults to ``_COUNTING_STATS``.
        """
        self._windows: list[int] = windows if windows is not None else [5, 10, 20]
        self._alphas: list[float] = alphas if alphas is not None else [0.15, 0.20]
        self._alpha_fast = alpha_fast
        self._alpha_slow = alpha_slow
        self._stats: tuple[str, ...] = stats if stats is not None else _COUNTING_STATS

    def transform(
        self,
        team_games: pd.DataFrame,
        reference_day_num: int | None = None,
    ) -> pd.DataFrame:
        """Compute all sequential features for a team's game history.

        Input must be sorted by ``day_num`` ascending to ensure temporal
        integrity (no future data leakage).

        Orchestration order (critical for correctness):
        1. OT rescaling (before any aggregation)
        2. Time-decay weights
        3. Rolling stats (on OT-rescaled stats, with weights)
        4. EWMA (on OT-rescaled stats)
        5. Momentum
        6. Streak (on original won column)
        7. Possessions + per-possession stats
        8. Four Factors

        Args:
            team_games: Per-team game DataFrame sorted by ``day_num``.
            reference_day_num: Reference day for time-decay weights.
                Defaults to the last game's ``day_num``.

        Returns:
            New DataFrame with all feature columns appended to originals.
            Preserves input row order.
        """
        if team_games.empty:
            return team_games.copy()

        # Step 1: OT rescaling (before any aggregation)
        scaled = apply_ot_rescaling(team_games, stats=self._stats)

        # Step 2: Time-decay weights (computed on original day_nums)
        weights = compute_game_weights(team_games["day_num"], reference_day_num)

        # Step 3: Rolling features (on OT-rescaled stats, with time-decay)
        rolling = compute_rolling_stats(scaled, self._windows, self._stats, weights)

        # Step 4: EWMA (on OT-rescaled stats)
        ewma = compute_ewma_stats(scaled, self._alphas, self._stats)

        # Step 5: Momentum
        momentum = compute_momentum(scaled, self._alpha_fast, self._alpha_slow, self._stats)

        # Step 6: Streak (on original won column, not rescaled)
        streak = compute_streak(team_games["won"])

        # Step 7: Possessions + per-possession stats
        possessions = compute_possessions(scaled)
        per_poss = compute_per_possession_stats(scaled, self._stats, possessions)

        # Step 8: Four Factors
        four_factors = compute_four_factors(scaled, possessions)

        return pd.concat(
            [
                team_games,
                rolling,
                ewma,
                momentum,
                streak.to_frame(),
                per_poss,
                four_factors,
                possessions.to_frame(),
            ],
            axis=1,
        )
