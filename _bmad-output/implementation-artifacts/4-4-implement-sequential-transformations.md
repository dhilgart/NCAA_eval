# Story 4.4: Implement Sequential Transformations

Status: ready-for-dev

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a data scientist,
I want rolling windows, EWMA, momentum, streak, per-possession, and Four Factor features computed from chronologically ordered game data,
so that I can capture recent team form, efficiency, and trends as predictive features without data leakage.

## Acceptance Criteria

1. **Given** chronological game data (detailed box scores) is available, **When** the developer applies sequential transformations to a team's game history, **Then** rolling averages are computed over configurable windows of **5, 10, and 20 games** (plus full-season aggregate) for all box-score counting stats; all three window sizes are parallel feature columns — not competing features, but modeler-configurable parameters of the same building block.
2. **And** all sequential features respect chronological ordering — no feature for game N uses data from games N+1 or later (no future data leakage).
3. **And** features are computed using vectorized operations (numpy/pandas) per NFR1 — no Python `for` loops iterating over DataFrame rows.
4. **And** EWMA (Exponentially Weighted Moving Average) is implemented with configurable α (range 0.10–0.30; recommended start α=0.15–0.20 mapping to effective window of 9–12 games); uses `pandas.DataFrame.ewm(alpha=α).mean()` per team per season.
5. **And** a momentum/trajectory feature is produced as `ewma_fast − ewma_slow` (rate of change of efficiency; positive = improving form into tournament) for each stat.
6. **And** win/loss streaks are encoded as a signed integer: `+N` for winning streak of N games, `−N` for losing streak, capturing pure win/loss sequence dynamics independent of efficiency magnitude.
7. **And** `rescale_overtime(score, num_ot)` from `serving.py` is applied to raw score columns before any rolling/EWMA aggregation (normalizes OT games to 40-minute equivalent); box-score counting stats are similarly rescaled using the same formula.
8. **And** time-decay game weighting applies the BartTorvik formula before rolling aggregations: `weight = max(0.6, 1 − 0.01 × max(0, days_ago − 40))`; the weighted rolling mean uses per-game weights in numerator and denominator; `days_ago` is computed as `reference_day_num − game_day_num` where `reference_day_num` defaults to the last game's `day_num` if not provided.
9. **And** per-possession normalization is applied to all counting stats: `possessions = FGA − OR + TO + 0.44 × FTA`; stat values are divided by possession count to remove pace confound.
10. **And** Four Factors are computed: `eFG% = (FGM + 0.5 × FGM3) / FGA`, `ORB% = OR / (OR + opponent_DR)`, `FTR = FTA / FGA`, `TO% = TO / possessions`.
11. **And** home court encoding converts `loc` (from WLoc field) to a numeric feature: H=+1, A=−1, N=0 for the winning team; inverted (H=−1, A=+1, N=0) for the losing team.
12. **And** edge cases are handled gracefully: `min_periods=1` for rolling windows (early-season games with fewer than N prior games), zero-possession games (guard against divide-by-zero), and missing detailed stats for pre-2003 seasons (compact stats only).
13. **And** sequential transformations are covered by unit tests in `tests/unit/test_sequential.py` validating correctness and temporal integrity.

## Tasks / Subtasks

- [ ] Task 1: Create `src/ncaa_eval/transform/sequential.py` with constants and `DetailedResultsLoader` (AC: 1–12)
  - [ ] 1.1: Add module header with `from __future__ import annotations`, imports, and module-level logger
  - [ ] 1.2: Define `_COUNTING_STATS: tuple[str, ...]` constant — the per-team box-score columns computed from the long-format reshape: `("fgm", "fga", "fgm3", "fga3", "ftm", "fta", "oreb", "dreb", "ast", "to", "stl", "blk", "pf", "score", "opp_score")`
  - [ ] 1.3: Define `_LONG_COLS: tuple[str, ...]` — all columns present in the long-format team-perspective DataFrame: `("season", "day_num", "team_id", "opp_id", "won", "loc_encoded", "num_ot", "is_tournament", "score", "opp_score", "fgm", "fga", "fgm3", "fga3", "ftm", "fta", "oreb", "dreb", "ast", "to", "stl", "blk", "pf", "opp_oreb", "opp_dreb")`
  - [ ] 1.4: Implement `DetailedResultsLoader` class — wraps a combined long-format DataFrame
  - [ ] 1.5: `classmethod from_csvs(cls, regular_path: Path, tourney_path: Path) -> DetailedResultsLoader` — loads both CSVs (each with columns Season,DayNum,WTeamID,WScore,LTeamID,LScore,WLoc,NumOT,WFGM,...,LPF), reshapes to long format (vectorized rename+concat, no iterrows), adds `is_tournament` flag (False for regular, True for tourney)
  - [ ] 1.6: Implement `_reshape_to_long(df: pd.DataFrame, is_tournament: bool) -> pd.DataFrame` module-level helper — uses `pd.concat` of winner-side and loser-side renames; loc_encoded for winner: H→+1, A→-1, N→0; for loser: H→-1, A→+1, N→0
  - [ ] 1.7: `get_season_long_format(self, season: int) -> pd.DataFrame` — returns all games for a season (regular + tourney) in long format sorted by `(day_num, team_id)`
  - [ ] 1.8: `get_team_season(self, team_id: int, season: int) -> pd.DataFrame` — returns all games for one team in one season, sorted by `day_num`; returns empty DataFrame if team or season not found

- [ ] Task 2: Implement OT rescaling and time-decay weighting helpers (AC: 7, 8)
  - [ ] 2.1: Implement `apply_ot_rescaling(team_games: pd.DataFrame, stats: tuple[str, ...] = _COUNTING_STATS) -> pd.DataFrame` — applies `rescale_overtime(score, num_ot)` formula vectorized to all counting stats: `df[stats] = df[stats].div(df["num_ot"].mul(5).add(40).div(40), axis=0)` (equivalent to `stat × 40 / (40 + 5 × num_ot)`); returns a copy, does not modify in-place
  - [ ] 2.2: Implement `compute_game_weights(day_nums: pd.Series, reference_day_num: int | None = None) -> pd.Series` — BartTorvik decay: `days_ago = reference - day_num` where reference defaults to `day_nums.max()`; `weight = (1 - 0.01 * max(0, days_ago - 40)).clip(lower=0.6)`; fully vectorized using pandas clip

- [ ] Task 3: Implement rolling window features (AC: 1, 2, 3, 12)
  - [ ] 3.1: Implement `compute_rolling_stats(team_games: pd.DataFrame, windows: list[int], stats: tuple[str, ...], weights: pd.Series | None = None) -> pd.DataFrame` — for each window and stat, compute `rolling(window, min_periods=1).mean()`; if `weights` provided, compute weighted rolling mean using: `(stat × weight).rolling(w, min_periods=1).sum() / weight.rolling(w, min_periods=1).sum()`; output column names: `rolling_{w}_{stat}` (e.g., `rolling_5_score`, `rolling_10_fgm`); also include `rolling_full_{stat}` (expanding mean = full-season aggregate)

- [ ] Task 4: Implement EWMA and momentum features (AC: 4, 5)
  - [ ] 4.1: Implement `compute_ewma_stats(team_games: pd.DataFrame, alphas: list[float], stats: tuple[str, ...]) -> pd.DataFrame` — uses `df[stats].ewm(alpha=α, adjust=False).mean()`; output column names: `ewma_{alpha_str}_{stat}` where `alpha_str = str(α).replace(".", "p")` (e.g., `ewma_0p15_score`, `ewma_0p20_fgm`); one alpha = one set of columns; the modeler selects which alpha to use
  - [ ] 4.2: Implement `compute_momentum(team_games: pd.DataFrame, alpha_fast: float, alpha_slow: float, stats: tuple[str, ...]) -> pd.DataFrame` — `momentum_{stat} = ewma(fast) − ewma(slow)` for each stat; output column names: `momentum_{stat}` (e.g., `momentum_score`); uses `pd.DataFrame.ewm(alpha=α, adjust=False).mean()` internally

- [ ] Task 5: Implement streak features (AC: 6)
  - [ ] 5.1: Implement `compute_streak(won: pd.Series) -> pd.Series` — signed integer encoding; positive for win streaks, negative for loss streaks; vectorized using `(won != won.shift()).cumsum()` to group consecutive identical outcomes, then `groupby.cumcount() + 1` for streak length, then `.where(won, -streak_len)`; output Series named `"streak"`

- [ ] Task 6: Implement per-possession normalization and Four Factors (AC: 9, 10)
  - [ ] 6.1: Implement `compute_possessions(team_games: pd.DataFrame) -> pd.Series` — `FGA − OR + TO + 0.44 × FTA`; returns a Series named `"possessions"`; guard: replace zero values with `np.nan` (use `pd.Series.replace(0, np.nan)`) to prevent division-by-zero in downstream normalization
  - [ ] 6.2: Implement `compute_per_possession_stats(team_games: pd.DataFrame, stats: tuple[str, ...], possessions: pd.Series) -> pd.DataFrame` — divides each stat column by possessions; output column names: `{stat}_per100` (multiply by 100 for interpretability: `stat × 100 / possessions`); e.g., `fgm_per100`
  - [ ] 6.3: Implement `compute_four_factors(team_games: pd.DataFrame, possessions: pd.Series) -> pd.DataFrame` — computes: `efg_pct = (fgm + 0.5 × fgm3) / fga`, `orb_pct = oreb / (oreb + opp_dreb)`, `ftr = fta / fga`, `to_pct = to / possessions`; output DataFrame with columns `["efg_pct", "orb_pct", "ftr", "to_pct"]`; guard all denominators against zero (return NaN when denominator is zero)

- [ ] Task 7: Implement top-level `SequentialTransformer` class (AC: 1–12)
  - [ ] 7.1: Define `SequentialTransformer` class with `__init__(self, windows: list[int] | None = None, alphas: list[float] | None = None, alpha_fast: float = 0.20, alpha_slow: float = 0.10, stats: tuple[str, ...] | None = None) -> None`; defaults: `windows=[5, 10, 20]`, `alphas=[0.15, 0.20]`, `stats=_COUNTING_STATS`
  - [ ] 7.2: Implement `transform(self, team_games: pd.DataFrame, reference_day_num: int | None = None) -> pd.DataFrame` — orchestrates all feature computation steps in order: (1) OT rescaling, (2) time-decay weights, (3) rolling stats (with weights), (4) EWMA, (5) momentum, (6) streak, (7) possessions + per-possession, (8) four factors; returns a new DataFrame with all feature columns appended to the original columns; preserves input row order
  - [ ] 7.3: Ensure returned DataFrame has `"streak"` column, all `"rolling_N_stat"` columns, all `"ewma_alpha_stat"` columns, all `"momentum_stat"` columns, `"possessions"`, all `"stat_per100"` columns, and four-factor columns `["efg_pct", "orb_pct", "ftr", "to_pct"]`

- [ ] Task 8: Export public API from `src/ncaa_eval/transform/__init__.py` (AC: 1–12)
  - [ ] 8.1: Import and re-export `DetailedResultsLoader`, `SequentialTransformer`, `compute_streak`, `compute_possessions`, `compute_four_factors`, `compute_game_weights` from `transform/__init__.py`
  - [ ] 8.2: Add all new names to `__all__`

- [ ] Task 9: Write unit tests in `tests/unit/test_sequential.py` (AC: 13)
  - [ ] 9.1: `test_reshape_to_long_winner_row` — fixture CSV with one game (WLoc=H), verify winner row has `loc_encoded=+1`, `won=True`, correct `fgm` value from WFGM column
  - [ ] 9.2: `test_reshape_to_long_loser_row` — same fixture, verify loser row has `loc_encoded=-1`, `won=False`, correct `fgm` value from LFGM column
  - [ ] 9.3: `test_reshape_to_long_neutral` — WLoc=N: winner row `loc_encoded=0`, loser row `loc_encoded=0`
  - [ ] 9.4: `test_detailed_results_loader_from_csvs` — fixture with 2 regular-season games + 1 tourney game; verify row count = `(2+1) × 2 = 6`; verify `is_tournament` flags
  - [ ] 9.5: `test_get_team_season_sorted` — verify returned DataFrame is sorted by `day_num` ascending
  - [ ] 9.6: `test_get_team_season_empty` — unknown `team_id` returns empty DataFrame (no exception)
  - [ ] 9.7: `test_apply_ot_rescaling_regulation` — `num_ot=0`: scores unchanged (`40/40 = 1.0` multiplier)
  - [ ] 9.8: `test_apply_ot_rescaling_one_ot` — `num_ot=1`: score rescaled to `score × 40/45`
  - [ ] 9.9: `test_compute_game_weights_no_decay` — game played 0 days ago: weight = 1.0; game 40 days ago: weight = 1.0; game 41 days ago: weight = 0.99
  - [ ] 9.10: `test_compute_game_weights_floor` — game 140 days ago: weight = 0.6 (floor enforced)
  - [ ] 9.11: `test_rolling_stats_window_5` — 10-game fixture; verify `rolling_5_score` at position 0 = score[0], at position 4 = mean(score[0:5])
  - [ ] 9.12: `test_rolling_stats_min_periods` — single-game team history: `rolling_5_score` is not NaN (min_periods=1)
  - [ ] 9.13: `test_rolling_full_is_expanding_mean` — `rolling_full_score` at each row equals `score.expanding().mean()` to that row
  - [ ] 9.14: `test_rolling_no_future_leakage` — for position i, `rolling_5_score` only uses games at positions ≤ i (not i+1)
  - [ ] 9.15: `test_ewma_stats_alpha` — 5-game fixture; verify `ewma_0p20_score` at row 1 equals `alpha × score[1] + (1-alpha) × score[0]` (with `adjust=False`)
  - [ ] 9.16: `test_compute_momentum_positive_improving` — increasing score series: `momentum_score` is positive (fast ewma > slow ewma as recent games are higher)
  - [ ] 9.17: `test_compute_streak_win_streak` — 3 consecutive wins: streak = +3 at position 2
  - [ ] 9.18: `test_compute_streak_loss_streak` — 3 consecutive losses: streak = -3 at position 2
  - [ ] 9.19: `test_compute_streak_reset` — WWLWW: streaks are +1, +2, -1, +1, +2
  - [ ] 9.20: `test_compute_possessions_formula` — known FGA, OR, TO, FTA values → verify `possessions = FGA - OR + TO + 0.44 × FTA`
  - [ ] 9.21: `test_compute_possessions_zero_guard` — if possessions = 0 for a row, result is NaN (no ZeroDivisionError downstream)
  - [ ] 9.22: `test_compute_per_possession_stats` — known score and possessions → verify `score_per100 = score × 100 / possessions`
  - [ ] 9.23: `test_compute_four_factors_efg` — known FGM, FGM3, FGA → verify `efg_pct = (FGM + 0.5×FGM3) / FGA`
  - [ ] 9.24: `test_compute_four_factors_orb` — known oreb, opp_dreb → verify `orb_pct = oreb / (oreb + opp_dreb)`
  - [ ] 9.25: `test_compute_four_factors_zero_fga` — FGA = 0: `efg_pct = NaN` (no exception)
  - [ ] 9.26: `test_sequential_transformer_transform_columns` — run `SequentialTransformer().transform(team_games)` on a 10-game fixture; verify all expected column groups exist: streak, rolling_5/10/20/full, ewma_0p15/0p20, momentum, possessions, per100, four factors
  - [ ] 9.27: `test_sequential_transformer_preserves_row_count` — output has same number of rows as input

- [ ] Task 10: Commit (AC: all)
  - [ ] 10.1: Stage `src/ncaa_eval/transform/sequential.py`, `src/ncaa_eval/transform/__init__.py`, `tests/unit/test_sequential.py`
  - [ ] 10.2: Commit: `feat(transform): implement sequential transformations (Story 4.4)`
  - [ ] 10.3: Update `_bmad-output/implementation-artifacts/sprint-status.yaml`: `4-4-implement-sequential-transformations` → `review`

## Dev Notes

### Story Nature: Third Code Story in Epic 4 — sequential.py in transform/

This is a **code story** — `mypy --strict`, Ruff, `from __future__ import annotations`, and the no-iterrows mandate all apply. No notebook deliverables.

This story delivers **sequential feature computation infrastructure** consumed by:
- Story 4.7 (stateful feature serving) — needs `SequentialTransformer` and `DetailedResultsLoader`
- Story 4.5 (graph builders) — independent, but both work on the same per-team game data
- Story 4.8 (Elo features) — shares the per-team game history pattern

### 🚨 CRITICAL: Box Score Data Is NOT In the Repository

**The single most important architectural fact for this story:**

The `Game` schema ([src/ncaa_eval/ingest/schema.py](src/ncaa_eval/ingest/schema.py)) and `ParquetRepository` only store **compact game data**:
```python
# Game schema fields:
game_id, season, day_num, date, w_team_id, l_team_id, w_score, l_score, loc, num_ot, is_tournament
```

**Box score stats (FGM, FGA, FGM3, FGA3, FTM, FTA, OR, DR, Ast, TO, Stl, Blk, PF) are NOT in the repository.** They are in the Kaggle CSV files only:
- `data/kaggle/MRegularSeasonDetailedResults.csv`
- `data/kaggle/MNCAATourneyDetailedResults.csv`

Both files have the same column structure:
```
Season, DayNum, WTeamID, WScore, LTeamID, LScore, WLoc, NumOT,
WFGM, WFGA, WFGM3, WFGA3, WFTM, WFTA, WOR, WDR, WAst, WTO, WStl, WBlk, WPF,
LFGM, LFGA, LFGM3, LFGA3, LFTM, LFTA, LOR, LDR, LAst, LTO, LStl, LBlk, LPF
```

Note: these CSVs use `WLoc` (not `Loc`), and the `DayNum` column (not `day_num`).

**Important coverage caveat:** Detailed stats are only available from **2003 onwards**. Pre-2003 seasons (1985–2002) have only compact results — no FGM/FGA/etc. The `DetailedResultsLoader` will simply return an empty or compact-only DataFrame for pre-2003 seasons. Story 4.4 implementations must handle the absence of detailed stats gracefully.

The `DetailedResultsLoader` follows the exact same pattern as normalization.py — it is a CSV loading class in the transform layer. It does NOT use the Repository or ChronologicalDataServer.

### Module Placement

**New file:** `src/ncaa_eval/transform/sequential.py`

Alongside `serving.py` and `normalization.py` in the transform layer. Per Architecture Section 9, all feature engineering belongs in `src/ncaa_eval/transform/`.

**Modified file:** `src/ncaa_eval/transform/__init__.py` — add exports for new public API.

### Wide-to-Long Reshape (Vectorized, No IterRows)

The CSVs store one row per game with W/L prefixes. We need TWO rows per game (one per team). Use `pd.concat` of two renamed DataFrames — fully vectorized:

```python
def _reshape_to_long(df: pd.DataFrame, is_tournament: bool) -> pd.DataFrame:
    """Reshape a W/L-columnar game DataFrame to long (per-team) format."""
    common_cols = {"Season": "season", "DayNum": "day_num", "NumOT": "num_ot"}

    w_rename = {
        "WTeamID": "team_id", "LTeamID": "opp_id",
        "WScore": "score", "LScore": "opp_score",
        "WFGM": "fgm", "WFGA": "fga", "WFGM3": "fgm3", "WFGA3": "fga3",
        "WFTM": "ftm", "WFTA": "fta", "WOR": "oreb", "WDR": "dreb",
        "WAst": "ast", "WTO": "to", "WStl": "stl", "WBlk": "blk", "WPF": "pf",
        "LOR": "opp_oreb", "LDR": "opp_dreb",
    }
    l_rename = {
        "LTeamID": "team_id", "WTeamID": "opp_id",
        "LScore": "score", "WScore": "opp_score",
        "LFGM": "fgm", "LFGA": "fga", "LFGM3": "fgm3", "LFGA3": "fga3",
        "LFTM": "ftm", "LFTA": "fta", "LOR": "oreb", "LDR": "dreb",
        "LAst": "ast", "LTO": "to", "LStl": "stl", "LBlk": "blk", "LPF": "pf",
        "WOR": "opp_oreb", "WDR": "opp_dreb",
    }

    w_df = df.rename(columns={**common_cols, **w_rename})
    w_df["won"] = True
    # WLoc from winner's perspective: H=home (winner at home), A=away, N=neutral
    w_df["loc_encoded"] = df["WLoc"].map({"H": 1, "A": -1, "N": 0})

    l_df = df.rename(columns={**common_cols, **l_rename})
    l_df["won"] = False
    # For loser: invert H/A (if winner was home, loser was away)
    l_df["loc_encoded"] = df["WLoc"].map({"H": -1, "A": 1, "N": 0})

    for side in (w_df, l_df):
        side["is_tournament"] = is_tournament

    return pd.concat([w_df[list(_LONG_COLS)], l_df[list(_LONG_COLS)]], ignore_index=True)
```

### DetailedResultsLoader Design

```python
class DetailedResultsLoader:
    """Loads detailed box-score results and provides per-team game views.

    Reads MRegularSeasonDetailedResults.csv and MNCAATourneyDetailedResults.csv
    into a combined long-format DataFrame with one row per (team, game).
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df  # Long-format, all seasons

    @classmethod
    def from_csvs(cls, regular_path: Path, tourney_path: Path) -> DetailedResultsLoader:
        reg_df = pd.read_csv(regular_path)
        tur_df = pd.read_csv(tourney_path)
        long = pd.concat(
            [_reshape_to_long(reg_df, is_tournament=False),
             _reshape_to_long(tur_df, is_tournament=True)],
            ignore_index=True,
        )
        return cls(long)

    def get_season_long_format(self, season: int) -> pd.DataFrame:
        mask = self._df["season"] == season
        return self._df[mask].sort_values(["day_num", "team_id"]).reset_index(drop=True)

    def get_team_season(self, team_id: int, season: int) -> pd.DataFrame:
        mask = (self._df["team_id"] == team_id) & (self._df["season"] == season)
        return self._df[mask].sort_values("day_num").reset_index(drop=True)
```

### OT Rescaling (AC 7)

```python
def apply_ot_rescaling(
    team_games: pd.DataFrame,
    stats: tuple[str, ...] = _COUNTING_STATS,
) -> pd.DataFrame:
    """Rescale all counting stats to 40-minute equivalent for OT games.

    Applies: stat_adj = stat × 40 / (40 + 5 × num_ot)
    Returns a copy of the DataFrame with rescaled columns.
    """
    df = team_games.copy()
    # Multiplier: 40 / (40 + 5 * num_ot); regulation games multiply by 1.0
    multiplier = 40.0 / (40.0 + 5.0 * df["num_ot"])
    available = [s for s in stats if s in df.columns]
    df[available] = df[available].mul(multiplier, axis=0)
    return df
```

Note: `rescale_overtime(score, num_ot)` from `serving.py` is the scalar version. For DataFrames, use the vectorized formula directly (equivalent math).

### Time-Decay Weighting (AC 8)

```python
def compute_game_weights(
    day_nums: pd.Series,
    reference_day_num: int | None = None,
) -> pd.Series:
    """BartTorvik time-decay weights: 1% per day after 40 days old; floor 60%.

    Args:
        day_nums: Series of game day numbers (ascending order).
        reference_day_num: Reference point for 'days ago'. Defaults to max(day_nums).

    Returns:
        Series of weights in [0.6, 1.0] for each game.
    """
    ref = int(day_nums.max()) if reference_day_num is None else reference_day_num
    days_ago = ref - day_nums
    weight = (1.0 - 0.01 * (days_ago - 40).clip(lower=0)).clip(lower=0.6)
    return weight
```

### Rolling Window Features (AC 1–3)

```python
def compute_rolling_stats(
    team_games: pd.DataFrame,
    windows: list[int],
    stats: tuple[str, ...],
    weights: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute rolling mean features for all specified windows and stats."""
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
```

### EWMA Features (AC 4)

```python
def compute_ewma_stats(
    team_games: pd.DataFrame,
    alphas: list[float],
    stats: tuple[str, ...],
) -> pd.DataFrame:
    """Compute EWMA features for all specified alphas and stats."""
    result = {}
    for alpha in alphas:
        alpha_str = f"{alpha:.2f}".replace(".", "p")
        for stat in stats:
            if stat in team_games.columns:
                result[f"ewma_{alpha_str}_{stat}"] = (
                    team_games[stat].ewm(alpha=alpha, adjust=False).mean()
                )
    return pd.DataFrame(result, index=team_games.index)
```

**Note on `adjust=False`**: Use `adjust=False` for standard exponential smoothing where each update is `value_t = α × obs_t + (1−α) × value_{t−1}`. This is consistent with the community EWMA implementations (α=0.15 → effective window ≈12 games).

### Momentum Feature (AC 5)

```python
def compute_momentum(
    team_games: pd.DataFrame,
    alpha_fast: float,
    alpha_slow: float,
    stats: tuple[str, ...],
) -> pd.DataFrame:
    """Compute ewma_fast - ewma_slow momentum for each stat."""
    fast = team_games[list(stats)].ewm(alpha=alpha_fast, adjust=False).mean()
    slow = team_games[list(stats)].ewm(alpha=alpha_slow, adjust=False).mean()
    result = (fast - slow)
    result.columns = [f"momentum_{s}" for s in result.columns]
    return result
```

### Streak Feature (AC 6)

```python
def compute_streak(won: pd.Series) -> pd.Series:
    """Compute signed win/loss streak.

    Returns +N for a winning streak of N, -N for a losing streak of N.
    Uses vectorized groupby-cumcount (no iterrows).
    """
    # Group consecutive same outcomes
    group = (won != won.shift(fill_value=not won.iloc[0])).cumsum()
    streak_len = won.groupby(group).cumcount() + 1
    return streak_len.where(won, -streak_len).rename("streak")
```

**Edge case**: `won.shift()` produces NaN at position 0. Use `fill_value=not won.iloc[0]` to force a "group break" at the start so the first game always starts streak count at 1. Guard against empty Series before calling.

### Per-Possession Normalization (AC 9)

```python
def compute_possessions(team_games: pd.DataFrame) -> pd.Series:
    """Compute possession count: FGA - OR + TO + 0.44 × FTA."""
    poss = (
        team_games["fga"]
        - team_games["oreb"]
        + team_games["to"]
        + 0.44 * team_games["fta"]
    )
    # Guard: 0 or negative possessions → NaN (rare but possible in short OT fixtures)
    return poss.where(poss > 0, other=float("nan")).rename("possessions")
```

### Four Factors (AC 10)

```python
def compute_four_factors(
    team_games: pd.DataFrame,
    possessions: pd.Series,
) -> pd.DataFrame:
    """Compute Dean Oliver's Four Factors efficiency ratios."""
    fga = team_games["fga"].replace(0, float("nan"))
    return pd.DataFrame({
        "efg_pct": (team_games["fgm"] + 0.5 * team_games["fgm3"]) / fga,
        "orb_pct": team_games["oreb"] / (team_games["oreb"] + team_games["opp_dreb"]).replace(0, float("nan")),
        "ftr": team_games["fta"] / fga,
        "to_pct": team_games["to"] / possessions,
    }, index=team_games.index)
```

### `SequentialTransformer.transform()` Orchestration Order

The correct ordering is critical for temporal integrity:

```python
def transform(
    self,
    team_games: pd.DataFrame,
    reference_day_num: int | None = None,
) -> pd.DataFrame:
    if team_games.empty:
        return team_games.copy()

    # Step 1: OT rescaling (before any aggregation)
    scaled = apply_ot_rescaling(team_games, stats=self._stats)

    # Step 2: Time-decay weights (computed on original day_nums)
    weights = compute_game_weights(team_games["day_num"], reference_day_num)

    # Step 3: Rolling features (on OT-rescaled stats, with time-decay weights)
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
        [team_games, rolling, ewma, momentum, streak.to_frame(), per_poss, four_factors,
         possessions.to_frame()],
        axis=1,
    )
```

### mypy Strict Compliance Notes

- All function signatures must have complete type annotations
- `pd.DataFrame` and `pd.Series` are the annotation types (no subscripts — pandas is untyped)
- `import numpy as np` requires `# type: ignore[import-untyped]` (like pandas)
- `list[int] | None` as a parameter type requires `from __future__ import annotations` at top of file
- Avoid using `float("nan")` in return type annotations — return `pd.Series` or `pd.DataFrame` as expected
- `pd.DataFrame.ewm()` is part of pandas; no type ignore needed for `.mean()` calls
- `# type: ignore[import-untyped]` needed on `import numpy as np` and `import pandas as pd`

### Architecture Guardrails (Mandatory)

1. **`from __future__ import annotations` required** — first non-comment line
2. **`mypy --strict` mandatory** — zero errors; use `# type: ignore[import-untyped]` for pandas/numpy
3. **Vectorization First** — NO `for` loops over DataFrame rows; `itertuples` acceptable only for non-vectorizable string operations (none required here)
4. **No iterrows** — use vectorized operations exclusively
5. **No direct I/O from the transform module** — `DetailedResultsLoader` loads from CSV paths passed in; it does NOT hardcode data paths
6. **No imports from `ncaa_eval.ingest.repository`** — the sequential module is a pure transform-layer component; it does NOT use the Repository (the detailed CSVs are not in the Parquet store)
7. **Do NOT import `rescale_overtime` from `serving.py`** — implement the equivalent formula directly in `sequential.py` to avoid circular import risk; both implement `score × 40 / (40 + 5 × num_ot)`

### No New Dependencies Required

All needed libraries are in `pyproject.toml`:
- `pandas` — all DataFrame operations, rolling/ewm
- `numpy` — vectorized math (`np.nan`, clip operations)

### Import Pattern for Sequential Module

```python
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)
```

### Test File Structure

**File:** `tests/unit/test_sequential.py`

Follow the same pattern as `tests/unit/test_normalization.py`:

```python
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from ncaa_eval.transform.sequential import (
    DetailedResultsLoader,
    SequentialTransformer,
    compute_four_factors,
    compute_game_weights,
    compute_momentum,
    compute_possessions,
    compute_rolling_stats,
    compute_streak,
    compute_ewma_stats,
)
```

**Fixture approach:** Build minimal fixture DataFrames in-memory using `pd.DataFrame({...})`. For `from_csvs` tests, use `tmp_path` + `df.to_csv(tmp_path / "filename.csv", index=False)`.

**Minimal fixture CSV structure for `MRegularSeasonDetailedResults.csv`:**
```python
@pytest.fixture
def regular_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame({
        "Season": [2010, 2010],
        "DayNum": [30, 40],
        "WTeamID": [1001, 1002],
        "WScore": [75, 80],
        "LTeamID": [1002, 1001],
        "LScore": [65, 70],
        "WLoc": ["H", "N"],
        "NumOT": [0, 1],
        "WFGM": [28, 30], "WFGA": [60, 62], "WFGM3": [5, 6], "WFGA3": [15, 14],
        "WFTM": [14, 10], "WFTA": [18, 12], "WOR": [12, 10], "WDR": [22, 24],
        "WAst": [15, 18], "WTO": [14, 12], "WStl": [7, 8], "WBlk": [3, 4], "WPF": [20, 18],
        "LFGM": [24, 26], "LFGA": [58, 60], "LFGM3": [4, 3], "LFGA3": [12, 13],
        "LFTM": [13, 15], "LFTA": [17, 19], "LOR": [10, 11], "LDR": [20, 21],
        "LAst": [12, 14], "LTO": [16, 15], "LStl": [6, 5], "LBlk": [2, 3], "LPF": [22, 20],
    })
    path = tmp_path / "regular.csv"
    df.to_csv(path, index=False)
    return path
```

**Markers:**
- `@pytest.mark.smoke` on fast unit tests (< 1s each)
- `@pytest.mark.unit` on all tests in this file

### Previous Story Learnings (from Stories 4.2 and 4.3)

- **`Iterator` from `collections.abc`** — UP035 compliance; not from `typing`
- **`frozen=True` dataclasses** — only prevents attribute rebinding, not mutation of mutable contents; use `tuple[str, ...]` for sequence constants (they cannot be mutated)
- **`logger = logging.getLogger(__name__)`** at module level
- **Warning capture in tests** — use `unittest.mock.patch("ncaa_eval.transform.sequential.logger")` not `caplog` (the `ncaa_eval` logger has `propagate=False`)
- **Every public method must have a test** — map every public method to a test ID during implementation; the Story 4.3 review found `composite_pca()` was the only untested method and had the worst edge-case bugs
- **Empty-DataFrame guards** — always add `if df.empty: return pd.DataFrame()` before any sklearn/ML method or heavy computation
- **Empty-input guards for arithmetic** — validate non-empty inputs before division operations that could produce ZeroDivisionError

### What NOT to Do

- **Do not** use the `ChronologicalDataServer` or `ParquetRepository` for loading detailed stats — they don't have them
- **Do not** implement graph features (PageRank, betweenness, etc.) — that belongs in Story 4.5
- **Do not** implement opponent adjustment rating systems (SRS, Ridge, Colley, Elo) — that belongs in Story 4.6
- **Do not** implement the feature serving layer (combining all features, temporal slicing) — that belongs in Story 4.7
- **Do not** use `df.iterrows()` for DataFrame processing — use vectorized pandas operations
- **Do not** hardcode the data paths (`data/kaggle/...`) in the module — accept Path parameters
- **Do not** implement Women's data (WRegularSeasonDetailedResults.csv) — Men's only for MVP
- **Do not** skip tests for any public method — Story 4.3 review established the pattern: every public method → at least one test

### Running Quality Checks

```bash
POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval ruff check .
POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval mypy --strict src/ncaa_eval tests
POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval pytest tests/unit/test_sequential.py -v
```

### Project Structure Notes

**New files:**
- `src/ncaa_eval/transform/sequential.py` — `DetailedResultsLoader`, `SequentialTransformer`, module-level helper functions

**Modified files:**
- `src/ncaa_eval/transform/__init__.py` — add exports for new public API
- `tests/unit/test_sequential.py` — new test file (create in `tests/unit/`)
- `_bmad-output/implementation-artifacts/sprint-status.yaml` — update status to `review`
- `_bmad-output/implementation-artifacts/4-4-implement-sequential-transformations.md` — this story file (Dev Agent Record section)

**No changes to:**
- `src/ncaa_eval/ingest/` (stable)
- `src/ncaa_eval/transform/serving.py` (stable)
- `src/ncaa_eval/transform/normalization.py` (stable)
- `pyproject.toml` (no new dependencies)
- Any existing test files

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 4.4 — Acceptance Criteria]
- [Source: _bmad-output/planning-artifacts/epics.md#Epic 4 — Feature Engineering Suite (FR5: Sequential features, per-possession, Four Factors)]
- [Source: specs/research/feature-engineering-techniques.md#Section 3 — Sequential / Momentum Features]
- [Source: specs/research/feature-engineering-techniques.md#Section 6.7 — Per-Possession Normalization (Edwards 2021 pipeline)]
- [Source: specs/research/feature-engineering-techniques.md#Section 7.3 — Building Blocks by Story (Story 4.4 scope)]
- [Source: specs/05-architecture-fullstack.md#Section 9 — Unified Project Structure (`transform/` module)]
- [Source: specs/05-architecture-fullstack.md#Section 12 — Coding Standards (mypy --strict, vectorization)]
- [Source: data/kaggle/MRegularSeasonDetailedResults.csv — columns: Season, DayNum, WTeamID, WScore, LTeamID, LScore, WLoc, NumOT, WFGM..LPF]
- [Source: data/kaggle/MNCAATourneyDetailedResults.csv — same column structure as regular season]
- [Source: src/ncaa_eval/transform/serving.py — rescale_overtime formula, module structure]
- [Source: src/ncaa_eval/transform/normalization.py — CSV loader pattern, vectorized dict construction, frozen dataclass pattern]
- [Source: _bmad-output/implementation-artifacts/4-3-implement-canonical-team-id-mapping-data-cleaning.md#Dev Notes — itertuples acceptable, every public method needs a test, empty-guard pattern]
- [Source: _bmad-output/implementation-artifacts/4-2-implement-chronological-data-serving-api.md#Dev Agent Record — frozen dataclass mutable-field gotcha, Iterator from collections.abc]
- [Source: _bmad-output/planning-artifacts/template-requirements.md — ML .fit() empty guard, empty-dict ZeroDivisionError guard, all-methods-tested mandate]

## Dev Agent Record

### Agent Model Used

Claude Sonnet 4.6 (create-story workflow)

### Debug Log References

### Completion Notes List

### File List
