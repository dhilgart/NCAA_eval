"""Unit tests for sequential feature transformations (Story 4.4).

Tests cover: DetailedResultsLoader, apply_ot_rescaling, compute_game_weights,
compute_rolling_stats, compute_ewma_stats, compute_momentum, compute_streak,
compute_possessions, compute_per_possession_stats, compute_four_factors,
and SequentialTransformer.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest

from ncaa_eval.transform.sequential import (
    DetailedResultsLoader,
    SequentialTransformer,
    _reshape_to_long,
    apply_ot_rescaling,
    compute_ewma_stats,
    compute_four_factors,
    compute_game_weights,
    compute_momentum,
    compute_per_possession_stats,
    compute_possessions,
    compute_rolling_stats,
    compute_streak,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def regular_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "Season": [2010, 2010],
            "DayNum": [30, 40],
            "WTeamID": [1001, 1002],
            "WScore": [75, 80],
            "LTeamID": [1002, 1001],
            "LScore": [65, 70],
            "WLoc": ["H", "N"],
            "NumOT": [0, 1],
            "WFGM": [28, 30],
            "WFGA": [60, 62],
            "WFGM3": [5, 6],
            "WFGA3": [15, 14],
            "WFTM": [14, 10],
            "WFTA": [18, 12],
            "WOR": [12, 10],
            "WDR": [22, 24],
            "WAst": [15, 18],
            "WTO": [14, 12],
            "WStl": [7, 8],
            "WBlk": [3, 4],
            "WPF": [20, 18],
            "LFGM": [24, 26],
            "LFGA": [58, 60],
            "LFGM3": [4, 3],
            "LFGA3": [12, 13],
            "LFTM": [13, 15],
            "LFTA": [17, 19],
            "LOR": [10, 11],
            "LDR": [20, 21],
            "LAst": [12, 14],
            "LTO": [16, 15],
            "LStl": [6, 5],
            "LBlk": [2, 3],
            "LPF": [22, 20],
        }
    )
    path = tmp_path / "regular.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def tourney_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "Season": [2010],
            "DayNum": [134],
            "WTeamID": [1001],
            "WScore": [85],
            "LTeamID": [1003],
            "LScore": [72],
            "WLoc": ["N"],
            "NumOT": [0],
            "WFGM": [32],
            "WFGA": [65],
            "WFGM3": [7],
            "WFGA3": [18],
            "WFTM": [14],
            "WFTA": [18],
            "WOR": [14],
            "WDR": [25],
            "WAst": [20],
            "WTO": [10],
            "WStl": [9],
            "WBlk": [5],
            "WPF": [16],
            "LFGM": [26],
            "LFGA": [62],
            "LFGM3": [5],
            "LFGA3": [14],
            "LFTM": [15],
            "LFTA": [20],
            "LOR": [11],
            "LDR": [22],
            "LAst": [15],
            "LTO": [18],
            "LStl": [7],
            "LBlk": [3],
            "LPF": [21],
        }
    )
    path = tmp_path / "tourney.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def ten_game_team() -> pd.DataFrame:
    """10-game team history fixture with all box-score columns."""
    n = 10
    return pd.DataFrame(
        {
            "season": [2020] * n,
            "day_num": list(range(10, 10 + n * 5, 5)),
            "team_id": [1001] * n,
            "opp_id": [1002] * n,
            "won": [True, False, True, True, False, True, True, True, False, True],
            "loc_encoded": [1, -1, 0, 1, -1, 0, 1, -1, 0, 1],
            "num_ot": [0] * n,
            "is_tournament": [False] * n,
            "score": [70.0, 65.0, 75.0, 80.0, 60.0, 72.0, 78.0, 82.0, 58.0, 76.0],
            "opp_score": [65.0, 70.0, 68.0, 72.0, 75.0, 65.0, 70.0, 75.0, 72.0, 68.0],
            "fgm": [25.0, 22.0, 28.0, 30.0, 20.0, 26.0, 29.0, 31.0, 19.0, 27.0],
            "fga": [55.0, 52.0, 58.0, 60.0, 50.0, 56.0, 59.0, 61.0, 48.0, 57.0],
            "fgm3": [5.0, 4.0, 6.0, 7.0, 3.0, 5.0, 6.0, 7.0, 3.0, 5.0],
            "fga3": [15.0, 13.0, 16.0, 17.0, 12.0, 14.0, 15.0, 17.0, 11.0, 14.0],
            "ftm": [15.0, 17.0, 13.0, 16.0, 14.0, 15.0, 14.0, 17.0, 13.0, 17.0],
            "fta": [18.0, 20.0, 16.0, 19.0, 17.0, 18.0, 17.0, 20.0, 16.0, 20.0],
            "oreb": [10.0, 8.0, 12.0, 11.0, 7.0, 10.0, 11.0, 12.0, 6.0, 10.0],
            "dreb": [22.0, 20.0, 24.0, 23.0, 19.0, 22.0, 23.0, 24.0, 18.0, 22.0],
            "ast": [15.0, 13.0, 17.0, 18.0, 11.0, 15.0, 16.0, 18.0, 10.0, 16.0],
            "to": [12.0, 14.0, 11.0, 10.0, 15.0, 12.0, 11.0, 10.0, 16.0, 12.0],
            "stl": [6.0, 5.0, 7.0, 8.0, 4.0, 6.0, 7.0, 8.0, 3.0, 6.0],
            "blk": [3.0, 2.0, 4.0, 3.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0],
            "pf": [18.0, 20.0, 17.0, 16.0, 21.0, 18.0, 17.0, 16.0, 22.0, 18.0],
            "opp_oreb": [8.0, 10.0, 9.0, 7.0, 11.0, 8.0, 9.0, 7.0, 12.0, 8.0],
            "opp_dreb": [20.0, 22.0, 21.0, 19.0, 23.0, 20.0, 21.0, 19.0, 24.0, 20.0],
        }
    )


# ---------------------------------------------------------------------------
# Tests: _reshape_to_long
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.smoke
def test_reshape_to_long_winner_row(regular_csv: Path) -> None:
    """Winner row has loc_encoded=+1 (H game), won=True, correct fgm."""
    df = pd.read_csv(regular_csv)
    long = _reshape_to_long(df.iloc[[0]], is_tournament=False)

    winner = long[long["won"] == True]  # noqa: E712
    assert len(winner) == 1
    assert int(winner["loc_encoded"].iloc[0]) == 1  # WLoc="H"
    assert bool(winner["won"].iloc[0]) is True
    assert int(winner["fgm"].iloc[0]) == 28  # WFGM from first row


@pytest.mark.unit
@pytest.mark.smoke
def test_reshape_to_long_loser_row(regular_csv: Path) -> None:
    """Loser row has loc_encoded=-1 (H game → loser was away), won=False."""
    df = pd.read_csv(regular_csv)
    long = _reshape_to_long(df.iloc[[0]], is_tournament=False)

    loser = long[long["won"] == False]  # noqa: E712
    assert len(loser) == 1
    assert int(loser["loc_encoded"].iloc[0]) == -1  # inverted H → -1
    assert bool(loser["won"].iloc[0]) is False
    assert int(loser["fgm"].iloc[0]) == 24  # LFGM from first row


@pytest.mark.unit
@pytest.mark.smoke
def test_reshape_to_long_neutral(regular_csv: Path) -> None:
    """WLoc=N: both winner and loser get loc_encoded=0."""
    df = pd.read_csv(regular_csv)
    # Second row has WLoc="N"
    long = _reshape_to_long(df.iloc[[1]], is_tournament=False)

    assert (long["loc_encoded"] == 0).all()


# ---------------------------------------------------------------------------
# Tests: DetailedResultsLoader
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.smoke
def test_detailed_results_loader_from_csvs(regular_csv: Path, tourney_csv: Path) -> None:
    """2 regular + 1 tourney game → 6 long-format rows; is_tournament flags correct."""
    loader = DetailedResultsLoader.from_csvs(regular_csv, tourney_csv)
    df = loader._df

    assert len(df) == 6  # (2 + 1) games × 2 teams per game
    assert df[df["is_tournament"] == False].shape[0] == 4  # noqa: E712
    assert df[df["is_tournament"] == True].shape[0] == 2  # noqa: E712


@pytest.mark.unit
@pytest.mark.smoke
def test_get_team_season_sorted(regular_csv: Path, tourney_csv: Path) -> None:
    """get_team_season returns rows sorted by day_num ascending."""
    loader = DetailedResultsLoader.from_csvs(regular_csv, tourney_csv)
    games = loader.get_team_season(team_id=1001, season=2010)

    assert list(games["day_num"]) == sorted(games["day_num"].tolist())


@pytest.mark.unit
@pytest.mark.smoke
def test_get_team_season_empty(regular_csv: Path, tourney_csv: Path) -> None:
    """Unknown team_id returns empty DataFrame (no exception)."""
    loader = DetailedResultsLoader.from_csvs(regular_csv, tourney_csv)
    result = loader.get_team_season(team_id=9999, season=2010)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


@pytest.mark.unit
@pytest.mark.smoke
def test_get_season_long_format(regular_csv: Path, tourney_csv: Path) -> None:
    """get_season_long_format returns all rows for a season sorted by (day_num, team_id)."""
    loader = DetailedResultsLoader.from_csvs(regular_csv, tourney_csv)
    df = loader.get_season_long_format(season=2010)

    # 2 regular + 1 tourney game × 2 teams per game = 6 rows
    assert len(df) == 6
    # Sorted by day_num ascending
    assert list(df["day_num"]) == sorted(df["day_num"].tolist())
    # Within same day_num, sorted by team_id ascending
    for day in df["day_num"].unique():
        group = df[df["day_num"] == day]["team_id"].tolist()
        assert group == sorted(group)
    # reset_index(drop=True) → index starts at 0
    assert df.index[0] == 0


# ---------------------------------------------------------------------------
# Tests: apply_ot_rescaling
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.smoke
def test_apply_ot_rescaling_regulation(ten_game_team: pd.DataFrame) -> None:
    """num_ot=0: all counting stats unchanged (40/40 = 1.0 multiplier)."""
    assert (ten_game_team["num_ot"] == 0).all()
    result = apply_ot_rescaling(ten_game_team)

    pd.testing.assert_series_equal(result["score"], ten_game_team["score"])
    pd.testing.assert_series_equal(result["fgm"], ten_game_team["fgm"])


@pytest.mark.unit
@pytest.mark.smoke
def test_apply_ot_rescaling_one_ot() -> None:
    """num_ot=1: score rescaled to score × 40/45."""
    df = pd.DataFrame(
        {
            "num_ot": [1],
            "score": [90.0],
            "opp_score": [85.0],
            "fgm": [30.0],
            "fga": [60.0],
            "fgm3": [5.0],
            "fga3": [15.0],
            "ftm": [25.0],
            "fta": [30.0],
            "oreb": [10.0],
            "dreb": [20.0],
            "ast": [15.0],
            "to": [12.0],
            "stl": [6.0],
            "blk": [3.0],
            "pf": [18.0],
        }
    )
    result = apply_ot_rescaling(df)
    expected_score = 90.0 * 40.0 / 45.0
    assert abs(float(result["score"].iloc[0]) - expected_score) < 1e-9


# ---------------------------------------------------------------------------
# Tests: compute_game_weights
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.smoke
def test_compute_game_weights_no_decay() -> None:
    """Game 0 days ago: weight=1.0; 40 days ago: weight=1.0; 41 days ago: 0.99."""
    ref = 100
    day_nums = pd.Series([100, 60, 59])
    weights = compute_game_weights(day_nums, reference_day_num=ref)

    assert abs(float(weights.iloc[0]) - 1.0) < 1e-9  # 0 days ago
    assert abs(float(weights.iloc[1]) - 1.0) < 1e-9  # 40 days ago
    assert abs(float(weights.iloc[2]) - 0.99) < 1e-9  # 41 days ago


@pytest.mark.unit
@pytest.mark.smoke
def test_compute_game_weights_floor() -> None:
    """Game 140 days ago: weight = 0.6 (floor enforced)."""
    ref = 200
    day_nums = pd.Series([60])  # 140 days ago
    weights = compute_game_weights(day_nums, reference_day_num=ref)

    assert abs(float(weights.iloc[0]) - 0.6) < 1e-9


@pytest.mark.unit
@pytest.mark.smoke
def test_compute_game_weights_empty() -> None:
    """Empty day_nums returns empty weights Series without error."""
    weights = compute_game_weights(pd.Series([], dtype=int))
    assert isinstance(weights, pd.Series)
    assert len(weights) == 0


# ---------------------------------------------------------------------------
# Tests: compute_rolling_stats
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.smoke
def test_rolling_stats_window_5(ten_game_team: pd.DataFrame) -> None:
    """rolling_5_score[0] = score[0]; rolling_5_score[4] = mean(score[0:5])."""
    result = compute_rolling_stats(ten_game_team, windows=[5], stats=("score",))
    scores = ten_game_team["score"].tolist()

    assert abs(float(result["rolling_5_score"].iloc[0]) - scores[0]) < 1e-9
    expected_pos4 = sum(scores[:5]) / 5.0
    assert abs(float(result["rolling_5_score"].iloc[4]) - expected_pos4) < 1e-9


@pytest.mark.unit
@pytest.mark.smoke
def test_rolling_stats_min_periods() -> None:
    """Single-game team history: rolling_5_score is not NaN (min_periods=1)."""
    df = pd.DataFrame(
        {
            "score": [75.0],
            "day_num": [30],
        }
    )
    result = compute_rolling_stats(df, windows=[5], stats=("score",))
    assert not result["rolling_5_score"].isna().any()
    assert abs(float(result["rolling_5_score"].iloc[0]) - 75.0) < 1e-9


@pytest.mark.unit
@pytest.mark.smoke
def test_rolling_full_is_expanding_mean(ten_game_team: pd.DataFrame) -> None:
    """rolling_full_score at each row equals score.expanding().mean()."""
    result = compute_rolling_stats(ten_game_team, windows=[5], stats=("score",))
    expected = ten_game_team["score"].expanding().mean()
    pd.testing.assert_series_equal(
        result["rolling_full_score"],
        expected,
        check_names=False,
    )


@pytest.mark.unit
@pytest.mark.smoke
def test_rolling_no_future_leakage(ten_game_team: pd.DataFrame) -> None:
    """For position i, rolling_5_score only uses games at positions <= i."""
    result = compute_rolling_stats(ten_game_team, windows=[5], stats=("score",))
    scores = ten_game_team["score"].tolist()

    for i in range(len(scores)):
        window_data = scores[max(0, i - 4) : i + 1]
        expected = sum(window_data) / len(window_data)
        assert abs(float(result["rolling_5_score"].iloc[i]) - expected) < 1e-9


# ---------------------------------------------------------------------------
# Tests: compute_ewma_stats
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.smoke
def test_ewma_stats_alpha() -> None:
    """ewma_0p20_score at row 1 equals alpha*score[1] + (1-alpha)*score[0]."""
    alpha = 0.20
    scores = [70.0, 80.0, 75.0, 85.0, 72.0]
    df = pd.DataFrame({"score": scores})
    result = compute_ewma_stats(df, alphas=[alpha], stats=("score",))

    # adjust=False: value_t = alpha * obs_t + (1-alpha) * value_{t-1}
    expected_row1 = alpha * scores[1] + (1 - alpha) * scores[0]
    assert abs(float(result["ewma_0p20_score"].iloc[1]) - expected_row1) < 1e-9


# ---------------------------------------------------------------------------
# Tests: compute_momentum
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.smoke
def test_compute_momentum_positive_improving() -> None:
    """Increasing score series: momentum_score is positive (fast > slow)."""
    scores = [50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0]
    df = pd.DataFrame({"score": scores})
    result = compute_momentum(df, alpha_fast=0.20, alpha_slow=0.10, stats=("score",))

    # After several games of increasing scores, fast EWMA > slow EWMA
    assert float(result["momentum_score"].iloc[-1]) > 0


# ---------------------------------------------------------------------------
# Tests: compute_streak
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.smoke
def test_compute_streak_win_streak() -> None:
    """3 consecutive wins: streak = +3 at position 2."""
    won = pd.Series([True, True, True])
    streak = compute_streak(won)
    assert int(streak.iloc[2]) == 3


@pytest.mark.unit
@pytest.mark.smoke
def test_compute_streak_loss_streak() -> None:
    """3 consecutive losses: streak = -3 at position 2."""
    won = pd.Series([False, False, False])
    streak = compute_streak(won)
    assert int(streak.iloc[2]) == -3


@pytest.mark.unit
@pytest.mark.smoke
def test_compute_streak_reset() -> None:
    """WWLWW: streaks are +1, +2, -1, +1, +2."""
    won = pd.Series([True, True, False, True, True])
    streak = compute_streak(won)
    assert list(streak) == [1, 2, -1, 1, 2]


# ---------------------------------------------------------------------------
# Tests: compute_possessions
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.smoke
def test_compute_possessions_formula() -> None:
    """Known FGA, OR, TO, FTA: verify possessions = FGA - OR + TO + 0.44*FTA."""
    df = pd.DataFrame(
        {
            "fga": [60.0],
            "oreb": [10.0],
            "to": [14.0],
            "fta": [18.0],
        }
    )
    result = compute_possessions(df)
    expected = 60.0 - 10.0 + 14.0 + 0.44 * 18.0
    assert abs(float(result.iloc[0]) - expected) < 1e-9


@pytest.mark.unit
@pytest.mark.smoke
def test_compute_possessions_zero_guard() -> None:
    """If possessions = 0 for a row, result is NaN (no ZeroDivisionError downstream)."""
    # Craft a row where possessions = 0: FGA=OR, TO=0, FTA=0
    df = pd.DataFrame(
        {
            "fga": [5.0],
            "oreb": [5.0],
            "to": [0.0],
            "fta": [0.0],
        }
    )
    result = compute_possessions(df)
    assert pd.isna(result.iloc[0])


# ---------------------------------------------------------------------------
# Tests: compute_per_possession_stats
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.smoke
def test_compute_per_possession_stats() -> None:
    """Known score and possessions: score_per100 = score * 100 / possessions."""
    df = pd.DataFrame({"score": [75.0]})
    possessions = pd.Series([60.0])
    result = compute_per_possession_stats(df, stats=("score",), possessions=possessions)
    expected = 75.0 * 100.0 / 60.0
    assert abs(float(result["score_per100"].iloc[0]) - expected) < 1e-9


# ---------------------------------------------------------------------------
# Tests: compute_four_factors
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.smoke
def test_compute_four_factors_efg() -> None:
    """Known FGM, FGM3, FGA: efg_pct = (FGM + 0.5*FGM3) / FGA."""
    df = pd.DataFrame(
        {
            "fgm": [28.0],
            "fgm3": [5.0],
            "fga": [60.0],
            "oreb": [10.0],
            "opp_dreb": [20.0],
            "fta": [18.0],
            "to": [14.0],
        }
    )
    possessions = pd.Series([50.0])
    result = compute_four_factors(df, possessions)
    expected_efg = (28.0 + 0.5 * 5.0) / 60.0
    assert abs(float(result["efg_pct"].iloc[0]) - expected_efg) < 1e-9


@pytest.mark.unit
@pytest.mark.smoke
def test_compute_four_factors_orb() -> None:
    """Known oreb, opp_dreb: orb_pct = oreb / (oreb + opp_dreb)."""
    df = pd.DataFrame(
        {
            "fgm": [28.0],
            "fgm3": [5.0],
            "fga": [60.0],
            "oreb": [10.0],
            "opp_dreb": [20.0],
            "fta": [18.0],
            "to": [14.0],
        }
    )
    possessions = pd.Series([50.0])
    result = compute_four_factors(df, possessions)
    expected_orb = 10.0 / (10.0 + 20.0)
    assert abs(float(result["orb_pct"].iloc[0]) - expected_orb) < 1e-9


@pytest.mark.unit
@pytest.mark.smoke
def test_compute_four_factors_zero_fga() -> None:
    """FGA = 0: efg_pct = NaN (no exception)."""
    df = pd.DataFrame(
        {
            "fgm": [0.0],
            "fgm3": [0.0],
            "fga": [0.0],
            "oreb": [5.0],
            "opp_dreb": [10.0],
            "fta": [5.0],
            "to": [3.0],
        }
    )
    possessions = pd.Series([8.0])
    result = compute_four_factors(df, possessions)
    assert pd.isna(result["efg_pct"].iloc[0])
    assert pd.isna(result["ftr"].iloc[0])


# ---------------------------------------------------------------------------
# Tests: SequentialTransformer
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.smoke
def test_sequential_transformer_transform_columns(
    ten_game_team: pd.DataFrame,
) -> None:
    """All expected column groups exist after transform."""
    transformer = SequentialTransformer()
    result = transformer.transform(ten_game_team)

    # Streak
    assert "streak" in result.columns

    # Rolling windows
    for w in [5, 10, 20]:
        assert f"rolling_{w}_score" in result.columns
    assert "rolling_full_score" in result.columns

    # EWMA
    assert "ewma_0p15_score" in result.columns
    assert "ewma_0p20_score" in result.columns

    # Momentum
    assert "momentum_score" in result.columns

    # Possessions
    assert "possessions" in result.columns

    # Per-100
    assert "score_per100" in result.columns

    # Four Factors
    for col in ["efg_pct", "orb_pct", "ftr", "to_pct"]:
        assert col in result.columns


@pytest.mark.unit
@pytest.mark.smoke
def test_sequential_transformer_preserves_row_count(
    ten_game_team: pd.DataFrame,
) -> None:
    """Output has same number of rows as input."""
    transformer = SequentialTransformer()
    result = transformer.transform(ten_game_team)
    assert len(result) == len(ten_game_team)


@pytest.mark.unit
@pytest.mark.smoke
def test_sequential_transformer_empty_input() -> None:
    """transform() on empty DataFrame returns empty DataFrame without error (pre-2003 guard)."""
    empty_df = pd.DataFrame(columns=["season", "day_num", "team_id", "won", "num_ot", "score"])
    transformer = SequentialTransformer()
    result = transformer.transform(empty_df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
