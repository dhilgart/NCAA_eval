"""Unit tests for batch opponent adjustment rating solvers (Story 4.6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from ncaa_eval.transform.opponent import (
    BatchRatingSolver,
    compute_colley_ratings,
    compute_ridge_ratings,
    compute_srs_ratings,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _linear_chain() -> pd.DataFrame:
    """4-team linear chain: 1 beat 2 beat 3 beat 4, each by margin 10."""
    return pd.DataFrame(
        {
            "w_team_id": [1, 2, 3],
            "l_team_id": [2, 3, 4],
            "w_score": [110, 110, 110],
            "l_score": [100, 100, 100],
        }
    )


def _balanced_round_robin() -> pd.DataFrame:
    """6-team balanced round-robin: each team plays every other team once.

    Teams are numbered 1-6. For each pair (i,j) with i<j:
    - i beats j by margin 10 (so team 1 is best, team 6 is worst).
    """
    rows = []
    for i in range(1, 7):
        for j in range(i + 1, 7):
            rows.append(
                {
                    "w_team_id": i,
                    "l_team_id": j,
                    "w_score": 110,
                    "l_score": 100,
                }
            )
    return pd.DataFrame(rows)


def _symmetric_two_game() -> pd.DataFrame:
    """A beat B by 10, B beat A by 10 — perfectly symmetric."""
    return pd.DataFrame(
        {
            "w_team_id": [1, 2],
            "l_team_id": [2, 1],
            "w_score": [110, 110],
            "l_score": [100, 100],
        }
    )


def _blowout_fixture() -> pd.DataFrame:
    """A beat B by 100 (blowout)."""
    return pd.DataFrame(
        {
            "w_team_id": [1],
            "l_team_id": [2],
            "w_score": [200],
            "l_score": [100],
        }
    )


def _capped_fixture() -> pd.DataFrame:
    """A beat B by exactly 25 (cap)."""
    return pd.DataFrame(
        {
            "w_team_id": [1],
            "l_team_id": [2],
            "w_score": [125],
            "l_score": [100],
        }
    )


def _win_heavy_fixture() -> pd.DataFrame:
    """Team 1 wins 8 of 10 games; team 2 wins 2 of 10 games (same opponents)."""
    rows = []
    # 10 games: team 1 is winner in 8, team 2 in 2
    for _ in range(8):
        rows.append({"w_team_id": 1, "l_team_id": 2, "w_score": 110, "l_score": 100})
    for _ in range(2):
        rows.append({"w_team_id": 2, "l_team_id": 1, "w_score": 110, "l_score": 100})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SRS tests
# ---------------------------------------------------------------------------


def test_srs_linear_chain_ordering() -> None:
    """4-team linear chain: ratings must be monotonically decreasing 1>2>3>4."""
    df = _linear_chain()
    result = compute_srs_ratings(df)
    ratings = result.set_index("team_id")["srs_rating"]
    assert ratings[1] > ratings[2] > ratings[3] > ratings[4]


def test_srs_convergence_assertion() -> None:
    """SRS is deterministic: two calls on same fixture yield identical results."""
    df = _balanced_round_robin()
    r1 = compute_srs_ratings(df).set_index("team_id")["srs_rating"]
    r2 = compute_srs_ratings(df).set_index("team_id")["srs_rating"]
    pd.testing.assert_series_equal(r1, r2, atol=1e-4, check_names=False)


def test_srs_symmetric_wins_equal_ratings() -> None:
    """Symmetric schedule (A beat B, B beat A, same margin) → equal ratings."""
    df = _symmetric_two_game()
    result = compute_srs_ratings(df).set_index("team_id")["srs_rating"]
    assert abs(result[1] - result[2]) < 1e-6


def test_srs_margin_cap_applied() -> None:
    """SRS with margin_cap=25 on blowout == SRS on pre-capped fixture."""
    blowout = _blowout_fixture()
    capped = _capped_fixture()

    r_blowout = compute_srs_ratings(blowout, margin_cap=25)
    r_capped = compute_srs_ratings(capped, margin_cap=25)

    # Both fixtures should produce the same spread because the blowout is capped to 25
    spread_blowout = abs(
        r_blowout.set_index("team_id")["srs_rating"][1] - r_blowout.set_index("team_id")["srs_rating"][2]
    )
    spread_capped = abs(
        r_capped.set_index("team_id")["srs_rating"][1] - r_capped.set_index("team_id")["srs_rating"][2]
    )
    assert abs(spread_blowout - spread_capped) < 1e-6


def test_srs_ratings_sum_to_zero() -> None:
    """Zero-centering: sum of SRS ratings must be ~0 for any connected fixture."""
    df = _balanced_round_robin()
    result = compute_srs_ratings(df)
    assert abs(result["srs_rating"].sum()) < 1e-4


def test_srs_empty_input() -> None:
    """Empty DataFrame → empty result with correct schema, no exception."""
    empty = pd.DataFrame(columns=["w_team_id", "l_team_id", "w_score", "l_score"])
    result = compute_srs_ratings(empty)
    assert list(result.columns) == ["team_id", "srs_rating"]
    assert len(result) == 0


# ---------------------------------------------------------------------------
# Ridge tests
# ---------------------------------------------------------------------------


def test_ridge_lambda_high_shrinks_ratings() -> None:
    """Higher λ produces ratings with smaller magnitude (closer to zero)."""
    df = _balanced_round_robin()
    r_low = compute_ridge_ratings(df, lam=1.0)
    r_high = compute_ridge_ratings(df, lam=100.0)
    mean_abs_low = r_low["ridge_rating"].abs().mean()
    mean_abs_high = r_high["ridge_rating"].abs().mean()
    assert mean_abs_high < mean_abs_low


def test_ridge_lambda_sensitivity_ordering_preserved() -> None:
    """λ affects magnitude but not the rank ordering of teams."""
    df = _balanced_round_robin()
    r_low = compute_ridge_ratings(df, lam=1.0).set_index("team_id")["ridge_rating"]
    r_high = compute_ridge_ratings(df, lam=100.0).set_index("team_id")["ridge_rating"]
    # Rankings should be identical (rank 1 → 6, lower value = worse)
    assert list(r_low.rank().values) == list(r_high.rank().values)


def test_ridge_empty_input() -> None:
    """Empty DataFrame → empty result with correct schema, no exception."""
    empty = pd.DataFrame(columns=["w_team_id", "l_team_id", "w_score", "l_score"])
    result = compute_ridge_ratings(empty)
    assert list(result.columns) == ["team_id", "ridge_rating"]
    assert len(result) == 0


# ---------------------------------------------------------------------------
# Colley tests
# ---------------------------------------------------------------------------


def test_colley_ignores_margin() -> None:
    """Colley ratings are identical regardless of point margin (win/loss only)."""
    # Same teams, same win/loss results, but different margins
    df_small = pd.DataFrame(
        {
            "w_team_id": [1, 2, 3],
            "l_team_id": [2, 3, 1],
            "w_score": [105, 105, 105],
            "l_score": [100, 100, 100],
        }
    )
    df_large = pd.DataFrame(
        {
            "w_team_id": [1, 2, 3],
            "l_team_id": [2, 3, 1],
            "w_score": [150, 150, 150],
            "l_score": [100, 100, 100],
        }
    )
    r_small = compute_colley_ratings(df_small).set_index("team_id")["colley_rating"]
    r_large = compute_colley_ratings(df_large).set_index("team_id")["colley_rating"]
    for team_id in [1, 2, 3]:
        assert abs(r_small[team_id] - r_large[team_id]) < 1e-10


def test_colley_win_loss_ratio_reflected() -> None:
    """Team with better win/loss record gets higher Colley rating."""
    df = _win_heavy_fixture()
    result = compute_colley_ratings(df).set_index("team_id")["colley_rating"]
    assert result[1] > result[2]


def test_colley_ratings_bounded() -> None:
    """All Colley ratings must be in [0, 1] by construction."""
    df = _balanced_round_robin()
    result = compute_colley_ratings(df)
    assert (result["colley_rating"] >= 0.0).all()
    assert (result["colley_rating"] <= 1.0).all()


def test_colley_empty_input() -> None:
    """Empty DataFrame → empty result with correct schema, no exception."""
    empty = pd.DataFrame(columns=["w_team_id", "l_team_id"])
    result = compute_colley_ratings(empty)
    assert list(result.columns) == ["team_id", "colley_rating"]
    assert len(result) == 0


# ---------------------------------------------------------------------------
# Schema and cross-method tests
# ---------------------------------------------------------------------------


def test_all_methods_return_correct_schema() -> None:
    """All methods return DataFrame with correct column names, dtypes, and row count."""
    df = _linear_chain()
    unique_teams = 4

    srs = BatchRatingSolver().compute_srs(df)
    assert list(srs.columns) == ["team_id", "srs_rating"]
    assert pd.api.types.is_integer_dtype(srs["team_id"])
    assert srs["srs_rating"].dtype == np.float64
    assert len(srs) == unique_teams

    ridge = BatchRatingSolver().compute_ridge(df)
    assert list(ridge.columns) == ["team_id", "ridge_rating"]
    assert ridge["ridge_rating"].dtype == np.float64
    assert len(ridge) == unique_teams

    colley = BatchRatingSolver().compute_colley(df)
    assert list(colley.columns) == ["team_id", "colley_rating"]
    assert colley["colley_rating"].dtype == np.float64
    assert len(colley) == unique_teams


def test_convenience_functions_match_class_methods() -> None:
    """Convenience functions produce identical results to class methods."""
    df = _balanced_round_robin()
    solver = BatchRatingSolver()

    srs_conv = compute_srs_ratings(df).sort_values("team_id").reset_index(drop=True)
    srs_class = solver.compute_srs(df).sort_values("team_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(srs_conv, srs_class)

    ridge_conv = compute_ridge_ratings(df).sort_values("team_id").reset_index(drop=True)
    ridge_class = solver.compute_ridge(df).sort_values("team_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(ridge_conv, ridge_class)

    colley_conv = compute_colley_ratings(df).sort_values("team_id").reset_index(drop=True)
    colley_class = solver.compute_colley(df).sort_values("team_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(colley_conv, colley_class)


@pytest.mark.no_mutation
def test_no_iterrows() -> None:
    """opponent.py must not use iterrows (vectorization mandate)."""
    # Navigate from test file location to src/ncaa_eval/transform/opponent.py
    test_dir = Path(__file__).parent  # tests/unit/
    repo_root = test_dir.parent.parent  # repo root
    source = repo_root / "src" / "ncaa_eval" / "transform" / "opponent.py"
    assert source.exists(), f"opponent.py not found at {source}"
    content = source.read_text()
    assert "iterrows" not in content, "opponent.py must not use iterrows()"
