"""Shared test helpers for tests/unit/.

Plain helper functions (not pytest fixtures) that can be imported
by any test module within this directory scope.

Note: Because tests/unit/ is a Python package (has __init__.py), these helpers
cannot be auto-injected by pytest as fixtures — they require an explicit import:
    from tests.unit.conftest import _make_season_df, _make_feature_server
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


def _make_season_df(
    year: int,
    n_regular: int = 10,
    n_tournament: int = 3,
    *,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Create a minimal synthetic season DataFrame for testing.

    Returns a DataFrame with 11 columns:
        game_id, season, day_num, date, team_a_id, team_b_id,
        is_tournament, loc_encoding, team_a_won,
        elo_diff, win_pct_diff (synthetic non-metadata feature columns).

    The extra ``elo_diff`` / ``win_pct_diff`` columns ensure ``_feature_cols()``
    returns a non-empty list, exercising the stateless column-filtering code path.
    team_a_id draws from [1000, 2000) and team_b_id from [2000, 3000) to
    guarantee distinct team identities per game.
    """
    if rng is None:
        rng = np.random.default_rng(seed=year)

    total = n_regular + n_tournament
    is_tournament = [False] * n_regular + [True] * n_tournament
    # Include synthetic feature columns (not in METADATA_COLS) so _feature_cols()
    # returns a non-empty list, exercising the stateless column-filtering code path.
    return pd.DataFrame(
        {
            "game_id": [f"{year}_{i}" for i in range(total)],
            "season": year,
            "day_num": list(range(total)),
            "date": pd.date_range(f"{year}-01-01", periods=total, freq="D"),
            "team_a_id": rng.integers(1000, 2000, size=total),
            # Use a separate range [2000, 3000) to guarantee team_b != team_a.
            "team_b_id": rng.integers(2000, 3000, size=total),
            "is_tournament": is_tournament,
            "loc_encoding": rng.choice([1, -1, 0], size=total),
            "team_a_won": rng.choice([True, False], size=total),
            # Synthetic features — used to verify stateless models receive only
            # non-metadata columns and _DataDependentModel has real values to use.
            "elo_diff": rng.normal(0.0, 50.0, size=total),
            "win_pct_diff": rng.uniform(-0.5, 0.5, size=total),
        }
    )


def _make_feature_server(
    season_dfs: dict[int, pd.DataFrame],
) -> MagicMock:
    """Build a mock StatefulFeatureServer for testing.

    Mocks ``StatefulFeatureServer.serve_season_features(year, mode='batch')``
    to return pre-built DataFrames from ``season_dfs``. Returns an empty
    DataFrame for any year not present in the dict.
    """
    mock = MagicMock()
    mock.serve_season_features.side_effect = lambda year, mode="batch": season_dfs.get(year, pd.DataFrame())
    return mock
