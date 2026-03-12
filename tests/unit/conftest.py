"""Shared test helpers for tests/unit/.

Plain helper functions (not pytest fixtures) that can be imported
by any test module within this directory scope.
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
    """Create a minimal synthetic season DataFrame for testing."""
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
            "team_b_id": rng.integers(1000, 2000, size=total),
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
    """Build a mock StatefulFeatureServer returning pre-built DataFrames."""
    mock = MagicMock()
    mock.serve_season_features.side_effect = lambda year, mode="batch": season_dfs.get(year, pd.DataFrame())
    return mock
