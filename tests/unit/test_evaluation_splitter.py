"""Unit tests for the walk-forward cross-validation splitter."""

from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import MagicMock

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from ncaa_eval.evaluation.splitter import CVFold, walk_forward_splits

# ── Test helpers ─────────────────────────────────────────────────────────────


def _make_season_df(
    year: int,
    n_regular: int = 10,
    n_tournament: int = 3,
    *,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Create a minimal synthetic season DataFrame for testing.

    Produces ``n_regular`` regular-season rows and ``n_tournament`` tournament
    rows.  Uses deterministic game_id generation so callers can assert equality.
    """
    if rng is None:
        rng = np.random.default_rng(seed=year)

    total = n_regular + n_tournament
    is_tournament = [False] * n_regular + [True] * n_tournament
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
        }
    )


def _make_feature_server(
    season_dfs: dict[int, pd.DataFrame],
) -> MagicMock:
    """Build a mock StatefulFeatureServer returning pre-built DataFrames."""
    mock = MagicMock()
    mock.serve_season_features.side_effect = lambda year, mode="batch": season_dfs.get(year, pd.DataFrame())
    return mock


# ── TestCVFold ───────────────────────────────────────────────────────────────


class TestCVFold:
    """Tests for the CVFold frozen dataclass."""

    def test_frozen(self) -> None:
        fold = CVFold(train=pd.DataFrame(), test=pd.DataFrame(), year=2023)
        with pytest.raises(AttributeError):
            fold.year = 2024  # type: ignore[misc]

    def test_attributes(self) -> None:
        train = pd.DataFrame({"a": [1]})
        test = pd.DataFrame({"b": [2]})
        fold = CVFold(train=train, test=test, year=2021)
        assert fold.year == 2021
        pd.testing.assert_frame_equal(fold.train, train)
        pd.testing.assert_frame_equal(fold.test, test)


# ── TestWalkForwardSplits ────────────────────────────────────────────────────


class TestWalkForwardSplits:
    """Tests for walk_forward_splits fold generation."""

    def test_basic_fold_count(self) -> None:
        """3.1: Correct number of folds for a season range."""
        seasons = [2010, 2011, 2012, 2013]
        dfs = {y: _make_season_df(y) for y in seasons}
        server = _make_feature_server(dfs)

        folds = list(walk_forward_splits(seasons, server))
        # First season is training-only → 3 folds (2011, 2012, 2013)
        assert len(folds) == 3

    def test_temporal_integrity(self) -> None:
        """3.2: All training data seasons < test year."""
        seasons = [2010, 2011, 2012, 2013]
        dfs = {y: _make_season_df(y) for y in seasons}
        server = _make_feature_server(dfs)

        for fold in walk_forward_splits(seasons, server):
            train_seasons = fold.train["season"].unique()
            for s in train_seasons:
                assert s < fold.year, f"Training season {s} is not strictly before test year {fold.year}"

    def test_2020_exclusion(self) -> None:
        """3.3: No fold has year=2020."""
        seasons = [2018, 2019, 2020, 2021]
        dfs = {y: _make_season_df(y) for y in seasons}
        # 2020 has no tournament games
        dfs[2020] = _make_season_df(2020, n_tournament=0)
        server = _make_feature_server(dfs)

        fold_years = [f.year for f in walk_forward_splits(seasons, server)]
        assert 2020 not in fold_years

    def test_2020_training_inclusion(self) -> None:
        """3.4: Folds for years > 2020 include 2020 regular-season data in training."""
        seasons = [2018, 2019, 2020, 2021]
        dfs = {y: _make_season_df(y) for y in seasons}
        dfs[2020] = _make_season_df(2020, n_tournament=0)
        server = _make_feature_server(dfs)

        folds = list(walk_forward_splits(seasons, server))
        fold_2021 = next(f for f in folds if f.year == 2021)
        train_seasons = set(fold_2021.train["season"].unique())
        assert 2020 in train_seasons, "2020 regular-season data should be in 2021 training"

    def test_fold_determinism(self) -> None:
        """3.5: Two identical calls produce identical results."""
        seasons = [2010, 2011, 2012]
        dfs = {y: _make_season_df(y) for y in seasons}

        server1 = _make_feature_server(dfs)
        server2 = _make_feature_server(dfs)

        folds1 = list(walk_forward_splits(seasons, server1))
        folds2 = list(walk_forward_splits(seasons, server2))

        assert len(folds1) == len(folds2)
        for f1, f2 in zip(folds1, folds2):
            assert f1.year == f2.year
            pd.testing.assert_frame_equal(f1.train, f2.train)
            pd.testing.assert_frame_equal(f1.test, f2.test)

    def test_single_season_raises(self) -> None:
        """3.6: Single-season range raises ValueError."""
        server = _make_feature_server({})
        with pytest.raises(ValueError, match="at least 2 seasons"):
            list(walk_forward_splits([2023], server))

    def test_empty_seasons_raises(self) -> None:
        """3.6 variant: Empty seasons raises ValueError."""
        server = _make_feature_server({})
        with pytest.raises(ValueError, match="at least 2 seasons"):
            list(walk_forward_splits([], server))

    def test_empty_season_still_yields(self) -> None:
        """3.7: Fold is still generated if repository returns data."""
        seasons = [2010, 2011, 2012]
        dfs = {
            2010: _make_season_df(2010),
            2011: pd.DataFrame(),  # empty season
            2012: _make_season_df(2012),
        }
        server = _make_feature_server(dfs)

        folds = list(walk_forward_splits(seasons, server))
        # 2011 is empty but has no tournament games → skip
        # 2012 should still yield a fold
        fold_years = [f.year for f in folds]
        assert 2012 in fold_years

    def test_test_data_contains_only_tournament_games(self) -> None:
        """3.8: Test data contains only tournament games."""
        seasons = [2010, 2011, 2012]
        dfs = {y: _make_season_df(y) for y in seasons}
        server = _make_feature_server(dfs)

        for fold in walk_forward_splits(seasons, server):
            if not fold.test.empty:
                assert fold.test[
                    "is_tournament"
                ].all(), f"Test data for year {fold.year} contains non-tournament games"

    def test_training_data_contains_all_games(self) -> None:
        """3.9: Training data contains all games (regular season + tournament) from prior years."""
        seasons = [2010, 2011, 2012]
        dfs = {y: _make_season_df(y, n_regular=10, n_tournament=3) for y in seasons}
        server = _make_feature_server(dfs)

        folds = list(walk_forward_splits(seasons, server))
        # For fold year=2012, training should include ALL games from 2010 and 2011
        fold_2012 = next(f for f in folds if f.year == 2012)
        train_2010 = fold_2012.train[fold_2012.train["season"] == 2010]
        train_2011 = fold_2012.train[fold_2012.train["season"] == 2011]
        # Each season has 10 regular + 3 tournament = 13 games
        assert len(train_2010) == 13
        assert len(train_2011) == 13

    def test_mode_passed_to_feature_server(self) -> None:
        """3.10: mode parameter is passed through to feature server."""
        seasons = [2010, 2011]
        dfs = {y: _make_season_df(y) for y in seasons}
        server = _make_feature_server(dfs)

        list(walk_forward_splits(seasons, server, mode="stateful"))

        # Verify serve_season_features was called with mode="stateful"
        for call_args in server.serve_season_features.call_args_list:
            assert call_args == ((call_args[0][0],), {"mode": "stateful"}) or (
                call_args.kwargs.get("mode") == "stateful"
            )


# ── TestEdgeCases ────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge case tests for the splitter."""

    def test_unsorted_seasons_produce_same_folds(self) -> None:
        """Seasons not sorted → splitter sorts internally, deterministic."""
        seasons_sorted: Sequence[int] = [2010, 2011, 2012, 2013]
        seasons_unsorted: Sequence[int] = [2013, 2010, 2012, 2011]
        dfs = {y: _make_season_df(y) for y in seasons_sorted}

        server1 = _make_feature_server(dfs)
        server2 = _make_feature_server(dfs)

        folds1 = list(walk_forward_splits(seasons_sorted, server1))
        folds2 = list(walk_forward_splits(seasons_unsorted, server2))

        assert len(folds1) == len(folds2)
        for f1, f2 in zip(folds1, folds2):
            assert f1.year == f2.year
            pd.testing.assert_frame_equal(f1.train, f2.train)
            pd.testing.assert_frame_equal(f1.test, f2.test)

    def test_two_seasons_yields_one_fold(self) -> None:
        """Two seasons total → exactly 1 fold."""
        seasons = [2010, 2011]
        dfs = {y: _make_season_df(y) for y in seasons}
        server = _make_feature_server(dfs)

        folds = list(walk_forward_splits(seasons, server))
        assert len(folds) == 1
        assert folds[0].year == 2011

    def test_non_2020_season_no_tournament_yields_empty_test(self) -> None:
        """Test year with 0 tournament games (non-2020) → fold with empty test DataFrame."""
        seasons = [2010, 2011]
        dfs = {
            2010: _make_season_df(2010),
            2011: _make_season_df(2011, n_tournament=0),
        }
        server = _make_feature_server(dfs)

        folds = list(walk_forward_splits(seasons, server))
        # 2011 is NOT in _NO_TOURNAMENT_SEASONS, but has 0 tournament games
        # → yield fold with empty test
        assert len(folds) == 1
        assert folds[0].year == 2011
        assert folds[0].test.empty

    def test_feature_caching(self) -> None:
        """Feature server is called once per season, not once per fold."""
        seasons = [2010, 2011, 2012, 2013]
        dfs = {y: _make_season_df(y) for y in seasons}
        server = _make_feature_server(dfs)

        list(walk_forward_splits(seasons, server))

        # Each season should be served exactly once
        assert server.serve_season_features.call_count == len(seasons)

    def test_train_index_is_reset(self) -> None:
        """Training DataFrame index is reset after concat (no duplicate indices)."""
        seasons = [2010, 2011, 2012]
        dfs = {y: _make_season_df(y) for y in seasons}
        server = _make_feature_server(dfs)

        for fold in walk_forward_splits(seasons, server):
            assert fold.train.index.is_unique, f"Fold year={fold.year} has duplicate indices in training data"

    def test_fold_years_are_ascending(self) -> None:
        """Folds are yielded in ascending year order."""
        seasons = [2010, 2011, 2012, 2013, 2014]
        dfs = {y: _make_season_df(y) for y in seasons}
        server = _make_feature_server(dfs)

        fold_years = [f.year for f in walk_forward_splits(seasons, server)]
        assert fold_years == sorted(fold_years)
