"""Walk-forward cross-validation splitter with Leave-One-Tournament-Out folds.

Provides :func:`walk_forward_splits`, which partitions historical game data into
train/test folds where each fold uses one tournament year as the test set and all
prior years as training data.  The 2020 COVID year is handled gracefully: its
regular-season data is included in training but no test fold is yielded (the
tournament was cancelled).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Sequence

import pandas as pd  # type: ignore[import-untyped]

from ncaa_eval.transform.feature_serving import StatefulFeatureServer
from ncaa_eval.transform.serving import _NO_TOURNAMENT_SEASONS


@dataclasses.dataclass(frozen=True)
class CVFold:
    """A single cross-validation fold.

    Attributes
    ----------
    train
        All games from seasons strictly before the test year.
    test
        Tournament games only from the test year.
    year
        The test season year.
    """

    train: pd.DataFrame
    test: pd.DataFrame
    year: int


def walk_forward_splits(
    seasons: Sequence[int],
    feature_server: StatefulFeatureServer,
    *,
    mode: str = "batch",
) -> Iterator[CVFold]:
    """Generate walk-forward CV folds with Leave-One-Tournament-Out splits.

    Parameters
    ----------
    seasons
        Ordered sequence of season years to include (e.g., range(2008, 2026)).
        Must contain at least 2 seasons.
    feature_server
        Configured StatefulFeatureServer for building feature matrices.
    mode
        Feature serving mode: "batch" (stateless models) or "stateful"
        (sequential-update models like Elo).

    Yields
    ------
    CVFold
        For each eligible test year (skipping no-tournament years like 2020):
        - train: All games from seasons strictly before the test year
        - test: Tournament games only from the test year
        - year: The test season year

    Raises
    ------
    ValueError
        If ``seasons`` has fewer than 2 elements.
    """
    sorted_seasons = sorted(seasons)

    if len(sorted_seasons) < 2:
        msg = "seasons must contain at least 2 seasons (need at least one training and one test season)"
        raise ValueError(msg)

    # Cache feature DataFrames — serve each season exactly once
    season_cache: dict[int, pd.DataFrame] = {}
    for year in sorted_seasons:
        season_cache[year] = feature_server.serve_season_features(year, mode=mode)

    # Walk-forward: iterate from second season onward as test candidates
    for i, test_year in enumerate(sorted_seasons[1:], start=1):
        # Skip no-tournament seasons (e.g., 2020 COVID cancellation)
        if test_year in _NO_TOURNAMENT_SEASONS:
            continue

        # Accumulate training data from all prior seasons
        train_frames: list[pd.DataFrame] = []
        for train_year in sorted_seasons[:i]:
            df = season_cache[train_year]
            if not df.empty:
                train_frames.append(df)

        train_df = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame()

        # Test data: tournament games only from the test year
        test_season_df = season_cache[test_year]
        if test_season_df.empty:
            test_df = test_season_df
        else:
            test_df = test_season_df[test_season_df["is_tournament"] == True].reset_index(  # noqa: E712
                drop=True,
            )

        yield CVFold(train=train_df, test=test_df, year=test_year)
