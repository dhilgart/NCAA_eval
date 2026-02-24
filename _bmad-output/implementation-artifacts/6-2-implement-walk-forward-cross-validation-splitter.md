# Story 6.2: Implement Walk-Forward Cross-Validation Splitter

Status: review

## Story

As a data scientist,
I want a "Leave-One-Tournament-Out" cross-validation splitter with strict temporal boundaries,
So that I can backtest models across multiple years without data leakage.

## Acceptance Criteria

1. **Given** historical game data spanning multiple seasons, **When** the developer uses the CV splitter to generate train/test folds, **Then** each fold uses one tournament year as the test set and all prior years as training data.
2. Strict temporal boundaries ensure no future data appears in any training fold.
3. The 2020 COVID year is handled gracefully: models receive training data but no test evaluation is attempted (season has `has_tournament=False`).
4. The splitter yields `(train_data, test_data, year)` tuples for each fold.
5. The splitter is compatible with both stateful models (chronological iteration) and stateless models (batch splits).
6. The splitter is covered by unit tests validating temporal integrity and 2020 handling.
7. Fold boundaries are deterministic and reproducible.

## Tasks / Subtasks

- [x] Task 1: Create `src/ncaa_eval/evaluation/splitter.py` (AC: #1–#5, #7)
  - [x] 1.1 Define `CVFold` frozen dataclass: `train: pd.DataFrame`, `test: pd.DataFrame`, `year: int`
  - [x] 1.2 Implement `walk_forward_splits(seasons: Sequence[int], feature_server: StatefulFeatureServer, *, mode: str = "batch") -> Iterator[CVFold]`
  - [x] 1.3 Implement temporal boundary logic: for test year `Y`, training data = all seasons `< Y`
  - [x] 1.4 Skip 2020 as a test year: include 2020 regular-season data in training folds for subsequent years, but never yield a fold with `year=2020` (no tournament to evaluate)
  - [x] 1.5 Support `mode` parameter for stateful (`"stateful"`) vs stateless (`"batch"`) feature serving
  - [x] 1.6 Ensure deterministic output: same inputs always produce identical fold DataFrames
- [x] Task 2: Export public API from `src/ncaa_eval/evaluation/__init__.py` (AC: #4)
  - [x] 2.1 Add `CVFold` and `walk_forward_splits` to imports and `__all__`
- [x] Task 3: Create `tests/unit/test_evaluation_splitter.py` (AC: #6)
  - [x] 3.1 Test basic fold generation: correct number of folds for a season range
  - [x] 3.2 Test temporal integrity: for every fold, all training data seasons < test year
  - [x] 3.3 Test 2020 exclusion: no fold has `year=2020`
  - [x] 3.4 Test 2020 training inclusion: folds for years > 2020 include 2020 regular-season data in training
  - [x] 3.5 Test fold determinism: two identical calls produce identical results
  - [x] 3.6 Test single-season range: raises `ValueError` (need at least 2 seasons for train + test)
  - [x] 3.7 Test empty season: fold is still generated if repository returns data
  - [x] 3.8 Test that test data contains only tournament games (the evaluation target)
  - [x] 3.9 Test that training data contains all games (regular season + tournament) from prior years
  - [x] 3.10 Test `mode` parameter is passed through to feature server

## Dev Notes

### Design Specification

The CV splitter implements **Leave-One-Tournament-Out (LOTO)** cross-validation:

```
For seasons [2008, 2009, ..., 2025]:
  Fold 1: train=[2008],           test=2009 tournament games
  Fold 2: train=[2008, 2009],     test=2010 tournament games
  ...
  Fold N: train=[2008..2019],     test=2020 → SKIPPED (no tournament)
  Fold N+1: train=[2008..2020],   test=2021 tournament games  (2020 regular season IS in training)
  ...
```

Each fold's **test set** contains ONLY tournament games (`is_tournament == True`) from the test year. Each fold's **training set** contains ALL games (regular season + tournament) from all prior years.

### Critical Implementation Constraints

1. **Library-First Rule**: No external CV library needed — scikit-learn's `BaseCrossValidator` is designed for i.i.d. row-level splits, not temporal walk-forward. A custom iterator is the correct approach here.
2. **`from __future__ import annotations`** required in all Python files (Ruff enforcement).
3. **`mypy --strict`** mandatory — use proper type annotations with `Sequence`, `Iterator`, `pd.DataFrame`.
4. **Mutation testing**: `src/ncaa_eval/evaluation/` is in `[tool.mutmut]` paths — tests must catch subtle mutations (e.g., `<` vs `<=` in temporal boundaries, off-by-one in season ranges).
5. **Vectorization (NFR1)**: DataFrame filtering operations, not Python loops over games. Use boolean indexing for train/test splits.

### File Structure

```
src/ncaa_eval/evaluation/
├── __init__.py          # MODIFY — add CVFold, walk_forward_splits exports
├── metrics.py           # Existing — DO NOT MODIFY
├── splitter.py          # NEW — CV splitter implementation
```

```
tests/unit/
├── test_evaluation_metrics.py    # Existing — DO NOT MODIFY
├── test_evaluation_splitter.py   # NEW — comprehensive splitter tests
```

### Dependencies

All required libraries are already in `pyproject.toml`:
- `pandas` — DataFrame operations for train/test splits
- No new dependencies needed.

### Existing Codebase Context — DO NOT Reimplement

- **`src/ncaa_eval/transform/serving.py`**: `ChronologicalDataServer` provides `get_chronological_season(year)` returning `SeasonGames` with `has_tournament: bool`. The splitter should use `SeasonGames.has_tournament` to detect 2020 (not hardcode the year check). The constant `_NO_TOURNAMENT_SEASONS = frozenset({2020})` is defined there.
- **`src/ncaa_eval/transform/feature_serving.py`**: `StatefulFeatureServer.serve_season_features(year, mode)` returns a `pd.DataFrame` with columns including `game_id`, `season`, `day_num`, `date`, `team_a_id`, `team_b_id`, `is_tournament`, `loc_encoding`, `team_a_won`, plus feature columns. The splitter should call this to get feature matrices per season.
- **`src/ncaa_eval/cli/train.py`**: The existing `run_training()` function builds features per season and concatenates them. The splitter follows the same pattern but partitions into train/test folds. Use `METADATA_COLS` from `train.py` to understand which columns are metadata vs features.
- **`src/ncaa_eval/model/base.py`**: `Model.fit(X, y)` and `Model.predict_proba(X)` accept DataFrames. `StatefulModel.fit()` processes games chronologically (needs full DataFrame with metadata). Stateless models need only feature columns. The splitter yields raw DataFrames — downstream consumers (Story 6.3) handle column selection.

### Data Flow Architecture

```
Repository → ChronologicalDataServer → StatefulFeatureServer → CV Splitter
                                                                    ↓
                                                            CVFold(train_df, test_df, year)
                                                                    ↓
                                                        Story 6.3: Parallel Execution
                                                                    ↓
                                                        Story 6.1: Metric Library
```

The splitter sits between feature serving and parallel execution. It produces DataFrames that Story 6.3 will consume for parallel model training and evaluation.

### Function Signature

```python
@dataclasses.dataclass(frozen=True)
class CVFold:
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
    """
```

### Key Implementation Details

1. **Season ordering**: Sort `seasons` ascending. Iterate from the second season onward (first season is training-only).
2. **Feature caching**: Call `feature_server.serve_season_features(year, mode=mode)` once per season and cache the DataFrame. Do NOT re-serve features for the same season across folds.
3. **Tournament filtering**: `test_df = season_df[season_df["is_tournament"] == True]`. If a test year has no tournament games AND `has_tournament` is True, yield an empty test DataFrame (the caller decides what to do).
4. **2020 handling**: Use `ChronologicalDataServer`'s `SeasonGames.has_tournament` logic. Since the splitter uses `StatefulFeatureServer` (not `ChronologicalDataServer` directly), check the `is_tournament` column: if a season has ZERO tournament games, check whether it's a known no-tournament season. The simplest approach: import `_NO_TOURNAMENT_SEASONS` from `serving.py` OR just check if the test year's DataFrame has any `is_tournament == True` rows. If no tournament rows exist, skip yielding a fold for that year.
5. **Training accumulation**: Use `pd.concat()` to accumulate training DataFrames across seasons. Reset index after concat to avoid duplicate indices.
6. **Input validation**: Raise `ValueError` if `seasons` has fewer than 2 elements (need at least one training season and one test season).

### Testing Approach

Tests should use **mock/fake** feature servers to avoid needing real data. Create a helper that generates synthetic DataFrames with the required columns:

```python
def _make_season_df(year: int, n_regular: int = 10, n_tournament: int = 3) -> pd.DataFrame:
    """Create a minimal synthetic season DataFrame for testing."""
    # Include required columns: season, is_tournament, game_id, team_a_id, etc.
    # Regular season games + tournament games
```

Use `unittest.mock.MagicMock` or a simple wrapper class for `StatefulFeatureServer` that returns pre-built DataFrames. This keeps tests fast, isolated, and independent of the data pipeline.

### Edge Cases

| Edge Case | Expected Behavior |
|---|---|
| `seasons` has < 2 elements | `ValueError` |
| 2020 in seasons | 2020 included in training data but no fold yielded with `year=2020` |
| Test year has 0 tournament games (non-2020) | Yield fold with empty `test` DataFrame |
| Empty season (no games at all) | Include in training (empty concat is harmless), skip as test year |
| Single test year (2 seasons total) | Yield exactly 1 fold |
| `seasons` not sorted | Splitter sorts internally — deterministic regardless of input order |

### Previous Story Learnings (Story 6.1)

- **`# type: ignore[import-untyped]`** required for pandas imports (`import pandas as pd  # type: ignore[import-untyped]`)
- **`frozen=True` dataclass**: Use for `CVFold` — immutable fold results prevent caller mutation bugs
- **Test organization**: Class-based test structure using `TestClassName` pattern (e.g., `TestWalkForwardSplits`, `TestCVFold`, `TestEdgeCases`)
- **Assertion patterns**: Use `pd.testing.assert_frame_equal()` for DataFrame comparison in determinism tests
- **`np.random.default_rng(seed)`** for reproducible synthetic test data
- **sklearn import pattern**: Not needed here — no sklearn dependencies in the splitter

### Project Structure Notes

- `src/ncaa_eval/evaluation/` already exists with `__init__.py` and `metrics.py` — new file `splitter.py` follows the same module pattern
- No conflicts with existing code — the splitter is a new component that consumes existing APIs
- The splitter's output (`CVFold`) is the input to Story 6.3 (Parallel CV Execution) — design the interface to be simple and self-contained

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Epic 6, Story 6.2 (lines 724–740)]
- [Source: src/ncaa_eval/transform/serving.py — ChronologicalDataServer, SeasonGames, _NO_TOURNAMENT_SEASONS]
- [Source: src/ncaa_eval/transform/feature_serving.py — StatefulFeatureServer.serve_season_features()]
- [Source: src/ncaa_eval/cli/train.py — run_training() pattern, METADATA_COLS]
- [Source: src/ncaa_eval/model/base.py — Model ABC, StatefulModel template]
- [Source: src/ncaa_eval/evaluation/metrics.py — module pattern, type annotations]
- [Source: _bmad-output/implementation-artifacts/6-1-implement-metric-library-scikit-learn-numpy.md — previous story learnings]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

No issues encountered during implementation.

### Completion Notes List

- Implemented `CVFold` frozen dataclass and `walk_forward_splits` iterator in `splitter.py`
- Uses `_NO_TOURNAMENT_SEASONS` from `serving.py` for 2020 skip logic (not hardcoded year check)
- Feature caching: each season served exactly once, cached in dict for O(1) lookup across folds
- Training accumulation via `pd.concat(..., ignore_index=True)` prevents duplicate indices
- Tournament filtering via boolean indexing (`is_tournament == True`) — vectorized, no Python loops
- Seasons sorted internally for deterministic output regardless of input order
- Input validation raises `ValueError` for < 2 seasons
- Exported `CVFold` and `walk_forward_splits` from `evaluation/__init__.py` and `__all__`
- 19 unit tests covering: fold count, temporal integrity, 2020 exclusion/inclusion, determinism, edge cases (empty seasons, unsorted input, single fold, feature caching, index reset, ascending fold order)
- All 572 project tests pass. Ruff and mypy --strict clean.

### File List

- `src/ncaa_eval/evaluation/splitter.py` — NEW: CVFold dataclass + walk_forward_splits function
- `src/ncaa_eval/evaluation/__init__.py` — MODIFIED: added CVFold, walk_forward_splits exports
- `tests/unit/test_evaluation_splitter.py` — NEW: 19 unit tests for splitter
- `_bmad-output/implementation-artifacts/6-2-implement-walk-forward-cross-validation-splitter.md` — MODIFIED: story status/tasks
- `_bmad-output/implementation-artifacts/sprint-status.yaml` — MODIFIED: story status → review

## Change Log

- 2026-02-23: Implemented walk-forward CV splitter with Leave-One-Tournament-Out logic, 2020 COVID handling, feature caching, and comprehensive test suite (19 tests)
