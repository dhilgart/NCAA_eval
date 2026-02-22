# Story 4.8: Implement Dynamic Rating Features (Elo Feature Building Block)

Status: review

## Story

As a data scientist,
I want a game-by-game Elo rating system that produces team ratings as features for the walk-forward feature pipeline,
So that I can capture in-season trajectory and momentum in addition to the full-season batch ratings from Story 4.6.

**Note:** This story implements Elo ratings as a **feature building block** (a rating computed from game history to feed as input to another model, e.g., XGBoost). Story 5.3 implements Elo as a complete predictive **model** — these are architecturally distinct. Do NOT implement model-level `train`/`predict` interfaces here.

## Acceptance Criteria

1. **Standard Elo Update** — Elo ratings are updated game-by-game from a configurable initial rating (default 1500): `r_new = r_old + K_eff × (actual − expected)`, where `expected = 1 / (1 + 10^((r_opponent − r_team)/400))`.

2. **Variable K-Factor** — The K-factor is configurable and supports variable-K: K=56 (early season, first 20 games per team) → K=38 (regular season) → K=47.5 (tournament games). K boundaries are configurable.

3. **Margin-of-Victory Scaling** — `K_eff = K × min(margin, max_margin)^0.85` (Silver/SBCB formula; diminishing returns on blowouts). `max_margin` is configurable (default 25).

4. **Home-Court Adjustment** — Subtract a configurable number of Elo points (default 3.5) from the home team's effective rating before computing expected outcome. Applied only when `loc == "H"` (home team is the winner); when `loc == "A"`, the adjustment favors the away team (loser).

5. **Season Mean-Reversion** — Between seasons, regress a configurable fraction (default 25%, range 20–35%) of each team's rating toward its **conference mean** to account for roster turnover. Requires `ConferenceLookup` from Story 4.3.

6. **Pre-Tournament Elo Snapshot** — A pre-tournament Elo snapshot (as of the last regular-season game) is available as a team-level feature column compatible with Story 4.7 matchup delta computation (`delta_elo`).

7. **Walk-Forward Compatibility** — Elo updates are computed incrementally game-by-game from the chronological serving API with no future data leakage. The Elo rating used for a game at day_num D reflects only games with day_num < D.

8. **Feature Serving Integration** — The `StatefulFeatureServer` (Story 4.7) is updated to populate `delta_elo` with actual computed values instead of `np.nan` when Elo is enabled in `FeatureConfig`.

9. **Unit Tests** — The Elo feature generator is covered by unit tests validating: rating updates, margin scaling, home court adjustment, variable K-factor transitions, season mean-reversion, and walk-forward temporal correctness.

## Tasks / Subtasks

- [x] Task 1: Implement `EloConfig` frozen dataclass (AC: #1, #2, #3, #4, #5)
  - [x] 1.1 Create `src/ncaa_eval/transform/elo.py`
  - [x] 1.2 Define `EloConfig` frozen dataclass with fields: `initial_rating`, `k_early`, `k_regular`, `k_tournament`, `early_game_threshold`, `margin_exponent`, `max_margin`, `home_advantage_elo`, `mean_reversion_fraction`
  - [x] 1.3 Sensible defaults: `initial_rating=1500`, `k_early=56`, `k_regular=38`, `k_tournament=47.5`, `early_game_threshold=20`, `margin_exponent=0.85`, `max_margin=25`, `home_advantage_elo=3.5`, `mean_reversion_fraction=0.25`
  - [x] 1.4 Unit tests for config defaults and frozen immutability

- [x] Task 2: Implement `EloFeatureEngine` core class (AC: #1, #2, #3, #4, #7)
  - [x] 2.1 Constructor accepts `EloConfig` and optional `ConferenceLookup`
  - [x] 2.2 Internal state: `_ratings: dict[int, float]` mapping `team_id → current_elo`; `_game_counts: dict[int, int]` mapping `team_id → games_played_this_season`
  - [x] 2.3 Implement `expected_score(rating_a: float, rating_b: float) -> float` — logistic function `1 / (1 + 10^((r_b − r_a)/400))`
  - [x] 2.4 Implement `_effective_k(team_id: int, is_tournament: bool) -> float` — variable K based on game count and tournament flag
  - [x] 2.5 Implement `_margin_multiplier(margin: int) -> float` — `min(margin, max_margin)^margin_exponent`
  - [x] 2.6 Implement `update_game(w_team_id, l_team_id, w_score, l_score, loc, is_tournament) -> tuple[float, float]` — returns (elo_w_before, elo_l_before) for feature use BEFORE updating ratings
  - [x] 2.7 Implement `get_rating(team_id: int) -> float` — returns current rating (or initial_rating if unseen)
  - [x] 2.8 Unit tests: basic update, expected score symmetry, margin scaling effect, home court effect, variable K transitions

- [x] Task 3: Implement season management (AC: #5, #7)
  - [x] 3.1 Implement `apply_season_mean_reversion(season: int) -> None` — regress each team toward conference mean; no-op if no `ConferenceLookup`
  - [x] 3.2 Conference mean computation: group teams by conference, compute mean Elo per conference, regress each team's rating by `fraction × (conference_mean − current_rating)` toward conference mean
  - [x] 3.3 Implement `reset_game_counts() -> None` — reset per-team game counts for new season (K-factor depends on season game count)
  - [x] 3.4 Implement `start_new_season(season: int) -> None` — orchestrates: `apply_season_mean_reversion(season)` then `reset_game_counts()`
  - [x] 3.5 Teams with no conference info (e.g., first appearance): regress toward global mean
  - [x] 3.6 Unit tests: mean reversion correctness, conference grouping, global fallback, game count reset

- [x] Task 4: Implement snapshot and bulk processing (AC: #6, #7)
  - [x] 4.1 Implement `get_all_ratings() -> dict[int, float]` — returns copy of current ratings dict
  - [x] 4.2 Implement `process_season(games: list[Game], season: int) -> pd.DataFrame` — processes all games for a season in chronological order, returns DataFrame with columns `[game_id, elo_w_before, elo_l_before]`
  - [x] 4.3 `process_season` calls `start_new_season(season)` at the beginning if prior-season ratings exist
  - [x] 4.4 Unit tests: snapshot correctness, process_season output schema, multi-season continuity

- [x] Task 5: Integrate with `FeatureConfig` and `StatefulFeatureServer` (AC: #8)
  - [x] 5.1 Add `elo_enabled: bool = False` and `elo_config: EloConfig | None = None` fields to `FeatureConfig`
  - [x] 5.2 Update `FeatureConfig.active_blocks()`: include `FeatureBlock.ELO` when `elo_enabled` is True
  - [x] 5.3 Update `StatefulFeatureServer.__init__()`: accept optional `elo_engine: EloFeatureEngine | None` parameter
  - [x] 5.4 Update `_serve_batch()`: if ELO active, run `elo_engine.process_season()` and populate `elo_a`/`elo_b` columns, then compute `delta_elo` in `_compute_matchup_deltas()`
  - [x] 5.5 Update `_serve_stateful()`: if ELO active, call `elo_engine.update_game()` per game, using the **before** ratings for feature values
  - [x] 5.6 Update `_build_game_row()`: add `elo_a`/`elo_b` from engine state before game update
  - [x] 5.7 Update `_compute_matchup_deltas()`: compute `delta_elo = elo_a − elo_b` when ELO active
  - [x] 5.8 Remove `np.nan` placeholder assignments for `delta_elo` in both modes when ELO is active; keep `np.nan` when ELO is disabled
  - [x] 5.9 Update `_empty_frame()` if needed
  - [x] 5.10 Unit tests: feature serving with Elo enabled vs disabled, delta_elo values populated

- [x] Task 6: Integration tests (AC: #7, #9)
  - [x] 6.1 Walk-forward temporal integrity: Elo rating for game at day_num D reflects only games with day_num < D
  - [x] 6.2 Multi-season continuity: ratings carry forward across seasons with mean-reversion
  - [x] 6.3 Batch/stateful equivalence: both modes produce identical `delta_elo` values for the same season
  - [x] 6.4 Feature serving round-trip: `StatefulFeatureServer` with Elo enabled produces non-NaN `delta_elo`

- [x] Task 7: Update `__init__.py` exports (AC: all)
  - [x] 7.1 Add `EloConfig`, `EloFeatureEngine` to `ncaa_eval/transform/__init__.py` imports and `__all__`
  - [x] 7.2 Run `mypy --strict`, `ruff check`, full `pytest` to verify no regressions

## Dev Notes

### Architecture & Design Constraints

- **Pure transform layer**: `elo.py` lives in `src/ncaa_eval/transform/` — no direct Parquet/SQLite IO. Accepts `Game` objects or pre-loaded DataFrames.
- **Not a model**: This is a **feature building block** — it computes Elo ratings to feed as input features to XGBoost or other models. Story 5.3 implements Elo as a Model ABC plugin with `train`/`predict`/`save` — DO NOT implement those interfaces here.
- **No `iterrows`**: All DataFrame construction must use list-accumulate-then-assign pattern (see Story 4.7 code review fix).
- **`from __future__ import annotations`**: Required first non-comment line in the new file.
- **`mypy --strict`**: Use `# type: ignore[import-untyped]` for pandas, numpy. No bare `Any`.
- **Logger pattern**: `logger = logging.getLogger(__name__)` at module level.
- **Empty-input guards**: Return empty DataFrame with correct columns on empty input; never raise.
- **Frozen dataclasses**: Use `tuple[str, ...]` for sequence fields to prevent content mutation.

### Existing Building Blocks (DO NOT Reimplement)

| Building Block | Module | Key API | Story |
|:---|:---|:---|:---|
| Chronological game serving | `transform.serving` | `ChronologicalDataServer.get_chronological_season(year)`, `.iter_games_by_date(year)` | 4.2 |
| OT rescaling | `transform.serving` | `rescale_overtime(score, num_ot)` | 4.2 |
| Conference lookup | `transform.normalization` | `ConferenceLookup.from_csv(path)`, `.get(season, team_id)` → `str \| None` | 4.3 |
| Feature serving | `transform.feature_serving` | `StatefulFeatureServer`, `FeatureConfig`, `FeatureBlock.ELO` | 4.7 |
| Batch rating solvers | `transform.opponent` | `compute_srs_ratings()`, `compute_ridge_ratings()`, `compute_colley_ratings()` | 4.6 |

### Elo Formula Reference

```python
# Expected score (logistic)
expected = 1.0 / (1.0 + 10.0 ** ((r_opponent - r_team) / 400.0))

# Effective K-factor with margin scaling
k_eff = k_base * min(margin, max_margin) ** 0.85

# Rating update
r_new = r_old + k_eff * (actual - expected)
# actual = 1.0 for win, 0.0 for loss

# Home court: subtract home_advantage_elo from home team's effective rating
# before computing expected score (not from the rating itself)

# Season mean-reversion toward conference mean:
# r_new = r_old + fraction * (conf_mean - r_old)
# = r_old * (1 - fraction) + conf_mean * fraction
```

### Variable K-Factor Logic

```python
# Per-team game count determines K phase:
if is_tournament:
    k_base = k_tournament  # 47.5 (tournament importance)
elif game_count < early_game_threshold:  # first 20 games
    k_base = k_early  # 56 (volatile early)
else:
    k_base = k_regular  # 38 (stabilized)
```

### Home Court Adjustment Design

The `loc` field in `Game` describes the **winner's** perspective: `H` = winner was home, `A` = winner was away, `N` = neutral. For Elo computation:

- If `loc == "H"`: winner had home advantage → subtract `home_advantage_elo` from winner's effective rating before computing expected score (deflates their expected win probability, giving them less credit for winning at home)
- If `loc == "A"`: loser had home advantage → subtract `home_advantage_elo` from loser's effective rating before computing expected score
- If `loc == "N"`: no adjustment

### Feature Serving Integration Points

The `StatefulFeatureServer` in `feature_serving.py` needs these changes:

1. **`FeatureConfig`** — Add `elo_enabled: bool = False` and `elo_config: EloConfig | None = None`
2. **`FeatureConfig.active_blocks()`** — Include `FeatureBlock.ELO` when `elo_enabled is True`
3. **`StatefulFeatureServer.__init__()`** — Accept optional `elo_engine: EloFeatureEngine | None`
4. **`_serve_batch()`** — Run `elo_engine.process_season()`, merge results as `elo_a`/`elo_b` columns, compute `delta_elo` in deltas
5. **`_serve_stateful()`** — Call `elo_engine.update_game()` per game; use **before** ratings
6. **`_build_game_row()`** — Add `elo_a`/`elo_b` from engine pre-update state
7. **`_compute_matchup_deltas()`** — Add `delta_elo = elo_a − elo_b`
8. **Keep `delta_elo = np.nan`** when Elo is disabled (backward compatibility)

### Critical Temporal Safety Invariant

For game G at day_num D:
- The Elo rating used as a feature for game G must reflect ONLY games processed before G in chronological order
- `update_game()` returns the **before** ratings, then updates internal state
- This ensures the feature value for G does not include G's outcome

### Season Mean-Reversion Details

- Between seasons, call `start_new_season(season)` which:
  1. Groups all rated teams by their conference (via `ConferenceLookup.get(season, team_id)`)
  2. Computes conference mean Elo
  3. Regresses each team: `new_rating = old_rating + fraction × (conf_mean - old_rating)`
  4. Teams without conference info → regress toward global mean (mean of all rated teams)
  5. Resets per-team game counts to 0 (for variable K-factor)
- **First season**: no reversion (no prior ratings exist), all teams start at `initial_rating`

### OT Score Rescaling

The `Game` objects from `ChronologicalDataServer` already have raw scores. Apply `rescale_overtime(score, num_ot)` from `transform.serving` to normalize OT scores to 40-minute equivalent before computing margin for K_eff. Import and use the existing function — do NOT reimplement.

### Previous Story Learnings (Critical)

**From Story 4.7 (Feature Serving):**
- `StatefulFeatureServer` uses list-accumulate-then-assign pattern (not `df.at` cell writes)
- `_build_batch_indexed()` pre-indexes batch ratings by `team_id` once for O(1) lookups
- `active_blocks()` returns a `frozenset[FeatureBlock]` — check `FeatureBlock.ELO in active`
- `_compute_matchup_deltas()` pattern: check column existence before computing delta
- Code review H2/M1/M2: performance-critical paths cache computed values above loops
- `elo_config` field needs `field(default=None)` in the dataclass (mutable default avoidance)

**From Story 4.6 (Batch Ratings):**
- Batch solvers operate on regular-season games only (`is_tournament == False`)
- Data contract: return `DataFrame[team_id, rating_column]`
- Margin capping at 25 points prevents extreme-blowout distortion

**Cross-story patterns:**
- Every public method needs ≥1 test
- Empty-input guards mandatory (return empty DF, don't raise)
- Frozen dataclasses only prevent attribute rebinding — use `tuple` for immutable sequences
- Logger testing: `unittest.mock.patch("ncaa_eval.transform.elo.logger")`
- `# type: ignore[import-untyped]` for pandas, numpy

### Project Structure Notes

- New file: `src/ncaa_eval/transform/elo.py`
- New test files: `tests/unit/test_elo.py`, `tests/integration/test_elo_integration.py`
- Modified: `src/ncaa_eval/transform/feature_serving.py` (add Elo integration)
- Modified: `src/ncaa_eval/transform/__init__.py` (add exports)
- Do NOT modify existing test files for other stories unless fixing a broken import

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 4.8]
- [Source: specs/research/feature-engineering-techniques.md — Section 6.7 (Elo specification), Section 7.2 (Distinct Building Blocks)]
- [Source: src/ncaa_eval/transform/feature_serving.py — StatefulFeatureServer, FeatureConfig, FeatureBlock.ELO placeholder]
- [Source: src/ncaa_eval/transform/serving.py — ChronologicalDataServer, rescale_overtime()]
- [Source: src/ncaa_eval/transform/normalization.py — ConferenceLookup API]
- [Source: src/ncaa_eval/transform/opponent.py — BatchRatingSolver pattern (contrast: batch vs. stateful)]
- [Source: _bmad-output/implementation-artifacts/4-7-implement-stateful-feature-serving.md — Previous story learnings, integration points]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

- Home-court test direction: Initially wrote tests expecting home-win to give less credit, but AC #4 literally says "subtract from home team's effective rating" which lowers expected score → bigger surprise → more credit. Tests corrected to match AC spec.

### Completion Notes List

- **Tasks 1-4**: Implemented `EloConfig` frozen dataclass and `EloFeatureEngine` in `src/ncaa_eval/transform/elo.py`. Engine supports: logistic expected score, variable K-factor (early/regular/tournament), margin-of-victory scaling with configurable cap and exponent, home-court adjustment via effective rating subtraction, season mean-reversion toward conference mean (with global mean fallback), OT score rescaling via `rescale_overtime()`, and bulk `process_season()` returning before-ratings. 36 unit tests.
- **Task 5**: Integrated Elo into `StatefulFeatureServer` — added `elo_enabled`/`elo_config` to `FeatureConfig`, `elo_engine` parameter to server, wired both batch mode (via `process_season()`) and stateful mode (via `update_game()` per game), computed `delta_elo` in `_compute_matchup_deltas()`. Backward-compatible: `delta_elo = np.nan` when Elo disabled. 10 new unit tests in `test_feature_serving.py`.
- **Task 6**: 10 integration tests validating walk-forward temporal integrity, multi-season continuity with conference mean-reversion, batch/stateful mode equivalence for `delta_elo`, and feature serving round-trip.
- **Task 7**: Exported `EloConfig` and `EloFeatureEngine` from `transform/__init__.py`. Full suite: 388 tests pass, mypy --strict clean (47 files), ruff clean.

### Change Log

- 2026-02-22: Implemented Story 4.8 — Elo feature building block with full test coverage (46 new unit tests + 10 integration tests). All acceptance criteria satisfied.

### File List

- `src/ncaa_eval/transform/elo.py` (new) — EloConfig, EloFeatureEngine
- `src/ncaa_eval/transform/feature_serving.py` (modified) — FeatureConfig elo_enabled/elo_config, StatefulFeatureServer elo_engine integration
- `src/ncaa_eval/transform/__init__.py` (modified) — EloConfig, EloFeatureEngine exports
- `tests/unit/test_elo.py` (new) — 36 unit tests for Elo engine
- `tests/unit/test_feature_serving.py` (modified) — 10 new Elo integration tests
- `tests/integration/test_elo_integration.py` (new) — 10 integration tests
- `_bmad-output/implementation-artifacts/sprint-status.yaml` (modified) — status update
