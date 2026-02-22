# Story 4.7: Implement Stateful Feature Serving

Status: ready-for-dev

## Story

As a data scientist,
I want a feature serving layer that combines all active feature transformations into a temporally-safe feature matrix, with in-fold probability calibration and matchup-level feature support,
So that models receive a consistent, leakage-free feature matrix with calibrated probability outputs.

## Acceptance Criteria

1. **Declarative Feature Configuration** — The serving layer combines all active feature transformations (sequential, graph, batch rating, dynamic rating, normalization) into a unified feature matrix via declarative configuration (specify which building blocks to activate).

2. **Chronological Ordering** — Features are served in strict chronological order matching the data serving API (Story 4.2's `ChronologicalDataServer`).

3. **Temporal Safety** — The serving layer enforces that no feature computation uses future data relative to the prediction point.

4. **Dual Consumption Modes** — The serving layer supports both stateful (per-game iteration) and stateless (batch) consumption modes.

5. **Massey Ordinal Temporal Slicing** — For each game at date D, only ordinals with `RankingDayNum` published ≤ D are used — prevents ordinal leakage during walk-forward backtesting.

6. **Matchup-Level Features** — Matchup features are computed as team_A − team_B deltas: seed differential (`seed_num_A − seed_num_B`), ordinal rank deltas, Elo delta, SRS delta — these are the primary matchup signals for tournament prediction.

7. **In-Fold Probability Calibration** — Isotonic regression or cubic-spline calibration is fitted on training fold predictions and applied to test fold predictions. The `goto_conversion` package is assessed as an alternative. Calibration is always in-fold to prevent leakage.

8. **Scope Parameters** — `gender_scope` and `dataset_scope` are configurable parameters on the feature server (e.g., men's vs. women's; Kaggle-only vs. ESPN-enriched games).

9. **Integration Tests** — The feature serving pipeline is covered by integration tests validating end-to-end temporal integrity, calibration leakage prevention, and matchup-level delta correctness.

## Tasks / Subtasks

- [ ] Task 1: Define `FeatureConfig` dataclass (AC: #1, #4, #8)
  - [ ] 1.1 Create `src/ncaa_eval/transform/feature_serving.py`
  - [ ] 1.2 Define `FeatureConfig` frozen dataclass with fields: `sequential_windows`, `ewma_alphas`, `graph_features_enabled`, `batch_rating_types`, `ordinal_systems`, `ordinal_composite`, `matchup_deltas`, `gender_scope`, `dataset_scope`, `calibration_method`
  - [ ] 1.3 Define `FeatureBlock` enum: `SEQUENTIAL`, `GRAPH`, `BATCH_RATING`, `ORDINAL`, `SEED`, `ELO` (Elo placeholder for Story 4.8)
  - [ ] 1.4 Unit tests for config validation and defaults

- [ ] Task 2: Implement `StatefulFeatureServer` core class (AC: #1, #2, #3, #4)
  - [ ] 2.1 Constructor accepts `FeatureConfig`, `ChronologicalDataServer`, data lookup objects (normalization tables, CSV paths)
  - [ ] 2.2 Implement `serve_season_features(year, mode="batch"|"stateful")` returning a `pd.DataFrame` with one row per game and all configured feature columns
  - [ ] 2.3 Batch mode: compute all features for the full season at once (for stateless models like XGBoost)
  - [ ] 2.4 Stateful mode: yield feature rows game-by-game via iterator, accumulating state (graph, sequential) incrementally
  - [ ] 2.5 Internal orchestration order: load games → compute sequential features → compute graph features → load batch ratings → load ordinals → assemble matchup deltas
  - [ ] 2.6 Unit tests for both consumption modes

- [ ] Task 3: Implement Massey ordinal temporal slicing (AC: #5)
  - [ ] 3.1 Use `MasseyOrdinalsStore.get_snapshot(season, day_num)` — this already filters by `RankingDayNum ≤ day_num` (implemented in Story 4.3)
  - [ ] 3.2 For each game, slice ordinals at the game's `day_num` (not a global season snapshot)
  - [ ] 3.3 Unit tests: verify ordinal features at game G use only ordinals published before G

- [ ] Task 4: Implement matchup-level feature computation (AC: #6)
  - [ ] 4.1 Compute team-pair deltas: `feature_A − feature_B` for all configured features
  - [ ] 4.2 Seed differential: `seed_num_A − seed_num_B` (using `TourneySeedTable` from Story 4.3; `None`/NaN for non-tournament or unseeded games)
  - [ ] 4.3 Ordinal rank deltas per composite system
  - [ ] 4.4 Batch rating deltas (SRS, Ridge, Colley)
  - [ ] 4.5 Elo delta placeholder (column present with NaN; populated when Story 4.8 is implemented)
  - [ ] 4.6 Unit tests for delta correctness (A−B symmetry: if team order flips, deltas negate)

- [ ] Task 5: Implement in-fold probability calibration (AC: #7)
  - [ ] 5.1 Create `src/ncaa_eval/transform/calibration.py`
  - [ ] 5.2 Implement `IsotonicCalibrator` wrapping `sklearn.isotonic.IsotonicRegression`
  - [ ] 5.3 Implement `fit(y_true, y_prob)` and `transform(y_prob)` interface
  - [ ] 5.4 Evaluate `goto_conversion` package: if it provides better calibration than isotonic for small samples, document assessment and expose as an option
  - [ ] 5.5 Ensure calibration is in-fold only: `fit()` on training fold predictions, `transform()` on test fold predictions — never fit on the data being calibrated
  - [ ] 5.6 Unit tests: calibration leakage prevention test (fit and transform on disjoint data), output probabilities in [0,1], monotonicity for isotonic

- [ ] Task 6: Implement scope filtering (AC: #8)
  - [ ] 6.1 `gender_scope` filters team/game data (men's vs. women's) — currently men-only in MVP; parameter presence future-proofs the API
  - [ ] 6.2 `dataset_scope` filters by data source: `"kaggle"` (Kaggle-only games), `"all"` (Kaggle + ESPN enrichment) — controls 2025 dedup behavior
  - [ ] 6.3 Unit tests for scope filtering

- [ ] Task 7: Integration tests (AC: #9)
  - [ ] 7.1 End-to-end temporal integrity: features at game G contain no data from games after G
  - [ ] 7.2 Calibration leakage prevention: calibrator not fit on test fold data
  - [ ] 7.3 Matchup delta correctness: A−B = −(B−A) for all feature deltas
  - [ ] 7.4 Roundtrip test: batch mode and stateful mode produce identical feature values for the same season

- [ ] Task 8: Update `__init__.py` exports (AC: all)
  - [ ] 8.1 Add all new public classes/functions to `ncaa_eval/transform/__init__.py` in alphabetical order
  - [ ] 8.2 Run `mypy --strict`, `ruff check`, full `pytest` to verify no regressions

## Dev Notes

### Architecture & Design Constraints

- **Pure transform layer**: `feature_serving.py` lives in `src/ncaa_eval/transform/` — no direct Parquet/SQLite IO. Accepts pre-loaded data or uses `ChronologicalDataServer` (which wraps Repository).
- **No `iterrows`**: All feature computation must be vectorized (pandas/numpy). `itertuples` acceptable only for non-vectorizable branching.
- **`from __future__ import annotations`**: Required first non-comment line in all new Python files.
- **`mypy --strict`**: Use `# type: ignore[import-untyped]` for pandas, numpy, sklearn, networkx. No bare `Any`.
- **Logger pattern**: `logger = logging.getLogger(__name__)` at module level.
- **Empty-input guards**: Return empty DataFrame with correct columns on empty input; never raise.

### Existing Building Blocks (DO NOT Reimplement)

| Building Block | Module | Key API | Story |
|:---|:---|:---|:---|
| Chronological game serving | `transform.serving` | `ChronologicalDataServer.get_chronological_season(year)`, `.iter_games_by_date(year)` | 4.2 |
| OT rescaling | `transform.serving` | `rescale_overtime(score, num_ot)` | 4.2 |
| Team name normalization | `transform.normalization` | `TeamNameNormalizer.from_csv(path)` | 4.3 |
| Tourney seeds | `transform.normalization` | `TourneySeedTable.from_csv(path)`, `.get(season, team_id)` → `TourneySeed` | 4.3 |
| Conference lookup | `transform.normalization` | `ConferenceLookup.from_csv(path)` | 4.3 |
| Massey ordinals (temporal) | `transform.normalization` | `MasseyOrdinalsStore.from_csv(path)`, `.get_snapshot(season, day_num)`, `.composite_simple_average(...)` etc. | 4.3 |
| Sequential features | `transform.sequential` | `SequentialTransformer.transform(team_games, ...)` → DataFrame with rolling, EWMA, momentum, streak, per-poss, four-factors | 4.4 |
| Detailed results loading | `transform.sequential` | `DetailedResultsLoader.load(csv_paths)` → long-format DataFrame (1 row per team per game, 2003+) | 4.4 |
| Graph centrality | `transform.graph` | `GraphTransformer.transform(games_df, reference_day_num)` → DataFrame with pagerank, betweenness, hits_hub, hits_authority, clustering | 4.5 |
| Incremental graph | `transform.graph` | `GraphTransformer.add_game_to_graph(graph, ...)` | 4.5 |
| SRS ratings | `transform.opponent` | `compute_srs_ratings(games_df)` → DataFrame[team_id, srs] | 4.6 |
| Ridge ratings | `transform.opponent` | `compute_ridge_ratings(games_df)` → DataFrame[team_id, ridge] | 4.6 |
| Colley ratings | `transform.opponent` | `compute_colley_ratings(games_df)` → DataFrame[team_id, colley] | 4.6 |

### Data Coverage Notes

- **Detailed box scores** (for sequential features): 2003+ only. Pre-2003 has compact data (scores only).
- **Graph features**: Work on compact data (scores, margins, team IDs). Available 1985+.
- **Batch ratings**: Work on compact data. Available 1985+.
- **Massey ordinals**: 2003–2025 (100+ systems). Coverage varies by system — use `CoverageGateResult` from Story 4.3.
- **Tournament seeds**: Available for all tournament years.
- **2025 dedup**: Handled by `ChronologicalDataServer` — caller does NOT need to deduplicate.

### Feature Matrix Schema

The output DataFrame should have one row per game with columns:

```
game_id, season, day_num, date, team_a_id, team_b_id, is_tournament,
# Sequential feature deltas (team_a − team_b):
delta_rolling_5_score, delta_rolling_10_score, ..., delta_ewma_020_score, ...,
delta_momentum_score, ..., delta_streak,
delta_efg_pct, delta_orb_pct, delta_ftr, delta_to_pct,
# Graph feature deltas:
delta_pagerank, delta_betweenness, delta_hits_authority, delta_clustering,
# Batch rating deltas:
delta_srs, delta_ridge, delta_colley,
# Ordinal deltas:
delta_ordinal_composite, delta_ordinal_pom, ...,
# Seed differential:
seed_diff,  # seed_num_A − seed_num_B (NaN if non-tournament)
# Elo delta (placeholder):
delta_elo,  # NaN until Story 4.8
# Game metadata:
loc_encoding,  # H=+1, A=−1, N=0 (from team_a perspective)
# Target:
team_a_won  # boolean label
```

**Convention**: `team_a` is always the winning team from the Game record (`w_team_id`), but features should be computed such that the model can predict from either team's perspective. For tournament predictions, team_a/team_b are the two teams in the matchup (not necessarily winner/loser).

### Calibration Design Notes

- **`sklearn.isotonic.IsotonicRegression`**: Use `y_min=0.0, y_max=1.0, out_of_bounds="clip"` for probability bounds.
- **In-fold only**: The calibrator is fitted on training predictions and applied to test predictions. Never fit calibration on the same data being calibrated (leakage).
- **Sample size concern**: Isotonic regression overfits below ~1000 calibration samples. For small folds, consider sigmoid (Platt scaling) as fallback.
- **`goto_conversion`** (PyPI: `goto-conversion`): A Kaggle-proven package for probability calibration, specifically designed for betting odds conversion. Assess whether its methods (Shin model) apply to game probability calibration. If not directly applicable, document and stick with isotonic/sigmoid.
- Calibration is a **separate concern** from feature computation — keep it in its own module (`calibration.py`) so it can be reused by the evaluation engine (Epic 6).

### Stateful vs. Batch Mode

- **Batch mode** (for XGBoost-style models): Compute all features for all games in a season, return a single DataFrame. Graph features use full-season snapshot. Sequential features use pre-computed rolling stats.
- **Stateful mode** (for Elo-style models): Iterate game-by-game. After each game, update:
  - Sequential accumulators (rolling windows, EWMA)
  - Graph (add edge via `add_game_to_graph`)
  - Elo ratings (placeholder — Story 4.8)
  - Yield the feature row for the current game **before** the game outcome is known to the model (temporal safety).

### Walk-Forward Temporal Safety Invariant

For any game G at date D:
1. Sequential features for teams in G use only games with date < D
2. Graph features use only edges from games with date < D
3. Ordinal features use only ordinals with `RankingDayNum` < D's `day_num`
4. Batch ratings are full-season pre-tournament snapshots (computed from all regular-season games) — these are **allowed** for tournament prediction because they use only data before the tournament
5. Seed information is available once seeds are published (day_num ≈ 134) — for regular-season games, seed features are NaN

### Previous Story Learnings (Critical)

**From Story 4.4 (Sequential):**
- Box scores NOT in Repository — load from CSV via `DetailedResultsLoader`
- Orchestration order matters: OT rescale → weights → rolling → EWMA → momentum → streak → possessions → per-poss → four factors
- Per-possession formula: `FGA − OR + TO + 0.44 × FTA` with zero-guard

**From Story 4.5 (Graph):**
- PageRank warm-start via `nstart` parameter — use for walk-forward efficiency
- HITS may fail to converge — handle `PowerIterationFailedConvergence` (returns uniform scores)
- Clustering coefficient requires `G.to_undirected()`

**From Story 4.6 (Opponent):**
- Batch solvers operate on **regular-season games only** (`is_tournament == False`) — caller must pre-filter
- SRS convergence: ~3000–5000 iterations typical; log warning if max_iter exhausted
- Data contract: all three solvers return `DataFrame[team_id, {rating_type}]`

**From Story 4.3 (Normalization):**
- `MasseyOrdinalsStore.get_snapshot()` uses `groupby + idxmax` for temporal filtering
- Coverage gate checks SAG & WLK for 2003–2025; fallback to MOR+POM+DOL
- sklearn PCA lazy-imported to avoid top-level `# type: ignore`

**Cross-story patterns:**
- Every public method needs ≥1 test
- Empty-input guards mandatory (return empty DF, don't raise)
- Frozen dataclasses only prevent attribute rebinding, not content mutation — use `tuple[str, ...]`
- Logger testing: `unittest.mock.patch("ncaa_eval.transform.MODULE.logger")`
- `# type: ignore[import-untyped]` for pandas, numpy, sklearn, networkx

### Project Structure Notes

- New files: `src/ncaa_eval/transform/feature_serving.py`, `src/ncaa_eval/transform/calibration.py`
- New test files: `tests/unit/test_feature_serving.py`, `tests/unit/test_calibration.py`
- Modified: `src/ncaa_eval/transform/__init__.py` (add exports)
- No changes to existing transformation modules — compose them, don't modify them

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 4.7]
- [Source: specs/05-architecture-fullstack.md — Repository Pattern, Strategy Pattern, Transform module]
- [Source: specs/03-prd.md — FR4 (Chronological Serving), FR5 (Advanced Transformations), NFR1 (Vectorization), NFR4 (Leakage Prevention)]
- [Source: src/ncaa_eval/transform/serving.py — ChronologicalDataServer API]
- [Source: src/ncaa_eval/transform/normalization.py — MasseyOrdinalsStore, TourneySeedTable, ConferenceLookup]
- [Source: src/ncaa_eval/transform/sequential.py — SequentialTransformer, DetailedResultsLoader]
- [Source: src/ncaa_eval/transform/graph.py — GraphTransformer]
- [Source: src/ncaa_eval/transform/opponent.py — BatchRatingSolver, compute_srs/ridge/colley_ratings]
- [Source: sklearn.isotonic.IsotonicRegression — probability calibration]
- [Source: goto-conversion PyPI — Kaggle probability calibration package]
- [Source: _bmad-output/implementation-artifacts/4-6-implement-opponent-adjustments.md — Previous story learnings]
- [Source: _bmad-output/implementation-artifacts/4-5-implement-graph-builders-centrality-features.md — Graph patterns]
- [Source: _bmad-output/implementation-artifacts/4-4-implement-sequential-transformations.md — Sequential patterns]

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
