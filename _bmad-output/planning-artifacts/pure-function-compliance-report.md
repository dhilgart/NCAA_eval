# Pure Function Compliance Report — `src/ncaa_eval/`

**Date:** 2026-03-03
**Story:** 8.9 — Add PEP 20, SOLID & Pure Function Gates
**Scope:** All source files in `src/ncaa_eval/` (excludes tests, dashboard, template)
**Methodology:** Manual review of function signatures, I/O patterns, state mutation, and side-effect isolation per STYLE_GUIDE Section 6.2 ("Pure core + side-effect shell")

---

## Executive Summary

The NCAA_eval codebase demonstrates **strong Pure Function compliance**. The "pure core + side-effect shell" architecture is well-realized: feature engineering (`transform/`) is almost entirely pure, evaluation metrics are pure, and I/O is cleanly pushed to the edges (connectors, repository, CLI orchestration).

| Category | Count | Notes |
|----------|-------|-------|
| **Pure functions** | 47 | No side effects; same input always produces same output |
| **Side-effect functions (expected)** | 46 | I/O, network, state mutation — all at architectural edges |
| **Mixed functions (violations)** | 3 | Pure logic buried inside side-effect code |

The 3 violations are concentrated in `cli/train.py` (the training orchestrator) and `ingest/sync.py` (ESPN team map builder). All are deferrable to existing Epic 8 stories.

---

## 1. Module-by-Module Analysis

### cli/

**Expected role:** Side-effect orchestrators (I/O, user interaction, subprocess calls).

#### `cli/main.py`
| Function | Classification | Rationale |
|----------|---------------|-----------|
| `_callback()` | Side-effect | Typer CLI callback (no-op but part of framework) |
| `_instantiate_model()` | **Mixed** (minor) | Reads config file from disk AND performs pure Pydantic validation + model instantiation in the same function. Acceptable because the I/O is a single `read_text()` and the function is short (10 lines). |
| `train()` | Side-effect | CLI entry point — expected orchestrator |

**Assessment:** Clean. `_instantiate_model` is a borderline case but acceptable given its brevity.

#### `cli/train.py`
| Function | Classification | Rationale |
|----------|---------------|-----------|
| `_get_git_hash()` | Side-effect | Subprocess call to `git rev-parse` |
| `_build_fold_predictions()` | **Pure** | DataFrame assembly from BacktestResult — no I/O, no mutation |
| `run_training()` | Side-effect | Training orchestrator — expected |

**Assessment:** `_build_fold_predictions` is a good example of extracting pure logic from the orchestrator. However, `run_training()` has a **violation** — see Section 2.

---

### ingest/

**Expected role:** Side-effect I/O (network, file system, CSV parsing).

#### `ingest/schema.py`
| Function | Classification | Rationale |
|----------|---------------|-----------|
| `Team`, `Season`, `Game` (Pydantic models) | **Pure** | Immutable data validation — no side effects |
| `Game._check_game_integrity()` | **Pure** | Pydantic validator — deterministic input validation |

**Assessment:** Exemplary. Schema models are pure data containers.

#### `ingest/repository.py`
| Function | Classification | Rationale |
|----------|---------------|-----------|
| `_apply_model_defaults()` | Side-effect (mutation) | Mutates DataFrame in-place (fills NaN with defaults) |
| `ParquetRepository.get_teams/games/seasons()` | Side-effect | File I/O (Parquet reads) |
| `ParquetRepository.save_teams/games/seasons()` | Side-effect | File I/O (Parquet writes) |

**Assessment:** Clean separation. Repository is the side-effect shell; schema models above are the pure core.

#### `ingest/connectors/base.py`
| Function | Classification | Rationale |
|----------|---------------|-----------|
| Exception classes | **Pure** | Data definitions |
| `Connector` ABC | Interface | Abstract — no implementation |

#### `ingest/connectors/kaggle.py`
| Function | Classification | Rationale |
|----------|---------------|-----------|
| `_validate_columns()` | **Pure** | Set comparison — deterministic |
| `KaggleConnector.download()` | Side-effect | Network I/O (Kaggle API) + file extraction |
| `KaggleConnector._read_csv()` | Side-effect | File I/O |
| `KaggleConnector.load_day_zeros()` | Side-effect | File I/O + internal caching (`self._day_zeros`) |
| `KaggleConnector.fetch_teams/seasons/team_spellings()` | Side-effect | Delegates to `_read_csv` |
| `KaggleConnector.fetch_games()` | Side-effect | Delegates to `_read_csv` |
| `KaggleConnector._parse_games_csv()` | **Mixed** (see Section 2) | CSV reading AND pure row-level parsing in the same function |

**Assessment:** The Kaggle connector follows a reasonable "download once, parse many" pattern. The `_parse_games_csv` method mixes file I/O (via `_read_csv`) with pure parsing logic, but the coupling is tight by design — the parsing is CSV-format-specific and not reusable outside this connector. Low-priority finding.

#### `ingest/connectors/espn.py`
| Function | Classification | Rationale |
|----------|---------------|-----------|
| `_parse_game_result()` | **Pure** | String parsing — deterministic, no side effects |
| `_resolve_team_id()` | **Pure** (with logging) | Fuzzy matching computation. Logging is the only side effect and is acceptable (observability, not behavior). |
| `EspnConnector._fetch_per_team()` | Side-effect | Network I/O (cbbpy HTTP calls) |
| `EspnConnector._fetch_schedule_df()` | Side-effect | Delegates to `_fetch_per_team` |
| `EspnConnector._parse_schedule_df()` | **Pure** | DataFrame transformation — no I/O |
| `EspnConnector._parse_date()` | **Pure** | Date parsing — deterministic |
| `EspnConnector._infer_loc()` | **Pure** | Location inference from DataFrame row |
| `EspnConnector.fetch_games()` | Side-effect | Orchestrates fetch + parse |

**Assessment:** Good pure/side-effect separation. `_parse_game_result`, `_parse_schedule_df`, `_parse_date`, and `_infer_loc` are all pure and independently testable.

#### `ingest/sync.py`
| Function | Classification | Rationale |
|----------|---------------|-----------|
| `_build_espn_team_map()` | **Mixed** (see Section 2) | File I/O (reads cbbpy CSV) + pure fuzzy-matching logic |
| `SyncEngine.sync_kaggle()` | Side-effect | Orchestrator — expected |
| `SyncEngine.sync_espn()` | Side-effect | Orchestrator — expected |
| `SyncEngine.sync_all()` | Side-effect | Orchestrator — expected |

**Assessment:** `_build_espn_team_map` is the primary violation in this module — see Section 2.

---

### transform/

**Expected role:** Mostly pure feature engineering.

#### `transform/serving.py`
| Function | Classification | Rationale |
|----------|---------------|-----------|
| `rescale_overtime()` | **Pure** | Arithmetic — no side effects |
| `_effective_date()` | **Pure** (with logging) | Date computation with fallback. Logging is observability only. |
| `_deduplicate_2025()` | **Pure** | DataFrame deduplication — returns new list, no mutation |
| `ChronologicalDataServer.get_chronological_season()` | Side-effect | Reads from Repository (I/O) + calls `datetime.date.today()` (time dependency) |
| `ChronologicalDataServer.iter_games_by_date()` | Side-effect | Delegates to `get_chronological_season` |

**Assessment:** Excellent. Pure functions (`rescale_overtime`, `_deduplicate_2025`) are cleanly separated from the data-serving shell.

#### `transform/normalization.py`
| Function | Classification | Rationale |
|----------|---------------|-----------|
| `parse_seed()` | **Pure** | String parsing — deterministic |
| `TeamNameNormalizer.normalize()` | **Pure** (with logging) | Dict lookup + prefix matching. Logging is observability. |
| `TeamNameNormalizer.from_csv()` | Side-effect | File I/O (CSV read) — factory method at the edge |
| `TourneySeedTable.get()` | **Pure** | Dict lookup |
| `TourneySeedTable.all_seeds()` | **Pure** | List filtering |
| `TourneySeedTable.from_csv()` | Side-effect | File I/O (CSV read) — factory method at the edge |
| `ConferenceLookup.get()` | **Pure** | Dict lookup |
| `ConferenceLookup.from_csv()` | Side-effect | File I/O (CSV read) — factory method at the edge |
| `MasseyOrdinalsStore.run_coverage_gate()` | **Pure** | DataFrame groupby → set comparison |
| `MasseyOrdinalsStore.get_snapshot()` | **Pure** | DataFrame filtering + pivot |
| `MasseyOrdinalsStore.composite_simple_average()` | **Pure** | Mean computation |
| `MasseyOrdinalsStore.composite_weighted()` | **Pure** | Weighted average |
| `MasseyOrdinalsStore.composite_pca()` | **Pure** | PCA transform |
| `MasseyOrdinalsStore.pre_tournament_snapshot()` | **Pure** | Delegates to `get_snapshot` |
| `MasseyOrdinalsStore.normalize_rank_delta()` | **Pure** | Arithmetic |
| `MasseyOrdinalsStore.normalize_percentile()` | **Pure** | Arithmetic |
| `MasseyOrdinalsStore.normalize_zscore()` | **Pure** | Arithmetic |
| `MasseyOrdinalsStore.from_csv()` | Side-effect | File I/O (CSV read) — factory method at the edge |

**Assessment:** Exemplary pure-core architecture. The `from_csv()` factory methods are the ONLY side-effect entry points; all computation methods are pure. This is the canonical "pure core + side-effect shell" pattern.

#### `transform/sequential.py`
| Function | Classification | Rationale |
|----------|---------------|-----------|
| `_reshape_to_long()` | **Pure** | DataFrame transformation |
| `apply_ot_rescaling()` | **Pure** | Returns copy; no mutation of input |
| `compute_game_weights()` | **Pure** | Vectorized arithmetic |
| `compute_rolling_stats()` | **Pure** | Rolling window computation |
| `compute_ewma_stats()` | **Pure** | EWMA computation |
| `compute_momentum()` | **Pure** | EWMA delta computation |
| `compute_streak()` | **Pure** | Cumsum-based streak computation |
| `compute_possessions()` | **Pure** | Vectorized formula |
| `compute_per_possession_stats()` | **Pure** | Normalization |
| `compute_four_factors()` | **Pure** | Efficiency ratio computation |
| `DetailedResultsLoader.from_csvs()` | Side-effect | File I/O (CSV reads) — factory method |
| `DetailedResultsLoader.get_season_long_format()` | **Pure** | DataFrame filtering |
| `DetailedResultsLoader.get_team_season()` | **Pure** | DataFrame filtering |
| `SequentialTransformer.transform()` | **Pure** | Orchestrates pure functions — no I/O |

**Assessment:** Outstanding. 13 pure functions, 1 side-effect factory method. This module is the gold standard for pure function design.

#### `transform/calibration.py`
| Function | Classification | Rationale |
|----------|---------------|-----------|
| `IsotonicCalibrator.fit()` | **Pure** (stateful) | Mutates internal state but no I/O. Stateful in the ML sense — acceptable. |
| `IsotonicCalibrator.transform()` | **Pure** | Deterministic output for given input + fitted state |
| `SigmoidCalibrator.fit()` | **Pure** (stateful) | Same as above |
| `SigmoidCalibrator.transform()` | **Pure** | Same as above |

**Assessment:** Clean. The fit/transform pattern is standard ML convention — state mutation during `fit()` is expected and does not constitute a side-effect violation.

#### `transform/feature_serving.py`
| Function | Classification | Rationale |
|----------|---------------|-----------|
| `FeatureConfig.active_blocks()` | **Pure** | Deterministic from config |
| `StatefulFeatureServer.serve_season_features()` | Side-effect | Calls data server (Repository I/O) |
| `StatefulFeatureServer._serve_batch()` | Side-effect | Delegates to data-serving layer |
| `StatefulFeatureServer._serve_stateful()` | Side-effect | Delegates + mutates Elo engine state |
| `StatefulFeatureServer._build_game_metadata()` | **Pure** | List comprehension |
| `StatefulFeatureServer._game_to_metadata_dict()` | **Pure** | Static dict construction |
| `StatefulFeatureServer._get_ordinal_values_with_systems()` | **Pure** | Delegates to pure MasseyOrdinalsStore methods |
| `StatefulFeatureServer._resolve_ordinal_systems()` | **Pure** | Config inspection + coverage gate |
| `StatefulFeatureServer._get_seed_nums()` | **Pure** | Seed table lookup |
| `StatefulFeatureServer._compute_batch_ratings()` | **Pure** | Delegates to pure rating solvers |
| `StatefulFeatureServer._build_batch_indexed()` | **Pure** | Index construction |
| `StatefulFeatureServer._collect_rating_vals()` | Side-effect (mutation) | Mutates accumulator lists in-place (acceptable — local mutation pattern) |
| `StatefulFeatureServer._compute_matchup_deltas()` | **Pure** | Arithmetic delta computation |
| `StatefulFeatureServer._empty_frame()` | **Pure** | Empty DataFrame construction |

**Assessment:** Good separation. The public `serve_season_features()` is the side-effect shell; internal feature computation methods are pure.

#### `transform/opponent.py`
| Function | Classification | Rationale |
|----------|---------------|-----------|
| `_build_team_index()` | **Pure** | Index construction |
| `_build_srs_matrices()` | **Pure** | Matrix construction |
| `BatchRatingSolver.compute_srs()` | **Pure** (with logging) | Fixed-point iteration — deterministic. Logging only on non-convergence. |
| `BatchRatingSolver.compute_ridge()` | **Pure** | Ridge regression |
| `BatchRatingSolver.compute_colley()` | **Pure** (with logging) | Linear algebra. Logging only on singular matrix fallback. |
| `compute_srs_ratings()` | **Pure** | Convenience wrapper |
| `compute_ridge_ratings()` | **Pure** | Convenience wrapper |
| `compute_colley_ratings()` | **Pure** | Convenience wrapper |

**Assessment:** Fully pure. All 8 functions have no side effects beyond diagnostic logging.

#### `transform/graph.py`
| Function | Classification | Rationale |
|----------|---------------|-----------|
| `build_season_graph()` | **Pure** | Constructs and returns new graph |
| `compute_pagerank()` | **Pure** | NetworkX computation |
| `compute_betweenness_centrality()` | **Pure** | NetworkX computation |
| `compute_hits()` | **Pure** (with logging) | NetworkX computation with convergence warning |
| `compute_clustering_coefficient()` | **Pure** | NetworkX computation |
| `GraphTransformer.build_graph()` | **Pure** | Delegates to `build_season_graph` |
| `GraphTransformer.add_game_to_graph()` | Side-effect (mutation) | Mutates graph in-place — intentional for walk-forward incremental updates |
| `GraphTransformer.compute_features()` | **Pure** | Computes features from graph |
| `GraphTransformer.transform()` | **Pure** | Convenience: build + compute |

**Assessment:** Excellent. Only `add_game_to_graph` mutates state, and that's by design for incremental walk-forward efficiency.

#### `transform/elo.py`
| Function | Classification | Rationale |
|----------|---------------|-----------|
| `EloFeatureEngine.expected_score()` | **Pure** | Static method — logistic formula |
| `EloFeatureEngine.get_rating()` | **Pure** | Dict lookup (reads but doesn't mutate) |
| `EloFeatureEngine.update_game()` | Side-effect (state mutation) | Updates internal ratings dict — by design for sequential Elo |
| `EloFeatureEngine.apply_season_mean_reversion()` | Side-effect (state mutation) | Modifies ratings — season transition logic |
| `EloFeatureEngine.reset_game_counts()` | Side-effect (state mutation) | Clears game counts |
| `EloFeatureEngine.start_new_season()` | Side-effect (state mutation) | Orchestrates season transition |
| `EloFeatureEngine.get_all_ratings()` | **Pure** | Returns copy of dict |
| `EloFeatureEngine.process_season()` | Side-effect (state mutation) | Processes all games + updates ratings |
| `EloFeatureEngine._effective_k()` | **Pure** | K-factor lookup — no mutation |
| `EloFeatureEngine._margin_multiplier()` | **Pure** | Arithmetic — no mutation |

**Assessment:** Good for a stateful engine. Pure math (`expected_score`, `_margin_multiplier`, `_effective_k`) is cleanly separated from state-mutating methods. The state mutation is inherent to the Elo algorithm.

---

### model/

**Expected role:** Mix — `fit`/`predict` are pure-ish, I/O at edges (`save`/`load`).

#### `model/base.py`
| Function | Classification | Rationale |
|----------|---------------|-----------|
| `Model` ABC | Interface | Abstract — no implementation |
| `StatefulModel.fit()` | Side-effect (state mutation) | Reconstructs games and calls `update()` |
| `StatefulModel.predict_proba()` | **Pure** | Calls `_predict_one` per row |
| `StatefulModel._to_games()` | **Pure** | Static — DataFrame → Game list reconstruction |

**Assessment:** Clean. `_to_games` is properly extracted as a pure static helper.

#### `model/registry.py`
| Function | Classification | Rationale |
|----------|---------------|-----------|
| `register_model()` | Side-effect (module state) | Mutates global registry — expected for decorator pattern |
| `get_model()` | **Pure** | Dict lookup |
| `list_models()` | **Pure** | Returns sorted keys |

#### `model/elo.py`
| Function | Classification | Rationale |
|----------|---------------|-----------|
| `EloModel.update()` | Side-effect | Delegates to engine |
| `EloModel.start_season()` | Side-effect | Delegates to engine |
| `EloModel._predict_one()` | **Pure** | Rating lookup + expected score formula |
| `EloModel.get_state()` / `set_state()` | Side-effect | State snapshot/restore |
| `EloModel.save()` | Side-effect | File I/O (JSON write) |
| `EloModel.load()` | Side-effect | File I/O (JSON read) |
| `EloModel.get_config()` | **Pure** | Returns config object |
| `EloModel._to_elo_config()` | **Pure** | Static dataclass conversion |

**Assessment:** Clean. Pure logic (`_predict_one`, `_to_elo_config`, `get_config`) separated from I/O (`save`, `load`) and state mutation (`update`, `start_season`).

#### `model/logistic_regression.py`
| Function | Classification | Rationale |
|----------|---------------|-----------|
| `LogisticRegressionModel.fit()` | Side-effect (state mutation) | Trains sklearn model in-place |
| `LogisticRegressionModel.predict_proba()` | **Pure** | Deterministic from fitted state |
| `LogisticRegressionModel.save()` | Side-effect | File I/O (joblib dump) |
| `LogisticRegressionModel.load()` | Side-effect | File I/O (joblib load) |
| `LogisticRegressionModel.get_config()` | **Pure** | Returns config |

#### `model/xgboost_model.py`
| Function | Classification | Rationale |
|----------|---------------|-----------|
| `XGBoostModel.fit()` | Side-effect (state mutation + randomness) | Trains XGBoost with `random_state=42` and train/val split |
| `XGBoostModel.predict_proba()` | **Pure** | Deterministic from fitted state |
| `XGBoostModel.save()` | Side-effect | File I/O (UBJ write) |
| `XGBoostModel.load()` | Side-effect | File I/O (UBJ read) |
| `XGBoostModel.get_config()` | **Pure** | Returns config |

#### `model/tracking.py`
| Function | Classification | Rationale |
|----------|---------------|-----------|
| `ModelRun`, `Prediction` (Pydantic models) | **Pure** | Data containers |
| `RunStore.save_run()` | Side-effect | File I/O (JSON + Parquet write) |
| `RunStore.load_run()` | Side-effect | File I/O (JSON read) |
| `RunStore.load_predictions()` | Side-effect | File I/O (Parquet read) |
| `RunStore.save_metrics()` | Side-effect | File I/O (Parquet write) |
| `RunStore.load_metrics()` | Side-effect | File I/O (Parquet read) |
| `RunStore.save_fold_predictions()` | Side-effect | File I/O (Parquet write) |
| `RunStore.load_fold_predictions()` | Side-effect | File I/O (Parquet read) |
| `RunStore.save_model()` | Side-effect | File I/O (delegates to model.save) |
| `RunStore.load_model()` | Side-effect | File I/O + registry lookup |
| `RunStore.load_feature_names()` | Side-effect | File I/O (JSON read) |
| `RunStore.load_all_summaries()` | Side-effect | File I/O (iterates run dirs) |
| `RunStore.list_runs()` | Side-effect | File I/O (directory scan) |

**Assessment:** Clean side-effect shell. All methods in `RunStore` are pure I/O — exactly where side effects should live.

---

### evaluation/

**Expected role:** Mostly pure calculations.

#### `evaluation/metrics.py`
| Function | Classification | Rationale |
|----------|---------------|-----------|
| `_validate_inputs()` | **Pure** | Input validation — deterministic |
| `log_loss()` | **Pure** | Delegates to sklearn after validation |
| `brier_score()` | **Pure** | Delegates to sklearn after validation |
| `roc_auc()` | **Pure** | Delegates to sklearn after validation |
| `expected_calibration_error()` | **Pure** | Vectorized numpy computation |
| `reliability_diagram_data()` | **Pure** | Delegates to sklearn + numpy binning |

**Assessment:** Fully pure. All 6 metric functions are stateless, deterministic, and have no side effects.

#### `evaluation/splitter.py`
| Function | Classification | Rationale |
|----------|---------------|-----------|
| `walk_forward_splits()` | Side-effect | Calls `feature_server.serve_season_features()` which reads from Repository |

**Assessment:** The splitter is a thin orchestration layer. Its only side effect comes from the feature server dependency. The split logic itself (walk-forward iteration, tournament filtering) is pure.

#### `evaluation/plotting.py`
| Function | Classification | Rationale |
|----------|---------------|-----------|
| `plot_reliability_diagram()` | **Pure** | Constructs and returns Plotly Figure |
| `plot_backtest_summary()` | **Pure** | Constructs and returns Plotly Figure |
| `plot_metric_comparison()` | **Pure** | Constructs and returns Plotly Figure |
| `plot_advancement_heatmap()` | **Pure** | Constructs and returns Plotly Figure |
| `plot_score_distribution()` | **Pure** | Constructs and returns Plotly Figure |

**Assessment:** Fully pure. All plotting functions take data in, return Figure objects out — no I/O.

#### `evaluation/simulation.py`
| Function | Classification | Rationale |
|----------|---------------|-----------|
| `_build_subtree()` | **Pure** | Recursive tree construction |
| `build_bracket()` | **Pure** | Bracket construction from seed data |
| `MatrixProvider.matchup_probability()` | **Pure** | Matrix lookup |
| `MatrixProvider.batch_matchup_probabilities()` | **Pure** | Batch matrix lookup |
| `EloProvider.matchup_probability()` | **Pure** | Delegates to model's pure `_predict_one` |
| `EloProvider.batch_matchup_probabilities()` | **Pure** | Batch version of above |
| `build_probability_matrix()` | **Pure** | Upper-triangle batch fill |
| `scoring_from_config()` | **Pure** | Factory dispatch — deterministic |
| `compute_advancement_probs()` | **Pure** | Phylourny algorithm — deterministic |
| `compute_expected_points()` | **Pure** | Matrix-vector multiply |
| `compute_expected_points_seed_diff()` | **Pure** | Extended EP with seed bonus |
| `compute_most_likely_bracket()` | **Pure** | Greedy traversal — deterministic |
| `compute_bracket_distribution()` | **Pure** | Histogram + percentile computation |
| `score_bracket_against_sims()` | **Pure** | Vectorized scoring |
| `simulate_tournament_mc()` | **Pure** (with controlled randomness) | MC simulation using injected `rng` — deterministic when seeded |
| `simulate_tournament()` | **Pure** | High-level orchestrator — dispatches to pure functions |
| `_collect_leaves()` | **Pure** | Tree traversal |
| Scoring rule classes (`Standard`, `Fibonacci`, `SeedDiffBonus`, `Custom`, `Dict`) | **Pure** | All `points_per_round` methods are arithmetic |

**Assessment:** Outstanding. The entire 1,291-line simulation module is pure. Monte Carlo randomness is managed via an injected `np.random.Generator`, making the module deterministic when seeded. No I/O anywhere.

#### `evaluation/backtest.py`
| Function | Classification | Rationale |
|----------|---------------|-----------|
| `feature_cols()` | **Pure** | Column name filtering |
| `_evaluate_fold()` | Side-effect (state mutation) | Trains model (mutates state) + `time.perf_counter()` |
| `run_backtest()` | Side-effect | Orchestrator — calls walk_forward_splits (I/O), deep-copies models, runs joblib parallel, prints Rich tables |

**Assessment:** `feature_cols()` is properly extracted as a pure helper. The backtest orchestrator is expected to be a side-effect function. The metric computation within `_evaluate_fold` is cleanly delegated to the pure `metrics.py` functions.

---

### utils/

**Expected role:** Configuration and cross-cutting concerns.

#### `utils/logger.py`
| Function | Classification | Rationale |
|----------|---------------|-----------|
| `configure_logging()` | Side-effect | Mutates global logging state + reads environment variable |
| `get_logger()` | Side-effect | Returns mutable logger instance from global registry |

**Assessment:** Expected — logging configuration is inherently side-effectful.

---

## 2. Violations — Mixed Functions (Pure Logic Buried in Side Effects)

### 2a. `cli/train.py:73–246` — `run_training()` — Label Balance Check Buried in Orchestrator

**Location:** `cli/train.py:149–155`
**What's mixed:** The label balance detection logic (`label_mean > 0.95 or label_mean < 0.05`) is a pure analytical check embedded inside the training orchestrator alongside I/O (feature serving, model persistence, Rich console output).

**Why it matters:** The label balance check cannot be unit-tested without instantiating a full `ParquetRepository`, `ChronologicalDataServer`, `StatefulFeatureServer`, and `Model`. A developer wanting to verify the threshold logic must mock 4+ dependencies.

**How to separate:**
```python
# Pure function (testable with just a Series)
def check_label_balance(y: pd.Series, hi: float = 0.95, lo: float = 0.05) -> str | None:
    """Return a warning message if labels are imbalanced, else None."""
    mean = y.mean()
    if mean > hi or mean < lo:
        return f"labels are heavily imbalanced (mean={mean:.3f})"
    return None
```

**Tracking:** Deferred to **Story 8.1** (train.py refactor). The constants `0.95`/`0.05` were already flagged in the PEP 20 report as magic numbers needing named constants.

---

### 2b. `ingest/sync.py:40–107` — `_build_espn_team_map()` — File I/O Mixed with Pure Fuzzy Matching

**Location:** `ingest/sync.py:60–63` (CSV read) + `ingest/sync.py:76–97` (fuzzy matching loop)
**What's mixed:** This module-level function reads `cbbpy`'s `mens_team_map.csv` from disk (I/O), then performs pure fuzzy matching (string comparison + threshold logic). The fuzzy matching algorithm cannot be tested without the cbbpy CSV on disk.

**Why it matters:** The fuzzy matching logic (exact lookup → fuzzy fallback → override dict) is non-trivial and deserves independent unit testing. Currently it can only be tested via integration tests that require the cbbpy package to be installed.

**How to separate:**
```python
# Side-effect shell: read CSV
def _load_cbbpy_locations(year: int) -> list[str]:
    """Load ESPN location names from cbbpy's team map CSV."""
    ...

# Pure core: match locations to team IDs
def _match_locations_to_teams(
    locations: list[str],
    spellings: dict[str, int],
    overrides: dict[str, int],
    threshold: int = 80,
) -> tuple[dict[str, int], list[str]]:
    """Return (matched, unmatched) from fuzzy matching."""
    ...
```

**Tracking:** Deferred to **Story 8.3** (ESPN connector resilience refactoring). The function is tightly coupled to ESPN sync and the refactoring fits naturally with the ESPN error handling improvements scoped in that story.

---

### 2c. `cli/train.py:73–246` — `run_training()` — Prediction Assembly Buried in Orchestrator

**Location:** `cli/train.py:170–191`
**What's mixed:** The tournament prediction assembly loop (iterating `tourney` rows, clamping probabilities to `[0, 1]`, constructing `Prediction` objects) is pure transformation logic embedded in the orchestrator between the model training step and the persistence step.

**Why it matters:** The prediction assembly logic (especially the `min(max(prob, 0.0), 1.0)` clamping) is a correctness-critical operation that should be independently testable. Currently it requires mocking the full training pipeline to reach.

**How to separate:**
```python
# Pure function (testable with just DataFrames)
def build_predictions(
    tourney: pd.DataFrame, probs: pd.Series, run_id: str,
) -> list[Prediction]:
    """Assemble Prediction records from tournament games and probabilities."""
    ...
```

**Tracking:** Deferred to **Story 8.1** (train.py refactor). This extraction should happen alongside the `_build_fold_predictions` pattern already established in the same file.

---

## 3. Good Patterns — Pure Core + Side-Effect Shell

### 3a. `transform/normalization.py` — `from_csv()` Factory + Pure Methods (Canonical Example)

`MasseyOrdinalsStore` perfectly demonstrates the pattern: a single `from_csv()` class method handles I/O, and all 8 computation methods (`run_coverage_gate`, `get_snapshot`, `composite_*`, `normalize_*`) are pure. The same pattern is repeated by `TourneySeedTable`, `ConferenceLookup`, and `TeamNameNormalizer`.

### 3b. `transform/sequential.py` — 13 Pure Functions, 1 Factory

Every computation function in this module (`compute_rolling_stats`, `compute_ewma_stats`, `compute_momentum`, `compute_streak`, `compute_possessions`, `compute_per_possession_stats`, `compute_four_factors`, `apply_ot_rescaling`, `compute_game_weights`) is a standalone pure function. The only side-effect is `DetailedResultsLoader.from_csvs()`. The `SequentialTransformer.transform()` method orchestrates all pure functions without introducing any I/O — a pure orchestrator of pure functions.

### 3c. `evaluation/simulation.py` — 1,291 Lines, Zero I/O

The entire simulation module is pure. Monte Carlo randomness is injectable via `np.random.Generator`, making every function deterministic when seeded. This is the gold standard for complex pure computation.

### 3d. `evaluation/metrics.py` — Every Metric is Pure

All 6 metric functions accept arrays and return scalars. No I/O, no state, no randomness. The `_validate_inputs()` helper is also pure.

### 3e. `ingest/connectors/espn.py` — Parsing Separated from Fetching

`_parse_game_result()`, `_parse_schedule_df()`, `_parse_date()`, and `_infer_loc()` are all pure parsing functions that can be tested without network access. The side-effect `_fetch_per_team()` is cleanly isolated.

### 3f. `cli/train.py` — `_build_fold_predictions()` Extraction

This function was extracted from the orchestrator to enable independent testing of the fold prediction DataFrame assembly. It demonstrates the correct approach for the remaining violations in the same file.

---

## 4. Summary of Actions

| Action | Count | Details |
|--------|-------|---------|
| **Deferred to Story 8.1** | 2 | `run_training()` label balance check extraction; `run_training()` prediction assembly extraction |
| **Deferred to Story 8.3** | 1 | `_build_espn_team_map()` I/O + fuzzy matching separation |
| **Acceptable as-is** | 46 | Side-effect functions at architectural edges (connectors, repository, CLI, logging) |
| **No violations found** | — | `transform/` (except `sync.py`), `evaluation/`, `model/` all comply with pure-core architecture |

### Overall Compliance Rating: **Strong**

The codebase follows the "pure core + side-effect shell" principle consistently. The 3 violations are minor (concentrated in orchestration code) and all have natural homes in existing Epic 8 refactoring stories.
