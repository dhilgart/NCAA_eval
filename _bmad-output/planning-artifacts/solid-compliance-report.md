# SOLID Compliance Report — `src/ncaa_eval/`

**Date:** 2026-03-03
**Scope:** All source files in `src/ncaa_eval/` (excludes tests, dashboard, template)
**Methodology:** Manual review of class hierarchies, module responsibilities, interface design, dependency patterns, and import graph across 37 source files (8,264 lines)

---

## Executive Summary

The NCAA_eval codebase demonstrates **strong SOLID compliance** overall. The architecture was designed with SOLID principles in mind: the Model ABC / plugin registry, Repository abstraction, Connector hierarchy, and Protocol-based interfaces in the simulation engine are all textbook applications of these principles. The primary areas of concern are (1) the simulation module's size (SRP), (2) cross-module private attribute access (LSP/encapsulation), and (3) the sync engine's hard-coded dependency on concrete connectors (DIP).

| Principle | Violations | Compliant Patterns | Notes |
|-----------|-----------|-------------------|-------|
| **S — Single Responsibility** | 2 | 35 modules | `simulation.py` (1,290 lines) and `cli/train.py` multi-concern orchestration |
| **O — Open/Closed** | 1 | Strong | `scoring_from_config()` uses if/elif dispatch instead of registry |
| **L — Liskov Substitution** | 0 | Strong | All Model subclasses honor the ABC contract |
| **I — Interface Segregation** | 0 | Strong | Lean ABCs and Protocols throughout |
| **D — Dependency Inversion** | 2 | Moderate | `sync.py` and `cli/train.py` depend on concrete implementations |

---

## 1. Single Responsibility Principle (SRP)

### Violations

#### 1a. `evaluation/simulation.py` — 1,290 Lines, 6+ Distinct Responsibilities

- **File:** `src/ncaa_eval/evaluation/simulation.py`
- **Lines:** 1,290 (project convention: ~300 max per module)
- **Responsibilities identified:**
  1. **Bracket data structures** — `BracketNode`, `BracketStructure`, `MatchupContext`, `build_bracket()`, `_build_subtree()` (lines 70–210)
  2. **Probability provider protocol + implementations** — `ProbabilityProvider`, `MatrixProvider`, `EloProvider`, `build_probability_matrix()` (lines 218–372)
  3. **Scoring rule protocol + registry + implementations** — `ScoringRule`, `StandardScoring`, `FibonacciScoring`, `SeedDiffBonusScoring`, `CustomScoring`, `DictScoring`, `scoring_from_config()`, `register_scoring()`, `get_scoring()`, `list_scorings()` (lines 380–640)
  4. **Result data structures** — `SimulationResult`, `BracketDistribution`, `MostLikelyBracket` (lines 648–728)
  5. **Analytical computation (Phylourny)** — `compute_advancement_probs()`, `compute_expected_points()`, `compute_expected_points_seed_diff()`, `compute_most_likely_bracket()`, `compute_bracket_distribution()`, `score_bracket_against_sims()` (lines 735–1034)
  6. **Monte Carlo engine** — `simulate_tournament_mc()`, `_collect_leaves()` (lines 1042–1210)
  7. **High-level orchestrator** — `simulate_tournament()` (lines 1218–1290)
- **Impact:** High — this module is the most complex in the codebase. The interleaving of data structures, protocols, implementations, and algorithms makes it difficult to modify one concern without reading the entire file.
- **Tracking:** Story 8.1 (code-architecture-cleanup-simulation-module-split) — backlog

#### 1b. `cli/train.py` — Training Orchestration Mixes Multiple Concerns

- **File:** `src/ncaa_eval/cli/train.py`
- **Lines:** 246
- **Issue:** `run_training()` (lines 73–246) handles feature serving setup, model training, prediction generation, run persistence, backtest execution, metric persistence, fold prediction persistence, model artifact persistence, and Rich console output — all in a single function.
- **Impact:** Medium — the function is linear (no deep nesting), but its breadth means any change to the training pipeline requires modifying this one function.
- **Tracking:** Story 8.1 (refactoring scope includes `run_training()` decomposition)

### Compliant Patterns

The remaining 35 modules demonstrate excellent SRP adherence:

- **`model/base.py`** (198 lines) — Model ABC and StatefulModel template only
- **`model/registry.py`** (57 lines) — Plugin registration only
- **`model/elo.py`** (187 lines), **`model/xgboost_model.py`** (176 lines), **`model/logistic_regression.py`** (56 lines) — Each model is a single file with a single responsibility: ABC conformance for one model type
- **`model/tracking.py`** (291 lines) — Run metadata and persistence only
- **`ingest/schema.py`** (60 lines) — Pydantic data models only
- **`ingest/connectors/base.py`** (72 lines) — ABC and exception hierarchy only
- **`ingest/connectors/kaggle.py`** (250 lines), **`ingest/connectors/espn.py`** (268 lines) — Each connector handles exactly one data source
- **`ingest/repository.py`** (206 lines) — Repository ABC + Parquet implementation
- **`evaluation/metrics.py`** (286 lines) — Pure metric functions only
- **`evaluation/splitter.py`** (109 lines) — Walk-forward splitting only
- **`evaluation/backtest.py`** (320 lines) — Cross-validation orchestration only
- **`evaluation/plotting.py`** (341 lines) — Plotly visualization adapters only
- **`transform/` modules** — Each module handles exactly one feature transformation family (sequential, graph, elo, opponent, calibration, normalization, serving, feature_serving)
- **`utils/logger.py`** (136 lines) — Logging configuration only

---

## 2. Open/Closed Principle (OCP)

### Violations

#### 2a. `evaluation/simulation.py:596–640` — `scoring_from_config()` Uses If/Elif Dispatch

- **File:** `src/ncaa_eval/evaluation/simulation.py:596–640`
- **Issue:** `scoring_from_config()` dispatches on `config["type"]` using a chain of `if`/`elif` statements:
  ```python
  if scoring_type == "standard": return StandardScoring()
  if scoring_type == "fibonacci": return FibonacciScoring()
  if scoring_type == "seed_diff_bonus": ...
  if scoring_type == "dict": ...
  if scoring_type == "custom": ...
  ```
  Adding a new scoring type requires modifying this function.
- **Impact:** Low — the project already has `register_scoring()` / `get_scoring()` (the decorator-based registry pattern). `scoring_from_config()` exists as a convenience for config-dict instantiation and could delegate to the registry instead of reimplementing dispatch. The violation is contained to this one function.
- **Tracking:** Story 8.1 (simulation module split) — when the scoring module is extracted, `scoring_from_config()` should delegate to the registry

### Compliant Patterns

The codebase demonstrates **excellent OCP compliance** in its core extension points:

- **Model plugin registry** (`model/registry.py`) — `@register_model("name")` decorator pattern. Adding a new model type (e.g., LightGBM) requires zero changes to existing code — just create a new file with the decorator. Verified: `EloModel`, `XGBoostModel`, and `LogisticRegressionModel` are all registered this way.
- **Scoring rule registry** (`evaluation/simulation.py`) — `@register_scoring("name")` pattern mirrors the model registry. `StandardScoring`, `FibonacciScoring`, and `SeedDiffBonusScoring` are registered via decorator.
- **`ProbabilityProvider` Protocol** — Any class implementing `matchup_probability()` and `batch_matchup_probabilities()` can be used without modifying the simulation engine. `MatrixProvider` and `EloProvider` demonstrate this.
- **`ScoringRule` Protocol** — Any class with `name` and `points_per_round()` satisfies the protocol. `CustomScoring` and `DictScoring` show extensibility without modification.
- **`Connector` ABC hierarchy** — Adding a new data source (e.g., BartTorvik) means adding a new `Connector` subclass — no changes to existing connectors.
- **`Repository` ABC** — `ParquetRepository` is the sole implementation today, but the abstraction is ready for a future SQLite backend (noted in Story 5.5).
- **Batch rating solvers** (`transform/opponent.py`) — `BatchRatingSolver` encapsulates SRS, Ridge, and Colley solvers. Adding a new solver method (e.g., Massey) is a single new method addition, with the module-level convenience function pattern providing backward compatibility.
- **Feature block enum** (`transform/feature_serving.py`) — `FeatureBlock` enum + `FeatureConfig.active_blocks()` allows toggling feature families via configuration rather than code modification.

---

## 3. Liskov Substitution Principle (LSP)

### Violations

None found. All subclass hierarchies honor their parent contracts.

### Compliant Patterns

#### 3a. Model Hierarchy — Full Contract Compliance

The `Model` ABC defines 5 abstract methods (`fit`, `predict_proba`, `save`, `load`, `get_config`). All three concrete implementations honor the contract:

| Subclass | `fit` | `predict_proba` | `save` | `load` | `get_config` |
|----------|-------|-----------------|--------|--------|-------------|
| `EloModel` (via `StatefulModel`) | Template method → `update()` per game | Template method → `_predict_one()` per row | JSON directory | JSON directory | `EloModelConfig` |
| `XGBoostModel` | XGBClassifier.fit with early stopping | XGBClassifier.predict_proba[:,1] | UBJSON + config JSON | UBJSON + config JSON | `XGBoostModelConfig` |
| `LogisticRegressionModel` | sklearn LR.fit | sklearn LR.predict_proba[:,1] | joblib + config JSON | joblib + config JSON | `LogisticRegressionConfig` |

- **`StatefulModel`** is a well-implemented Template Method pattern: concrete `fit()` and `predict_proba()` delegate to abstract hooks (`update()`, `_predict_one()`, `start_season()`, `get_state()`, `set_state()`). `EloModel` implements all hooks correctly.
- **Return type covariance** is properly applied: `EloModel.get_config()` returns `EloModelConfig` (a subtype of `ModelConfig`), which is valid LSP behavior.

#### 3b. Connector Hierarchy — Optional Capability Pattern

`Connector` defines `fetch_games()` as abstract and `fetch_teams()` / `fetch_seasons()` as optional capabilities with `NotImplementedError` defaults. This is a deliberate design choice documented in the docstring:

- `KaggleConnector` implements all three methods
- `EspnConnector` implements only `fetch_games()`, inheriting the default `NotImplementedError` for the others

This pattern is LSP-compliant because the base class explicitly documents the optionality contract and recommends `isinstance` or `try/except` probing.

#### 3c. Calibrator Classes — Consistent Interface

`IsotonicCalibrator` and `SigmoidCalibrator` both implement `fit(y_true, y_prob)` and `transform(y_prob)` with identical signatures and semantics. They are not formally linked by an ABC or Protocol, but they follow a consistent duck-typing pattern.

#### 3d. ProbabilityProvider Protocol — Structural Subtyping

`MatrixProvider` and `EloProvider` both satisfy the `ProbabilityProvider` Protocol's two-method contract (`matchup_probability`, `batch_matchup_probabilities`). The `@runtime_checkable` decorator enables isinstance verification.

---

## 4. Interface Segregation Principle (ISP)

### Violations

None found. All interfaces are lean and focused.

### Compliant Patterns

#### 4a. ABCs — Minimal Method Counts

| ABC / Protocol | Method Count | Assessment |
|---------------|-------------|------------|
| `Model` | 5 methods | Minimum viable: train, predict, save, load, config |
| `StatefulModel` | 5 additional hooks | Template Method pattern; each hook has a single purpose |
| `Repository` | 6 methods | 3 reads + 3 writes — symmetric and cohesive |
| `Connector` | 1 required + 2 optional | `fetch_games()` is the universal capability; teams/seasons are optional |
| `ProbabilityProvider` | 2 methods | Single + batch — both needed for simulation |
| `ScoringRule` | 1 property + 1 method | `name` + `points_per_round()` — minimal |

No ABC requires implementers to provide methods they don't need. The `Connector` ABC's optional capabilities pattern (`fetch_teams`/`fetch_seasons` raise `NotImplementedError` by default) is an explicit ISP design — `EspnConnector` only needs `fetch_games()` and doesn't need to stub out team/season methods.

#### 4b. Protocols vs ABCs — Correct Separation

The project correctly uses **ABCs for inheritance hierarchies** (Model, Repository, Connector) and **Protocols for structural typing** (ProbabilityProvider, ScoringRule). This is documented in the Style Guide (Section 10) and consistently applied:

- ABCs enforce implementation completeness at class definition time
- Protocols allow third-party or ad-hoc implementations without inheritance (e.g., `MatrixProvider` satisfies `ProbabilityProvider` without inheriting from it)

#### 4c. FeatureConfig — Declarative Configuration

`FeatureConfig` (frozen dataclass) provides a declarative interface for feature block selection without forcing consumers to understand every block's implementation. The `active_blocks()` method encapsulates the logic of which blocks are enabled, keeping the `StatefulFeatureServer` focused on orchestration.

---

## 5. Dependency Inversion Principle (DIP)

### Violations

#### 5a. `ingest/sync.py` — Hard-Coded Concrete Connector Dependencies

- **File:** `src/ncaa_eval/ingest/sync.py:18–19`
- **Issue:** `SyncEngine` imports and directly instantiates concrete connectors:
  ```python
  from ncaa_eval.ingest.connectors.espn import EspnConnector
  from ncaa_eval.ingest.connectors.kaggle import KaggleConnector
  ```
  The `sync_kaggle()` method (line 151) creates `KaggleConnector(extract_dir=...)` directly. The `sync_espn()` method (line 240) creates `EspnConnector(...)` directly. `SyncEngine.__init__()` accepts a `Repository` abstraction (good DIP), but the connector layer is not injected.
- **Impact:** Medium — adding a new data source requires modifying `SyncEngine`. However, this is somewhat mitigated by the fact that the sync engine is inherently source-specific (Kaggle-specific caching logic, ESPN-specific marker files).
- **Tracking:** Story 8.3 (ESPN connector resilience) may partially address this

#### 5b. `cli/train.py` — Direct Dependency on Concrete Data Stack

- **File:** `src/ncaa_eval/cli/train.py:21–25`
- **Issue:** `run_training()` directly instantiates the concrete data stack:
  ```python
  from ncaa_eval.ingest import ParquetRepository
  ...
  repo = ParquetRepository(base_path=data_dir)
  data_server = ChronologicalDataServer(repo)
  server = StatefulFeatureServer(config=feature_config, data_server=data_server)
  ```
  The function depends on `ParquetRepository` (concrete) instead of `Repository` (abstract). Additionally, `FeatureConfig` is hard-coded with default parameters rather than being injected.
- **Impact:** Low-Medium — the CLI is inherently a composition root (it's where abstractions are bound to implementations), so some concrete dependencies are expected. However, the `run_training()` function also serves as the entry point for test code, and hard-coding `ParquetRepository` limits testability.
- **Tracking:** Story 8.1 (decomposing `run_training()`) could introduce constructor injection for the data stack

### Compliant Patterns

#### 5a. Repository Abstraction — Textbook DIP

`ChronologicalDataServer` depends on the `Repository` ABC:
```python
class ChronologicalDataServer:
    def __init__(self, repository: Repository) -> None:
        self._repo = repository
```
All downstream consumers (`StatefulFeatureServer`, `SyncEngine`) work through this abstraction. Swapping `ParquetRepository` for a future SQLite implementation requires no changes to these consumers.

#### 5b. Model Plugin Registry — Indirection via Registration

The training CLI resolves model types through the registry (`get_model(name)`) rather than importing concrete model classes. This allows new models to be added without changing the CLI.

#### 5c. Metric Functions — Dependency on Abstractions

`backtest.py` defines `DEFAULT_METRICS` as a mapping of `str → Callable[[NDArray, NDArray], float]`. The `_evaluate_fold()` function accepts arbitrary metric functions via the `metric_fns` parameter. No concrete metric is hard-coded — all four defaults (log_loss, brier_score, roc_auc, ece) are injected through the mapping.

#### 5d. Feature Server — Injected Dependencies

`StatefulFeatureServer.__init__()` accepts all its dependencies via constructor injection:
- `config: FeatureConfig` — configuration
- `data_server: ChronologicalDataServer` — data access
- `seed_table: TourneySeedTable | None` — optional seed data
- `ordinals_store: MasseyOrdinalsStore | None` — optional ordinals
- `elo_engine: EloFeatureEngine | None` — optional Elo engine

This clean injection pattern allows full control over dependencies in tests and different deployment contexts.

---

## 6. Cross-Cutting Observations

### 6a. Private Attribute Access Across Module Boundaries

Three instances of cross-module private attribute access were identified (tracked under Story 8.2):

| Accessor | Target Private Attribute | File:Line |
|----------|------------------------|-----------|
| `EloModel.get_state()` | `self._engine._game_counts` | `model/elo.py:78` |
| `EloModel.set_state()` | `self._engine._ratings`, `self._engine._game_counts` | `model/elo.py:115–116` |
| `StatefulFeatureServer._serve_stateful()` | `self._elo_engine._ratings` | `transform/feature_serving.py:301` |
| `EloProvider` | `self._model._predict_one()` | `evaluation/simulation.py:325, 339` |

These are not strictly SOLID violations (they don't violate any single SOLID principle in isolation) but they represent **encapsulation violations** that increase coupling between modules. The `EloModel` comment at line 112–114 explicitly acknowledges this: "EloFeatureEngine has no public setter — direct attribute assignment is intentional here."

**Tracking:** Story 8.2 (expose-public-apis-eliminate-private-attribute-access) — backlog

### 6b. isinstance Checks for Stateful/Stateless Dispatch

Two locations use `isinstance(model, StatefulModel)` to dispatch between stateful and stateless model handling:

- `cli/train.py:111` — Training pipeline
- `evaluation/backtest.py:166` — Fold evaluation

This is an acceptable pattern because `StatefulModel` is a public ABC in the Model hierarchy (not a concrete class), and the dispatch controls which columns to pass to `fit()`/`predict_proba()` (full DataFrame vs. feature-only columns). An alternative pattern (e.g., a method on Model that returns required columns) could eliminate these checks, but the current approach is clear and well-documented.

---

## 7. Summary of Actions

| Finding | Principle | Severity | Status | Tracking |
|---------|-----------|----------|--------|----------|
| `simulation.py` 1,290 lines / 6+ responsibilities | SRP | High | Deferred | Story 8.1 |
| `cli/train.py` `run_training()` multi-concern | SRP | Medium | Deferred | Story 8.1 |
| `scoring_from_config()` if/elif dispatch | OCP | Low | Deferred | Story 8.1 |
| `sync.py` hard-coded concrete connectors | DIP | Medium | Deferred | Story 8.3 |
| `cli/train.py` direct `ParquetRepository` instantiation | DIP | Low | Acceptable | CLI is a composition root |
| Cross-module private attribute access (4 instances) | Encapsulation | Medium | Deferred | Story 8.2 |
| All Model subclasses honor ABC contract | LSP | N/A | Compliant | — |
| All ABCs have lean, focused interfaces | ISP | N/A | Compliant | — |
| Model registry / Scoring registry (OCP) | OCP | N/A | Compliant | — |
| Repository abstraction (DIP) | DIP | N/A | Compliant | — |
| Protocol-based structural typing | ISP/OCP | N/A | Compliant | — |

**Total findings:** 4 violations (all deferred to existing Epic 8 stories), 2 acceptable patterns, 0 fixes required in this review.
