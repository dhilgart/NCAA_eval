# NCAA_eval Codebase Audit Report

**Date:** 2026-03-02
**Version Audited:** 0.9.0
**Branch:** `improvements`
**Methodology:** Multi-agent party-mode analysis — 7 BMAD agents conducted independent deep-dive reviews, followed by cross-pollinated second pass

---

## Executive Summary

The NCAA_eval project has strong fundamentals: comprehensive type safety (mypy --strict), good test coverage, clean separation of concerns in the transform layer, and disciplined use of design patterns (Repository, Plugin Registry, Strategy). All 7 core epics are functionally complete.

However, the audit identified **85+ distinct issues** across architecture, code quality, test coverage, requirements compliance, documentation, and product gaps. These range from critical product-level gaps (no user-editable bracket, no Kaggle submission export) to minor cosmetic inconsistencies.

**Pass 1 Findings Summary:**

| Agent | Issues Found | Critical | Major | Minor | Cosmetic |
|-------|:-----------:|:--------:|:-----:|:-----:|:--------:|
| 🏗️ Architect (Winston) | 28 | 0 | 9 | 17 | 2 |
| 🧪 QA (Quinn) | 38 | 0 | 3 | 14 | 5 |
| 🏃 Scrum Master (Bob) | 10 | 1 | 3 | 5 | 1 |
| 📋 Product Manager (John) | 24 | 4 | 13 | 6 | 1 |
| 📚 Tech Writer (Paige) | 16 | 0 | 5 | 8 | 3 |
| 🧪 Test Architect (Murat) | 12 | 0 | 2 | 8 | 2 |
| 💻 Developer (Amelia) | partial | - | - | - | - |

*Note: Developer agent exhausted context mid-analysis; findings are included where available.*

---

## Category 1: Definitely Requires Human PO Direction

These issues involve product strategy, feature scope decisions, or tradeoffs that cannot be resolved without stakeholder input.

### 1.1 Game Theory Sliders Never Implemented (SM #2)
- **Epic AC says:** "Game Theory sliders in the sidebar (Upset Aggression, Chalk Bias, Seed-Weight) perturb the model's base probabilities in real-time"
- **Reality:** Story 7.5 marked them OUT OF SCOPE. Story 7.7 spike completed research. **No follow-up implementation story was ever created.**
- **Impact:** A documented feature from the AC is completely missing
- **Agents concurring:** SM (Bob), PM (John), Architect (Winston)

### 1.2 No User-Editable Bracket (PM #1.6)
- **Description:** Users cannot click matchups to override model picks. The Pool Scorer only scores the model's most-likely bracket, not the user's picks.
- **Impact:** The core use case — helping users fill out their March Madness pool bracket — is incomplete
- **Files:** `dashboard/pages/2_Presentation.py`, `dashboard/pages/4_Pool_Scorer.py`

### 1.3 No Kaggle Submission Export (PM #6.3)
- **Description:** The product's tagline references Kaggle March Mania, but there is no export producing the required `ID,Pred` format (`2025_1104_1112`)
- **Impact:** Users cannot submit predictions to the Kaggle competition directly
- **File:** `dashboard/lib/filters.py` (export_bracket_csv)

### 1.4 No Model Ensemble/Blending Support (PM #6.2)
- **Description:** No mechanism to blend predictions from multiple models (e.g., 60% XGBoost + 40% Elo)
- **Impact:** Top Kaggle performers universally use ensembles; single-model predictions are strictly suboptimal

### 1.5 No Demo/Sample Data (PM #7.2)
- **Description:** Product requires full Kaggle API setup (account, phone verification, competition rules, API token) before any functionality works
- **Impact:** Very high barrier to first value; potential users abandon before seeing the product

### 1.6 Feature Engineering Pipeline Not Configurable from CLI (Architect #7.2)
- **Description:** `run_training()` hardcodes `FeatureConfig(graph_features_enabled=False, batch_rating_types=(), ...)`. No `--feature-config` CLI option exists.
- **Impact:** Users cannot experiment with different feature combinations without editing source code
- **File:** `src/ncaa_eval/cli/train.py:101-109`

### 1.7 `team_a_won = True` Label Bias (Architect #2.5)
- **Description:** `_game_to_metadata_dict()` always assigns team_a = w_team_id, making training label always 1.0. The pipeline warns if label mean > 0.95 but doesn't fix it.
- **Decision needed:** Should the feature server randomize team_a/team_b assignment?
- **File:** `src/ncaa_eval/transform/feature_serving.py:523`

### 1.8 Fibonacci Scoring Values Mismatch (SM #1)
- **Epic says:** (1-1-2-3-5-8)
- **Code says:** (2-3-5-8-13-21)
- **Decision needed:** Which sequence is canonical?
- **File:** `src/ncaa_eval/evaluation/simulation.py:480`

### 1.9 Metric Explorer Missing 3 of 4 Drill-Down Dimensions (SM #3)
- **Epic AC says:** "drill-down by year, round, seed matchup, or conference"
- **Reality:** Only year is implemented. Round, seed matchup, conference explicitly deferred as "post-MVP"
- **File:** `dashboard/pages/3_Model_Deep_Dive.py:33`

### 1.10 "Candidate Entry" Bracket Flagging Not Implemented (SM #4)
- **Epic AC says:** "the user can flag a specific bracket configuration as a 'Candidate Entry'"
- **Reality:** No such feature exists in the Presentation page

### 1.11 CLI Has No `predict` Command (PM #2.2)
- **Description:** During March Madness, users need predictions for specific matchups NOW without retraining
- **Impact:** Primary use case (tournament predictions) not accessible via CLI

### 1.12 No Per-Game Prediction Explainability (PM #5.2)
- **Description:** "Duke has a 72% chance of beating UNC" but no way to understand contributing factors
- **Impact:** Users cannot make informed decisions about when to trust vs. override the model

### 1.13 `StatefulModel.fit()` Interface Impedance Mismatch (Architect #3.6, #4.5)
- **Description:** Stateful models receive `(X, y)` DataFrames but need `Game` objects, causing wasteful round-trip serialization. Backtest checks `isinstance(model, StatefulModel)` violating LSP.
- **Decision needed:** Should `StatefulModel.fit()` accept `list[Game]` directly?
- **Files:** `src/ncaa_eval/model/base.py:107-169`, `src/ncaa_eval/evaluation/backtest.py:163`

### 1.14 Pool Scorer: CSV Export Only, Not CSV/JSON (SM #6)
- **Epic AC says:** "CSV/JSON" export
- **Reality:** Only CSV via st.download_button

### 1.15 Feature Importance Only Available for XGBoost (PM #5.1)
- **Description:** Elo shows "Feature importance is not available for stateful models" despite Elo being inherently explainable (team ratings). LR also missing despite having `.coef_`.
- **Impact:** Users can't understand model behavior for 2 of 3 model types

---

## Category 2: Might Require Human Judgment

These issues have reasonable arguments on both sides or may be acceptable tradeoffs.

### 2.1 `sync.py` at Project Root vs Inside CLI Package (Architect #1.1)
- **Description:** Root-level `sync.py` creates a parallel CLI entry point outside the package boundary
- **File:** `sync.py` (1-77)
- **Tradeoff:** Convenience of `python sync.py` vs architectural consistency

### 2.2 `serving.py` Imports from `ncaa_eval.ingest` — Tight Coupling (Architect #2.1)
- **Description:** `ChronologicalDataServer` imports `Repository` and `Game` from ingest, breaking the "no ingest imports" invariant documented in other transform modules
- **File:** `src/ncaa_eval/transform/serving.py:19-20`

### 2.3 Repository `get_games` Constructs Game Objects Per Row (Architect #1.4)
- **Description:** `df.to_dict(orient="records")` → `Game(**row)` for every row. Wasteful when downstream consumers immediately convert back to DataFrames.
- **Tradeoff:** Domain integrity vs performance

### 2.4 `KaggleConnector` Uses `iterrows()` (Architect #1.5)
- **Description:** Despite project's "no iterrows()" convention. The ingest layer is not the transform layer — is the convention universal?
- **Files:** `src/ncaa_eval/ingest/connectors/kaggle.py:157,168,202,219`

### 2.5 Connector ABC Has Optional Methods That Raise NotImplementedError (Architect #1.6)
- **Description:** "Header Interface" anti-pattern. Could use separate protocols/mixins instead.
- **File:** `src/ncaa_eval/ingest/connectors/base.py:56-72`

### 2.6 Giant `__init__.py` Re-exports (Architect #2.6)
- **Description:** `transform/__init__.py` re-exports 37 symbols, triggering loading of all submodules (networkx, sklearn, etc.)
- **Tradeoff:** Import convenience vs startup time

### 2.7 EloModelConfig Duplicates EloConfig Fields (Architect #3.2)
- **Description:** Same 9 fields defined in both Pydantic model and frozen dataclass
- **File:** `src/ncaa_eval/model/elo.py:22-38`

### 2.8 Model Registry is a Global Mutable Singleton (Architect #3.4)
- **Description:** Module-level mutable `_MODEL_REGISTRY` dict. Common pattern but makes testing harder.
- **File:** `src/ncaa_eval/model/registry.py:16`

### 2.9 `RunStore.load_model()` Has Deferred Import (Architect #3.5)
- **Description:** Circular dependency avoidance via deferred `from ncaa_eval.model.registry import get_model`
- **File:** `src/ncaa_eval/model/tracking.py:239`

### 2.10 Deferred sklearn Imports in metrics.py (Architect #7.5)
- **Description:** Every call to `log_loss()`, `brier_score()`, etc. does deferred import. Minor overhead, cached by Python.
- **File:** `src/ncaa_eval/evaluation/metrics.py:93,123,154,258`

### 2.11 `EspnConnector._fetch_per_team` Swallows Exceptions at DEBUG Level (Architect #7.7)
- **Description:** `except Exception: logger.debug(...)` — failures invisible at default log level
- **File:** `src/ncaa_eval/ingest/connectors/espn.py:141-143`

### 2.12 `get_data_dir()` Uses `__file__`-Relative Path Navigation (Architect #6.2)
- **Description:** `Path(__file__).resolve().parent.parent.parent / "data"` — fragile if directory structure changes
- **File:** `dashboard/lib/filters.py:56-58`

### 2.13 Dashboard Pages Use Module-Level `_render_*()` Pattern (Architect #6.4)
- **Description:** All page logic runs on import — Streamlit convention but surprising for non-Streamlit imports
- **Files:** `dashboard/pages/1_Lab.py:132`, `dashboard/pages/4_Pool_Scorer.py:248`

### 2.14 Leaderboard Click-to-Navigate Uses Undocumented Streamlit API (PM #1.10)
- **Description:** `event.selection.rows` with `# type: ignore[attr-defined]` — may break on Streamlit updates
- **File:** `dashboard/pages/1_Lab.py:116-129`

### 2.15 Plotly Adapter API Design Changed from AC (SM #10)
- **Description:** Epic says `model.plot_calibration()` (methods), implementation uses standalone functions. Story documents this as deliberate.

### 2.16 st.spinner Instead of st.progress for Simulation (SM #5)
- **Description:** AC specifies `st.progress` bar, implementation uses `st.spinner()` (indeterminate)
- **File:** `dashboard/pages/4_Pool_Scorer.py:92,162`

### 2.17 Story 2.3 Open AI-Review Follow-ups (SM #9)
- **Description:** Pandera schema validation not added to KaggleConnector; iterrows not replaced

### 2.18 Top-Level `__init__.py` Does Not Re-Export Public API (Tech Writer #4.7)
- **Description:** Style Guide says "Public symbols should be importable from the package level" but `from ncaa_eval import EloModel` fails
- **File:** `src/ncaa_eval/__init__.py:1-3`

### 2.19 User Guide Documents Planned Feature (Game Theory Sliders) As If It Exists (PM #3.3)
- **Description:** 50 lines of specification for a non-existent feature, with only a small {note} admonition
- **File:** `docs/user-guide.md:527-575`

### 2.20 No Data Post-Sync Validation (PM #4.4)
- **Description:** No validation step checks game count reasonableness, duplicate games, team reference integrity after sync

### 2.21 _make_season_df Duplicated Across Test Files (QA)
- **Description:** Same helper defined in `test_evaluation_splitter.py` and `test_evaluation_backtest.py` — should be a shared fixture
- **Files:** `tests/unit/test_evaluation_splitter.py:18`, `tests/unit/test_evaluation_backtest.py:28`

---

## Category 3: Flaws So Obvious No Human Insight Needed

These are clearly bugs, code smells, or violations that should be fixed regardless of product direction.

### CODE ARCHITECTURE

#### 3.1 `simulation.py` is a 1,291-Line Mega-Module (Architect #4.1)
- **File:** `src/ncaa_eval/evaluation/simulation.py` (1-1291)
- **Description:** Contains 7+ distinct responsibilities: bracket structures, providers, scoring rules + registry, simulation results, analytical algorithm, Monte Carlo engine, orchestrator
- **Fix:** Split into `bracket.py`, `scoring.py`, `providers.py`, `analytical.py`, `monte_carlo.py`

#### 3.2 `SyncEngine` Directly Couples to `typer.echo` (Architect #1.2, #1.3)
- **File:** `src/ncaa_eval/ingest/sync.py:15,160,162,170,173,180,185,234,251`
- **Description:** Data layer module `import typer` and calls `typer.echo()` for progress. Makes SyncEngine unusable from notebooks, tests, or dashboard.
- **Fix:** Use logging or accept a callback/progress interface

#### 3.3 Hardcoded 2025 Deduplication Logic (Architect #1.7)
- **File:** `src/ncaa_eval/transform/serving.py:185-186`
- **Description:** `if year == 2025: games = _deduplicate_2025(games)` — will silently fail for 2026+ data
- **Fix:** Deduplicate all years checking for ESPN prefix pattern, or make configurable

#### 3.4 Private Attribute Access on EloFeatureEngine (3 violations) (Architect #2.2, #3.1, #3.3)
- **Files:**
  - `src/ncaa_eval/transform/feature_serving.py:301` — accesses `_ratings`
  - `src/ncaa_eval/model/elo.py:78,115-116` — reads `_game_counts`, writes `_ratings` and `_game_counts`
  - `src/ncaa_eval/evaluation/simulation.py:313,325,339` — calls `_predict_one`
- **Fix:** Expose public `has_ratings()`, `set_ratings()`, `set_game_counts()`, `predict_matchup()` methods

#### 3.5 Dashboard `filters.py` is a 621-Line Kitchen Sink (Architect #6.1)
- **File:** `dashboard/lib/filters.py` (1-621)
- **Description:** Contains data loading, simulation orchestration, probability provider construction, scoring, CSV export, and win probability calculation
- **Fix:** Split into `data_loaders.py`, `simulation_helpers.py`, `export.py`

#### 3.6 `run_training` God Function (Architect #5.2)
- **File:** `src/ncaa_eval/cli/train.py:73`
- **Description:** Suppresses 3 complexity lints (`PLR0913`, `C901`, `PLR0912`). Handles feature building, training, prediction, persistence, backtesting, metric saving, fold saving, and model saving in one function.
- **Fix:** Decompose into smaller functions

#### 3.7 Duplicated `DEFAULT_MARGIN_CAP` Constant (Architect #2.3)
- **Files:** `src/ncaa_eval/transform/graph.py:34`, `src/ncaa_eval/transform/opponent.py:14`
- **Description:** Both define `DEFAULT_MARGIN_CAP = 25` independently
- **Fix:** Centralize in a shared constants module

#### 3.8 Duplicated Fuzzy Match Logic (Architect #7.4)
- **Files:** `src/ncaa_eval/ingest/sync.py:86-94`, `src/ncaa_eval/ingest/connectors/espn.py:73-85`
- **Description:** Both implement `rapidfuzz.fuzz.token_set_ratio` with threshold 80
- **Fix:** Centralize in a shared utility

#### 3.9 `FeatureConfig` Uses Stringly-Typed Fields (Architect #2.4)
- **File:** `src/ncaa_eval/transform/feature_serving.py:82-87`
- **Description:** `batch_rating_types`, `ordinal_composite`, `gender_scope`, `calibration_method` are plain `str` with magic values
- **Fix:** Use `Literal` types or enums

#### 3.10 Scoring Registry Uses Untyped `dict[str, type]` (Architect #4.2)
- **File:** `src/ncaa_eval/evaluation/simulation.py:407`
- **Description:** `_SCORING_REGISTRY: dict[str, type] = {}` loses type safety
- **Fix:** Use `dict[str, type[ScoringRule]]`

#### 3.11 `splitter.py` Imports Private `_NO_TOURNAMENT_SEASONS` (Architect #4.3)
- **File:** `src/ncaa_eval/evaluation/splitter.py:19`
- **Description:** Importing a private constant from another module is fragile
- **Fix:** Make the constant public or centralize

#### 3.12 `backtest.py` Silently Swallows Exceptions (Architect #4.4)
- **File:** `src/ncaa_eval/evaluation/backtest.py:183`
- **Description:** `except Exception: metrics[name] = float("nan")` hides bugs in metric implementations
- **Fix:** At minimum log the exception

#### 3.13 Dashboard Accesses Private `_clf` Attribute (Architect #6.3)
- **File:** `dashboard/lib/filters.py:207-208`
- **Description:** `getattr(model, "_clf", None)` reaches into model internals for feature importances
- **Fix:** Model ABC should provide public `get_feature_importances()` method

#### 3.14 `Ellipsis` Used as Sentinel for PydanticUndefined (Architect #7.3)
- **File:** `src/ncaa_eval/ingest/repository.py:102`
- **Description:** `sentinel: Any = ...` — fragile if Pydantic internals change
- **Fix:** Use `pydantic.fields.PydanticUndefined` directly

#### 3.15 No Abstract Base for Calibrators (Architect #7.6)
- **File:** `src/ncaa_eval/transform/calibration.py:34,104`
- **Description:** `IsotonicCalibrator` and `SigmoidCalibrator` share identical interfaces but no common base
- **Fix:** Create a `Calibrator` Protocol or ABC

### DOCUMENTATION

#### 3.16 NumPy-Style Docstrings (5 modules violate Google-style mandate) (Tech Writer #4.1-4.5)
- **Files:**
  - `src/ncaa_eval/evaluation/metrics.py` — entire module
  - `src/ncaa_eval/transform/elo.py` — class docstring, `update_game`, `process_season`
  - `src/ncaa_eval/model/elo.py` — `set_state`, `load`
  - `src/ncaa_eval/model/base.py` — `_to_games`
  - `src/ncaa_eval/model/tracking.py` — `load_run`, `load_predictions` (mixed with Google-style!)
- **Fix:** Mechanical find-and-replace: `Parameters ----------` → `Args:`, etc.

#### 3.17 Getting Started Tutorial Shows Inaccurate Expected Output (PM #3.1)
- **File:** `docs/tutorials/getting-started.md:19-67`
- **Description:** Expected CLI output doesn't match actual sync.py / training output
- **Fix:** Run the commands and capture actual output

#### 3.18 No Troubleshooting Section Anywhere (PM #3.2)
- **Description:** No guidance for common failures: Kaggle auth, ESPN rate limits, missing conda env, Parquet version mismatches
- **Fix:** Add troubleshooting section to user guide or getting-started tutorial

#### 3.19 Missing License Section in README (Tech Writer #1.1)
- **File:** `README.md`
- **Fix:** Add "## License" section referencing GPL-3.0

#### 3.20 Missing `Returns` Section in `_resolve_team_id` Docstring (Tech Writer #4.6)
- **File:** `src/ncaa_eval/ingest/connectors/espn.py:54-85`

### TESTING

#### 3.21 `scoring_from_config` Completely Untested (QA)
- **File:** `src/ncaa_eval/evaluation/simulation.py:596-640`
- **Description:** Exported factory function with 5 dispatch branches — zero test coverage

#### 3.22 CLI Training Tests Only Cover Logistic Regression (QA)
- **File:** `tests/unit/test_cli_train.py`
- **Description:** XGBoost and Elo model types never exercised through the training pipeline

#### 3.23 Dead Code in Test Conftest (QA)
- **File:** `tests/conftest.py`
- **Description:** `sample_game_records` fixture defined but never used

#### 3.24 Empty Test File (QA)
- **File:** `tests/test_ncaa_eval.py`
- **Description:** Contains only `from __future__ import annotations`

#### 3.25 Tests Access Private `._P` Attribute (QA #12.1)
- **File:** `tests/unit/test_dashboard_filters.py:517-518,565`
- **Description:** Tests assert on `result._P[0, 1]` instead of using public `matchup_probability()`

### SPRINT HOUSEKEEPING

#### 3.26 Epic Statuses Not Updated to "done" (SM #7)
- **File:** `_bmad-output/implementation-artifacts/sprint-status.yaml`
- **Description:** All stories in Epics 1-7 are "done" but all epic statuses remain "in-progress"
- **Fix:** Flip each epic from "in-progress" to "done"

#### 3.27 Story 6.6 Dev Agent Record Incomplete (SM #8)
- **File:** `_bmad-output/implementation-artifacts/6-6-implement-tournament-scoring-user-defined-point-schedules.md:239`
- **Description:** `{{agent_model_name_version}}` template placeholder unfilled

### DATA PIPELINE

#### 3.28 ESPN Connector Silently Swallows All Per-Team Exceptions (PM #4.1)
- **File:** `src/ncaa_eval/ingest/connectors/espn.py:137-143`
- **Description:** `except Exception: logger.debug(...)` at DEBUG level. Users get partial data with zero indication.
- **Fix:** Log at WARNING level, include team name and exception; report failure count

#### 3.29 No Retry Logic for Network Operations (PM #4.3)
- **Files:** `src/ncaa_eval/ingest/connectors/kaggle.py`, `src/ncaa_eval/ingest/connectors/espn.py`
- **Description:** Neither connector uses retry logic despite hundreds of HTTP requests (ESPN) and the project's Library-First Rule (tenacity)

#### 3.30 No Data Freshness Indicators (PM #4.2)
- **Description:** Dashboard shows no timestamp of last sync, no latest game date, no staleness warning
- **Impact:** Users may make bracket decisions based on stale data

### DASHBOARD UX

#### 3.31 Bracket Font Size Too Small (PM #1.7)
- **File:** `dashboard/lib/bracket_renderer.py:79-84`
- **Description:** 10px team names, 9px probability labels in 700px frame — likely unreadable
- **Fix:** Increase font sizes or add zoom/scroll

#### 3.32 No Dashboard Setup/First-Run Validation (PM #7.3)
- **Description:** New users land on a mostly blank dashboard with a small sidebar info message
- **Fix:** Add prominent "Setup needed" indicator or setup wizard

#### 3.33 No Manual Cache Refresh Button (PM #1.8)
- **File:** `dashboard/lib/filters.py`
- **Description:** 5-minute TTL cache with no "Refresh" button. Users who train a model see stale data.

#### 3.34 Inconsistent Breadcrumb Navigation (PM #1.9)
- **Files:** `dashboard/pages/1_Lab.py`, `dashboard/pages/3_Model_Deep_Dive.py`
- **Description:** Some pages have breadcrumbs, others don't

---

## Metrics Summary

### By Severity (deduplicated)

| Severity | Count |
|----------|:-----:|
| Critical | 4 |
| Major | 25 |
| Minor | 30 |
| Cosmetic | 6 |

### By Category

| Category | Count |
|----------|:-----:|
| 1. Requires PO Direction | 15 |
| 2. Might Require Human Judgment | 21 |
| 3. Obviously Needs Fixing | 34 |

---

## Agent Sign-offs

- 🏗️ **Winston (Architect):** "The architecture is generally clean. The main structural debt is concentrated in simulation.py, the ingest-CLI coupling, and private attribute access patterns around EloFeatureEngine."
- 🧪 **Quinn (QA):** "Test suite is well-designed and comprehensive. Most impactful gaps: scoring_from_config untested, CLI only tests logistic_regression."
- 🏃 **Bob (SM):** "9 AC discrepancies found. Game Theory sliders are the biggest gap — spike done, no implementation story created."
- 📋 **John (PM):** "3 critical product gaps: no user-editable bracket, no Kaggle submission export, ESPN silent data loss. Top strategic recommendation: build the user-editable bracket."
- 📚 **Paige (Tech Writer):** "Project is well-documented overall. Single biggest finding: 5 modules use NumPy-style docstrings instead of Google-style mandate."
- 🧪 **Murat (TEA):** "Test pyramid is healthy. Mutation testing targets evaluation/ingest. Property-based testing (Hypothesis) used for Elo. Main gap: no E2E integration tests through the full pipeline."
- 💻 **Amelia (Dev):** "Code quality is strong. Key code smells: God functions in train.py, stringly-typed configs, duplicate constants."
