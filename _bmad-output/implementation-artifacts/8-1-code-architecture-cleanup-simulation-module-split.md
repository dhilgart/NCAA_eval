# Story 8.1: Code Architecture Cleanup — Simulation Module Split & Kitchen Sink Refactors

Status: ready-for-dev

## Story

As a developer,
I want `simulation.py` (1,290 lines, 7+ responsibilities), `dashboard/lib/filters.py` (621 lines, 4 responsibilities), and the `run_training()` God Function decomposed into focused, single-responsibility modules,
so that the codebase is maintainable, testable, and aligned with SRP and the project's PEP 20 / SOLID quality gates.

## Acceptance Criteria

### AC1: Split `simulation.py` into Focused Modules

1. `src/ncaa_eval/evaluation/bracket.py` contains: `MatchupContext`, `BracketNode`, `BracketStructure`, `build_bracket`, `_build_subtree`, `N_ROUNDS`, `N_GAMES` constants
2. `src/ncaa_eval/evaluation/scoring.py` contains: `ScoringRule` protocol, `StandardScoring`, `FibonacciScoring`, `SeedDiffBonusScoring`, `CustomScoring`, `DictScoring`, `ScoringNotFoundError`, `register_scoring`, `get_scoring`, `list_scorings`, `scoring_from_config`, `_SCORING_REGISTRY`
3. `src/ncaa_eval/evaluation/providers.py` contains: `ProbabilityProvider` protocol, `MatrixProvider`, `EloProvider`, `build_probability_matrix`
4. `src/ncaa_eval/evaluation/simulation.py` is reduced to: `SimulationResult`, `BracketDistribution`, `MostLikelyBracket` dataclasses, `compute_advancement_probs`, `compute_expected_points`, `compute_expected_points_seed_diff`, `compute_most_likely_bracket`, `compute_bracket_distribution`, `score_bracket_against_sims`, `simulate_tournament_mc`, `simulate_tournament` orchestrator, `_collect_leaves`
5. All symbols currently exported from `ncaa_eval.evaluation.simulation` remain importable from that path (backward compatibility via re-exports in `simulation.py` or `evaluation/__init__.py`)
6. The `evaluation/__init__.py` `__all__` list is updated to import from the correct new submodules

### AC2: Split `dashboard/lib/filters.py` into Focused Modules

7. `dashboard/lib/data_loaders.py` contains: `get_data_dir`, `load_available_years`, `load_available_runs`, `load_leaderboard_data`, `load_fold_predictions`, `load_feature_importances`, `load_available_scorings`, `load_tourney_seeds`, `_load_team_names_uncached`, `load_team_names` (all with existing `@st.cache_data` decorators)
8. `dashboard/lib/simulation_helpers.py` contains: `BracketSimulationResult`, `_build_provider_from_folds`, `_build_team_labels`, `run_bracket_simulation`
9. `dashboard/lib/export.py` contains: `export_bracket_csv`, `_game_win_probability`
10. `dashboard/lib/filters.py` retains only: `score_chosen_bracket`, `build_custom_scoring`, `_ROUND_OF_64_DAY_NUM`, `_ROUND_LABELS` constants
11. All dashboard pages continue to work — imports updated in: `app.py`, `pages/home.py`, `pages/1_Lab.py`, `pages/2_Presentation.py`, `pages/3_Model_Deep_Dive.py`, `pages/4_Pool_Scorer.py`

### AC3: Decompose `run_training()` God Function

12. `run_training()` in `src/ncaa_eval/cli/train.py` decomposed into 4-6 smaller functions (see Tasks for suggested decomposition)
13. All three `noqa` suppressions (`PLR0913`, `C901`, `PLR0912`) removed — the refactored functions must pass Ruff without suppressions
14. The orchestrator function (`run_training`) reduced to ~40-60 lines calling the extracted helper functions

### AC4: Quality Gates

15. All existing tests pass without modification (or with import path updates only)
16. `ruff check .` passes (no new violations)
17. `mypy --strict src/ncaa_eval tests` passes
18. No behavioral changes — pure refactoring (same inputs produce same outputs)

## Tasks / Subtasks

- [ ] Task 1: Split `simulation.py` — Extract `bracket.py` (AC: #1, #5, #6)
  - [ ] 1.1 Create `src/ncaa_eval/evaluation/bracket.py` with `MatchupContext`, `BracketNode`, `BracketStructure`, `build_bracket`, `_build_subtree`, `N_ROUNDS`, `N_GAMES`
  - [ ] 1.2 Add `from __future__ import annotations` and required imports (`TourneySeed`, `dataclasses`, `typing`)
  - [ ] 1.3 Remove extracted code from `simulation.py`, add imports from `bracket`
  - [ ] 1.4 Verify `ruff check .` and `mypy --strict` pass

- [ ] Task 2: Split `simulation.py` — Extract `scoring.py` (AC: #2, #5, #6)
  - [ ] 2.1 Create `src/ncaa_eval/evaluation/scoring.py` with `ScoringRule` protocol, all 5 concrete scoring classes, registry functions, `scoring_from_config`, `ScoringNotFoundError`, `_SCORING_REGISTRY`
  - [ ] 2.2 Remove extracted code from `simulation.py`, add imports from `scoring`
  - [ ] 2.3 Verify `ruff check .` and `mypy --strict` pass

- [ ] Task 3: Split `simulation.py` — Extract `providers.py` (AC: #3, #5, #6)
  - [ ] 3.1 Create `src/ncaa_eval/evaluation/providers.py` with `ProbabilityProvider` protocol, `MatrixProvider`, `EloProvider`, `build_probability_matrix`
  - [ ] 3.2 Remove extracted code from `simulation.py`, add imports from `providers`
  - [ ] 3.3 Verify `ruff check .` and `mypy --strict` pass

- [ ] Task 4: Clean up `simulation.py` and ensure backward compatibility (AC: #4, #5, #6)
  - [ ] 4.1 Verify `simulation.py` retains only: result dataclasses, analytical functions, MC engine, orchestrator
  - [ ] 4.2 Add re-exports in `simulation.py` for all symbols that were previously importable from `ncaa_eval.evaluation.simulation` — use explicit imports from new submodules
  - [ ] 4.3 Update `evaluation/__init__.py` to import from new submodules (bracket, scoring, providers) while keeping the same `__all__` list
  - [ ] 4.4 Verify all 10 files that import from `ncaa_eval.evaluation.simulation` still work (see Dev Notes — Import Consumers)
  - [ ] 4.5 Run full test suite — `pytest`

- [ ] Task 5: Split `dashboard/lib/filters.py` (AC: #7-11)
  - [ ] 5.1 Create `dashboard/lib/data_loaders.py` — move all `load_*` functions and `get_data_dir`
  - [ ] 5.2 Create `dashboard/lib/simulation_helpers.py` — move `BracketSimulationResult`, `_build_provider_from_folds`, `_build_team_labels`, `run_bracket_simulation`
  - [ ] 5.3 Create `dashboard/lib/export.py` — move `export_bracket_csv`, `_game_win_probability`
  - [ ] 5.4 Update `filters.py` to retain only scoring orchestration functions + constants
  - [ ] 5.5 Update imports in all 6 dashboard consumer files (see Dev Notes — Dashboard Import Consumers)
  - [ ] 5.6 Update imports in test files: `tests/unit/test_dashboard_app.py` and `tests/unit/test_dashboard_filters.py`
  - [ ] 5.7 Run `pytest tests/unit/test_dashboard_app.py tests/unit/test_dashboard_filters.py` to verify

- [ ] Task 6: Decompose `run_training()` God Function (AC: #12-14)
  - [ ] 6.1 Extract `_setup_feature_server()` (lines 101-109): Initialize repo, data server, feature config, feature server
  - [ ] 6.2 Extract `_build_season_features()` (lines 113-141): Loop seasons with progress, build feature frames
  - [ ] 6.3 Extract `_prepare_training_data()` (lines 143-167): Combine frames, extract labels, check imbalance, compute feat_cols
  - [ ] 6.4 Extract `_generate_tournament_predictions()` (lines 169-191): Filter tourney games, predict probs, build Prediction objects
  - [ ] 6.5 Extract `_run_backtest_and_persist()` (lines 206-228): Walk-forward backtest, save metrics/folds
  - [ ] 6.6 Extract `_persist_artifacts_and_summarize()` (lines 230-244): Save model, print summary table
  - [ ] 6.7 Reduce `run_training()` to orchestrator calling the 6 helpers; remove all 3 `noqa` suppressions
  - [ ] 6.8 Verify `ruff check src/ncaa_eval/cli/train.py` passes without any noqa
  - [ ] 6.9 Run `pytest tests/` to verify no regressions

- [ ] Task 7: Final validation (AC: #15-18)
  - [ ] 7.1 `ruff check .` — zero violations
  - [ ] 7.2 `mypy --strict src/ncaa_eval tests` — zero errors
  - [ ] 7.3 `pytest` — all tests pass
  - [ ] 7.4 Verify no behavioral changes (pure refactoring)

## Dev Notes

### Key Files to Modify

| File | Changes |
|------|---------|
| `src/ncaa_eval/evaluation/simulation.py` | Extract bracket, scoring, providers into new modules; retain result dataclasses + analytical + MC + orchestrator; add re-exports for backward compat |
| `src/ncaa_eval/evaluation/__init__.py` | Update imports to source from new submodules |
| `src/ncaa_eval/evaluation/plotting.py` | Update `N_ROUNDS`, `BracketDistribution`, `SimulationResult` import to new submodule (or keep importing from simulation.py re-exports) |
| `src/ncaa_eval/cli/train.py` | Decompose `run_training()` into 6 helper functions |
| `dashboard/lib/filters.py` | Strip to scoring orchestration only |
| `dashboard/app.py` | Update data_loader imports |
| `dashboard/pages/home.py` | Update data_loader imports |
| `dashboard/pages/1_Lab.py` | Update data_loader imports |
| `dashboard/pages/2_Presentation.py` | Update simulation_helpers imports |
| `dashboard/pages/3_Model_Deep_Dive.py` | Update data_loader imports |
| `dashboard/pages/4_Pool_Scorer.py` | Update simulation_helpers + export imports |
| `tests/unit/test_evaluation_simulation.py` | Import path updates only (or keep via re-exports) |
| `tests/unit/test_evaluation_plotting.py` | Import path updates if needed |
| `tests/unit/test_dashboard_app.py` | Import path updates for dashboard.lib split |
| `tests/unit/test_dashboard_filters.py` | Import path updates for dashboard.lib split |

### Key Files to Create

| File | Purpose |
|------|---------|
| `src/ncaa_eval/evaluation/bracket.py` | Bracket data structures and construction |
| `src/ncaa_eval/evaluation/scoring.py` | Scoring protocols, implementations, registry |
| `src/ncaa_eval/evaluation/providers.py` | Probability provider protocols and implementations |
| `dashboard/lib/data_loaders.py` | All `@st.cache_data` data loading functions |
| `dashboard/lib/simulation_helpers.py` | Bracket simulation orchestration for dashboard |
| `dashboard/lib/export.py` | CSV export and win-probability formatting |

### Architecture Patterns and Constraints

- **`from __future__ import annotations`** required in ALL new Python files (Ruff-enforced)
- **Google-style docstrings** — not NumPy-style (5 modules were just fixed in Story 8.9)
- **`mypy --strict`** mandatory for `src/ncaa_eval/` and `tests/` — dashboard files are NOT under mypy strict
- **No behavioral changes** — this is a pure structural refactoring. Same inputs, same outputs, same test assertions
- **Backward compatibility** — `from ncaa_eval.evaluation.simulation import X` MUST keep working for all X. Use re-exports in `simulation.py` or route through `evaluation/__init__.py`
- **Ruff complexity limits**: McCabe ≤10, returns ≤6, branches ≤12, args ≤5. The extracted helper functions must satisfy these without `noqa` suppressions

### Simulation Module Split — Dependency Graph

```
bracket.py          → TourneySeed (from transform.normalization)
providers.py        → numpy, bracket.BracketStructure
scoring.py          → logging, typing (zero project imports)
simulation.py       → bracket, scoring, providers, numpy, tqdm (lazy)
```

No circular dependencies. Each new module has a clean, one-directional dependency graph.

### Import Consumers — `ncaa_eval.evaluation.simulation`

Files importing from `ncaa_eval.evaluation.simulation` (10 total):

| File | Symbols Imported | Strategy |
|------|------------------|----------|
| `evaluation/__init__.py` | All 31 public symbols | Update to import from new submodules |
| `evaluation/plotting.py` | `N_ROUNDS`, `BracketDistribution`, `SimulationResult` | Import from `bracket` (N_ROUNDS) and `simulation` (results) |
| `dashboard/lib/filters.py` | 16 symbols | Keep importing from `ncaa_eval.evaluation.simulation` (re-exports) |
| `dashboard/pages/4_Pool_Scorer.py` | `BracketDistribution`, `get_scoring` | Keep importing from `ncaa_eval.evaluation.simulation` (re-exports) |
| `tests/unit/test_evaluation_simulation.py` | All 32 symbols (incl. `_SCORING_REGISTRY`) | Keep importing from `ncaa_eval.evaluation.simulation` (re-exports) |
| `tests/unit/test_evaluation_plotting.py` | 3 symbols | Keep importing from `ncaa_eval.evaluation.simulation` (re-exports) |
| `docs/tutorials/custom-metric.md` | Code examples | Documentation — update code examples to show new import paths as alternatives |

**Strategy**: Update `evaluation/__init__.py` and `evaluation/plotting.py` to import from new submodules directly. All other consumers keep importing from `ncaa_eval.evaluation.simulation` via re-exports — zero external breakage.

### Dashboard Import Consumers — `dashboard.lib.filters`

| File | Current Imports | New Import Source |
|------|-----------------|-------------------|
| `dashboard/app.py` | `get_data_dir`, `load_available_runs`, `load_available_scorings`, `load_available_years` | `dashboard.lib.data_loaders` |
| `dashboard/pages/home.py` | `get_data_dir`, `load_available_runs`, `load_available_years` | `dashboard.lib.data_loaders` |
| `dashboard/pages/1_Lab.py` | `get_data_dir`, `load_available_runs`, `load_leaderboard_data` | `dashboard.lib.data_loaders` |
| `dashboard/pages/2_Presentation.py` | `BracketSimulationResult`, `get_data_dir`, `load_tourney_seeds`, `run_bracket_simulation` | `data_loaders` (get_data_dir, load_tourney_seeds) + `simulation_helpers` (BracketSimulationResult, run_bracket_simulation) |
| `dashboard/pages/3_Model_Deep_Dive.py` | `get_data_dir`, `load_available_runs`, `load_feature_importances`, `load_fold_predictions`, `load_leaderboard_data` | `dashboard.lib.data_loaders` |
| `dashboard/pages/4_Pool_Scorer.py` | `BracketSimulationResult`, `build_custom_scoring`, `export_bracket_csv`, `get_data_dir`, `load_tourney_seeds`, `run_bracket_simulation`, `score_chosen_bracket` | `data_loaders` + `simulation_helpers` + `export` + `filters` |

### `run_training()` Decomposition Guide

Current function signature (line 73):
```python
def run_training(  # noqa: PLR0913, C901, PLR0912
    model: Model,
    *,
    start_year: int,
    end_year: int,
    data_dir: Path,
    output_dir: Path,
    model_name: str,
    console: Console | None = None,
) -> ModelRun:
```

Suggested extraction — each helper is a **module-private** function (prefixed `_`):

1. **`_setup_feature_server(data_dir, model)`** → returns `(StatefulFeatureServer, bool)` — initializes repo, data server, feature config, feature server, detects stateful
2. **`_build_season_features(server, start_year, end_year, is_stateful, console)`** → returns `list[pd.DataFrame]` — loops seasons, builds features per year with progress bar
3. **`_prepare_training_data(season_frames, server)`** → returns `(pd.DataFrame, pd.Series, list[str])` — concatenates, extracts y, computes feat_cols
4. **`_generate_tournament_predictions(combined, model, feat_cols, model_name, run_id)`** → returns `list[Prediction]` — filters tourney games, predicts, builds Prediction records
5. **`_run_backtest_and_persist(model, start_year, end_year, data_dir, output_dir, run_store, model_run, console)`** → returns `BacktestResult | None` — walk-forward if ≥2 seasons
6. **`_persist_artifacts_and_summarize(model, output_dir, model_run, run_store, backtest_result, console)`** → saves model and prints summary

The `PLR0913` concern: the orchestrator can take the same 8 args but pass subsets to helpers. If Ruff still flags it, consider bundling `data_dir` + `output_dir` + `console` into a config dataclass.

### Testing Strategy

- **No new tests required** — this is pure refactoring
- **All existing tests must pass as-is** (or with import path updates only)
- Test file `tests/unit/test_evaluation_simulation.py` comprehensively tests all simulation symbols — verify it passes unchanged (via re-exports)
- Test file `tests/unit/test_dashboard_filters.py` has 35+ test functions covering all dashboard filter functions — update import paths to match new module locations
- Run `pytest -x` after each task to catch regressions early

### Previous Story Learnings (Story 8.9)

- **Ruff scope**: `ruff check .` from project root includes EDA notebooks which are excluded from enforcement. If Ruff reports notebook violations, ignore them — they are not in scope
- **`# noqa` annotations**: When removing the 3 suppressions from `run_training()`, ensure the refactored code actually satisfies the constraints (McCabe ≤10, branches ≤12, args ≤5). Do NOT just remove `noqa` and hope — verify with `ruff check src/ncaa_eval/cli/train.py`
- **Pre-commit hooks**: `debug-statements`, `check-yaml`, `ruff`, `ruff-format` all run. The `template/` directory is excluded from pre-commit (Jinja2 syntax)

### Source Document References

- [Source: `_bmad-output/planning-artifacts/codebase-audit-report.md` — Finding 3.1 (simulation.py mega-module)]
- [Source: `_bmad-output/planning-artifacts/codebase-audit-report.md` — Finding 3.5 (filters.py kitchen sink)]
- [Source: `_bmad-output/planning-artifacts/codebase-audit-report.md` — Finding 3.6 (run_training God function)]
- [Source: `_bmad-output/planning-artifacts/epic-8-codebase-improvements.md` — Story 8.1 section]
- [Source: `_bmad-output/planning-artifacts/template-requirements.md` — Story 8.9 learnings (Ruff scope, noqa annotations)]

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
