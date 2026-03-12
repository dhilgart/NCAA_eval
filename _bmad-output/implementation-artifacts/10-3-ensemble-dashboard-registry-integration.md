# Story 10.3: Dashboard and Model Registry Integration

Status: done

## Story

As a **data scientist**,
I want to **see ensemble models in the dashboard leaderboard and inspect their components**,
so that **I can compare ensemble performance against single models and understand which base model the meta-learner is relying on**.

## Acceptance Criteria

1. **Leaderboard visibility** — Given a trained `StackedEnsemble` artifact in the output directory, when the user opens the Model Leaderboard dashboard page, then the ensemble appears as a single entry with its `model_name` (just like Elo/XGBoost runs today).

2. **Ensemble components section** — Given an ensemble entry in the leaderboard, when the user navigates to the Model Deep Dive page, then an expandable "Ensemble Components" section shows each base model's name and its OOF log loss (from the manifest).

3. **Feature importance with interpretable labels** — Given the ensemble is selected in Model Deep Dive, if the meta-learner supports `get_feature_importances()`, then the importance chart shows `[pred_base_0, pred_base_1, ..., seed_diff, ...]` with interpretable labels (e.g., "XGBoost Prediction" instead of "pred_base_0").

4. **Bracket visualizer integration** — Given the ensemble is selected in the dashboard, when the user navigates to the Bracket Visualizer page, then `ensemble.predict_bracket(data_dir, season)` is called to generate the probability matrix, and the rest of the bracket visualizer works identically to single-model mode.

5. **Model registry CLI compatibility** — `StackedEnsemble` is registered in the model registry under the name provided to `run_training()` so the CLI `predict` and `export` commands work on ensemble run IDs.

## Tasks / Subtasks

- [x] Task 1: Fix `simulation_helpers.py` to handle ensemble model type (AC: #4)
  - [x] 1.1: In `run_bracket_simulation()`, add an `elif run.model_type == "ensemble"` branch that loads the ensemble via `StackedEnsemble.load()` and creates an `EnsembleProvider`
  - [x] 1.2: In `run_bracket_simulation_with_progress()`, add the same ensemble provider branch for the MC simulation unperturbed matrix re-load
  - [x] 1.3: Update the provider type annotation from `EloProvider | MatrixProvider` to include the ensemble path

- [x] Task 2: Fix `data_loaders.py` `load_feature_importances()` for ensembles (AC: #3)
  - [x] 2.1: When `store.load_model()` returns a `StackedEnsemble`, call `ensemble.meta_learner.get_feature_importances()` instead of `model.get_feature_importances()`
  - [x] 2.2: Map raw column names to interpretable labels: `pred_base_N` -> base model type name from manifest, contextual features keep their names
  - [x] 2.3: Read manifest to get `base_model_types` for label mapping

- [x] Task 3: Add "Ensemble Components" section to Model Deep Dive (AC: #2)
  - [x] 3.1: In `_render_deep_dive()`, detect `model_type == "ensemble"` and render an expandable "Ensemble Components" section
  - [x] 3.2: Load the manifest from the run's model directory to get `base_model_types` and `contextual_features`
  - [x] 3.3: Load OOF run summaries if available (the manifest contains `oof_backtest_run_ids` or similar) to show per-base-model OOF log loss
  - [x] 3.4: Display a table: Base Model Name | OOF Log Loss

- [x] Task 4: Verify leaderboard displays ensembles correctly (AC: #1)
  - [x] 4.1: Verify `load_leaderboard_data()` already works for ensemble runs (it reads `model_type` from `ModelRun.model_type` which is set to `"ensemble"` by `_run_ensemble_training`)
  - [x] 4.2: If any filtering/display logic excludes unknown model types, fix it

- [x] Task 5: Verify CLI predict/export works for ensemble run IDs (AC: #5)
  - [x] 5.1: Verify `cli/predict.py` `build_predictions()` already handles ensembles (Story 10.2 replaced the `NotImplementedError`)
  - [x] 5.2: Verify `cli/export.py` handles ensembles or add a clear error message if not supported

- [x] Task 6: Tests (AC: all)
  - [x] 6.1: Unit test: `run_bracket_simulation` with ensemble model type returns valid `BracketSimulationResult`
  - [x] 6.2: Unit test: `load_feature_importances` for ensemble returns interpretable labels
  - [x] 6.3: Unit test: Model Deep Dive renders ensemble components section when `model_type == "ensemble"`
  - [x] 6.4: Unit test: Leaderboard displays ensemble runs alongside single-model runs

- [x] Task 7: Quality gates (AC: all)
  - [x] 7.1: `ruff check .` clean
  - [x] 7.2: `mypy --strict src/ncaa_eval tests` clean
  - [x] 7.3: Full `pytest` suite passes (existing + new tests)
  - [x] 7.4: No regressions in single-model dashboard paths

## Dev Notes

### Critical Context: What Already Works

`StackedEnsemble` is already registered in the model registry via `_EnsembleSentinel` (a sentinel class). `RunStore.load_model()` already dispatches `model_type == "ensemble"` to `StackedEnsemble.load()`. The CLI predict command already handles ensembles (Story 10.2). The `ModelRun` saved by `_run_ensemble_training` has `model_type="ensemble"`.

**What does NOT work yet**: The dashboard's `simulation_helpers.py` only handles `"elo"` and falls through to a fold-predictions-based `MatrixProvider` for all other model types. Ensembles don't have fold predictions in the standard format — they use `predict_bracket()` to generate the probability matrix directly. This is the primary integration gap.

### Dashboard Ensemble Provider Gap (Task 1 — Most Critical)

`simulation_helpers.py:run_bracket_simulation()` has this logic (lines 177-184):

```python
provider: EloProvider | MatrixProvider
if run.model_type == "elo":
    provider = EloProvider(model)
else:
    mp = _build_provider_from_folds(store, run_id, season, bracket)
    if mp is None:
        return None
    provider = mp
```

For ensembles, `_build_provider_from_folds()` will return `None` (no standard fold predictions) and the bracket page will show "Could not simulate bracket." **Fix**: Add an `elif run.model_type == "ensemble"` branch that uses the existing `EnsembleProvider` from `evaluation/providers.py`:

```python
from ncaa_eval.evaluation.providers import EnsembleProvider as _EnsembleProvider
from ncaa_eval.model.ensemble import StackedEnsemble

if run.model_type == "elo":
    provider = EloProvider(model)
elif run.model_type == "ensemble":
    assert isinstance(model, StackedEnsemble)  # type narrowing
    ensemble_provider = _EnsembleProvider(model, path, season)
    provider = ensemble_provider
else:
    mp = _build_provider_from_folds(store, run_id, season, bracket)
    ...
```

**Same fix needed** in `run_bracket_simulation_with_progress()` (lines 332-339) for the MC simulation unperturbed matrix re-load.

**Type annotation update**: The `provider` type needs to include the ensemble path. Since `EnsembleProvider` satisfies the `ProbabilityProvider` protocol, and `build_probability_matrix` accepts `ProbabilityProvider`, the type annotation change is straightforward.

### Feature Importance Interpretable Labels (Task 2)

`load_feature_importances()` in `data_loaders.py` calls `model.get_feature_importances()`. For a `StackedEnsemble`, this will fail because `StackedEnsemble` has no `get_feature_importances()` method — it's not a `Model` subclass.

**Fix**: Check if the loaded model is a `StackedEnsemble`. If so:
1. Call `ensemble.meta_learner.get_feature_importances()` to get raw importance tuples
2. Load the manifest from `store.model_dir(run_id) / "manifest.json"` to get `base_model_types`
3. Map `pred_base_N` to `"{base_model_types[N]} Prediction"` (e.g., "XGBoost Prediction", "Elo Prediction")
4. Keep contextual feature names as-is (e.g., "seed_diff", "is_tournament")

### Ensemble Components Section (Task 3)

In `3_Model_Deep_Dive.py:_render_deep_dive()`, add a section after the feature importance section that:
1. Detects `model_type == "ensemble"`
2. Loads the manifest from `store.model_dir(run_id) / "manifest.json"`
3. Renders an `st.expander("Ensemble Components")` with:
   - Base model types from manifest
   - OOF log loss per base model (if available in manifest — check `oof_backtest_run_ids`)
   - Meta-learner type
   - Contextual features list

The manifest schema (from Story 10.1) contains:
```json
{
  "base_model_types": ["xgboost", "elo"],
  "base_model_count": 2,
  "contextual_features": ["seed_diff", "is_tournament", "loc_encoding"],
  "meta_learner_type": "logistic_regression",
  "meta_column_order": ["pred_base_0", "pred_base_1", "seed_diff", "is_tournament", "loc_encoding"]
}
```

Note: The manifest MAY also contain `"oof_backtest_run_ids"` and `"oof_game_count"` / `"oof_drop_pct"` from `_run_ensemble_training`. Check the actual `cli/train.py:_run_ensemble_training()` to confirm what fields are persisted.

### Leaderboard Already Works (Task 4 — Verification Only)

`load_leaderboard_data()` merges `store.list_runs()` metadata with `store.load_all_summaries()` on `run_id`. For ensemble runs:
- `ModelRun.model_type` is `"ensemble"` — this flows through correctly
- The summary parquet contains the same year × metric format as single models
- The leaderboard display columns are `["run_id", "model_type", "year"] + metric_cols` — ensembles will appear as `model_type="ensemble"`

This should work without changes. Verify by checking that nothing in the leaderboard logic filters or excludes unknown model types.

### CLI Predict/Export Already Works (Task 5 — Verification Only)

- **predict**: `cli/predict.py:build_predictions()` was fixed in Story 10.2 — the `NotImplementedError` guard for ensembles was replaced with `_build_ensemble_predictions()` which calls `ensemble.predict_bracket()`
- **export**: `cli/export.py:build_kaggle_submission()` uses the predict path. Verify it handles ensemble run IDs or returns a clear error

### Existing Code to Reuse (DO NOT Reinvent)

| Need | Existing Solution | Location |
|------|-------------------|----------|
| Ensemble probability provider | `EnsembleProvider` class | `evaluation/providers.py` |
| Build probability matrix | `build_probability_matrix()` | `evaluation/providers.py` |
| Ensemble model loading | `StackedEnsemble.load()` via `RunStore.load_model()` | `model/tracking.py:261` |
| Bracket simulation | `simulate_tournament()` | `evaluation/simulation.py` |
| Feature importances from meta-learner | `ensemble.meta_learner.get_feature_importances()` | via `Model` ABC |
| Manifest loading | `json.loads((model_dir / "manifest.json").read_text())` | `model/ensemble.py:642` |
| Team labels | `_build_team_labels()` | `dashboard/lib/simulation_helpers.py` |
| Seed loading | `TourneySeedTable.from_csv()` | `transform/normalization.py` |

### Anti-Patterns to Avoid

- **DO NOT** create a new `ProbabilityProvider` implementation for ensembles — `EnsembleProvider` already exists
- **DO NOT** modify `StackedEnsemble` to be a `Model` subclass — it deliberately avoids the ABC
- **DO NOT** add `get_feature_importances()` to `StackedEnsemble` — route through `meta_learner` instead
- **DO NOT** duplicate the manifest loading logic — load the JSON once and pass data through
- **DO NOT** use `iterrows` for any DataFrame operations
- **DO NOT** break existing Elo/XGBoost/LogisticRegression dashboard paths

### Dashboard File Structure — NO New Files

All changes go in existing dashboard files:

| File | Action | Description |
|------|--------|-------------|
| `dashboard/lib/simulation_helpers.py` | MODIFY | Add ensemble branch to `run_bracket_simulation` and `run_bracket_simulation_with_progress` |
| `dashboard/lib/data_loaders.py` | MODIFY | Handle `StackedEnsemble` in `load_feature_importances`; add manifest loader helper |
| `dashboard/pages/3_Model_Deep_Dive.py` | MODIFY | Add "Ensemble Components" expander section |
| `tests/unit/test_leaderboard_page.py` | MODIFY | Add ensemble row in sample data, verify display |
| `tests/unit/test_dashboard_app.py` or new test file | MODIFY/CREATE | Tests for ensemble simulation path, feature importance labels, deep dive components |

### Testing Strategy

**Unit tests** (mock RunStore and models):
- `test_bracket_simulation_ensemble_model_type`: Mock `store.load_model()` to return a `StackedEnsemble` stub, verify `run_bracket_simulation` returns a valid `BracketSimulationResult`
- `test_feature_importances_ensemble_interpretable_labels`: Mock model loading to return a `StackedEnsemble` with known manifest, verify labels are human-readable
- `test_deep_dive_renders_ensemble_components`: Mock Streamlit and verify `st.expander("Ensemble Components")` is called when `model_type == "ensemble"`
- `test_leaderboard_includes_ensemble_rows`: Add ensemble rows to sample data, verify they appear in output

**Integration verification** (not automated tests — manual):
- Train an ensemble, verify it appears in leaderboard
- Select ensemble, verify bracket visualizer works
- Verify Model Deep Dive shows components

### Story 10.2 Learnings to Apply

- `caplog` tests need logger propagation re-enabled: `logging.getLogger("ncaa_eval").propagate = True`
- Use `type: ignore[attr-defined]` for `model.feature_names_` access on dynamically-set attributes
- Keep helper functions extracted to stay under C901 complexity threshold
- Dashboard tests use `patch.object(_module, "st", mock_st)` pattern for mocking Streamlit
- `EnsembleProvider` lazily calls `predict_bracket()` on first use — no upfront computation

### Project Structure Notes

- `from __future__ import annotations` required in all modified files (already present)
- `mypy --strict` applies to all files in `src/` and `tests/`
- Google-style docstrings required for public methods
- Dashboard files in `dashboard/` are NOT type-checked by mypy (not in `src/ncaa_eval` or `tests/`)
- Dashboard tests use `importlib.import_module("dashboard.pages.X")` pattern to load page modules

### References

- [Source: specs/ensemble-architecture.md#5.1] — Dashboard integration design spec
- [Source: _bmad-output/planning-artifacts/epics.md] — Epic 10, Story 10.3 acceptance criteria
- [Source: _bmad-output/implementation-artifacts/10-2-ensemble-inference-interface.md] — Previous story dev notes, EnsembleProvider implementation
- [Source: dashboard/lib/simulation_helpers.py] — Bracket simulation orchestration (primary modification target)
- [Source: dashboard/lib/data_loaders.py:168-204] — Feature importance loading (needs ensemble handling)
- [Source: dashboard/pages/3_Model_Deep_Dive.py] — Model Deep Dive page (needs ensemble components section)
- [Source: src/ncaa_eval/evaluation/providers.py:170-229] — EnsembleProvider (reuse, do not reinvent)
- [Source: src/ncaa_eval/model/tracking.py:242-264] — RunStore.load_model ensemble dispatch
- [Source: src/ncaa_eval/model/ensemble.py:639-666] — StackedEnsemble.load() and manifest schema

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

No blocking issues encountered.

### Completion Notes List

- **Task 1**: Added `elif run.model_type == "ensemble"` branch to both `run_bracket_simulation()` and `run_bracket_simulation_with_progress()` in `simulation_helpers.py`. Imports `EnsembleProvider` from `evaluation/providers.py` and `StackedEnsemble` from `model/ensemble.py`. Updated provider type annotation to `EloProvider | MatrixProvider | _EnsembleProvider`.
- **Task 2**: Modified `load_feature_importances()` to detect `StackedEnsemble` via `isinstance` check and route through `meta_learner.get_feature_importances()`. Added `_map_ensemble_feature_labels()` helper that reads `manifest.json` to map `pred_base_N` column names to human-readable labels (e.g., "Xgboost Prediction"). Added `load_ensemble_manifest()` cached function for the deep dive page.
- **Task 3**: Added `_render_ensemble_components()` function to Model Deep Dive page. Renders an `st.expander("Ensemble Components")` showing meta-learner type, contextual features, and a base model table. Called from `_render_deep_dive()` only when `model_type == "ensemble"`. Added "Meta-Learner Feature Importance" chart title for ensemble model type.
- **Task 4**: Verified — leaderboard displays ensembles without changes. `load_leaderboard_data()` includes `model_type` from `ModelRun.model_type` and no filtering excludes unknown types.
- **Task 5**: Verified — `cli/predict.py` handles ensembles via `_build_ensemble_predictions()` (Story 10.2). `cli/export.py` raises `TypeError` for non-Elo models with a clear message.
- **Task 6**: Added 11 new test methods across 3 files covering bracket simulation ensemble path, feature importance label mapping, ensemble manifest loading, deep dive ensemble components section, and leaderboard ensemble row display.
- **Task 7**: All quality gates pass — `ruff check .` clean, `mypy --strict` clean, 1180 tests pass, 0 regressions.

### Code Review Fixes — Round 2 (Claude Sonnet 4.6, 2026-03-12)

**M1 Fixed**: `_load_oof_log_losses` in `3_Model_Deep_Dive.py` used `__import__("pathlib").Path(data_dir)` (import anti-pattern). Added `from pathlib import Path` at module level; function now uses `Path(data_dir)` directly. Local `from ncaa_eval.model.tracking import RunStore` renamed to `_RunStore` to avoid shadowing and clarify it is a local import.

**L3 Fixed**: `_map_ensemble_feature_labels` used `.title()` for model type names, producing `"Xgboost Prediction"` instead of `"XGBoost Prediction"`. Added `_MODEL_TYPE_DISPLAY_NAMES` lookup table (`{"xgboost": "XGBoost", "elo": "Elo", "logistic_regression": "Logistic Regression"}`) with `.title()` fallback for unknown types. Updated 4 test assertions to use the correct `"XGBoost Prediction"` label.

### Code Review Fixes (Claude Sonnet 4.6, 2026-03-12)

**H1 Fixed**: AC #2 partially unimplemented — OOF log loss was missing from Ensemble Components table. Added `_load_oof_log_losses()` helper that reads `store.load_metrics()` for each `oof_backtest_run_ids` entry. Table now shows "Base Model | OOF Log Loss" per AC spec.

**M1 Fixed**: `_map_ensemble_feature_labels()` and `load_ensemble_manifest()` called `store.model_dir(run_id)` which creates the directory as a side effect. Changed to use `store._runs_dir / run_id / "model" / "manifest.json"` directly.

**M2 Fixed**: Ensemble Components table had a useless "Index" column (0, 1, 2...) — removed. Table now shows only "Base Model" and "OOF Log Loss".

**M3 Fixed**: Both `run_bracket_simulation()` and `run_bracket_simulation_with_progress()` used `assert isinstance(model, StackedEnsemble)` which is unsafe in production (not caught by `except (OSError, ValueError, KeyError, TypeError)`). Replaced with graceful `if not isinstance(...) return None` pattern. Extracted `_build_probability_provider()` helper to keep both functions under C901/PLR0911 complexity limits.

**M4 Fixed**: Added `test_returns_empty_on_corrupt_json` to `TestLoadEnsembleManifest` — covers `json.JSONDecodeError` path for truncated manifest files.

**M5 Fixed**: Added `TestRunBracketSimulationWithProgress::test_returns_result_for_ensemble_model_with_progress` — covers the MC simulation ensemble provider path (previously untested).

### File List

- `dashboard/lib/simulation_helpers.py` — Modified: ensemble branch in provider selection, `_build_probability_provider()` helper extracted
- `dashboard/lib/data_loaders.py` — Modified: ensemble feature importances, `_map_ensemble_feature_labels()`, `_MODEL_TYPE_DISPLAY_NAMES` lookup table, `load_ensemble_manifest()`, `_runs_dir` path pattern
- `dashboard/pages/3_Model_Deep_Dive.py` — Modified: `_render_ensemble_components()` with OOF log loss, `_load_oof_log_losses()` helper, ensemble chart title, `from pathlib import Path` added
- `tests/unit/test_dashboard_filters.py` — Modified: ensemble bracket simulation tests, manifest tests, `_make_manifest_store()` helper, `TestRunBracketSimulationWithProgress`, XGBoost label assertions updated
- `tests/unit/test_deep_dive_page.py` — Modified: `TestEnsembleDeepDive` class (3 tests), manifest now includes `oof_backtest_run_ids`, XGBoost label fixture updated
- `tests/unit/test_leaderboard_page.py` — Modified: `TestEnsembleLeaderboardDisplay` class (1 test)
- `_bmad-output/planning-artifacts/template-requirements.md` — Modified: Story 10.3 learnings added
- `_bmad-output/implementation-artifacts/sprint-status.yaml` — Modified: status update
- `_bmad-output/implementation-artifacts/10-3-ensemble-dashboard-registry-integration.md` — Modified: story status

## Change Log

- 2026-03-12: Implemented dashboard ensemble integration — bracket simulation, feature importance labels, deep dive components section, verification of leaderboard and CLI paths (Story 10.3)
- 2026-03-12: Code review fixes — OOF log loss in components table (H1), model_dir side-effect fix (M1), Index column removed (M2), assert isinstance → graceful check (M3), JSONDecodeError test (M4), with_progress ensemble test (M5)
- 2026-03-12: Code review fixes (round 2) — `__import__("pathlib")` anti-pattern fixed (M1), `_MODEL_TYPE_DISPLAY_NAMES` lookup for correct "XGBoost" capitalization (L3), 4 test assertions updated
