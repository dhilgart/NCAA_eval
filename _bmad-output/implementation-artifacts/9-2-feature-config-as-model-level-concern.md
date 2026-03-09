# Story 9.2: Feature Config as Model-Level Concern

Status: done

## Story

As a **data scientist**,
I want to **embed feature engineering configuration directly in my model class**,
so that **my model always receives inputs in the correct format, I can experiment with different feature combinations by passing constructor kwargs, and loaded model artifacts carry their own feature requirements without external configuration files**.

## Acceptance Criteria

1. **Given** any concrete `Model` subclass (`XGBoostModel`, `LogisticRegressionModel`, `EloModel`)
   **When** the developer instantiates the model
   **Then** the model exposes a `feature_config: FeatureConfig` attribute derived from its constructor kwargs

2. **Given** feature-relevant kwargs (e.g., `batch_rating_types`, `graph_features_enabled`, `ordinal_composite`)
   **When** passed to a model's `__init__`
   **Then** they are threaded into the model's `FeatureConfig`

3. **Given** `run_training()` is called with a model
   **When** training begins
   **Then** `run_training()` reads `model.feature_config` to build the feature server instead of using the hardcoded defaults in `_setup_feature_server()`

4. **Given** `model.save(path)` is called
   **When** the model is persisted
   **Then** a `feature_config.json` sidecar file is written alongside model weights

5. **Given** `model.load(path)` is called
   **When** the model is reconstructed
   **Then** the `FeatureConfig` is loaded from the sidecar so the loaded model knows exactly what columns it expects

6. **Given** `fit()` has completed on any stateless model
   **When** the model is inspected
   **Then** `self.feature_names_: list[str]` contains the ordered list of feature columns it was trained on

7. **Given** `calibration_method` currently exists on `FeatureConfig`
   **When** this story is complete
   **Then** `calibration_method` has been removed from `FeatureConfig` and added to `ModelConfig` (calibration is a model-output concern, not a feature-computation concern)

8. **Given** `EloModel` is instantiated
   **When** its `feature_config` is inspected
   **Then** it uses a minimal `FeatureConfig` (no batch ratings, no ordinals, `elo_enabled=True`) since it reconstructs `Game` objects from metadata columns only

9. **Given** the CLI command `ncaa-eval train` is executed
   **When** a model is trained with default settings
   **Then** behavior is identical to the current implementation (backward compatible — models carry their own default feature configs)

## Tasks / Subtasks

- [x] Task 1: Relocate `calibration_method` from `FeatureConfig` to `ModelConfig` (AC: #7)
  - [x] 1.1 Remove `calibration_method` field from `FeatureConfig` in `src/ncaa_eval/transform/feature_serving.py`
  - [x] 1.2 Keep `CalibrationMethod` type alias in transform module (still needed for `ModelConfig`)
  - [x] 1.3 Add `calibration_method: CalibrationMethod | None = None` field to `ModelConfig` in `src/ncaa_eval/model/base.py`
  - [x] 1.4 No code reads `FeatureConfig.calibration_method` — field was vestigial
  - [x] 1.5 Update tests that construct `FeatureConfig` with `calibration_method` kwarg
  - [x] 1.6 `CalibrationMethod` imported from `feature_serving` into `base.py`

- [x] Task 2: Add `feature_config` attribute to `Model` ABC (AC: #1)
  - [x] 2.1 Add `feature_config: FeatureConfig` as a declared attribute on `Model` base class
  - [x] 2.2 Add `FeatureConfig` import to `src/ncaa_eval/model/base.py`

- [x] Task 3: Add `feature_names_` convention to stateless models (AC: #6)
  - [x] 3.1 `XGBoostModel`: renamed `self._feature_names` → `self.feature_names_` (sklearn convention)
  - [x] 3.2 `LogisticRegressionModel`: added `self.feature_names_: list[str] = []` init, set in `fit()`

- [x] Task 4: Refactor `XGBoostModel.__init__` to accept feature kwargs (AC: #1, #2)
  - [x] 4.1 Added keyword-only args: `batch_rating_types`, `graph_features_enabled`, `ordinal_composite` with defaults matching current hardcoded values
  - [x] 4.2 Construct `self.feature_config = FeatureConfig(...)` from those kwargs
  - [x] 4.3 Keep existing `config: XGBoostModelConfig | None` parameter for hyperparams

- [x] Task 5: Refactor `EloModel.__init__` to set minimal `feature_config` (AC: #8)
  - [x] 5.1 Set `self.feature_config = FeatureConfig(sequential_windows=(), graph_features_enabled=False, batch_rating_types=(), ordinal_composite=None, elo_enabled=True, elo_config=...)`

- [x] Task 6: Refactor `LogisticRegressionModel.__init__` to accept feature kwargs (AC: #1, #2)
  - [x] 6.1 Same pattern as XGBoostModel — accept feature kwargs, construct `FeatureConfig`

- [x] Task 7: Serialize `feature_config` in `save()` / `load()` (AC: #4, #5)
  - [x] 7.1 Created shared `_feature_config_io.py` helper (DRY — avoids triplicating logic)
  - [x] 7.2 `XGBoostModel.save()` / `.load()` — uses shared helpers
  - [x] 7.3 `EloModel.save()` / `.load()` — uses shared helpers
  - [x] 7.4 `LogisticRegressionModel.save()` / `.load()` — uses shared helpers
  - [x] 7.5 Backward compatibility: if `feature_config.json` is missing on `load()`, default `FeatureConfig` preserved

- [x] Task 8: Refactor `_setup_feature_server()` and `run_training()` (AC: #3, #9)
  - [x] 8.1 Changed `_setup_feature_server(data_dir)` → `_setup_feature_server(data_dir, feature_config)`
  - [x] 8.2 `run_training()` reads `model.feature_config` and passes to `_setup_feature_server()`
  - [x] 8.3 CLI behavior unchanged — models carry default configs matching previous hardcoded values

- [x] Task 9: Update tests (all ACs)
  - [x] 9.1 Updated existing model unit tests for new `feature_config` attribute
  - [x] 9.2 Added tests: model instantiation → `feature_config` present with correct values (all 3 models)
  - [x] 9.3 Added tests: custom feature kwargs → `FeatureConfig` reflects overrides (XGBoost, LogReg)
  - [x] 9.4 Added tests: `save()` writes `feature_config.json`, `load()` reads it back (all 3 models)
  - [x] 9.5 Added tests: backward compat — `load()` without sidecar uses defaults (all 3 models)
  - [x] 9.6 Added tests: `feature_names_` set after `fit()` on stateless models (XGBoost, LogReg)
  - [x] 9.7 Updated test: `calibration_method` on `ModelConfig`, not on `FeatureConfig`
  - [x] 9.8 Training pipeline tests pass with new `_setup_feature_server` signature (existing CLI tests cover AC #9)
  - [x] 9.9 Full test suite: 964 passed, 1 failed (check-manifest — expected, new file not in VCS), 1 skipped
  - [x] 9.10 Type checks: `mypy --strict` passes on all 94 source files
  - [x] 9.11 Linter: `ruff check .` passes

- [ ] Review Follow-ups (AI)
  - [ ] [AI-Review][MEDIUM] `feature_config` annotation on `Model` ABC is not enforced — a future subclass that forgets `self.feature_config = ...` in `__init__` will pass mypy and only crash at runtime. Consider an `__init_subclass__` check or abstract property. [src/ncaa_eval/model/base.py:44]
  - [ ] [AI-Review][MEDIUM] `save_feature_config` uses `default=str` fallback in `json.dumps` — silently converts non-JSON-native types to strings if FeatureConfig gains new field types. Remove `default=str`; all current fields are JSON-native after `asdict()`. [src/ncaa_eval/model/_feature_config_io.py:15]
  - [ ] [AI-Review][MEDIUM] No test for `ordinal_systems` non-None round-trip via save/load sidecar. Add a test: save model with `FeatureConfig(ordinal_systems=("massey", "rpi"))`, load it, assert `ordinal_systems == ("massey", "rpi")`. [tests/unit/test_model_xgboost.py or test_model_logistic_regression.py]
  - [ ] [AI-Review][LOW] `EloModel.__init__` calls `_to_elo_config(self._config)` twice — store result in a local variable and reuse. [src/ncaa_eval/model/elo.py:54-62]
  - [ ] [AI-Review][LOW] `FeatureConfig` not exported from `ncaa_eval.model.__init__.py` despite being a model-level concern. Add to `__all__` for clean public API. [src/ncaa_eval/model/__init__.py]
  - [ ] [AI-Review][LOW] No test asserts that `run_training()` passes model's custom `feature_config` to `StatefulFeatureServer` (AC3). Current CLI tests only verify backward-compat defaults; a regression where `model.feature_config` is ignored would not be caught. [tests/unit/test_cli_train.py]

## Dev Notes

### Why This Story Matters

This is a **hard prerequisite for Epic 10 (Ensemble Modeling)**. The ensemble framework needs each sub-model to know its own feature requirements so that:
- `run_training()` can build per-model feature servers
- `StackedEnsemble.predict_proba()` can route the right column slice to each sub-model via `feature_names_`
- Loaded models carry their feature config for inference without external configuration

### Current State (What's Broken)

`FeatureConfig` is constructed inside `_setup_feature_server()` in `src/ncaa_eval/cli/train.py:90-100` with hardcoded defaults:

```python
def _setup_feature_server(data_dir: Path) -> StatefulFeatureServer:
    repo = ParquetRepository(base_path=data_dir)
    data_server = ChronologicalDataServer(repo)
    feature_config = FeatureConfig(
        graph_features_enabled=False,
        batch_rating_types=("srs",),
        ordinal_composite=None,
        calibration_method=None,
    )
    return StatefulFeatureServer(config=feature_config, data_server=data_server)
```

Library users cannot vary feature engineering without editing source code. Models have no knowledge of what features they were trained on.

### Target Design Pattern

From `specs/ensemble-architecture.md` §2.2:

```python
class XGBoostModel(Model):
    def __init__(
        self,
        config: XGBoostModelConfig | None = None,
        *,
        batch_rating_types: tuple[BatchRatingType, ...] = ("srs",),
        graph_features_enabled: bool = False,
        ordinal_composite: OrdinalCompositeMethod | None = None,
        # ... other feature kwargs ...
    ) -> None:
        self._config = config or XGBoostModelConfig()
        self.feature_config = FeatureConfig(
            batch_rating_types=batch_rating_types,
            graph_features_enabled=graph_features_enabled,
            ordinal_composite=ordinal_composite,
        )
```

```python
class EloModel(StatefulModel):
    def __init__(self, config: EloModelConfig | None = None) -> None:
        self._config = config or EloModelConfig()
        self.feature_config = FeatureConfig(
            sequential_windows=(),
            graph_features_enabled=False,
            batch_rating_types=(),
            ordinal_composite=None,
            elo_enabled=True,
            elo_config=self._to_elo_config(self._config),
        )
```

### Critical Implementation Details

**Default feature kwargs must match current `_setup_feature_server()` hardcoded values** so that existing CLI behavior is preserved:
- `graph_features_enabled=False`
- `batch_rating_types=("srs",)`
- `ordinal_composite=None`
- `calibration_method=None` (now on ModelConfig, default None)

**`FeatureConfig` serialization**: Use `dataclasses.asdict(feature_config)` → `json.dumps()` for save. For load, `json.loads()` → `FeatureConfig(**data)`. Handle `EloConfig` nested dataclass — it needs special deserialization since `FeatureConfig.elo_config` is an `EloConfig | None`.

**`feature_names_` on XGBoostModel**: Already stores `self._feature_names` internally — just rename to public `feature_names_` and ensure it's set during `fit()`. This is line 123: `self._feature_names = list(X.columns)`. Also update references in `save()` (line 156), `load()` (line 178), and `get_feature_importances()` (lines 187-190).

**`feature_names_` on LogisticRegressionModel**: Add `self.feature_names_: list[str] = []` in `__init__` and `self.feature_names_ = list(X.columns)` in `fit()`.

**RunStore.save_model already saves feature_names externally** via `feature_names` kwarg (see `tracking.py:205-230`). This is the run-level tracking. The model-level `feature_names_` attribute is separate — it travels with the model object in memory and is used by ensemble routing.

**Backward compatibility for `load()`**: If `feature_config.json` doesn't exist (old model artifacts), construct a default `FeatureConfig` matching the current hardcoded defaults. Do NOT raise an error.

### `calibration_method` Relocation Details

`FeatureConfig.calibration_method` (line 86 of `feature_serving.py`) is declared but **never used by `StatefulFeatureServer`** — it's a vestigial field. Calibration is performed by the `Calibrator` classes in `transform/calibration.py` as a post-prediction step, not during feature serving. Moving it to `ModelConfig` is a correctness fix.

Search for all references to `calibration_method` before removing:
- `feature_serving.py:86` — field declaration
- `train.py:98` — `calibration_method=None` in `FeatureConfig()` constructor
- Tests that construct `FeatureConfig` with this kwarg
- `CalibrationMethod` type alias export in `transform/__init__.py` (keep — still needed for the ModelConfig field)

### Project Structure Notes

Files to modify:
- `src/ncaa_eval/model/base.py` — Add `feature_config` attribute declaration, add `calibration_method` to `ModelConfig`
- `src/ncaa_eval/model/xgboost_model.py` — Add feature kwargs, `feature_config`, rename `_feature_names` → `feature_names_`, add sidecar save/load
- `src/ncaa_eval/model/elo.py` — Add minimal `feature_config`, add sidecar save/load
- `src/ncaa_eval/model/logistic_regression.py` — Add feature kwargs, `feature_config`, `feature_names_`, add sidecar save/load
- `src/ncaa_eval/model/__init__.py` — May need to export `FeatureConfig` if it becomes part of model public API
- `src/ncaa_eval/transform/feature_serving.py` — Remove `calibration_method` from `FeatureConfig`
- `src/ncaa_eval/transform/__init__.py` — Update exports if `CalibrationMethod` moves
- `src/ncaa_eval/cli/train.py` — Refactor `_setup_feature_server()` to accept `FeatureConfig` parameter
- Test files: `tests/unit/test_model_*.py`, `tests/unit/test_feature_serving.py`, `tests/unit/test_cli_train.py`

### Conventions

- `from __future__ import annotations` in all Python files
- `mypy --strict` for all `src/` and `tests/` files
- Google-style docstrings
- `@dataclass(frozen=True)` for `FeatureConfig` — immutable by design
- Pydantic `BaseModel` for `ModelConfig` subclasses
- `FeatureConfig` is a frozen dataclass, NOT a Pydantic model

### References

- [Source: specs/ensemble-architecture.md §2] — Complete target design for this story
- [Source: src/ncaa_eval/cli/train.py:90-100] — Current hardcoded `_setup_feature_server()`
- [Source: src/ncaa_eval/model/base.py] — Model ABC definition
- [Source: src/ncaa_eval/model/xgboost_model.py] — XGBoost model implementation
- [Source: src/ncaa_eval/model/elo.py] — Elo model implementation
- [Source: src/ncaa_eval/model/logistic_regression.py] — LogReg model implementation
- [Source: src/ncaa_eval/transform/feature_serving.py:59-113] — FeatureConfig dataclass
- [Source: src/ncaa_eval/model/tracking.py:205-230] — RunStore.save_model with feature_names
- [Source: _bmad-output/implementation-artifacts/9-1-kaggle-submission-export.md] — Previous story learnings

### Previous Story Intelligence (Story 9.1)

- **Pure function pattern**: Story 9.1 established the pattern of core pure functions with thin CLI/dashboard wrappers. Apply same principle here — keep `FeatureConfig` serialization as pure functions.
- **DRY violations caught in review**: Code review found CLI and dashboard duplicating 25 lines of model-loading logic. Watch for similar duplication when adding `feature_config` handling to three model classes — extract shared serialization helpers if warranted.
- **Pre-commit ruff-format**: First commit attempts may auto-fix formatting. Re-stage fixed files.
- **Test count baseline**: 946 tests passed at Story 9.1 completion. All tests must continue to pass.

### Git Intelligence

Recent commits show the project is stable on main with all epics 1-8 done and 9.1 merged. No in-flight changes to conflict with. Latest commit: `bd13cb7 feat(evaluation): add Kaggle submission export`.

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

- `instance._feature_names` in `xgboost_model.py` `load()` wasn't caught by `replace_all` (only matched `self.` prefix) — required manual fix
- `dict[str, object]` type on `_feature_config_io.py:_deserialize_feature_config` caused mypy `arg-type` error on `tuple(data[key])` — fixed by using `dict[str, Any]`

### Completion Notes List

- Created shared `_feature_config_io.py` helper module to avoid triplicating FeatureConfig serialization logic across 3 model classes (DRY principle from Story 9.1 review learnings)
- `CalibrationMethod` type alias stays in `feature_serving.py` (still importable from transform layer) — only the field was relocated to `ModelConfig`
- `FeatureConfig` is a frozen dataclass with nested `EloConfig` — deserialization pops `elo_config` and reconstructs it separately, converts list→tuple for frozen fields
- Backward compatibility: all 3 model `load()` methods gracefully handle missing `feature_config.json` by keeping the model's constructor-default `FeatureConfig`
- No circular imports: `base.py` imports from `feature_serving.py` (confirmed at runtime)
- Test count: 964 passed (up from 946 baseline), +18 new tests for Story 9.2

### Change Log

- `src/ncaa_eval/transform/feature_serving.py` — Removed `calibration_method` field from `FeatureConfig`
- `src/ncaa_eval/model/base.py` — Added `feature_config: FeatureConfig` to `Model` ABC, added `calibration_method` to `ModelConfig`
- `src/ncaa_eval/model/_feature_config_io.py` — **NEW** shared FeatureConfig serialization helpers
- `src/ncaa_eval/model/xgboost_model.py` — Added feature kwargs, `feature_config`, renamed `_feature_names` → `feature_names_`, sidecar save/load
- `src/ncaa_eval/model/elo.py` — Added minimal `feature_config` (elo_enabled=True), sidecar save/load
- `src/ncaa_eval/model/logistic_regression.py` — Added feature kwargs, `feature_config`, `feature_names_`, sidecar save/load
- `src/ncaa_eval/cli/train.py` — Refactored `_setup_feature_server()` to accept `FeatureConfig` parameter from model
- `tests/unit/test_feature_serving.py` — Removed `calibration_method` references, updated test to use `ModelConfig`
- `tests/unit/test_model_xgboost.py` — Added `TestFeatureConfig` (3 tests), `TestFeatureConfigSaveLoad` (3 tests)
- `tests/unit/test_model_elo.py` — Added `TestFeatureConfig` (4 tests), `TestFeatureConfigSaveLoad` (3 tests)
- `tests/unit/test_model_logistic_regression.py` — Added `TestFeatureConfig` (3 tests), `TestFeatureConfigSaveLoad` (4 tests)
- `tests/integration/test_feature_serving_integration.py` — Removed `calibration_method=None` kwargs
- `tests/integration/test_elo_integration.py` — Removed `calibration_method=None` kwargs

### File List

- `src/ncaa_eval/transform/feature_serving.py`
- `src/ncaa_eval/model/base.py`
- `src/ncaa_eval/model/_feature_config_io.py` (new)
- `src/ncaa_eval/model/xgboost_model.py`
- `src/ncaa_eval/model/elo.py`
- `src/ncaa_eval/model/logistic_regression.py`
- `src/ncaa_eval/cli/train.py`
- `tests/unit/test_feature_serving.py`
- `tests/unit/test_model_xgboost.py`
- `tests/unit/test_model_elo.py`
- `tests/unit/test_model_logistic_regression.py`
- `tests/integration/test_feature_serving_integration.py`
- `tests/integration/test_elo_integration.py`
- `_bmad-output/implementation-artifacts/9-2-feature-config-as-model-level-concern.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
