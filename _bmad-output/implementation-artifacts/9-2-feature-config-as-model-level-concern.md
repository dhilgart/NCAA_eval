# Story 9.2: Feature Config as Model-Level Concern

Status: ready-for-dev

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

- [ ] Task 1: Relocate `calibration_method` from `FeatureConfig` to `ModelConfig` (AC: #7)
  - [ ] 1.1 Remove `calibration_method` field from `FeatureConfig` in `src/ncaa_eval/transform/feature_serving.py`
  - [ ] 1.2 Remove `CalibrationMethod` import/export from `FeatureConfig` if it becomes unused there (keep type alias in transform module if still needed elsewhere)
  - [ ] 1.3 Add `calibration_method: CalibrationMethod | None = None` field to `ModelConfig` in `src/ncaa_eval/model/base.py`
  - [ ] 1.4 Update any code that reads `FeatureConfig.calibration_method` to read from `ModelConfig` instead
  - [ ] 1.5 Update tests that construct `FeatureConfig` with `calibration_method` kwarg
  - [ ] 1.6 Update `__init__.py` exports if `CalibrationMethod` needs to be exported from `model` module

- [ ] Task 2: Add `feature_config` attribute to `Model` ABC (AC: #1)
  - [ ] 2.1 Add `feature_config: FeatureConfig` as a declared attribute on `Model` base class (not abstract — subclasses set it in `__init__`)
  - [ ] 2.2 Add `FeatureConfig` import to `src/ncaa_eval/model/base.py`

- [ ] Task 3: Add `feature_names_` convention to stateless models (AC: #6)
  - [ ] 3.1 `XGBoostModel` already stores `self._feature_names: list[str]` — rename to public `feature_names_` (sklearn convention) and set it in `fit()`
  - [ ] 3.2 `LogisticRegressionModel` — add `self.feature_names_: list[str] = []` init, set in `fit()`

- [ ] Task 4: Refactor `XGBoostModel.__init__` to accept feature kwargs (AC: #1, #2)
  - [ ] 4.1 Add feature-relevant kwargs: `batch_rating_types`, `graph_features_enabled`, `ordinal_composite`, `sequential_windows`, `elo_enabled` etc. with defaults matching current `_setup_feature_server()` defaults
  - [ ] 4.2 Construct `self.feature_config = FeatureConfig(...)` from those kwargs
  - [ ] 4.3 Keep existing `config: XGBoostModelConfig | None` parameter for hyperparams

- [ ] Task 5: Refactor `EloModel.__init__` to set minimal `feature_config` (AC: #8)
  - [ ] 5.1 Set `self.feature_config = FeatureConfig(sequential_windows=(), graph_features_enabled=False, batch_rating_types=(), ordinal_composite=None, elo_enabled=True, elo_config=...)` in `__init__`

- [ ] Task 6: Refactor `LogisticRegressionModel.__init__` to accept feature kwargs (AC: #1, #2)
  - [ ] 6.1 Same pattern as XGBoostModel — accept feature kwargs, construct `FeatureConfig`

- [ ] Task 7: Serialize `feature_config` in `save()` / `load()` (AC: #4, #5)
  - [ ] 7.1 `XGBoostModel.save()` — write `feature_config.json` sidecar via `dataclasses.asdict()` + `json.dumps()`
  - [ ] 7.2 `XGBoostModel.load()` — read sidecar and reconstruct `FeatureConfig`
  - [ ] 7.3 `EloModel.save()` — same sidecar pattern
  - [ ] 7.4 `EloModel.load()` — same reconstruction
  - [ ] 7.5 `LogisticRegressionModel.save()` / `.load()` — same pattern
  - [ ] 7.6 Handle backward compatibility: if `feature_config.json` is missing on `load()`, use the model's default `FeatureConfig`

- [ ] Task 8: Refactor `_setup_feature_server()` and `run_training()` (AC: #3, #9)
  - [ ] 8.1 Change `_setup_feature_server(data_dir)` → `_setup_feature_server(data_dir, feature_config)` — accept `FeatureConfig` parameter
  - [ ] 8.2 In `run_training()`, read `model.feature_config` and pass to `_setup_feature_server(data_dir, model.feature_config)`
  - [ ] 8.3 Verify CLI `ncaa-eval train` still works unchanged (models carry their own default configs)

- [ ] Task 9: Update tests (all ACs)
  - [ ] 9.1 Update existing model unit tests for new `feature_config` attribute
  - [ ] 9.2 Add tests: model instantiation → `feature_config` present with correct values
  - [ ] 9.3 Add tests: custom feature kwargs → `FeatureConfig` reflects overrides
  - [ ] 9.4 Add tests: `save()` writes `feature_config.json`, `load()` reads it back
  - [ ] 9.5 Add tests: backward compat — `load()` without sidecar uses defaults
  - [ ] 9.6 Add tests: `feature_names_` set after `fit()` on stateless models
  - [ ] 9.7 Add tests: `calibration_method` on `ModelConfig`, not on `FeatureConfig`
  - [ ] 9.8 Update training pipeline tests for new `_setup_feature_server` signature
  - [ ] 9.9 Run full test suite: `pytest`
  - [ ] 9.10 Run type checks: `mypy --strict src/ncaa_eval tests`
  - [ ] 9.11 Run linter: `ruff check .`

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

{{agent_model_name_version}}

### Debug Log References

### Completion Notes List

### File List
