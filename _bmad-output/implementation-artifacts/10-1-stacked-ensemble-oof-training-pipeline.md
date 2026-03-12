# Story 10.1: StackedEnsemble Class and OOF Training Pipeline

Status: ready-for-dev

## Story

As a **data scientist**,
I want to **define a stacked ensemble by listing base models and a meta-learner and train the whole thing in one `run_training()` call**,
so that **I can build an ensemble that learns optimal, game-context-dependent weights without manually orchestrating out-of-fold prediction generation or alignment**.

## Acceptance Criteria

1. **Given** a `StackedEnsemble` instance with `base_models`, `meta_learner`, and `contextual_features`
   **When** the user calls `run_training(ensemble, data_dir=..., start_year=..., end_year=..., output_dir=..., model_name=...)`
   **Then** `run_training()` detects the `StackedEnsemble` type and routes to `_run_ensemble_training()`

2. **Given** the ensemble training pipeline
   **When** OOF generation runs for each base model
   **Then** for each base model, a walk-forward backtest is run using that model's own `feature_config` to produce out-of-fold (OOF) predictions

3. **Given** OOF predictions from all base models
   **When** they are aligned by `game_id`
   **Then** an inner join is performed; a warning is logged if >5% of games are dropped by the join

4. **Given** aligned OOF predictions
   **When** the meta-training DataFrame is assembled
   **Then** it has columns `[pred_base_0, pred_base_1, ..., <contextual_features>]` plus the label `team_a_won`

5. **Given** the meta-training DataFrame
   **When** the meta-learner is trained
   **Then** `meta_learner.fit(meta_X, meta_y)` is called where `meta_y` is `team_a_won`

6. **Given** training is complete
   **When** final base model retraining runs
   **Then** each base model is retrained on the full dataset (all seasons) using its own feature server

7. **Given** training is complete
   **When** artifacts are saved
   **Then** each base model, the meta-learner, and a manifest recording base model names, `contextual_features`, and the run IDs of the OOF backtest runs are persisted

8. **Given** a `StackedEnsemble` instance
   **When** `ensemble.feature_config` is accessed
   **Then** it returns the union of all base models' `feature_config`s

## Tasks / Subtasks

- [ ] Task 1: Create `StackedEnsemble` dataclass (AC: #1, #8)
  - [ ] 1.1: Create `src/ncaa_eval/model/ensemble.py` with `StackedEnsemble` dataclass
  - [ ] 1.2: Implement `feature_config` property returning the union of all base model configs
  - [ ] 1.3: Create `StackedEnsembleConfig(ModelConfig)` Pydantic class
  - [ ] 1.4: Register via `@register_model("ensemble")` and import in `model/__init__.py`
- [ ] Task 2: Implement `_run_ensemble_training()` in `cli/train.py` (AC: #1–#7)
  - [ ] 2.1: Add `StackedEnsemble` type check in `run_training()` to dispatch to `_run_ensemble_training()`
  - [ ] 2.2: Implement OOF generation loop — per base model, build feature server from `base_model.feature_config`, run `run_backtest()`, collect `FoldResult` predictions
  - [ ] 2.3: Implement OOF alignment — inner join all base model OOF predictions on `game_id`; log warning if >5% dropped
  - [ ] 2.4: Implement meta-training set construction — join contextual features onto aligned OOF preds
  - [ ] 2.5: Train meta-learner on meta-training DataFrame
  - [ ] 2.6: Retrain each base model on full dataset using its own feature server
  - [ ] 2.7: Persist ensemble artifact — save each base model, meta-learner, and manifest
- [ ] Task 3: Implement `StackedEnsemble.save()` / `load()` (AC: #7)
  - [ ] 3.1: Save base models to `model/base_models/<idx>/`, meta-learner to `model/meta_learner/`
  - [ ] 3.2: Save manifest JSON with base model names, contextual features, meta-input column order
  - [ ] 3.3: Implement `load()` classmethod to reconstruct from disk
- [ ] Task 4: Write unit tests (all ACs)
  - [ ] 4.1: Test `StackedEnsemble.feature_config` union logic
  - [ ] 4.2: Test `_run_ensemble_training()` with mock models — verify OOF alignment, meta-training set shape, and artifact persistence
  - [ ] 4.3: Test save/load round-trip
  - [ ] 4.4: Test >5% OOF drop warning is logged
  - [ ] 4.5: Test model registry includes `"ensemble"`
- [ ] Task 5: Quality gates
  - [ ] 5.1: `ruff check .`
  - [ ] 5.2: `mypy --strict src/ncaa_eval tests`
  - [ ] 5.3: Full test suite passes (`pytest`)

## Dev Notes

### Critical Design Decisions

**`StackedEnsemble` is NOT a `Model` subclass.** It is a standalone `@dataclass` (see `specs/ensemble-architecture.md` §3.2). `run_training()` dispatches on `isinstance(model, StackedEnsemble)` via a union type `Model | StackedEnsemble`. The ensemble holds `Model` instances as children but does not implement the `Model` ABC directly — it has a fundamentally different lifecycle (no single `fit(X, y)` / `predict_proba(X)` contract; instead, it orchestrates multiple sub-model training runs and a meta-learner).

**However**, for run tracking and dashboard discovery, the ensemble needs to be discoverable via `RunStore`. The `run_training()` return type stays `ModelRun`. The `model_type` field in `ModelRun` should be set to `"ensemble"`.

### `StackedEnsemble` Dataclass Location

Create as `src/ncaa_eval/model/ensemble.py`. This parallels `elo.py`, `xgboost_model.py`, `logistic_regression.py`.

```python
@dataclass
class StackedEnsemble:
    base_models: list[Model]
    meta_learner: Model  # Must be a stateless Model (not StatefulModel)
    contextual_features: list[str] = field(
        default_factory=lambda: ["seed_diff", "is_tournament", "loc_encoding"]
    )
```

### `feature_config` Property — Union Logic

The ensemble's `feature_config` must be the **superset** of all base model configs:

```python
@property
def feature_config(self) -> FeatureConfig:
    """Union of all base model feature configs."""
    # Merge sequential_windows, batch_rating_types, etc. from all base models
    # elo_enabled = True if ANY base model has it
    # graph_features_enabled = True if ANY base model has it
    # sequential_windows = union of all windows
    # batch_rating_types = union of all types
    # ordinal_composite = pick the first non-None, or None
```

Key rules:
- `sequential_windows`: union of all tuples → `tuple(sorted(set(...)))`
- `ewma_alphas`: union of all tuples → `tuple(sorted(set(...)))`
- `batch_rating_types`: union of all tuples → `tuple(sorted(set(...)))`
- `graph_features_enabled`: `True` if ANY base model has it `True`
- `elo_enabled`: `True` if ANY base model has it `True`
- `elo_config`: take from the first Elo-enabled base model (there should be at most one)
- `ordinal_composite`: take the first non-`None` value across base models
- `ordinal_systems`: union if any non-`None`
- `matchup_deltas`, `gender_scope`, `dataset_scope`: assert all base models agree (raise `ValueError` if not)

### `run_training()` Dispatch Modification

In `src/ncaa_eval/cli/train.py`, change the signature:

```python
def run_training(
    model: Model | StackedEnsemble,
    *,
    ...
) -> ModelRun:
    if isinstance(model, StackedEnsemble):
        return _run_ensemble_training(model, ...)
    # existing leaf-model path unchanged
```

### `_run_ensemble_training()` Pipeline Steps

**Step 1 — OOF generation (per base model):**
For each `base_model` in `ensemble.base_models`:
- `server = _setup_feature_server(data_dir, base_model.feature_config)`
- `seasons = list(range(start_year, end_year + 1))`
- `mode = "stateful" if isinstance(base_model, StatefulModel) else "batch"`
- `result = run_backtest(copy.deepcopy(base_model), server, seasons=seasons, mode=mode, n_jobs=1)`
- Collect `FoldResult.predictions` and `FoldResult.test_game_ids` from each fold

**Step 2 — OOF alignment:**
- Each base model produces a DataFrame: `game_id`, `pred_base_<i>`, `team_a_won`
- Inner join all on `game_id`
- Calculate drop percentage: `1 - len(joined) / len(largest_base)`. If >5%, log warning via `logger.warning()`

**Step 3 — Meta-training set construction:**
- Build a minimal feature server using the ensemble's `feature_config` (or just the features needed for `contextual_features` — `seed_diff`, `is_tournament`, `loc_encoding` are always present in any feature server output since they're metadata/SEED block columns)
- However, `seed_diff`, `is_tournament`, and `loc_encoding` are in `METADATA_COLS` and are present in every backtest fold's test DataFrame. So the contextual features can be extracted from the same OOF DataFrames — no need for a separate feature server
- **Important**: The OOF predictions from `run_backtest` come from `_evaluate_fold`, which randomizes team assignment on the test set. The `seed_diff` and `loc_encoding` values in the randomized test data are already correct (signs flipped when teams are swapped). These values can be joined onto the aligned OOF predictions by `game_id`
- **Approach**: During OOF generation, also collect `test_game_ids`, plus the contextual feature columns from the test DataFrame. Then join these onto the aligned OOF DataFrame
- **Simpler approach**: Actually, `FoldResult` already stores `test_game_ids`, `test_team_a_ids`, `test_team_b_ids`, `predictions`, `actuals`. But it does NOT store the raw feature columns. So you need to reconstruct the contextual features. Since `seed_diff`, `is_tournament`, `loc_encoding` are metadata columns present in the pre-randomized data, retrieve them from the feature server's season output and join by `game_id`
- **Simplest approach**: Build one feature server with the ensemble's union `feature_config`, serve all seasons, and extract contextual features from there. Then join onto the aligned OOF predictions by `game_id`. This avoids needing to store extra columns in FoldResult

**Step 4 — Meta-learner training:**
```python
meta_X = aligned_oof[["pred_base_0", "pred_base_1", ..., "seed_diff", "is_tournament", "loc_encoding"]]
meta_y = aligned_oof["team_a_won"].astype(int)
ensemble.meta_learner.fit(meta_X, meta_y)
```
Store `meta_column_order = list(meta_X.columns)` for inference-time reconstruction.

**Step 5 — Final base model retraining:**
For each base model, build its own feature server, build all season features, and call `_prepare_and_train()` equivalent logic (randomize for stateless, fit on full data).

**Step 6 — Artifact persistence:**
- Save each base model via `model.save(path / "base_models" / f"base_{i}")`
- Save meta-learner via `ensemble.meta_learner.save(path / "meta_learner")`
- Save manifest JSON: `{"base_model_names": [...], "contextual_features": [...], "meta_column_order": [...], "oof_run_ids": [...]}`
- Save `feature_config.json` for the ensemble's union config

### Existing Code to Reuse — DO NOT Reinvent

| Facility | Location | Usage |
|---|---|---|
| `run_backtest()` | `src/ncaa_eval/evaluation/backtest.py` | OOF generation — call it per base model |
| `_setup_feature_server()` | `src/ncaa_eval/cli/train.py:90` | Build feature server from a `FeatureConfig` |
| `_build_fold_predictions()` | `src/ncaa_eval/cli/train.py:59` | Convert `BacktestResult` → fold predictions DataFrame |
| `_randomize_team_assignment()` | `src/ncaa_eval/evaluation/backtest.py` | Label balancing for stateless models |
| `_feature_cols()` | `src/ncaa_eval/evaluation/backtest.py` | Extract feature column names |
| `RunStore` | `src/ncaa_eval/model/tracking.py` | Persist `ModelRun`, predictions, metrics, model artifacts |
| `save_feature_config()` / `load_feature_config()` | `src/ncaa_eval/model/_feature_config_io.py` | FeatureConfig serialization |
| `FoldResult`, `BacktestResult` | `src/ncaa_eval/evaluation/backtest.py` | OOF prediction containers |
| `walk_forward_splits()` | `src/ncaa_eval/evaluation/splitter.py` | Walk-forward CV folds |

### Key Type Signatures to Match

```python
# run_training signature change
def run_training(
    model: Model | StackedEnsemble,   # <-- union type
    *,
    start_year: int,
    end_year: int,
    data_dir: Path,
    output_dir: Path,
    model_name: str,
    console: Console | None = None,
) -> ModelRun:

# _run_ensemble_training (new function)
def _run_ensemble_training(
    ensemble: StackedEnsemble,
    *,
    start_year: int,
    end_year: int,
    data_dir: Path,
    output_dir: Path,
    model_name: str,
    console: Console,
) -> ModelRun:
```

### Meta-Learner Constraints

The meta-learner MUST be a **stateless** model (not `StatefulModel`). It receives a tabular DataFrame of `[pred_base_0, ..., seed_diff, is_tournament, loc_encoding]` — these are numeric features, not game sequences. The meta-learner calls `.fit(meta_X, meta_y)` and `.predict_proba(meta_X)` in the standard stateless pattern.

Validate this at `StackedEnsemble.__post_init__()`:
```python
def __post_init__(self) -> None:
    if isinstance(self.meta_learner, StatefulModel):
        raise TypeError("meta_learner must be a stateless Model, not StatefulModel")
    if len(self.base_models) < 2:
        raise ValueError("StackedEnsemble requires at least 2 base models")
```

### Manifest Schema

```json
{
  "base_model_types": ["xgboost", "elo"],
  "base_model_count": 2,
  "contextual_features": ["seed_diff", "is_tournament", "loc_encoding"],
  "meta_column_order": ["pred_base_0", "pred_base_1", "seed_diff", "is_tournament", "loc_encoding"],
  "oof_backtest_run_ids": ["uuid-base-0", "uuid-base-1"],
  "oof_game_count": 1234,
  "oof_drop_pct": 0.02
}
```

### Testing Strategy

**Unit tests in `tests/unit/test_model_ensemble.py`:**

1. **`test_feature_config_union`** — Create two models with different `FeatureConfig`s, verify ensemble's `feature_config` is the correct union.
2. **`test_meta_learner_stateful_rejected`** — Verify `TypeError` raised when `meta_learner` is `StatefulModel`.
3. **`test_min_two_base_models`** — Verify `ValueError` raised with <2 base models.
4. **`test_oof_alignment_inner_join`** — Mock two base models with overlapping but not identical `game_id` sets; verify inner join drops correctly and warning fires at >5%.
5. **`test_save_load_roundtrip`** — Create a `StackedEnsemble`, save, load, verify manifest and base model count.
6. **`test_ensemble_registered`** — `"ensemble" in list_models()`.

For the OOF pipeline integration test, use `LogisticRegressionModel` as base models (cheap to train) with a small synthetic dataset. Do NOT use `XGBoostModel` in unit tests — it's slow and requires real data shape.

**Test fixtures**: Reuse existing conftest fixtures for model instances and synthetic DataFrames. See `tests/conftest.py` for `sample_feature_df`, `sample_labels`, `mock_run_store`.

### File Structure

```
src/ncaa_eval/model/
├── __init__.py          # Add ensemble import
├── base.py              # No changes
├── ensemble.py          # NEW — StackedEnsemble dataclass
├── elo.py               # No changes
├── xgboost_model.py     # No changes
├── logistic_regression.py # No changes
├── registry.py          # No changes
├── tracking.py          # No changes
├── _feature_config_io.py # No changes

src/ncaa_eval/cli/
├── train.py             # MODIFIED — add StackedEnsemble dispatch + _run_ensemble_training()

tests/unit/
├── test_model_ensemble.py # NEW — ensemble tests
```

### `from __future__ import annotations` Required

All new Python files MUST have `from __future__ import annotations` as the first import (enforced by Ruff FA100).

### What NOT To Do

- **Do NOT implement `predict_proba()` or `predict_bracket()` on `StackedEnsemble`** — that's Story 10.2
- **Do NOT add dashboard integration** — that's Story 10.3
- **Do NOT create a tutorial notebook** — that's Story 10.4
- **Do NOT make `StackedEnsemble` inherit from `Model`** — it has a different lifecycle; the spec says `@dataclass`, not ABC subclass
- **Do NOT modify `FoldResult` to store extra columns** — work with the existing interface and build contextual features separately
- **Do NOT modify the backtest module** — use it as-is via `run_backtest()`
- **Do NOT add CLI support for ensemble training** — that can come later; `run_training()` is the programmatic API for now
- **Do NOT use `Plotly` in any notebook outputs** (project convention: matplotlib only for large datasets)

### Project Structure Notes

- New file `src/ncaa_eval/model/ensemble.py` follows the established pattern of one module per model type
- `model/__init__.py` must import `ensemble as _ensemble` to trigger `@register_model`
- `model/__init__.py` `__all__` should add `"StackedEnsemble"`
- All paths use `Path` objects, not string concatenation
- `mypy --strict` compliance: all type annotations required, no `Any` where avoidable

### References

- [Source: specs/ensemble-architecture.md — §3–§4 (constructor, training flow)]
- [Source: specs/ensemble-architecture.md — §2 (FeatureConfig as model-level concern, prerequisite Story 9.2)]
- [Source: _bmad-output/planning-artifacts/epics.md — Epic 10, Story 10.1]
- [Source: src/ncaa_eval/model/base.py — Model ABC, StatefulModel template]
- [Source: src/ncaa_eval/cli/train.py — run_training(), _setup_feature_server(), _build_fold_predictions()]
- [Source: src/ncaa_eval/evaluation/backtest.py — run_backtest(), FoldResult, BacktestResult, METADATA_COLS]
- [Source: src/ncaa_eval/evaluation/splitter.py — walk_forward_splits(), CVFold]
- [Source: src/ncaa_eval/model/tracking.py — RunStore, ModelRun, Prediction]
- [Source: src/ncaa_eval/model/_feature_config_io.py — save_feature_config(), load_feature_config()]
- [Source: src/ncaa_eval/model/xgboost_model.py — feature_names_ pattern, FeatureConfig usage]
- [Source: src/ncaa_eval/model/elo.py — StatefulModel FeatureConfig pattern]
- [Source: src/ncaa_eval/transform/feature_serving.py:59-111 — FeatureConfig dataclass, active_blocks()]

## Dev Agent Record

### Agent Model Used

{{agent_model_name_version}}

### Debug Log References

### Completion Notes List

### File List
