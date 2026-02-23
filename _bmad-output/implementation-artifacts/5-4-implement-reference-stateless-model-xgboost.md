# Story 5.4: Implement Reference Stateless Model (XGBoost)

Status: review

## Story

As a data scientist,
I want an XGBoost wrapper as the reference stateless model,
So that I have a powerful gradient-boosting baseline and a template for building other batch-trained models.

## Acceptance Criteria

1. **XGBoostModel wraps XGBClassifier** — `XGBoostModel(Model)` wraps `xgboost.XGBClassifier` implementing `Model` directly (no `StatefulModel` subclass — stateless models bypass the per-game lifecycle).
2. **fit** — `fit(X: pd.DataFrame, y: pd.Series)` calls `XGBClassifier.fit(X, y, eval_set=..., early_stopping_rounds=...)` using a validation split from `X`.
3. **predict_proba** — `predict_proba(X: pd.DataFrame) -> pd.Series` returns `XGBClassifier.predict_proba(X)[:, 1]` — P(team_a wins) as probabilities (XGBoost `binary:logistic` objective).
4. **XGBoostModelConfig** — `XGBoostModelConfig(ModelConfig)` is the Pydantic config with: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_alpha`, `reg_lambda`, `early_stopping_rounds`, `validation_fraction` — defaults from `specs/research/modeling-approaches.md` §5.5 and §6.4.
5. **Label balance** — Label balance is verified before training: if team_a/team_b assignment is non-random (e.g., always winner = team_a), `scale_pos_weight` must be set accordingly; document the convention in the implementation.
6. **save** — `save(path: Path)` calls `clf.save_model(str(path / "model.ubj"))` (XGBoost UBJSON native format, stable across versions) and writes config JSON to `path / "config.json"`.
7. **load** — `load(cls, path: Path) -> Self` instantiates `XGBClassifier()` then calls `clf.load_model(str(path / "model.ubj"))` — `load_model` is an instance method, NOT a class method.
8. **Plugin registration** — The model registers via the plugin registry as `"xgboost"`.
9. **Tests** — The model is covered by unit tests validating `fit`/`predict_proba`/`save`/`load` round-trip.

## Tasks / Subtasks

- [x] Task 1: Create `XGBoostModelConfig` (AC: #4)
  - [x] 1.1 Define `XGBoostModelConfig(ModelConfig)` with `model_name: Literal["xgboost"] = "xgboost"` and all hyperparameters from §5.5
  - [x] 1.2 Add `validation_fraction: float = 0.1` for automatic eval_set splitting in `fit()`
  - [x] 1.3 Verify Pydantic JSON round-trip with default and custom values

- [x] Task 2: Create `XGBoostModel(Model)` (AC: #1, #2, #3, #5)
  - [x] 2.1 Implement `__init__(config: XGBoostModelConfig | None = None)` — store config, instantiate `XGBClassifier` with config params
  - [x] 2.2 Implement `fit(X, y)` — split validation set from X/y using `validation_fraction`, call `XGBClassifier.fit(X_train, y_train, eval_set=[(X_val, y_val)])` with `early_stopping_rounds` from config
  - [x] 2.3 Guard `fit()` against empty DataFrame input (`if X.empty: raise ValueError`)
  - [x] 2.4 Implement `predict_proba(X)` — return `pd.Series(clf.predict_proba(X)[:, 1], index=X.index)`
  - [x] 2.5 Implement `get_config() -> XGBoostModelConfig`
  - [x] 2.6 Document label balance convention in docstring (team_a assignment and scale_pos_weight)

- [x] Task 3: Implement `save` / `load` (AC: #6, #7)
  - [x] 3.1 `save(path)` — create directory, save model via `clf.save_model(str(path / "model.ubj"))`, config via `path / "config.json"` with `model_dump_json()`
  - [x] 3.2 `load(cls, path) -> Self` — check BOTH `config.json` and `model.ubj` exist before reading either; `XGBClassifier()` then `clf.load_model(str(path / "model.ubj"))`; reconstruct from config; return Self
  - [x] 3.3 Raise `FileNotFoundError` with clear message on incomplete saves

- [x] Task 4: Register as plugin (AC: #8)
  - [x] 4.1 Add `@register_model("xgboost")` decorator to `XGBoostModel`
  - [x] 4.2 Update `model/__init__.py` to import `xgboost_model` module for auto-registration

- [x] Task 5: Write unit tests (AC: #9)
  - [x] 5.1 Test `XGBoostModelConfig` creation with defaults
  - [x] 5.2 Test `XGBoostModelConfig` creation with custom values
  - [x] 5.3 Test `XGBoostModelConfig` JSON round-trip
  - [x] 5.4 Test `XGBoostModel.fit(X, y)` trains successfully on synthetic data
  - [x] 5.5 Test `XGBoostModel.predict_proba(X)` returns probabilities in [0, 1]
  - [x] 5.6 Test `predict_proba` output length matches input length
  - [x] 5.7 Test `predict_proba` returns `pd.Series` with correct index
  - [x] 5.8 Test `save()` / `load()` round-trip produces identical predictions (`tmp_path`)
  - [x] 5.9 Test `save()` / `load()` preserves config values
  - [x] 5.10 Test `load()` raises `FileNotFoundError` on missing model file
  - [x] 5.11 Test `load()` raises `FileNotFoundError` on missing config file
  - [x] 5.12 Test plugin registration: `get_model("xgboost")` returns `XGBoostModel`
  - [x] 5.13 Test `get_config()` returns the config instance
  - [x] 5.14 Test `fit()` raises `ValueError` on empty DataFrame
  - [x] 5.15 Hypothesis property test: `predict_proba` output is bounded [0, 1] for random feature inputs
  - [x] 5.16 Test early stopping: model with `early_stopping_rounds` stops before `n_estimators` on easy data

- [x] Task 6: Run quality gates (AC: all)
  - [x] 6.1 `ruff check src/ tests/` passes
  - [x] 6.2 `mypy --strict src/ncaa_eval tests` passes
  - [x] 6.3 `pytest` passes with all new tests green and zero regressions

## Dev Notes

### Design Reference

The complete Model ABC interface specification is in `specs/research/modeling-approaches.md`:
- Section 5.2: ABC pseudocode (import-verified across 3 code review rounds)
- Section 5.5: `XGBoostModelConfig` schema with all parameter defaults
- Section 6.2: XGBoost implementation approach
- Section 6.4: Hyperparameter ranges for tuning
- Section 5.7: Persistence format (UBJSON)

### Critical: XGBoostModel Is Stateless — Implements `Model` Directly

`XGBoostModel` implements `Model` directly — there is no `StatelessModel` subclass. Stateless models do not need lifecycle hooks (`start_season`, `get_state`, `set_state`). The only abstract methods to implement are: `fit`, `predict_proba`, `save`, `load`, `get_config`.

This follows the exact same pattern as `LogisticRegressionModel` (Story 5.2 test fixture) — see `src/ncaa_eval/model/logistic_regression.py` as the canonical example.

### Critical: XGBClassifier API Specifics (v3.2.0 installed)

**XGBoost 3.2.0** is installed in the conda environment. Key API details:

1. **`XGBClassifier` constructor** accepts all hyperparameters:
   ```python
   XGBClassifier(
       n_estimators=config.n_estimators,
       max_depth=config.max_depth,
       learning_rate=config.learning_rate,
       subsample=config.subsample,
       colsample_bytree=config.colsample_bytree,
       min_child_weight=config.min_child_weight,
       reg_alpha=config.reg_alpha,
       reg_lambda=config.reg_lambda,
       objective="binary:logistic",
       eval_metric="logloss",
       early_stopping_rounds=config.early_stopping_rounds,
       random_state=42,
       use_label_encoder=False,
   )
   ```

2. **`fit(X, y, eval_set=...)`** — `eval_set` is a list of `(X, y)` tuples for validation. Early stopping uses the last eval set.

3. **`predict_proba(X)`** returns a 2D array `(n_samples, 2)` for binary classification. Use `[:, 1]` for P(class=1).

4. **`save_model(fname)`** — instance method on `XGBClassifier`. Use `.ubj` extension for UBJSON (default since XGBoost 2.1+, stable across versions).

5. **`load_model(fname)`** — **instance method**, NOT a class method. Must instantiate `XGBClassifier()` first, THEN call `clf.load_model(path)`.

6. **Early stopping with `best_iteration`** — When early stopping fires, `predict_proba()` automatically uses `best_iteration` (no manual intervention needed in XGBoost 3.x).

### Critical: Validation Split for Early Stopping

`fit()` must create a validation split for `eval_set`. Strategy:

```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=config.validation_fraction, random_state=42, stratify=y,
)
self._clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
```

- Use `stratify=y` to maintain class balance in both splits
- `verbose=False` to suppress XGBoost training output during normal use
- Default `validation_fraction=0.1` reserves 10% for early stopping monitoring

### Critical: Label Balance Convention

The `StatefulFeatureServer` (Story 4.7) produces feature matrices where `team_a` and `team_b` are assigned based on the game data structure — typically `team_a = w_team_id` (winner). This means `y` may be heavily biased toward 1 (team_a wins).

**Two approaches (document both, implement the simpler one):**
1. **Randomize team assignment** in the feature server before training — then `y` is balanced and `scale_pos_weight=1.0`
2. **Set `scale_pos_weight`** = `count(y==0) / count(y==1)` to compensate for imbalance

For this story, document the convention in the `fit()` docstring and set `scale_pos_weight=1.0` in the config (default). The feature server's team assignment convention will be resolved in Story 5.5 (CLI) or the evaluation pipeline (Epic 6).

### Critical: `save` / `load` File Checking Pattern

From Story 5.3 code review: multi-file `load()` must check ALL expected files exist BEFORE reading any. Pattern:

```python
@classmethod
def load(cls, path: Path) -> Self:
    config_path = path / "config.json"
    model_path = path / "model.ubj"
    if not config_path.exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)
    if not model_path.exists():
        msg = f"Model file not found: {model_path}"
        raise FileNotFoundError(msg)
    config = XGBoostModelConfig.model_validate_json(config_path.read_text())
    instance = cls(config)
    instance._clf.load_model(str(model_path))
    return instance
```

### Critical: Persistence Format (UBJSON)

Use XGBoost's native UBJSON format (`.ubj`), NOT pickle/joblib:
- **Backward compatible** across XGBoost versions (guaranteed by XGBoost docs)
- `save_model("model.ubj")` saves trees + objective — stable
- Pickle saves full training state — NOT stable across versions
- Config is saved separately as JSON via Pydantic `model_dump_json()`

Do NOT use `joblib.dump()` for the XGBoost model (that's for sklearn models like `LogisticRegressionModel`).

### Critical: `use_label_encoder=False` Deprecation

In XGBoost 3.x, `use_label_encoder` parameter is removed (it was deprecated in 2.x). Do NOT pass this parameter. The epics description mentions it but it no longer applies to XGBoost 3.2.0. Check the constructor signature before using — if it raises a `TypeError`, remove the parameter.

### File Structure

```
src/ncaa_eval/model/
├── __init__.py              # Add xgboost_model import for auto-registration
├── base.py                  # Model ABC, StatefulModel (EXISTING — DO NOT MODIFY)
├── registry.py              # Plugin registry (EXISTING — DO NOT MODIFY)
├── logistic_regression.py   # Test fixture (EXISTING — DO NOT MODIFY)
├── elo.py                   # Elo reference (EXISTING — DO NOT MODIFY)
└── xgboost_model.py         # NEW — XGBoostModel, XGBoostModelConfig
```

```
tests/unit/
├── test_model_base.py       # EXISTING — DO NOT MODIFY
├── test_model_registry.py   # EXISTING — DO NOT MODIFY
├── test_model_logistic_regression.py  # EXISTING — DO NOT MODIFY
├── test_model_elo.py        # EXISTING — DO NOT MODIFY
└── test_model_xgboost.py   # NEW — all XGBoostModel tests
```

**Note:** The module is named `xgboost_model.py` (not `xgboost.py`) to avoid shadowing the `xgboost` package import.

### Existing Test Patterns to Follow

See `tests/unit/test_model_logistic_regression.py` (Story 5.2) for the canonical stateless model test pattern:
- `_make_train_data()` helper: generates linearly separable synthetic data with `np.random.default_rng(42)`
- Test config defaults and custom values
- Test `fit` / `predict_proba` with the synthetic data
- Test `save` / `load` round-trip with `tmp_path`
- Test registration via `list_models()`
- Test `get_config()` returns the config instance

For Hypothesis property tests, see `tests/unit/test_model_elo.py` (Story 5.3):
- `@given(st.floats(...))` for bounded-output tests
- `@settings(max_examples=50)` for reasonable test speed

### Dependencies Already Available

| Package | Usage | In pyproject.toml? |
|:---|:---|:---|
| `xgboost` (3.2.0 installed) | `XGBClassifier` | Yes (`xgboost = "*"`) |
| `scikit-learn` | `train_test_split` for validation split | Yes |
| `pydantic` (^2.12.5) | `XGBoostModelConfig` base | Yes |
| `numpy` | Synthetic test data | Yes |
| `pandas` | DataFrame/Series interface | Yes |

No new dependencies needed.

### Existing Codebase Context (DO NOT Reimplement)

| Building Block | Module | Relevant API | Story |
|:---|:---|:---|:---|
| Model ABC | `model.base` | `Model`, `ModelConfig` | 5.2 |
| Plugin registry | `model.registry` | `@register_model`, `get_model`, `list_models` | 5.2 |
| LogisticRegression (pattern) | `model.logistic_regression` | Canonical stateless Model pattern | 5.2 |
| EloModel (stateful pattern) | `model.elo` | Reference for save/load file-checking pattern | 5.3 |

### XGBoostModelConfig Default Values

From `specs/research/modeling-approaches.md` §5.5 and §6.4:

| Parameter | Default | Tuning Range | Notes |
|:---|:---|:---|:---|
| `n_estimators` | 500 | 100–2000 | With early stopping |
| `max_depth` | 5 | 3–8 | Small data → shallow trees |
| `learning_rate` | 0.05 | 0.01–0.3 | Trade off with n_estimators |
| `subsample` | 0.8 | 0.5–1.0 | Row sampling |
| `colsample_bytree` | 0.8 | 0.5–1.0 | Feature sampling |
| `min_child_weight` | 3 | 1–10 | Regularization for small NCAA datasets |
| `reg_alpha` | 0.0 | 0–1.0 | L1 (Lasso) regularization |
| `reg_lambda` | 1.0 | 0.5–5.0 | L2 (Ridge) regularization |
| `early_stopping_rounds` | 50 | 10–100 | Rounds without improvement before stopping |
| `validation_fraction` | 0.1 | 0.05–0.2 | Fraction of training data held out for eval_set |

### Project Conventions (Must Follow)

- `from __future__ import annotations` required in all Python files
- Conventional commits: `feat(model): implement reference XGBoost model — story 5.4`
- `mypy --strict` mandatory
- `Literal["xgboost"]` for `model_name` type (follows `LogisticRegressionConfig` and `EloModelConfig` patterns)
- No `for` loops over DataFrames for metric calculations (NFR1) — XGBoost's vectorized `predict_proba` naturally satisfies this
- Test file naming: `tests/unit/test_model_xgboost.py`

### Project Structure Notes

- New file: `src/ncaa_eval/model/xgboost_model.py`
- New file: `tests/unit/test_model_xgboost.py`
- Modified: `src/ncaa_eval/model/__init__.py` (add xgboost_model import for auto-registration)

### References

- [Source: specs/research/modeling-approaches.md — Section 5.2 (ABC pseudocode), Section 5.5 (XGBoostModelConfig), Section 6.2 (XGBoost implementation approach), Section 6.4 (hyperparameter ranges), Section 5.7 (persistence format)]
- [Source: src/ncaa_eval/model/base.py — Model, ModelConfig (abstract methods to implement)]
- [Source: src/ncaa_eval/model/registry.py — @register_model, get_model, list_models]
- [Source: src/ncaa_eval/model/logistic_regression.py — LogisticRegressionModel pattern to follow exactly]
- [Source: src/ncaa_eval/model/elo.py — EloModel save/load file-checking pattern]
- [Source: _bmad-output/planning-artifacts/epics.md — Story 5.4 AC]
- [Source: _bmad-output/implementation-artifacts/5-3-implement-reference-stateful-model-elo.md — Previous story learnings, code review patterns]
- [Source: _bmad-output/planning-artifacts/template-requirements.md — Accumulated project conventions and code review findings]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

- XGBoost 3.x confirmed: `use_label_encoder` parameter removed — omitted from constructor as specified in Dev Notes
- `xgboost` package provides type stubs — no `type: ignore[import-untyped]` needed (unlike pandas/sklearn)
- Ruff auto-fixed import sorting (merged `from hypothesis import` lines)

### Completion Notes List

- Implemented `XGBoostModelConfig(ModelConfig)` with all 10 hyperparameters matching §5.5/§6.4 defaults
- Implemented `XGBoostModel(Model)` — stateless model wrapping `XGBClassifier` with `binary:logistic` objective
- `fit()` uses `train_test_split` with `stratify=y` for early stopping validation set
- `fit()` guards against empty DataFrame input
- Label balance convention documented in class and method docstrings
- `save()` uses XGBoost native UBJSON format (`model.ubj`) + Pydantic JSON config
- `load()` checks both files exist before reading either (Story 5.3 pattern)
- Plugin registered as `"xgboost"` via `@register_model` decorator
- Auto-registration via `model/__init__.py` import
- 16 unit tests covering all ACs: config, fit/predict, save/load, registration, property-based, early stopping
- All 475 tests pass (zero regressions), ruff clean, mypy --strict clean

### Change Log

- 2026-02-23: Implemented XGBoostModel reference stateless model — all 6 tasks complete, 16 tests added

### File List

- `src/ncaa_eval/model/xgboost_model.py` (NEW) — XGBoostModelConfig, XGBoostModel
- `src/ncaa_eval/model/__init__.py` (MODIFIED) — added xgboost_model import for auto-registration
- `tests/unit/test_model_xgboost.py` (NEW) — 16 unit tests
- `_bmad-output/implementation-artifacts/5-4-implement-reference-stateless-model-xgboost.md` (MODIFIED) — story tracking
- `_bmad-output/implementation-artifacts/sprint-status.yaml` (MODIFIED) — status update
