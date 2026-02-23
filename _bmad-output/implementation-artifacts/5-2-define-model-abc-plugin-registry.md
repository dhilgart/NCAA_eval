# Story 5.2: Define Model ABC & Plugin Registry

Status: ready-for-dev

## Story

As a data scientist,
I want an abstract base class (`Model`) with a plugin-registry architecture,
So that I can implement custom models that plug into the training and evaluation pipeline without modifying core code.

## Acceptance Criteria

1. **Model ABC** — `Model` ABC enforces implementation of `fit(X, y)`, `predict_proba(X)`, `save(path)`, `load(path)`, and `get_config()` abstract methods.
2. **fit interface** — `fit(X: pd.DataFrame, y: pd.Series) -> None` is the unified training interface for all model types (sklearn naming convention).
3. **predict_proba interface** — `predict_proba(X: pd.DataFrame) -> pd.Series` returns calibrated P(team_a wins) in [0.0, 1.0] for each row in X (unified for stateful and stateless).
4. **load returns Self** — `load(cls, path: Path) -> Self` is a classmethod returning `Self` (PEP 673 / Python 3.12) so that `EloModel.load(path)` is typed as `EloModel`, not `Model`.
5. **get_config** — `get_config() -> ModelConfig` returns the Pydantic-validated config; `ModelConfig` (Pydantic BaseModel) is the base class for all model configs.
6. **StatefulModel subclass** — `StatefulModel(Model)` is defined with: (1) concrete template `fit()` that reconstructs `Game` objects from X and calls `update()` per game; (2) concrete template `predict_proba()` that dispatches to `_predict_one()` per row; (3) abstract hooks `update(game: Game)`, `_predict_one(team_a_id, team_b_id)`, `start_season(season)`, `get_state()`, `set_state(state)`.
7. **No StatelessModel subclass** — Stateless models (XGBoost, logistic regression) implement `Model` directly.
8. **Plugin registry** — `@register_model("name")` decorator, `get_model(name) -> type[Model]`, and `list_models() -> list[str]`; built-in models auto-register on package import; external users register via `@register_model` before invoking the pipeline.
9. **LogisticRegression test fixture** — A minimal logistic regression implementation (`LogisticRegressionModel(Model)`) is included as a test fixture (not production) demonstrating the stateless `Model` contract in ~30 lines.
10. **Tests** — The ABC and registry are covered by unit tests including the logistic regression test fixture.
11. **mypy --strict** — Type annotations satisfy `mypy --strict`.

## Tasks / Subtasks

- [x] Task 1: Create `ModelConfig` base class (AC: #5)
  - [x] 1.1 Define `ModelConfig(PydanticBaseModel)` with `model_name: str` in `src/ncaa_eval/model/base.py`
  - [x] 1.2 Verify Pydantic v2 JSON serialization round-trip

- [x] Task 2: Create `Model` ABC (AC: #1, #2, #3, #4)
  - [x] 2.1 Define `Model(ABC)` with abstract methods: `fit`, `predict_proba`, `save`, `load`, `get_config`
  - [x] 2.2 Ensure `load` is `@classmethod @abstractmethod` returning `Self`
  - [x] 2.3 Verify `mypy --strict` passes on the ABC definition

- [x] Task 3: Create `StatefulModel` subclass (AC: #6)
  - [x] 3.1 Implement concrete `fit(X, y)` template method — reconstruct `Game` objects from X columns, iterate sequentially calling `start_season()` at season boundaries and `update()` per game
  - [x] 3.2 Implement concrete `_to_games(X, y) -> list[Game]` — map DataFrame columns to `Game` fields; this is a concrete method on `StatefulModel`, NOT abstract
  - [x] 3.3 Implement concrete `predict_proba(X)` template — iterate rows with `itertuples()`, call `_predict_one(team_a_id, team_b_id)` per row
  - [x] 3.4 Define abstract hooks: `_predict_one`, `update`, `start_season`, `get_state`, `set_state`

- [x] Task 4: Create plugin registry (AC: #8)
  - [x] 4.1 Implement `_MODEL_REGISTRY` dict, `@register_model("name")` decorator, `get_model(name)`, `list_models()` in `src/ncaa_eval/model/registry.py`
  - [x] 4.2 Implement `ModelNotFoundError` exception for unknown model names
  - [x] 4.3 Ensure auto-registration on package import via `__init__.py` imports

- [x] Task 5: Create LogisticRegression test fixture (AC: #9)
  - [x] 5.1 Implement `LogisticRegressionModel(Model)` wrapping `sklearn.linear_model.LogisticRegression` in ~30 lines
  - [x] 5.2 Implement `LogisticRegressionConfig(ModelConfig)` with minimal hyperparameters
  - [x] 5.3 Register as `@register_model("logistic_regression")`
  - [x] 5.4 Place in `src/ncaa_eval/model/logistic_regression.py`
  - [x] 5.5 Implement `save`/`load` using `joblib.dump()`/`joblib.load()` + config JSON

- [x] Task 6: Update `model/__init__.py` exports (AC: #8)
  - [x] 6.1 Export: `Model`, `StatefulModel`, `ModelConfig`, `register_model`, `get_model`, `list_models`
  - [x] 6.2 Import `LogisticRegressionModel` for auto-registration

- [ ] Task 7: Write unit tests (AC: #10)
  - [ ] 7.1 Test Model ABC cannot be instantiated directly
  - [ ] 7.2 Test StatefulModel ABC cannot be instantiated without implementing all abstract methods
  - [ ] 7.3 Test `_to_games()` correctly reconstructs `Game` objects from DataFrame
  - [ ] 7.4 Test `StatefulModel.fit()` calls `start_season` at boundaries and `update` per game
  - [ ] 7.5 Test `StatefulModel.predict_proba()` dispatches to `_predict_one` per row
  - [ ] 7.6 Test plugin registry: `register_model`, `get_model`, `list_models`, `ModelNotFoundError`
  - [ ] 7.7 Test LogisticRegressionModel: `fit`/`predict_proba`/`save`/`load` round-trip
  - [ ] 7.8 Test `ModelConfig` Pydantic validation and JSON serialization

- [ ] Task 8: Run quality gates (AC: #11)
  - [ ] 8.1 `ruff check .` passes
  - [ ] 8.2 `mypy --strict src/ncaa_eval tests` passes
  - [ ] 8.3 `pytest` passes with all new tests green

## Dev Notes

### Design Reference

The complete Model ABC interface specification is in `specs/research/modeling-approaches.md` Section 5 — this was import-verified across 3 rounds of adversarial code review. **Use that pseudocode as the implementation starting point.**

### Existing ABC Patterns to Follow

Two ABC patterns already exist in the codebase. Follow them for consistency:

| Pattern | File | Key Traits |
|:---|:---|:---|
| `Repository` ABC | `src/ncaa_eval/ingest/repository.py` | `abc.ABC`, abstract methods, concrete `ParquetRepository` in same package |
| `Connector` ABC | `src/ncaa_eval/ingest/connectors/base.py` | `abc.ABC`, abstract methods, concrete implementations in sibling modules, exception hierarchy |

### File Structure

```
src/ncaa_eval/model/
├── __init__.py          # Public exports + auto-registration imports
├── base.py              # Model ABC, StatefulModel, ModelConfig
├── registry.py          # @register_model, get_model, list_models, ModelNotFoundError
└── logistic_regression.py  # Test fixture: LogisticRegressionModel
```

Story 5.3 will add `elo.py`, Story 5.4 will add `xgboost_model.py` — do NOT create these files now.

### Critical: `_to_games()` Column Mapping

`StatefulModel.fit(X, y)` receives the feature DataFrame from `StatefulFeatureServer`. The `_to_games()` method must reconstruct `Game` objects from X. The required columns in X are:

| X Column | Game Field | Notes |
|:---|:---|:---|
| `team_a_id` | Maps to `w_team_id` or `l_team_id` | Determined by `y` (label) — if `team_a_won == True`, `team_a_id` is `w_team_id` |
| `team_b_id` | Maps to the other team ID | |
| `season` | `season` | Always present in feature DataFrame |
| `day_num` | `day_num` | Always present |
| `date` | `date` | Always present (derived from day_num for Kaggle games) |
| `loc_encoding` | `loc` | Numeric in X (+1/−1/0); reconstruct H/A/N for Game |
| `game_id` | `game_id` | Always present |
| `is_tournament` | `is_tournament` | Always present |

The `y` Series contains `team_a_won` (bool/int). To reconstruct winner/loser:
- If `team_a_won == True`: `w_team_id = team_a_id`, `l_team_id = team_b_id`
- If `team_a_won == False`: `w_team_id = team_b_id`, `l_team_id = team_a_id`

**Score columns**: The feature DataFrame may or may not contain raw scores. If scores are not available in X, use dummy scores (e.g., `w_score=1, l_score=0`) since `StatefulModel.update()` subclass implementations can choose whether they need scores (Elo does; other stateful models might not). Document this convention clearly.

### Critical: `predict_proba` Uses `itertuples()`

`StatefulModel.predict_proba()` uses `itertuples()` to call `_predict_one()` per row. This is acceptable for stateful models because:
1. Stateful predictions are inherently per-row (look up in-memory ratings by team ID)
2. Tournament prediction is ~63 games — iteration overhead is negligible
3. The "Vectorization First" rule applies to metric calculations (NFR1), not to per-game state lookups

### Critical: Plugin Registry Auto-Registration

Built-in models must auto-register when `import ncaa_eval.model` is executed. Implementation pattern:

```python
# src/ncaa_eval/model/__init__.py
from ncaa_eval.model.base import Model, ModelConfig, StatefulModel
from ncaa_eval.model.registry import get_model, list_models, register_model

# Auto-register built-in models (import triggers @register_model decorator)
from ncaa_eval.model import logistic_regression as _lr  # noqa: F401
```

### Critical: `load()` as `@classmethod @abstractmethod` Returning `Self`

The `load` method must be BOTH a classmethod AND abstract. Python 3.12 supports `typing.Self` natively. The ordering of decorators matters:

```python
@classmethod
@abstractmethod
def load(cls, path: Path) -> Self:
    ...
```

This ensures `EloModel.load(path)` returns type `EloModel` (not `Model`), enabling proper type narrowing.

### Critical: No `predict()` Method

The Model ABC does NOT define a `predict()` method (no label prediction). Only `predict_proba(X) -> pd.Series` exists. This is intentional — NCAA evaluation metrics (Brier Score, LogLoss) operate on probabilities, not labels.

### Critical: LogisticRegression Test Fixture — Not Production

The `LogisticRegressionModel` is a TEST FIXTURE demonstrating the stateless `Model` contract. It:
- Wraps `sklearn.linear_model.LogisticRegression` with L2 penalty
- Uses `joblib` for model persistence
- Has minimal config: `C` (regularization strength), `max_iter`
- Is NOT a production reference model (that role belongs to XGBoost in Story 5.4)
- Should be ~30 lines of implementation code

### Dependencies Already Available

| Package | Usage | Already in pyproject.toml? |
|:---|:---|:---|
| `pydantic` (^2.12.5) | `ModelConfig` base class | Yes |
| `pandas` | `fit(X, y)`, `predict_proba(X)` signatures | Yes |
| `scikit-learn` | `LogisticRegression` (test fixture), `joblib` | Yes |
| `joblib` | Model serialization for sklearn-based models | Yes |
| `xgboost` | NOT used in this story (Story 5.4) | Yes (for later) |

### Existing Codebase Context (DO NOT Reimplement)

| Building Block | Module | Relevant API | Story |
|:---|:---|:---|:---|
| Game schema | `ingest.schema` | `Game(BaseModel)` — Pydantic v2 with validators | 2.2 |
| Elo feature engine | `transform.elo` | `EloFeatureEngine`, `EloConfig` — Story 5.3 wraps this | 4.8 |
| Feature serving | `transform.feature_serving` | `StatefulFeatureServer.serve_season_features()` → DataFrame | 4.7 |
| Calibration | `transform.calibration` | `IsotonicCalibrator`, `SigmoidCalibrator` | 4.7 |

### `Game` Model Fields (from `ingest/schema.py`)

```python
class Game(BaseModel):
    game_id: str
    season: int          # >= 1985
    day_num: int         # >= 0
    date: datetime.date | None
    w_team_id: int       # >= 1
    l_team_id: int       # >= 1
    w_score: int         # >= 0, must be > l_score
    l_score: int         # >= 0
    loc: Literal["H", "A", "N"]
    num_ot: int          # >= 0, default 0
    is_tournament: bool  # default False
```

### Project Structure Notes

- New files in `src/ncaa_eval/model/`: `base.py`, `registry.py`, `logistic_regression.py`
- New test files in `tests/`: `tests/unit/model/` directory with `test_base.py`, `test_registry.py`, `test_logistic_regression.py` (or similar structure following existing test layout)
- Modified: `src/ncaa_eval/model/__init__.py` (currently has only `from __future__ import annotations`)

### References

- [Source: specs/research/modeling-approaches.md — Section 5 (Model ABC interface spec, import-verified)]
- [Source: specs/research/modeling-approaches.md — Section 5.4 (Plugin registry requirements)]
- [Source: specs/research/modeling-approaches.md — Section 5.5 (ModelConfig schema)]
- [Source: specs/05-architecture-fullstack.md — FR6 (Model ABC), NFR3 (Plugin Registry), Strategy Pattern]
- [Source: _bmad-output/planning-artifacts/epics.md — Story 5.2 AC]
- [Source: src/ncaa_eval/ingest/repository.py — Repository ABC pattern]
- [Source: src/ncaa_eval/ingest/connectors/base.py — Connector ABC pattern]
- [Source: src/ncaa_eval/ingest/schema.py — Game Pydantic model]
- [Source: src/ncaa_eval/transform/elo.py — EloConfig, EloFeatureEngine (Story 5.3 will wrap)]
- [Source: src/ncaa_eval/transform/feature_serving.py — StatefulFeatureServer output columns]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

### Completion Notes List

- Tasks 1-3: Created `ModelConfig`, `Model` ABC, and `StatefulModel` in `base.py`. All 13 unit tests pass covering: config validation/serialization, ABC enforcement, `_to_games()` reconstruction, `fit()` season boundaries, `predict_proba()` dispatch, and `get_state`/`set_state` round-trip.
- Task 4: Created plugin registry with `@register_model` decorator, `get_model()`, `list_models()`, `ModelNotFoundError`. 5 registry tests pass including duplicate name detection and unknown model error.
- Task 5: Created `LogisticRegressionModel` test fixture wrapping sklearn LR in ~30 lines with `LogisticRegressionConfig(C, max_iter)`, `save`/`load` via joblib+JSON, `@register_model("logistic_regression")`. 6 tests pass.
- Task 6: Updated `__init__.py` with public exports (`Model`, `StatefulModel`, `ModelConfig`, `register_model`, `get_model`, `list_models`, `ModelNotFoundError`) and auto-registration import. Verified `list_models()` returns `["logistic_regression"]` on package import.

### File List

- `src/ncaa_eval/model/base.py` (new)
- `src/ncaa_eval/model/registry.py` (new)
- `src/ncaa_eval/model/logistic_regression.py` (new)
- `tests/unit/test_model_base.py` (new)
- `tests/unit/test_model_registry.py` (new)
- `tests/unit/test_model_logistic_regression.py` (new)
- `src/ncaa_eval/model/__init__.py` (modified)
