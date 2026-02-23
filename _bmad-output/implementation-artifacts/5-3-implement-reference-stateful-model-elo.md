# Story 5.3: Implement Reference Stateful Model (Elo)

Status: done

## Story

As a data scientist,
I want a working Elo rating system as the reference stateful model,
So that I have a proven baseline for tournament prediction and a template for building other stateful models.

## Acceptance Criteria

1. **EloModel wraps EloFeatureEngine** — `EloModel(StatefulModel)` wraps `EloFeatureEngine` from `transform.elo` — it does NOT re-implement Elo from scratch; `fit(X, y)` is inherited from `StatefulModel` (calls `update()` per reconstructed game).
2. **update delegates** — `update(game: Game)` delegates to `EloFeatureEngine.update_game()` to advance ratings.
3. **start_season delegates** — `start_season(season: int)` delegates to `EloFeatureEngine.start_new_season(season)` for mean reversion.
4. **_predict_one** — `_predict_one(team_a_id: int, team_b_id: int) -> float` returns P(team_a wins) via the Elo expected-score formula using current ratings; public prediction is via inherited `predict_proba(X: pd.DataFrame) -> pd.Series`.
5. **EloModelConfig** — `EloModelConfig(ModelConfig)` is the Pydantic config with parameters: `initial_rating`, `k_early`, `early_game_threshold`, `k_regular`, `k_tournament`, `margin_exponent`, `max_margin`, `home_advantage_elo`, `mean_reversion_fraction` — defaults matching `EloConfig` from Story 4.8.
6. **get_state / set_state** — `get_state() -> dict[str, Any]` returns a snapshot dict with `"ratings"` (dict[int, float]) and `"game_counts"` (dict[int, int]) keys; `set_state(state)` validates the dict structure and restores both.
7. **save / load** — `save(path: Path)` JSON-dumps ratings dict + config; `load(cls, path: Path) -> Self` reconstructs from JSON.
8. **Plugin registration** — The model registers via the plugin registry as `"elo"`.
9. **Validation** — The Elo model is validated against known rating calculations on a small fixture dataset.
10. **Tests** — The model is covered by unit tests for rating updates, `_predict_one`, state persistence (`get_state`/`set_state`), and `save`/`load` round-trip.

## Tasks / Subtasks

- [x] Task 1: Create `EloModelConfig` (AC: #5)
  - [x] 1.1 Define `EloModelConfig(ModelConfig)` with all Elo parameters matching `EloConfig` defaults
  - [x] 1.2 Use `Literal["elo"]` for `model_name` field
  - [x] 1.3 Verify Pydantic JSON round-trip

- [x] Task 2: Create `EloModel(StatefulModel)` (AC: #1, #2, #3, #4, #6)
  - [x] 2.1 Implement `__init__(config: EloModelConfig | None = None)` — instantiate `EloFeatureEngine` from config
  - [x] 2.2 Implement `update(game: Game)` — delegate to `EloFeatureEngine.update_game()` extracting args from Game fields
  - [x] 2.3 Implement `start_season(season: int)` — delegate to `EloFeatureEngine.start_new_season(season)`
  - [x] 2.4 Implement `_predict_one(team_a_id, team_b_id) -> float` — call `EloFeatureEngine.expected_score(r_a, r_b)` with current ratings
  - [x] 2.5 Implement `get_state() -> dict[str, Any]` — return `{"ratings": engine.get_all_ratings(), "game_counts": engine._game_counts}`
  - [x] 2.6 Implement `set_state(state)` — restore ratings and game counts from dict
  - [x] 2.7 Implement `get_config() -> EloModelConfig`

- [x] Task 3: Implement `save` / `load` (AC: #7)
  - [x] 3.1 `save(path)` — create directory, JSON-dump ratings dict to `path / "state.json"`, config to `path / "config.json"`; include game_counts in state.json
  - [x] 3.2 `load(cls, path) -> Self` — read config.json, instantiate EloModel(config), read state.json, call `set_state()`; return Self

- [x] Task 4: Register as plugin (AC: #8)
  - [x] 4.1 Add `@register_model("elo")` decorator to `EloModel`
  - [x] 4.2 Update `model/__init__.py` to import `elo` module for auto-registration

- [x] Task 5: Write unit tests (AC: #9, #10)
  - [x] 5.1 Test `EloModelConfig` creation and JSON round-trip
  - [x] 5.2 Test `EloModel.update(game)` delegates to engine and changes ratings
  - [x] 5.3 Test `EloModel._predict_one()` returns correct probability based on rating difference
  - [x] 5.4 Test `EloModel.start_season()` triggers mean reversion via engine
  - [x] 5.5 Test `get_state()` / `set_state()` round-trip
  - [x] 5.6 Test `save()` / `load()` round-trip (file-system test with tmp_path)
  - [x] 5.7 Test full `fit()` → `predict_proba()` end-to-end with known fixture data
  - [x] 5.8 Test plugin registration: `get_model("elo")` returns `EloModel`
  - [x] 5.9 Test known rating calculation: verify specific numeric outcomes on a small fixture dataset

- [x] Task 6: Run quality gates (AC: all)
  - [x] 6.1 `ruff check src/ tests/` passes
  - [x] 6.2 `mypy --strict src/ncaa_eval tests` passes
  - [x] 6.3 `pytest` passes with all new tests green and zero regressions

## Dev Notes

### Design Reference

The complete Model ABC interface specification is in `specs/research/modeling-approaches.md`:
- Section 5.2: ABC pseudocode (import-verified across 3 code review rounds)
- Section 5.5: `EloModelConfig` schema with all parameter defaults
- Section 6.1: Elo implementation approach
- Section 6.4: Hyperparameter ranges for tuning

### Critical: Wrap `EloFeatureEngine` — Do NOT Reimplement Elo

`EloModel` is a thin wrapper around `EloFeatureEngine` (Story 4.8, `src/ncaa_eval/transform/elo.py`). It does NOT recompute Elo from scratch. The engine already implements:
- `update_game(w_team_id, l_team_id, w_score, l_score, loc, is_tournament, num_ot)` — returns `(elo_w_before, elo_l_before)`
- `expected_score(rating_a, rating_b)` — static method, logistic formula
- `start_new_season(season)` — applies mean reversion + resets game counts
- `get_all_ratings()` — returns `dict[int, float]` copy
- `get_rating(team_id)` — returns single rating (initial_rating if unseen)
- `_game_counts: dict[int, int]` — per-team game count (affects variable K)

The `EloModel` delegates all Elo math to the engine. The model class adds:
1. `StatefulModel` ABC conformance (inherits concrete `fit()` and `predict_proba()` template methods)
2. `EloModelConfig` Pydantic config (mirrors `EloConfig` dataclass fields)
3. JSON persistence (`save`/`load`)
4. Plugin registration (`@register_model("elo")`)

### Critical: `EloConfig` (dataclass) vs `EloModelConfig` (Pydantic)

`EloFeatureEngine` uses `EloConfig` (a frozen dataclass from `transform.elo`). `EloModel` uses `EloModelConfig` (a Pydantic `ModelConfig` subclass from `model/`). These have identical fields and defaults but are distinct types:

| Field | `EloConfig` default | `EloModelConfig` default | Must match? |
|:---|:---|:---|:---|
| `initial_rating` | 1500.0 | 1500.0 | Yes |
| `k_early` | 56.0 | 56.0 | Yes |
| `k_regular` | 38.0 | 38.0 | Yes |
| `k_tournament` | 47.5 | 47.5 | Yes |
| `early_game_threshold` | 20 | 20 | Yes |
| `margin_exponent` | 0.85 | 0.85 | Yes |
| `max_margin` | 25 | 25 | Yes |
| `home_advantage_elo` | 3.5 | 3.5 | Yes |
| `mean_reversion_fraction` | 0.25 | 0.25 | Yes |

`EloModel.__init__` must convert `EloModelConfig` → `EloConfig` to construct the engine. Pattern:

```python
EloConfig(
    initial_rating=config.initial_rating,
    k_early=config.k_early,
    # ... all fields
)
```

### Critical: `update(game)` Argument Mapping

`StatefulModel.fit(X, y)` reconstructs `Game` objects via `_to_games()` (see `base.py:107-169`) and calls `update(game)` per game. `EloModel.update()` must extract engine arguments from the `Game`:

```python
def update(self, game: Game) -> None:
    self._engine.update_game(
        w_team_id=game.w_team_id,
        l_team_id=game.l_team_id,
        w_score=game.w_score,
        l_score=game.l_score,
        loc=game.loc,
        is_tournament=game.is_tournament,
        num_ot=game.num_ot,
    )
```

Note: `update_game()` returns `(elo_w_before, elo_l_before)` but `update()` returns `None` per the ABC. The return value is discarded — it's only used when computing features (Story 4.8), not when fitting a model.

### Critical: `_predict_one` Uses `expected_score`

```python
def _predict_one(self, team_a_id: int, team_b_id: int) -> float:
    r_a = self._engine.get_rating(team_a_id)
    r_b = self._engine.get_rating(team_b_id)
    return EloFeatureEngine.expected_score(r_a, r_b)
```

This returns P(team_a wins) directly — no calibration needed because the logistic Elo formula IS a calibrated probability estimate.

### Critical: `save` / `load` Format

Save to a directory at `path`:
- `path/config.json` — `EloModelConfig.model_dump_json()`
- `path/state.json` — `json.dumps({"ratings": {str(k): v for k, v in ratings.items()}, "game_counts": {str(k): v for k, v in counts.items()}})`

JSON keys must be strings (JSON spec), so team_id ints need `str()` conversion on save and `int()` conversion on load.

Load:
1. Read `config.json` → `EloModelConfig.model_validate_json(text)`
2. Construct `EloModel(config)`
3. Read `state.json` → parse ratings dict, convert keys back to `int`
4. Call `set_state({"ratings": ratings, "game_counts": game_counts})`
5. Return `Self`

### Critical: `ConferenceLookup` Is Optional

`EloFeatureEngine.__init__` takes an optional `conference_lookup: ConferenceLookup | None`. When `None`, mean reversion uses global mean instead of per-conference mean. For the `EloModel`, pass `None` — conference-aware mean reversion is a feature-engineering concern (Story 4.8), not a model concern. The model just needs valid ratings.

If future use cases require conference-aware reversion in the model, `EloModelConfig` can add a `use_conference_reversion: bool` field — but NOT in this story.

### File Structure

```
src/ncaa_eval/model/
├── __init__.py              # Add elo import for auto-registration
├── base.py                  # Model ABC, StatefulModel (EXISTING — DO NOT MODIFY)
├── registry.py              # Plugin registry (EXISTING — DO NOT MODIFY)
├── logistic_regression.py   # Test fixture (EXISTING — DO NOT MODIFY)
└── elo.py                   # NEW — EloModel, EloModelConfig
```

```
tests/unit/
├── test_model_base.py       # EXISTING — DO NOT MODIFY
├── test_model_registry.py   # EXISTING — DO NOT MODIFY
├── test_model_logistic_regression.py  # EXISTING — DO NOT MODIFY
└── test_model_elo.py        # NEW — all EloModel tests
```

### Existing Test Patterns to Follow

Look at `tests/unit/test_model_logistic_regression.py` (Story 5.2) for the pattern:
- Test config creation and JSON round-trip
- Test `fit` / `predict_proba` with minimal synthetic data
- Test `save` / `load` round-trip with `tmp_path`
- Test registration via `get_model("name")`

Look at `tests/unit/test_model_base.py` for `_make_game_dataframe()` helper — reuse it or create a similar fixture for Elo-specific tests.

### Known Numeric Test Case

For a simple 2-game fixture, verify exact numeric outcomes:

```
Initial rating: 1500
Game 1: Team A (home) beats Team B, score 80-70
  - Effective R_A = 1500 - 3.5 = 1496.5 (home deflation)
  - Expected_A = 1 / (1 + 10^((1500 - 1496.5)/400)) ≈ 0.4980
  - Margin = 10, K = 56 (early), mult = 10^0.85 ≈ 7.079
  - K_eff = 56 × 7.079 ≈ 396.44
  - r_A_new = 1500 + 396.44 × (1 - 0.498) ≈ 1698.94
```

Use this kind of hand-calculated example to validate the model's `_predict_one` and `update` outputs.

### Dependencies Already Available

| Package | Usage | In pyproject.toml? |
|:---|:---|:---|
| `pydantic` (^2.12.5) | `EloModelConfig` base | Yes |
| `json` (stdlib) | Save/load persistence | N/A |
| `transform.elo` | `EloFeatureEngine`, `EloConfig` | Internal |

No new dependencies needed.

### Existing Codebase Context (DO NOT Reimplement)

| Building Block | Module | Relevant API | Story |
|:---|:---|:---|:---|
| Elo feature engine | `transform.elo` | `EloFeatureEngine`, `EloConfig`, `expected_score()` | 4.8 |
| Model ABC | `model.base` | `Model`, `StatefulModel`, `ModelConfig` | 5.2 |
| Plugin registry | `model.registry` | `@register_model`, `get_model`, `list_models` | 5.2 |
| Game schema | `ingest.schema` | `Game(BaseModel)` with validators | 2.2 |
| LogisticRegression (pattern) | `model.logistic_regression` | Example of stateless Model contract | 5.2 |

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

### `EloFeatureEngine` API Summary (from `transform/elo.py`)

```python
class EloFeatureEngine:
    def __init__(self, config: EloConfig, conference_lookup: ConferenceLookup | None = None) -> None: ...
    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float: ...
    def get_rating(self, team_id: int) -> float: ...
    def update_game(self, w_team_id, l_team_id, w_score, l_score, loc, is_tournament, *, num_ot=0) -> tuple[float, float]: ...
    def start_new_season(self, season: int) -> None: ...
    def get_all_ratings(self) -> dict[int, float]: ...
    # Internal: _game_counts: dict[int, int], _ratings: dict[int, float]
```

### `StatefulModel` Template Methods (from `model/base.py`)

- `fit(X, y)` — concrete: calls `_to_games(X, y)`, iterates games, calls `start_season()` at boundaries, calls `update()` per game
- `predict_proba(X)` — concrete: calls `_predict_one(team_a_id, team_b_id)` per row via `itertuples()`
- `_to_games(X, y)` — concrete: reconstructs `Game` objects from DataFrame columns (`team_a_id`, `team_b_id`, `season`, `day_num`, `date`, `loc_encoding`, `game_id`, `is_tournament`, optional `w_score`/`l_score`/`num_ot`)

The developer inherits all three and only implements the abstract hooks: `update`, `_predict_one`, `start_season`, `get_state`, `set_state`, `save`, `load`, `get_config`.

### Project Conventions (Must Follow)

- `from __future__ import annotations` required in all Python files
- Conventional commits: `feat(model): implement reference Elo model — story 5.3`
- `mypy --strict` mandatory
- `Literal["elo"]` for `model_name` type (follows `LogisticRegressionConfig` pattern)
- No `for` loops over DataFrames for metric calculations (NFR1) — not applicable here since Elo updates are inherently per-game
- Test file naming: `tests/unit/test_model_elo.py`

### Project Structure Notes

- New file: `src/ncaa_eval/model/elo.py`
- New file: `tests/unit/test_model_elo.py`
- Modified: `src/ncaa_eval/model/__init__.py` (add elo import for auto-registration)

### References

- [Source: specs/research/modeling-approaches.md — Section 5.2 (ABC pseudocode), Section 5.5 (EloModelConfig), Section 6.1 (Elo implementation approach), Section 6.4 (hyperparameter ranges)]
- [Source: src/ncaa_eval/transform/elo.py — EloConfig, EloFeatureEngine API]
- [Source: src/ncaa_eval/model/base.py — Model, StatefulModel, ModelConfig, _to_games()]
- [Source: src/ncaa_eval/model/registry.py — @register_model, get_model, list_models]
- [Source: src/ncaa_eval/model/logistic_regression.py — LogisticRegressionModel pattern to follow]
- [Source: src/ncaa_eval/ingest/schema.py — Game Pydantic model fields]
- [Source: _bmad-output/planning-artifacts/epics.md — Story 5.3 AC]
- [Source: _bmad-output/implementation-artifacts/5-2-define-model-abc-plugin-registry.md — Previous story learnings, file structure, test patterns]
- [Source: _bmad-output/implementation-artifacts/5-1-research-modeling-approaches.md — Spike findings, Elo feature vs model distinction]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

- Minor ruff import-sorting fixes in `__init__.py` and test file (auto-fixed)
- Removed unnecessary `type: ignore[attr-defined]` on `_to_games` call (inherited from StatefulModel, visible to mypy)

### Completion Notes List

- `EloModelConfig(ModelConfig)` with `Literal["elo"]` model_name and all 9 Elo parameters matching `EloConfig` defaults
- `EloModel(StatefulModel)` — thin wrapper delegating all Elo math to `EloFeatureEngine`; converts `EloModelConfig` → `EloConfig` in `__init__`, passes `conference_lookup=None`
- `update(game)` extracts 7 fields from `Game` and delegates to `engine.update_game()`
- `start_season(season)` delegates to `engine.start_new_season(season)`
- `_predict_one(team_a_id, team_b_id)` calls `engine.get_rating()` for both teams then `EloFeatureEngine.expected_score()`
- `get_state()`/`set_state()` snapshot/restore both `_ratings` and `_game_counts` dicts
- `save(path)` writes `config.json` + `state.json` with string-key JSON; `load(path)` reconstructs with int-key conversion
- `@register_model("elo")` decorator + `__init__.py` import for auto-registration
- 25 unit tests covering: config defaults/custom/round-trip, update delegation, _predict_one correctness, start_season mean reversion, get/set state round-trip, save/load file-system round-trip, fit→predict_proba end-to-end, plugin registration, known numeric calculations, home advantage verification
- All 454 tests pass (30 new + 424 existing), ruff clean, mypy --strict clean
- Code review fixes applied: set_state() input validation with KeyError/TypeError on malformed state; load() FileNotFoundError with clear message on partial saves; _to_elo_config() uses dataclasses.fields to auto-map EloConfig fields (future-proof); set_state() private-access comment added; fit()-twice accumulation documented and tested

### Change Log

- 2026-02-23: Implemented EloModel reference stateful model (Story 5.3) — all ACs satisfied, 25 tests added
- 2026-02-23: Code review fixes — 5 issues fixed: set_state() validation (HIGH), fit() accumulation documented+tested (MEDIUM), load() partial-save guard (MEDIUM), set_state() private-access comment (MEDIUM), _to_elo_config() future-proofed via dataclasses.fields (MEDIUM); 5 new tests added (30 total); 454 tests pass

### File List

- `src/ncaa_eval/model/elo.py` (NEW)
- `src/ncaa_eval/model/__init__.py` (MODIFIED — added elo import)
- `tests/unit/test_model_elo.py` (NEW)
- `_bmad-output/implementation-artifacts/5-3-implement-reference-stateful-model-elo.md` (MODIFIED — task completion)
- `_bmad-output/implementation-artifacts/sprint-status.yaml` (MODIFIED — status update)
