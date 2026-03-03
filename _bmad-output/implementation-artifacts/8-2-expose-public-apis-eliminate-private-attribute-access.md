# Story 8.2: Expose Public APIs & Eliminate Private Attribute Access

Status: review

## Story

As a developer,
I want all cross-module interactions to use public methods and constants instead of accessing private (`_`-prefixed) attributes,
so that the codebase has explicit API contracts, is safe to refactor, and passes `mypy --strict` without implicit coupling.

## Acceptance Criteria

### AC1: EloFeatureEngine Public API Expansion

1. `EloFeatureEngine` in `src/ncaa_eval/transform/elo.py` exposes a new public method `has_ratings() -> bool` that returns `True` if `_ratings` is non-empty
2. `EloFeatureEngine` exposes `set_ratings(ratings: dict[int, float]) -> None` to replace direct `_ratings` assignment
3. `EloFeatureEngine` exposes `set_game_counts(counts: dict[int, int]) -> None` to replace direct `_game_counts` assignment
4. `EloFeatureEngine` exposes `get_game_counts() -> dict[int, int]` returning a copy (mirrors existing `get_all_ratings()` pattern)
5. `EloFeatureEngine` exposes `predict_matchup(team_a_id: int, team_b_id: int) -> float` that returns P(team_a wins) via the Elo expected-score formula — this is the public equivalent of the current `_predict_one` private method

### AC2: Feature Serving Uses Public API

6. `src/ncaa_eval/transform/feature_serving.py:301` uses `self._elo_engine.has_ratings()` instead of accessing `self._elo_engine._ratings` directly

### AC3: EloModel Uses Public Setters/Getters

7. `src/ncaa_eval/model/elo.py` `get_state()` uses `self._engine.get_game_counts()` instead of `dict(self._engine._game_counts)`
8. `src/ncaa_eval/model/elo.py` `set_state()` uses `self._engine.set_ratings(...)` and `self._engine.set_game_counts(...)` instead of direct `_ratings` / `_game_counts` assignment

### AC4: StatefulModel Public Prediction Hook

9. `StatefulModel` ABC in `src/ncaa_eval/model/base.py` exposes a public `predict_matchup(team_a_id: int, team_b_id: int) -> float` method that delegates to the existing abstract `_predict_one` hook — this gives external consumers a public API without breaking the internal hook contract
10. `src/ncaa_eval/evaluation/providers.py` `EloProvider` uses `self._model.predict_matchup(...)` instead of `self._model._predict_one(...)` (including the `hasattr` check in `__init__`)

### AC5: Model Feature Importances Public API

11. `Model` ABC in `src/ncaa_eval/model/base.py` provides a default `get_feature_importances() -> list[tuple[str, float]] | None` method returning `None` (base behavior: no importances available)
12. `XGBoostModel` in `src/ncaa_eval/model/xgboost_model.py` overrides `get_feature_importances()` to return feature names + importance values from `self._clf.feature_importances_` (requires storing `feature_names` during `fit()`)
13. `dashboard/lib/data_loaders.py` `load_feature_importances()` uses `model.get_feature_importances()` instead of `getattr(model, "_clf", None)`

### AC6: Public Constant for No-Tournament Seasons

14. `src/ncaa_eval/transform/serving.py` renames `_NO_TOURNAMENT_SEASONS` to `NO_TOURNAMENT_SEASONS` (public constant)
15. `src/ncaa_eval/evaluation/splitter.py` imports `NO_TOURNAMENT_SEASONS` (public name)
16. `src/ncaa_eval/transform/__init__.py` exports `NO_TOURNAMENT_SEASONS` if it currently exports the private version

### AC7: Calibrator Protocol

17. A `Calibrator` Protocol (or ABC) is created in `src/ncaa_eval/transform/calibration.py` defining `fit(probs, outcomes)` and `transform(probs)` as the common interface
18. `IsotonicCalibrator` and `SigmoidCalibrator` explicitly implement the `Calibrator` protocol
19. Any code that accepts calibrator instances uses the `Calibrator` type annotation instead of `IsotonicCalibrator | SigmoidCalibrator`

### AC8: MatrixProvider Public Accessor

20. `MatrixProvider` in `src/ncaa_eval/evaluation/providers.py` exposes a public `matchup_probability(team_a_id: int, team_b_id: int) -> float` method (may already exist — verify and add if missing)

### AC9: Test Updates

21. `tests/unit/test_dashboard_filters.py` assertions on `result._P[i, j]` replaced with `result.matchup_probability(team_a_id, team_b_id)` calls
22. `tests/unit/test_model_elo.py` tests calling `model._predict_one(...)` updated to use `model.predict_matchup(...)`
23. `tests/unit/test_model_elo.py` tests assigning `model._engine._ratings[...]` updated to use `model._engine.set_ratings(...)` (or test via the public `set_state()` / `get_state()` round-trip)
24. `tests/unit/test_model_elo.py` tests assigning `model._engine._game_counts[...]` updated to use `model._engine.set_game_counts(...)`

### AC10: Quality Gates

25. `ruff check .` passes (no new violations)
26. `mypy --strict src/ncaa_eval tests` passes
27. All existing tests pass (import path / API changes only — no behavioral changes)
28. No behavioral changes — same inputs produce same outputs; this is a pure API surface refactoring

## Tasks / Subtasks

- [x] Task 1: Expand EloFeatureEngine public API (AC: #1-5)
  - [x] 1.1 Add `has_ratings() -> bool` method to `EloFeatureEngine`
  - [x] 1.2 Add `set_ratings(ratings: dict[int, float]) -> None` method
  - [x] 1.3 Add `set_game_counts(counts: dict[int, int]) -> None` method
  - [x] 1.4 Add `get_game_counts() -> dict[int, int]` method (return copy like `get_all_ratings()`)
  - [x] 1.5 Add `predict_matchup(team_a_id: int, team_b_id: int) -> float` public method wrapping the expected-score formula currently in `_predict_one`
  - [x] 1.6 Run `ruff check src/ncaa_eval/transform/elo.py` and `mypy --strict src/ncaa_eval/transform/elo.py`

- [x] Task 2: Update feature_serving.py to use public API (AC: #6)
  - [x] 2.1 Replace `self._elo_engine._ratings` check with `self._elo_engine.has_ratings()` in `_serve_stateful()`
  - [x] 2.2 Run `mypy --strict src/ncaa_eval/transform/feature_serving.py`

- [x] Task 3: Update model/elo.py to use public setters/getters (AC: #7-8)
  - [x] 3.1 In `get_state()`, replace `dict(self._engine._game_counts)` with `self._engine.get_game_counts()`
  - [x] 3.2 In `set_state()`, replace `self._engine._ratings = {...}` with `self._engine.set_ratings({...})`
  - [x] 3.3 In `set_state()`, replace `self._engine._game_counts = {...}` with `self._engine.set_game_counts({...})`
  - [x] 3.4 Run `mypy --strict src/ncaa_eval/model/elo.py`

- [x] Task 4: Add public predict_matchup to StatefulModel ABC (AC: #9-10)
  - [x] 4.1 Add concrete `predict_matchup(team_a_id: int, team_b_id: int) -> float` method to `StatefulModel` that delegates to `self._predict_one(team_a_id, team_b_id)`
  - [x] 4.2 In `providers.py` `EloProvider.__init__`, replace `hasattr(model, "_predict_one")` check with `hasattr(model, "predict_matchup")`
  - [x] 4.3 In `providers.py` `EloProvider.matchup_probability` and `batch_matchup_probabilities`, replace `self._model._predict_one(...)` with `self._model.predict_matchup(...)`
  - [x] 4.4 Run `mypy --strict src/ncaa_eval/model/base.py src/ncaa_eval/evaluation/providers.py`

- [x] Task 5: Add get_feature_importances to Model ABC (AC: #11-13)
  - [x] 5.1 Add default `get_feature_importances() -> list[tuple[str, float]] | None` returning `None` to `Model` ABC in `base.py`
  - [x] 5.2 In `XGBoostModel.fit()`, store `self._feature_names = list(X.columns)` before training
  - [x] 5.3 Override `get_feature_importances()` in `XGBoostModel` to return `list(zip(self._feature_names, self._clf.feature_importances_))` if fitted, else `None`
  - [x] 5.4 Update `dashboard/lib/data_loaders.py` `load_feature_importances()` to call `model.get_feature_importances()` first, with legacy fallback
  - [x] 5.5 Run `mypy --strict src/ncaa_eval/model/base.py src/ncaa_eval/model/xgboost_model.py`

- [x] Task 6: Rename _NO_TOURNAMENT_SEASONS to public (AC: #14-16)
  - [x] 6.1 In `serving.py`, rename `_NO_TOURNAMENT_SEASONS` to `NO_TOURNAMENT_SEASONS`
  - [x] 6.2 Update all references in `serving.py` itself
  - [x] 6.3 Update import in `splitter.py`
  - [x] 6.4 Update `transform/__init__.py` to re-export `NO_TOURNAMENT_SEASONS`
  - [x] 6.5 Update test references in `test_evaluation_splitter.py`
  - [x] 6.6 Run `ruff check .` and `mypy --strict src/ncaa_eval tests`

- [x] Task 7: Create Calibrator Protocol (AC: #17-19)
  - [x] 7.1 Define `Calibrator` Protocol in `calibration.py` with `fit()` and `transform()` signatures
  - [x] 7.2 Verify `IsotonicCalibrator` and `SigmoidCalibrator` structurally satisfy the protocol (confirmed)
  - [x] 7.3 Export `Calibrator` from `transform/__init__.py`
  - [x] 7.4 Run `mypy --strict src/ncaa_eval/transform/calibration.py`

- [x] Task 8: MatrixProvider public accessor (AC: #20)
  - [x] 8.1 Verified `MatrixProvider` already has `matchup_probability()` from `ProbabilityProvider` protocol — no changes needed
  - [x] 8.2 Verified the public method returns equivalent value to `self._P[i, j]` access
  - [x] 8.3 Run `mypy --strict src/ncaa_eval/evaluation/providers.py`

- [x] Task 9: Update test files (AC: #21-24)
  - [x] 9.1 In `test_dashboard_filters.py`, replaced `result._P[i, j]` assertions with `result.matchup_probability(team_a, team_b, ctx)`
  - [x] 9.2 In `test_model_elo.py`, replaced `model._predict_one(...)` calls with `model.predict_matchup(...)`
  - [x] 9.3 In `test_model_elo.py`, replaced direct `model._engine._ratings[...]` assignments with `model._engine.set_ratings({...})`
  - [x] 9.4 In `test_model_elo.py`, replaced direct `model._engine._game_counts[...]` assignments with `model._engine.set_game_counts({...})`
  - [x] 9.5 In `test_evaluation_simulation.py`, updated `TestEloProvider` FakeElo classes to use `predict_matchup` instead of `_predict_one`
  - [x] 9.6 In `test_dashboard_filters.py`, updated `test_returns_sorted_importances` mock to use `get_feature_importances()` return value
  - [x] 9.7 Run `pytest tests/unit/test_model_elo.py tests/unit/test_dashboard_filters.py tests/unit/test_evaluation_simulation.py -x`

- [x] Task 10: Final validation (AC: #25-28)
  - [x] 10.1 `ruff check .` — zero new violations
  - [x] 10.2 `mypy --strict src/ncaa_eval tests` — zero errors (85 source files)
  - [x] 10.3 `pytest` — 865 passed, 1 skipped
  - [x] 10.4 Verified no behavioral changes — pure API surface refactoring

## Dev Notes

### Key Principle

This story follows **Pattern B** from the codebase audit ("Private API Leakage Across Module Boundaries"). The goal is to convert every cross-module `_`-prefixed access into a public method call, creating explicit API contracts that `mypy --strict` can enforce.

**This is a pure refactoring story** — no behavioral changes. The same inputs produce the same outputs. The only change is that what was implicit (reaching into internals) becomes explicit (calling public methods).

### Architecture Patterns and Constraints

- **`from __future__ import annotations`** required in ALL Python files (Ruff-enforced)
- **Google-style docstrings** — not NumPy-style
- **`mypy --strict`** mandatory for `src/ncaa_eval/` and `tests/`
- **Dashboard files are NOT under mypy strict** — but the `data_loaders.py` changes should still use proper types
- **Backward compatibility**: Private methods like `_predict_one` should NOT be removed — they remain as the internal hook. The public method delegates to them.

### EloFeatureEngine — Current State

**File:** `src/ncaa_eval/transform/elo.py` (lines 73–296)

**Existing public methods (keep as-is):**
- `get_rating(team_id: int) -> float` — returns single team rating
- `get_all_ratings() -> dict[int, float]` — returns copy of ratings dict
- `reset_game_counts() -> None` — resets game counts
- `process_season(games, season)` — orchestrates full season
- `update_game(game)` — updates ratings for a single game
- `start_new_season(season)` — applies mean reversion

**Private attributes being accessed cross-module:**
- `_ratings: dict[int, float]` — accessed by `feature_serving.py:301` (truthiness check) and `model/elo.py:115` (assignment)
- `_game_counts: dict[int, int]` — accessed by `model/elo.py:78` (read) and `model/elo.py:116` (assignment)
- `_predict_one(team_a_id, team_b_id)` — This is NOT on `EloFeatureEngine`; it is on `EloModel`. However, the public `predict_matchup()` on `EloFeatureEngine` should wrap the expected-score formula that `EloModel._predict_one()` currently implements.

**Important distinction**: The `_predict_one` accessed in `providers.py` is on `EloModel` (a `StatefulModel` subclass), NOT on `EloFeatureEngine`. The fix is to add `predict_matchup()` as a concrete method on `StatefulModel` that delegates to `_predict_one()`.

### StatefulModel._predict_one Architecture

**File:** `src/ncaa_eval/model/base.py`

`_predict_one(team_a_id: int, team_b_id: int) -> float` is an **abstract hook** on `StatefulModel`. It is the internal contract that subclasses implement. Making it public would expose the hook contract to external consumers, which is undesirable.

**Correct design:**
```python
class StatefulModel(Model):
    # Public API for external consumers:
    def predict_matchup(self, team_a_id: int, team_b_id: int) -> float:
        """Return P(team_a wins) for a single matchup."""
        return self._predict_one(team_a_id, team_b_id)

    # Internal hook for subclasses:
    @abstractmethod
    def _predict_one(self, team_a_id: int, team_b_id: int) -> float: ...
```

This preserves the Template Method pattern: `_predict_one` remains the subclass hook, `predict_matchup` is the public API.

### EloProvider — Current Private Access

**File:** `src/ncaa_eval/evaluation/providers.py` (lines 117–143)

```python
# Current (line 117-120):
if not hasattr(model, "_predict_one"):
    msg = "model must have a _predict_one(team_a_id, team_b_id) method"
    raise TypeError(msg)

# Current (line 129):
result: float = self._model._predict_one(team_a_id, team_b_id)
```

**Fix:** Replace `_predict_one` with `predict_matchup` throughout.

### Dashboard Feature Importances — Current Private Access

**File:** `dashboard/lib/data_loaders.py` (line 189)

```python
# Current:
clf = getattr(model, "_clf", None)
importances = getattr(clf, "feature_importances_", None)
```

**Fix:** Model ABC provides `get_feature_importances()` with a default `None` return. `XGBoostModel` overrides it. `LogisticRegressionModel` (test fixture) could also override it (`.coef_`), but this is optional — only implement for models where it's clearly useful.

**Note on `_feature_names`:** `XGBoostModel.fit()` currently receives `X: pd.DataFrame`, so `X.columns` is available. Store `self._feature_names: list[str] = list(X.columns)` during fit. Include feature names in `save()`/`load()` round-trip — write to `config.json` or a separate `feature_names.json`.

### Calibrator Protocol Design

**File:** `src/ncaa_eval/transform/calibration.py`

Both `IsotonicCalibrator` and `SigmoidCalibrator` share:
- `fit(probs: npt.NDArray[np.float64], outcomes: npt.NDArray[np.float64]) -> None`
- `transform(probs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]`

Use a `Protocol` (not ABC) since both classes already structurally conform:

```python
class Calibrator(Protocol):
    def fit(self, probs: npt.NDArray[np.float64], outcomes: npt.NDArray[np.float64]) -> None: ...
    def transform(self, probs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
```

This matches the project's convention: ABCs for the major contracts (Model, Repository), Protocols for lightweight structural typing.

### MatrixProvider — _P Access

**File:** `src/ncaa_eval/evaluation/providers.py`

`MatrixProvider` stores `self._P` (the probability matrix). Check if it already has a `matchup_probability()` method (from the `ProbabilityProvider` protocol). If so, tests should use that. If not, add one.

### NO_TOURNAMENT_SEASONS Rename

**File:** `src/ncaa_eval/transform/serving.py:28`

```python
# Current:
_NO_TOURNAMENT_SEASONS: frozenset[int] = frozenset({2020})

# Fix:
NO_TOURNAMENT_SEASONS: frozenset[int] = frozenset({2020})
```

Check `transform/__init__.py` for re-exports. Also check `tests/unit/test_evaluation_splitter.py` for references.

### Previous Story Learnings (Story 8.1)

- **Backward compatibility via re-exports**: Story 8.1 used re-exports extensively when splitting modules. Same principle applies here — private methods should continue to exist (as internal hooks), just with new public methods wrapping them.
- **mypy strict**: Dashboard files are NOT under mypy strict. Don't add `# type: ignore` to fix dashboard type issues — they are pre-existing.
- **Pre-commit hooks**: `debug-statements`, `check-yaml`, `ruff`, `ruff-format` all run. The `template/` directory is excluded.
- **Story 8.1 review follow-up** (line 100 of 8-1 story): `EloProvider` accessing `model._predict_one` was explicitly called out as "Pre-existing pattern — not a regression from this story. Track under Story 8.2." This story resolves that exact item.

### Git Intelligence

Recent commits show Story 8.1 just completed:
- `6f96429` — Story 8.1: decompose simulation.py, filters.py, and run_training() God Function
- `219d29a` — Add PEP 20, SOLID & pure function gates to PR template + codebase audit

The codebase is freshly refactored. The new module locations from Story 8.1 are:
- `src/ncaa_eval/evaluation/providers.py` — where `EloProvider._predict_one` access lives (moved from `simulation.py`)
- `src/ncaa_eval/evaluation/scoring.py` — `_SCORING_REGISTRY` is already typed as `dict[str, type[ScoringRule]]` (fixed in Story 8.1, confirmed by explore agent)
- `dashboard/lib/data_loaders.py` — where `_clf` access lives (moved from `filters.py`)

### Scoring Registry — Already Fixed

The codebase audit finding 3.10 (scoring registry uses untyped `dict[str, type]`) was already resolved in Story 8.1. The registry in `scoring.py:60` is now `dict[str, type[ScoringRule]]`. **No work needed for this AC.**

### Files to Modify

| File | Changes |
|------|---------|
| `src/ncaa_eval/transform/elo.py` | Add `has_ratings()`, `set_ratings()`, `set_game_counts()`, `get_game_counts()`, `predict_matchup()` |
| `src/ncaa_eval/transform/feature_serving.py` | Replace `_ratings` access with `has_ratings()` |
| `src/ncaa_eval/transform/serving.py` | Rename `_NO_TOURNAMENT_SEASONS` → `NO_TOURNAMENT_SEASONS` |
| `src/ncaa_eval/transform/calibration.py` | Add `Calibrator` Protocol |
| `src/ncaa_eval/transform/__init__.py` | Update re-export if applicable |
| `src/ncaa_eval/model/base.py` | Add `predict_matchup()` to `StatefulModel`, `get_feature_importances()` to `Model` |
| `src/ncaa_eval/model/elo.py` | Use `set_ratings()`, `set_game_counts()`, `get_game_counts()` |
| `src/ncaa_eval/model/xgboost_model.py` | Override `get_feature_importances()`, store `_feature_names` |
| `src/ncaa_eval/evaluation/providers.py` | Use `predict_matchup()` instead of `_predict_one()` |
| `src/ncaa_eval/evaluation/splitter.py` | Import `NO_TOURNAMENT_SEASONS` (public name) |
| `dashboard/lib/data_loaders.py` | Use `model.get_feature_importances()` |
| `tests/unit/test_model_elo.py` | Use public APIs in all test assertions |
| `tests/unit/test_dashboard_filters.py` | Replace `._P` access with `matchup_probability()` |
| `tests/unit/test_evaluation_splitter.py` | Update import to public constant name |

### Source Document References

- [Source: `_bmad-output/planning-artifacts/codebase-audit-report.md` — Finding 3.4 (EloFeatureEngine private access x3)]
- [Source: `_bmad-output/planning-artifacts/codebase-audit-report.md` — Finding 3.10 (scoring registry — already fixed in 8.1)]
- [Source: `_bmad-output/planning-artifacts/codebase-audit-report.md` — Finding 3.11 (private constant import)]
- [Source: `_bmad-output/planning-artifacts/codebase-audit-report.md` — Finding 3.13 (dashboard _clf access)]
- [Source: `_bmad-output/planning-artifacts/codebase-audit-report.md` — Finding 3.15 (no calibrator ABC)]
- [Source: `_bmad-output/planning-artifacts/codebase-audit-report.md` — Finding 3.25 (tests access private ._P)]
- [Source: `_bmad-output/planning-artifacts/codebase-audit-pass2-addendum.md` — Pattern B (Private API Leakage Across Module Boundaries)]
- [Source: `_bmad-output/planning-artifacts/epic-8-codebase-improvements.md` — Story 8.2 section]
- [Source: `_bmad-output/implementation-artifacts/8-1-code-architecture-cleanup-simulation-module-split.md` — Review Follow-up: EloProvider._predict_one tracked for Story 8.2]
- [Source: `_bmad-output/planning-artifacts/template-requirements.md` — Defensive Private Attribute Access in Tests]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6 (claude-opus-4-6)

### Debug Log References

None — clean implementation with no blocking issues.

### Completion Notes List

- AC #20 (MatrixProvider public accessor): Already satisfied — `MatrixProvider.matchup_probability()` exists from the `ProbabilityProvider` protocol. No code changes needed.
- AC #13 (dashboard feature importances): Kept legacy `getattr(model, "_clf")` fallback for backward compatibility with saved model runs that predate `_feature_names` storage. Consolidated return paths to satisfy PLR0911 (max 6 return statements).
- AC #19 (Calibrator protocol usage): No code currently accepts `IsotonicCalibrator | SigmoidCalibrator` as a union type annotation — both calibrator classes already structurally satisfy the new `Calibrator` Protocol without any annotation changes needed elsewhere.
- Test file `test_evaluation_simulation.py` also needed updates — `TestEloProvider` FakeElo classes used `_predict_one` which was not caught in the original AC list. Fixed by renaming to `predict_matchup`.

### Change Log

| Change | Reason |
|--------|--------|
| Added 5 public methods to `EloFeatureEngine` | Replace cross-module `_ratings`/`_game_counts` access (AC #1-5) |
| `feature_serving.py` uses `has_ratings()` | Eliminate `_ratings` truthiness check (AC #6) |
| `model/elo.py` uses public setters/getters | Eliminate `_ratings`/`_game_counts` assignment (AC #7-8) |
| `StatefulModel.predict_matchup()` delegates to `_predict_one()` | Template Method: public API + internal hook (AC #9) |
| `EloProvider` uses `predict_matchup` | Eliminate `_predict_one` cross-module access (AC #10) |
| `Model.get_feature_importances()` default method | Base returns `None`, XGBoost overrides (AC #11-12) |
| `data_loaders.py` uses `get_feature_importances()` | Eliminate `_clf` access with legacy fallback (AC #13) |
| Renamed `_NO_TOURNAMENT_SEASONS` → `NO_TOURNAMENT_SEASONS` | Public constant (AC #14-16) |
| Added `Calibrator` Protocol | Structural typing for calibrator interface (AC #17-19) |
| Updated all test files to use public APIs | Eliminate private access in tests (AC #21-24) |

### File List

| File | Action |
|------|--------|
| `src/ncaa_eval/transform/elo.py` | Modified — added 5 public methods |
| `src/ncaa_eval/transform/feature_serving.py` | Modified — `has_ratings()` call |
| `src/ncaa_eval/transform/serving.py` | Modified — renamed constant to public |
| `src/ncaa_eval/transform/calibration.py` | Modified — added `Calibrator` Protocol |
| `src/ncaa_eval/transform/__init__.py` | Modified — exports `NO_TOURNAMENT_SEASONS`, `Calibrator` |
| `src/ncaa_eval/model/base.py` | Modified — added `predict_matchup()`, `get_feature_importances()` |
| `src/ncaa_eval/model/elo.py` | Modified — uses public setters/getters |
| `src/ncaa_eval/model/xgboost_model.py` | Modified — `_feature_names`, `get_feature_importances()` override |
| `src/ncaa_eval/evaluation/providers.py` | Modified — uses `predict_matchup` |
| `src/ncaa_eval/evaluation/splitter.py` | Modified — imports public constant |
| `dashboard/lib/data_loaders.py` | Modified — uses `get_feature_importances()` with legacy fallback |
| `tests/unit/test_model_elo.py` | Modified — uses public APIs |
| `tests/unit/test_dashboard_filters.py` | Modified — uses `matchup_probability()`, updated mock |
| `tests/unit/test_evaluation_simulation.py` | Modified — FakeElo uses `predict_matchup` |
| `tests/unit/test_evaluation_splitter.py` | Modified — updated comment |
| `_bmad-output/implementation-artifacts/8-2-expose-public-apis-eliminate-private-attribute-access.md` | Modified — story status/tasks |
| `_bmad-output/implementation-artifacts/sprint-status.yaml` | Modified — status in-progress → review |
