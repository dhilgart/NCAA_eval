# Story 8.6: Type Safety & Configuration Improvements

Status: done

## Story

As a developer,
I want `FeatureConfig` fields to use `Literal` types instead of bare `str`, duplicated constants to be centralized, and the Fibonacci scoring UI label to display actual point values,
so that type-checkers catch invalid configuration values at development time and users see accurate scoring information.

## Acceptance Criteria

1. `FeatureConfig` fields use `Literal` types: `batch_rating_types`, `ordinal_composite`, `gender_scope`, `calibration_method`
2. `DEFAULT_MARGIN_CAP` centralized in a single shared constants location (not duplicated in `graph.py` and `opponent.py`)
3. Fibonacci scoring UI label displays actual point values (e.g., "Fibonacci (2-3-5-8-13-21)") so users are not misled regardless of which values PO chooses
4. `ruff check .` and `mypy --strict src/ncaa_eval tests` pass

## Tasks / Subtasks

- [x] Task 1: Add `Literal` types to `FeatureConfig` fields (AC: #1, #4)
  - [x] 1.1 Define type aliases for the Literal unions near the top of `feature_serving.py`
  - [x] 1.2 Change `batch_rating_types: tuple[str, ...]` → `tuple[BatchRatingType, ...]` where `BatchRatingType = Literal["srs", "ridge", "colley"]`
  - [x] 1.3 Change `ordinal_composite: str | None` → `Literal["simple_average", "weighted", "pca"] | None`
  - [x] 1.4 Change `gender_scope: str` → `Literal["M", "W"]`
  - [x] 1.5 Change `calibration_method: str | None` → `Literal["isotonic", "sigmoid"] | None`
  - [x] 1.6 Also change `dataset_scope: str` → `Literal["kaggle", "all"]` (same pattern, same docstring-documented values)
  - [x] 1.7 Update all test files that construct `FeatureConfig` to satisfy the new types
  - [x] 1.8 Run `mypy --strict src/ncaa_eval tests` — fix any type errors

- [x] Task 2: Centralize `DEFAULT_MARGIN_CAP` (AC: #2)
  - [x] 2.1 Add `DEFAULT_MARGIN_CAP: int = 25` to `src/ncaa_eval/transform/constants.py` (new file — `__init__.py` would cause circular imports)
  - [x] 2.2 Remove `DEFAULT_MARGIN_CAP` definition from `src/ncaa_eval/transform/graph.py:34`
  - [x] 2.3 Remove `DEFAULT_MARGIN_CAP` definition from `src/ncaa_eval/transform/opponent.py:14`
  - [x] 2.4 Add `from ncaa_eval.transform.constants import DEFAULT_MARGIN_CAP` to both `graph.py` and `opponent.py`
  - [x] 2.5 Verify no other files import `DEFAULT_MARGIN_CAP` from the old locations

- [x] Task 3: Fibonacci scoring UI label with point values (AC: #3)
  - [x] 3.1 Add registry-level `display_name` parameter to `register_scoring()` (per Dev Notes anti-pattern guidance — avoids requiring Protocol property on all implementations)
  - [x] 3.2 Register `FibonacciScoring` with `display_name="Fibonacci (2-3-5-8-13-21)"`
  - [x] 3.3 Register `StandardScoring` with `display_name="Standard (1-2-4-8-16-32)"`
  - [x] 3.4 Add `list_scoring_display_names()` → `dict[str, str]` mapping registry keys to display names
  - [x] 3.5 Update `dashboard/app.py` sidebar selectbox to show `display_name` values via `format_func` parameter
  - [x] 3.6 Update `2_Presentation.py` and `4_Pool_Scorer.py` subheaders/chart titles to use display names

- [x] Task 4: Run quality gates (AC: #4)
  - [x] 4.1 Run `ruff check .` — fix any violations
  - [x] 4.2 Run `mypy --strict src/ncaa_eval tests` — fix any type errors
  - [x] 4.3 Run `pytest` — all tests pass, no regressions

## Dev Notes

### Key Code Locations

| Purpose | File Path | Lines |
|---|---|---|
| FeatureConfig class | `src/ncaa_eval/transform/feature_serving.py` | 50–79 |
| DEFAULT_MARGIN_CAP #1 | `src/ncaa_eval/transform/graph.py` | 34 |
| DEFAULT_MARGIN_CAP #2 | `src/ncaa_eval/transform/opponent.py` | 14 |
| Batch rating dispatch | `src/ncaa_eval/transform/feature_serving.py` | 122–126 |
| ScoringRule Protocol | `src/ncaa_eval/evaluation/scoring.py` | 34–51 |
| FibonacciScoring | `src/ncaa_eval/evaluation/scoring.py` | 129–142 |
| StandardScoring | `src/ncaa_eval/evaluation/scoring.py` | 113–126 |
| Scoring registry | `src/ncaa_eval/evaluation/scoring.py` | 58–105 |
| Dashboard scoring selectbox | `dashboard/app.py` | 83–88 |
| Dashboard data loaders | `dashboard/lib/data_loaders.py` | 197–204 |
| FeatureConfig tests | `tests/unit/test_feature_serving.py` | 41–114 |

### Critical Implementation Details

**FeatureConfig is a frozen `@dataclass`, NOT a Pydantic `BaseModel`.**
- Use `from typing import Literal` — `Literal` types work with dataclasses for mypy enforcement (static check, no runtime validation)
- The class is at `feature_serving.py:50` with `@dataclass(frozen=True)`
- Do NOT convert to Pydantic — that would be scope creep

**Literal tuple element types:**
```python
# For batch_rating_types, each element must be one of the valid values:
from typing import Literal
BatchRatingType = Literal["srs", "ridge", "colley"]
batch_rating_types: tuple[BatchRatingType, ...] = ("srs", "ridge", "colley")
```
This tells mypy that `FeatureConfig(batch_rating_types=("invalid",))` is a type error.

**Valid value sets (extracted from dispatch logic):**
| Field | Valid Values | Source |
|---|---|---|
| `batch_rating_types` | `"srs"`, `"ridge"`, `"colley"` | `_BATCH_RATING_FUNCS` dict in `feature_serving.py:122–126` |
| `ordinal_composite` | `"simple_average"`, `"weighted"`, `"pca"`, `None` | `MasseyOrdinalsStore` methods in `normalization.py` |
| `gender_scope` | `"M"`, `"W"` | Docstring convention — M=men's, W=women's |
| `calibration_method` | `"isotonic"`, `"sigmoid"`, `None` | `IsotonicCalibrator`, `SigmoidCalibrator` in `calibration.py` |
| `dataset_scope` | `"kaggle"`, `"all"` | Docstring convention — kaggle-only vs enriched |

**DEFAULT_MARGIN_CAP centralization:**
Both `graph.py:34` and `opponent.py:14` define `DEFAULT_MARGIN_CAP: int = 25` independently. Move to a single location. Check `src/ncaa_eval/transform/__init__.py` — if it already re-exports public symbols, add the constant there. Otherwise, create `src/ncaa_eval/transform/constants.py`.

**Fibonacci UI label — use `ScoringRule.display_name` property:**
- The `ScoringRule` Protocol at `scoring.py:34` defines `name` property. Adding a `display_name` property keeps backward compatibility — callers that need the registry key still use `name`, UI code uses `display_name`.
- The dashboard sidebar at `dashboard/app.py:88` uses `st.selectbox("Scoring Format", options=scorings, ...)` where `scorings` is a `list[str]` of registry keys. Use `format_func` to display human-readable names.
- `list_scoring_display_names()` can instantiate each registered class and call `display_name` — but be careful: `SeedDiffBonusScoring` requires a `seed_map` arg in `__init__`. Either add a class-level `DISPLAY_NAME` attribute or provide display names in the registry itself.

**Simpler approach for display names — registry-level display names:**
Instead of adding `display_name` to the Protocol (which forces all implementations to add it), add an optional `display_name` parameter to `register_scoring()`:
```python
_SCORING_DISPLAY_NAMES: dict[str, str] = {}

def register_scoring(name: str, *, display_name: str | None = None) -> ...:
    ...
    _SCORING_DISPLAY_NAMES[name] = display_name or name
```
Then `list_scoring_display_names() -> dict[str, str]` reads from this dict. This avoids instantiating scoring objects just for labels.

### Anti-Patterns to Avoid

- Do NOT convert `FeatureConfig` from dataclass to Pydantic BaseModel — out of scope
- Do NOT add runtime validation to the frozen dataclass (`__post_init__`) unless mypy doesn't catch it — `Literal` types are sufficient for AC #1
- Do NOT change the `ScoringRule` Protocol's `name` property to return display names — `name` is used as the registry key internally
- Do NOT add display name support by requiring `ScoringRule` implementations to define `display_name` on instances — `SeedDiffBonusScoring.__init__` takes `seed_map` and cannot be instantiated without args. Use registry-level display names instead.
- Do NOT move constants shared ONLY within `transform/` to a project-wide `src/ncaa_eval/constants.py` — keep them scoped to the transform package

### Previous Story Intelligence

**From Story 8.5:** Registered `unit` marker in `pyproject.toml`. Confirmed `scoring_from_config` has full test coverage. Fibonacci point values are `(2.0, 3.0, 5.0, 8.0, 13.0, 21.0)` — tested by `test_fibonacci_scoring_values`.

**From Story 8.9:** PR template now includes PEP 20, SOLID, and pure-function quality gates. All new code must follow these. Silent exception swallowing is prohibited.

**From Story 8.2:** `ScoringRule` was already given proper `type[ScoringRule]` typing in `_SCORING_REGISTRY`. Any additions to the scoring module should maintain this pattern.

### Project Structure Notes

- `src/ncaa_eval/transform/` is the package for all feature transformation — `FeatureConfig`, `graph.py`, `opponent.py` all live here
- `src/ncaa_eval/evaluation/scoring.py` — scoring registry and implementations
- `dashboard/` is NOT under `mypy --strict` — dashboard changes only need to pass `ruff check .`
- All Python files require `from __future__ import annotations`

### References

- [Source: _bmad-output/planning-artifacts/epic-8-codebase-improvements.md#Story 8.6]
- [Source: _bmad-output/planning-artifacts/codebase-audit-report.md] — Audit items 3.7, 3.9, 1.8
- [Source: docs/STYLE_GUIDE.md] — Type safety patterns, Literal usage
- [Source: src/ncaa_eval/transform/feature_serving.py] — FeatureConfig definition
- [Source: src/ncaa_eval/evaluation/scoring.py] — ScoringRule Protocol, FibonacciScoring

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

### Completion Notes List

- Task 1: Added 5 Literal type aliases (`BatchRatingType`, `OrdinalCompositeMethod`, `GenderScope`, `DatasetScope`, `CalibrationMethod`) to `feature_serving.py`. Updated all 5 `FeatureConfig` fields to use them. Re-exported from `transform/__init__.py`. All existing tests pass without modification — all callsites already used valid literal values. mypy --strict passes.
- Task 2: Created `transform/constants.py` with `DEFAULT_MARGIN_CAP = 25`. Removed duplicate definitions from `graph.py` and `opponent.py`, replaced with imports from `constants.py`. Re-exported from `transform/__init__.py`. Used `constants.py` instead of `__init__.py` directly to avoid circular imports.
- Task 3: Added registry-level `display_name` parameter to `register_scoring()` decorator. StandardScoring registered with "Standard (1-2-4-8-16-32)", FibonacciScoring with "Fibonacci (2-3-5-8-13-21)". Added `list_scoring_display_names()`. Updated dashboard sidebar selectbox via `format_func`, plus `2_Presentation.py` and `4_Pool_Scorer.py` to use display names in subheaders/chart titles. Added 6 unit tests.

### Change Log

- 2026-03-04: Task 1 — Added Literal types to FeatureConfig fields
- 2026-03-04: Task 2 — Centralized DEFAULT_MARGIN_CAP in transform/constants.py
- 2026-03-04: Task 3 — Added scoring display names with point values
- 2026-03-04: Task 4 — Quality gates passed (ruff, mypy --strict, pytest 912/912)
- 2026-03-04: Code review fixes — Literal mode type propagated to splitter/backtest/train; misleading parameter name fixed in 4_Pool_Scorer.py; new tests for Literal aliases and DEFAULT_MARGIN_CAP centralization; quality gates re-passed (ruff clean, mypy strict clean, 912/912 tests pass)
- 2026-03-04: Adversarial code review — removed dead `_VALID_MODES` constant from backtest.py; added explanatory comments to Literal+runtime guard coexistence; quality gates pass (ruff clean, mypy strict clean, 922/922 tests pass)

### File List

- `src/ncaa_eval/transform/feature_serving.py` (modified — added Literal type aliases + updated FeatureConfig fields; serve_season_features mode now Literal["batch","stateful"])
- `src/ncaa_eval/transform/__init__.py` (modified — re-exported new type aliases + DEFAULT_MARGIN_CAP)
- `src/ncaa_eval/transform/constants.py` (new — centralized DEFAULT_MARGIN_CAP)
- `src/ncaa_eval/transform/graph.py` (modified — imports DEFAULT_MARGIN_CAP from constants)
- `src/ncaa_eval/transform/opponent.py` (modified — imports DEFAULT_MARGIN_CAP from constants)
- `src/ncaa_eval/evaluation/scoring.py` (modified — added display_name to register_scoring + list_scoring_display_names)
- `src/ncaa_eval/evaluation/__init__.py` (modified — re-exported list_scoring_display_names)
- `src/ncaa_eval/evaluation/splitter.py` (modified — mode parameter now Literal["batch","stateful"])
- `src/ncaa_eval/evaluation/backtest.py` (modified — mode parameter now Literal["batch","stateful"])
- `src/ncaa_eval/cli/train.py` (modified — Literal annotations on mode variables)
- `dashboard/app.py` (modified — selectbox uses format_func for display names)
- `dashboard/lib/data_loaders.py` (modified — added load_scoring_display_names)
- `dashboard/pages/2_Presentation.py` (modified — subheaders/chart titles use display names)
- `dashboard/pages/4_Pool_Scorer.py` (modified — renamed scoring_label param to scoring_registry_key)
- `tests/unit/test_evaluation_simulation.py` (modified — added TestScoringDisplayNames with 6 tests)
- `tests/unit/test_feature_serving.py` (modified — added TestFeatureConfigLiteralAliases + TestDefaultMarginCap)
- `tests/unit/test_evaluation_splitter.py` (modified — type: ignore on invalid mode test)
- `tests/unit/test_evaluation_backtest.py` (modified — type: ignore on invalid mode test)
