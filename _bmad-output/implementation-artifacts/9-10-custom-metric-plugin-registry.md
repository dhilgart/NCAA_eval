# Story 9.10: Custom Metric Plugin Registry

Status: review

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a **data scientist**,
I want to **register a custom metric function and have it appear in the metric explorer and leaderboard alongside the built-in metrics**,
so that **I can evaluate models on domain-specific criteria without modifying library source code**.

## Acceptance Criteria

1. **Given** a function decorated with `@register_metric("my_metric")`, **when** the user runs a backtest or opens the metric explorer dashboard, **then** `my_metric` appears alongside `log_loss`, `brier_score`, and the other built-in metrics.

2. **Given** the registry is implemented, **then** the `@register_metric` decorator and `MetricRegistry` lookup functions (`get_metric`, `list_metrics`) are publicly exported from `ncaa_eval.evaluation`.

3. **Given** the registry is implemented, **then** Story 7.9 tutorial `docs/tutorials/custom-metric.md` is updated to accurately document the registry API (replacing the current manual `metric_fns=` dict approach with decorator-based registration).

4. **Given** the registry is implemented, **then** the feature-generator registry is NOT implemented in this story (remains in Post-MVP Backlog).

5. **Given** the four built-in metrics (`log_loss`, `brier_score`, `roc_auc`, `ece`), **then** all four are auto-registered via `@register_metric()` on module import, and `list_metrics()` returns all four by default.

6. **Given** a custom metric is registered and a backtest is run with `metric_fns=None` (default), **then** `run_backtest()` uses all registered metrics (built-in + user-registered) — not just the hardcoded `DEFAULT_METRICS`.

7. **Given** a registered metric name that conflicts with an existing name, **when** `@register_metric("log_loss")` is called a second time, **then** a `ValueError` is raised with a descriptive message (same behavior as `register_model` and `register_scoring`).

8. **Given** a metric name not in the registry, **when** `get_metric("nonexistent")` is called, **then** a `MetricNotFoundError` (subclass of `KeyError`) is raised with available metric names listed.

9. **Given** the dashboard leaderboard page (`1_Lab.py`) and model deep dive page (`3_Model_Deep_Dive.py`), **then** the hardcoded `_METRIC_COLS` lists are replaced with dynamic discovery from the metric registry via `list_metrics()`, so custom metrics appear automatically in both pages.

## Tasks / Subtasks

- [x] Task 1: Add metric registry to `src/ncaa_eval/evaluation/metrics.py` (AC: #2, #5, #7, #8)
  - [x] 1.1: Define `MetricFn` type alias: `Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], float]`
  - [x] 1.2: Add module-level `_METRIC_REGISTRY: dict[str, MetricFn] = {}`
  - [x] 1.3: Add `MetricNotFoundError(KeyError)` exception class
  - [x] 1.4: Implement `register_metric(name: str) -> Callable` decorator (mirrors `register_scoring` pattern from `scoring.py`)
  - [x] 1.5: Implement `get_metric(name: str) -> MetricFn` (raises `MetricNotFoundError`)
  - [x] 1.6: Implement `list_metrics() -> list[str]` (returns sorted names)
  - [x] 1.7: Decorate existing `log_loss`, `brier_score`, `roc_auc`, `expected_calibration_error` with `@register_metric("log_loss")`, `@register_metric("brier_score")`, `@register_metric("roc_auc")`, `@register_metric("ece")`

- [x] Task 2: Wire `run_backtest()` to use the registry (AC: #1, #6)
  - [x] 2.1: In `backtest.py`, change `DEFAULT_METRICS` from hardcoded `MappingProxyType` to a function `default_metrics()` that returns `{name: get_metric(name) for name in list_metrics()}`
  - [x] 2.2: Update `run_backtest()` — when `metric_fns is None`, call `default_metrics()` to get all registered metrics (built-in + user-registered)
  - [x] 2.3: Preserve backward compatibility: `DEFAULT_METRICS` can remain as a module-level constant for existing callers who import it, but document that it only contains built-in metrics (users should prefer the registry)

- [x] Task 3: Export registry API from `evaluation/__init__.py` (AC: #2)
  - [x] 3.1: Add `register_metric`, `get_metric`, `list_metrics`, `MetricNotFoundError`, `MetricFn` to imports and `__all__`

- [x] Task 4: Update dashboard to use dynamic metric discovery (AC: #9)
  - [x] 4.1: In `dashboard/pages/1_Lab.py`, replace hardcoded `_METRIC_COLS` with dynamic `list_metrics()` call (import from `ncaa_eval.evaluation`)
  - [x] 4.2: In `dashboard/pages/3_Model_Deep_Dive.py`, replace hardcoded `_METRIC_COLS` with dynamic `list_metrics()` call
  - [x] 4.3: Handle gracefully when registered metrics are not present in DataFrame columns (custom metrics won't have data for old runs) — filter `list_metrics()` to only columns present in the DataFrame

- [x] Task 5: Write unit tests in `tests/unit/test_metric_registry.py` (AC: #1, #5, #6, #7, #8)
  - [x] 5.1: Test `list_metrics()` returns all 4 built-in metrics on import
  - [x] 5.2: Test `get_metric("log_loss")` returns the `log_loss` function
  - [x] 5.3: Test `get_metric("nonexistent")` raises `MetricNotFoundError`
  - [x] 5.4: Test `@register_metric("custom")` registers and is discoverable via `list_metrics()`
  - [x] 5.5: Test duplicate registration raises `ValueError`
  - [x] 5.6: Test `run_backtest` with `metric_fns=None` uses all registered metrics (including custom ones registered before backtest call)
  - [x] 5.7: Test `run_backtest` with explicit `metric_fns=` dict still works (backward compatibility)

- [x] Task 6: Update tutorial `docs/tutorials/custom-metric.md` (AC: #3)
  - [x] 6.1: Add a new Step 3 (after "Use in a Backtest") showing `@register_metric("my_mae")` decorator usage
  - [x] 6.2: Show that registered metrics appear automatically in `run_backtest()` without passing `metric_fns`
  - [x] 6.3: Show `list_metrics()` output including the custom metric
  - [x] 6.4: Note that the `metric_fns=` dict approach still works for ad-hoc metrics that should not be globally registered
  - [x] 6.5: Add a note clarifying that custom metrics vs. custom scoring rules are different extension mechanisms

## Dev Notes

### Architecture Pattern — Follow `scoring.py` and `model/registry.py` Exactly

The project already has **two proven registry implementations** with identical structure. The metric registry MUST follow this exact pattern:

```
scoring.py (ScoringRule registry)          →  metrics.py (MetricFn registry)
─────────────────────────────────          ─────────────────────────────────
_SCORING_REGISTRY: dict[str, type]         _METRIC_REGISTRY: dict[str, MetricFn]
ScoringNotFoundError(KeyError)             MetricNotFoundError(KeyError)
register_scoring(name) → decorator         register_metric(name) → decorator
get_scoring(name) → type                   get_metric(name) → MetricFn
list_scorings() → list[str]                list_metrics() → list[str]
```

**Key difference:** Scoring and model registries store **classes** (`type[ScoringRule]`, `type[Model]`). The metric registry stores **functions** (`Callable`). This means `register_metric` is a **function decorator**, not a class decorator. The decorator wraps the function, registers it, and returns it unchanged.

### MetricFn Type Alias

```python
from collections.abc import Callable
import numpy.typing as npt
import numpy as np

MetricFn = Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], float]
```

This is the same signature already enforced by `_evaluate_fold()` in `backtest.py` (line 182–185).

### Registry Implementation Pattern

```python
_METRIC_REGISTRY: dict[str, MetricFn] = {}

_MF = TypeVar("_MF", bound=MetricFn)

class MetricNotFoundError(KeyError):
    """Raised when a requested metric name is not in the registry."""

def register_metric(name: str) -> Callable[[_MF], _MF]:
    """Function decorator that registers a metric function."""
    def decorator(fn: _MF) -> _MF:
        if name in _METRIC_REGISTRY:
            msg = f"Metric name {name!r} is already registered"
            raise ValueError(msg)
        _METRIC_REGISTRY[name] = fn
        return fn
    return decorator

def get_metric(name: str) -> MetricFn:
    """Return the metric function registered under *name*."""
    try:
        return _METRIC_REGISTRY[name]
    except KeyError:
        msg = f"No metric registered with name {name!r}. Available: {list_metrics()}"
        raise MetricNotFoundError(msg) from None

def list_metrics() -> list[str]:
    """Return all registered metric names (sorted)."""
    return sorted(_METRIC_REGISTRY)
```

### Built-In Metric Registration

Apply `@register_metric` to the four existing metric functions. The decorator name for `expected_calibration_error` is `"ece"` (matching the existing key in `DEFAULT_METRICS`):

```python
@register_metric("log_loss")
def log_loss(...) -> float: ...

@register_metric("brier_score")
def brier_score(...) -> float: ...

@register_metric("roc_auc")
def roc_auc(...) -> float: ...

@register_metric("ece")
def expected_calibration_error(...) -> float: ...
```

**CRITICAL:** `reliability_diagram_data` is NOT registered because it returns `ReliabilityData`, not `float`. It does not match the `MetricFn` contract.

### Backtest Integration — `DEFAULT_METRICS` Transition

**Current state (`backtest.py`):**
```python
DEFAULT_METRICS: Mapping[str, Callable[...]] = types.MappingProxyType({
    "log_loss": log_loss,
    "brier_score": brier_score,
    "roc_auc": roc_auc,
    "ece": expected_calibration_error,
})

# In run_backtest():
resolved_metrics = dict(DEFAULT_METRICS) if metric_fns is None else dict(metric_fns)
```

**Target state:**
```python
# Keep DEFAULT_METRICS for backward compatibility (callers who import it directly)
DEFAULT_METRICS: Mapping[str, Callable[...]] = types.MappingProxyType({
    "log_loss": log_loss,
    "brier_score": brier_score,
    "roc_auc": roc_auc,
    "ece": expected_calibration_error,
})

def default_metrics() -> dict[str, MetricFn]:
    """Return all registered metric functions (built-in + user-registered)."""
    return {name: get_metric(name) for name in list_metrics()}

# In run_backtest():
resolved_metrics = default_metrics() if metric_fns is None else dict(metric_fns)
```

This way:
- `DEFAULT_METRICS` constant still works for anyone who imports it (backward compat)
- `run_backtest(metric_fns=None)` now picks up custom registered metrics automatically
- `run_backtest(metric_fns={...})` still works for ad-hoc metric dicts

### Dashboard Dynamic Metric Columns

**Current hardcoded pattern (`1_Lab.py` line 14, `3_Model_Deep_Dive.py` line 27):**
```python
_METRIC_COLS = ["log_loss", "brier_score", "roc_auc", "ece"]
```

**Target dynamic pattern:**
```python
from ncaa_eval.evaluation import list_metrics

def _get_metric_cols(df: pd.DataFrame) -> list[str]:
    """Return metric column names that exist in both registry and DataFrame."""
    registered = list_metrics()
    return [m for m in registered if m in df.columns]
```

**Why filter against DataFrame columns:** Old backtest runs (before custom metric registration) won't have the custom metric columns. The dashboard must handle this gracefully — show only metrics that have data.

### Key File Locations

| File | Action |
|------|--------|
| `src/ncaa_eval/evaluation/metrics.py` | **MODIFY** — add registry dict, decorator, getter, lister, error class; decorate built-in metrics |
| `src/ncaa_eval/evaluation/backtest.py` | **MODIFY** — add `default_metrics()` function, update `run_backtest` to use registry when `metric_fns=None` |
| `src/ncaa_eval/evaluation/__init__.py` | **MODIFY** — export `register_metric`, `get_metric`, `list_metrics`, `MetricNotFoundError`, `MetricFn` |
| `dashboard/pages/1_Lab.py` | **MODIFY** — replace hardcoded `_METRIC_COLS` with dynamic `list_metrics()` + DataFrame intersection |
| `dashboard/pages/3_Model_Deep_Dive.py` | **MODIFY** — same as `1_Lab.py` |
| `tests/unit/test_metric_registry.py` | **NEW** — unit tests for registry + backtest integration |
| `docs/tutorials/custom-metric.md` | **MODIFY** — update Part 1 Step 3 with registry usage, add registry verification example |

### Project Structure Notes

- No new directories needed — all changes are in existing `src/ncaa_eval/evaluation/`, `dashboard/pages/`, `tests/unit/`, `docs/tutorials/`
- `from __future__ import annotations` required in all new/modified Python files
- All code must pass `mypy --strict` and `ruff check`
- The `MetricFn` type alias needs careful typing — `Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], float]` must satisfy `mypy --strict`. Consider using `Protocol` if `TypeVar` bound to `Callable` creates issues.

### Test Pattern — Registry Isolation

**CRITICAL:** Tests that call `register_metric()` mutate the global `_METRIC_REGISTRY`. Tests must clean up after themselves to avoid cross-test pollution. Use a fixture or `try/finally` to remove test-registered metrics:

```python
@pytest.fixture()
def _clean_registry():
    """Remove test-registered metrics after each test."""
    from ncaa_eval.evaluation.metrics import _METRIC_REGISTRY
    before = set(_METRIC_REGISTRY.keys())
    yield
    for key in set(_METRIC_REGISTRY.keys()) - before:
        del _METRIC_REGISTRY[key]
```

Or test with a fresh module reload. The fixture approach is cleaner.

### Previous Story Intelligence (9.9)

- Quality gates: 1101 tests passing, ruff clean, mypy --strict clean (101 files)
- Code review patterns from 9.9: watch for dead code from branching, false docstring claims, weak test assertions
- Pattern: always separate pure logic from CLI/UI concerns — the registry is pure logic, dashboard code consumes it
- Pipe-safety was a recurring issue in 9.9 — not directly relevant here, but shows the importance of testing interaction boundaries

### Git Intelligence

Recent commits all follow `feat(scope): description (Story X.Y)` pattern. Stories 9.1–9.9 are done, all merged to main. The scoring registry (Story 6.6) and model registry (Story 5.2) are the most directly relevant precedents — follow their patterns exactly.

### What NOT to Implement

Per AC #4, the **feature-generator registry** is explicitly OUT OF SCOPE. Do not implement `register_feature_generator()` or similar — that remains in the Post-MVP Backlog. Only the metric registry is implemented in this story.

### References

- [Source: `src/ncaa_eval/evaluation/metrics.py` — current metric functions, `_validate_inputs`, `ReliabilityData`]
- [Source: `src/ncaa_eval/evaluation/scoring.py` — proven registry pattern: `_SCORING_REGISTRY`, `register_scoring()`, `get_scoring()`, `list_scorings()`, `ScoringNotFoundError`]
- [Source: `src/ncaa_eval/model/registry.py` — proven registry pattern: `_MODEL_REGISTRY`, `register_model()`, `get_model()`, `list_models()`, `ModelNotFoundError`]
- [Source: `src/ncaa_eval/evaluation/backtest.py` — `DEFAULT_METRICS` constant (line 58–68), `run_backtest()` `metric_fns` parameter (line 263), `_evaluate_fold()` signature (line 179)]
- [Source: `src/ncaa_eval/evaluation/__init__.py` — public API exports for evaluation module]
- [Source: `dashboard/pages/1_Lab.py` — hardcoded `_METRIC_COLS` (line 14), `_DISPLAY_COLS` (line 15)]
- [Source: `dashboard/pages/3_Model_Deep_Dive.py` — hardcoded `_METRIC_COLS` (line 27)]
- [Source: `docs/tutorials/custom-metric.md` — current tutorial with manual `metric_fns=` dict approach]
- [Source: `_bmad-output/planning-artifacts/epics.md#Story 9.10` — acceptance criteria and source audit items]
- [Source: Audit item P3-17; PO decision 2026-03-09 — metric registry only, feature-generator deferred]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

- Fixed ruff C901/PLR0912 in `1_Lab.py` by extracting `_style_metric_table()` helper
- Fixed test collection error in `test_leaderboard_page.py` — test imported removed `_METRIC_COLS` module attribute; updated to use local constant

### Completion Notes List

- Implemented decorator-based metric registry (`register_metric`, `get_metric`, `list_metrics`, `MetricNotFoundError`, `MetricFn`) in `metrics.py`, mirroring the scoring and model registry patterns exactly
- All 4 built-in metrics (`log_loss`, `brier_score`, `roc_auc`, `ece`) auto-registered via `@register_metric()` on module import
- Added `default_metrics()` function in `backtest.py` that returns all registered metrics; `run_backtest(metric_fns=None)` now uses registry-based discovery
- `DEFAULT_METRICS` constant preserved for backward compatibility
- Dashboard pages (`1_Lab.py`, `3_Model_Deep_Dive.py`) now use dynamic `_get_metric_cols(df)` with registry + DataFrame intersection
- 11 new unit tests covering: built-in registration, get/list, duplicate errors, MetricNotFoundError, backtest integration
- Tutorial updated with new Step 3 showing `@register_metric` usage, `list_metrics()` verification, and note distinguishing custom metrics from custom scoring rules
- Quality gates: 1112 tests passing, ruff clean, mypy --strict clean (102 files)

### Change Log

- 2026-03-10: Implemented custom metric plugin registry (Story 9.10)

### File List

- `src/ncaa_eval/evaluation/metrics.py` — MODIFIED (added registry infrastructure + decorators on built-in metrics)
- `src/ncaa_eval/evaluation/backtest.py` — MODIFIED (added `default_metrics()`, updated `run_backtest` to use registry)
- `src/ncaa_eval/evaluation/__init__.py` — MODIFIED (exported `register_metric`, `get_metric`, `list_metrics`, `MetricNotFoundError`, `MetricFn`)
- `dashboard/pages/1_Lab.py` — MODIFIED (replaced hardcoded `_METRIC_COLS`/`_DISPLAY_COLS` with dynamic `_get_metric_cols()`)
- `dashboard/pages/3_Model_Deep_Dive.py` — MODIFIED (replaced hardcoded `_METRIC_COLS` with dynamic `_get_metric_cols()`)
- `tests/unit/test_metric_registry.py` — NEW (11 unit tests for registry + backtest integration)
- `tests/unit/test_leaderboard_page.py` — MODIFIED (updated to use local metric constants instead of removed module attribute)
- `docs/tutorials/custom-metric.md` — MODIFIED (added Step 3 registry usage, updated summary table)
- `_bmad-output/implementation-artifacts/sprint-status.yaml` — MODIFIED (9-10 status: in-progress → review)
- `_bmad-output/implementation-artifacts/9-10-custom-metric-plugin-registry.md` — MODIFIED (tasks marked complete, status → review)
