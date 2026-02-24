# Story 6.3: Implement Parallel Cross-Validation Execution

Status: ready-for-dev

## Story

As a data scientist,
I want cross-validation folds and model evaluations to run in parallel via joblib,
So that multi-year backtests complete faster by utilizing all available CPU cores.

## Acceptance Criteria

1. **Given** the CV splitter (Story 6.2) generates multiple folds, **When** the developer runs a parallelized backtest, **Then** independent CV folds execute concurrently using `joblib.Parallel`.
2. The number of parallel workers is configurable (default: all cores via `n_jobs=-1`).
3. Progress is reported during parallel execution (fold completion, elapsed time).
4. Results from all folds are collected and aggregated into a summary DataFrame.
5. The 10-year Elo backtest (training & inference) completes in under 60 seconds per the PRD performance target.
6. Parallel execution produces identical results to sequential execution (determinism).
7. Parallel CV is covered by integration tests comparing parallel vs. sequential results.

## Tasks / Subtasks

- [ ] Task 1: Create `src/ncaa_eval/evaluation/backtest.py` (AC: #1–#6)
  - [ ] 1.1 Define `FoldResult` frozen dataclass: `year: int`, `predictions: pd.Series`, `actuals: pd.Series`, `metrics: dict[str, float]`, `elapsed_seconds: float`
  - [ ] 1.2 Define `BacktestResult` frozen dataclass: `fold_results: tuple[FoldResult, ...]`, `summary: pd.DataFrame`, `elapsed_seconds: float`
  - [ ] 1.3 Implement `_evaluate_fold(fold: CVFold, model: Model, metric_fns: ...) -> FoldResult` — the single-fold worker function
  - [ ] 1.4 Implement `run_backtest(model, feature_server, *, seasons, mode, n_jobs, metric_fns, console) -> BacktestResult` — the main parallel orchestrator
  - [ ] 1.5 Collect all `CVFold` objects eagerly (materialize the generator) before dispatching to joblib
  - [ ] 1.6 Deep-copy the model for each fold to avoid shared-state corruption (stateful models)
  - [ ] 1.7 Aggregate fold results into a summary DataFrame (year as index, metric columns + elapsed_seconds)
  - [ ] 1.8 Report progress via Rich console (fold completion count, per-fold timing)
- [ ] Task 2: Export public API from `src/ncaa_eval/evaluation/__init__.py` (AC: #4)
  - [ ] 2.1 Add `FoldResult`, `BacktestResult`, `run_backtest` to imports and `__all__`
- [ ] Task 3: Create `tests/unit/test_evaluation_backtest.py` (AC: #6, #7)
  - [ ] 3.1 Test `_evaluate_fold` with a mock model producing known predictions
  - [ ] 3.2 Test `run_backtest` sequential (`n_jobs=1`) produces correct fold count and metrics
  - [ ] 3.3 Test `run_backtest` parallel (`n_jobs=2`) produces identical results to sequential
  - [ ] 3.4 Test stateful model gets deep-copied per fold (original model state unchanged after backtest)
  - [ ] 3.5 Test stateless model column selection (only feature columns passed to fit/predict_proba)
  - [ ] 3.6 Test progress reporting (console output captured and validated)
  - [ ] 3.7 Test empty test fold (0 tournament games) is handled gracefully — metrics are NaN or skipped
  - [ ] 3.8 Test `n_jobs` parameter is passed through to `joblib.Parallel`
  - [ ] 3.9 Test summary DataFrame structure (correct columns, index is years, sorted ascending)
  - [ ] 3.10 Test single fold (2 seasons minimum) works correctly
  - [ ] 3.11 Test default metric_fns includes log_loss, brier_score, roc_auc, expected_calibration_error

## Dev Notes

### Design Specification

The parallel backtest orchestrates the full train → predict → evaluate cycle per fold:

```
walk_forward_splits() → list[CVFold]
                              ↓
                    joblib.Parallel(n_jobs)
                    ┌────────────────────────┐
                    │ _evaluate_fold(fold, model_copy, metrics)  │
                    │   1. model.fit(train)                      │
                    │   2. preds = model.predict_proba(test)     │
                    │   3. compute metrics(actuals, preds)       │
                    │   4. return FoldResult                     │
                    └────────────────────────┘
                              ↓
                    BacktestResult(fold_results, summary_df)
```

Each fold is **fully independent**: separate model copy, separate train/test data. This is embarrassingly parallel — ideal for `joblib.Parallel`.

### Critical Implementation Constraints

1. **`from __future__ import annotations`** required in all Python files (Ruff enforcement).
2. **`mypy --strict`** mandatory — use proper type annotations. `joblib.Parallel` and `joblib.delayed` need `# type: ignore[import-untyped]`.
3. **Use Google docstring style** (`Args:`, `Returns:`, `Raises:`), NOT NumPy style. This is the #1 docstring drift issue caught in code review for Stories 6.1–6.2.
4. **Mutation testing**: `src/ncaa_eval/evaluation/` is in `[tool.mutmut]` paths — tests must catch subtle mutations (e.g., `n_jobs=1` vs `n_jobs=-1` default, wrong metric aggregation).
5. **Library-First Rule**: Use `joblib.Parallel` + `joblib.delayed` (already in `pyproject.toml`). Do NOT reimplement with `multiprocessing` or `concurrent.futures`.
6. **Vectorization (NFR1)**: Metric calculations are already vectorized (Story 6.1). Do NOT add Python `for` loops for metric computation.

### File Structure

```
src/ncaa_eval/evaluation/
├── __init__.py          # MODIFY — add FoldResult, BacktestResult, run_backtest exports
├── metrics.py           # Existing — DO NOT MODIFY (consumed by _evaluate_fold)
├── splitter.py          # Existing — DO NOT MODIFY (consumed by run_backtest)
├── backtest.py          # NEW — parallel backtest orchestrator
```

```
tests/unit/
├── test_evaluation_metrics.py     # Existing — DO NOT MODIFY
├── test_evaluation_splitter.py    # Existing — DO NOT MODIFY
├── test_evaluation_backtest.py    # NEW — backtest tests
```

### Dependencies

All required libraries are already in `pyproject.toml`:
- `joblib = "*"` — `Parallel`, `delayed` for parallel fold execution
- `pandas` — DataFrame operations for summary aggregation
- `numpy` — array operations for metric inputs
- No new dependencies needed.

### Existing Codebase Context — DO NOT Reimplement

- **`src/ncaa_eval/evaluation/splitter.py`**: `walk_forward_splits(seasons, feature_server, mode=mode)` returns `Iterator[CVFold]`. The backtest MUST materialize this into a `list[CVFold]` before passing to joblib (generators cannot be pickled). The splitter already caches features per season internally — no need to duplicate this caching.

- **`src/ncaa_eval/evaluation/metrics.py`**: Provides `log_loss`, `brier_score`, `roc_auc`, `expected_calibration_error`. All accept `npt.NDArray[np.float64]` and return `float`. The backtest worker should convert `pd.Series` predictions/actuals to `.to_numpy()` before calling metrics.

- **`src/ncaa_eval/model/base.py`**: `Model.fit(X, y)` and `Model.predict_proba(X) -> pd.Series`. `StatefulModel` needs the full DataFrame (metadata + features) for `fit()`. Stateless models need only feature columns. Use `isinstance(model, StatefulModel)` to dispatch. The `get_state()`/`set_state()` hooks on `StatefulModel` are available but `copy.deepcopy(model)` is simpler for fork isolation.

- **`src/ncaa_eval/cli/train.py`**: `METADATA_COLS` (frozenset of 12 column names) and `_feature_cols(df)` helper. Import and reuse `METADATA_COLS` and `_feature_cols` — do NOT redefine them. Move these to a shared location if needed (e.g., `evaluation/backtest.py` can import from `cli.train`), OR better: since the backtest module is lower-level than CLI, consider moving `METADATA_COLS` and `_feature_cols` to `evaluation/backtest.py` and having `cli/train.py` import from there. **Preferred approach**: define `METADATA_COLS` and `_feature_cols` in `backtest.py` and update `cli/train.py` to import from the new location (keeps the dependency arrow correct: cli → evaluation, not evaluation → cli).

- **`src/ncaa_eval/transform/feature_serving.py`**: `StatefulFeatureServer.serve_season_features(year, mode)` — consumed by `walk_forward_splits`, not directly by the backtest. The backtest receives pre-built `CVFold` objects.

### `_evaluate_fold` Worker Function

```python
def _evaluate_fold(
    fold: CVFold,
    model: Model,
    metric_fns: Mapping[str, Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], float]],
) -> FoldResult:
    """Train model on fold.train, predict on fold.test, compute metrics.

    Args:
        fold: A single CV fold with train/test DataFrames.
        model: A deep-copied model instance (caller is responsible for copying).
        metric_fns: Mapping of metric name → callable(y_true, y_prob) → float.

    Returns:
        FoldResult with predictions, actuals, computed metrics, and timing.
    """
```

Key implementation details:
1. If `fold.test.empty`, return a `FoldResult` with empty predictions/actuals and NaN metrics.
2. Extract `y_train = fold.train["team_a_won"].astype(int)` and `y_test = fold.test["team_a_won"].astype(int)`.
3. For stateful models: `model.fit(fold.train, y_train)`. For stateless: `model.fit(fold.train[feat_cols], y_train)`.
4. For stateful models: `preds = model.predict_proba(fold.test)`. For stateless: `preds = model.predict_proba(fold.test[feat_cols])`.
5. Compute each metric: `{name: fn(y_test.to_numpy().astype(np.float64), preds.to_numpy().astype(np.float64)) for name, fn in metric_fns.items()}`. Wrap individual metric calls in try/except to handle edge cases (e.g., single-class test set for ROC-AUC).
6. Time the entire fold with `time.perf_counter()`.

### `run_backtest` Orchestrator

```python
def run_backtest(
    model: Model,
    feature_server: StatefulFeatureServer,
    *,
    seasons: Sequence[int],
    mode: str = "batch",
    n_jobs: int = -1,
    metric_fns: Mapping[str, Callable[...]] | None = None,
    console: Console | None = None,
) -> BacktestResult:
    """Run parallelized walk-forward cross-validation backtest.

    Args:
        model: Model instance to evaluate (will be deep-copied per fold).
        feature_server: Configured feature server for building CV folds.
        seasons: Season years to include (passed to walk_forward_splits).
        mode: Feature serving mode ("batch" or "stateful").
        n_jobs: Number of parallel workers. -1 = all cores, 1 = sequential.
        metric_fns: Metric functions to compute per fold. Defaults to
            {log_loss, brier_score, roc_auc, expected_calibration_error}.
        console: Rich Console for progress output.

    Returns:
        BacktestResult with per-fold results and summary DataFrame.
    """
```

Key implementation details:
1. Materialize folds: `folds = list(walk_forward_splits(seasons, feature_server, mode=mode))`.
2. Deep-copy model per fold: `models = [copy.deepcopy(model) for _ in folds]`.
3. Default metric_fns: `{"log_loss": log_loss, "brier_score": brier_score, "roc_auc": roc_auc, "ece": expected_calibration_error}`.
4. Dispatch: `results = Parallel(n_jobs=n_jobs)(delayed(_evaluate_fold)(fold, m, metric_fns) for fold, m in zip(folds, models))`.
5. Sort results by year ascending (joblib may return out of order with multi-processing).
6. Build summary DataFrame: rows = fold results, columns = metric names + `elapsed_seconds`, index = year.
7. Print progress: after `Parallel` completes, print per-fold timing and aggregate stats via Rich console.

### Model Deep-Copy Strategy

**Stateful models** (Elo) maintain internal rating state. Each fold must start from a fresh model to avoid cross-contamination. `copy.deepcopy(model)` handles this because:
- `EloModel` wraps `EloFeatureEngine` which holds a `dict[int, float]` ratings dict — fully picklable and deepcopy-safe.
- `XGBoostModel` wraps `XGBClassifier` — deepcopy creates a fresh unfitted classifier.

The `get_state()`/`set_state()` API exists but is unnecessary here — deepcopy is simpler and works for all model types uniformly.

**Important**: Do NOT modify the original `model` passed by the caller. After `run_backtest` returns, the caller's model should be in the same state as before the call.

### joblib Configuration

- **Backend**: Use default `loky` (multi-processing). Do NOT use `threading` — model training holds the GIL (XGBoost C++ extensions release it, but Elo's Python loops don't).
- **`n_jobs=-1`**: All available cores. `n_jobs=1`: sequential execution (useful for debugging, determinism comparison).
- **Verbose**: Do NOT pass `verbose` to `joblib.Parallel` — use Rich console for user-facing progress instead.
- **`prefer` parameter**: Do NOT specify `prefer` — let joblib use its default backend selection.

### Progress Reporting

Since `joblib.Parallel` with `loky` backend runs in separate processes, real-time per-fold progress bars are not straightforward. Instead:

1. Print fold count before dispatching: `"Running backtest: {n_folds} folds, n_jobs={n_jobs}"`.
2. After `Parallel` completes, print a Rich table with per-fold timing.
3. Print total elapsed time.

This is a post-hoc summary, not real-time — acceptable for AC #3. Real-time progress would require `joblib.Parallel(verbose=...)` or a callback mechanism, which adds complexity without significant user value for a few dozen folds.

### Default Metrics

```python
DEFAULT_METRICS: dict[str, Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], float]] = {
    "log_loss": log_loss,
    "brier_score": brier_score,
    "roc_auc": roc_auc,
    "ece": expected_calibration_error,
}
```

Import these from `ncaa_eval.evaluation.metrics`. The user can override by passing a custom `metric_fns` dict.

### Summary DataFrame Structure

```
         log_loss  brier_score  roc_auc    ece  elapsed_seconds
year
2009        0.65        0.22     0.73   0.04            2.31
2010        0.61        0.20     0.76   0.03            1.98
...
2025        0.58        0.18     0.79   0.02            3.12
```

- Index: `year` (int)
- Columns: one per metric in `metric_fns` + `elapsed_seconds`
- Sorted ascending by year

### Edge Cases

| Edge Case | Expected Behavior |
|---|---|
| Empty test fold (0 tournament games) | `FoldResult` with empty predictions/actuals, NaN metrics, fold still in results |
| Single fold (2 seasons) | Works normally, 1 fold in results |
| `n_jobs=1` | Sequential execution, deterministic baseline |
| `n_jobs=-1` on single-core machine | Equivalent to `n_jobs=1` (joblib handles this) |
| ROC-AUC on single-class test set | Catch `ValueError`, store `float('nan')` for that metric |
| Metric function raises exception | Catch per-metric, store NaN, continue with other metrics for that fold |
| Very large number of folds (20+) | joblib handles scheduling; memory is bounded by fold DataFrames |

### Testing Approach

Tests should use **mock/fake** models and feature servers, consistent with the patterns in `test_evaluation_splitter.py`:

```python
class _FakeStatelessModel(Model):
    """Minimal stateless model for testing — always predicts 0.5."""
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(0.5, index=X.index)
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> Self: ...
    def get_config(self) -> ModelConfig: ...

class _FakeStatefulModel(StatefulModel):
    """Minimal stateful model for testing — tracks state mutations."""
    ...
```

Use `unittest.mock.MagicMock` for `StatefulFeatureServer` (same pattern as splitter tests). Create pre-built `CVFold` objects directly for `_evaluate_fold` tests — no need to go through the splitter.

For the determinism test (AC #6): run `run_backtest` with `n_jobs=1` and `n_jobs=2`, compare all fold metrics with `pytest.approx`. Note that floating-point results should be identical (same input data, same operations), but use `approx` with tight tolerance as a safety margin.

### Previous Story Learnings (Stories 6.1 + 6.2)

- **`# type: ignore[import-untyped]`** required for `import joblib` and `import pandas as pd`.
- **`frozen=True` dataclass**: Use for `FoldResult` and `BacktestResult` — immutable results prevent caller mutation bugs.
- **Google docstring style**: NOT NumPy style. This was the #1 code review fix in Story 6.2.
- **`mode` validation at entry point**: `run_backtest` must validate `mode` immediately, not delegate to `walk_forward_splits` (template-requirements.md: "public APIs must validate at entry point").
- **Test organization**: Class-based grouping: `TestEvaluateFold`, `TestRunBacktest`, `TestDeterminism`, `TestEdgeCases`.
- **`pd.testing.assert_frame_equal()`** for DataFrame comparison in determinism tests.
- **`np.random.default_rng(seed)`** for reproducible synthetic test data.

### METADATA_COLS Refactoring

`METADATA_COLS` and `_feature_cols` are currently defined in `cli/train.py`. The backtest module needs the same logic to separate features from metadata. Two options:

1. **Import from cli.train** — creates `evaluation → cli` dependency (wrong direction).
2. **Move to evaluation.backtest, update cli.train to import** — correct dependency direction.

**Choose option 2.** Define `METADATA_COLS` and `_feature_cols` in `backtest.py`. Update `cli/train.py` to `from ncaa_eval.evaluation.backtest import METADATA_COLS, _feature_cols` (remove the local definitions). This is a minimal refactor — the values are identical, only the import source changes.

### Data Flow Architecture

```
StatefulFeatureServer
        ↓
walk_forward_splits() → list[CVFold]
        ↓
run_backtest()
  ├── copy.deepcopy(model) × N folds
  ├── joblib.Parallel(n_jobs)
  │     └── _evaluate_fold(fold, model_copy, metrics)
  │           ├── model.fit(train)
  │           ├── model.predict_proba(test)
  │           └── metrics.log_loss / brier / roc_auc / ece
  ├── collect FoldResults
  └── aggregate → BacktestResult(fold_results, summary_df)
```

### Project Structure Notes

- `src/ncaa_eval/evaluation/` already exists with `__init__.py`, `metrics.py`, `splitter.py` — new file `backtest.py` follows the same module pattern
- Moving `METADATA_COLS` from `cli/train.py` to `evaluation/backtest.py` corrects the dependency direction; `cli/train.py` import updated accordingly
- The backtest's output (`BacktestResult`) will be consumed by Story 5.5's training CLI (future enhancement to add `backtest` sub-command) and Story 7.3 (leaderboard visualization)

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Epic 6, Story 6.3 — acceptance criteria]
- [Source: specs/03-prd.md line 173 — 60-second Elo backtest performance target]
- [Source: specs/05-architecture-fullstack.md line 123 — Joblib for parallel CV (NFR2)]
- [Source: src/ncaa_eval/evaluation/splitter.py — walk_forward_splits, CVFold]
- [Source: src/ncaa_eval/evaluation/metrics.py — log_loss, brier_score, roc_auc, expected_calibration_error]
- [Source: src/ncaa_eval/model/base.py — Model ABC, StatefulModel, fit/predict_proba contract]
- [Source: src/ncaa_eval/cli/train.py — METADATA_COLS, _feature_cols, run_training pattern]
- [Source: _bmad-output/implementation-artifacts/6-2-implement-walk-forward-cross-validation-splitter.md — previous story learnings, testing patterns]
- [Source: _bmad-output/implementation-artifacts/6-1-implement-metric-library-scikit-learn-numpy.md — metric function signatures, code review learnings]
- [Source: _bmad-output/planning-artifacts/template-requirements.md — Google docstring mandate, mode validation, frozen dataclass patterns]

## Dev Agent Record

### Agent Model Used

{{agent_model_name_version}}

### Debug Log References

### Completion Notes List

### File List
