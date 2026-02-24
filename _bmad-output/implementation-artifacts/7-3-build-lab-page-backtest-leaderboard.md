# Story 7.3: Build Lab Page — Backtest Leaderboard

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a data scientist,
I want a sortable leaderboard comparing all trained models by various metrics,
So that I can quickly identify the best-performing models and spot trends.

## Acceptance Criteria

1. **Given** model run results are persisted in the local store (Epic 5), **When** the user navigates to the Lab Leaderboard page, **Then** all model runs are displayed in a sortable table with columns for each metric (LogLoss, Brier, ROC-AUC, ECE).

2. **Given** the leaderboard is displayed, **When** the user clicks a column header, **Then** the table supports sorting by any metric column.

3. **Given** the leaderboard is displayed, **When** the user views the top of the page, **Then** `st.metric` diagnostic cards display top-line KPIs (best LogLoss, best Brier Score, best ROC-AUC, lowest ECE) with performance deltas vs. a baseline model.

4. **Given** the leaderboard is displayed, **When** the user views metric columns, **Then** conditional formatting (Green-to-Red gradients) highlights model outliers per the UX spec.

5. **Given** the leaderboard is displayed, **When** the user clicks a model run ID, **Then** the app navigates to the Model Deep Dive view (Story 7.4) with that run selected.

6. **Given** the global filters are set, **When** the user views the leaderboard, **Then** the leaderboard filters by the global Tournament Year and Model Version selections from `st.session_state`.

7. **Given** data must load quickly, **When** the page renders, **Then** data loads within the 500ms interaction response target via `@st.cache_data`.

## Tasks / Subtasks

- [x] Task 1: Extend `RunStore` to persist and load backtest metric summaries (AC: #1, #7)
  - [x] 1.1 Add `save_metrics(run_id: str, summary: pd.DataFrame) -> None` to `RunStore` — writes `summary.parquet` under `runs/<run_id>/`
  - [x] 1.2 Add `load_metrics(run_id: str) -> pd.DataFrame | None` to `RunStore` — reads `summary.parquet` if it exists, returns `None` for legacy runs without metrics
  - [x] 1.3 Add `load_all_summaries() -> pd.DataFrame` to `RunStore` — iterates all runs, loads available metrics, and concatenates into a single DataFrame with `run_id` column; skips runs without `summary.parquet`
  - [x] 1.4 Write unit tests for `save_metrics`/`load_metrics`/`load_all_summaries` round-trip, missing-file handling, and empty-store edge case

- [x] Task 2: Wire `run_training()` CLI to persist backtest metrics after training (AC: #1)
  - [x] 2.1 In `src/ncaa_eval/cli/train.py` `run_training()`, after `store.save_run()`, run walk-forward backtest and call `store.save_metrics(run.run_id, result.summary)` to persist the backtest summary alongside the run
  - [x] 2.2 No existing CLI integration tests to update (story referenced nonexistent `src/ncaa_eval/model/cli.py`; actual file is `src/ncaa_eval/cli/train.py`)

- [x] Task 3: Add leaderboard data-loading function to `dashboard/lib/filters.py` (AC: #1, #6, #7)
  - [x] 3.1 Implement `load_leaderboard_data(data_dir: str) -> list[dict[str, object]]` decorated with `@st.cache_data(ttl=300)` — calls `RunStore(Path(data_dir)).load_all_summaries()`, joins with `ModelRun` metadata (model_type, timestamp, start_year, end_year), returns list of dicts (cache-safe)
  - [x] 3.2 Write unit tests for `load_leaderboard_data` with mocked `RunStore` (4 tests: joined data, empty summaries, missing dir, OSError)

- [x] Task 4: Implement the leaderboard page in `dashboard/pages/1_Lab.py` (AC: #1–#7)
  - [x] 4.1 Read global filters from `st.session_state`: `selected_year`, `selected_run_id`
  - [x] 4.2 Call `load_leaderboard_data(str(get_data_dir()))` to get the full leaderboard DataFrame
  - [x] 4.3 Apply year filter: if `selected_year` is set, filter the summary to show only metrics for that tournament year; if no year filter, show aggregate (mean across years) per run
  - [x] 4.4 Render `st.metric` diagnostic cards in a row of 4 columns — best LogLoss, best Brier, best ROC-AUC, lowest ECE — with delta vs. the worst model's value (or vs. baseline if a "baseline" run exists)
  - [x] 4.5 Render the leaderboard as `st.dataframe(styled_df)` with `use_container_width=True`
  - [x] 4.6 Apply Pandas Styler `background_gradient(cmap=..., subset=[metric_cols])` with a Red-to-Green colormap for ROC-AUC (higher=better), Green-to-Red for LogLoss/Brier/ECE (lower=better)
  - [x] 4.7 Make the run_id column clickable — use `st.dataframe` with `on_select="rerun"` and `selection_mode="single-row"`; on select, set `st.session_state.selected_run_id` and call `st.switch_page("pages/3_Model_Deep_Dive.py")`
  - [x] 4.8 Handle empty state: if no runs exist, display `st.info("No model runs available. Train a model first: python -m ncaa_eval.cli train --model elo")`

- [x] Task 5: Write tests for the leaderboard page (AC: all)
  - [x] 5.1 Smoke test: `import dashboard.pages.1_Lab` succeeds without error (existing test passes — refactored page to use function wrapper for import safety)
  - [x] 5.2 Unit test `load_leaderboard_data` with mocked `RunStore` returning known data (done in Task 3 — 4 tests in test_dashboard_filters.py)
  - [x] 5.3 Test that the page handles empty data gracefully (2 tests in test_leaderboard_page.py)
  - [x] 5.4 Test that year filtering logic correctly subsets the DataFrame (3 tests: specific year, missing year, aggregate)

- [x] Task 6: Verify quality gates (AC: all)
  - [x] 6.1 `mypy --strict src/ncaa_eval tests` passes — 78 files, no issues
  - [x] 6.2 `mypy dashboard/` passes — 11 files, no issues
  - [x] 6.3 `ruff check src/ tests/ dashboard/` — all passed (notebook warnings are pre-existing)
  - [x] 6.4 Full test suite passes — 777 passed, 1 skipped, 0 failures

## Dev Notes

### Critical Design Decision: Metric Persistence Gap

**Problem:** `RunStore` currently persists `run.json` (metadata) and `predictions.parquet` (raw predictions) but does NOT persist `BacktestResult.summary` (per-year metrics: log_loss, brier_score, roc_auc, ece). The leaderboard needs metric values to display.

**Solution:** Add `save_metrics()` / `load_metrics()` / `load_all_summaries()` methods to `RunStore` that persist `BacktestResult.summary` as a Parquet file (`summary.parquet`) alongside existing run artifacts. Wire `run_training()` in the CLI to persist the backtest summary after training completes.

**Why NOT recompute on the fly:** `predictions.parquet` stores `pred_win_prob` but NOT `team_a_won` (the actual outcome). Reconstructing actuals requires joining against `ParquetRepository.get_games()` by `game_id`, resolving `w_team_id`/`l_team_id` → `team_a_id`/`team_b_id`, then computing metrics — significant logic that violates the 500ms performance target and "no direct IO in dashboard" principle.

**Why NOT add metrics to `ModelRun`:** `ModelRun` stores run-level metadata. Metrics are per-year per-fold data (a DataFrame), not a single scalar. Conflating them would bloat the Pydantic model. A separate sidecar file (`summary.parquet`) is the cleanest separation.

**Backward compatibility:** `load_metrics()` returns `None` for runs created before this change (no `summary.parquet`). The leaderboard skips these runs or shows "N/A" metrics.

### RunStore Extension — Exact API

Add to `src/ncaa_eval/model/tracking.py`:

```python
def save_metrics(self, run_id: str, summary: pd.DataFrame) -> None:
    """Persist backtest metric summary for a run.

    Args:
        run_id: The run identifier.
        summary: BacktestResult.summary DataFrame (index=year,
            columns=[log_loss, brier_score, roc_auc, ece, elapsed_seconds]).
    """
    run_dir = self._runs_dir / run_id
    if not run_dir.exists():
        msg = f"Run directory not found: {run_id}"
        raise FileNotFoundError(msg)
    summary.to_parquet(run_dir / "summary.parquet")

def load_metrics(self, run_id: str) -> pd.DataFrame | None:
    """Load backtest metric summary for a run.

    Args:
        run_id: The run identifier.

    Returns:
        Summary DataFrame or None if no summary exists (legacy run).
    """
    path = self._runs_dir / run_id / "summary.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)

def load_all_summaries(self) -> pd.DataFrame:
    """Load metric summaries for all runs that have them.

    Returns:
        DataFrame with columns [run_id, year, log_loss, brier_score,
        roc_auc, ece, elapsed_seconds]. Empty DataFrame if no summaries.
    """
    frames: list[pd.DataFrame] = []
    for run in self.list_runs():
        summary = self.load_metrics(run.run_id)
        if summary is not None:
            df = summary.reset_index()
            df["run_id"] = run.run_id
            frames.append(df)
    if not frames:
        return pd.DataFrame(
            columns=["run_id", "year", "log_loss", "brier_score",
                     "roc_auc", "ece", "elapsed_seconds"]
        )
    return pd.concat(frames, ignore_index=True)
```

### CLI Wiring — `run_training()` Change

In `src/ncaa_eval/model/cli.py`, after the `run_backtest()` call, add:

```python
store.save_metrics(run.run_id, result.summary)
```

Check exact location: the `run_training()` function calls `run_backtest()` and receives a `BacktestResult`. After `store.save_run(run, predictions)`, add the `save_metrics` call.

### Leaderboard Data Loading

`dashboard/lib/filters.py` — new function:

```python
@st.cache_data(ttl=300)
def load_leaderboard_data(data_dir: str) -> list[dict[str, object]]:
    """Load leaderboard data: run metadata joined with metric summaries.

    Returns list of dicts (serializable for st.cache_data) with keys:
    run_id, model_type, timestamp, start_year, end_year, year,
    log_loss, brier_score, roc_auc, ece.
    """
```

**Cache pattern:** Return `list[dict]` not `pd.DataFrame` (Streamlit `@st.cache_data` serializes return values; dicts are safe). Reconstruct DataFrame in the page script.

### Leaderboard Page — `dashboard/pages/1_Lab.py`

**Replace** the current placeholder with the full leaderboard implementation.

**Layout:**

```
┌─────────────────────────────────────────────────┐
│  Backtest Leaderboard                           │
├────────┬────────┬────────┬────────┬─────────────┤
│Best    │Best    │Best    │Lowest  │             │
│LogLoss │Brier   │ROC-AUC │ECE     │  (metrics)  │
│0.543   │0.195   │0.742   │0.032   │             │
│▼ -0.02 │▼ -0.01 │▲ +0.03 │▼ -0.01 │             │
├────────┴────────┴────────┴────────┴─────────────┤
│ ┌─────────────────────────────────────────────┐ │
│ │ Sortable DataFrame                          │ │
│ │ run_id | model | year | LL   | BS   | AUC  │ │
│ │ abc123 | elo   | 2024 | 0.54 | 0.20 | 0.74 │ │
│ │ def456 | xgb   | 2024 | 0.51 | 0.19 | 0.76 │ │
│ │ ...    | ...   | ...  | ...  | ...  | ...  │ │
│ └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

**Year filter behavior:**
- If `selected_year` is set: show per-year rows for that year only
- If `selected_year` is None or "All": show mean metrics across all years per run (aggregated view)

**Metric gradient direction:**
- LogLoss, Brier Score, ECE: lower is better → Green (low) to Red (high)
- ROC-AUC: higher is better → Red (low) to Green (high)

Use `matplotlib.colors.LinearSegmentedColormap.from_list("gr", [COLOR_GREEN, COLOR_RED])` and its reverse for the opposite direction. Or use Pandas `background_gradient(cmap="RdYlGn")` for ROC-AUC and `background_gradient(cmap="RdYlGn_r")` for LogLoss/Brier/ECE.

**Click-to-navigate:** Use `st.dataframe(on_select="rerun", selection_mode="single-row")` to detect row selection, then `st.switch_page("pages/3_Model_Deep_Dive.py")` after setting `st.session_state.selected_run_id`. If `on_select` is not available in Streamlit 1.54 (check API), use a workaround: render run_id as a column, add a `st.selectbox` or `st.button` per row. As a fallback, use `st.data_editor` with a checkbox column.

**Alternative click approach (simpler):** Below the DataFrame, add a "View Details" button that reads the DataFrame selection state and navigates. Or use `st.page_link("pages/3_Model_Deep_Dive.py")` with query params.

### Pandas Styler with `st.dataframe` — Confirmed Working

`st.dataframe` (Streamlit 1.54) supports Pandas Styler objects. `df.style.background_gradient()` renders correctly. Use:

```python
styled = df.style.background_gradient(
    cmap="RdYlGn_r",  # Red=high (bad), Green=low (good)
    subset=["log_loss", "brier_score", "ece"],
).background_gradient(
    cmap="RdYlGn",    # Green=high (good), Red=low (bad)
    subset=["roc_auc"],
).format({
    "log_loss": "{:.4f}",
    "brier_score": "{:.4f}",
    "roc_auc": "{:.4f}",
    "ece": "{:.4f}",
})
st.dataframe(styled, use_container_width=True)
```

### `st.switch_page` for Navigation

`st.switch_page(page)` programmatically navigates to another page. The `page` argument must match the path used in `st.Page()`. From Story 7.2, the deep dive page is registered as:

```python
deep_dive = st.Page("pages/3_Model_Deep_Dive.py", title="Model Deep Dive", ...)
```

So call `st.switch_page("pages/3_Model_Deep_Dive.py")` after setting `st.session_state.selected_run_id`.

### `st.metric` Diagnostic Cards

```python
col1, col2, col3, col4 = st.columns(4)
col1.metric("Best Log Loss", f"{best_ll:.4f}", delta=f"{best_ll - worst_ll:.4f}")
col2.metric("Best Brier", f"{best_bs:.4f}", delta=f"{best_bs - worst_bs:.4f}")
col3.metric("Best ROC-AUC", f"{best_auc:.4f}", delta=f"{best_auc - worst_auc:.4f}", delta_color="normal")
col4.metric("Lowest ECE", f"{best_ece:.4f}", delta=f"{best_ece - worst_ece:.4f}")
```

**Delta direction:** `st.metric` uses `delta_color="normal"` by default (green=positive, red=negative). For LogLoss/Brier/ECE where lower is better, the delta should be negative (improvement = decrease), so use `delta_color="inverse"`. For ROC-AUC where higher is better, use `delta_color="normal"`.

### Existing Code — DO NOT Reimplement

| Concern | Existing Code | Location |
|:---|:---|:---|
| Color constants | `COLOR_GREEN`, `COLOR_RED`, `COLOR_NEUTRAL` | `evaluation.plotting` (re-exported from `evaluation`) |
| Plotly template | `TEMPLATE = "plotly_dark"` | `evaluation.plotting` |
| Data dir resolution | `get_data_dir()` | `dashboard.lib.filters` |
| Available runs | `load_available_runs(data_dir)` | `dashboard.lib.filters` |
| Available years | `load_available_years(data_dir)` | `dashboard.lib.filters` |
| Available scorings | `load_available_scorings()` | `dashboard.lib.filters` |
| CSS styles | `MONOSPACE_CSS` | `dashboard.lib.styles` |
| Metric functions | `log_loss`, `brier_score`, `roc_auc`, `expected_calibration_error` | `evaluation.metrics` |
| Plotting functions | `plot_backtest_summary`, `plot_metric_comparison` | `evaluation.plotting` |
| RunStore | `RunStore.list_runs()`, `.load_run()`, `.load_predictions()` | `model.tracking` |
| ModelRun | `ModelRun` Pydantic model | `model.tracking` |

### Filesystem Layout After This Story

```
data/
  runs/
    <run_id>/
      run.json              # ModelRun JSON (unchanged)
      predictions.parquet   # Predictions (unchanged)
      summary.parquet       # NEW: BacktestResult.summary (year × metrics)
```

### Project Structure Notes

- New methods added to `src/ncaa_eval/model/tracking.py` (`RunStore` class)
- Small change to `src/ncaa_eval/model/cli.py` (`run_training()` function)
- New function in `dashboard/lib/filters.py`
- Full rewrite of `dashboard/pages/1_Lab.py` (replacing placeholder)
- New test file: `tests/unit/test_run_store_metrics.py` (or extend existing `test_tracking.py`)
- Update existing: `tests/unit/test_dashboard_app.py`, `tests/unit/test_dashboard_filters.py`

### mypy Considerations

- `RunStore` changes are in `src/ncaa_eval/` → `mypy --strict` applies
- `load_metrics` return type is `pd.DataFrame | None` — callers must handle `None`
- Dashboard files: `mypy dashboard/` non-strict
- `from __future__ import annotations` required in ALL files

### Edge Cases

- **No runs exist:** Leaderboard shows `st.info("No model runs available...")`
- **Runs exist but no summaries (legacy):** `load_all_summaries()` returns empty DataFrame → show `st.warning("No backtest metrics available. Re-run training to generate metrics.")`
- **Single run:** Diagnostic cards show metrics but deltas are 0 (or omit delta)
- **Year filter with no data for that year:** Show `st.info("No backtest results for {year}")`
- **Mixed runs (some with summaries, some without):** Only runs with summaries appear in leaderboard; add a note about legacy runs

### Previous Story Learnings (Story 7.2)

- **Session state:** Use `st.session_state.setdefault(key, value)` not conditional assignment
- **Data loading guards:** Use `except OSError` not `except Exception` in cache functions
- **Cache serialization:** Return `list[dict]` from `@st.cache_data` functions, not Pydantic models
- **`from __future__ import annotations`:** Required in ALL files
- **Google docstring style:** `Args:`, `Returns:`, `Raises:`
- **`lib/` .gitignore override:** Already in place from Story 7.2
- **`st.plotly_chart(fig, use_container_width=True)`** for responsive Plotly rendering

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 7.3 — acceptance criteria and epic context]
- [Source: specs/05-architecture-fullstack.md §5.2 — Lab Page responsibilities: diagnostics and leaderboard]
- [Source: specs/05-architecture-fullstack.md §7 — Frontend Architecture (Streamlit multipage, session state, caching)]
- [Source: specs/05-architecture-fullstack.md §12 — Coding Standards (type sharing, no direct IO in UI)]
- [Source: specs/04-front-end-spec.md §3.1 — Backtest-to-Selection Diagnostic Loop user flow]
- [Source: specs/04-front-end-spec.md §4.1 — Heatmap DataFrame with Green-to-Red gradients]
- [Source: specs/04-front-end-spec.md §4.2 — Dark Mode, monospace fonts, functional color palette]
- [Source: specs/04-front-end-spec.md §5.2 — 500ms interaction response target, st.cache_data]
- [Source: _bmad-output/implementation-artifacts/7-2-build-streamlit-app-shell-navigation.md — previous story, session state patterns, data loading patterns]
- [Source: src/ncaa_eval/model/tracking.py — RunStore API, ModelRun schema]
- [Source: src/ncaa_eval/evaluation/backtest.py — BacktestResult, FoldResult schemas]
- [Source: src/ncaa_eval/evaluation/plotting.py — COLOR_GREEN, COLOR_RED, COLOR_NEUTRAL, TEMPLATE]
- [Source: src/ncaa_eval/evaluation/__init__.py — public API re-exports]
- [Source: _bmad-output/planning-artifacts/template-requirements.md — session state patterns, OSError guard, cache serialization]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

### Completion Notes List

- Task 1: Added `save_metrics()`, `load_metrics()`, `load_all_summaries()` to `RunStore` in tracking.py. 8 unit tests all pass covering round-trip, missing-file, empty-store, mixed legacy/new runs.
- Task 2: Wired `run_training()` in `src/ncaa_eval/cli/train.py` to run walk-forward backtest after training and persist `summary.parquet` via `store.save_metrics()`. Skips backtest when <2 seasons. No existing CLI integration tests existed (story referenced wrong file path).
- Task 3: Added `load_leaderboard_data()` to `dashboard/lib/filters.py` — cached, returns list[dict] joining summaries with run metadata. 4 unit tests pass.
- Task 4: Replaced placeholder in `dashboard/pages/1_Lab.py` with full leaderboard: year filtering, st.metric KPI cards, Pandas Styler gradients, st.dataframe with on_select for row click navigation to Deep Dive, and empty state handling.
- Task 5: Refactored page into `_render_leaderboard()` function for import safety. Added 5 tests in test_leaderboard_page.py (year filtering + empty state). Existing smoke import test passes. load_leaderboard_data tests already in Task 3.
- Task 6: All quality gates pass. mypy --strict (78 files), mypy dashboard (11 files), ruff check, full test suite (777 passed, 1 skipped).
- Code Review (2026-02-24): 7 issues fixed: H1 (iterrows → vectorized merge in load_leaderboard_data), H2 (_feature_cols promoted to public feature_cols in evaluation API), M1 (legacy-run st.warning added to 1_Lab.py), M2 (CLI test extended to assert summary.parquet + new test_train_persists_backtest_metrics), M3 (session_state.setdefault), L1 (weak empty-state tests replaced with actual mock-based assertions), L2 (RunStore docstring updated). Post-review: 778 passed, 1 skipped.

### Change Log

- 2026-02-24: Story 7.3 implementation complete — all 6 tasks done, all ACs satisfied
- 2026-02-24: Code review complete — 7 issues fixed (2 HIGH, 3 MEDIUM, 2 LOW); story marked done

### File List

- src/ncaa_eval/model/tracking.py (modified — added 3 new methods + updated docstring)
- src/ncaa_eval/cli/train.py (modified — added backtest + save_metrics after training; updated import)
- src/ncaa_eval/evaluation/backtest.py (modified — _feature_cols renamed to feature_cols)
- src/ncaa_eval/evaluation/__init__.py (modified — feature_cols added to public API)
- dashboard/lib/filters.py (modified — added load_leaderboard_data; fixed iterrows → merge)
- dashboard/pages/1_Lab.py (rewritten — full leaderboard with function wrapper; legacy-run warning; setdefault)
- tests/unit/test_run_store_metrics.py (new — 8 unit tests)
- tests/unit/test_dashboard_filters.py (modified — added 4 leaderboard tests)
- tests/unit/test_leaderboard_page.py (new — 5 tests; empty-state tests strengthened with mock-based assertions)
- tests/unit/test_cli_train.py (modified — assert summary.parquet + new test_train_persists_backtest_metrics)
- tests/unit/test_evaluation_backtest.py (modified — updated import from _feature_cols to feature_cols)
