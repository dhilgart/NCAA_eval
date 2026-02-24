# Story 7.4: Build Lab Page — Model Deep Dive & Reliability Diagrams

Status: review

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a data scientist,
I want detailed diagnostic views for a specific model showing calibration, confusion, and feature importance,
So that I can understand where a model succeeds and fails beyond aggregate metrics.

## Acceptance Criteria

1. **Given** the user has selected a specific model run from the Leaderboard (Story 7.3), **When** the user views the Model Deep Dive page, **Then** a reliability diagram (predicted vs. actual probability) is rendered via `st.plotly_chart`.

2. **Given** a reliability diagram is displayed, **When** the user examines the plot, **Then** the diagram clearly identifies model over-confidence or under-confidence per the PRD success metric (scatter above/below diagonal).

3. **Given** the Deep Dive page is displayed, **When** the user interacts with metric explorer controls, **Then** a metric explorer allows drill-down by year, round, seed matchup, or conference.

4. **Given** the Deep Dive page displays a stateless model (e.g., XGBoost), **When** the user views the feature importance section, **Then** feature importance is displayed (for stateless models like XGBoost).

5. **Given** the Deep Dive page is displayed, **When** the user interacts with any visualization, **Then** all visualizations use the functional color palette (`COLOR_GREEN`, `COLOR_RED`, `COLOR_NEUTRAL`) and are interactive (Plotly).

6. **Given** the Deep Dive page is displayed, **When** the user views the page header, **Then** breadcrumb navigation shows context (e.g., Home > Lab > elo-abc123).

## Tasks / Subtasks

- [x] Task 1: Extend `RunStore` to persist fold-level predictions and actuals at training time (AC: #1, #2, #3)
  - [x] 1.1 Add `save_fold_predictions(run_id: str, fold_preds: pd.DataFrame) -> None` to `RunStore` — writes a `fold_predictions.parquet` under `runs/<run_id>/` containing columns `[year, game_id, team_a_id, team_b_id, pred_win_prob, team_a_won]`
  - [x] 1.2 Add `load_fold_predictions(run_id: str) -> pd.DataFrame | None` to `RunStore` — reads `fold_predictions.parquet` if exists, returns `None` for legacy runs
  - [x] 1.3 Write unit tests for `save_fold_predictions` / `load_fold_predictions` round-trip, missing-file handling, and legacy-run behavior (extend `tests/unit/test_run_store_metrics.py`)

- [x] Task 2: Wire `run_training()` to persist fold predictions after backtest (AC: #1)
  - [x] 2.1 In `src/ncaa_eval/cli/train.py` `run_training()`, after `store.save_metrics(...)`, build fold predictions DataFrame from FoldResult objects and call `store.save_fold_predictions()` to persist per-fold y_true/y_prob alongside the run
  - [x] 2.2 Verify the save call is inside the `if len(seasons) >= 2:` guard (same scope as `save_metrics`)

- [x] Task 3: Extend `RunStore` to persist trained model artifacts (AC: #4)
  - [x] 3.1 Add `save_model(run_id, model, feature_names=None)` to `RunStore`
  - [x] 3.2 Add `load_model(run_id)` / `load_feature_names(run_id)` to `RunStore`
  - [x] 3.3 Wire `store.save_model(run.run_id, model, feature_names=feat_cols)` in `run_training()`
  - [x] 3.4 Write unit tests for save/load round-trip and legacy-run behavior

- [x] Task 4: Add deep-dive data-loading functions to `dashboard/lib/filters.py` (AC: #1, #3, #4, #7)
  - [x] 4.1 Implement `load_fold_predictions(data_dir, run_id)` with `@st.cache_data(ttl=300)`
  - [x] 4.2 Skipped — `load_run_predictions` not needed; fold predictions are sufficient for reliability diagrams
  - [x] 4.3 Implement `load_feature_importances(data_dir, run_id)` with `@st.cache_data(ttl=300)`
  - [x] 4.4 Write unit tests for both new cache functions (mocked `RunStore`)

- [x] Task 5: Implement the Model Deep Dive page in `dashboard/pages/3_Model_Deep_Dive.py` (AC: #1–#6)
  - [x] 5.1 Wrap all page logic in `_render_deep_dive()` with helper functions `_render_reliability_section()`, `_render_metric_summary()`, `_render_feature_importance()`
  - [x] 5.2 No-run-selected guard: `st.info` + `st.page_link` to Leaderboard
  - [x] 5.3 Breadcrumb navigation: `st.caption("Home > Lab > {model_type}-{run_id[:8]}")`
  - [x] 5.4 Legacy run handling: `st.warning` for missing fold predictions
  - [x] 5.5 Reliability diagram with year selector and `plot_reliability_diagram()`
  - [x] 5.6 Per-year metric summary table with gradient styling
  - [x] 5.7 Metric Explorer: year drill-down via fold year selector (seed/round drill-down deferred per Dev Notes scope decision)
  - [x] 5.8 Feature importance horizontal bar chart (XGBoost) / info message (Elo)
  - [x] 5.9 Back to Leaderboard navigation at top of page

- [x] Task 6: Write tests for the Model Deep Dive page (AC: all)
  - [x] 6.1 Smoke test: existing `test_dashboard_app.py` import test passes
  - [x] 6.2 Test `_render_deep_dive()` with no `selected_run_id` — st.info + st.page_link called
  - [x] 6.3 Test with valid run and fold predictions — st.plotly_chart called
  - [x] 6.4 Test legacy run (empty fold predictions) — st.warning called
  - [x] 6.5 Test feature importance: XGBoost renders chart, Elo shows info message
  - [x] 6.6 All tests use `patch.object(module, ...)` pattern

- [x] Task 7: Verify quality gates (AC: all)
  - [x] 7.1 `mypy --strict src/ncaa_eval tests` passes — 79 source files, no issues
  - [x] 7.2 `mypy dashboard/` passes — 11 source files, no issues
  - [x] 7.3 `ruff check src/ tests/ dashboard/` passes — all checks passed
  - [x] 7.4 Full test suite passes — 807 passed, 1 skipped

## Dev Notes

### Critical Architecture Decision: Fold Predictions Persistence

**Problem:** The reliability diagram requires per-game `y_true` (actuals) and `y_prob` (predictions) arrays from walk-forward CV folds. Currently, `FoldResult` objects (containing `.predictions` and `.actuals` Series per fold) are computed at training time by `run_backtest()` but are NOT persisted. `predictions.parquet` stores the final training run predictions (not CV fold predictions), and it does NOT contain `team_a_won` (the actual outcome).

**Solution:** Extend `RunStore` to save a `fold_predictions.parquet` file at training time. Wire `run_training()` to call `store.save_fold_predictions(run.run_id, result.fold_results)` after the backtest. This is the minimal-cost approach: ~20 lines of new code in tracking.py, ~1 line in train.py.

**Schema for `fold_predictions.parquet`:**
```
year: int64        — CV fold year (maps to FoldResult.year)
game_index: int64  — position within fold
pred_win_prob: float64 — P(team_a wins)
team_a_won: float64    — actual outcome (0.0 or 1.0)
```

**Why NOT recompute on the fly:** Reconstructing actuals from `predictions.parquet` requires joining against `ParquetRepository.get_games()` to resolve `w_team_id`/`l_team_id` → `team_a_id`/`team_b_id` → `team_a_won`. This violates the 500ms performance target and the "no direct IO in dashboard" principle.

**Why NOT add to existing `predictions.parquet`:** The existing file stores final-training-run predictions (all games used for training), not cross-validation fold test-set predictions. CV fold predictions cover only tournament games per fold year. Conflating them would break the existing `Prediction` schema and `RunStore.load_predictions()` contract.

**Backward compatibility:** `load_fold_predictions()` returns `None` for legacy runs (no `fold_predictions.parquet`). The page shows `st.warning("No fold predictions available. Re-run training to generate diagnostic data.")`.

### Model Persistence for Feature Importance

**Problem:** The training CLI does NOT call `model.save()`. Feature importance for XGBoost requires access to the fitted `XGBClassifier.feature_importances_` array, which is only available on a trained model object.

**Solution:** Extend `RunStore` with `save_model()` / `load_model()` methods. Wire `run_training()` to save the model alongside other artifacts. The dashboard loads the model on demand when displaying feature importance.

**File layout after this story:**
```
data/runs/<run_id>/
    run.json                  # ModelRun JSON (unchanged)
    predictions.parquet       # Training predictions (unchanged)
    summary.parquet           # BacktestResult.summary (from Story 7.3)
    fold_predictions.parquet  # NEW: CV fold y_true/y_prob per year
    model/                    # NEW: trained model artifacts
        model.ubj             # XGBoost native format (if XGBoost)
        model.json            # Elo ratings (if Elo)
        config.json           # Model config
```

**`load_model()` implementation notes:**
- Read `run.json` to get `model_type` string
- Use `get_model(model_type)` from `ncaa_eval.model.base` registry to get the model class
- Call `ModelClass.load(run_dir / "model")` — this is a classmethod returning `Self`
- Return `None` if `model/` directory does not exist (legacy runs)
- Feature importance extraction: check `hasattr(model, "_clf")` and `hasattr(model._clf, "feature_importances_")` — only XGBoost exposes this

### Feature Importance — Feature Names

Feature names come from `feature_cols(df)` — the same list used during `.fit()`. **Problem:** The feature names are not persisted alongside the model. **Solution:** Save the feature names list when saving the model.

Add to `save_model()`:
```python
# If model was fit with known feature columns, save them
if hasattr(model, "_feature_names"):
    json.dump(model._feature_names, open(model_dir / "feature_names.json", "w"))
```

**Alternative (simpler, preferred):** Store `feature_names_in_` from the sklearn/XGBoost classifier directly. XGBoost's `_clf.feature_names_in_` is set after `.fit()` and is an ndarray of feature names. Save this alongside the model:

```python
def save_model(self, run_id: str, model: Model, feature_names: list[str] | None = None) -> None:
    model_dir = self._runs_dir / run_id / "model"
    model_dir.mkdir(exist_ok=True)
    model.save(model_dir)
    if feature_names is not None:
        import json
        with open(model_dir / "feature_names.json", "w") as f:
            json.dump(feature_names, f)
```

The CLI passes the feature column names from the training DataFrame.

### Metric Explorer — Drill-Down Dimensions

**AC3 specifies drill-down by:** year, round, seed matchup, or conference.

**Year drill-down:** Already covered by the year selector on the reliability diagram + per-year metric summary table (from `load_leaderboard_data` filtered to this run).

**Round / seed / conference drill-down — data availability challenge:** `fold_predictions.parquet` stores `year, game_index, pred_win_prob, team_a_won` but NOT `round`, `seed`, `team_a_id`, `team_b_id`, or `conference`. To drill down by round/seed/conference, we need to join fold predictions against game metadata.

**Practical approach:** Extend `fold_predictions.parquet` to also store `team_a_id`, `team_b_id`, and `game_id`. Then the dashboard can join against `load_run_predictions()` or a teams/seeds lookup to derive round/seed/conference dimensions. Alternatively, store `seed_a`, `seed_b`, and `round` directly in the fold predictions for self-contained drill-down.

**Recommended fold_predictions schema (enriched):**
```
year: int64
game_id: str
team_a_id: int64
team_b_id: int64
pred_win_prob: float64
team_a_won: float64
seed_a: int64 | null     — tournament seed of team_a (null for regular season games)
seed_b: int64 | null     — tournament seed of team_b
round_num: int64 | null  — tournament round (1-6, null for non-tournament)
```

**Note:** The enriched fields (seed_a, seed_b, round_num) require access to the feature DataFrame during backtest. The `CVFold` already contains the full feature DataFrame with metadata columns including `team_a_id`, `team_b_id`, and `is_tournament`. Seeds and rounds can be derived from game metadata at save time.

**Scope decision:** For MVP, store `year, game_id, team_a_id, team_b_id, pred_win_prob, team_a_won` in fold_predictions. Drill-down by year is fully supported. Round and seed matchup drill-down require an additional lookup against game/tournament data — implement as a best-effort feature (show groupings where data is available, gracefully degrade otherwise).

### Reliability Diagram — Already Implemented

`plot_reliability_diagram(y_true, y_prob, n_bins=10, title=None) -> go.Figure` in `evaluation.plotting` is ready to use. It:
- Calls `reliability_diagram_data()` internally
- Returns dual-axis figure: scatter (calibration curve) + bar (bin counts)
- Uses `plotly_dark` template and project color palette
- Hover template shows predicted/observed/count

### Breadcrumb Navigation

The UX spec calls for minimalist breadcrumbs: `Home > Lab > v1.2-GraphModel`. For our implementation:
```python
run = next((r for r in runs if r["run_id"] == run_id), None)
if run:
    label = f"{run['model_type']}-{run_id[:8]}"
    st.caption(f"Home > Lab > {label}")
```

### Page Layout

```
┌─────────────────────────────────────────────────────────────┐
│  ← Back to Leaderboard          Home > Lab > elo-abc12345   │
├─────────────────────────────────────────────────────────────┤
│  Model Deep Dive: elo-abc12345                              │
├─────────────────────────────────────────────────────────────┤
│  [Year Selector: All Years | 2024 | 2023 | ...]            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Reliability Diagram (Plotly)                       │    │
│  │  (Calibration curve + bin counts)                   │    │
│  └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│  Per-Year Metric Summary (styled DataFrame)                 │
│  ┌──────┬───────┬───────┬────────┬───────┐                  │
│  │ Year │ LL    │ BS    │ AUC    │ ECE   │                  │
│  │ 2024 │ 0.54  │ 0.19  │ 0.74   │ 0.03  │                  │
│  │ 2023 │ 0.57  │ 0.20  │ 0.72   │ 0.04  │                  │
│  └──────┴───────┴───────┴────────┴───────┘                  │
├─────────────────────────────────────────────────────────────┤
│  Feature Importance (horizontal bar chart, XGBoost only)    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  elo_delta          ████████████████████  0.312     │    │
│  │  seed_diff          ██████████████  0.198           │    │
│  │  srs_delta          ████████████  0.167             │    │
│  │  ...                                                │    │
│  └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│  Hyperparameters                                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  st.json(run.hyperparameters)                       │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Existing Code — DO NOT Reimplement

| Concern | Existing Code | Location |
|:---|:---|:---|
| Reliability diagram | `plot_reliability_diagram(y_true, y_prob, ...)` | `evaluation.plotting` |
| Backtest summary chart | `plot_backtest_summary(result, ...)` | `evaluation.plotting` |
| Color constants | `COLOR_GREEN`, `COLOR_RED`, `COLOR_NEUTRAL` | `evaluation.plotting` |
| Plotly template | `TEMPLATE = "plotly_dark"` | `evaluation.plotting` |
| Data dir resolution | `get_data_dir()` | `dashboard.lib.filters` |
| Available runs | `load_available_runs(data_dir)` | `dashboard.lib.filters` |
| Leaderboard data | `load_leaderboard_data(data_dir)` | `dashboard.lib.filters` |
| CSS styles | `MONOSPACE_CSS` | `dashboard.lib.styles` |
| Metric functions | `log_loss`, `brier_score`, `roc_auc`, `expected_calibration_error` | `evaluation.metrics` |
| ReliabilityData | `reliability_diagram_data(y_true, y_prob)` | `evaluation.metrics` |
| RunStore | `list_runs()`, `load_run()`, `load_predictions()`, `save_metrics()`, `load_metrics()`, `load_all_summaries()` | `model.tracking` |
| ModelRun | `ModelRun` Pydantic model | `model.tracking` |
| Prediction | `Prediction` Pydantic model | `model.tracking` |
| Feature cols | `feature_cols(df)` | `evaluation.backtest` (via `evaluation.__init__`) |
| FoldResult | `FoldResult` frozen dataclass | `evaluation.backtest` |
| BacktestResult | `BacktestResult` frozen dataclass | `evaluation.backtest` |
| Model ABC | `Model`, `StatefulModel`, `get_model()` | `model.base` |
| Plugin registry | `@register_model`, `get_model(name)` | `model.base` |
| Gradient styling pattern | `df.style.background_gradient(cmap=...)` | `dashboard/pages/1_Lab.py` |
| Page navigation | `st.switch_page("pages/3_Model_Deep_Dive.py")` | `dashboard/pages/1_Lab.py` |
| Session state keys | `selected_run_id`, `selected_year` | `dashboard/app.py` |

### Streamlit API Notes

- `st.plotly_chart(fig, use_container_width=True)` for responsive Plotly rendering
- `st.page_link("pages/1_Lab.py", label="Back to Leaderboard")` for navigation links
- `st.selectbox(key=...)` binds to `st.session_state` for persistent filter state
- `st.json(data)` renders collapsible JSON for hyperparameters
- `st.caption(text)` for breadcrumb text (lightweight, secondary text)

### Pandas Styler Gradient Pattern (from Story 7.3)

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
```

### Feature Importance Bar Chart

```python
import plotly.graph_objects as go

fig = go.Figure(go.Bar(
    x=importances,
    y=feature_names,
    orientation="h",
    marker_color=COLOR_GREEN,
))
fig.update_layout(
    template=TEMPLATE,
    title="Feature Importance (Gain)",
    xaxis_title="Importance",
    yaxis_title="Feature",
    yaxis=dict(autorange="reversed"),  # highest importance at top
    height=max(400, len(feature_names) * 25),
)
```

### Project Structure Notes

- `src/ncaa_eval/model/tracking.py` — add `save_fold_predictions`, `load_fold_predictions`, `save_model`, `load_model` to `RunStore`
- `src/ncaa_eval/cli/train.py` — add `save_fold_predictions` and `save_model` calls to `run_training()`
- `dashboard/lib/filters.py` — add `load_fold_predictions`, `load_run_predictions`, `load_feature_importances`
- `dashboard/pages/3_Model_Deep_Dive.py` — full rewrite (replace placeholder)
- `tests/unit/test_run_store_metrics.py` — extend with fold prediction and model persistence tests
- `tests/unit/test_dashboard_filters.py` — extend with new cache function tests
- `tests/unit/test_deep_dive_page.py` — new test file for page logic
- `tests/unit/test_cli_train.py` — extend to assert `fold_predictions.parquet` and `model/` are created

### mypy Considerations

- `RunStore` changes in `src/ncaa_eval/` — `mypy --strict` applies
- `load_fold_predictions` return type `pd.DataFrame | None` — callers handle `None`
- `load_model` return type `Model | None` — callers handle `None`
- Dashboard files: `mypy dashboard/` (non-strict)
- `from __future__ import annotations` required in ALL new/modified files

### Edge Cases

- **No selected run:** Page shows `st.info("Select a model run...")` with link back to leaderboard
- **Legacy run (no fold_predictions.parquet):** Page shows `st.warning` for reliability diagram, still shows metric summary from `summary.parquet`
- **Legacy run (no model/ directory):** Feature importance section shows `st.info("Feature importance not available...")`
- **Elo model (no feature importance):** Feature importance section shows `st.info("Feature importance is not available for stateful models.")`
- **Single fold year:** Year selector shows only one option; aggregate view is identical to single-year view
- **NaN in metrics:** Use `_fmt(v)` guard pattern from Story 7.3 (return `"N/A"` for NaN values)
- **Empty fold predictions for a year:** Skip that year in reliability diagram selector

### Previous Story Learnings (Story 7.3)

- **Session state:** Use `st.session_state.setdefault(key, value)` not conditional assignment
- **Data loading guards:** Use `except OSError` not `except Exception` in cache functions
- **Cache serialization:** Return `list[dict]` from `@st.cache_data` functions, not Pydantic models or DataFrames
- **`from __future__ import annotations`:** Required in ALL files
- **Google docstring style:** `Args:`, `Returns:`, `Raises:`
- **`patch.object(module, ...)`:** Use for digit-prefixed page modules, NOT `patch("dashboard.pages.3_Model_Deep_Dive.func")`
- **No `iterrows()`:** Use merge + vectorized operations
- **Public API boundary:** Import from public `ncaa_eval.evaluation` or `ncaa_eval.model`, not private `_` functions
- **`_DISPLAY_COLS` pattern:** Define explicit column constants for displayed tables
- **`st.metric` NaN guard:** Always guard formatted values against NaN
- **`lib/` .gitignore override:** Already in place from Story 7.2

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 7.4 — acceptance criteria and epic context]
- [Source: specs/05-architecture-fullstack.md §5.2 — Lab Page: render_reliability_diagram()]
- [Source: specs/05-architecture-fullstack.md §7 — Frontend Architecture (Streamlit multipage, session state, caching)]
- [Source: specs/05-architecture-fullstack.md §12 — Coding Standards (type sharing, no direct IO in UI)]
- [Source: specs/04-front-end-spec.md §2.1 — Site Map: Lab > Reliability Diagrams + Metric Explorer]
- [Source: specs/04-front-end-spec.md §3.1 — Backtest-to-Selection Diagnostic Loop (step 2: reliability side-panel)]
- [Source: specs/04-front-end-spec.md §4.2 — Dark Mode, monospace fonts, functional color palette]
- [Source: specs/04-front-end-spec.md §5.2 — 500ms interaction response target, st.cache_data]
- [Source: src/ncaa_eval/evaluation/plotting.py — plot_reliability_diagram, COLOR_GREEN/RED/NEUTRAL, TEMPLATE]
- [Source: src/ncaa_eval/evaluation/metrics.py — ReliabilityData, reliability_diagram_data]
- [Source: src/ncaa_eval/evaluation/backtest.py — BacktestResult, FoldResult, feature_cols, METADATA_COLS]
- [Source: src/ncaa_eval/model/tracking.py — RunStore, ModelRun, Prediction, save_metrics/load_metrics]
- [Source: src/ncaa_eval/model/base.py — Model ABC, StatefulModel, get_model() registry]
- [Source: src/ncaa_eval/cli/train.py — run_training(), save_run/save_metrics wiring]
- [Source: dashboard/pages/1_Lab.py — leaderboard gradient styling, click-to-navigate pattern]
- [Source: dashboard/pages/3_Model_Deep_Dive.py — current placeholder]
- [Source: dashboard/lib/filters.py — get_data_dir, load_available_runs, load_leaderboard_data patterns]
- [Source: _bmad-output/implementation-artifacts/7-3-build-lab-page-backtest-leaderboard.md — previous story learnings]
- [Source: _bmad-output/planning-artifacts/template-requirements.md — session state, OSError guard, cache serialization, patch.object, NaN guard patterns]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

### Completion Notes List

### File List

**Modified:**
- `src/ncaa_eval/model/tracking.py` — Added `save_fold_predictions`, `load_fold_predictions`, `save_model`, `load_model`, `load_feature_names` to RunStore
- `src/ncaa_eval/evaluation/backtest.py` — Extended `FoldResult` with `test_game_ids`, `test_team_a_ids`, `test_team_b_ids` fields; updated `_evaluate_fold()` to populate them
- `src/ncaa_eval/cli/train.py` — Added `_build_fold_predictions()` helper; wired fold prediction and model persistence into `run_training()`
- `dashboard/lib/filters.py` — Added `load_fold_predictions()` and `load_feature_importances()` cached loaders
- `dashboard/pages/3_Model_Deep_Dive.py` — Full rewrite from placeholder to diagnostic page with reliability diagram, metric summary, feature importance, hyperparameters
- `tests/unit/test_run_store_metrics.py` — Added 12 tests for fold predictions and model persistence
- `tests/unit/test_dashboard_filters.py` — Added 8 tests for new data loader functions

**New:**
- `tests/unit/test_deep_dive_page.py` — 6 tests for Model Deep Dive page rendering logic

### Change Log

| Commit | Message |
|:---|:---|
| `8344868` | feat(tracking): add fold predictions persistence to RunStore |
| `648fcc1` | feat(train): wire fold predictions persistence into training pipeline |
| `2d3d728` | feat(tracking): add model persistence to RunStore |
| `69583d1` | feat(dashboard): add fold predictions and feature importance loaders |
| `8500265` | feat(dashboard): implement Model Deep Dive page |
| `f75d8ed` | test(dashboard): add Model Deep Dive page tests |
