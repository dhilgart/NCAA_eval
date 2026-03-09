# Story 9.3: Feature Importance for Elo and Logistic Regression

Status: in-progress

## Story

As a **data scientist**,
I want to **see feature importance / interpretability information for all model types, not just XGBoost**,
so that **I can understand what drives predictions across Elo, Logistic Regression, and XGBoost models**.

## Acceptance Criteria

1. **Given** a trained model is selected in the Model Deep Dive dashboard page
   **When** the user views the Feature Importance section
   **Then** XGBoost shows feature importance (existing behavior, unchanged)

2. **Given** a trained Logistic Regression model is selected
   **When** the user views the Feature Importance section
   **Then** coefficient absolute values are shown as feature importance, paired with `feature_names_`

3. **Given** a trained Elo model is selected
   **When** the user views the Feature Importance section
   **Then** team rating values are shown as interpretability information (top-N teams by rating)
   **And** the display communicates that these are Elo ratings, not traditional feature importances

4. **Given** the Elo Model Deep Dive page
   **When** the Feature Importance section renders
   **Then** the `"Feature importance is not available for stateful models."` message is replaced with meaningful Elo interpretability

## Tasks / Subtasks

- [x] Task 1: Add `get_feature_importances()` to `LogisticRegressionModel` (AC: #2)
  - [x] 1.1 Override `get_feature_importances()` in `logistic_regression.py`
  - [x] 1.2 Return `list[tuple[str, float]]` pairing `feature_names_` with `abs(self._clf.coef_[0])`
  - [x] 1.3 Return `None` if `feature_names_` is empty (unfitted model)
  - [x] 1.4 Add `import numpy as np` (needed for `np.abs`)

- [ ] Task 2: Add `get_feature_importances()` to `EloModel` (AC: #3, #4)
  - [ ] 2.1 Override `get_feature_importances()` in `elo.py`
  - [ ] 2.2 Return team ratings as `list[tuple[str, float]]` — format: `("team_{id}", rating)`
  - [ ] 2.3 Sort descending by rating, limit to top 50 teams (avoid overwhelming the chart)
  - [ ] 2.4 Return `None` if no ratings exist (fresh model, never fitted)

- [ ] Task 3: Update dashboard to handle model-specific interpretability (AC: #3, #4)
  - [ ] 3.1 Update `_render_feature_importance()` in `3_Model_Deep_Dive.py` to show model-appropriate title/labels
  - [ ] 3.2 For `logistic_regression`: title "Feature Importance (|Coefficient|)", x-axis "Absolute Coefficient"
  - [ ] 3.3 For `elo`: title "Team Elo Ratings (Top 50)", x-axis "Rating"
  - [ ] 3.4 For `xgboost`: keep existing title "Feature Importance (Gain)" — unchanged
  - [ ] 3.5 Remove the generic `"not available for stateful models"` fallback message (line 96)
  - [ ] 3.6 Pass `model_type` context into the rendering to select chart title/axis labels

- [ ] Task 4: Update `load_feature_importances()` sort logic (AC: #3)
  - [ ] 4.1 The current sort in `data_loaders.py:192` sorts by `p[1]` descending — this works for XGBoost (higher = more important) and LogReg (higher abs coef = more important)
  - [ ] 4.2 For Elo, the model's `get_feature_importances()` should return data pre-sorted by rating descending, so the dashboard sort still works (highest rated teams first)
  - [ ] 4.3 No changes needed to `load_feature_importances()` itself — the model contract handles it

- [x] Task 5: Add unit tests for LogisticRegression feature importance (AC: #2)
  - [x] 5.1 Test: `get_feature_importances()` returns `None` before `fit()`
  - [x] 5.2 Test: after `fit()`, returns `list[tuple[str, float]]` with correct feature names
  - [x] 5.3 Test: returned values are absolute coefficient magnitudes (all non-negative)
  - [x] 5.4 Test: length matches `len(feature_names_)`
  - [x] 5.5 Test: save/load round-trip preserves `get_feature_importances()` behavior

- [ ] Task 6: Add unit tests for Elo interpretability (AC: #3, #4)
  - [ ] 6.1 Test: `get_feature_importances()` returns `None` on fresh model (no ratings)
  - [ ] 6.2 Test: after `fit()` or manual `set_ratings()`, returns `list[tuple[str, float]]`
  - [ ] 6.3 Test: returned entries are `("team_{id}", rating_value)` format
  - [ ] 6.4 Test: results are sorted descending by rating
  - [ ] 6.5 Test: limit to top 50 when more than 50 teams exist

- [ ] Task 7: Update dashboard tests (AC: #3, #4)
  - [ ] 7.1 Update `test_returns_empty_for_elo_model` in `test_dashboard_filters.py` — Elo now returns data, not empty
  - [ ] 7.2 Add test for logistic_regression model_type in dashboard rendering

- [ ] Task 8: Verify existing XGBoost behavior unchanged (AC: #1)
  - [ ] 8.1 Run existing XGBoost feature importance tests — all must pass without changes
  - [ ] 8.2 Run full test suite — baseline: 964 tests (from Story 9.2)

## Dev Notes

### Current State

**XGBoost** — `get_feature_importances()` already implemented at `xgboost_model.py:211-216`:
```python
def get_feature_importances(self) -> list[tuple[str, float]] | None:
    if not self._is_fitted or not self.feature_names_:
        return None
    importances = self._clf.feature_importances_
    return list(zip(self.feature_names_, importances.tolist()))
```

**LogisticRegression** — inherits base class `return None`. Has `feature_names_` (set in `fit()` at line 67) and `self._clf` is sklearn `LogisticRegression` which exposes `.coef_` after fitting.

**Elo** — inherits base class `return None`. Has no `feature_names_` (stateful model). Has `self._engine.get_all_ratings() -> dict[int, float]` which returns team→rating mapping.

### Data Flow: Model → Dashboard

1. Dashboard calls `load_feature_importances(data_dir, run_id)` at `data_loaders.py:158-195`
2. This calls `model.get_feature_importances()` (the method we're adding/overriding)
3. If `None`, falls back to legacy `_clf.feature_importances_` path
4. Returns `list[dict[str, object]]` with keys `"feature"` and `"importance"`, sorted descending
5. `_render_feature_importance()` at `3_Model_Deep_Dive.py:88-110` renders a horizontal bar chart

### LogisticRegression Implementation Pattern

```python
# In logistic_regression.py — add after get_config() method
def get_feature_importances(self) -> list[tuple[str, float]] | None:
    """Return absolute coefficient values as feature importance."""
    if not self.feature_names_:
        return None
    coefs = np.abs(self._clf.coef_[0])
    return list(zip(self.feature_names_, coefs.tolist()))
```

**Why `abs(coef_[0])`**: For binary classification, `coef_` has shape `(1, n_features)`. The magnitude indicates importance; the sign indicates direction (positive = increases P(team_a wins)). The dashboard already sorts by value, so absolute values rank features by strength of effect.

**Note**: `numpy` is not currently imported in `logistic_regression.py` — add `import numpy as np` with appropriate `# type: ignore[import-untyped]` comment.

### Elo Implementation Pattern

```python
# In elo.py — add after get_config() method
def get_feature_importances(self) -> list[tuple[str, float]] | None:
    """Return top team Elo ratings as interpretability information."""
    ratings = self._engine.get_all_ratings()
    if not ratings:
        return None
    sorted_ratings = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    top_n = sorted_ratings[:50]
    return [(f"team_{team_id}", rating) for team_id, rating in top_n]
```

**Design decision — team ID as string**: The model only knows numeric team IDs (e.g., `1181`). Mapping to team names (e.g., "Duke") requires loading team reference data, which is outside the model's responsibility. The dashboard could optionally resolve these names later, but the model returns `"team_{id}"` format.

### Dashboard Changes

The `_render_feature_importance()` function at `3_Model_Deep_Dive.py:88-110` needs model-type-aware titles:

```python
def _render_feature_importance(data_dir: str, run_id: str, model_type: str) -> None:
    st.subheader("Feature Importance")
    importances = load_feature_importances(data_dir, run_id)
    if not importances:
        st.info("Feature importance not available. Re-run training to persist model artifacts.")
        return

    # Model-type-aware chart configuration
    if model_type == "elo":
        chart_title = "Team Elo Ratings (Top 50)"
        x_label = "Rating"
    elif model_type == "logistic_regression":
        chart_title = "Feature Importance (|Coefficient|)"
        x_label = "Absolute Coefficient"
    else:
        chart_title = "Feature Importance (Gain)"
        x_label = "Importance"

    feature_names = [d["feature"] for d in importances]
    importance_values = [d["importance"] for d in importances]
    fig = go.Figure(go.Bar(...))
    fig.update_layout(title=chart_title, xaxis_title=x_label, ...)
```

**Key change**: The old code had two fallback messages (lines 93-96) — one for XGBoost and one for "stateful models". Since all model types now return data, simplify to a single generic message.

### Files to Modify

| File | Change |
|------|--------|
| `src/ncaa_eval/model/logistic_regression.py` | Add `get_feature_importances()`, add `numpy` import |
| `src/ncaa_eval/model/elo.py` | Add `get_feature_importances()` |
| `dashboard/pages/3_Model_Deep_Dive.py` | Model-type-aware chart titles, remove "stateful models" message |
| `tests/unit/test_model_logistic_regression.py` | Add `TestFeatureImportance` class |
| `tests/unit/test_model_elo.py` | Add `TestFeatureImportance` class |
| `tests/unit/test_dashboard_filters.py` | Update Elo test, add LogReg test |

### Files NOT to Modify

| File | Reason |
|------|--------|
| `src/ncaa_eval/model/base.py` | `get_feature_importances()` already has correct signature and default |
| `src/ncaa_eval/model/xgboost_model.py` | AC #1 — existing behavior unchanged |
| `dashboard/lib/data_loaders.py` | `load_feature_importances()` already calls `model.get_feature_importances()` and sorts; no changes needed |

### Project Structure Notes

- All model files are in `src/ncaa_eval/model/`
- Dashboard pages in `dashboard/pages/`
- Dashboard data loading in `dashboard/lib/data_loaders.py`
- Unit tests mirror source structure under `tests/unit/`
- `from __future__ import annotations` required in all Python files
- `mypy --strict` required for all `src/` and `tests/` files
- Google-style docstrings; no type duplication in docstrings

### Conventions

- `get_feature_importances()` return type: `list[tuple[str, float]] | None` — defined by base class
- `feature_names_` follows sklearn trailing-underscore convention for fitted attributes
- `# type: ignore[import-untyped]` on `numpy` and `pandas` imports (project convention)
- Test class naming: `TestFeatureImportance` (Story 9.3 scope)
- Plotly charts use `TEMPLATE` and `COLOR_GREEN` from `ncaa_eval.evaluation.plotting`

### References

- [Source: src/ncaa_eval/model/base.py:72-78] — `get_feature_importances()` base class method
- [Source: src/ncaa_eval/model/xgboost_model.py:211-216] — XGBoost reference implementation
- [Source: src/ncaa_eval/model/logistic_regression.py:63-68] — `feature_names_` and `_clf` attributes
- [Source: src/ncaa_eval/model/elo.py:93-98] — `get_state()` / `get_all_ratings()`
- [Source: dashboard/pages/3_Model_Deep_Dive.py:88-110] — Current `_render_feature_importance()`
- [Source: dashboard/lib/data_loaders.py:158-195] — `load_feature_importances()` pipeline
- [Source: tests/unit/test_dashboard_filters.py:306-416] — Existing feature importance tests

### Previous Story Intelligence (Story 9.2)

- **DRY pattern**: Story 9.2 extracted shared `_feature_config_io.py` helper to avoid triplicating save/load logic across 3 models. Feature importance is simpler (no shared serialization needed) — each model's override is self-contained (~5 lines).
- **`feature_names_` already exists on LogReg**: Set in `fit()` at line 67, persisted via `feature_names.json` in `save()`/`load()`. The new `get_feature_importances()` just reads it — no new persistence needed.
- **Test count baseline**: 964 tests passed at Story 9.2 completion. All must continue to pass.
- **Pre-commit ruff-format**: First commit may auto-fix formatting. Re-stage after.

### Git Intelligence

- Latest commit: `3f79949 feat(model): embed FeatureConfig as model-level concern (Story 9.2)`
- No in-flight changes on main. Clean starting point.
- Story 9.2 touched all 3 model files — developer should read the current state of each file before modifying (don't assume code from the story template is current).

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6 (claude-opus-4-6)

### Debug Log References

### Completion Notes List

### File List
