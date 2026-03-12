# Story 9.11: Replace Undocumented Streamlit API Usage

Status: ready-for-dev

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a **developer**,
I want to **replace the undocumented `event.selection.rows` Streamlit API in the leaderboard with the official documented alternative**,
so that **the dashboard does not break on Streamlit upgrades and all API usage is supported and type-safe**.

## Acceptance Criteria

1. **Given** the leaderboard page uses `event.selection.rows` with `# type: ignore[attr-defined]`, **when** the developer investigates the official Streamlit selection API, **then** the undocumented API call is replaced with the documented, type-safe equivalent.

2. **Given** the replacement is implemented, **then** the two `# type: ignore[attr-defined]` comments on lines 137-138 of `dashboard/pages/1_Lab.py` are removed.

3. **Given** the replacement is implemented, **then** the leaderboard click-to-navigate behavior is preserved: clicking a row sets `st.session_state["selected_run_id"]` and calls `st.switch_page("pages/3_Model_Deep_Dive.py")`.

4. **Given** the replacement is implemented, **then** `mypy --strict` passes on `dashboard/pages/1_Lab.py` with no new `# type: ignore` comments (existing `# type: ignore[import-untyped]` for pandas is acceptable).

5. **Given** the replacement is implemented, **then** all existing tests in `tests/unit/test_leaderboard_page.py` pass with any necessary mock updates.

## Tasks / Subtasks

- [ ] Task 1: Replace undocumented API with typed, documented equivalent in `1_Lab.py` (AC: #1, #2, #3, #4)
  - [ ] 1.1: Import `DataframeState` from `streamlit.elements.arrow` (use `TYPE_CHECKING` guard for runtime safety)
  - [ ] 1.2: Use `typing.cast()` to narrow `st.dataframe()` return from `DeltaGenerator | DataframeState` to `DataframeState` (since `on_select="rerun"` guarantees `DataframeState` return)
  - [ ] 1.3: Access `event["selection"]["rows"]` using dict-style access (TypedDict supports this with proper mypy typing) — OR use attribute access if mypy resolves correctly after cast
  - [ ] 1.4: Remove both `# type: ignore[attr-defined]` comments
  - [ ] 1.5: Verify `mypy --strict dashboard/pages/1_Lab.py` passes cleanly

- [ ] Task 2: Update test mocks in `test_leaderboard_page.py` (AC: #5)
  - [ ] 2.1: Update `mock_st.dataframe.return_value` to use dict-style return matching `DataframeState` TypedDict shape: `{"selection": {"rows": []}}` — OR keep `MagicMock(selection=MagicMock(rows=[]))` if attribute access is used
  - [ ] 2.2: Run `pytest tests/unit/test_leaderboard_page.py` to confirm all tests pass

- [ ] Task 3: Run full quality gates (AC: #4, #5)
  - [ ] 3.1: Run `mypy --strict src/ncaa_eval tests dashboard`
  - [ ] 3.2: Run `ruff check .`
  - [ ] 3.3: Run `pytest` (full suite)

## Dev Notes

### The Problem

`dashboard/pages/1_Lab.py` lines 128-141 use the Streamlit `st.dataframe()` selection API with `# type: ignore[attr-defined]` comments:

```python
event = st.dataframe(
    styled,
    use_container_width=True,
    on_select="rerun",
    selection_mode="single-row",
    key="leaderboard_selection",
)

if event and event.selection and event.selection.rows:  # type: ignore[attr-defined]
    selected_idx = event.selection.rows[0]  # type: ignore[attr-defined]
```

### Research Finding: The API IS Officially Documented

As of Streamlit 1.54.0 (the installed version), the `on_select`, `selection_mode`, and `event.selection.rows` API is **officially documented and typed**. The `# type: ignore` comments exist because mypy cannot automatically narrow the return type union.

**Return type:** `st.dataframe()` returns `DeltaGenerator | DataframeState` depending on whether `on_select` is `"ignore"` or `"rerun"`.

**Type definitions** (in `streamlit.elements.arrow`):

```python
class DataframeSelectionState(TypedDict, total=False):
    rows: list[int]
    columns: list[str]
    cells: list[tuple[int, str]]

class DataframeState(TypedDict, total=False):
    selection: DataframeSelectionState
```

Both are `TypedDict` subclasses — they support dict-style access (`event["selection"]["rows"]`) and attribute access (`event.selection.rows`) at runtime.

### The Fix: `typing.cast()` to Narrow the Union

Since `on_select="rerun"` guarantees the return is `DataframeState` (not `DeltaGenerator`), use `cast()` to tell mypy:

```python
from __future__ import annotations
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from streamlit.elements.arrow import DataframeState

event = cast("DataframeState", st.dataframe(
    styled,
    use_container_width=True,
    on_select="rerun",
    selection_mode="single-row",
    key="leaderboard_selection",
))

# Now mypy knows event is DataframeState — no type: ignore needed
selected_rows = event.get("selection", {}).get("rows", [])
if selected_rows:
    selected_idx = selected_rows[0]
    selected_run_id = str(display_df.iloc[selected_idx]["run_id"])
    st.session_state["selected_run_id"] = selected_run_id
    st.switch_page("pages/3_Model_Deep_Dive.py")
```

**IMPORTANT:** Use `event.get("selection", {}).get("rows", [])` (dict-style safe access) rather than `event["selection"]["rows"]` to handle the `total=False` TypedDict gracefully — keys may be absent before any selection occurs.

**Alternative:** If dict-style `.get()` causes mypy issues with TypedDict, use:
```python
selection = event.get("selection")
if selection:
    rows = selection.get("rows", [])
    if rows:
        ...
```

### Import Pattern for `DataframeState`

Import from `streamlit.elements.arrow` — this is the module where the classes are defined:

```python
if TYPE_CHECKING:
    from streamlit.elements.arrow import DataframeState
```

The `TYPE_CHECKING` guard means the import only runs during mypy analysis, not at runtime. This avoids coupling to Streamlit's internal module structure at runtime.

### Test Mock Update

The existing test mocks use:
```python
mock_st.dataframe.return_value = MagicMock(selection=MagicMock(rows=[]))
```

If switching to dict-style access, update mocks to return dict-like objects:
```python
mock_st.dataframe.return_value = {"selection": {"rows": []}}
```

If keeping attribute-style access (which works at runtime since the return object supports both), the existing mocks remain valid.

### Key File Locations

| File | Action |
|------|--------|
| `dashboard/pages/1_Lab.py` | **MODIFY** — lines 128-141: add `cast()` import, narrow return type, remove `# type: ignore` |
| `tests/unit/test_leaderboard_page.py` | **MODIFY** — update `mock_st.dataframe.return_value` if access pattern changes |

### Project Structure Notes

- No new files or directories needed
- `from __future__ import annotations` already present in `1_Lab.py`
- `mypy --strict` must pass on all dashboard files
- `ruff check` must pass

### Previous Story Intelligence (9.10)

- Quality gates: 1112 tests passing, ruff clean, mypy --strict clean (102 files)
- Story 9.10 moved `get_metric_cols` to shared `dashboard/lib/data_loaders.py` — `1_Lab.py` imports from there
- Test pattern: mock `st` module via `patch.object(_lab_mod, "st", mock_st)` — works well, keep this pattern
- Review feedback from 9.10: watch for hardcoded values that should be dynamic

### Git Intelligence

Recent commits follow `feat(scope): description (Story X.Y)` pattern. Stories 9.1-9.10 are done, all merged to main. This is a minor refactor story — scope is `dashboard`.

### What NOT to Implement

- Do NOT change the functional behavior of the leaderboard selection
- Do NOT change `on_select="rerun"` or `selection_mode="single-row"` parameters
- Do NOT modify other `st.dataframe` calls in other pages (they don't use selection)
- Do NOT add a Streamlit version pin constraint — the API is already documented in the installed version (1.54.0) and covered by the existing `>=1.36,<2` constraint

### References

- [Source: `dashboard/pages/1_Lab.py:128-141` — current undocumented API usage with `# type: ignore`]
- [Source: `streamlit.elements.arrow:94-179` — `DataframeSelectionState` and `DataframeState` TypedDict definitions]
- [Source: `tests/unit/test_leaderboard_page.py:100,122` — existing mocks for `st.dataframe.return_value`]
- [Source: Streamlit docs — `st.dataframe` API reference: https://docs.streamlit.io/develop/api-reference/data/st.dataframe]
- [Source: Streamlit docs — dataframe row selections tutorial: https://docs.streamlit.io/develop/tutorials/elements/dataframe-row-selections]
- [Source: `_bmad-output/planning-artifacts/epics.md#Story 9.11` — acceptance criteria and PO decision]
- [Source: Audit item 2.14; PO decision 2026-03-11 (A — Rewrite to use official Streamlit API)]

## Dev Agent Record

### Agent Model Used

{{agent_model_name_version}}

### Debug Log References

### Completion Notes List

### Change Log

### File List
