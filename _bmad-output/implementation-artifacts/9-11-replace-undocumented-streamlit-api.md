# Story 9.11: Replace Undocumented Streamlit API Usage

Status: review

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

- [x] Task 1: Replace undocumented API with typed, documented equivalent in `1_Lab.py` (AC: #1, #2, #3, #4)
  - [x] 1.1: Import `DataframeState` from `streamlit.elements.arrow` — NOT NEEDED: Streamlit 1.54.0 has `@overload` signatures that automatically narrow `st.dataframe()` return to `DataframeState` when `on_select="rerun"` is passed
  - [x] 1.2: Use `typing.cast()` — NOT NEEDED: mypy reports `redundant-cast` because the overload already narrows the type
  - [x] 1.3: Access `event["selection"]["rows"]` using dict-style safe `.get()` access — `event.get("selection", {}).get("rows", [])`
  - [x] 1.4: Remove both `# type: ignore[attr-defined]` comments
  - [x] 1.5: Verify `mypy --strict dashboard/pages/1_Lab.py` passes cleanly — confirmed clean

- [x] Task 2: Update test mocks in `test_leaderboard_page.py` (AC: #5)
  - [x] 2.1: Updated `mock_st.dataframe.return_value` to dict-style return: `{"selection": {"rows": []}}`
  - [x] 2.2: All 7 tests pass in `tests/unit/test_leaderboard_page.py`

- [x] Task 3: Run full quality gates (AC: #4, #5)
  - [x] 3.1: `mypy --strict src/ncaa_eval tests dashboard` — clean (118 source files)
  - [x] 3.2: `ruff check .` — clean
  - [x] 3.3: `pytest` — 1114 passed, 1 skipped, no regressions

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

Claude Opus 4.6

### Debug Log References

- mypy initially reported `redundant-cast` when using `typing.cast("DataframeState", ...)` — investigation revealed Streamlit 1.54.0 has `@overload` decorators on `ArrowMixin.dataframe()` that narrow the return type to `DataframeState` when `on_select=Literal["rerun"]` is passed. No cast or TYPE_CHECKING import needed.
- ruff flagged C901 complexity (11 > 10) when using two-level nested `if` for dict access. Simplified to single-line chained `.get()` to reduce complexity back to 10.

### Completion Notes List

- Replaced attribute-style access (`event.selection.rows`) with dict-style safe access (`event.get("selection", {}).get("rows", [])`) — proper TypedDict usage
- Removed both `# type: ignore[attr-defined]` comments
- No new imports needed — Streamlit's `@overload` signatures already provide full type narrowing
- Updated test mocks from `MagicMock(selection=MagicMock(rows=[]))` to `{"selection": {"rows": []}}` to match dict-style access
- Key discovery: The story's Dev Notes suggested cast/TYPE_CHECKING import, but Streamlit's overloads made these unnecessary — simpler solution

### Change Log

- 2026-03-11: Replaced undocumented attribute-style Streamlit dataframe selection API with typed dict-style access; removed type: ignore comments; updated test mocks (Story 9.11)

### File List

- `dashboard/pages/1_Lab.py` — MODIFIED: replaced `event.selection.rows` with `event.get("selection", {}).get("rows", [])`, removed `# type: ignore[attr-defined]`
- `tests/unit/test_leaderboard_page.py` — MODIFIED: updated `mock_st.dataframe.return_value` from MagicMock to dict
- `_bmad-output/implementation-artifacts/9-11-replace-undocumented-streamlit-api.md` — MODIFIED: task checkboxes, dev agent record, status
- `_bmad-output/implementation-artifacts/sprint-status.yaml` — MODIFIED: story status ready-for-dev → review
