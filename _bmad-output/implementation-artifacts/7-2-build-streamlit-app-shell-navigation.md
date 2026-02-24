# Story 7.2: Build Streamlit App Shell & Navigation

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a data scientist,
I want a Streamlit multipage app with sidebar navigation, dark mode, and persistent global filters,
So that I can seamlessly switch between Lab and Presentation views while maintaining context.

## Acceptance Criteria

1. **Given** the dashboard application is launched via `streamlit run dashboard/app.py`, **When** the user opens the application in a browser, **Then** the app renders in Dark Mode by default with Wide Mode layout.

2. **Given** the app is running, **When** the user views the sidebar, **Then** a persistent sidebar provides navigation between "Lab" and "Presentation" sections with appropriate page entries for downstream stories.

3. **Given** the sidebar is visible, **When** the user interacts with global filters, **Then** Tournament Year, Model Version, and Scoring Format selectors are available and their selections persist across page navigation via `st.session_state`.

4. **Given** the user selects a filter value, **When** they navigate to a different page, **Then** the filter selections are preserved and applied on the new page.

5. **Given** the app is running, **When** heavy datasets are loaded (model results, game data), **Then** `@st.cache_data` is used to ensure sub-500ms interaction response.

6. **Given** the app is running, **When** the dashboard renders data, **Then** it imports and calls `ncaa_eval` functions exclusively (no direct file IO in the dashboard layer).

7. **Given** the app is running, **When** the user views data tables, **Then** monospace fonts are applied per the UX spec.

## Tasks / Subtasks

- [x] Task 1: Create `.streamlit/config.toml` for dark theme and wide layout defaults (AC: #1)
  - [x] 1.1 Create `.streamlit/` directory in project root
  - [x] 1.2 Configure `[theme]` section with project color palette (primaryColor `#28a745`, dark backgrounds)
  - [x] 1.3 Configure `[server]` section (runOnSave=false)
  - [x] 1.4 Configure `[client]` section (toolbarMode="minimal")

- [x] Task 2: Implement `dashboard/app.py` — the app entrypoint and navigation router (AC: #1, #2, #3, #4)
  - [x] 2.1 Call `st.set_page_config(page_title="NCAA Eval", page_icon=":material/sports_basketball:", layout="wide", initial_sidebar_state="expanded")` as the FIRST Streamlit command
  - [x] 2.2 Define page objects using `st.Page` for each section: Home (default), Lab pages (Leaderboard placeholder, Model Deep Dive placeholder), Presentation pages (Bracket Visualizer placeholder, Pool Scorer placeholder)
  - [x] 2.3 Configure `st.navigation()` with a dict to group pages into "Lab" and "Presentation" sections in the sidebar
  - [x] 2.4 Call `pg.run()` to execute the selected page
  - [x] 2.5 Render global filter widgets in `st.sidebar` BEFORE `pg.run()` (so filters appear on every page): Tournament Year selectbox, Model Run selectbox, Scoring Format selectbox
  - [x] 2.6 Initialize and read/write all filter values via `st.session_state` for cross-page persistence

- [x] Task 3: Implement `dashboard/lib/filters.py` — shared data-loading and filter logic (AC: #3, #5, #6)
  - [x] 3.1 Create `dashboard/lib/` package with `__init__.py`
  - [x] 3.2 Implement `load_available_years() -> list[int]` using `@st.cache_data` — calls `ParquetRepository.get_seasons()` and extracts years
  - [x] 3.3 Implement `load_available_runs() -> list[ModelRun]` using `@st.cache_data` — calls `RunStore.list_runs()`
  - [x] 3.4 Implement `load_available_scorings() -> list[str]` using `@st.cache_data` — calls `list_scorings()`
  - [x] 3.5 Implement `get_data_dir() -> Path` helper that resolves the project's `data/` directory for RunStore/ParquetRepository construction

- [x] Task 4: Implement Home page callable (AC: #1, #2)
  - [x] 4.1 Create `dashboard/pages/home.py` with a welcome page showing project title, brief description, and navigation hints
  - [x] 4.2 Display placeholder summary cards showing available model runs count, available seasons, etc. using `st.metric`

- [x] Task 5: Convert existing page stubs to work with `st.navigation` pattern (AC: #2)
  - [x] 5.1 Rewrite `dashboard/pages/1_Lab.py` as a proper page function or script that reads global filters from `st.session_state` and displays a placeholder ("Backtest Leaderboard — coming in Story 7.3")
  - [x] 5.2 Rewrite `dashboard/pages/2_Presentation.py` as a proper page function or script that reads global filters from `st.session_state` and displays a placeholder ("Bracket Visualizer — coming in Story 7.5")
  - [x] 5.3 Add additional placeholder page files for remaining Epic 7 pages: `3_Model_Deep_Dive.py` (Story 7.4), `4_Pool_Scorer.py` (Story 7.6)

- [x] Task 6: Apply monospace font styling (AC: #7)
  - [x] 6.1 Add custom CSS via `st.markdown(unsafe_allow_html=True)` in `app.py` to apply monospace font (`IBM Plex Mono`, fallback to system monospace) to `st.dataframe` and `st.table` elements
  - [x] 6.2 Define shared CSS string in `dashboard/lib/styles.py` for reuse across pages

- [x] Task 7: Write tests (AC: all)
  - [x] 7.1 Create `tests/unit/test_dashboard_filters.py` — test data-loading functions with mocked RunStore/ParquetRepository (return type validation, caching behavior)
  - [x] 7.2 Create `tests/unit/test_dashboard_app.py` — smoke test that `app.py` imports without error, page modules are importable
  - [x] 7.3 Test that session state keys are initialized correctly
  - [x] 7.4 Test filter loader functions return correct types

- [x] Task 8: Verify quality gates (AC: all)
  - [x] 8.1 `mypy --strict src/ncaa_eval tests` (dashboard/ is NOT under mypy --strict — it's a Streamlit app, not library code; but run `mypy dashboard/` non-strict to catch obvious errors)
  - [x] 8.2 `ruff check .`
  - [x] 8.3 Full test suite passes

## Dev Notes

### Streamlit Version & Multipage API

**Installed:** Streamlit 1.54.0 (latest as of Feb 2026).

**Use `st.navigation()` + `st.Page()` API** — this is the modern, recommended approach for multipage apps in Streamlit >=1.36. Do NOT use the legacy `pages/` directory auto-discovery pattern.

Key pattern for the entrypoint (`dashboard/app.py`):

```python
import streamlit as st

st.set_page_config(
    page_title="NCAA Eval",
    page_icon=":material/sports_basketball:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define pages
home = st.Page("pages/home.py", title="Home", icon=":material/home:", default=True)
leaderboard = st.Page("pages/1_Lab.py", title="Backtest Leaderboard", icon=":material/leaderboard:")
deep_dive = st.Page("pages/3_Model_Deep_Dive.py", title="Model Deep Dive", icon=":material/analytics:")
bracket = st.Page("pages/2_Presentation.py", title="Bracket Visualizer", icon=":material/account_tree:")
pool_scorer = st.Page("pages/4_Pool_Scorer.py", title="Pool Scorer", icon=":material/calculate:")

pg = st.navigation({
    "": [home],
    "Lab": [leaderboard, deep_dive],
    "Presentation": [bracket, pool_scorer],
})

# Global filters in sidebar (rendered on every page)
with st.sidebar:
    # ... filter widgets ...
    pass

pg.run()
```

**Critical:** `st.set_page_config()` MUST be the first Streamlit command. `st.navigation()` is called exactly once in the entrypoint. All page files are scripts executed via `pg.run()`.

### Navigation API Details

- `st.navigation(dict)` groups pages into labeled sidebar sections. Use `""` (empty string) as key for ungrouped pages (e.g., Home).
- `st.Page(source, title, icon, default)` — source is a relative file path (relative to entrypoint `dashboard/app.py`) or a callable.
- `position="sidebar"` is the default (no need to specify).
- Icons use Material Design syntax: `:material/icon_name:`.

### Global Filters — Session State Pattern

Filters must persist across page navigation. Use `st.session_state` with explicit initialization:

```python
# In app.py, BEFORE rendering sidebar widgets:
if "selected_year" not in st.session_state:
    st.session_state.selected_year = None  # or latest available year
if "selected_run_id" not in st.session_state:
    st.session_state.selected_run_id = None
if "selected_scoring" not in st.session_state:
    st.session_state.selected_scoring = "standard"

# Sidebar widgets bound to session state via `key=` parameter:
with st.sidebar:
    st.selectbox("Tournament Year", options=years, key="selected_year")
    st.selectbox("Model Run", options=run_options, key="selected_run_id")
    st.selectbox("Scoring Format", options=scorings, key="selected_scoring")
```

**Important:** Use the `key=` parameter on widgets to bind directly to session state. Do NOT manually assign `st.session_state.x = widget_value` — Streamlit handles this automatically when you provide a `key`.

### Data Loading via `ncaa_eval` API — NO Direct File IO

The dashboard MUST call `ncaa_eval` functions, never read files directly. Key API surfaces:

| Dashboard Need | ncaa_eval API Call | Module |
|:---|:---|:---|
| List available seasons | `ParquetRepository(data_dir).get_seasons()` | `ingest.repository` |
| List model runs | `RunStore(data_dir).list_runs()` | `model.tracking` |
| List scoring formats | `list_scorings()` | `evaluation` (re-exported) |
| Load backtest result | `RunStore(data_dir).load_run(run_id)` | `model.tracking` |
| Load predictions | `RunStore(data_dir).load_predictions(run_id)` | `model.tracking` |

**Data directory resolution:** The `data/` directory is at project root. Use `Path(__file__).resolve().parent.parent / "data"` from `dashboard/app.py` to find it. Encapsulate in `dashboard/lib/filters.py:get_data_dir()`.

### Caching Strategy

Use `@st.cache_data` for all data-loading functions. These decorators cache based on input arguments and survive page navigation:

```python
@st.cache_data(ttl=300)  # 5 min TTL for run listing (may change during session)
def load_available_runs(data_dir: str) -> list[dict]:
    store = RunStore(Path(data_dir))
    return [run.model_dump() for run in store.list_runs()]
```

**Note:** `@st.cache_data` requires serializable arguments and return values. Pass `str(data_dir)` not `Path`, and return dicts/lists not Pydantic models directly (serialize via `.model_dump()`).

### Dark Mode & Theme Configuration

Create `.streamlit/config.toml` at project root:

```toml
[theme]
primaryColor = "#28a745"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#1C2128"
textColor = "#FAFAFA"
font = "monospace"

[server]
runOnSave = false

[client]
toolbarMode = "minimal"
```

This enforces dark mode by default. The `font = "monospace"` setting applies monospace globally per UX spec.

### Monospace Font CSS Supplement

For more precise control over data tables:

```python
CUSTOM_CSS = """
<style>
    .stDataFrame, .stTable {
        font-family: 'IBM Plex Mono', 'Fira Code', 'Consolas', monospace !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
```

### Color Palette Constants

Reuse from `ncaa_eval.evaluation.plotting`:

```python
from ncaa_eval.evaluation import COLOR_GREEN, COLOR_RED, COLOR_NEUTRAL
```

These are already defined: `#28a745`, `#dc3545`, `#6c757d`.

### Dashboard File Structure

```
dashboard/
├── app.py                    # Entrypoint: st.set_page_config, st.navigation, sidebar filters
├── lib/
│   ├── __init__.py
│   ├── filters.py            # Data-loading functions with @st.cache_data
│   └── styles.py             # Shared CSS constants
├── pages/
│   ├── home.py               # NEW: Welcome/summary page (default)
│   ├── 1_Lab.py              # REWRITE: Leaderboard placeholder (Story 7.3)
│   ├── 2_Presentation.py     # REWRITE: Bracket placeholder (Story 7.5)
│   ├── 3_Model_Deep_Dive.py  # NEW: Deep dive placeholder (Story 7.4)
│   └── 4_Pool_Scorer.py      # NEW: Pool scorer placeholder (Story 7.6)
└── components/
    └── __init__.py            # Existing stub — keep for future widget components
```

### Existing Code — DO NOT Reimplement

- `evaluation/plotting.py`: All Plotly visualization functions — USE for downstream stories, not this one
- `evaluation/simulation.py`: `list_scorings()` — USE for scoring format filter
- `model/tracking.py`: `RunStore`, `ModelRun` — USE for model run filter
- `ingest/repository.py`: `ParquetRepository` — USE for year filter
- `evaluation/__init__.py`: Re-exports all the above — import from here

### mypy Considerations

- Dashboard files (`dashboard/`) are NOT part of `mypy --strict` scope (they're Streamlit scripts, not library code)
- Run `mypy dashboard/` in non-strict mode to catch obvious type errors
- `streamlit` ships with type stubs as of v1.38+ — `import streamlit as st` should type-check without `type: ignore`
- `from __future__ import annotations` is still required in all files per project convention

### Edge Cases

- **No data directory:** If `data/` doesn't exist (fresh clone), filters should show empty lists with a helpful message ("No data available — run `python sync.py` first")
- **No model runs:** If RunStore returns empty, Model Run filter shows "No runs available"
- **Session state initialization race:** `st.set_page_config` MUST come before any other `st` call. Initialize session state defaults AFTER page config but BEFORE sidebar widgets.

### Previous Story Learnings (Story 7.1)

- **Google docstring style**: NOT NumPy style. Use `Args:`, `Returns:`, `Raises:`.
- **Frozen dataclasses**: Result containers are frozen. Use standalone functions.
- **`from __future__ import annotations`**: Required in ALL Python files.
- **Color constants**: Already defined in `evaluation.plotting` — reuse, don't redefine.
- **`plotly_dark` template**: Already set up in plotting module. Dashboard should use `st.plotly_chart(fig, use_container_width=True)` for responsive Plotly rendering.
- **tqdm progress bars**: Added to `run_backtest(progress=True)` and `simulate_tournament_mc(progress=True)` in Story 7.1.

### Performance Targets

| Operation | Target |
|:---|:---|
| App initial load (cold start) | < 3 seconds |
| Page navigation (cached data) | < 500 ms |
| Filter change re-render | < 500 ms |
| Data loading (cached) | < 100 ms |

### Architecture Constraints

- **Type Sharing (§12 Architecture):** Data structures passed between Logic and UI use Pydantic models or TypedDicts.
- **No Direct IO in UI (§12):** Dashboard MUST call `ncaa_eval` functions — never `open()`, `pd.read_parquet()`, etc.
- **Vectorization (NFR1):** Not directly relevant to this story (no metric calculations).
- **Desktop Only (§5.1 UX Spec):** Wide mode enforced, mobile not supported.
- **`from __future__ import annotations`:** Required in all Python files.

### Project Structure Notes

- Dashboard lives in `dashboard/` at repo root (NOT under `src/ncaa_eval/`)
- Dashboard is NOT an installable package — it's a Streamlit app that imports `ncaa_eval`
- The `dashboard/lib/` package is internal to the dashboard (not importable from `ncaa_eval`)
- `.streamlit/config.toml` goes at project root (Streamlit auto-discovers it)

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 7.2 — acceptance criteria]
- [Source: specs/05-architecture-fullstack.md §7 — Frontend Architecture (Streamlit multipage, session state, caching)]
- [Source: specs/05-architecture-fullstack.md §9 — Unified Project Structure (dashboard/ layout)]
- [Source: specs/05-architecture-fullstack.md §12 — Coding Standards (type sharing, no direct IO in UI)]
- [Source: specs/04-front-end-spec.md §2.2 — Navigation Structure (sidebar, contextual filters, breadcrumbs)]
- [Source: specs/04-front-end-spec.md §4.2 — Branding & Style (dark mode, color palette, monospace fonts)]
- [Source: specs/04-front-end-spec.md §5 — Responsiveness & Performance (desktop only, 500ms target, caching)]
- [Source: _bmad-output/implementation-artifacts/7-1-build-plotly-adapters-jupyter-lab-visualization.md — previous story learnings]
- [Source: src/ncaa_eval/evaluation/__init__.py — public API surface for dashboard consumption]
- [Source: src/ncaa_eval/model/tracking.py — RunStore, ModelRun for model run listing]
- [Source: src/ncaa_eval/ingest/repository.py — ParquetRepository for season listing]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

### Completion Notes List

- Implemented complete Streamlit multipage app shell using modern `st.navigation()` + `st.Page()` API (Streamlit 1.54.0)
- Dark mode enforced by `.streamlit/config.toml` with project green (`#28a745`) primary color
- Global filters (Tournament Year, Model Run, Scoring Format) persist across page navigation via `st.session_state` with `key=` binding
- All data loading goes through `ncaa_eval` public APIs: `ParquetRepository.get_seasons()`, `RunStore.list_runs()`, `list_scorings()`
- `@st.cache_data` with 5-min TTL on all data-loading functions for sub-500ms interaction response
- Edge cases handled: empty data directory shows "run sync.py" hint, empty runs shows "No model runs available"
- Monospace font CSS applied globally via config.toml `font = "monospace"` plus supplemental CSS for data tables (IBM Plex Mono with fallbacks)
- Added `dashboard/__init__.py` for mypy module resolution (not in original plan, but required for `mypy dashboard/` to work)
- 21 new tests: 9 filter tests (mocked dependencies, return types, cache unwrapping), 12 smoke tests (imports, signatures, CSS, session state)
- All quality gates pass: `mypy --strict` (src+tests), `mypy dashboard/` (non-strict), `ruff check .`, 751 tests passed + 1 skipped

### Change Log

- 2026-02-24: Implemented Story 7.2 — Streamlit app shell with navigation, global filters, dark theme, and monospace fonts
- 2026-02-24: Code review fixes — session state defaults now use first available value (not None), data-loading functions gracefully return [] when data dir missing, run_id cast to str for type safety, 4 new tests added (25 total)

### File List

**New files:**
- `.streamlit/config.toml`
- `dashboard/__init__.py`
- `dashboard/lib/__init__.py`
- `dashboard/lib/filters.py`
- `dashboard/lib/styles.py`
- `dashboard/pages/home.py`
- `dashboard/pages/3_Model_Deep_Dive.py`
- `dashboard/pages/4_Pool_Scorer.py`
- `tests/unit/test_dashboard_filters.py`
- `tests/unit/test_dashboard_app.py`

**Modified files:**
- `.gitignore` (added `!dashboard/lib/` override for `lib/` gitignore rule)
- `dashboard/app.py`
- `dashboard/pages/1_Lab.py`
- `dashboard/pages/2_Presentation.py`
- `_bmad-output/implementation-artifacts/7-2-build-streamlit-app-shell-navigation.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
