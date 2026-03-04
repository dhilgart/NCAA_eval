# Story 8.8: Dashboard UX Quick Fixes

Status: ready-for-dev

## Story

As a **data scientist using the Streamlit dashboard**,
I want bracket text readable at a glance, clear first-run guidance, a cache refresh button, consistent navigation, and data freshness indicators,
so that the dashboard is immediately usable without squinting, confusion, or stale-data surprises.

## Acceptance Criteria

1. **Bracket renderer font sizes increased** — Team names minimum 12px, probability labels minimum 10px (currently 10px / 9px in `dashboard/lib/bracket_renderer.py:79-83`)
2. **Dashboard home page shows prominent "Setup needed" message** when no data exists — not just the small sidebar `st.info()` message; must be a prominent banner or callout on the home page body itself
3. **"Refresh Data" button added to sidebar** to manually clear `st.cache_data` — users who train a model or run sync see stale data until the 5-minute TTL expires
4. **Breadcrumb navigation consistent across all pages** — currently missing from `dashboard/pages/1_Lab.py`; add breadcrumbs matching the pattern used in `2_Presentation.py`, `3_Model_Deep_Dive.py`, and `4_Pool_Scorer.py`
5. **Data freshness indicator in sidebar** — show last sync date and latest game date so users know how current their data is

## Tasks / Subtasks

- [x] Task 1: Increase bracket renderer font sizes (AC: #1)
  - [x] 1.1 In `dashboard/lib/bracket_renderer.py`, change `.name` font-size from `10px` to `12px` (line 79)
  - [x] 1.2 Change `.prob` font-size from `9px` to `10px` (line 83)
  - [x] 1.3 Consider increasing `.seed` font-size from `10px` to `11px` for proportionality
  - [x] 1.4 Adjust `.team` min-height from `18px` to `20px` if needed to accommodate larger text
  - [x] 1.5 Verify bracket still fits within the 700px-height iframe (`components.html(bracket_html, height=700)` in `2_Presentation.py:46`) — increase height if needed

- [ ] Task 2: Add prominent "Setup needed" message on home page (AC: #2)
  - [ ] 2.1 In `dashboard/pages/home.py`, check if `years` is empty (already computed on line 13)
  - [ ] 2.2 If empty, display `st.warning()` or `st.error()` with large banner explaining: data directory not found, run `ncaa-eval sync --source all --dest data/` to download data
  - [ ] 2.3 If years exist but runs is empty, display `st.info()` explaining: data exists but no models trained yet, run `ncaa-eval train --model elo --start-year 2015 --end-year 2025`
  - [ ] 2.4 Preserve the existing summary metrics for the happy path (data + runs both present)

- [ ] Task 3: Add "Refresh Data" button to sidebar (AC: #3)
  - [ ] 3.1 In `dashboard/app.py`, add a button at the bottom of the sidebar section (after the Scoring Format filter)
  - [ ] 3.2 On click, call `st.cache_data.clear()` followed by `st.rerun()`
  - [ ] 3.3 Use a subtle button style — this is a utility action, not a primary CTA (consider `st.button("🔄 Refresh Data", use_container_width=True)`)

- [ ] Task 4: Add breadcrumbs to Leaderboard page (AC: #4)
  - [ ] 4.1 In `dashboard/pages/1_Lab.py`, add breadcrumb block before the `st.header("Backtest Leaderboard")` call (around line 20)
  - [ ] 4.2 Use the same pattern as other pages: `col_nav, col_bc = st.columns([1, 3])` with `st.page_link("pages/home.py", label="← Home")` and `st.caption("Home > Lab > Backtest Leaderboard")`

- [ ] Task 5: Add data freshness indicator to sidebar (AC: #5)
  - [ ] 5.1 In `dashboard/lib/data_loaders.py`, add a cached function `load_data_freshness(data_dir: str) -> dict[str, str | None]` that returns `{"last_sync_date": ..., "latest_game_date": ...}`
  - [ ] 5.2 `last_sync_date`: Check modification time of the Parquet files in `data_dir` (e.g., `max(p.stat().st_mtime for p in Path(data_dir).rglob("*.parquet"))`) and format as date string
  - [ ] 5.3 `latest_game_date`: Load the most recent season and find the max game date via `ParquetRepository`
  - [ ] 5.4 In `dashboard/app.py`, display freshness info at the bottom of the sidebar: small caption showing "Data synced: {date}" and "Latest game: {date}"
  - [ ] 5.5 If data dir doesn't exist, skip freshness display (the "no data" info message already covers this)

- [ ] Task 6: Run quality gates
  - [ ] 6.1 `ruff check .` passes
  - [ ] 6.2 `ruff format --check .` passes
  - [ ] 6.3 `mypy --strict src/ncaa_eval tests` passes (note: `dashboard/` is excluded from mypy — this is intentional per P2-6)
  - [ ] 6.4 `pytest` passes (full suite)

## Dev Notes

### Current State Analysis

**Bracket Font Sizes** (`dashboard/lib/bracket_renderer.py`):
- `.name` (team names): `font-size: 10px` → needs `12px`
- `.prob` (win probability): `font-size: 9px` → needs `10px`
- `.seed` (seed number): `font-size: 10px` → consider `11px`
- `.team` container: `min-height: 18px` — may need bump to accommodate larger text
- Body font: `font-size: 11px` (used for labels like "Final Four") — leave as-is
- The bracket renders inside a `components.html(bracket_html, height=700, scrolling=True)` iframe in `2_Presentation.py:46`

**Home Page Empty State** (`dashboard/pages/home.py`):
- Currently shows `st.metric("Available Seasons", 0)` and `st.metric("Model Runs", 0)` when no data — not helpful
- Sidebar already has `st.info("No data available — run ...")` but it's easy to miss
- The home page body needs a prominent banner for first-time users

**Cache Refresh** (`dashboard/app.py`):
- All data loaders use `@st.cache_data(ttl=300)` (5 minutes)
- `load_available_scorings` and `load_scoring_display_names` use `ttl=None` (permanent)
- `st.cache_data.clear()` clears ALL cached data — this is the correct approach since partial invalidation is not supported by Streamlit's API
- After clearing, `st.rerun()` forces all data to be re-fetched from disk

**Breadcrumbs** — Current state by page:
| Page | Has Breadcrumbs | Pattern |
|------|----------------|---------|
| `home.py` | N/A (root) | — |
| `1_Lab.py` | ❌ Missing | — |
| `2_Presentation.py` | ✅ | `st.columns([1, 3])` + `st.page_link` + `st.caption` |
| `3_Model_Deep_Dive.py` | ✅ | `st.columns([1, 3])` + `st.page_link` + `st.caption` |
| `4_Pool_Scorer.py` | ✅ | `st.columns([1, 3])` + `st.page_link` + `st.caption` |

**Data Freshness**:
- `ParquetRepository` stores game data in `data/seasons/` as Parquet files
- `RunStore` stores model runs in `data/runs/`
- Modification time of Parquet files is a proxy for "last sync date"
- Latest game date requires loading season data and finding max date
- Keep it lightweight — cache with same TTL as other data loaders (300s)

### Key File Locations

| Purpose | File Path | Lines |
|---|---|---|
| Bracket CSS/HTML renderer | `dashboard/lib/bracket_renderer.py` | 24-105 (CSS), 192-237 (cells), 240-313 (main) |
| Home page | `dashboard/pages/home.py` | 1-25 |
| Main app + sidebar | `dashboard/app.py` | 1-102 |
| Leaderboard (missing breadcrumbs) | `dashboard/pages/1_Lab.py` | 1-132 |
| Presentation (breadcrumb reference) | `dashboard/pages/2_Presentation.py` | 107-112 |
| Data loaders (cache layer) | `dashboard/lib/data_loaders.py` | 1-281 |
| CSS styles | `dashboard/lib/styles.py` | 1-12 |
| Parquet repository | `src/ncaa_eval/ingest/repository.py` | — |

### Architecture & Convention Compliance

- **Dashboard is excluded from `mypy --strict`** (intentional per P2-6 — Streamlit has poor type stubs)
- **Dashboard uses `ncaa_eval` public APIs only** — no direct file IO (enforced pattern)
- **`@st.cache_data`** is the caching mechanism — all data access goes through cached loaders
- **No `from __future__ import annotations`** requirement for dashboard (EDA notebook rule), but dashboard files already use it — maintain consistency
- **Conventional commits**: `fix(dashboard): ...` scope for this story
- **No dashboard tests currently exist** that would need updating (dashboard test coverage is minimal — limited to `tests/unit/test_dashboard_filters.py` which tests scoring orchestration, not page rendering)

### Previous Story Intelligence (8.7)

Story 8.7 was a housekeeping/CI story — no dashboard changes. Key learnings:
- 922 tests passing as of story 8.7
- All quality gates green
- Pre-commit is canonical quality gate (Section 11 of STYLE_GUIDE.md)
- CI divergence from nox is documented and accepted

### Git Intelligence

Recent commits are all squash-merge PRs for Epic 8 stories. Pattern: `type(scope): Story X.Y — Title (#PR)`. This story should follow the same pattern.

### Risks & Gotchas

1. **`st.cache_data.clear()` is global** — it clears ALL cached data, not just stale entries. This is acceptable because re-fetching from local Parquet files is fast (<100ms). No partial invalidation alternative exists in Streamlit's API.
2. **Data freshness from file mtime** — Parquet file modification time is a proxy for sync date, not an authoritative timestamp. This is acceptable for a UX indicator.
3. **Bracket height** — Increasing font sizes may cause the bracket to overflow its 700px iframe. Test with a full 64-team bracket and increase `height` in `components.html()` if needed (e.g., to 800px).
4. **Scope discipline** — This story covers 5 specific UX fixes. Do NOT expand to add game theory sliders, user-editable brackets, theme toggles, or other deferred items (those are Category 1/2 items for Story 8.13).
5. **`PLR0913` noqa comments** — `_team_cell` and `_render_region_html` already have `# noqa: PLR0913 — REFACTOR Story 8.1` comments. These are acknowledged tech debt, not part of this story's scope.

### References

- [Source: `_bmad-output/planning-artifacts/codebase-audit-report.md` — Findings 3.30 (freshness), 3.31 (bracket font), 3.32 (first-run), 3.33 (cache refresh), 3.34 (breadcrumbs)]
- [Source: `_bmad-output/planning-artifacts/epic-8-codebase-improvements.md` — Story 8.8 definition]
- [Source: `dashboard/lib/bracket_renderer.py` — Current bracket CSS]
- [Source: `dashboard/app.py` — Current sidebar implementation]
- [Source: `dashboard/pages/home.py` — Current home page]
- [Source: `dashboard/pages/1_Lab.py` — Missing breadcrumbs]
- [Source: `dashboard/pages/2_Presentation.py:107-112` — Breadcrumb reference pattern]
- [Source: `dashboard/lib/data_loaders.py` — Cache strategy and data access patterns]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

### Completion Notes List

### Change Log

### File List
