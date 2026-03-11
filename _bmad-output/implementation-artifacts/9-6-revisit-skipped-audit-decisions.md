# Story 9.6: Revisit Skipped Audit Decisions

Status: ready-for-dev

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a **product owner**,
I want to **review and make final decisions on the 19 remaining "S — skip" items from the Epic 8 audit PO decision log**,
so that **no potential improvements are permanently lost, every deferred item has a clear final disposition, and the `po-decision-log-epic8.md` has no remaining "skip" entries**.

## Acceptance Criteria

1. **Given** the `po-decision-log-epic8.md` file currently has 21 items marked `S — skip, come back later`
   **When** the PO reviews each skipped item
   **Then** 2 items are marked as already resolved (2.18 → Story 9.4, 2.20 → Story 9.5)
   **And** each of the remaining 19 items receives a final decision: **Implement** (create new story), **Defer to Post-MVP** (add/confirm in Post-MVP Backlog), or **Accept as-is** (close with rationale)

2. **Given** any item decided as "Implement"
   **When** the decision is recorded
   **Then** a new story is added to `epics.md` (under Epic 9, a new epic, or the next available sprint) with full AC and source references

3. **Given** any item decided as "Defer to Post-MVP"
   **When** the decision is recorded
   **Then** the item is confirmed present in the Post-MVP Backlog section of `epics.md` (most already exist from Story 8.12)

4. **Given** any item decided as "Accept as-is"
   **When** the decision is recorded
   **Then** the item's `PO Decision` field in `po-decision-log-epic8.md` is updated from "S — skip" to the chosen disposition with rationale

5. **Given** all 21 items have been reviewed
   **When** the story completes
   **Then** `po-decision-log-epic8.md` has **zero** remaining items in "S — skip" status
   **And** the Summary section's decision counts are updated to reflect final totals

## The 21 Skipped Items (Grouped by Theme)

### Code Architecture (9 items)

| # | Item | Current State | SM Recommendation |
|---|------|--------------|-------------------|
| 2.2 | `serving.py` imports from ingest layer | `ChronologicalDataServer` imports `Repository`/`Game` from `ncaa_eval.ingest` | Accept as-is — practical data access, not a true violation |
| 2.3 | Repository `get_games` per-row construction | `df.to_dict("records")` → `Game(**row)` per row; wasteful when downstream re-converts to DF | Accept as-is — Pydantic validation on every record is valuable |
| 2.4 | KaggleConnector uses `iterrows()` | 4 `iterrows()` calls despite project convention | Defer — one-time sync cost, not a performance bottleneck |
| 2.5 | Connector ABC optional methods raise NotImplementedError | "Header Interface" anti-pattern; only 2 concrete implementations | Accept as-is — only 2 implementations, not worth refactoring |
| 2.6 | Giant `__init__.py` re-exports (37 symbols) | `transform/__init__.py` loads all submodules on import | Accept as-is — import convenience outweighs startup cost |
| 2.7 | EloModelConfig duplicates EloConfig fields | Same 9 fields in Pydantic model and frozen dataclass | Accept as-is — de-duplication risks breaking the model/config boundary |
| 2.8 | Model Registry is a global mutable singleton | Module-level `_MODEL_REGISTRY` dict | Accept as-is — standard Python registry pattern |
| 2.9 | RunStore deferred import | Circular dependency avoidance via deferred import | Accept as-is — Python-standard circular dependency resolution |
| 2.10 | Deferred sklearn imports in metrics.py | Every metric call does deferred import; cached by Python | Accept as-is — minimal overhead, cached after first call |

### Dashboard/UX (5 items)

| # | Item | Current State | SM Recommendation |
|---|------|--------------|-------------------|
| 2.12 | `get_data_dir()` uses `__file__`-relative path | `Path(__file__).parent.parent.parent / "data"` — fragile | Defer — dashboard dir structure stable since Epic 7 |
| 2.13 | Dashboard pages use module-level `_render_*()` | All page logic runs on import — Streamlit convention | Accept as-is — this IS the Streamlit pattern |
| 2.14 | Leaderboard uses undocumented Streamlit API | `event.selection.rows` with `# type: ignore` | Defer — works currently; address if Streamlit breaks it |
| 2.15 | Plotly adapter API design changed from AC | Epic says methods; impl uses standalone functions | Accept as-is — deliberate design decision documented in Story 7.1 |
| 2.16 | `st.spinner` instead of `st.progress` | AC specifies `st.progress` bar; uses `st.spinner` | Defer — already in Post-MVP Backlog |

### Testing & Quality Gates (3 items)

| # | Item | Current State | SM Recommendation |
|---|------|--------------|-------------------|
| 2.17 | Story 2.3 open AI-review follow-ups | Pandera schema + iterrows replacement not done | Defer — code works correctly; quality improvement |
| 2.21 | `_make_season_df` duplicated in tests | Same helper in 2 test files | Accept as-is — minor duplication; consolidate opportunistically |
| P2-5 | No coverage threshold enforced | CI runs coverage but no `--cov-fail-under` | Defer — need to measure current coverage first |

### Documentation & Quality Gates (2 items)

| # | Item | Current State | SM Recommendation |
|---|------|--------------|-------------------|
| P2-6 | Dashboard excluded from all quality gates | No mypy/ruff for `dashboard/` | Defer — Streamlit has poor type stubs |
| P3-20 | Architecture spec stale | Multiple discrepancies from implementation | Accept as-is — Story 8.12 added historical-document banner |

### Already Resolved (2 items — auto-close)

| # | Item | Resolved By |
|---|------|-------------|
| 2.18 | Top-level `__init__.py` missing re-exports | Story 9.4 (Fix Public API Documentation) |
| 2.20 | No data post-sync validation | Story 9.5 (Post-Sync Data Validation) |

## Tasks / Subtasks

- [ ] Task 1: Auto-close resolved items (AC: #1)
  - [ ] 1.1 Update item 2.18 in `po-decision-log-epic8.md`: change `S — skip` to `Resolved — Story 9.4 fixed public API documentation and import paths`
  - [ ] 1.2 Update item 2.20 in `po-decision-log-epic8.md`: change `S — skip` to `Resolved — Story 9.5 implemented post-sync data validation`

- [ ] Task 2: PO reviews code architecture items 2.2–2.10 (AC: #1, #3, #4)
  - [ ] 2.1 Present items 2.2–2.10 with SM recommendations (see table above)
  - [ ] 2.2 Record PO decision for each item in `po-decision-log-epic8.md`
  - [ ] 2.3 For any "Implement" decisions: create story in `epics.md`
  - [ ] 2.4 For any "Defer" decisions: verify item exists in Post-MVP Backlog

- [ ] Task 3: PO reviews dashboard/UX items 2.12–2.16 (AC: #1, #3, #4)
  - [ ] 3.1 Present items 2.12–2.16 with SM recommendations
  - [ ] 3.2 Record PO decision for each item
  - [ ] 3.3 For any "Implement" decisions: create story in `epics.md`
  - [ ] 3.4 For any "Defer" decisions: verify item exists in Post-MVP Backlog

- [ ] Task 4: PO reviews testing/quality items 2.17, 2.21, P2-5 (AC: #1, #3, #4)
  - [ ] 4.1 Present items with SM recommendations
  - [ ] 4.2 Record PO decision for each item
  - [ ] 4.3 For any "Implement" decisions: create story in `epics.md`
  - [ ] 4.4 For any "Defer" decisions: verify item exists in Post-MVP Backlog

- [ ] Task 5: PO reviews documentation/quality items P2-6, P3-20 (AC: #1, #3, #4)
  - [ ] 5.1 Present items with SM recommendations
  - [ ] 5.2 Record PO decision for each item

- [ ] Task 6: Update decision log summary and verify completeness (AC: #5)
  - [ ] 6.1 Update Summary section decision counts in `po-decision-log-epic8.md`
  - [ ] 6.2 Verify zero "S — skip" entries remain (grep validation)
  - [ ] 6.3 Update `sprint-status.yaml` with any new stories created

- [ ] Task 7: Update `epics.md` Post-MVP Backlog (AC: #2, #3)
  - [ ] 7.1 Add any new "Implement" stories to appropriate epic
  - [ ] 7.2 Confirm "Defer" items exist in Post-MVP Backlog
  - [ ] 7.3 Remove any promoted items from Post-MVP Backlog (if promoted to stories)

## Dev Notes

### Story Nature
This is a **PO decision-gathering story**, not a code implementation story. The "developer" role is the SM/PO facilitating decisions and documenting outcomes. **No production code changes expected** — only planning artifact updates.

### What Changed Since Story Was Originally Imagined

The original story (in `epics.md`) listed items 2.2–2.10, 2.12–2.17, 2.21, P2-5, P2-6, P3-20 as needing decisions. Since then:

1. **Item 2.18 (Top-level `__init__.py` missing re-exports)** — Fully resolved by Story 9.4 (Fix Public API Documentation). The story updated import paths and documentation to match reality.

2. **Item 2.20 (No data post-sync validation)** — Fully resolved by Story 9.5 (Post-Sync Data Validation). Implemented `validate_sync()` with game count, duplicate, and team reference checks.

3. **P3-17 (Metric plugin registry)** — While P3-17 itself was handled in Story 8.13's Cat-1 section (not a skip item), the related tutorial correction item from the Post-MVP Backlog should be verified as addressed by Story 9.10's tutorial update.

4. **All other Epic 9 stories (9.1–9.3, 9.7–9.10) completed** — The Cat-1 "Implement" decisions from Story 8.13 are all done. This story handles only the remaining Cat-2 "skip" items.

### Process Guidance

1. **Present items grouped by theme** (as organized in the tables above) — this helps the PO make coherent batch decisions for related items
2. **SM recommendations are pre-loaded** — each item already has a recommendation from the original decision framework. The PO can accept, override, or modify.
3. **Most items lean toward "Accept as-is"** — these are code architecture decisions where the current state is functional and the fix effort outweighs the benefit. Only a few items warrant "Defer" (items with existing Post-MVP Backlog entries).
4. **Expected outcome:** ~12 Accept-as-is, ~5 Defer, ~0 Implement, 2 Already Resolved

### Artifact Locations
- PO Decision Log: `_bmad-output/planning-artifacts/po-decision-log-epic8.md`
- Epics file: `_bmad-output/planning-artifacts/epics.md`
- Sprint status: `_bmad-output/implementation-artifacts/sprint-status.yaml`
- Audit reports (reference only):
  - `_bmad-output/planning-artifacts/codebase-audit-report.md`
  - `_bmad-output/planning-artifacts/codebase-audit-pass2-addendum.md`
  - `_bmad-output/planning-artifacts/codebase-audit-pass3-addendum.md`

### Previous Story Intelligence (Story 8.13)
- Story 8.13 established the PO Decision Log format with structured decision frameworks per item
- Each item has: Decision needed, Context, Options table, Recommendation, Rationale, Follow-up, PO Decision
- The 21 "S — skip" items were deferred in batch during the 2026-03-09 PO session — they received the decision framework text but no final PO decision
- Story 8.12 established the Post-MVP Backlog entry format (title, description, effort, distinctness, source, deferral reason)

### Project Structure Notes
- No source tree changes expected — this is a planning/documentation story
- Artifacts modified: `po-decision-log-epic8.md`, `epics.md`, `sprint-status.yaml`

### References
- [Source: _bmad-output/planning-artifacts/po-decision-log-epic8.md] — 21 items with "S — skip" status
- [Source: _bmad-output/planning-artifacts/epics.md#post-mvp-backlog] — existing deferred items
- [Source: _bmad-output/implementation-artifacts/8-13-gather-po-direction-category-1-2-items.md] — parent story context
- [Source: _bmad-output/implementation-artifacts/9-4-fix-public-api-documentation.md] — resolved item 2.18
- [Source: _bmad-output/implementation-artifacts/9-5-post-sync-data-validation.md] — resolved item 2.20
- [Source: _bmad-output/implementation-artifacts/9-10-custom-metric-plugin-registry.md] — related P3-17 tutorial update

## Dev Agent Record

### Agent Model Used

claude-opus-4-6

### Debug Log References

### Completion Notes List

### File List
