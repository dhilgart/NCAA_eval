# Story 9.6: Revisit Skipped Audit Decisions

Status: review

## Story

As a **product owner**,
I want to **review and make final decisions on all Epic 8 audit items deferred with "S — skip, come back later"**,
so that **no potential improvements are permanently lost and each item receives a clear disposition (Implement, Defer to Post-MVP, or Accept as-is)**.

## Acceptance Criteria

1. **Given** the `po-decision-log-epic8.md` file contains items marked `S — skip, come back later`
   **When** the PO reviews each skipped item
   **Then** every item listed below receives a final decision (Implement, Defer to Post-MVP, or Accept as-is):
   - 2.2 (`serving.py` imports from ingest)
   - 2.3 (Repository `get_games` per-row construction)
   - 2.4 (KaggleConnector uses `iterrows()`)
   - 2.5 (Connector ABC optional methods)
   - 2.6 (Giant `__init__.py` re-exports)
   - 2.7 (EloModelConfig duplicates EloConfig)
   - 2.8 (Model Registry global singleton)
   - 2.9 (RunStore deferred import)
   - 2.10 (Deferred sklearn imports)
   - 2.12 (`get_data_dir()` `__file__`-relative path)
   - 2.13 (Dashboard module-level `_render_*()` pattern)
   - 2.14 (Undocumented Streamlit API)
   - 2.15 (Plotly adapter API changed from AC)
   - 2.16 (`st.spinner` instead of `st.progress`)
   - 2.17 (Story 2.3 open AI-review follow-ups)
   - 2.18 (Top-level `__init__.py` missing re-exports)
   - 2.21 (`_make_season_df` duplicated in tests)
   - P2-5 (No coverage threshold)
   - P2-6 (Dashboard excluded from quality gates)
   - P3-20 (Architecture spec stale)

2. **Given** a decision of "Implement" for any item
   **When** the decision is recorded
   **Then** a new Epic 9 story is created in `epics.md` (or the item is added to an existing story) and `sprint-status.yaml` is updated with the new story key in backlog status

3. **Given** a decision of "Defer to Post-MVP" for any item
   **When** the decision is recorded
   **Then** the item is added to (or confirmed already in) the Post-MVP Backlog section of `epics.md`

4. **Given** a decision of "Accept as-is" for any item
   **When** the decision is recorded
   **Then** the item is marked as resolved in `po-decision-log-epic8.md` with rationale

5. **Given** all items have been reviewed
   **When** the story is complete
   **Then** `po-decision-log-epic8.md` has NO remaining items in "S — skip" status
   **And** the Decision Counts summary at the top of `po-decision-log-epic8.md` is updated to reflect new totals

## Tasks / Subtasks

- [x] Task 1: Present skipped items to PO for batch review (AC: #1)
  - [x] 1.1: Load `po-decision-log-epic8.md` and identify all 20 items with "S — skip" status
  - [x] 1.2: For each item, present the original context, options, and recommendation to the PO
  - [x] 1.3: Record PO decision for each item (Implement / Defer to Post-MVP / Accept as-is)

- [x] Task 2: Process "Implement" decisions (AC: #2)
  - [x] 2.1: For each "Implement" decision, create a new story entry in `epics.md` under Epic 9 (or confirm existing coverage)
  - [x] 2.2: Add corresponding story key to `sprint-status.yaml` with status `backlog`
  - [x] 2.3: Update the item's entry in `po-decision-log-epic8.md` with the PO decision and follow-up story reference

- [x] Task 3: Process "Defer to Post-MVP" decisions (AC: #3)
  - [x] 3.1: For each "Defer" decision, verify the item exists in the Post-MVP Backlog of `epics.md` (add if missing)
  - [x] 3.2: Update the item's entry in `po-decision-log-epic8.md` replacing "S — skip" with the defer decision and rationale

- [x] Task 4: Process "Accept as-is" decisions (AC: #4)
  - [x] 4.1: For each "Accept" decision, update the item's entry in `po-decision-log-epic8.md` replacing "S — skip" with accept-as-is and rationale

- [x] Task 5: Finalize decision log and verify completeness (AC: #5)
  - [x] 5.1: Verify ALL 20 skipped items now have a final decision (no "S — skip" remaining)
  - [x] 5.2: Update the Decision Counts summary at the top of `po-decision-log-epic8.md`
  - [x] 5.3: Run a grep/search to confirm zero "skip, come back later" entries remain

## Dev Notes

### Nature of This Story

This is a **PO decision-making facilitation story**, not a code implementation story. The dev agent's role is to:
1. Present each skipped item to the PO with full context
2. Record decisions
3. Update project tracking artifacts (`po-decision-log-epic8.md`, `epics.md`, `sprint-status.yaml`)

No source code changes are expected. All changes are to planning/tracking documents.

### Items Under Review — Quick Reference

The 20 skipped items fall into three categories:

**Architecture/Design Decisions (likely Accept as-is):**
- 2.2: `serving.py` imports from ingest — practical data access, not a real layer violation
- 2.3: Repository per-row Game construction — Pydantic validation worth the cost
- 2.5: Connector ABC optional methods — only 2 implementations, protocols overkill
- 2.6: Giant `__init__.py` re-exports — import convenience valued for personal project
- 2.7: EloModelConfig duplicates EloConfig — Pydantic vs dataclass serve different purposes
- 2.8: Model Registry global singleton — standard plugin pattern
- 2.9: RunStore deferred import — well-established circular dependency pattern
- 2.10: Deferred sklearn imports — cached after first call, negligible overhead
- 2.13: Dashboard module-level `_render_*()` — standard Streamlit convention
- 2.15: Plotly adapter API changed from AC — deliberate documented design decision

**Code Quality Improvements (likely Defer to Post-MVP):**
- 2.4: KaggleConnector `iterrows()` — not a perf bottleneck, runs once per sync
- 2.12: `get_data_dir()` `__file__`-relative path — fragile but stable since Epic 7
- 2.14: Undocumented Streamlit API — works, fix if/when Streamlit breaks it
- 2.16: `st.spinner` vs `st.progress` — UX polish, already in Post-MVP Backlog #19
- 2.17: Story 2.3 AI-review follow-ups (Pandera + iterrows) — quality, not functional
- 2.21: `_make_season_df` duplicated in tests — consolidate opportunistically
- P2-5: No coverage threshold — needs baseline measurement first
- P2-6: Dashboard excluded from quality gates — Streamlit type stubs are poor

**Already Partially Resolved:**
- 2.18: Top-level `__init__.py` re-exports — Story 9.4 ALREADY fixed this (updated Style Guide). Decision log still says "S — skip" but the work is done. Mark as resolved.
- P3-20: Architecture spec stale — Story 8.12 already added historical-document banner. Mark as resolved.

### Previous Story Intelligence

**Story 8.13** (Gather PO Direction) established the decision framework:
- Presented items with context, options, recommendations
- PO made decisions in batch (some overriding SM recommendations)
- Items marked "S — skip" were explicitly deferred to this follow-up story
- The PO created 5 new stories (9.1–9.5) and made custom decisions (e.g., 1.6 became model-level feature config instead of CLI flag)

**Stories 9.1–9.5** demonstrate the PO's decision pattern:
- Tends to implement features that complete the core user journey (1.2 user-editable bracket, 1.11 CLI predict)
- Respects architecture decisions documented in stories (1.7 label bias, 1.13 StatefulModel.fit())
- Values low-effort/high-value items (1.3 Kaggle export, 1.15 feature importance)

### Files to Modify

- `_bmad-output/planning-artifacts/po-decision-log-epic8.md` — update each "S — skip" to final decision
- `_bmad-output/planning-artifacts/epics.md` — add new stories (if any "Implement") or Post-MVP items (if any "Defer")
- `_bmad-output/implementation-artifacts/sprint-status.yaml` — add new story keys (if any "Implement")

### Files NOT to Modify

- No source code files — this is a planning/decision story
- No test files
- No configuration files

### Testing Standards

No code changes = no tests. Verification is:
- Zero "S — skip" entries remaining in `po-decision-log-epic8.md`
- Decision Counts summary is accurate
- Any new stories appear in both `epics.md` and `sprint-status.yaml`
- Any new Post-MVP items appear in the Post-MVP Backlog section of `epics.md`

### Project Structure Notes

- All changes are within `_bmad-output/` planning and implementation artifact directories
- No structural changes to the source tree

### References

- [Source: _bmad-output/planning-artifacts/po-decision-log-epic8.md] — Primary artifact with all 20 skipped items
- [Source: _bmad-output/planning-artifacts/epics.md#Epic-9] — Current Epic 9 story list
- [Source: _bmad-output/planning-artifacts/epics.md#Post-MVP-Backlog] — Current Post-MVP Backlog
- [Source: _bmad-output/implementation-artifacts/sprint-status.yaml] — Sprint tracking
- [Source: _bmad-output/implementation-artifacts/9-5-post-sync-data-validation.md] — Previous story (patterns and PO decision style)
- [Source: _bmad-output/planning-artifacts/codebase-audit-report.md] — Original audit findings
- [Source: _bmad-output/planning-artifacts/codebase-audit-pass2-addendum.md] — Pass 2 findings
- [Source: _bmad-output/planning-artifacts/codebase-audit-pass3-addendum.md] — Pass 3 findings

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

N/A — no code implementation, no debugging needed.

### Completion Notes List

- Reviewed all 20 "S — skip" items from `po-decision-log-epic8.md`
- **10 items → Accept as-is**: 2.2, 2.3, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.13, 2.15 (architecture/design decisions that are pragmatic and well-justified)
- **8 items → Defer to Post-MVP**: 2.4, 2.12, 2.14, 2.16, 2.17, 2.21, P2-5, P2-6 (code quality improvements, not functional bugs; all confirmed present in Post-MVP Backlog)
- **2 items → Already Resolved**: 2.18 (fixed by Story 9.4), P3-20 (fixed by Story 8.12)
- **Bonus**: Also resolved 2.20 (was "S — skip" but Story 9.5 implemented the work)
- **No "Implement" decisions** — all 20 skipped items were architecture decisions, code quality improvements, or already resolved
- Updated Decision Counts summary to reflect final totals: Cat 1 (8 Implement, 6 Defer, 2 Accept-as-is), Cat 2 (11 Accept-as-is, 8 Defer, 5 Already Resolved)
- Updated Follow-up Actions Summary with comprehensive disposition breakdown
- Verified zero "S — skip" entries remaining via grep
- No changes to `epics.md` needed — all Defer items already present in Post-MVP Backlog
- No changes to `sprint-status.yaml` needed (beyond marking this story in-progress → review) — no new stories created

### Change Log

- 2026-03-09: Resolved all 20 "S — skip" items in `po-decision-log-epic8.md` (10 Accept-as-is, 8 Defer, 2 Already Resolved + 1 bonus). Updated Decision Counts and Follow-up Actions Summary. Story status → review.

### File List

- `_bmad-output/planning-artifacts/po-decision-log-epic8.md` (modified — 20 skip decisions resolved, counts updated, summary rewritten)
- `_bmad-output/implementation-artifacts/sprint-status.yaml` (modified — story status ready-for-dev → in-progress → review)
- `_bmad-output/implementation-artifacts/9-6-revisit-skipped-audit-decisions.md` (modified — tasks marked complete, Dev Agent Record filled)
