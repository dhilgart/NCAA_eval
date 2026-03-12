# Story 9.15: Document Iterrows Convention Exception for Ingest Layer

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a **developer**,
I want to **document the explicit exception allowing `iterrows()` in the ingest layer**,
so that **future audits do not re-flag this as a convention violation and the rationale is clear**.

## Acceptance Criteria

1. **Given** the project convention prohibits `iterrows()` usage (Style Guide Section 5)
   **When** a developer reads the Style Guide / conventions documentation
   **Then** they find a documented exception for the ingest layer's one-time-per-sync operations
   **And** the rationale (one-time sync cost, Pydantic validation per row) is explained

## Tasks / Subtasks

- [x] Task 1: Add ingest-layer exception to Style Guide Section 5 (AC: #1)
  - [x] 1.1: In `docs/STYLE_GUIDE.md` Section 5 ("Vectorization First"), add a 4th exception to the "Exceptions" list for ingest-layer one-time-per-sync operations
  - [x] 1.2: Include rationale: one-time sync cost (not in hot paths), Pydantic per-row validation justifies row iteration, and Pandera schema validation guards the DataFrame boundary
  - [x] 1.3: Reference the PO decision (Audit item 2.4, PO Decision C — 2026-03-11)
- [x] Task 2: Update forbidden-pattern documentation to acknowledge the exception (AC: #1)
  - [x] 2.1: In `docs/testing/test-purpose-guide.md` (around line 147), add a comment or note that `src/ncaa_eval/ingest/` is excluded from the forbidden-pattern check
  - [x] 2.2: In `docs/testing/domain-testing.md` (around line 142), add the same note about the ingest-layer exception
- [x] Task 3: Run quality gates (AC: #1)
  - [x] 3.1: `ruff check .` — clean
  - [x] 3.2: Verify no markdown formatting issues in changed files

## Dev Notes

### Key Implementation Details

**This is a documentation-only story.** No source code changes. No test changes. Only markdown documentation files are modified.

**Target files (all in `docs/`):**
1. `docs/STYLE_GUIDE.md` — Primary change: add exception #4 to Section 5 "Exceptions" list (currently at lines 302-313)
2. `docs/testing/test-purpose-guide.md` — Minor addition: note that ingest layer is excluded from forbidden-pattern detection (around lines 146-151)
3. `docs/testing/domain-testing.md` — Minor addition: same note about ingest-layer exclusion (around lines 141-146)

**Current Style Guide Section 5 "Exceptions" (lines 302-313):**
Three exceptions are currently documented:
1. Small, fixed collection (e.g., 5 model configs)
2. Graph traversal (NetworkX operations)
3. Side effects (file writing per team)

**New exception #4 to add:**
4. **Ingest-layer one-time-per-sync operations** — `src/ncaa_eval/ingest/connectors/kaggle.py` uses `iterrows()` in three methods (`load_day_zeros`, `fetch_teams`, `_parse_games_csv`) to construct Pydantic model instances (Game, Team) from CSV rows. These run exactly once per `sync` operation and are not in any hot path. The per-row Pydantic validation catches data integrity issues at the boundary, which justifies row-by-row iteration. Pandera schema validation (Story 9.14) guards the DataFrame boundary before iteration begins.

**PO Decision reference:**
- Audit item 2.4 in `_bmad-output/planning-artifacts/codebase-audit-report.md`
- PO Decision: C — Add explicit exception to convention (2026-03-11)
- Decision log: `_bmad-output/planning-artifacts/po-decision-log-epic8.md` Section 2.4

**Existing `test_no_iterrows` smoke tests:**
- `tests/unit/test_graph.py:509` — asserts `graph.py` has no iterrows
- `tests/unit/test_opponent.py:294` — asserts `opponent.py` has no iterrows
- These tests are scoped to specific transform-layer modules, NOT to the ingest layer, so they already exclude `kaggle.py`. No test changes needed.

**What NOT to do:**
- Do NOT change any Python source code — this is documentation only
- Do NOT remove or modify existing iterrows usages in `kaggle.py`
- Do NOT add new `test_no_iterrows` tests for the ingest layer (the exception explicitly allows iterrows there)
- Do NOT weaken the existing vectorization convention — the exception is narrowly scoped to ingest-layer sync operations only
- Do NOT add `itertuples` to the exception — only `iterrows` is covered (and only in the ingest layer)
- Do NOT touch `docs/TESTING_STRATEGY.md` — the reference there (line 315) is an example snippet, not a rule definition

### Previous Story Intelligence (Story 9.14)

- Pandera schema validation was added to `kaggle.py` — validates DataFrame structure before iterrows iteration
- The iterrows usage was explicitly preserved per story AC ("iterrows usage is NOT changed — accepted per item 2.4 carve-out for ingest layer")
- Story 9.14 code review confirmed 1132 tests passing
- Import pattern for Pandera: `import pandera.pandas as pa` (v0.29+ deprecation of `import pandera as pa`)

### Git Intelligence

Recent commit pattern: `docs(convention): ...` or `docs(style): ...` would be appropriate for this documentation change. Last 10 commits are all Epic 9 stories (9.5 through 9.14), following conventional commit format with scope.

### Project Structure Notes

- All changes confined to `docs/` directory — no `src/` or `tests/` changes
- `from __future__ import annotations` not applicable (markdown files only)
- `mypy --strict` not applicable (no Python changes)
- `ruff check` still applies to verify no issues introduced

### References

- [Source: docs/STYLE_GUIDE.md — Section 5, lines 264-314]
- [Source: docs/testing/test-purpose-guide.md — lines 144-151, forbidden pattern list]
- [Source: docs/testing/domain-testing.md — lines 139-146, forbidden pattern list]
- [Source: _bmad-output/planning-artifacts/codebase-audit-report.md — Section 2.4]
- [Source: _bmad-output/planning-artifacts/po-decision-log-epic8.md — Section 2.4, PO Decision C]
- [Source: _bmad-output/planning-artifacts/epics.md — Epic 9, Story 9.15]
- [Source: src/ncaa_eval/ingest/connectors/kaggle.py — iterrows at lines 187, 208, 269]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

(none — documentation-only story, no debug issues encountered)

### Completion Notes List

- Added Exception #4 to Style Guide Section 5 "Exceptions" list covering ingest-layer one-time-per-sync `iterrows()` usage with full rationale (Pydantic per-row validation, Pandera boundary guard, PO Decision C reference)
- Added blockquote notes to both `test-purpose-guide.md` and `domain-testing.md` clarifying the ingest-layer exclusion from forbidden-pattern checks, with cross-reference to Style Guide Section 5 Exception #4
- All quality gates pass: `ruff check .` clean, markdown formatting verified
- No Python source code or test changes — documentation only as specified

### Change Log

- 2026-03-12: Documented ingest-layer iterrows exception in Style Guide and testing docs (Story 9.15)
- 2026-03-12: Code review fixes — corrected Style Guide Exception #4 rationale for `load_day_zeros` (MEDIUM-1); clarified itertuples scope exclusion in both testing doc notes (MEDIUM-2)

## Senior Developer Review (AI)

**Reviewer:** Claude Sonnet 4.6 | **Date:** 2026-03-12

**Verdict:** ✅ APPROVED (with fixes applied)

**Git vs Story Discrepancies:** 0
**Issues Found:** 0 High, 2 Medium, 3 Low
**Issues Fixed:** 2 (MEDIUM-1, MEDIUM-2)
**Action Items Created:** 0

### Findings

**[FIXED] MEDIUM-1 — STYLE_GUIDE.md:312-318: Overstated Pydantic rationale for `load_day_zeros`**
- Original text claimed all three methods use iterrows "to construct Pydantic model instances (`Game`, `Team`) from CSV rows"
- `load_day_zeros` builds `dict[int, datetime.date]` — no Pydantic models; it uses per-row date parsing with custom `DataFormatError`
- Fixed: Reworded to accurately describe per-row transformation for each method

**[FIXED] MEDIUM-2 — test-purpose-guide.md:158 / domain-testing.md:153: Ambiguous scope of forbidden-pattern exclusion**
- Notes appeared adjacent to a forbidden-patterns list containing `.iterrows()`, `.itertuples()`, and `for row in df`
- Story explicitly prohibits adding `.itertuples()` to the exception; notes could mislead inattentive readers
- Fixed: Reworded to explicitly state "the `iterrows()` forbidden-pattern check only" and that all other patterns remain prohibited

**LOW-1** — PO decision log has stale kaggle.py line numbers (157, 168 vs actual 187, 208) — planning artifact, no fix needed in this story.
**LOW-2** — Exception #4 title/body scope mismatch (broad title vs narrow named file) — acceptable as written; ingest-layer scope is clear from context.
**LOW-3** — Testing doc notes don't cross-reference PO Decision C directly — low traceability value; Style Guide cross-reference is sufficient.

### AC Coverage
- AC #1: IMPLEMENTED ✅ — Exception documented in Style Guide, with rationale and testing doc cross-references.

### File List

- `docs/STYLE_GUIDE.md` (modified — added Exception #4 to Section 5)
- `docs/testing/test-purpose-guide.md` (modified — added ingest-layer exclusion note)
- `docs/testing/domain-testing.md` (modified — added ingest-layer exclusion note)
- `_bmad-output/implementation-artifacts/9-15-document-iterrows-convention-exception.md` (modified — task completion, status)
- `_bmad-output/implementation-artifacts/sprint-status.yaml` (modified — status update)
- `_bmad-output/planning-artifacts/template-requirements.md` (modified — template learnings from code review, Story 9.15)
