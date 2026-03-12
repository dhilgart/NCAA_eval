# Story 9.15: Document Iterrows Convention Exception for Ingest Layer

Status: ready-for-dev

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

- [ ] Task 1: Add ingest-layer exception to Style Guide Section 5 (AC: #1)
  - [ ] 1.1: In `docs/STYLE_GUIDE.md` Section 5 ("Vectorization First"), add a 4th exception to the "Exceptions" list for ingest-layer one-time-per-sync operations
  - [ ] 1.2: Include rationale: one-time sync cost (not in hot paths), Pydantic per-row validation justifies row iteration, and Pandera schema validation guards the DataFrame boundary
  - [ ] 1.3: Reference the PO decision (Audit item 2.4, PO Decision C — 2026-03-11)
- [ ] Task 2: Update forbidden-pattern documentation to acknowledge the exception (AC: #1)
  - [ ] 2.1: In `docs/testing/test-purpose-guide.md` (around line 147), add a comment or note that `src/ncaa_eval/ingest/` is excluded from the forbidden-pattern check
  - [ ] 2.2: In `docs/testing/domain-testing.md` (around line 142), add the same note about the ingest-layer exception
- [ ] Task 3: Run quality gates (AC: #1)
  - [ ] 3.1: `ruff check .` — clean
  - [ ] 3.2: Verify no markdown formatting issues in changed files

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

{{agent_model_name_version}}

### Debug Log References

### Completion Notes List

### File List
