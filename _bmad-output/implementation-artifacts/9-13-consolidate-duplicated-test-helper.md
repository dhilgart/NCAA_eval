# Story 9.13: Consolidate Duplicated Test Helper

Status: ready-for-dev

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want to consolidate the duplicated `_make_season_df` helper from `test_evaluation_splitter.py` and `test_evaluation_backtest.py` into a shared conftest fixture,
so that test helpers follow DRY principles and future test files can reuse the fixture.

## Acceptance Criteria

1. **Given** `_make_season_df` is duplicated in `tests/unit/test_evaluation_splitter.py` (lines 18-47) and `tests/unit/test_evaluation_backtest.py` (lines 29-60)
   **When** the developer moves it to a shared conftest
   **Then** both test files import from the shared fixture
   **And** all existing tests pass without modification

## Tasks / Subtasks

- [ ] Task 1: Consolidate `_make_season_df` into shared location (AC: #1)
  - [ ] 1.1 Create the unified `_make_season_df` in `tests/unit/conftest.py` — use the **backtest version** (it's a superset with `elo_diff` and `win_pct_diff` columns the splitter version lacks)
  - [ ] 1.2 Remove `_make_season_df` from `tests/unit/test_evaluation_splitter.py`
  - [ ] 1.3 Remove `_make_season_df` from `tests/unit/test_evaluation_backtest.py`
  - [ ] 1.4 Verify both test files can access the helper (pytest auto-discovers conftest.py fixtures/helpers)
- [ ] Task 2: Consolidate `_make_feature_server` into shared location (AC: #1)
  - [ ] 2.1 Move the identical `_make_feature_server` helper to `tests/unit/conftest.py`
  - [ ] 2.2 Remove from both `test_evaluation_splitter.py` and `test_evaluation_backtest.py`
- [ ] Task 3: Run full test suite and verify no regressions (AC: #1)
  - [ ] 3.1 `pytest` — all tests pass
  - [ ] 3.2 `mypy --strict src/ncaa_eval tests` — clean
  - [ ] 3.3 `ruff check .` — clean

## Dev Notes

### Key Implementation Details

**The two `_make_season_df` implementations differ slightly:**
- **Splitter version** (`test_evaluation_splitter.py:18-47`): 9 columns — `game_id`, `season`, `day_num`, `date`, `team_a_id`, `team_b_id`, `is_tournament`, `loc_encoding`, `team_a_won`
- **Backtest version** (`test_evaluation_backtest.py:29-60`): 11 columns — same 9 plus `elo_diff` and `win_pct_diff` (synthetic feature columns needed by stateless model code path in `_feature_cols()`)

**Use the backtest version** as the canonical implementation. The extra columns are harmless to splitter tests (they don't inspect column names) but required by backtest tests that exercise the stateless model column-filtering logic.

**`_make_feature_server` is identical** in both files (lines 50-56 splitter, lines 63-69 backtest). Move verbatim.

### Placement Decision: `tests/unit/conftest.py`

Place shared helpers in `tests/unit/conftest.py` (NOT root `tests/conftest.py`) because:
- Both consuming files are in `tests/unit/`
- Keeps scope minimal — integration tests don't need these helpers
- Root `tests/conftest.py` currently has only `temp_data_dir` and should stay lean
- pytest auto-discovers `conftest.py` — no explicit imports needed

**Important:** These are plain helper functions, NOT pytest fixtures (no `@pytest.fixture` decorator). Place them as module-level functions in conftest.py — pytest makes conftest contents importable within the directory scope. Test files will use `from conftest import _make_season_df, _make_feature_server` or simply call them directly since conftest is auto-loaded.

**Correction:** Actually, since these are plain functions (not fixtures), test files in `tests/unit/` cannot auto-import them from conftest without an explicit import. Two options:
1. Keep as plain functions in conftest and add `from conftest import _make_season_df` in each test file
2. Convert to `@pytest.fixture` returning a factory callable

Option 1 is simpler and preserves current call patterns. Use explicit imports.

### Existing `tests/unit/conftest.py`

Check if `tests/unit/conftest.py` exists. If not, create it. If it exists, append to it.

### What NOT to Do

- Do NOT consolidate other duplicated helpers in this story (e.g., `_make_game`, `repo` fixture, `_make_mock_stateful_model`). Scope is strictly `_make_season_df` and `_make_feature_server` per the AC.
- Do NOT change any function signatures or behavior.
- Do NOT rename the functions — keep the leading underscore convention.
- Do NOT move helpers to a new `tests/helpers.py` module — conftest is the project convention.

### Project Structure Notes

- Tests mirror `src/` structure: `tests/unit/`, `tests/integration/`, `tests/fixtures/`
- Root conftest: `tests/conftest.py` (contains `temp_data_dir` only)
- No existing `tests/unit/conftest.py` — will need to be created
- All Python files require `from __future__ import annotations`
- `mypy --strict` applies to all files in `tests/`

### Testing Standards

- Run full suite: `pytest` (expect ~1123 tests to pass)
- Type check: `mypy --strict src/ncaa_eval tests`
- Lint: `ruff check .`
- No new tests needed — this is a pure refactor of existing test infrastructure

### References

- [Source: _bmad-output/planning-artifacts/epics.md — Epic 9, Story 9.13]
- [Source: tests/unit/test_evaluation_splitter.py — lines 18-56]
- [Source: tests/unit/test_evaluation_backtest.py — lines 29-69]
- [Source: tests/conftest.py — root conftest structure]
- [Source: Audit item 2.21; PO decision 2026-03-11]
- [Source: Story 9.12 — previous story learnings on test patterns]

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
