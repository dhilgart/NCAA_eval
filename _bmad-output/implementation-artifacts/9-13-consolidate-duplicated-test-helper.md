# Story 9.13: Consolidate Duplicated Test Helper

Status: done

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

- [x] Task 1: Consolidate `_make_season_df` into shared location (AC: #1)
  - [x] 1.1 Create the unified `_make_season_df` in `tests/unit/conftest.py` — use the **backtest version** (it's a superset with `elo_diff` and `win_pct_diff` columns the splitter version lacks)
  - [x] 1.2 Remove `_make_season_df` from `tests/unit/test_evaluation_splitter.py`
  - [x] 1.3 Remove `_make_season_df` from `tests/unit/test_evaluation_backtest.py`
  - [x] 1.4 Verify both test files can access the helper (pytest auto-discovers conftest.py fixtures/helpers)
- [x] Task 2: Consolidate `_make_feature_server` into shared location (AC: #1)
  - [x] 2.1 Move the identical `_make_feature_server` helper to `tests/unit/conftest.py`
  - [x] 2.2 Remove from both `test_evaluation_splitter.py` and `test_evaluation_backtest.py`
- [x] Task 3: Run full test suite and verify no regressions (AC: #1)
  - [x] 3.1 `pytest` — all tests pass (1123 passed, 1 skipped)
  - [x] 3.2 `mypy --strict src/ncaa_eval tests` — clean (103 source files)
  - [x] 3.3 `ruff check .` — clean

### Review Follow-ups (AI)
- [ ] [AI-Review][HIGH] Add `spec=StatefulFeatureServer` to `MagicMock()` in `_make_feature_server` and fix return type to `StatefulFeatureServer` via `cast` — prevents silent swallowing of protocol typos [tests/unit/conftest.py:65-76]
- [ ] [AI-Review][MEDIUM] Add `assert year >= 1` (or docstring note) to `_make_season_df` — year=0 produces `DateParseError` from malformed date string [tests/unit/conftest.py:19]
- [ ] [AI-Review][LOW] Expand `_make_fold` docstring to note that shared `rng` state means output size of first call affects second call's random sequence [tests/unit/test_evaluation_backtest.py:30]

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

Claude Opus 4.6

### Debug Log References

- Initial `from conftest import ...` failed with `ModuleNotFoundError` because `tests/unit/` is a Python package (has `__init__.py`). Fixed by using fully-qualified import: `from tests.unit.conftest import ...`.
- Ruff flagged import ordering (I001) after adding the new import — resolved by running `ruff check . --fix`.

### Completion Notes List

- Created `tests/unit/conftest.py` with the backtest version of `_make_season_df` (11-column superset including `elo_diff` and `win_pct_diff`) and `_make_feature_server`.
- Removed both helpers from `test_evaluation_splitter.py` and `test_evaluation_backtest.py`.
- Both test files now import via `from tests.unit.conftest import _make_feature_server, _make_season_df`.
- Cleaned up unused imports (`MagicMock`, `numpy`) from `test_evaluation_splitter.py`.
- All 1123 tests pass, mypy --strict clean, ruff clean.

### File List

- `tests/unit/conftest.py` (new) — shared test helpers
- `tests/unit/test_evaluation_splitter.py` (modified) — removed duplicated helpers, added import from conftest
- `tests/unit/test_evaluation_backtest.py` (modified) — removed duplicated helpers, added import from conftest
- `_bmad-output/implementation-artifacts/9-13-consolidate-duplicated-test-helper.md` (modified) — story updates
- `_bmad-output/implementation-artifacts/sprint-status.yaml` (modified) — status update

### Change Log

- 2026-03-11: Consolidated `_make_season_df` and `_make_feature_server` from two test files into shared `tests/unit/conftest.py` to follow DRY principles
- 2026-03-11: [Code Review] Fixed self-play collision defect in `_make_season_df` — `team_b_id` now draws from `[2000,3000)` instead of `[1000,2000)` to guarantee distinct team IDs per game. Expanded module and function docstrings for clarity.
