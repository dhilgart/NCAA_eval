# Story 9.14: Add Pandera Schema Validation to KaggleConnector

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want to add Pandera schema validation to the KaggleConnector's CSV parsing,
so that data integrity issues are caught at the ingest boundary with clear, structured error messages.

## Acceptance Criteria

1. **Given** the KaggleConnector parses CSV files into DataFrames
   **When** a CSV file has unexpected columns, types, or value ranges
   **Then** Pandera schema validation catches the issue with a descriptive error
   **And** the existing `DataFormatError` exception type is preserved (wrap Pandera `SchemaError`)
   **And** the iterrows usage is NOT changed (accepted per item 2.4 carve-out for ingest layer)

## Tasks / Subtasks

- [x] Task 1: Define Pandera schemas for each CSV type (AC: #1)
  - [x] 1.1 Create a `schemas` module or add schemas to `kaggle.py` — define `TeamsSchema`, `SpellingsSchema`, `GamesSchema`, `SeasonsSchema` using `pa.DataFrameSchema` with `pa.Column` objects
  - [x] 1.2 `TeamsSchema`: `TeamID` (int, >= 1), `TeamName` (str, non-null)
  - [x] 1.3 `SpellingsSchema`: `TeamNameSpelling` (str, non-null), `TeamID` (int, >= 1)
  - [x] 1.4 `GamesSchema`: `Season` (int, >= 1985), `DayNum` (int, >= 0), `WTeamID` (int, >= 1), `LTeamID` (int, >= 1), `WScore` (int, >= 0), `LScore` (int, >= 0), `WLoc` (str, isin ["H", "A", "N"]), `NumOT` (int, >= 0)
  - [x] 1.5 `SeasonsSchema`: `Season` (int, >= 1985), `DayZero` (str, non-null)
- [x] Task 2: Integrate Pandera validation into KaggleConnector (AC: #1)
  - [x] 2.1 Replace `_validate_columns()` calls with `schema.validate(df)` calls
  - [x] 2.2 Wrap `pandera.errors.SchemaError` in `DataFormatError` to preserve the existing exception contract
  - [x] 2.3 Remove the manual `_validate_columns` function and the `_*_COLUMNS` sets (replaced by Pandera schemas)
  - [x] 2.4 Remove the manual `WLoc` validation in `_parse_games_csv` (line 244-247) — now handled by Pandera `isin` check
- [x] Task 3: Update tests for Pandera validation (AC: #1)
  - [x] 3.1 Existing tests for missing columns (`test_fetch_teams_missing_columns`) should still pass — Pandera also rejects missing columns
  - [x] 3.2 Add test: wrong type in a column (e.g., "abc" in TeamID) raises `DataFormatError`
  - [x] 3.3 Add test: value range violation (e.g., negative TeamID) raises `DataFormatError`
  - [x] 3.4 Add test: invalid WLoc value raises `DataFormatError` (replaces manual check)
  - [x] 3.5 Verify all existing tests still pass unchanged
- [x] Task 4: Run quality gates (AC: #1)
  - [x] 4.1 `pytest` — all tests pass
  - [x] 4.2 `mypy --strict src/ncaa_eval tests` — clean
  - [x] 4.3 `ruff check .` — clean

## Dev Notes

### Key Implementation Details

**Pandera is already installed** (v0.29.0) and listed in `pyproject.toml` (`pandera = "*"`). No dependency changes needed.

**Target file:** `src/ncaa_eval/ingest/connectors/kaggle.py` (265 lines)

**Current validation approach** (lines 34-56): Manual `_validate_columns()` function checks only for missing columns via set difference. No type or value range validation exists at the DataFrame level. Value validation happens later in the Pydantic models (Game, Team, Season) but only row-by-row during iterrows.

**Why Pandera adds value over the current approach:**
- Catches type mismatches before row-by-row iteration (fail-fast on corrupt CSV data)
- Validates value ranges at the DataFrame level (e.g., negative TeamIDs, unknown WLoc values)
- Provides structured, descriptive error messages automatically
- Replaces 4 separate `_*_COLUMNS` sets + `_validate_columns()` function with declarative schemas

**Pandera API pattern to use:**
```python
import pandera as pa  # v0.29.0

_GAMES_SCHEMA = pa.DataFrameSchema({
    "Season": pa.Column(int, pa.Check.ge(1985)),
    "DayNum": pa.Column(int, pa.Check.ge(0)),
    "WTeamID": pa.Column(int, pa.Check.ge(1)),
    "LTeamID": pa.Column(int, pa.Check.ge(1)),
    "WScore": pa.Column(int, pa.Check.ge(0)),
    "LScore": pa.Column(int, pa.Check.ge(0)),
    "WLoc": pa.Column(str, pa.Check.isin(["H", "A", "N"])),
    "NumOT": pa.Column(int, pa.Check.ge(0)),
})
```

**Error wrapping pattern:**
```python
try:
    schema.validate(df)
except pa.errors.SchemaError as exc:
    raise DataFormatError(f"kaggle: {filename} schema validation failed: {exc}") from exc
```

**What NOT to do:**
- Do NOT change iterrows usage — accepted per item 2.4 carve-out for ingest layer
- Do NOT add Pandera schemas to `schema.py` — that file is for Pydantic models. Keep Pandera schemas in `kaggle.py` near the parsing logic they guard
- Do NOT use `coerce=True` — we want to catch type mismatches, not silently fix them
- Do NOT use class-based Pandera `SchemaModel` — the functional `DataFrameSchema` API is simpler and sufficient for CSV column validation
- Do NOT use lazy validation — fail-fast is correct here; first error should halt parsing
- Do NOT modify the Pydantic models in `schema.py` — they already validate individual records correctly
- Do NOT add Pandera validation to the ESPN connector — out of scope (only KaggleConnector per story AC)
- Do NOT change column names or add aliases — match the exact CSV column names from Kaggle

**MSeasons.csv has extra columns** (`RegionW`, `RegionX`, `RegionY`, `RegionZ`) beyond what `_SEASONS_COLUMNS` requires. The Pandera schema must NOT reject extra columns — use the default `strict=False` setting (which allows additional columns).

**MTeamSpellings.csv** is parsed by `fetch_team_spellings()` — this method also needs Pandera validation. No fixture file exists for it in tests, so either add one or skip testing that path specifically.

### Previous Story Intelligence (Story 9.13)

- Created `tests/unit/conftest.py` with shared helpers
- Import pattern: `from tests.unit.conftest import _make_season_df` (explicit imports needed because `tests/unit/` is a Python package with `__init__.py`)
- Debug lesson: `from conftest import ...` fails with `ModuleNotFoundError` — use fully qualified path
- All 1123 tests passing as of Story 9.13 completion

### Git Intelligence

Recent commits follow pattern: `feat(ingest): ...` for ingest-layer changes. Story 9.5 (`c29be5f`) added post-sync validation to `src/ncaa_eval/ingest/validation.py` — this is a separate concern (post-sync vs. parse-time) and should NOT be confused with this story's scope.

### Test Fixture Files

Located at `tests/fixtures/kaggle/`:
- `MTeams.csv` — 4 teams (IDs 1101-1104)
- `MSeasons.csv` — 2 seasons (2023-2024) with DayZero dates and Region columns
- `MRegularSeasonCompactResults.csv` — 3 games (2 for 2024, 1 for 2023)
- `MNCAATourneyCompactResults.csv` — 2 tournament games (2024)
- No `MTeamSpellings.csv` fixture — spellings tests would need one added, or test via the `_read_csv` mock

### Pandera Import for mypy

Pandera's type stubs may need `# type: ignore[import-untyped]` like the existing pandas import on line 17. Check if `pandera` has inline types or needs the ignore comment.

### Project Structure Notes

- All changes confined to `src/ncaa_eval/ingest/connectors/kaggle.py` and `tests/unit/test_kaggle_connector.py`
- `from __future__ import annotations` required in all modified files (already present)
- `mypy --strict` applies — ensure Pandera schema objects are properly typed

### Testing Standards

- Run full suite: `pytest` (expect ~1123 tests to pass)
- Type check: `mypy --strict src/ncaa_eval tests`
- Lint: `ruff check .`
- New tests should go in `TestKaggleConnectorTeams` or new `TestKaggleConnectorSchemaValidation` class
- Follow existing test patterns: use `connector` fixture, `kaggle_dir`, `tmp_path`

### References

- [Source: _bmad-output/planning-artifacts/epics.md — Epic 9, Story 9.14]
- [Source: _bmad-output/planning-artifacts/po-decision-log-epic8.md — §2.17, PO Decision C]
- [Source: _bmad-output/planning-artifacts/codebase-audit-report.md — §2.17]
- [Source: src/ncaa_eval/ingest/connectors/kaggle.py — full file, current validation at lines 34-56]
- [Source: src/ncaa_eval/ingest/connectors/base.py — DataFormatError exception]
- [Source: src/ncaa_eval/ingest/schema.py — Pydantic models with field constraints]
- [Source: tests/unit/test_kaggle_connector.py — existing test patterns]
- [Source: Pandera docs — DataFrameSchema API: https://pandera.readthedocs.io/en/stable/dataframe_schemas.html]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

- Pandera 0.29.0 deprecation: `import pandera as pa` triggers FutureWarning; used `import pandera.pandas as pa` instead
- Pandera has `py.typed` marker — no `# type: ignore` needed for mypy
- Existing `test_fetch_teams_missing_columns` matched on "missing columns" but Pandera error says "column 'X' not in dataframe" — updated match pattern to "schema validation failed"
- `import pandera as pa` triggers FutureWarning in v0.29+; must use `import pandera.pandas as pa`. `pandera.errors` does not re-export through `pandera.pandas` so it is imported separately — explanatory comment added to source
- `fetch_team_spellings()` originally called `.astype(int)` on TeamID Series; removed as redundant after Pandera enforces int dtype
- `test_wrong_type_in_team_id`: feeding "abc" in TeamID causes pandas to infer `object` dtype; Pandera then rejects the type mismatch — this tests dtype inference, not a runtime type error per se
- Code review fix (2026-03-11): `fetch_seasons()` was re-reading MSeasons.csv independently of `load_day_zeros()`; refactored to delegate, eliminating duplicate disk read and Pandera validation
- Code review fix (2026-03-11): Added `tests/fixtures/kaggle/MTeamSpellings.csv` fixture and 3 tests covering `_SPELLINGS_SCHEMA` — this was the only schema path with zero test coverage

### Completion Notes List

- Defined 4 Pandera `DataFrameSchema` objects (`_TEAMS_SCHEMA`, `_SPELLINGS_SCHEMA`, `_GAMES_SCHEMA`, `_SEASONS_SCHEMA`) directly in `kaggle.py`
- Created `_validate_schema()` helper that wraps `pandera.errors.SchemaError` in `DataFormatError`
- Replaced all 5 `_validate_columns()` calls with `_validate_schema()` calls
- Removed `_validate_columns()` function and all `_*_COLUMNS` sets
- Removed manual WLoc validation (lines 244-247) — now handled by `_GAMES_SCHEMA` `isin` check
- Added 4 new tests in `TestKaggleConnectorSchemaValidation`: wrong type, negative value, invalid WLoc, negative score
- Updated `test_fetch_teams_missing_columns` match pattern for Pandera error format
- All 1127 tests pass, mypy --strict clean, ruff check clean (post-dev; post-review: 1130 tests)
- iterrows usage untouched per carve-out
- Code review (2026-03-11): Refactored `fetch_seasons()` to delegate to `load_day_zeros()`; added spellings fixture + 3 tests; removed redundant `.astype(int)`; added import comment

### Change Log

- 2026-03-11: Replaced manual CSV column validation with Pandera schema validation in KaggleConnector (Story 9.14)
- 2026-03-11: Code review fixes — eliminated duplicate MSeasons.csv validation, added spellings coverage, minor code cleanup

### File List

- `src/ncaa_eval/ingest/connectors/kaggle.py` (modified)
- `tests/unit/test_kaggle_connector.py` (modified)
- `tests/fixtures/kaggle/MTeamSpellings.csv` (added)
- `_bmad-output/implementation-artifacts/9-14-add-pandera-schema-validation-kaggle.md` (modified)
- `_bmad-output/implementation-artifacts/sprint-status.yaml` (modified)
