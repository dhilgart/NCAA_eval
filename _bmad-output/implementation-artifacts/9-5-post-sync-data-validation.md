# Story 9.5: Post-Sync Data Validation

Status: review

## Story

As a **data scientist**,
I want **automatic validation checks to run after data sync completes**,
so that **I can detect data quality issues (missing games, duplicates, team reference errors) before they silently corrupt downstream predictions**.

## Acceptance Criteria

1. **Given** a data sync (`ncaa-eval sync` or `python sync.py`) completes
   **When** the sync finishes downloading and persisting data
   **Then** a validation step runs automatically checking:
   - Game count per season is within expected range (±10% of historical average)
   - No duplicate games exist (same teams, same day)
   - All team IDs in games reference valid entries in the teams table
   **And** validation results are logged at INFO level with a summary
   **And** validation warnings do not block the sync (non-fatal) but are clearly visible

## Tasks / Subtasks

- [x] Task 1: Create `ValidationResult` dataclass and `validate_sync` function (AC: #1)
  - [x] 1.1: Create `src/ncaa_eval/ingest/validation.py` with a `ValidationResult` Pydantic model containing `check_name: str`, `passed: bool`, `message: str`, `details: dict[str, Any]`
  - [x] 1.2: Create `ValidationReport` Pydantic model aggregating multiple `ValidationResult` items, with `all_passed: bool` property
  - [x] 1.3: Implement `validate_sync(repo: Repository) -> ValidationReport` — top-level function that runs all three checks and returns results
  - [x] 1.4: Export `validate_sync` and `ValidationReport` from `ncaa_eval.ingest.__init__.py` / `__all__`

- [x] Task 2: Implement game count validation check (AC: #1 — game count)
  - [x] 2.1: Implement `_check_game_counts(repo: Repository) -> list[ValidationResult]` — loads all seasons, counts games per season, compares against expected range
  - [x] 2.2: Compute historical average from loaded data dynamically (median of all seasons' game counts) rather than hardcoding a constant — this handles dataset evolution gracefully
  - [x] 2.3: Flag any season where `count < median * 0.9` or `count > median * 1.1` (±10% threshold)
  - [x] 2.4: Special-case season 2020 (COVID) — skip or annotate as known anomaly (no tournament, shortened season)

- [x] Task 3: Implement duplicate game detection (AC: #1 — duplicates)
  - [x] 3.1: Implement `_check_duplicate_games(repo: Repository) -> list[ValidationResult]` — loads games per season, checks for duplicates by `(season, day_num, w_team_id, l_team_id)` tuple
  - [x] 3.2: Report duplicate count per season, include example duplicate pairs in details
  - [x] 3.3: The 2025 Kaggle+ESPN duplicate issue (4,545 doubled games) will be detected and reported — this is expected and validates the check works

- [x] Task 4: Implement team reference integrity check (AC: #1 — team references)
  - [x] 4.1: Implement `_check_team_references(repo: Repository) -> list[ValidationResult]` — loads teams, then checks every game's `w_team_id` and `l_team_id` against known team IDs
  - [x] 4.2: Report orphan team IDs (game references a team not in the teams table) with counts and affected seasons

- [x] Task 5: Integrate validation into sync pipeline (AC: #1)
  - [x] 5.1: In `sync.py` (project-root CLI), call `validate_sync(repo)` after sync completes
  - [x] 5.2: Log validation summary at INFO level — pass/fail per check with counts
  - [x] 5.3: Log individual warnings at WARNING level for each failed check
  - [x] 5.4: Ensure validation failures do NOT raise exceptions or exit with non-zero code — sync always completes successfully

- [x] Task 6: Add comprehensive tests (AC: all)
  - [x] 6.1: Unit tests in `tests/unit/test_validation.py` — test each check function with controlled data (known duplicates, missing teams, abnormal counts)
  - [x] 6.2: Integration test in `tests/integration/test_sync.py` — add a test verifying validation runs after sync and produces expected results
  - [x] 6.3: Run full test suite: `pytest`, `ruff check .`, `mypy --strict src/ncaa_eval tests`

## Dev Notes

### Architecture & Design Decisions

**New file:** `src/ncaa_eval/ingest/validation.py` — standalone validation module within the ingest package. This keeps validation concerns co-located with the data pipeline rather than in a separate top-level module.

**Function-based API, not a class:** Use a simple `validate_sync(repo: Repository)` function rather than a `DataValidator` class. The validation is stateless — it reads data from the repository and produces results. A class adds no value here.

**Depend on `Repository` interface, not `ParquetRepository`:** All validation functions accept `Repository` (the abstract base) for testability. Tests can use the real `ParquetRepository` with `tmp_path` fixtures (matching existing test patterns).

### Key Implementation Details

#### Game Count Validation
- Load all seasons via `repo.get_seasons()`, then `repo.get_games(season)` for each
- Compute median game count across all loaded seasons as the baseline
- Apply ±10% threshold: `median * 0.9 <= count <= median * 1.1`
- **COVID 2020:** The 2020 season has no tournament and a shortened regular season (~1,200 games vs ~5,500 typical). Either skip 2020 from the median calculation, or accept it as a known anomaly. Recommendation: exclude 2020 from the median but still validate it (it will flag as anomalous, which is correct behavior)
- **Performance:** Loading all games for all seasons may be slow (~40 seasons × ~5,500 games = ~220K records). Consider loading game counts from Parquet metadata (row counts) if performance is a concern, but for correctness start with full loads

#### Duplicate Detection
- Duplicate key: `(season, day_num, w_team_id, l_team_id)` — this is the natural dedup key per MEMORY.md
- The 2025 season is known to have ~4,545 duplicates (Kaggle + ESPN versions of the same game with different `game_id` formats). This is not a bug in the validation — it correctly identifies the duplication
- Build a `Counter` or `set` of tuples, report any with count > 1

#### Team Reference Integrity
- Load all teams via `repo.get_teams()`, build `set[int]` of valid team IDs
- For each season's games, check `w_team_id in team_ids` and `l_team_id in team_ids`
- Collect orphan IDs with their season context

### Logging Pattern

Follow the existing sync logging convention:
```python
from ncaa_eval.utils.logger import get_logger

log = get_logger("ingest.validation")

# Summary line
log.info("[validation] 3/3 checks passed")
# Or with warnings:
log.info("[validation] 2/3 checks passed, 1 warning")
log.warning("[validation] duplicate games: 4545 duplicates in season 2025")
```

### Error Handling

Validation is **non-fatal** per the AC. The `validate_sync` function:
- MUST NOT raise exceptions on validation failures (only on unexpected errors like I/O)
- Returns a `ValidationReport` with pass/fail status per check
- The caller (`sync.py`) logs results and continues

### Existing Patterns to Follow

| Pattern | Source | What to Reuse |
|---------|--------|---------------|
| Pydantic models for results | `ingest/schema.py` (Game, Team, Season) | Use `BaseModel` for `ValidationResult` |
| Module-level logger | `ingest/sync.py:21` | `log = get_logger("ingest.validation")` |
| Repository abstraction | `ingest/repository.py` (Repository ABC) | Accept `Repository` not `ParquetRepository` |
| Test fixtures | `tests/conftest.py` (`temp_data_dir`) | Use `tmp_path` + `ParquetRepository` for test isolation |
| Test structure | `tests/unit/test_repository.py` | Class-per-scenario, descriptive names |
| `SyncResult` dataclass | `ingest/sync.py:100-108` | Model `ValidationResult` similarly |

### Files to Create

- `src/ncaa_eval/ingest/validation.py` — validation functions and result models

### Files to Modify

- `src/ncaa_eval/ingest/__init__.py` — add `validate_sync`, `ValidationReport` to imports and `__all__`
- `sync.py` (project root) — call `validate_sync(repo)` after sync, log results
- `tests/unit/test_validation.py` — new test file for validation unit tests
- `tests/integration/test_sync.py` — add integration test for validation-after-sync

### Files NOT to Modify

- `src/ncaa_eval/ingest/sync.py` — validation is NOT called from `SyncEngine`; it's called from the CLI entry point (`sync.py`) after the engine finishes. This keeps `SyncEngine` focused on data fetching/persisting.
- `src/ncaa_eval/ingest/repository.py` — no changes needed to the repository
- `src/ncaa_eval/ingest/schema.py` — no changes to data models

### Previous Story Intelligence

Stories 9.1–9.4 established patterns:
- Each story added 13-18 tests
- Code review was thorough — expect scrutiny on edge cases (COVID year, 2025 duplicates)
- `mypy --strict` and `ruff check .` enforced on all new code
- Import paths use submodule pattern: `from ncaa_eval.ingest import validate_sync`

Story 8.3 (ESPN retry logic) decoupled `SyncEngine` from Typer and generalized dedup — the sync pipeline is clean and modular, making this story's integration point straightforward.

### Testing Standards

- `pytest` — full suite must pass
- `ruff check .` — all linting must pass
- `mypy --strict src/ncaa_eval tests` — full type checking
- New unit tests should cover:
  - All-pass scenario (clean data)
  - Each individual check failing independently
  - Multiple checks failing simultaneously
  - Edge cases: empty repository, single season, COVID 2020
  - The 2025 duplicate scenario (expected behavior)
- Integration test should verify end-to-end: sync → validate → log output

### Project Structure Notes

- New file `validation.py` goes in `src/ncaa_eval/ingest/` alongside `sync.py`, `repository.py`, `schema.py`
- Test file `test_validation.py` goes in `tests/unit/` alongside `test_repository.py`, `test_kaggle_connector.py`
- `from __future__ import annotations` required in all new Python files

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story-9.5] — Story requirements and AC
- [Source: _bmad-output/planning-artifacts/codebase-audit-report.md#2.20] — Original audit finding
- [Source: _bmad-output/planning-artifacts/po-decision-log-epic8.md#2.20] — PO decision to implement
- [Source: src/ncaa_eval/ingest/sync.py] — SyncEngine (fetch + persist pipeline)
- [Source: src/ncaa_eval/ingest/repository.py] — Repository ABC and ParquetRepository
- [Source: src/ncaa_eval/ingest/schema.py] — Game, Team, Season models
- [Source: sync.py] — CLI entry point where validation will be called
- [Source: tests/integration/test_sync.py] — Existing sync integration tests
- [Source: tests/unit/test_repository.py] — Repository unit test patterns

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

None — clean implementation with no blocking issues.

### Completion Notes List

- Implemented `ValidationResult` and `ValidationReport` as frozen Pydantic models in `src/ncaa_eval/ingest/validation.py`
- `validate_sync(repo)` runs three checks: game counts (±10% of median), duplicate games, team reference integrity
- Game count check excludes COVID 2020 from median calculation but still validates it (flags as expected anomaly)
- Duplicate detection uses `(season, day_num, w_team_id, l_team_id)` tuple key via `Counter`
- Team reference check validates all `w_team_id`/`l_team_id` against known teams
- Logging: INFO summary line + WARNING for each failed check (via `get_logger("ingest.validation")`)
- Integrated into `sync.py` CLI — `validate_sync(repo)` called after sync completes, non-fatal
- 23 unit tests covering all checks (pass, fail, edge cases: empty repo, single season, COVID, multiple failures)
- 1 integration test verifying validation runs after CLI sync and produces log output
- All 1004 tests pass, `ruff check .` clean, `mypy --strict` clean, `ruff format` clean

### Change Log

- 2026-03-09: Implemented post-sync data validation (Story 9.5) — added validation module with 3 checks, integrated into sync CLI, added 24 tests

### File List

- `src/ncaa_eval/ingest/validation.py` (new) — validation checks and result models
- `src/ncaa_eval/ingest/__init__.py` (modified) — export `validate_sync`, `ValidationReport`
- `sync.py` (modified) — call `validate_sync(repo)` after sync
- `tests/unit/test_validation.py` (new) — 23 unit tests for validation
- `tests/integration/test_sync.py` (modified) — 1 integration test for validation-after-sync
- `_bmad-output/implementation-artifacts/9-5-post-sync-data-validation.md` (modified) — story updates
- `_bmad-output/implementation-artifacts/sprint-status.yaml` (modified) — status → review
