# Story 8.3: Fix Data Pipeline Resilience — ESPN Error Handling, Retry Logic, Typer Decoupling

Status: done

## Story

As a data scientist,
I want the data pipeline to surface failures clearly, retry transient errors, and decouple from CLI-specific output,
so that ESPN sync reports partial failures visibly, retries transient network errors automatically, and the ingest layer can be used from notebooks and tests without pulling in Typer.

## Acceptance Criteria

### AC1: ESPN Per-Team Fetch Retry with Tenacity

1. `tenacity` is added to `[tool.poetry.dependencies]` in `pyproject.toml` (already installed transitively via streamlit — this makes it explicit)
2. The per-team `ms.get_team_schedule()` call in `espn.py:_fetch_per_team()` is wrapped with a `@tenacity.retry` decorator (or inline `Retrying` context) configured for: 3 attempts, exponential backoff starting at 2s (wait_exponential with multiplier=2, min=2, max=30), retry only on `Exception` (excluding `KeyboardInterrupt`)
3. After all retries are exhausted for a team, the failure is logged at WARNING with team name and exception, and the loop continues to the next team (current behavior preserved, but now with retries first)
4. The `# noqa: BLE001` comment on the bare `except Exception` in `_fetch_per_team` is retained — the broad catch is intentional here because cbbpy can raise any exception type on network/parsing failures

### AC2: ESPN Fetch Summary Reporting

5. After the per-team loop in `_fetch_per_team()` completes, a summary is logged at INFO level: `"espn: fetched {success}/{total} teams for season {year} ({failed} failed)"` where `success` = number of teams that returned non-empty DataFrames, `total` = total teams in mapping, `failed` = total - success
6. If `failed > 0`, the summary is logged at WARNING (not INFO) and includes a list of the first 5 failed team names
7. The return type and behavior of `_fetch_per_team()` is unchanged (still returns `pd.DataFrame | None`)

### AC3: ESPN Date Parsing Logging

8. The silent `except Exception: return None` in `EspnConnector._parse_date()` (line 240) now logs at DEBUG level: `"espn: could not parse date value %r for game"` before returning `None`
9. The `# noqa: BLE001` is retained on this handler

### AC4: SyncEngine Decoupled from Typer

10. All `typer.echo()` calls in `src/ncaa_eval/ingest/sync.py` are replaced with `logger.info()` calls using the existing module-level `logger`
11. The `import typer` line is removed from `src/ncaa_eval/ingest/sync.py`
12. The CLI entry point (`sync.py` at repo root) remains responsible for configuring log output format — the ingest layer only emits log messages
13. No behavioral change to CLI users: sync progress messages still appear (the CLI configures a handler that prints INFO-level ingest messages to stdout)

### AC5: Generalize Deduplication Beyond 2025

14. `_deduplicate_2025()` in `src/ncaa_eval/transform/serving.py` is renamed to `_deduplicate_espn_overlap(games)` (no year parameter — the function detects ESPN duplicates by checking for `espn_`-prefixed game IDs)
15. The `if year == 2025:` guard in `ChronologicalDataServer.get_chronological_season()` is replaced with a check that runs deduplication when **any** game in the list has a game_id starting with `"espn_"` — this automatically handles 2026+ seasons
16. The deduplication logic itself is unchanged (same triplet key, same ESPN-preferred keep strategy)
17. The function docstring is updated to reflect the generalized scope (remove "2025" references)

### AC6: Centralize Fuzzy Match Utility

18. A new module `src/ncaa_eval/ingest/fuzzy.py` is created containing a single public function: `fuzzy_match_team(name: str, candidates: dict[str, int], threshold: int = 80) -> int | None` that performs case-insensitive exact lookup then falls back to `rapidfuzz.fuzz.token_set_ratio` with the given threshold
19. `espn.py:_resolve_team_id()` is refactored to use `fuzzy_match_team()` for the fuzzy fallback (the exact-match-first + logging behavior is preserved)
20. `sync.py:_build_espn_team_map()` is refactored to use `fuzzy_match_team()` for the fuzzy fallback (the override-first + exact-match-first behavior is preserved)
21. The `_FUZZY_THRESHOLD` constants in both `espn.py` and `sync.py` are removed (the default parameter on `fuzzy_match_team` provides the canonical value)
22. Both `espn.py` and `sync.py` no longer import `from rapidfuzz import fuzz` directly — only `fuzzy.py` imports rapidfuzz

### AC7: Declare rapidfuzz Dependency

23. `rapidfuzz` is added to `[tool.poetry.dependencies]` in `pyproject.toml` with version constraint `"*"` (matching the project's convention for non-pinned deps)

### AC8: Fix PydanticUndefined Sentinel

24. `src/ncaa_eval/ingest/repository.py` line 102: `sentinel: Any = ...` is replaced with `from pydantic.fields import PydanticUndefined` and `sentinel = PydanticUndefined`
25. The comparison `default is not sentinel` continues to work correctly because `PydanticUndefined` is a singleton that supports identity comparison

### AC9: Backtest Exception Handler (Already Implemented — Verify Only)

26. `src/ncaa_eval/evaluation/backtest.py:186-187` already logs at WARNING with `exc_info=True` — verify this is the case and close this item as pre-existing fix. No code change needed.

### AC10: Quality Gates

27. `ruff check .` passes (no new violations)
28. `mypy --strict src/ncaa_eval tests` passes
29. All existing tests pass (behavioral equivalence for all sync/fetch/dedup operations)
30. No behavioral changes to CLI output for end users (sync progress messages still visible via logging)

## Tasks / Subtasks

- [x] Task 1: Add `tenacity` and `rapidfuzz` to pyproject.toml (AC: #1, #23)
  - [x] 1.1 Add `tenacity = "*"` to `[tool.poetry.dependencies]`
  - [x] 1.2 Add `rapidfuzz = "*"` to `[tool.poetry.dependencies]`
  - [x] 1.3 Run `ruff check pyproject.toml` to verify

- [x] Task 2: Create fuzzy match utility module (AC: #18)
  - [x] 2.1 Create `src/ncaa_eval/ingest/fuzzy.py` with `fuzzy_match_team()` function
  - [x] 2.2 Include `from __future__ import annotations`, type annotations, Google-style docstring
  - [x] 2.3 Run `mypy --strict src/ncaa_eval/ingest/fuzzy.py`

- [x] Task 3: Refactor ESPN connector (AC: #2-4, #8-9, #19, #21-22)
  - [x] 3.1 Add tenacity retry to `_fetch_per_team()` inner loop (3 retries, exponential backoff 2s-30s)
  - [x] 3.2 Track success/failure counts during per-team loop
  - [x] 3.3 Log summary after loop completion (AC #5-6)
  - [x] 3.4 Add DEBUG log to `_parse_date()` exception handler (AC #8)
  - [x] 3.5 Refactor `_resolve_team_id()` to use `fuzzy_match_team()` from `fuzzy.py` (AC #19)
  - [x] 3.6 Remove `from rapidfuzz import fuzz` import and `_FUZZY_THRESHOLD` constant (AC #21-22)
  - [x] 3.7 Run `mypy --strict src/ncaa_eval/ingest/connectors/espn.py`

- [x] Task 4: Decouple SyncEngine from Typer (AC: #10-13)
  - [x] 4.1 Replace all `typer.echo(...)` calls with `logger.info(...)` in `sync.py`
  - [x] 4.2 Remove `import typer` from `sync.py`
  - [x] 4.3 Refactor `_build_espn_team_map()` to use `fuzzy_match_team()` (AC #20-22)
  - [x] 4.4 Remove `from rapidfuzz import fuzz` import and `_FUZZY_THRESHOLD` constant
  - [x] 4.5 Verify CLI entry point (`sync.py` at repo root) configures logging so messages still appear
  - [x] 4.6 Run `mypy --strict src/ncaa_eval/ingest/sync.py`

- [x] Task 5: Generalize deduplication (AC: #14-17)
  - [x] 5.1 Rename `_deduplicate_2025` → `_deduplicate_espn_overlap` in `serving.py`
  - [x] 5.2 Replace `if year == 2025:` guard with ESPN-prefix detection
  - [x] 5.3 Update function docstring to remove 2025-specific language
  - [x] 5.4 Run `mypy --strict src/ncaa_eval/transform/serving.py`

- [x] Task 6: Fix PydanticUndefined sentinel (AC: #24-25)
  - [x] 6.1 Replace `sentinel: Any = ...` with `from pydantic_core import PydanticUndefined` and usage
  - [x] 6.2 Run `mypy --strict src/ncaa_eval/ingest/repository.py`

- [x] Task 7: Verify backtest handler (AC: #26)
  - [x] 7.1 Confirm `backtest.py:186-187` already logs WARNING with `exc_info=True` — no code change

- [x] Task 8: Update tests (AC: #27-30)
  - [x] 8.1 Update `tests/unit/test_espn_connector.py` — add tests for retry behavior (mock tenacity to verify retry attempts)
  - [x] 8.2 Update `tests/unit/test_espn_connector.py` — add test for fetch summary logging (success/failure counts)
  - [x] 8.3 Add `tests/unit/test_fuzzy.py` — test `fuzzy_match_team()` with exact, fuzzy, and no-match cases
  - [x] 8.4 Update `tests/integration/test_sync.py` — verify SyncEngine works without typer import (no `typer.echo` calls)
  - [x] 8.5 Update `tests/unit/test_chronological_serving.py` — verify deduplication triggers for any ESPN-prefixed season, not just 2025
  - [x] 8.6 Run full test suite: `pytest` — 883 passed, 1 skipped

- [x] Task 9: Final validation (AC: #27-30)
  - [x] 9.1 `ruff check .` — zero new violations (pre-existing notebook issues only)
  - [x] 9.2 `mypy --strict src/ncaa_eval tests` — zero errors (87 source files)
  - [x] 9.3 `pytest` — 883 passed, 1 skipped, 0 failures
  - [x] 9.4 Verify no behavioral changes for CLI users — `sync.py` (root) adds `logging.basicConfig(level=INFO, format="%(message)s")` so log messages print identically to previous `typer.echo` output

## Dev Notes

### Key Principle

This story addresses **Pattern D** from the codebase audit ("Silent Failure in Data Pipelines"). The ESPN connector, SyncEngine, and deduplication logic all independently suppress errors. The fix establishes a consistent pattern: log at WARNING, retry transient failures, report summaries, and never permanently cache incomplete results without indication.

### Architecture Patterns and Constraints

- **`from __future__ import annotations`** required in ALL Python files (Ruff-enforced)
- **Google-style docstrings** — not NumPy-style
- **`mypy --strict`** mandatory for `src/ncaa_eval/` and `tests/`
- **Library-First Rule**: Use `tenacity` for retry logic (already installed transitively via streamlit). Do NOT implement custom retry loops.
- **No behavioral changes to CLI output**: The sync CLI must still print progress messages to users. The change is architectural: `SyncEngine` emits log messages, the CLI configures log formatting.

### Tenacity Retry Configuration

The retry decorator should be applied to an inner helper, not the whole `_fetch_per_team` method. Suggested pattern:

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _fetch_single_team_schedule(team_name: str, season: int) -> pd.DataFrame | None:
    """Fetch a single team's schedule with retry on transient failures."""
    df = ms.get_team_schedule(team_name, season)
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df
    return None
```

Then `_fetch_per_team` calls this helper in the loop and catches the `RetryError` (or final exception after retries exhausted) to log the failure and continue.

**Important**: The retry wraps only the cbbpy HTTP call, NOT the entire loop. Each team gets its own 3 retries independently.

### SyncEngine Typer Decoupling Strategy

**Current state** (sync.py lines 160-251): 10 `typer.echo()` calls producing messages like:
- `"[kaggle] teams: 362 written"`
- `"[kaggle] season 2023: cache hit, skipped"`
- `"[espn] season 2025: 4545 games written"`

**Target state**: Replace each with `logger.info(...)` using the same message format. The CLI entry point (`sync.py` at repo root) already imports `logging` — it just needs to configure a handler that formats INFO messages to stdout. Example:

```python
# In sync.py (repo root CLI entry point):
logging.basicConfig(level=logging.INFO, format="%(message)s")
```

This ensures CLI users see the same progress messages, while notebook/test users get standard logging behavior.

### Deduplication Generalization

**Current** (`serving.py:185`):
```python
if year == 2025:
    games = _deduplicate_2025(games)
```

**Target**:
```python
if any(g.game_id.startswith("espn_") for g in games):
    games = _deduplicate_espn_overlap(games)
```

The function body is unchanged — only the guard condition and function name change. The `any()` check is O(N) but games lists are small (5K-10K per season) so performance is irrelevant.

### Fuzzy Match Centralization

**New module**: `src/ncaa_eval/ingest/fuzzy.py`

```python
def fuzzy_match_team(
    name: str,
    candidates: dict[str, int],
    threshold: int = 80,
) -> int | None:
    """Match a team name to a candidate mapping using exact then fuzzy lookup.

    Args:
        name: Team name to match.
        candidates: Mapping of known names to team IDs.
        threshold: Minimum rapidfuzz token_set_ratio score (0-100).

    Returns:
        Matched team ID, or None if no match meets the threshold.
    """
```

**espn.py refactoring**: `_resolve_team_id()` keeps its exact-match-first logic and its logging, but delegates the fuzzy fallback to `fuzzy_match_team()`.

**sync.py refactoring**: `_build_espn_team_map()` keeps its override-first + exact-match-first logic, but delegates the fuzzy fallback to `fuzzy_match_team()`.

### PydanticUndefined Fix

**Current** (`repository.py:102`):
```python
sentinel: Any = ...  # PydanticUndefined is represented as Ellipsis
```

**Target**:
```python
from pydantic.fields import PydanticUndefined
# ...
sentinel = PydanticUndefined
```

The comparison `default is not sentinel` still works because `PydanticUndefined` is a module-level singleton. The `Any` annotation can be dropped since the type is now concrete.

### ESPN Marker File Concern (Not In Scope — Document for Awareness)

The current marker-file caching (`_espn_marker(year)`) has a design flaw: `marker.touch()` runs even after partial failures, permanently caching incomplete data. This story does NOT fix the marker logic (scope creep), but the new summary logging (AC #5-6) makes partial failures visible so users know to re-run with `--force-refresh`. A future story could add a `.espn_synced_{year}.json` metadata file recording success/failure counts, but that is out of scope here.

### Files to Modify

| File | Changes |
|------|---------|
| `pyproject.toml` | Add `tenacity = "*"` and `rapidfuzz = "*"` to dependencies |
| `src/ncaa_eval/ingest/fuzzy.py` | **NEW** — centralized fuzzy match utility |
| `src/ncaa_eval/ingest/connectors/espn.py` | Add tenacity retry, fetch summary, date parse logging, use `fuzzy.py` |
| `src/ncaa_eval/ingest/sync.py` | Replace `typer.echo` → `logger.info`, remove typer import, use `fuzzy.py` |
| `src/ncaa_eval/transform/serving.py` | Rename/generalize deduplication function |
| `src/ncaa_eval/ingest/repository.py` | Use `PydanticUndefined` instead of `Ellipsis` |
| `tests/unit/test_espn_connector.py` | Add retry and summary tests |
| `tests/unit/test_fuzzy.py` | **NEW** — fuzzy match utility tests |
| `tests/integration/test_sync.py` | Verify no typer dependency in SyncEngine |
| `tests/unit/test_serving.py` | Verify generalized deduplication |

### Files NOT Modified

| File | Reason |
|------|--------|
| `src/ncaa_eval/evaluation/backtest.py` | AC #26 — already has WARNING + exc_info=True; no change needed |
| `sync.py` (repo root CLI) | May need minor logging config addition (verify existing setup) |

### Previous Story Learnings (Stories 8.1 & 8.2)

- **Backward compatibility via re-exports**: Not applicable here (no public API changes to existing modules)
- **mypy strict**: Dashboard files are NOT under mypy strict. Don't add `# type: ignore` to fix dashboard type issues
- **Pre-commit hooks**: `debug-statements`, `check-yaml`, `ruff`, `ruff-format` all run. The `template/` directory is excluded
- **`# noqa: BLE001` annotations**: Per template-requirements.md, keep `# noqa: BLE001` on all intentional broad except handlers, even after adding logging/retry (the handler is still broad by design)

### Git Intelligence

Recent commits show Stories 8.1 and 8.2 just completed (pure refactoring). The codebase is freshly reorganized:
- `evaluation/providers.py` — extracted from simulation.py in Story 8.1
- `evaluation/scoring.py` — extracted from simulation.py in Story 8.1
- `dashboard/lib/data_loaders.py` — extracted from filters.py in Story 8.1

None of these files are touched by Story 8.3. The `ingest/` module structure is unchanged since Epic 2.

### Source Document References

- [Source: `_bmad-output/planning-artifacts/codebase-audit-report.md` — Finding 3.2 (SyncEngine/typer coupling)]
- [Source: `_bmad-output/planning-artifacts/codebase-audit-report.md` — Finding 3.3 (hardcoded 2025 dedup)]
- [Source: `_bmad-output/planning-artifacts/codebase-audit-report.md` — Finding 3.8 (duplicate fuzzy match)]
- [Source: `_bmad-output/planning-artifacts/codebase-audit-report.md` — Finding 3.12 (backtest swallows exceptions)]
- [Source: `_bmad-output/planning-artifacts/codebase-audit-report.md` — Finding 3.14 (Ellipsis sentinel)]
- [Source: `_bmad-output/planning-artifacts/codebase-audit-report.md` — Finding 3.28 (ESPN silent exceptions)]
- [Source: `_bmad-output/planning-artifacts/codebase-audit-report.md` — Finding 3.29 (no retry logic)]
- [Source: `_bmad-output/planning-artifacts/codebase-audit-pass2-addendum.md` — Finding P2-1 (rapidfuzz undeclared dep)]
- [Source: `_bmad-output/planning-artifacts/codebase-audit-pass2-addendum.md` — Pattern D (Silent Failure in Data Pipelines)]
- [Source: `_bmad-output/planning-artifacts/epic-8-codebase-improvements.md` — Story 8.3 section]
- [Source: `_bmad-output/planning-artifacts/template-requirements.md` — `# noqa: BLE001` annotation pattern]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

None — clean implementation with no blocking issues.

### Completion Notes List

- **AC #1-4 (Retry)**: Created `_fetch_single_team_schedule()` with `@retry(stop=3, wait=exponential(2s-30s))`. `_fetch_per_team()` catches final exceptions after retries exhaust and continues to next team. `# noqa: BLE001` retained on broad except.
- **AC #5-7 (Summary)**: After loop, logs INFO for full success, WARNING for partial failure with first 5 failed team names. Return type unchanged.
- **AC #8-9 (Date parse)**: Added `logger.debug("espn: could not parse date value %r for game", value)` before `return None`. `# noqa: BLE001` retained.
- **AC #10-13 (Typer decoupling)**: Replaced 7 `typer.echo()` calls with `logger.info()` in `sync.py`, removed `import typer`. CLI entry point (`sync.py` root) adds `logging.basicConfig(level=INFO, format="%(message)s")` for behavioral equivalence.
- **AC #14-17 (Dedup generalization)**: Renamed `_deduplicate_2025` → `_deduplicate_espn_overlap`, replaced `if year == 2025:` guard with `if any(g.game_id.startswith("espn_") for g in games):`. Logic body unchanged.
- **AC #18-22 (Fuzzy centralization)**: Created `src/ncaa_eval/ingest/fuzzy.py` with `fuzzy_match_team()`. Both `espn.py` and `sync.py` refactored to use it. Removed duplicate `_FUZZY_THRESHOLD` constants and direct `rapidfuzz.fuzz` imports.
- **AC #23 (rapidfuzz dep)**: Added `rapidfuzz = "*"` to pyproject.toml.
- **AC #24-25 (PydanticUndefined)**: Replaced `sentinel: Any = ...` with `from pydantic_core import PydanticUndefined; sentinel = PydanticUndefined`. Used `pydantic_core` instead of `pydantic.fields` because mypy strict + pydantic plugin doesn't export it from `pydantic.fields`.
- **AC #26 (Backtest verify)**: Confirmed `backtest.py:186-187` already logs WARNING with `exc_info=True`. No change needed.
- **AC #27-30 (Quality gates)**: `ruff check .` clean (excluding pre-existing notebook issues), `mypy --strict` passes 87 files, `pytest` 883 passed / 1 skipped / 0 failures.

### Senior Developer Review (AI)

**Reviewer:** Code Review Agent (Claude Sonnet 4.6) — 2026-03-03

**Outcome:** APPROVED with fixes applied

**Issues Found:** 1 High, 3 Medium (all fixed), 5 Low (action items below)

**Fixes Applied:**
- **[HIGH] `_fetch_per_team` summary count bug** — Teams returning `None` (empty schedule, no exception) were counted in `failed` (total − success) but not tracked in `failed_teams`. Summary warning message was inconsistent (count > named list). Fixed by adding `else: failed_teams.append(team_name)` branch and computing `failed = len(failed_teams)`.
- **[MEDIUM] Redundant exact-match in `fuzzy_match_team`** — Both callers already do exact-match before calling `fuzzy_match_team`; the O(N) exact scan inside the function was dead code. Removed exact-match loop from `fuzzy_match_team` (now pure fuzzy). Updated docstring. Removed unused `logging` import. All `test_fuzzy.py` tests still pass (exact strings score 100 via fuzzy path).
- **[MEDIUM] Slow retry tests** — `test_retry_succeeds_on_second_attempt` and `test_retry_exhausted_raises` called the live `@retry` decorator with real `wait_exponential(min=2s)`, making each test take 2–10s. Added `patch("time.sleep")` to both tests. Test suite now runs in ~15s instead of ~25s+.
- **[MEDIUM] Weak assertion in `test_connector_continues_after_team_failure`** — `assert isinstance(games, list)` was trivially always true. Changed to `assert len(games) > 0` to verify non-failing teams contributed data.
- **[MEDIUM] `iterrows()` in `_parse_schedule_df`** — Violated project no-iterrows mandate. Converted to `df.itertuples(index=False)`. Updated `_infer_loc()` signature from `pd.Series` to `object` with `hasattr`/`getattr` for named-tuple compatibility. All 883 tests pass.

**Review Follow-ups (AI):**
- [x] [AI-Review][LOW] `test_empty_name` in `test_fuzzy.py` has weak assertion `result is None or isinstance(result, int)` — always true since return type is `int | None`. Define explicit contract: empty string should return `None`. [test_fuzzy.py:63-67] — **FIXED in Pass 2**
- [ ] [AI-Review][LOW] `sync.py` root: `format="%(message)s"` strips log level prefix — WARNING-level ESPN fetch failures appear without "WARNING:" label for CLI users. Consider `"%(levelname)s: %(message)s"` or tiered handler. [sync.py:40]
- [ ] [AI-Review][LOW] `fuzzy_match_team()` docstring says "callers are responsible for exact matching" — verify this contract is documented in the calling sites' docstrings to prevent future regression. [fuzzy.py:12-17]
- [ ] [AI-Review][LOW] `_resolve_team_id` logs WARNING on final no-match but no DEBUG on exact-miss-before-fuzzy — difficult to trace why fuzzy was attempted. [espn.py:77]
- [ ] [AI-Review][LOW] `rapidfuzz = "*"` and `tenacity = "*"` have no upper bound version pins. Consider adding upper bounds once transitive version requirements are confirmed stable.

### Senior Developer Review (AI) — Pass 2

**Reviewer:** Code Review Agent (Claude Sonnet 4.6) — 2026-03-03

**Outcome:** APPROVED with fixes applied

**Issues Found:** 0 Critical, 2 Medium (all fixed), 4 Low (action items below)

**Fixes Applied:**
- **[MEDIUM] `test_empty_name` vacuously true assertion** — `assert result is None or isinstance(result, int)` is always true (return type is `int | None`). Replaced with `assert result is None` — confirmed: `fuzz.token_set_ratio("", any_str) == 0.0 < threshold=80`. [test_fuzzy.py:67]
- **[MEDIUM] Missing regression test for None-return path in `_fetch_per_team`** — The HIGH bug from Pass 1 (teams returning None not counted in `failed_teams`) had no regression test. Added `test_none_return_counted_as_failed` to `TestFetchSummaryLogging` verifying: None return is counted in `failed`, summary is logged at WARNING, message contains "1 failed". 884 passed, 1 skipped.

**Review Follow-ups (AI):**
- [ ] [AI-Review][LOW] `sync.py` root: `format="%(message)s"` strips log level prefix — WARNING-level ESPN fetch failures appear without "WARNING:" label for CLI users. Consider `"%(levelname)s: %(message)s"` or tiered handler. [sync.py:40]
- [ ] [AI-Review][LOW] `fuzzy_match_team()` docstring says "callers are responsible for exact matching" — yet `test_empty_name` exercises the empty-string case silently. Either document empty-string behavior or note callers must guard against it. [fuzzy.py:12-17]
- [ ] [AI-Review][LOW] `_resolve_team_id` logs WARNING on final no-match but no DEBUG on exact-miss-before-fuzzy — difficult to trace why fuzzy was attempted. [espn.py:77]
- [ ] [AI-Review][LOW] `rapidfuzz = "*"` and `tenacity = "*"` have no upper bound version pins. Consider adding upper bounds once transitive version requirements are confirmed stable.

### Change Log

- 2026-03-03: Story 8.3 implemented — ESPN retry logic, fetch summary, date parse logging, Typer decoupling, generalized dedup, centralized fuzzy match, PydanticUndefined fix, rapidfuzz/tenacity deps declared
- 2026-03-03: Code review fixes (Pass 1) — fetch summary count bug (H1), redundant exact-match in fuzzy (M1), slow retry tests mock (M2), weak test assertion (M3), iterrows → itertuples (M5)
- 2026-03-03: Code review fixes (Pass 2) — test_empty_name vacuous assertion (M1), missing None-return regression test (M2); 884 passed

### File List

**New files:**
- `src/ncaa_eval/ingest/fuzzy.py`
- `tests/unit/test_fuzzy.py`

**Modified files:**
- `pyproject.toml` — added `tenacity = "*"` and `rapidfuzz = "*"`
- `poetry.lock` — regenerated
- `src/ncaa_eval/ingest/connectors/espn.py` — tenacity retry, fetch summary, date parse DEBUG log, fuzzy_match_team usage, itertuples migration, _infer_loc signature update, fetch summary count fix
- `src/ncaa_eval/ingest/sync.py` — typer.echo → logger.info, removed typer import, fuzzy_match_team usage
- `src/ncaa_eval/ingest/fuzzy.py` — removed redundant exact-match loop, removed unused logging import, updated docstring
- `src/ncaa_eval/transform/serving.py` — `_deduplicate_2025` → `_deduplicate_espn_overlap`, ESPN-prefix guard
- `src/ncaa_eval/ingest/repository.py` — `PydanticUndefined` sentinel
- `sync.py` (repo root CLI) — added `logging.basicConfig()` for CLI output
- `tests/unit/test_espn_connector.py` — retry and summary logging tests, time.sleep mock, strengthened assertion, None-return regression test
- `tests/unit/test_fuzzy.py` — test_empty_name assertion strengthened
- `tests/integration/test_sync.py` — no-typer-dependency test
- `tests/unit/test_chronological_serving.py` — generalized dedup test for non-2025 seasons
- `_bmad-output/planning-artifacts/template-requirements.md` — two new learnings: time.sleep mock for retry tests, fetch summary count pattern
- `_bmad-output/implementation-artifacts/sprint-status.yaml` — status updated
- `_bmad-output/implementation-artifacts/8-3-fix-data-pipeline-resilience-espn-error-handling.md` — story file updated
