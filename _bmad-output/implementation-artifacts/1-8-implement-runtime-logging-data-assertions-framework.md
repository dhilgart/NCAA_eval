# Story 1.8: Implement Runtime Logging & Data Assertions Framework

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want a structured logging system with configurable verbosity levels and a data assertions framework,
so that I can diagnose runtime issues efficiently and validate data integrity throughout the pipeline.

## Acceptance Criteria

1. A structured logging system is available using Python's `logging` module with project-specific configuration.
2. Custom verbosity levels are supported (QUIET, NORMAL, VERBOSE, DEBUG) controllable via CLI flag or environment variable.
3. Log output includes timestamps, module names, and configurable formatting.
4. Pandera is added as a production dependency and documented as the project's DataFrame validation library, replacing custom assertion helpers.
5. Pandera usage patterns (schema definition, `SchemaError` handling, `strict=False` for subset validation) are documented in `template-requirements.md` for downstream story adoption.
6. The logging framework is covered by unit tests with 100% branch coverage.
7. Usage examples are documented in the logging module docstrings.

## Tasks / Subtasks

- [x] Task 1: Implement structured logging module (AC: 1, 2, 3, 7)
  - [x] 1.1: Create `src/ncaa_eval/utils/logger.py` with `configure_logging()` and `get_logger()` public API
  - [x] 1.2: Define verbosity levels: QUIET->WARNING(30), NORMAL->INFO(20), VERBOSE->custom(15), DEBUG->DEBUG(10)
  - [x] 1.3: Register custom VERBOSE level via `logging.addLevelName(15, "VERBOSE")`
  - [x] 1.4: Implement `NCAA_EVAL_LOG_LEVEL` env var reader with NORMAL default
  - [x] 1.5: Configure log format: `%(asctime)s | %(name)s | %(levelname)-8s | %(message)s`
  - [x] 1.6: Add module docstring with usage examples per AC 7
  - [x] 1.7: Export `configure_logging`, `get_logger`, and level constants from `utils/__init__.py`

- [x] Task 2: Adopt Pandera as DataFrame validation library (AC: 4, 5)
  - [x] 2.1: Add `pandera` to production dependencies in `pyproject.toml`
  - [x] 2.2: Add `pandera` to edgetest upgrade list in `pyproject.toml`
  - [x] 2.3: Run `poetry lock --no-update && poetry install` to update lock file
  - [x] 2.4: Document Pandera usage patterns in `template-requirements.md` (import conventions, `strict=False`, `SchemaError` handling)
  - [x] 2.5: Remove custom `assertions.py` wrapper in favour of native Pandera API (initially implemented, then replaced per library-first rule)

- [x] Task 3: Write unit tests for logging module (AC: 6)
  - [x] 3.1: Create `tests/unit/test_logger.py`
  - [x] 3.2: Test `configure_logging()` applies correct level for each verbosity (QUIET, NORMAL, VERBOSE, DEBUG)
  - [x] 3.3: Test `NCAA_EVAL_LOG_LEVEL` env var override using `monkeypatch`
  - [x] 3.4: Test log output format includes timestamp, module name, and level (use `caplog`)
  - [x] 3.5: Test `get_logger(name)` returns logger under `ncaa_eval` hierarchy
  - [x] 3.6: Test default level is NORMAL when env var is unset

- [x] Task 4: End-to-end validation (all ACs)
  - [x] 4.1: `ruff check src/ncaa_eval/utils/ tests/unit/test_logger.py` — all checks passed
  - [x] 4.2: `mypy --strict src/ncaa_eval tests` — no issues found
  - [x] 4.3: `pytest tests/` — 25 tests pass, zero regressions
  - [x] 4.4: Verify logger module docstring contains usage examples

## Dev Notes

### Architecture Compliance

**CRITICAL — Follow these exactly:**

- **Package location:** `src/ncaa_eval/utils/` — per Architecture Section 9, `utils/` is the "Shared utilities" subpackage. [Source: specs/05-architecture-fullstack.md#Section 9]
- **`from __future__ import annotations`** required at the top of every new Python file. [Source: pyproject.toml ruff isort required-imports]
- **`mypy --strict` is mandatory** — all new files in `src/ncaa_eval/` and `tests/` must type-check cleanly. [Source: pyproject.toml#L40-L46]
- **Google-style docstrings** — enforced by Ruff `pydocstyle.convention = "google"`. Use single backtick `` `code` `` in docstrings, not RST double backtick. [Source: docs/STYLE_GUIDE.md]
- **Do NOT shadow stdlib modules** — name the logging module `logger.py` (NOT `logging.py`), which would shadow Python's `logging` and break all imports.
- **No third-party logging libraries** — use Python's standard `logging` module only. The AC explicitly specifies this. Do NOT add structlog, loguru, or similar.
- **Pandera added to `pyproject.toml`** — library-first rule applied: Pandera replaces custom assertion wrappers for DataFrame validation. Added as production dependency + edgetest entry.
- **Do NOT modify `noxfile.py`** — no new Nox sessions needed for this story.

### Technical Implementation Guide

#### Logging Module (`src/ncaa_eval/utils/logger.py`)

**Verbosity Level Mapping:**

| Project Level | Python Level | Value | Use Case |
|---|---|---|---|
| QUIET | WARNING | 30 | Suppress routine output; only warnings and errors |
| NORMAL | INFO | 20 | Standard operational output (default) |
| VERBOSE | VERBOSE (custom) | 15 | Detailed operational info; more than INFO, less than DEBUG |
| DEBUG | DEBUG | 10 | Full diagnostic output for troubleshooting |

**Custom VERBOSE level registration:**
```python
VERBOSE = 15
logging.addLevelName(VERBOSE, "VERBOSE")
```

**Public API:**
```python
def configure_logging(level: str = "NORMAL") -> None:
    """Configure project-wide logging with the given verbosity level.

    Example:
        >>> from ncaa_eval.utils.logger import configure_logging, get_logger
        >>> configure_logging("VERBOSE")
        >>> log = get_logger("ingest")
        >>> log.info("Loading data...")
    """

def get_logger(name: str) -> logging.Logger:
    """Return a logger under the `ncaa_eval` hierarchy.

    Example:
        >>> log = get_logger("transform.features")
        >>> log.info("Computing features for season %d", 2025)
    """
```

**Environment variable:** `NCAA_EVAL_LOG_LEVEL` — read at configure time. Valid values: `QUIET`, `NORMAL`, `VERBOSE`, `DEBUG` (case-insensitive). Fallback: `NORMAL`.

**Precedence:** explicit `level` parameter > env var > default (`NORMAL`).

**Log format:** `%(asctime)s | %(name)s | %(levelname)-8s | %(message)s`

**Logger hierarchy:** Root logger is `ncaa_eval`. `get_logger("ingest.kaggle")` returns `ncaa_eval.ingest.kaggle`.

**`configure_logging()` implementation pattern:**
1. Get the `ncaa_eval` root logger
2. Set its level based on verbosity
3. Remove existing handlers (prevent duplicates on re-call)
4. Add a `StreamHandler` to stderr with the configured formatter
5. Set `propagate = False` to prevent double-logging with Python root logger

#### Data Assertions: Pandera (native)

Custom assertion helpers (`assertions.py`) were initially implemented, then replaced with Pandera per the library-first rule. Downstream stories will define `pa.DataFrameSchema` instances directly using Pandera's native API. See `template-requirements.md` for documented usage patterns (import conventions, `strict=False`, `SchemaError` handling).

### Library / Framework Requirements

| Library | Status | Purpose |
|---|---|---|
| `logging` (stdlib) | Built-in | Structured logging with levels, formatters, handlers |
| `os` (stdlib) | Built-in | Read `NCAA_EVAL_LOG_LEVEL` environment variable |
| `pandera` | **NEW** — added to pyproject.toml | DataFrame validation (replaces custom assertions) |

### File Structure

**Files changed:**

```
src/ncaa_eval/utils/
├── __init__.py                 # MODIFIED — export logging public API
└── logger.py                   # NEW — structured logging with verbosity levels

tests/unit/
└── test_logger.py              # NEW — logging module tests (21 tests, 100% coverage)

pyproject.toml                  # MODIFIED — added pandera dependency + edgetest entry
poetry.lock                     # MODIFIED — updated with pandera + transitive deps
```

### Testing Requirements

**Test framework:** pytest with existing markers and conventions.

**Marker assignments:**
- Most tests -> `@pytest.mark.smoke` (fast, pure Python, no I/O)
- Property-based tests (if any) -> `@pytest.mark.property`

**Fixture patterns:**
- Use `return` not `yield` (Ruff PT022) unless teardown is needed
- Use `monkeypatch` for env var tests (no manual cleanup needed)
- Use `caplog` fixture for capturing log output in tests
- Create small test DataFrames inline — no fixture files needed

**Mutation testing note:** Current `[tool.mutmut]` targets `src/ncaa_eval/evaluation/` only. Consider adding `src/ncaa_eval/utils/` to `paths_to_mutate` in a future story (logger.py has branch logic in level resolution).

**Validation commands (all must pass):**
```bash
POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval ruff check src/ncaa_eval/utils/ tests/unit/test_logger.py
POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval mypy --strict src/ncaa_eval tests
POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval nox
```

### Previous Story Intelligence (Story 1.7)

**Patterns to follow:**
1. **`from __future__ import annotations`** at top of every file — Ruff auto-fixes but include it from the start.
2. **Google-style docstrings with single backticks** — not RST double backticks.
3. **`python=False` for Nox sessions** — but this story adds no new Nox sessions.
4. **Conventional commits:** `feat(utils): implement structured logging and data assertions framework`
5. **Story branch:** `story/1-8-implement-runtime-logging-data-assertions-framework`
6. **`docs/conf.py` is NOT in mypy scope** — but all files in `src/ncaa_eval/` and `tests/` are.

**Story 1.7 established:**
- Sphinx docs working (`nox -s docs`)
- All quality gates passing (lint, typecheck, tests)
- Commitizen validates commit format on every commit

### Git Intelligence

**Recent commits:**
```
7d74df9 Restructure docs/ as pure Sphinx source directory (#8)
dbac187 feat(toolchain): configure versioning, packaging & documentation (#7)
407cda0 Configure Nox session management & automation (#6)
3c07873 feat(testing): configure Hypothesis, Mutmut, and test framework (#5)
b823dc6 feat(toolchain): configure Ruff/Mypy/Pytest pre-commit hooks (#4)
```

**Patterns established:**
- Conventional commits enforced by commitizen pre-commit hook
- PR merges use squash-merge with conventional commit title
- Story branches: `story/{story-key}`

**Warning:** The commitizen pre-commit hook validates every commit message. Use conventional commit format: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `style`.

### Project Structure Notes

- `src/ncaa_eval/utils/__init__.py` already exists (created in Story 1.1) — currently contains only `from __future__ import annotations` and a module docstring
- `tests/unit/` directory already exists with `test_framework_validation.py`, `test_imports.py`, `test_package_structure.py`
- New modules integrate into the existing package structure without conflicts

### References

- [Source: specs/05-architecture-fullstack.md#Section 9] — Project structure: utils/ is "Shared utilities"
- [Source: specs/05-architecture-fullstack.md#Section 12] — Coding standards: mypy --strict, vectorization-first
- [Source: specs/03-prd.md#NFR5] — "deep logging, error traces, and data assertions to facilitate debugging"
- [Source: _bmad-output/planning-artifacts/epics.md#Story 1.8] — Acceptance criteria
- [Source: _bmad-output/implementation-artifacts/1-7-configure-versioning-packaging-documentation.md] — Previous story patterns
- [Source: pyproject.toml] — Tool configuration (ruff, mypy, pytest markers)
- [Source: _bmad-output/planning-artifacts/template-requirements.md] — Fixture patterns, test organization

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

- mypy `[import-untyped]` error for pandas despite `follow_imports = "silent"` — resolved with targeted `# type: ignore[import-untyped]` on pandas import lines (story Dev Notes incorrectly stated `follow_imports = "silent"` would suppress this under `--strict`)
- ruff format needed after initial file creation — ruff auto-formatter adjusted line lengths and import ordering
- Library-first pivot: custom `assertions.py` (5 functions, 34 tests) was implemented then replaced by Pandera dependency — wrapper removed entirely in favour of native Pandera API; see commits 957df6c and 802852d

### Completion Notes List

- ✅ Implemented `src/ncaa_eval/utils/logger.py` with `configure_logging()` and `get_logger()` public API, 4 verbosity levels (QUIET/NORMAL/VERBOSE/DEBUG), custom VERBOSE level registration, `NCAA_EVAL_LOG_LEVEL` env var support with precedence chain, and `StreamHandler` to stderr with pipe-delimited format
- ✅ Updated `src/ncaa_eval/utils/__init__.py` to export logging public API symbols
- ✅ 21 logging tests (100% branch coverage) covering all verbosity levels, env var override, level precedence, handler configuration, format validation, logger hierarchy
- ✅ Added Pandera as production dependency for DataFrame validation (library-first rule: replaces custom `assertions.py` wrapper that was initially implemented then removed)
- ✅ Documented Pandera usage patterns in `template-requirements.md` for downstream story adoption
- ✅ All 25 tests pass, ruff clean, mypy strict clean, zero regressions
- ✅ Logger module docstring contains comprehensive usage examples per AC 7

### File List

- `src/ncaa_eval/utils/__init__.py` — MODIFIED: export logging public API
- `src/ncaa_eval/utils/logger.py` — NEW: structured logging with verbosity levels
- `tests/unit/test_logger.py` — NEW: 21 unit tests (100% branch coverage); autouse fixture, capsys stderr capture, timestamp regex assertions
- `pyproject.toml` — MODIFIED: added `pandera = "*"` to dependencies + edgetest upgrade list
- `poetry.lock` — MODIFIED: updated with pandera + transitive dependencies
- `_bmad-output/planning-artifacts/template-requirements.md` — MODIFIED: documented Pandera usage patterns and library-first rule
- `_bmad-output/implementation-artifacts/sprint-status.yaml` — MODIFIED: story status updated
- `_bmad-output/implementation-artifacts/1-8-implement-runtime-logging-data-assertions-framework.md` — MODIFIED: story file updated

## Change Log

- 2026-02-18: Implemented structured logging module with 4 verbosity levels (QUIET/NORMAL/VERBOSE/DEBUG), env var control, and configurable formatting
- 2026-02-18: Initially implemented custom `assertions.py` with 5 DataFrame validation functions; code review hardened with column-existence guards and additional tests
- 2026-02-18: Replaced custom assertions with Pandera (library-first rule); removed `assertions.py` wrapper and `test_assertions.py` in favour of native Pandera API for downstream stories
- 2026-02-18: Code review (round 2) — fixed 9 issues (3C/2H/3M/1L): revised ACs 4–7 to reflect Pandera adoption; corrected stale File List, Completion Notes, and test counts; updated Dev Notes to remove obsolete constraints; documented pyproject.toml and poetry.lock changes; total: 25 tests, all passing, 100% logger coverage
