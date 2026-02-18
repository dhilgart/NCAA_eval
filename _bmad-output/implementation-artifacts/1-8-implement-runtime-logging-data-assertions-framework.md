# Story 1.8: Implement Runtime Logging & Data Assertions Framework

Status: ready-for-dev

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want a structured logging system with configurable verbosity levels and a data assertions framework,
so that I can diagnose runtime issues efficiently and validate data integrity throughout the pipeline.

## Acceptance Criteria

1. A structured logging system is available using Python's `logging` module with project-specific configuration.
2. Custom verbosity levels are supported (QUIET, NORMAL, VERBOSE, DEBUG) controllable via CLI flag or environment variable.
3. Log output includes timestamps, module names, and configurable formatting.
4. A data assertions module provides helper functions for validating DataFrame shapes, column types, value ranges, and null checks.
5. Assertion failures produce clear error messages with the specific validation that failed and the actual vs. expected values.
6. The logging and assertions framework is covered by unit tests.
7. Usage examples are documented in the module docstrings.

## Tasks / Subtasks

- [ ] Task 1: Implement structured logging module (AC: 1, 2, 3, 7)
  - [ ] 1.1: Create `src/ncaa_eval/utils/logger.py` with `configure_logging()` and `get_logger()` public API
  - [ ] 1.2: Define verbosity levels: QUIET->WARNING(30), NORMAL->INFO(20), VERBOSE->custom(15), DEBUG->DEBUG(10)
  - [ ] 1.3: Register custom VERBOSE level via `logging.addLevelName(15, "VERBOSE")`
  - [ ] 1.4: Implement `NCAA_EVAL_LOG_LEVEL` env var reader with NORMAL default
  - [ ] 1.5: Configure log format: `%(asctime)s | %(name)s | %(levelname)-8s | %(message)s`
  - [ ] 1.6: Add module docstring with usage examples per AC 7
  - [ ] 1.7: Export `configure_logging`, `get_logger`, and level constants from `utils/__init__.py`

- [ ] Task 2: Implement data assertions module (AC: 4, 5, 7)
  - [ ] 2.1: Create `src/ncaa_eval/utils/assertions.py`
  - [ ] 2.2: Implement `assert_shape(df, expected_rows, expected_cols)` — validates DataFrame dimensions
  - [ ] 2.3: Implement `assert_columns(df, required)` — validates required column names exist
  - [ ] 2.4: Implement `assert_dtypes(df, expected)` — validates column dtype mapping
  - [ ] 2.5: Implement `assert_no_nulls(df, columns=None)` — validates no null values in specified or all columns
  - [ ] 2.6: Implement `assert_value_range(df, column, min_val, max_val)` — validates values within bounds
  - [ ] 2.7: All failures raise `ValueError` with message format: `"<function> failed: expected <X>, got <Y>"`
  - [ ] 2.8: Add module docstring with usage examples per AC 7
  - [ ] 2.9: Export assertion functions from `utils/__init__.py`

- [ ] Task 3: Write unit tests for logging module (AC: 6)
  - [ ] 3.1: Create `tests/unit/test_logger.py`
  - [ ] 3.2: Test `configure_logging()` applies correct level for each verbosity (QUIET, NORMAL, VERBOSE, DEBUG)
  - [ ] 3.3: Test `NCAA_EVAL_LOG_LEVEL` env var override using `monkeypatch`
  - [ ] 3.4: Test log output format includes timestamp, module name, and level (use `caplog`)
  - [ ] 3.5: Test `get_logger(name)` returns logger under `ncaa_eval` hierarchy
  - [ ] 3.6: Test default level is NORMAL when env var is unset

- [ ] Task 4: Write unit tests for assertions module (AC: 6)
  - [ ] 4.1: Create `tests/unit/test_assertions.py`
  - [ ] 4.2: Test each assertion passes with valid DataFrame data
  - [ ] 4.3: Test each assertion raises `ValueError` with invalid data
  - [ ] 4.4: Test error messages contain the specific validation name, expected value, and actual value
  - [ ] 4.5: Test `assert_no_nulls` with both specific-columns and all-columns modes
  - [ ] 4.6: Test `assert_value_range` with boundary values (min, max, out-of-range)

- [ ] Task 5: End-to-end validation (all ACs)
  - [ ] 5.1: `ruff check src/ncaa_eval/utils/ tests/unit/test_logger.py tests/unit/test_assertions.py`
  - [ ] 5.2: `mypy --strict src/ncaa_eval tests`
  - [ ] 5.3: `nox` (full pipeline: lint -> typecheck -> tests) — all pass, zero regressions
  - [ ] 5.4: Verify module docstrings contain usage examples

## Dev Notes

### Architecture Compliance

**CRITICAL — Follow these exactly:**

- **Package location:** `src/ncaa_eval/utils/` — per Architecture Section 9, `utils/` is the "Shared utilities" subpackage. [Source: specs/05-architecture-fullstack.md#Section 9]
- **`from __future__ import annotations`** required at the top of every new Python file. [Source: pyproject.toml ruff isort required-imports]
- **`mypy --strict` is mandatory** — all new files in `src/ncaa_eval/` and `tests/` must type-check cleanly. [Source: pyproject.toml#L40-L46]
- **Google-style docstrings** — enforced by Ruff `pydocstyle.convention = "google"`. Use single backtick `` `code` `` in docstrings, not RST double backtick. [Source: docs/STYLE_GUIDE.md]
- **Do NOT shadow stdlib modules** — name the logging module `logger.py` (NOT `logging.py`), which would shadow Python's `logging` and break all imports.
- **Do NOT use `for` loops over DataFrames** in assertion implementations — use vectorized pandas operations (`.isnull().any()`, `.isin()`, etc.). [Source: specs/05-architecture-fullstack.md#Section 12]
- **No third-party logging libraries** — use Python's standard `logging` module only. The AC explicitly specifies this. Do NOT add structlog, loguru, or similar.
- **Do NOT modify `pyproject.toml`** — no new dependencies or tool config changes needed.
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

#### Data Assertions Module (`src/ncaa_eval/utils/assertions.py`)

**Public API signatures:**
```python
def assert_shape(
    df: pd.DataFrame,
    expected_rows: int | None = None,
    expected_cols: int | None = None,
) -> None: ...

def assert_columns(df: pd.DataFrame, required: Sequence[str]) -> None: ...

def assert_dtypes(df: pd.DataFrame, expected: Mapping[str, str | type]) -> None: ...

def assert_no_nulls(
    df: pd.DataFrame, columns: Sequence[str] | None = None,
) -> None: ...

def assert_value_range(
    df: pd.DataFrame,
    column: str,
    *,
    min_val: float | None = None,
    max_val: float | None = None,
) -> None: ...
```

**Import types from `collections.abc`:**
```python
from collections.abc import Mapping, Sequence
```

**Error message examples:**
```
assert_shape failed: expected (100, 5), got (98, 5)
assert_columns failed: missing columns {'TeamID', 'Score'}
assert_dtypes failed for column 'Score': expected int64, got float64
assert_no_nulls failed: null values found in columns ['TeamName'] (3 nulls)
assert_value_range failed for column 'Score': 2 values outside range [0, 200], min=-3, max=250
```

All functions raise `ValueError` (NOT `AssertionError`). `assert` statements can be disabled with `python -O`, silently skipping validations. `ValueError` is unconditional and the correct choice for data validation.

**Type annotations with pandas:** With `follow_imports = "silent"` in mypy config, `pd.DataFrame` resolves silently. Use `pd.DataFrame` directly — do NOT install `pandas-stubs`.

### Library / Framework Requirements

| Library | Status | Purpose |
|---|---|---|
| `logging` (stdlib) | Built-in | Structured logging with levels, formatters, handlers |
| `os` (stdlib) | Built-in | Read `NCAA_EVAL_LOG_LEVEL` environment variable |
| `pandas` | Already installed | DataFrame type used in assertions API |

**No new dependencies needed.** All are either stdlib or already in `pyproject.toml`.

### File Structure

**New files to create:**

```
src/ncaa_eval/utils/
├── __init__.py                 # MODIFIED — export public API
├── logger.py                   # NEW — structured logging with verbosity levels
└── assertions.py               # NEW — DataFrame validation helpers

tests/unit/
├── test_logger.py              # NEW — logging module tests
└── test_assertions.py          # NEW — assertions module tests
```

**Files NOT to touch:**
- `pyproject.toml` — no new deps or config changes
- `noxfile.py` — no new sessions
- `.pre-commit-config.yaml` — already finalized
- Any file outside `src/ncaa_eval/utils/` and `tests/unit/`

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

**Mutation testing note:** Current `[tool.mutmut]` targets `src/ncaa_eval/evaluation/` only. The assertions module has clear logic branches ideal for mutation testing — consider adding `src/ncaa_eval/utils/` to `paths_to_mutate` in a future story.

**Validation commands (all must pass):**
```bash
POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval ruff check src/ncaa_eval/utils/ tests/unit/test_logger.py tests/unit/test_assertions.py
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

{{agent_model_name_version}}

### Debug Log References

### Completion Notes List

### File List
