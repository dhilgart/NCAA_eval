# Story 1.5: Configure Testing Framework

Status: review

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want Pytest, Hypothesis, and Mutmut configured with the agreed testing strategy,
so that I can run tests locally and CI enforces the correct checks at each stage.

## Acceptance Criteria

1. **Given** the testing strategy from Story 1.3 is documented, **When** the developer runs `pytest`, **Then** the test suite discovers and runs tests from the defined directory structure (`tests/unit/`, `tests/integration/`, shared `conftest.py`).
2. **And** Hypothesis is available for property-based test generation — validated by at least one passing `@pytest.mark.property` test.
3. **And** Mutmut is configured for mutation testing on designated modules via `[tool.mutmut]` in `pyproject.toml`, runnable with `mutmut run`.
4. **And** test markers distinguish pre-commit tests from PR-time-only tests — `pytest -m smoke` runs only fast tests; `pytest -m property` runs Hypothesis tests (excluded from pre-commit).
5. **And** at least one passing smoke test exists to validate the framework is operational (already satisfied by Story 1.4's `test_imports.py` and `test_package_structure.py`; Story 1.5 must not break these).
6. **And** pytest configuration in `pyproject.toml` defines default options, markers, and test paths (already configured in Story 1.4; Story 1.5 adds only `[tool.mutmut]`).
7. **And** the CI/PR workflow runs the full pytest suite (not just pre-commit hooks) to enforce Tier 2 quality gates.

## Tasks / Subtasks

- [x] Task 1: Create integration test directory structure (AC: 1)
  - [x] 1.1: Create `tests/integration/__init__.py` (empty package file with `from __future__ import annotations` and a module docstring)
  - [x] 1.2: Create `tests/fixtures/` directory with a `.gitkeep` file so the directory is tracked by git

- [x] Task 2: Create shared fixtures in conftest.py (AC: 1, 2)
  - [x] 2.1: Create `tests/conftest.py` with `temp_data_dir` fixture (returns `Path` — Ruff PT022 requires `return` not `yield` when no teardown; type updated accordingly)
  - [x] 2.2: Add `sample_game_records` fixture to `tests/conftest.py` returning representative sample data (list of dicts until Game schema is defined in Story 2.2)
  - [x] 2.3: Ensure all fixtures have full type annotations to satisfy `mypy --strict` — verified: `mypy --strict` passes
  - [x] 2.4: Verify pytest discovers `conftest.py` correctly — run `pytest --collect-only` and confirm fixtures appear — confirmed: 4 tests collected

- [x] Task 3: Configure Mutmut for mutation testing (AC: 3)
  - [x] 3.1: Add `[tool.mutmut]` section to `pyproject.toml` targeting `src/ncaa_eval/evaluation/` (already exists with `__init__.py` — ready for mutation testing)
  - [x] 3.2: Add `.mutmut-cache` to `.gitignore` (mutmut creates this in the project root; not currently excluded)
  - [x] 3.3: Run `mutmut run` to verify configuration is valid — **WINDOWS BLOCKER**: mutmut 3.4.0 imports `resource` (Unix-only stdlib) unconditionally in `__main__.py`; crashes on Windows. Config TOML is syntactically valid (verified via `tomllib`). Will run correctly in CI (Linux). See Debug Log.
  - [x] 3.4: Document the `mutmut results` and `mutmut show <id>` workflow in a `pyproject.toml` comment — done

- [x] Task 4: Write Hypothesis property-based test to validate framework (AC: 2, 4)
  - [x] 4.1: Create `tests/unit/test_framework_validation.py` with at least one `@pytest.mark.property` Hypothesis test demonstrating the framework works
  - [x] 4.2: Verify the test runs correctly with `pytest -m property` — PASSED (1 test, 0.86s)
  - [x] 4.3: Verify `pytest -m smoke` does NOT pick up the property test (confirming marker separation) — confirmed: 3 smoke tests, 1 deselected

- [x] Task 5: Update CI workflow to run full test suite (AC: 7)
  - [x] 5.1: Add `pytest --cov=src/ncaa_eval --cov-report=term-missing` step to `.github/workflows/python-check.yaml` (runs after pre-commit hooks step)
  - [x] 5.2: Verify the CI step runs all tests (not filtered by marker) — step added without marker filter

- [x] Task 6: End-to-end validation (AC: 1–7)
  - [x] 6.1: Run `pytest` — 4 tests collected, all pass (0.84s)
  - [x] 6.2: Run `pytest -m smoke` — 3 smoke tests pass, 1 deselected, total 0.69s (< 10s budget)
  - [x] 6.3: Run `pytest -m property` — Hypothesis test passes (0.86s, multiple examples generated)
  - [x] 6.4: Run `pytest --cov=src/ncaa_eval --cov-report=term-missing` — 4 passed, coverage report generated (17% total; expected at this stage)
  - [x] 6.5: Run `mutmut run` — BLOCKED on Windows (see 3.3 note and Debug Log); config valid for CI

## Dev Notes

### Architecture Compliance

**CRITICAL — Follow these exactly:**

- **This story ACTIVATES the testing framework**, not rebuilds it. Stories 1.1–1.4 already installed all tools and configured `pyproject.toml`. Story 1.5's scope is: directory structure, shared fixtures, Mutmut config, and Hypothesis validation.
- **Do NOT modify the pytest markers section in `pyproject.toml`** — all 8 markers (`smoke`, `slow`, `integration`, `property`, `fuzz`, `performance`, `regression`, `mutation`) are already correctly configured. [Source: pyproject.toml#L106-L115]
- **Do NOT modify the `[tool.pytest.ini_options]` section** — `testpaths`, `addopts`, `minversion`, `norecursedirs` are already set correctly. [Source: pyproject.toml#L90-L115]
- **`mypy --strict` is mandatory** — all fixtures in `conftest.py` must have explicit return type annotations. Missing annotations will BLOCK pre-commit.
- **`from __future__ import annotations`** is required at the top of every Python file (enforced by Ruff isort). [Source: pyproject.toml#L69]
- **Google docstrings** for all public fixtures and test functions. [Source: docs/STYLE_GUIDE.md#Section 1]
- **Pre-commit smoke suite must stay < 10 seconds** — do NOT add any `@pytest.mark.smoke` tests in this story without verifying total pre-commit time stays within budget. [Source: docs/testing/execution.md#Tier 1]

### Technical Requirements

**Primary Deliverables:**

| File | Action | Purpose |
|---|---|---|
| `tests/integration/__init__.py` | **NEW** | Completes test directory structure (AC1) |
| `tests/fixtures/.gitkeep` | **NEW** | Tracks fixtures directory in git |
| `tests/conftest.py` | **NEW** | Shared project-wide fixtures (AC1, AC2) |
| `tests/unit/test_framework_validation.py` | **NEW** | Validates Hypothesis framework (AC2, AC4) |
| `pyproject.toml` | **MODIFY** | Add `[tool.mutmut]` section only (AC3) |
| `.github/workflows/python-check.yaml` | **MODIFY** | Add full pytest + coverage step (AC7) |

**CRITICAL — What NOT to Touch:**
- `tests/unit/test_imports.py` — existing smoke test from Story 1.4; do not modify
- `tests/unit/test_package_structure.py` — existing smoke test from Story 1.4; do not modify
- `tests/test_ncaa_eval.py` — exists but empty; can be left as-is or given a module docstring
- `[tool.pytest.ini_options]` section in `pyproject.toml` — fully configured; no changes needed
- `.pre-commit-config.yaml` — correctly configured for Story 1.4; do not touch

### Library / Framework Requirements

**Tools Already Installed (Story 1.1):**

| Tool | Installed As | Version | Purpose |
|---|---|---|---|
| `pytest` | `poetry.group.dev.dependencies` | `*` (≥8.0 enforced) | Test runner |
| `pytest-cov` | `poetry.group.dev.dependencies` | `*` | Coverage measurement |
| `hypothesis` | `poetry.group.dev.dependencies` | `*` | Property-based testing |
| `mutmut` | `poetry.group.dev.dependencies` | `*` | Mutation testing |

**Hypothesis Usage Patterns:**

```python
from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


@pytest.mark.property
@given(x=st.floats(0.0, 1.0), y=st.floats(0.0, 1.0))
def test_example_property(x: float, y: float) -> None:
    """Example: Hypothesis property test (NOT smoke — runs in PR/CI only)."""
    # Property: min of two values is always <= each value
    assert min(x, y) <= x
    assert min(x, y) <= y
```

**Mutmut `pyproject.toml` Configuration:**

```toml
[tool.mutmut]
paths_to_mutate = ["src/ncaa_eval/evaluation/"]
# Extend to more modules as they are implemented:
# paths_to_mutate = ["src/ncaa_eval/evaluation/", "src/ncaa_eval/model/"]
pytest_add_cli_args_test_selection = ["tests/"]
# Optional: only mutate lines covered by tests (faster runs):
# mutate_only_covered_lines = true
```

**Mutmut workflow commands:**
```bash
# Run mutation testing on Tier 1 modules
poetry run mutmut run

# Review results summary
poetry run mutmut results

# Inspect a specific surviving mutant
poetry run mutmut show <mutant-id>

# Target a specific file (faster for development)
poetry run mutmut run --paths-to-mutate=src/ncaa_eval/evaluation/metrics.py
```

**Conftest Fixture Pattern (mypy --strict compliant):**

```python
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Iterator[Path]:
    """Provide an isolated temporary directory for test data.

    Uses pytest's built-in tmp_path fixture which handles cleanup
    automatically (no manual teardown needed).

    Yields:
        Path: A temporary directory that exists for the duration of the test.
    """
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    yield data_dir
    # tmp_path cleanup is handled by pytest automatically
```

**Note on `tmp_path` vs manual `shutil.rmtree`:** Use pytest's built-in `tmp_path` fixture instead of creating and manually cleaning up a directory. It's simpler and avoids teardown failures on Windows (file locks).

**CI Workflow Pattern (Tier 2):**

```yaml
# Add after existing "Run pre-commit hooks" step:
- name: Run full test suite with coverage
  run: |
    poetry run pytest --cov=src/ncaa_eval --cov-report=term-missing
```

### File Structure Requirements

**Target state after Story 1.5:**

```
tests/
├── __init__.py                      # Existing (Story 1.1)
├── test_ncaa_eval.py                # Existing (empty, Story 1.1) — leave as-is
├── conftest.py                      # NEW (Story 1.5) — shared fixtures
├── unit/                            # Existing (Story 1.4)
│   ├── __init__.py                  # Existing (Story 1.4)
│   ├── test_imports.py              # Existing (Story 1.4) — DO NOT TOUCH
│   ├── test_package_structure.py    # Existing (Story 1.4) — DO NOT TOUCH
│   └── test_framework_validation.py # NEW (Story 1.5) — Hypothesis validation
├── integration/                     # NEW (Story 1.5)
│   └── __init__.py                  # NEW (Story 1.5)
└── fixtures/                        # NEW (Story 1.5) — test data files
    └── .gitkeep                     # NEW (Story 1.5) — tracks directory in git
```

### Testing Requirements

**Validation Tests (What Dev Agent Must Run):**

1. **Full test discovery (AC1):**
   ```bash
   poetry run pytest --collect-only
   ```
   Expected: All test files under `tests/unit/` and `tests/integration/` appear. No collection errors.

2. **Smoke suite timing (AC4, AC5):**
   ```bash
   poetry run pytest -m smoke -v
   ```
   Expected: Only `test_imports.py` and `test_package_structure.py` tests run. Total time < 10 seconds.

3. **Hypothesis property test (AC2):**
   ```bash
   poetry run pytest -m property -v
   ```
   Expected: `test_framework_validation.py` property test runs with Hypothesis generating multiple examples. No errors.

4. **Marker separation confirmed (AC4):**
   ```bash
   poetry run pytest -m smoke -v
   ```
   Expected: `test_framework_validation.py` (marked `property`) does NOT appear in output.

5. **Coverage report (AC7):**
   ```bash
   poetry run pytest --cov=src/ncaa_eval --cov-report=term-missing
   ```
   Expected: Coverage report generated. May show minimal coverage since `src/ncaa_eval/` has minimal code (just `__init__.py`). No errors.

6. **Mutmut validation (AC3):**
   ```bash
   poetry run mutmut run
   ```
   Expected: Mutmut executes without error. May report 0 mutations since `src/ncaa_eval/evaluation/` doesn't exist yet — that's acceptable. The goal is confirming config is syntactically valid.

**`test_framework_validation.py` Spec:**

```python
"""Framework validation tests for Hypothesis property-based testing.

These tests validate that the Hypothesis testing framework is correctly
configured. They do NOT test application logic.
"""
from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


@pytest.mark.property
@given(
    a=st.floats(allow_nan=False, allow_infinity=False, min_value=0.0, max_value=1.0),
    b=st.floats(allow_nan=False, allow_infinity=False, min_value=0.0, max_value=1.0),
)
def test_hypothesis_framework_available(a: float, b: float) -> None:
    """Validate Hypothesis framework is correctly installed and functional.

    Tests a trivial mathematical invariant (min/max relationship) to confirm
    Hypothesis can generate examples and run property-based tests. This test
    does NOT run in pre-commit (no @pytest.mark.smoke) — runs in PR/CI only.

    Args:
        a: A float in [0, 1] generated by Hypothesis.
        b: A float in [0, 1] generated by Hypothesis.
    """
    # Invariant: min is always <= both values, max is always >= both values
    assert min(a, b) <= a
    assert min(a, b) <= b
    assert max(a, b) >= a
    assert max(a, b) >= b
```

### Project Structure Notes

**Alignment with Story 1.1:**
- `tests/__init__.py` exists. The `tests/unit/__init__.py` exists. Story 1.5 adds `tests/integration/__init__.py` to complete the structure. [Source: docs/testing/conventions.md#Directory Structure]

**Alignment with Story 1.3 (Testing Strategy):**
- `docs/testing/conventions.md` explicitly names Story 1.5 as responsible for: `tests/conftest.py`, `tests/unit/` structure, `tests/integration/` structure, and fixtures directory. [Source: docs/testing/conventions.md#L14-L26]
- `docs/testing/execution.md` defines: Tier 1 (smoke), Tier 2 (full suite), Tier 3 (AI review), Tier 4 (human review). Story 1.5 completes Tier 1–2 setup. [Source: docs/testing/execution.md]

**Alignment with Story 1.4 (Toolchain):**
- pytest markers are already in `pyproject.toml` — Story 1.5 must not duplicate them.
- Pre-commit hook (`pytest -m smoke`) already runs smoke tests. Story 1.5 must not break this.
- CI workflow runs `pre-commit run --all-files` — Story 1.5 adds a `pytest` step AFTER this.

**Mutmut targeting strategy:**
- Story 1.5 targets `src/ncaa_eval/evaluation/` which already exists (with `__init__.py`). This is the correct Tier 1 target per `docs/testing/quality.md`.
- When `evaluation/metrics.py` is implemented in Epic 6, mutation testing will automatically find and test it with no config changes needed.
- `.hypothesis/` is already in `.gitignore` (line 55). ✓ No action needed there.
- `.mutmut-cache` is NOT yet in `.gitignore` — Task 3.2 adds it.

**Windows compatibility note:**
- Use pytest's built-in `tmp_path` fixture for temp directories. Manual `shutil.rmtree` in teardown fails intermittently on Windows when file handles aren't released. `tmp_path` handles this gracefully.
- Avoid hardcoded `/` path separators — use `pathlib.Path` for all path manipulation.

### Previous Story Intelligence

#### Story 1.4: Configure Code Quality Toolchain

**Critical learnings that directly impact Story 1.5:**

1. **Mypy local hook scope:** Mypy runs on `src/` AND `tests/` (files: `^(src/|tests/)`). All new test files in `tests/` are type-checked by the pre-commit hook. Missing type annotations will BLOCK commits.
   - [Source: .pre-commit-config.yaml#L67]

2. **`from __future__ import annotations` is mandatory:** Ruff's isort config enforces this as a required import in every Python file. Failing to include it in conftest.py or test_framework_validation.py will fail the pre-commit Ruff hook.
   - [Source: pyproject.toml#L69]

3. **codespell and blacken-docs removed:** These hooks were removed due to false positives and pedagogical doc corruption. Documentation code examples in `.md` files are safe. [Source: 1-4-configure-code-quality-toolchain.md#Human Review Fixes]

4. **Complexity gates are active:** Max 5 args, max 10 complexity, max 12 branches. Fixtures and test helper functions should be simple.

5. **Smoke test budget:** Current smoke suite (3 tests in test_imports.py and test_package_structure.py) must stay < 10 seconds total. Do NOT add `@pytest.mark.smoke` to the Hypothesis test in test_framework_validation.py.

6. **Windows bash context:** Development environment is Windows with bash shell. Use `poetry run pytest` (not raw `pytest`) to ensure the Poetry virtualenv is used.

7. **PLR0913 rule:** Max 5 function arguments applies to fixtures too. Don't create fixtures with more than 5 parameters.

#### Story 1.3: Define Testing Strategy (Reference)

- All documentation for this framework is in `docs/testing/` — the testing strategy itself is NOT being configured here, only the tooling that implements it.
- Fuzz tests (`@pytest.mark.fuzz`) target `ingest/` modules for malformed CSV handling. Story 1.5 does NOT implement fuzz tests — just ensures the framework can run them.
- `@pytest.mark.mutation` tags tests that are specifically good candidates for mutation testing evaluation.

### Git Intelligence

**Recent commit analysis:**

```
b823dc6 feat(toolchain): configure Ruff/Mypy/Pytest pre-commit hooks (#4)
3a2b8fd Merge pull request #3 from dhilgart/story/1-3-define-testing-strategy
e5dde87 docs(template): add PR template enforcement pattern from Story 1.3
f663cfb feat(workflow): update code-review to generate PRs using template format
```

**Patterns established:**
- Conventional commits format: `type(scope): description` (e.g., `feat(testing): configure Hypothesis and Mutmut`)
- One atomic commit per logical unit (not one commit per file)
- Pre-commit auto-fix commits preceded the story commit in Story 1.4 — pre-commit should auto-fix trailing whitespace / EOF issues on first commit

**Files touched in Story 1.4** (to avoid merge conflicts):
- `.pre-commit-config.yaml` — finalized, do NOT touch
- `.github/workflows/python-check.yaml` — Story 1.5 DOES modify this (add pytest step)
- `pyproject.toml` — Story 1.5 adds ONE section only: `[tool.mutmut]`

**Suggested commit message:**
```
feat(testing): configure Hypothesis, Mutmut, and test directory structure
```

### Architecture Analysis for Developer Guardrails

**From Architecture (docs/specs/05-architecture-fullstack.md):**

> **Section 10.2 — Development Workflow:**
> `nox` → Ruff → Mypy → Pytest
> *Note: Run this before every git commit.*

- Pre-commit (Story 1.4): fast smoke tests via `pytest -m smoke`
- CI/PR (Story 1.5): full `pytest` suite + coverage
- Nox orchestration (Story 1.6): will wrap the above into sessions

> **Section 12 — Coding Standards:**
> `mypy --strict` compliance mandatory.

All new files must pass mypy --strict. The pre-commit hook will catch this automatically.

**Testing strategy hierarchy for this story:**
- Tier 1 (pre-commit): `pytest -m smoke` — smoke tests only (< 10s) — NOT changed by Story 1.5
- Tier 2 (PR/CI): `pytest --cov=src/ncaa_eval --cov-report=term-missing` — ADDED by Story 1.5
- Tier 2 (PR/CI selective): `mutmut run` — CONFIGURED by Story 1.5 (not run in CI yet — too slow; configured for local use)

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 1.5] — Story acceptance criteria
- [Source: pyproject.toml#L90-L115] — Existing pytest configuration (markers, testpaths, addopts)
- [Source: pyproject.toml#L20-L34] — Dev dependencies (pytest, hypothesis, mutmut, pytest-cov all installed)
- [Source: .pre-commit-config.yaml] — Existing pre-commit hooks (do not modify)
- [Source: .github/workflows/python-check.yaml] — CI workflow (add pytest step)
- [Source: docs/testing/conventions.md#Directory Structure] — Target directory structure
- [Source: docs/testing/conventions.md#Fixture Conventions] — Fixture type annotation requirements
- [Source: docs/testing/conventions.md#Marker Definitions] — All 8 marker definitions
- [Source: docs/testing/execution.md#Tier 1] — Smoke test budget (< 10s total)
- [Source: docs/testing/execution.md#Tier 2] — PR/CI full suite requirements
- [Source: docs/testing/quality.md#Mutation Testing] — Mutmut Tier 1 targets (evaluation/metrics.py)
- [Source: _bmad-output/implementation-artifacts/1-4-configure-code-quality-toolchain.md] — Mypy scope, complexity gates, PLR0913 constraints

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-5-20250929

### Debug Log References

**[2026-02-17] Mutmut 3.4.0 Windows Incompatibility (Task 3.3)**
- mutmut 3.4.0 unconditionally imports `resource` (Unix-only stdlib) in `mutmut/__main__.py` line 9
- `ModuleNotFoundError: No module named 'resource'` on Windows; any `mutmut` invocation fails at import
- Config TOML is syntactically valid (verified via `python -c "import tomllib; ..."`)
- **Original resolution**: mutmut will execute correctly in CI (Linux/ubuntu-latest). Local Windows development requires WSL.
- **Recommendation for human reviewer**: Accept CI-only limitation, or pin `mutmut = "^2.5"` (requires different config format — setup.cfg style instead of TOML).

**[2026-02-17] WSL Setup — Mutmut 3.4.0 Verified Locally (Task 3.3 / Task 6.5 re-verification)**
- Developer WSL environment configured: Linux 6.6.87.2-microsoft-standard-WSL2
- Conda env `ncaa_eval` created with Python 3.12.12 (matches `python = ">=3.12,<4.0"` constraint)
- Poetry 2.3.2 + all dev deps (mutmut 3.4.0, pytest 9.0.2, hypothesis 6.151.6, etc.) installed via `pip install poetry && POETRY_VIRTUALENVS_CREATE=false poetry install --with dev`
- `mutmut run` runs successfully in WSL — no `resource` module error ✓
- Mutmut generates mutants, runs stats (3 passed, 1 deselected), and completes without import errors
- **New issue discovered**: `test_src_directory_structure` fails under mutmut because it uses `Path(__file__).parent.parent.parent` to compute project root. When mutmut runs from `mutants/` directory, this path resolves to `mutants/` not the project root; `mutants/src/ncaa_eval/__init__.py` doesn't exist since mutmut only copies mutation targets.
- **Fix applied**: Updated `[tool.mutmut]` `pytest_add_cli_args_test_selection` to add `-k "not test_src_directory_structure"`. Structural smoke tests check project layout, not evaluation module logic — exclusion is semantically correct.
- **Additional fix**: Added `mutants/` to `.gitignore` (mutmut 3.x creates this directory; was not excluded)
- "Stopping early, could not find any test case for any mutant" — expected: `evaluation/__init__.py` has no testable logic yet. Will work when `evaluation/metrics.py` is implemented in Epic 6.
- Full test suite: 4 passed, 0.53s ✓ | `pytest -m smoke`: 3 passed, 0.18s ✓ | `pytest -m property`: 1 passed ✓ | ruff: all checks passed ✓ | mypy --strict: no issues ✓
- **pytest-cov was missing** from installed packages despite being in `pyproject.toml`. Root cause: `poetry.lock` was out of date. Fixed with `poetry lock --no-update && poetry install --with dev`. pytest-cov 7.0.0 installed.

**[2026-02-17] Ruff PT022 — temp_data_dir fixture `yield` → `return` (Task 2.1)**
- Story Dev Notes specify `-> Iterator[Path]:` with `yield`. Ruff PT022 requires `return` when no teardown exists.
- Applied Ruff auto-fix: changed `yield` → `return`, updated return type `Iterator[Path]` → `Path`, removed now-unused `Iterator` import.
- mypy --strict passes with the corrected signature. Fixture behavior is identical (pytest handles cleanup via `tmp_path`).

### Completion Notes List

- Created `tests/integration/__init__.py` completing the test directory hierarchy from conventions.md
- Created `tests/fixtures/.gitkeep` to track fixtures directory in git
- Created `tests/conftest.py` with `temp_data_dir` (returns `Path`) and `sample_game_records` (returns `list[dict[str, object]]`) fixtures; mypy --strict + Ruff clean
- Created `tests/unit/test_framework_validation.py` with `@pytest.mark.property` Hypothesis test; passes in 0.86s; correctly excluded from `pytest -m smoke`
- Added `[tool.mutmut]` section to `pyproject.toml` targeting `src/ncaa_eval/evaluation/`; includes workflow command documentation
- Added `.mutmut-cache` to `.gitignore`
- Added full pytest + coverage step to `.github/workflows/python-check.yaml`
- Installed `pytest-cov 7.0.0` (was missing from environment; lock file was stale)
- All 4 tests pass: 0.84s full run, 0.69s smoke run — both within budget
- AC1 ✅ AC2 ✅ AC3 ✅ (config valid; CI-only due to Windows mutmut bug) AC4 ✅ AC5 ✅ AC6 ✅ AC7 ✅

**[2026-02-17] WSL Re-verification:**
- Mutmut 3.4.0 verified working locally via WSL (conda env `ncaa_eval`, Python 3.12.12)
- Fixed `[tool.mutmut]` to exclude `test_src_directory_structure` (path-resolution incompatible with mutmut's `mutants/` runner directory)
- Added `mutants/` to `.gitignore`
- AC3 ✅ (now verified locally via WSL, not CI-only) — All ACs remain satisfied

### Change Log

- 2026-02-17: Implemented Story 1.5 — configure testing framework (Hypothesis, Mutmut, conftest, CI coverage step)
- 2026-02-17: Re-verified with WSL — mutmut 3.4.0 runs locally; fixed `[tool.mutmut]` test exclusion and added `mutants/` to `.gitignore`

### File List

- `tests/integration/__init__.py` — NEW
- `tests/fixtures/.gitkeep` — NEW
- `tests/conftest.py` — NEW
- `tests/unit/test_framework_validation.py` — NEW
- `pyproject.toml` — MODIFIED (added `[tool.mutmut]` section, updated poetry.lock indirectly via `poetry lock --no-update`)
- `.gitignore` — MODIFIED (added `.mutmut-cache`)
- `.github/workflows/python-check.yaml` — MODIFIED (added full pytest + coverage step)
- `poetry.lock` — MODIFIED (updated to install pytest-cov 7.0.0)
- `pyproject.toml` — MODIFIED again (updated `[tool.mutmut]` `pytest_add_cli_args_test_selection` to exclude path-dependent test)
- `.gitignore` — MODIFIED again (added `mutants/` — mutmut 3.x runner directory)
