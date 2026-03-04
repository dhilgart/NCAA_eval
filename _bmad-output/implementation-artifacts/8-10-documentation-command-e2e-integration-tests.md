# Story 8.10: Documentation Command E2E Integration Tests

Status: ready-for-dev

## Story

As a developer,
I want E2E integration tests that validate every documented toolchain command actually works,
so that documentation rot is caught automatically and users following the guides encounter zero broken commands.

## Acceptance Criteria

### E2E Test Suite Creation

1. New test file `tests/integration/test_documented_commands.py` created
2. E2E test validates `pytest -m smoke` completes successfully and finishes in under 10 seconds
3. E2E test validates `pytest` (full suite) completes with exit code 0
4. E2E test validates `pytest --cov=src/ncaa_eval --cov-report=term-missing` produces coverage output
5. E2E test validates `ruff check .` exits with code 0
6. E2E test validates `ruff format --check .` exits with code 0
7. E2E test validates `mypy --strict src/ncaa_eval tests` exits with code 0
8. E2E test validates each `nox` session individually: `nox -s lint`, `nox -s typecheck`, `nox -s tests`
9. E2E test validates `python -m ncaa_eval.cli --help` prints help text (CLI is importable and functional)
10. E2E test validates `python -m ncaa_eval.cli train --help` prints train help
11. E2E test validates `check-manifest` runs (or the check is removed from documentation if not configured)
12. All E2E tests marked with `@pytest.mark.integration` and `@pytest.mark.slow`

### Documentation Fixes to Match Reality

13. Any documented command that does NOT work is either: (a) fixed so it works, or (b) removed/corrected in documentation
14. `docs/tutorials/getting-started.md` commands validated: each command runs and produces output matching the documented expected output (update expected output if different)

## Tasks / Subtasks

- [ ] Task 1: Create `tests/integration/test_documented_commands.py` with test scaffolding (AC: #1, #12)
  - [ ] 1.1 Create file with module docstring, imports (`subprocess`, `pytest`), and markers
  - [ ] 1.2 Define helper function `_run(cmd: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]` for subprocess invocation with timeout
- [ ] Task 2: Implement quality-gate command tests (AC: #2–#8)
  - [ ] 2.1 `test_pytest_smoke` — `pytest -m smoke` exit 0, under 10s wall-clock
  - [ ] 2.2 `test_pytest_full_suite` — `pytest` exit 0
  - [ ] 2.3 `test_pytest_coverage` — `pytest --cov=src/ncaa_eval --cov-report=term-missing` exit 0, stdout contains "TOTAL"
  - [ ] 2.4 `test_ruff_check` — `ruff check .` exit 0
  - [ ] 2.5 `test_ruff_format_check` — `ruff format --check .` exit 0
  - [ ] 2.6 `test_mypy_strict` — `mypy --strict src/ncaa_eval tests` exit 0
  - [ ] 2.7 `test_nox_lint` — `nox -s lint` exit 0
  - [ ] 2.8 `test_nox_typecheck` — `nox -s typecheck` exit 0
  - [ ] 2.9 `test_nox_tests` — `nox -s tests` exit 0
- [ ] Task 3: Implement CLI help tests (AC: #9–#10)
  - [ ] 3.1 `test_cli_help` — `python -m ncaa_eval.cli --help` exit 0, stdout contains "Usage"
  - [ ] 3.2 `test_cli_train_help` — `python -m ncaa_eval.cli train --help` exit 0, stdout contains "--model"
- [ ] Task 4: Implement check-manifest test (AC: #11)
  - [ ] 4.1 `test_check_manifest` — `check-manifest` exit 0 OR document/fix the gap
- [ ] Task 5: Validate documented commands against reality (AC: #13–#14)
  - [ ] 5.1 Test `ncaa-eval sync --help` — if no `[tool.poetry.scripts]` entry exists, either create it or fix the tutorial to use `python sync.py`
  - [ ] 5.2 Review all code blocks in `docs/tutorials/getting-started.md` for accuracy
  - [ ] 5.3 Update any expected output that differs from actual output
- [ ] Task 6: Run full test suite and verify all E2E tests pass (AC: all)
  - [ ] 6.1 `pytest tests/integration/test_documented_commands.py -v`
  - [ ] 6.2 `ruff check .`
  - [ ] 6.3 `mypy --strict src/ncaa_eval tests`

## Dev Notes

### Critical: `ncaa-eval` CLI Entry Point Does Not Exist

The getting-started tutorial (`docs/tutorials/getting-started.md:16`) documents:
```bash
ncaa-eval sync --source all --dest data/
```
But `pyproject.toml` has **no** `[tool.poetry.scripts]` section — there is no `ncaa-eval` console script. The README correctly uses `python sync.py` and `python -m ncaa_eval.cli train`. The tutorial also correctly notes `python sync.py` as a "legacy" invocation, but the "canonical" `ncaa-eval sync` form does not exist.

**Resolution options (dev must pick one and update the other):**
- **Option A:** Add `[tool.poetry.scripts]` entry: `ncaa-eval = "ncaa_eval.cli.main:app"` — then also wire up `sync` as a subcommand of the main Typer app
- **Option B:** Update `docs/tutorials/getting-started.md` to use `python sync.py` and `python -m ncaa_eval.cli train` as the canonical forms; remove `ncaa-eval sync` references

Option B is simpler and aligns with README. Prefer Option B unless the PO wants a unified CLI.

### CLI Structure

- **Training CLI**: `src/ncaa_eval/cli/main.py` — Typer app with `train` command
  - Entry: `python -m ncaa_eval.cli` (via `__main__.py`)
  - Test pattern: see `tests/unit/test_cli_train.py` (uses `typer.testing.CliRunner`)
- **Sync CLI**: `sync.py` (project root) — separate Typer app
  - Entry: `python sync.py`
  - Both CLIs have `--help` flags

### Subprocess vs. In-Process Testing

These are **E2E** tests — they MUST use `subprocess.run()` to invoke commands the same way a user would. Do NOT use `typer.testing.CliRunner` (that's unit-level testing, already done in `test_cli_train.py`). The point is to verify the full command-line invocation including import chain, entry point resolution, and environment correctness.

### Test Execution Time

These tests will be slow (each spawns a subprocess). Mark all tests with `@pytest.mark.integration` and `@pytest.mark.slow`. The full pytest run (AC #3) itself takes ~30-60s, so give generous timeouts:
- `pytest -m smoke`: 30s timeout
- `pytest` (full): 300s timeout (5 min)
- `pytest --cov`: 300s timeout
- `nox -s tests`: 300s timeout
- `mypy --strict`: 120s timeout
- Others: 60s timeout

### Nox Sessions

`noxfile.py` defines 4 sessions: `lint`, `typecheck`, `tests`, `docs`. Default sessions (no `-s`) run `lint`, `typecheck`, `tests`. Sessions use `python=False` (no virtualenv). The `docs` session runs sphinx-apidoc + sphinx-build.

The `nox -s docs` session is NOT in the E2E test ACs but could be added if desired.

### check-manifest

`check-manifest` is in `[tool.poetry.group.dev.dependencies]` and configured in `[tool.check-manifest]` in pyproject.toml. It is NOT in `.pre-commit-config.yaml`. Test whether `check-manifest` exits 0 from the repo root.

### Existing Test Patterns

- **Integration tests**: `tests/integration/` — 3 existing files, 34 tests
- **Markers**: `@pytest.mark.integration`, `@pytest.mark.slow`
- **No `@pytest.mark.unit` needed** — these are integration tests
- **`conftest.py`**: uses `temp_data_dir` fixture for temp directories

### Recursive Test Avoidance

AC #2-3 runs `pytest` as a subprocess, which will discover and run ALL tests including the E2E tests themselves. This creates infinite recursion. **Solution**: Either:
- Skip the E2E tests when running under subprocess (check env var)
- Use `-k "not test_documented_commands"` in the subprocess call
- Use `--ignore=tests/integration/test_documented_commands.py` in the subprocess call

**Recommended approach**: The subprocess pytest calls should use `--ignore=tests/integration/test_documented_commands.py` to prevent recursion. This is the cleanest — it explicitly documents the exclusion and avoids modifying test behavior based on environment.

### Project Structure Notes

- Tests live in `tests/integration/` (not `tests/e2e/`)
- Follow existing integration test patterns (module docstring, proper markers)
- The repo root is `/home/dhilg/git/NCAA_eval` — all subprocess commands should use `cwd=PROJECT_ROOT`

### References

- [Source: _bmad-output/planning-artifacts/epic-8-codebase-improvements.md#Story 8.10]
- [Source: docs/tutorials/getting-started.md — documented commands]
- [Source: README.md — documented commands]
- [Source: noxfile.py — session definitions]
- [Source: pyproject.toml — markers, check-manifest config]
- [Source: .pre-commit-config.yaml — hook configuration]
- [Source: tests/unit/test_cli_train.py — CLI test patterns (unit-level)]
- [Source: tests/integration/ — existing integration test patterns]

## Dev Agent Record

### Agent Model Used

{{agent_model_name_version}}

### Debug Log References

### Completion Notes List

### File List
