# Story 10.7: End-to-End User-Facing Execution Audit

Status: ready-for-dev

## Story

As a **project maintainer**,
I want to **execute every documented user-guide command and tutorial step verbatim and have CI permanently guard against execution-context failures**,
so that **users never encounter a broken workflow in the introductory documentation, and the "passes tests but fails for users" class of bug is structurally impossible to merge undetected**.

## Acceptance Criteria

1. **Full audit completed** — every command in `docs/user-guide.md`, `docs/tutorials/getting-started.md`, `docs/tutorials/custom-model.md`, `docs/tutorials/custom-metric.md`, and `notebooks/tutorials/03_ensemble_model.ipynb` is executed verbatim from the repo root; all failures are documented and fixed

2. **Known failures fixed:**
   - `streamlit run dashboard/app.py` starts cleanly without `ModuleNotFoundError` (the `dashboard` package must be resolvable from the process that Streamlit spawns, with no reliance on pytest's `sys.path` injection)
   - `python sync.py --source all --dest data/` imports and starts cleanly after a fresh `poetry install` (no import errors from any module introduced in Story 10.6 or later)

3. **New E2E test file** — `tests/e2e/test_user_facing_commands.py` is created with subprocess-based startup checks for all user-facing entry points. Tests invoke commands exactly as documented, with `cwd=REPO_ROOT`, and assert clean startup (returncode 0 and no `ModuleNotFoundError`/`ImportError` in stderr). All tests marked `@pytest.mark.integration`

4. **CI updated** — `python-check.yaml` gains a step that runs the E2E startup suite: `pytest tests/e2e/ -v`

5. **Testing strategy updated** — `docs/TESTING_STRATEGY.md` and `docs/testing/execution.md` gain a documented **"Execution Context Principle"**: tests must mirror user invocation (subprocess, correct `cwd`, no harness `sys.path` injection). The principle explicitly distinguishes *import-context tests* (what we had — vulnerable to path injection masking real failures) from *execution-context tests* (what we need — same process isolation as the user)

6. **No regressions** — full test suite passes: `pytest`, `mypy --strict src/ncaa_eval tests`, `ruff check .`

## Tasks / Subtasks

- [ ] Task 1: Audit — execute all documented commands verbatim (AC: #1)
  - [ ] 1.1 Execute every bash block in `docs/tutorials/getting-started.md` in order from repo root; record any failure with the exact error
  - [ ] 1.2 Execute every bash block in `docs/tutorials/custom-model.md`; record failures
  - [ ] 1.3 Execute every bash block in `docs/tutorials/custom-metric.md`; record failures
  - [ ] 1.4 Execute `streamlit run dashboard/app.py` from repo root (headless OK); confirm ImportError is reproducible and identify root cause
  - [ ] 1.5 Execute `python sync.py --help` and `python sync.py --source kaggle --dest /tmp/audit_data` from repo root; confirm clean import/startup; identify any 10-6 import error
  - [ ] 1.6 Execute `notebooks/tutorials/03_ensemble_model.ipynb` via `jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 --output-dir notebooks/tutorials/ notebooks/tutorials/03_ensemble_model.ipynb`; record any cell failure

- [ ] Task 2: Fix `streamlit run` execution-context failure (AC: #2)
  - [ ] 2.1 Reproduce the `ModuleNotFoundError: No module named 'dashboard'` — confirm exact Streamlit version and invocation context where it fails
  - [ ] 2.2 Identify root cause: Streamlit's subprocess `sys.path` does not include repo root in all environments/versions
  - [ ] 2.3 Implement fix — **preferred**: ensure `dashboard` is importable via installed package path (pyproject.toml) or via a `.pth` file, NOT via a `sys.path.insert` hack in `app.py`. If `dashboard` is not installable, document the required invocation (`PYTHONPATH=. streamlit run dashboard/app.py`) and add it to `docs/user-guide.md` and `docs/tutorials/getting-started.md`
  - [ ] 2.4 Confirm fix works with `streamlit run dashboard/app.py` (no env-var required, from bare repo root)

- [ ] Task 3: Fix `sync.py` / Story 10.6 import error (AC: #2)
  - [ ] 3.1 Identify the specific module introduced in Story 10.6 that causes the import failure
  - [ ] 3.2 Determine root cause: missing `poetry install` step, import side-effect, circular import, or missing `__init__.py` re-export
  - [ ] 3.3 Fix the import issue and verify `python sync.py --help` and `python -m ncaa_eval.cli --help` both succeed from a fresh `poetry install`

- [ ] Task 4: Fix any other audit failures found in Task 1 (AC: #1)
  - [ ] 4.1 For each failure documented in Task 1, apply the minimal fix (code, docs, or both)

- [ ] Task 5: Create `tests/e2e/test_user_facing_commands.py` (AC: #3)
  - [ ] 5.1 Add `tests/e2e/__init__.py` (empty)
  - [ ] 5.2 Implement startup-check tests using `subprocess.run()` — **not** `importlib`, **not** Python `import`. Each test must:
    - Set `cwd=REPO_ROOT` (`Path(__file__).parent.parent.parent` from `tests/e2e/`)
    - Capture stdout/stderr
    - Assert returncode == 0
    - Assert `"ModuleNotFoundError"` not in stderr
    - Assert `"ImportError"` not in stderr
  - [ ] 5.3 Implement tests for:
    - `python sync.py --help`
    - `python -m ncaa_eval.cli --help`
    - `python -m ncaa_eval.cli train --help`
    - `python -m ncaa_eval.cli predict --help`
    - `python -c "from ncaa_eval.model import list_models; print(list_models())"` — assert `elo` in stdout
    - Streamlit startup: launch `streamlit run dashboard/app.py --server.headless true --server.port 18501` via `subprocess.Popen`, wait 8 seconds, then poll; assert process has not exited with non-zero code AND stderr does not contain `ModuleNotFoundError`/`ImportError`
  - [ ] 5.4 All tests marked `@pytest.mark.integration` (they start subprocesses — too slow for smoke)

- [ ] Task 6: Update CI (`python-check.yaml`) (AC: #4)
  - [ ] 6.1 Add a step after the main pytest step: `poetry run pytest tests/e2e/ -v --tb=short`

- [ ] Task 7: Update testing strategy docs (AC: #5)
  - [ ] 7.1 Add "Execution Context" principle section to `docs/TESTING_STRATEGY.md` under Key Principles
  - [ ] 7.2 Add "Execution-Context Tests (E2E Startup)" section to `docs/testing/execution.md` explaining: what they test, why `subprocess.run()` is required, how they differ from import-context tests, and what the historical failure mode was
  - [ ] 7.3 Add `@pytest.mark.integration` example for E2E subprocess tests to the examples section

## Dev Notes

### Root Cause of the Historical Failure

The existing test suite used `importlib.import_module("dashboard.lib.filters")` to verify dashboard imports. This works under pytest because pytest's `rootdir` discovery adds the repo root to `sys.path`. However, `streamlit run dashboard/app.py` spawns a subprocess where `sys.path` may **not** include the repo root, depending on the Streamlit version and invocation environment. The result: dashboard passes all import tests but crashes for users.

The same structural gap applies to `sync.py` and `python -m ncaa_eval.cli`: if a new import is added and the test only exercises it via `importlib` under pytest, it passes; if the module is not installed (e.g., user hasn't re-run `poetry install` after pulling), the user-facing command fails.

**The fix is not just patching these individual cases — it is changing the testing philosophy**: execution-context tests must use `subprocess.run()` with an isolated process and correct `cwd`, never relying on the test harness to inject paths.

### Streamlit Path Investigation

When `streamlit run dashboard/app.py` is invoked, Streamlit adds the directory containing the script (`dashboard/`) to `sys.path`, not the repo root. This means `from dashboard.lib.data_loaders import ...` fails because `dashboard` as a package is found by looking for it *inside* `dashboard/` — a circular situation.

**Fix candidates (in preference order):**
1. Make `dashboard` an installable package: add `packages = ["dashboard", "dashboard.lib", ...]` to `pyproject.toml` so `poetry install` installs it into the conda env alongside `ncaa_eval`
2. Add a `.pth` file to the conda env `site-packages` pointing to the repo root (fragile, not portable)
3. Document `PYTHONPATH=. streamlit run dashboard/app.py` in all user-facing docs

Option 1 is preferred. Investigate whether `dashboard` is currently listed in `pyproject.toml` packages. If it is, the issue may be environment-specific (Streamlit version regression) — in that case, document the repro steps thoroughly.

### Subprocess Test Pattern

```python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent


@pytest.mark.integration
def test_sync_help() -> None:
    """sync.py --help must exit 0 with no import errors."""
    result = subprocess.run(
        [sys.executable, "sync.py", "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert "ModuleNotFoundError" not in result.stderr
    assert "ImportError" not in result.stderr
    assert result.returncode == 0
```

**Critical:** Never use `importlib.import_module()` in E2E tests. It inherits the test harness's `sys.path`. The whole point is to test without that inheritance.

### Notebook Execution Command (from MEMORY)

```bash
conda run -n ncaa_eval jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=600 --output-dir notebooks/tutorials \
  notebooks/tutorials/03_ensemble_model.ipynb
```

**Do NOT use `--output notebooks/tutorials/03_ensemble_model.ipynb`** — this doubles the path. Use `--output-dir` only. The nbconvert 7.x kernel CWD is the notebook's directory, so relative paths in cells resolve relative to `notebooks/tutorials/`, not the repo root.

The ensemble notebook requires real trained model artifacts and a synced `data/` directory. If these are not present, the notebook will fail. This is expected. The story's acceptance criterion is: the notebook executes without error *when run from a system with the correct prerequisites* — document those prerequisites clearly in the notebook's first cell.

### Commitizen Tag Issue (Out of Scope — Separate Hotfix)

The commitizen CI is failing with `fatal: tag '0.10.0' already exists`. This is because the tag was pushed in a prior partial run. Fix is a one-liner: `git push origin :refs/tags/0.10.0` to delete the remote tag, then re-trigger the CI bump workflow. This is **not** in scope for Story 10.7.

### Files to Touch

- `tests/e2e/__init__.py` — new (empty)
- `tests/e2e/test_user_facing_commands.py` — new
- `.github/workflows/python-check.yaml` — add E2E step
- `dashboard/` or `pyproject.toml` — streamlit path fix (exact file TBD by investigation)
- `sync.py` or `src/ncaa_eval/ingest/` — fix 10.6 import issue (exact file TBD)
- `docs/TESTING_STRATEGY.md` — add Execution Context Principle
- `docs/testing/execution.md` — add E2E Startup test section
- Possibly: `docs/user-guide.md`, `docs/tutorials/getting-started.md` — if invocation needs documenting

### References

- Streamlit path behavior: [Source: dashboard/app.py#L12] — `from dashboard.lib.data_loaders import ...`
- Existing (insufficient) import test: [Source: tests/unit/test_dashboard_app.py#L1]
- CI workflow: [Source: .github/workflows/python-check.yaml]
- Testing execution tiers: [Source: docs/testing/execution.md]
- Notebook execution pitfalls: [MEMORY: Jupyter/nbconvert Notes — use `--output-dir`, not `--output`]

## Dev Agent Record

### Agent Model Used

{{agent_model_name_version}}

### Debug Log References

### Completion Notes List

### File List
