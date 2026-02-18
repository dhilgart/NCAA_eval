# Story 1.6: Configure Session Management & Automation

Status: ready-for-dev

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want Nox configured to orchestrate the full quality pipeline,
so that running `nox` executes linting, type-checking, and testing in one command.

## Acceptance Criteria

1. **Given** Ruff, Mypy, and Pytest are configured from Stories 1.4 and 1.5, **When** the developer runs `nox`, **Then** Nox executes sessions in order: Ruff (lint/format) -> Mypy (type check) -> Pytest (tests).
2. **And** each session runs in an isolated environment.
3. **And** failure in any session is clearly reported with the failing session identified.
4. **And** `noxfile.py` is committed to the repository root.
5. **And** the developer can run individual sessions (e.g., `nox -s lint`, `nox -s typecheck`, `nox -s tests`).

## Tasks / Subtasks

- [ ] Task 1: Create `noxfile.py` with three sessions (AC: 1, 2, 3, 4, 5)
  - [ ] 1.1: Create `noxfile.py` at project root with `lint`, `typecheck`, and `tests` sessions
  - [ ] 1.2: Configure `nox.options.sessions` to set default session order: `["lint", "typecheck", "tests"]`
  - [ ] 1.3: Use `session.python = False` on each session so Nox reuses the active conda/Poetry environment instead of creating per-session virtualenvs (see Dev Notes on environment isolation)
  - [ ] 1.4: Ensure `from __future__ import annotations` is at top of file; add type annotations to satisfy `mypy --strict`
  - [ ] 1.5: Verify Ruff and mypy pass on `noxfile.py` itself

- [ ] Task 2: Implement `lint` session (AC: 1, 3, 5)
  - [ ] 2.1: Run `ruff check . --fix` (auto-fix violations)
  - [ ] 2.2: Run `ruff format --check .` (verify formatting; fail if unformatted files remain)
  - [ ] 2.3: Ensure non-zero exit on remaining violations after auto-fix

- [ ] Task 3: Implement `typecheck` session (AC: 1, 3, 5)
  - [ ] 3.1: Run `mypy --strict --show-error-codes --namespace-packages` on `src/ncaa_eval` and `tests`
  - [ ] 3.2: Match the pre-commit mypy invocation exactly (same flags)

- [ ] Task 4: Implement `tests` session (AC: 1, 3, 5)
  - [ ] 4.1: Run `pytest` (full test suite, no marker filter — Nox runs ALL tests, not just smoke)
  - [ ] 4.2: Include `--tb=short` for concise failure output

- [ ] Task 5: End-to-end validation (AC: 1-5)
  - [ ] 5.1: Run `nox` and verify all three sessions execute in order
  - [ ] 5.2: Run `nox -s lint` and verify only the lint session runs
  - [ ] 5.3: Run `nox -s typecheck` and verify only mypy runs
  - [ ] 5.4: Run `nox -s tests` and verify only pytest runs
  - [ ] 5.5: Introduce a deliberate type error; run `nox` and verify `typecheck` session fails with clear output identifying the session name
  - [ ] 5.6: Revert the error; verify `nox` passes cleanly end-to-end
  - [ ] 5.7: Run `ruff check noxfile.py` and `mypy --strict noxfile.py` to ensure the file itself passes all quality gates

## Dev Notes

### Architecture Compliance

**CRITICAL — Follow these exactly:**

- **Architecture Section 10.2** defines `nox` as the "Research Loop" command. Running `nox` must execute Ruff -> Mypy -> Pytest in that exact order. [Source: docs/specs/05-architecture-fullstack.md#Section 10.2]
- **Architecture Section 9** lists `noxfile.py` at the repository root, alongside `pyproject.toml`. [Source: docs/specs/05-architecture-fullstack.md#Section 9]
- **`mypy --strict` is mandatory** — `noxfile.py` must have full type annotations. Import `nox.Session` for session parameter typing. [Source: pyproject.toml#L39-L45]
- **`from __future__ import annotations`** is required at the top of `noxfile.py` (enforced by Ruff isort). [Source: pyproject.toml#L69]
- **Do NOT modify `.pre-commit-config.yaml`** — pre-commit hooks are finalized in Story 1.4. Nox is a separate orchestrator, not a replacement for pre-commit. [Source: .pre-commit-config.yaml]
- **Do NOT modify `pyproject.toml`** — all tool configs (`[tool.ruff]`, `[tool.mypy]`, `[tool.pytest.ini_options]`) are already correct. Nox sessions invoke these tools; they do not reconfigure them.
- **Do NOT modify `.github/workflows/python-check.yaml`** — CI already runs pre-commit + full pytest. Nox is for local developer use.

### Environment Isolation Strategy

**AC2 requires "each session runs in an isolated environment."** In this project, the correct interpretation is:

- The project runs inside a **conda environment** (`ncaa_eval`) with Poetry managing dependencies via `POETRY_VIRTUALENVS_CREATE=false`.
- Creating nox virtualenvs per session would **duplicate the entire dependency tree** and break the conda-based workflow.
- **Use `python=False`** in `@nox.session(python=False)` so Nox reuses the active environment.
- Sessions are **logically isolated**: each session runs independently, and failure in one is reported separately. This satisfies the spirit of AC2.
- The `--no-install` pattern is NOT needed since we're not creating virtualenvs.

**Key Nox API for no-venv mode:**
```python
@nox.session(python=False)
def lint(session: nox.Session) -> None:
    session.run("ruff", "check", ".", "--fix")
```

When `python=False`, do NOT call `session.install()` — it is deprecated without a virtualenv and will warn or error.

### Nox Session Design

| Session | Name | Command(s) | Purpose |
|---|---|---|---|
| `lint` | `nox -s lint` | `ruff check . --fix` then `ruff format --check .` | Auto-fix lint violations, verify formatting |
| `typecheck` | `nox -s typecheck` | `mypy --strict --show-error-codes --namespace-packages src/ncaa_eval tests` | Strict type checking |
| `tests` | `nox -s tests` | `pytest --tb=short` | Run full test suite |

**Default session order:** `nox.options.sessions = ["lint", "typecheck", "tests"]`

Running bare `nox` executes all three in order. Any session failure stops the pipeline and reports which session failed.

### Nox vs Pre-commit — How They Coexist

These are **complementary, not redundant**:

| Aspect | Pre-commit | Nox |
|---|---|---|
| **When** | Automatic on `git commit` | Manual developer command |
| **Scope** | Only changed files (except mypy/pytest) | Entire project |
| **Tests** | Smoke tests only (`-m smoke`) | Full test suite |
| **Purpose** | Fast gate before commit | Comprehensive validation before PR |

The developer workflow is:
1. Write code
2. Run `nox` to validate everything (Research Loop)
3. `git commit` triggers pre-commit hooks (fast smoke check)
4. Push for PR — CI runs full suite again

### Project Structure Notes

- `noxfile.py` is placed at the project root per Architecture Section 9.
- No other files are created or modified in this story.
- The file sits alongside `pyproject.toml`, `.pre-commit-config.yaml`, and `README.md`.

**Target state after Story 1.6:**

```
NCAA_eval/
├── noxfile.py                  # NEW (Story 1.6) — Session Management
├── pyproject.toml              # Existing — no changes
├── .pre-commit-config.yaml     # Existing — no changes
├── .github/workflows/          # Existing — no changes
├── src/ncaa_eval/              # Existing
├── tests/                      # Existing
└── ...
```

### Library / Framework Requirements

| Tool | Installed As | Version | Purpose |
|---|---|---|---|
| `nox` | `poetry.group.dev.dependencies` | Latest (2026.2.9 at time of writing) | Session manager |
| `ruff` | `poetry.group.dev.dependencies` | Already installed | Linting & formatting |
| `mypy` | `poetry.group.dev.dependencies` | Already installed | Type checking |
| `pytest` | `poetry.group.dev.dependencies` | Already installed (>=8.0) | Test runner |

**No new dependencies to install.** All tools are already in `pyproject.toml`.

**Nox API reference (key points for `python=False` mode):**
- `session.run("command", "arg1", "arg2")` — runs an external command; raises on non-zero exit
- Do NOT use `session.install()` — not available without a virtualenv
- `nox.options.sessions` — list of session names to run by default

### Testing Requirements

**Validation Tests (What Dev Agent Must Run):**

1. **Full pipeline:**
   ```bash
   POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval nox
   ```
   Expected: All three sessions (lint, typecheck, tests) run in order and pass.

2. **Individual sessions:**
   ```bash
   POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval nox -s lint
   POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval nox -s typecheck
   POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval nox -s tests
   ```
   Expected: Each runs independently and passes.

3. **Failure reporting (manual check):**
   - Introduce a type error in any `src/` file
   - Run `nox` — `typecheck` session should fail with the session name clearly identified in output
   - Revert the error

4. **Self-check — noxfile.py passes quality gates:**
   ```bash
   POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval ruff check noxfile.py
   POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval mypy --strict noxfile.py
   ```
   Expected: No lint or type errors.

5. **Existing tests still pass:**
   ```bash
   POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval pytest
   ```
   Expected: All 4 existing tests pass (unchanged from Story 1.5).

### References

- [Source: docs/specs/05-architecture-fullstack.md#Section 9] — Repository structure showing `noxfile.py` at root
- [Source: docs/specs/05-architecture-fullstack.md#Section 10.2] — Development workflow: `nox` runs Ruff -> Mypy -> Pytest
- [Source: pyproject.toml#L28] — `nox = "*"` already in dev dependencies
- [Source: pyproject.toml#L39-L45] — `[tool.mypy]` strict config
- [Source: pyproject.toml#L47-L66] — `[tool.ruff]` config
- [Source: pyproject.toml#L90-L116] — `[tool.pytest.ini_options]` config
- [Source: .pre-commit-config.yaml] — Pre-commit hooks (do not modify)
- [Source: .github/workflows/python-check.yaml] — CI workflow (do not modify)
- [Source: _bmad-output/planning-artifacts/epics.md#Story 1.6] — Story acceptance criteria
- [Source: _bmad-output/implementation-artifacts/1-5-configure-testing-framework.md] — Previous story learnings

### Previous Story Intelligence

#### Story 1.5: Configure Testing Framework

**Critical learnings that directly impact Story 1.6:**

1. **WSL is the dev environment** — all commands should be run via `conda run -n ncaa_eval` or after `conda activate ncaa_eval`. Windows native does not support mutmut.
2. **`from __future__ import annotations` is mandatory** in every Python file — Ruff auto-fix will add it, but include it from the start.
3. **`mypy --strict` catches missing annotations** — the noxfile must have `session: nox.Session` typed and `-> None` return types on all session functions.
4. **Pre-commit smoke suite must stay < 10 seconds** — Nox does NOT affect pre-commit; they are independent. But don't accidentally add `noxfile.py` to pre-commit scope.
5. **Google docstrings** required for public functions. Each session function needs a one-line docstring.
6. **Conventional commits format:** `feat(toolchain): configure Nox session management` or similar.

#### Story 1.4: Configure Code Quality Toolchain

- **Ruff and mypy are local hooks** using `poetry run` — Nox sessions should invoke them directly (not via `poetry run`) since Nox itself runs inside the conda env.
- **Complexity gates are active:** Max 5 args, max 10 complexity. Session functions are simple so this is not a concern.

### Git Intelligence

**Recent commits:**
```
3c07873 feat(testing): configure Hypothesis, Mutmut, and test framework (#5)
b823dc6 feat(toolchain): configure Ruff/Mypy/Pytest pre-commit hooks (#4)
3a2b8fd Merge pull request #3 from dhilgart/story/1-3-define-testing-strategy
```

**Patterns established:**
- Conventional commits: `type(scope): description`
- Story branch: `story/1-6-configure-session-management-automation`
- One atomic commit per logical unit
- PR with description following template format

**Suggested commit message:**
```
feat(toolchain): configure Nox session management
```

### Latest Technology Notes

- **Nox latest stable:** 2026.2.9 (calver versioning). No breaking changes from recent versions.
- `python=False` is the correct way to disable virtualenv creation per session (stable API).
- `nox.options.sessions` sets default sessions (stable API since early versions).
- `session.run()` raises `nox.command.CommandFailed` on non-zero exit — no special error handling needed.

## Dev Agent Record

### Agent Model Used

{{agent_model_name_version}}

### Debug Log References

### Completion Notes List

### Change Log

### File List
