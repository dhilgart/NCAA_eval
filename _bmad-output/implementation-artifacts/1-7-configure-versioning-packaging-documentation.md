# Story 1.7: Configure Versioning, Packaging & Documentation

Status: in-progress

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want Commitizen, check-manifest, edgetest, and Sphinx configured,
so that the project has automated versioning, package integrity checks, dependency management, and documentation generation.

## Acceptance Criteria

1. **Given** the Poetry project structure from Story 1.1 is in place, **When** the developer uses Commitizen for commits, **Then** commit messages follow the conventional commits format and version bumps are automated via `cz bump`.
2. **And** `check-manifest` validates that the package manifest includes all necessary files.
3. **And** edgetest is configured for dependency compatibility testing.
4. **And** `sphinx-build` generates HTML documentation from the `docs/` directory using the Furo theme.
5. **And** `sphinx-apidoc` can auto-generate API docs from module docstrings.
6. **And** a Nox session exists for documentation generation (`nox -s docs`).

## Tasks / Subtasks

- [x] Task 1: Verify and document Commitizen behavior (AC: 1)
  - [x] 1.1: Verify `cz check --rev-range HEAD~5..HEAD` validates recent commits pass conventional commits format
  - [x] 1.2: Verify `cz bump --dry-run` correctly reads commits and determines version bump
  - [x] 1.3: Update `CONTRIBUTING.md` to replace outdated pipenv/invoke workflow with Poetry/Nox/Commitizen

- [x] Task 2: Configure check-manifest (AC: 2)
  - [x] 2.1: Run `check-manifest` to discover any issues
  - [x] 2.2: Add `[tool.check-manifest]` ignore section to `pyproject.toml` for non-distribution files
  - [x] 2.3: Verify `check-manifest` passes cleanly

- [x] Task 3: Configure edgetest scaffolding (AC: 3)
  - [x] 3.1: Add `[tool.edgetest.envs.latest]` section to `pyproject.toml` specifying which deps to test against latest
  - [x] 3.2: Verify `edgetest` command can be invoked without errors

- [x] Task 4: Set up Sphinx documentation (AC: 4, 5)
  - [x] 4.1: Remove `mkdocs.yml` from project root (see Dev Notes — legacy cookiecutter stub, never configured)
  - [x] 4.2: Create `docs/conf.py` with Sphinx + Furo + Napoleon extension configuration
  - [x] 4.3: Create `docs/index.rst` as documentation root
  - [x] 4.4: Verify `sphinx-apidoc -f -e -o docs/api src/ncaa_eval` generates API stubs
  - [x] 4.5: Verify `sphinx-build -b html docs docs/_build/html` generates HTML without errors

- [ ] Task 5: Add `docs` Nox session (AC: 6)
  - [ ] 5.1: Add `docs` session to `noxfile.py` using `python=False` (same pattern as existing sessions)
  - [ ] 5.2: Verify `nox -s docs` succeeds end-to-end
  - [ ] 5.3: Verify updated `noxfile.py` passes `ruff check noxfile.py` and `mypy --strict noxfile.py`

- [ ] Task 6: End-to-end validation (all ACs)
  - [ ] 6.1: Run `nox -s docs` and verify `docs/_build/html/index.html` is generated
  - [ ] 6.2: Run `check-manifest` and verify clean pass
  - [ ] 6.3: Run `cz check --rev-range HEAD~5..HEAD` and verify recent commits pass
  - [ ] 6.4: Run `cz bump --dry-run` to preview version bump behavior
  - [ ] 6.5: Run `nox` (full pipeline) and verify no regressions (lint, typecheck, tests all pass)

## Dev Notes

### Architecture Compliance

**CRITICAL — Follow these exactly:**

- **Architecture Section 9** shows `docs/` at repository root alongside `pyproject.toml` and `noxfile.py`. [Source: docs/specs/05-architecture-fullstack.md#Section 9]
- **`from __future__ import annotations`** is required at the top of `noxfile.py` (already present — do not remove when adding the `docs` session). [Source: pyproject.toml#L69]
- **`mypy --strict` is mandatory** — `noxfile.py` must continue to type-check cleanly after adding the `docs` session. [Source: pyproject.toml#L39-L45]
- **Do NOT modify `.pre-commit-config.yaml`** — commitizen hooks are already configured and working.
- **Do NOT modify `[tool.commitizen]` in `pyproject.toml`** — already configured; Story 1.7 verifies and documents, not reconfigures.
- **`docs/conf.py` is a Sphinx config file, NOT a regular Python module** — do NOT add it to the `mypy` invocation in the `typecheck` Nox session. It uses implicit Sphinx globals and mypy cannot type-check it.

### Pre-Existing State (Critically Important)

**What is ALREADY configured — do not reconfigure:**

| Item | Status | Location |
|---|---|---|
| `commitizen` dep | ✅ Already installed | `pyproject.toml` line 29 |
| `check-manifest` dep | ✅ Already installed | `pyproject.toml` line 30 |
| `edgetest` dep | ✅ Already installed | `pyproject.toml` line 31 |
| `sphinx` dep | ✅ Already installed | `pyproject.toml` line 32 |
| `furo` dep | ✅ Already installed | `pyproject.toml` line 33 |
| `[tool.commitizen]` config | ✅ Already configured | `pyproject.toml` lines 135–139 |
| Commitizen pre-commit hooks | ✅ Already active | `.pre-commit-config.yaml` lines 31–38 |
| `CHANGELOG.md` | Empty stub | Project root (created by cookiecutter template) |
| `docs/_build/` | ✅ Already gitignored | `.gitignore` line 79 |

**Commitizen configuration already in `pyproject.toml`:**
```toml
[tool.commitizen]
name = "cz_conventional_commits"
version = "0.1.0"
tag_format = "$version"
version_files = ["pyproject.toml:version"]
```

**Commitizen pre-commit hooks already in `.pre-commit-config.yaml`:**
```yaml
- repo: https://github.com/commitizen-tools/commitizen
  rev: v3.29.0
  hooks:
    - id: commitizen           # Validates commit-msg format on every commit
    - id: commitizen-branch    # Validates branch commits on post-commit and push
```

The `commitizen` hook runs at the `commit-msg` stage and **already blocks commits that don't follow conventional commits format**. Story 1.7 just needs to verify it works and document the workflow — not configure it from scratch.

**IMPORTANT: All deps may already be installed in the conda env.** If any tool is not found, run:
```bash
POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval poetry install --with dev
```

### The `mkdocs.yml` Problem (Action Required)

`mkdocs.yml` exists at the project root — it came from `Lee-W/cookiecutter-python-template` v1.11.0 and was **never configured beyond the stub**. It references MkDocs Material theme but no content was ever set up.

**The story ACs specify Sphinx + Furo. These two documentation systems conflict.**

**Action: Delete `mkdocs.yml`** — it was never used, conflicts with the Sphinx approach, and the `.gitignore` already has `/site` as the MkDocs build output (also safe to leave as-is). No content will be lost.

### `CONTRIBUTING.md` Is Outdated (Action Required)

`CONTRIBUTING.md` was generated from the cookiecutter template and describes the **old pipenv/invoke workflow** (`inv env.init-dev`, `inv git.commit`, `inv test`, `inv style`, etc.). None of these tools are used in this project.

**Update `CONTRIBUTING.md` to reflect the actual workflow:**
1. Clone and setup: `poetry install --with dev` + `pre-commit install`
2. Commit workflow: `git commit` (Commitizen hook validates message format automatically)
3. Run quality gates: `nox` (lint → typecheck → tests)
4. Generate docs: `nox -s docs`
5. Preview version bump: `cz bump --dry-run`

**Keep the existing conventional commits badge** in README.md (already correct).

### Sphinx Configuration for src-layout

**`docs/conf.py` MUST add `src/` to `sys.path` for autodoc to find the package:**

```python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

project = "ncaa_eval"
author = "Dan Hilgart"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",    # REQUIRED: converts Google-style docstrings to Sphinx format
]

html_theme = "furo"

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autosummary_generate = True
```

**`sphinx.ext.napoleon` is mandatory** — this project uses **Google-style docstrings** (per `STYLE_GUIDE.md` and the Ruff `pydocstyle.convention = "google"` setting). Without napoleon, Google-style docstrings render as plain text instead of formatted documentation.

**`sys.path.insert` must point to `src/`** (not project root) for src-layout packages. Pointing to the project root means Python finds `src/ncaa_eval/` correctly via the installed package, but autodoc may fail to resolve imports.

**Do NOT add** `sphinx_autodoc_typehints` — requires separate install and conflicts with napoleon's type rendering.

**`docs/index.rst` minimal structure:**
```rst
NCAA Eval Documentation
=======================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
```

### Nox `docs` Session

Add this session to `noxfile.py` after the existing `tests` session:

```python
@nox.session(python=False)
def docs(session: nox.Session) -> None:
    """Generate Sphinx HTML documentation from source docstrings."""
    session.run("sphinx-apidoc", "-f", "-e", "-o", "docs/api", "src/ncaa_eval")
    session.run("sphinx-build", "-b", "html", "docs", "docs/_build/html")
```

**Critical design decisions:**
- `python=False` — reuse conda env (same as all other sessions). Do NOT call `session.install()`.
- Run `sphinx-apidoc` first (regenerates API stubs), then `sphinx-build` (builds HTML).
- `-f`: Force overwrite existing stubs. `-e`: One file per module.
- Output: `docs/_build/html/` (already gitignored via `.gitignore` line 79).

**Do NOT add `docs` to `nox.options.sessions`** — documentation generation is on-demand, not part of the Ruff → Mypy → Pytest quality pipeline. Keep defaults as `["lint", "typecheck", "tests"]`.

**`docs/api/` directory** — sphinx-apidoc generates `.rst` stub files here. These are generated output. Options:
- Add `docs/api/` to `.gitignore` (regenerated on each `nox -s docs` run)
- OR commit the stubs (allows PRs to show API doc changes)

For now, commit the stubs so PR diffs show when API changes affect documentation.

### check-manifest Configuration

check-manifest verifies that `sdist` distributions include all version-controlled files. For Poetry projects using `packages = [{include = "ncaa_eval", from = "src"}]`, many project-management files are correctly excluded from the distribution.

Add to `pyproject.toml` after the `[tool.commitizen]` section:

```toml
[tool.check-manifest]
ignore = [
    "_bmad/**",
    "_bmad-output/**",
    "docs/specs/**",
    "docs/testing/**",
    "docs/archive/**",
    "docs/_build/**",
    "docs/api/**",
    "tasks/**",
    "*.yaml",
    "*.yml",
    ".pre-commit-config.yaml",
    "CONTRIBUTING.md",
    "STYLE_GUIDE.md",
    "TESTING_STRATEGY.md",
]
```

Run `check-manifest` first without config to see what it flags, then tune the ignore list. The goal is for `check-manifest` to pass cleanly — these non-source files are correctly not included in the Python package distribution.

### edgetest Configuration

edgetest tests packages against minimum/maximum dependency versions. Since this project uses `"*"` (any version) for all dependencies, configure the scaffolding for future use:

```toml
[tool.edgetest]
extras = []

[[tool.edgetest.envs]]
name = "latest"
upgrade = ["pandas", "numpy", "scikit-learn", "xgboost", "networkx", "joblib"]
command = "pytest tests/ -m smoke --tb=short -q"
```

**Important context**: edgetest becomes valuable when dependency bounds are added in later epics. For now, the AC requires "configured for dependency compatibility testing" — scaffolding the config satisfies this. Do not block on edgetest CI failures; the tool tests against upper bounds which may not exist yet.

### Nox `typecheck` Session — No Changes Needed

The existing `typecheck` session already includes `noxfile.py`:
```
mypy --strict --show-error-codes --namespace-packages src/ncaa_eval tests noxfile.py
```

After adding the `docs` session:
- `noxfile.py` is already in scope — the new session will be type-checked automatically
- `docs/conf.py` should NOT be added to mypy scope (Sphinx config, uses implicit globals)

### Library / Framework Requirements

| Tool | Status | Purpose | Key Commands |
|---|---|---|---|
| `commitizen` | Already installed | Conventional commits + version bumps | `cz check`, `cz bump --dry-run`, `cz bump --changelog` |
| `check-manifest` | Already installed | Package manifest validation | `check-manifest` |
| `edgetest` | Already installed | Dependency edge testing | `edgetest` |
| `sphinx` | Already installed | Documentation generation | `sphinx-build`, `sphinx-apidoc` |
| `furo` | Already installed | Sphinx HTML theme | (used via `html_theme = "furo"` in conf.py) |

**No new dependencies needed.** All are in `pyproject.toml` dev group.

**Commitizen API reference:**
- `cz check` — validate current commit message
- `cz check --rev-range HEAD~5..HEAD` — validate recent commit history
- `cz bump --dry-run` — preview version bump without applying
- `cz bump --changelog` — bump version + update `CHANGELOG.md`
- `cz changelog` — update CHANGELOG without bumping version

**Sphinx API reference:**
- `sphinx-apidoc -f -e -o docs/api src/ncaa_eval` — generate API stubs
- `sphinx-build -b html docs docs/_build/html` — build HTML from `docs/`
- `-W` flag makes warnings into errors (optional, use if build is noisy)

### Testing Requirements

**Validation commands (all must pass):**

1. **Documentation build:**
   ```bash
   POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval nox -s docs
   ```
   Expected: `docs/_build/html/index.html` exists; exit code 0.

2. **Commitizen validation:**
   ```bash
   POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval cz check --rev-range HEAD~5..HEAD
   ```
   Expected: All recent commits pass conventional commits format check.

3. **Version bump dry run:**
   ```bash
   POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval cz bump --dry-run
   ```
   Expected: Shows predicted version bump OR "No commits found to generate a changelog" (if no tag exists yet — v0.1.0 has not been tagged). Either result is valid.

4. **Package manifest:**
   ```bash
   POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval check-manifest
   ```
   Expected: Clean pass (exit code 0).

5. **Noxfile quality gates after changes:**
   ```bash
   POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval ruff check noxfile.py
   POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval mypy --strict noxfile.py
   ```
   Expected: No lint or type errors.

6. **Regression — full pipeline:**
   ```bash
   POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval nox
   ```
   Expected: All three sessions (lint, typecheck, tests) pass. No regressions from Story 1.6.

### Project Structure Notes

**Target state after Story 1.7:**

```
NCAA_eval/
├── noxfile.py                  # MODIFIED — Add docs session
├── docs/
│   ├── conf.py                 # NEW — Sphinx configuration with Furo + Napoleon
│   ├── index.rst               # NEW — Documentation root
│   ├── api/                    # NEW — sphinx-apidoc output stubs (committed)
│   ├── _build/html/            # GENERATED — Sphinx output (already gitignored)
│   ├── specs/                  # Existing — no changes
│   └── testing/                # Existing — no changes
├── CONTRIBUTING.md             # MODIFIED — Replace pipenv/invoke with Poetry/Nox
├── mkdocs.yml                  # DELETED — Legacy stub, replaced by Sphinx
├── CHANGELOG.md                # Existing empty stub — cz bump will populate
├── pyproject.toml              # MODIFIED — Add check-manifest + edgetest config
└── ...
```

**Files NOT to touch:**
- `.pre-commit-config.yaml` — commitizen hooks already correct
- `.gitignore` — already has `docs/_build/` and `/site` entries
- `src/ncaa_eval/` — no source changes in this story
- `tests/` — no test changes in this story

### References

- [Source: docs/specs/05-architecture-fullstack.md#Section 9] — Repository structure
- [Source: pyproject.toml#L29-L33] — All dev deps already present
- [Source: pyproject.toml#L135-L139] — Commitizen already configured
- [Source: .pre-commit-config.yaml#L31-L38] — Commitizen hooks already active
- [Source: docs/STYLE_GUIDE.md] — Google-style docstrings (critical for sphinx.ext.napoleon)
- [Source: _bmad-output/planning-artifacts/epics.md#Story 1.7] — Acceptance criteria
- [Source: _bmad-output/implementation-artifacts/1-6-configure-session-management-automation.md] — Previous story patterns (python=False, nox session design)
- [Source: README.md] — "Created from Lee-W/cookiecutter-python-template v1.11.0" (explains mkdocs.yml and CONTRIBUTING.md origin)

### Previous Story Intelligence (Story 1.6)

**Critical learnings that directly impact Story 1.7:**

1. **`python=False` for all Nox sessions** — The `docs` session must use `@nox.session(python=False)`. Do NOT call `session.install()` in python=False mode.
2. **Google docstrings with single backticks** — The `docs` session function docstring should use single backtick `` `code` `` not RST double backtick ` `` code `` `. Ruff does not catch this automatically.
3. **Include noxfile.py in typecheck scope** — Already done in Story 1.6 (the typecheck session includes `noxfile.py`). After adding the `docs` session, run `mypy --strict noxfile.py` to verify no new type errors.
4. **Do NOT modify `.pre-commit-config.yaml`** — already finalized in earlier stories.
5. **Ruff will auto-fix `from __future__ import annotations`** — the noxfile already has it; any new Python files created must include it too.

### Git Intelligence

**Recent commits:**
```
407cda0 Configure Nox session management & automation (#6)
3c07873 feat(testing): configure Hypothesis, Mutmut, and test framework (#5)
b823dc6 feat(toolchain): configure Ruff/Mypy/Pytest pre-commit hooks (#4)
```

**Patterns established:**
- Conventional commits: `type(scope): description` (enforced by commitizen hook)
- Story branch: `story/1-7-configure-versioning-packaging-documentation`
- PR descriptions follow `.github/pull_request_template.md` structure

**Warning:** The `commitizen` pre-commit hook validates every commit message. Use conventional commit format: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `style`.

**Suggested commit messages:**
```
feat(docs): configure Sphinx with Furo theme and nox docs session
feat(packaging): configure check-manifest and edgetest scaffolding
docs(contributing): update workflow to Poetry/Nox toolchain
chore: remove legacy mkdocs.yml stub
```

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

### Completion Notes List

- Task 4 (Sphinx): Created docs/conf.py with Furo theme, sphinx.ext.napoleon (for Google docstrings), autodoc, autosummary. sys.path.insert points to src/ for src-layout. Created docs/index.rst. Deleted mkdocs.yml (legacy stub). sphinx-apidoc generated 6 module stub files in docs/api/. sphinx-build succeeded with zero warnings — HTML at docs/_build/html/index.html.
- Task 2 (check-manifest): Added [tool.check-manifest] ignore section covering .claude/**, .windsurf/**, _bmad/**, _bmad-output/**, docs/{specs,testing,archive,_build,api}/**, tasks/**, *.yaml, *.yml, dashboard/**, data/**, tests/**, noxfile.py, lock files, and developer docs. check-manifest passes cleanly.
- Task 3 (edgetest): Added [tool.edgetest] scaffolding with 'latest' env testing pandas/numpy/scikit-learn/xgboost/networkx/joblib against smoke tests. edgetest --help invokes without errors.
- Task 1 (Commitizen): Verified cz check behavior — 4/5 recent commits pass; 1 GitHub squash-merge commit fails (PR title "Configure Nox session management & automation (#6)" lacks conventional commits type prefix). Documented in CONTRIBUTING.md that PR titles must also follow conventional commits format when using GitHub squash-merge. `cz bump --dry-run --yes` works (--yes required in non-interactive env when no initial tag exists); shows 0.1.0 → 0.2.0 MINOR bump.

### File List

- `CONTRIBUTING.md` — modified: replaced pipenv/invoke workflow with Poetry/Nox/Commitizen
- `pyproject.toml` — modified: added [tool.check-manifest] ignore section and [tool.edgetest] scaffolding
- `mkdocs.yml` — deleted: legacy cookiecutter stub, replaced by Sphinx
- `docs/conf.py` — new: Sphinx configuration with Furo theme, Napoleon, autodoc, autosummary
- `docs/index.rst` — new: Documentation root with toctree pointing to API docs
- `docs/api/modules.rst` — new: generated by sphinx-apidoc
- `docs/api/ncaa_eval.rst` — new: generated by sphinx-apidoc
- `docs/api/ncaa_eval.evaluation.rst` — new: generated by sphinx-apidoc
- `docs/api/ncaa_eval.ingest.rst` — new: generated by sphinx-apidoc
- `docs/api/ncaa_eval.model.rst` — new: generated by sphinx-apidoc
- `docs/api/ncaa_eval.transform.rst` — new: generated by sphinx-apidoc
- `docs/api/ncaa_eval.utils.rst` — new: generated by sphinx-apidoc
