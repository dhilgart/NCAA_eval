# Story 1.1: Initialize Repository & Package Structure

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want a Poetry-managed Python project with src layout and core directory scaffolding,
so that I can `poetry install` into a working virtualenv with the correct package structure.

## Acceptance Criteria

1. **Given** a fresh clone of the repository, **When** the developer runs `poetry install`, **Then** a virtualenv is created with all core dependencies installed.
2. **And** the `src/ncaa_eval/` package is importable (`import ncaa_eval` succeeds).
3. **And** the directory structure matches the Architecture spec: `src/ncaa_eval/{ingest,transform,model,evaluation,utils}/`, `dashboard/`, `tests/`, `data/`.
4. **And** `pyproject.toml` specifies Python 3.12+ and declares all PRD-required dependencies (pandas, numpy, xgboost, scikit-learn, networkx, joblib, plotly, streamlit).
5. **And** a `.gitignore` excludes `data/`, virtualenvs, and common Python artifacts.

## Tasks / Subtasks

- [x] Task 1: Create `pyproject.toml` with Poetry configuration (AC: 1, 4)
  - [x] 1.1: Set project name `ncaa-eval`, version `0.1.0`, Python `>=3.12`
  - [x] 1.2: Add all core dependencies: pandas, numpy, xgboost, scikit-learn, networkx, joblib, plotly, streamlit
  - [x] 1.3: Add dev dependencies: pytest, hypothesis, mutmut, ruff, mypy, pre-commit, nox, commitizen, check-manifest, edgetest, sphinx, furo (note: sphinx-apidoc is bundled with sphinx, not a separate package)
  - [x] 1.4: Configure src layout: `packages = [{include = "ncaa_eval", from = "src"}]`
- [x] Task 2: Create directory scaffolding (AC: 2, 3)
  - [x] 2.1: Create `src/ncaa_eval/__init__.py`
  - [x] 2.2: Create `src/ncaa_eval/ingest/__init__.py`
  - [x] 2.3: Create `src/ncaa_eval/transform/__init__.py`
  - [x] 2.4: Create `src/ncaa_eval/model/__init__.py`
  - [x] 2.5: Create `src/ncaa_eval/evaluation/__init__.py`
  - [x] 2.6: Create `src/ncaa_eval/utils/__init__.py`
  - [x] 2.7: Create `dashboard/app.py` (empty Streamlit entry point)
  - [x] 2.8: Create `dashboard/pages/` directory with placeholder (1_Lab.py, 2_Presentation.py)
  - [x] 2.9: Create `dashboard/components/` directory with placeholder (__init__.py)
  - [x] 2.10: Create `tests/__init__.py` (pre-existing)
  - [x] 2.11: Create `data/` directory with `.gitkeep`
- [x] Task 3: Create `.gitignore` (AC: 5)
  - [x] 3.1: Exclude `data/` (except `.gitkeep`), virtualenvs (`.venv/`, `venv/`), common Python artifacts (`__pycache__/`, `*.pyc`, `*.egg-info/`, `dist/`, `build/`, `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/`)
- [x] Task 4: Run `poetry install` and verify (AC: 1, 2)
  - [x] 4.1: Run `poetry install` and confirm no errors (118 packages installed)
  - [x] 4.2: Run `poetry run python -c "import ncaa_eval"` and confirm success
  - [x] 4.3: Verify virtualenv was created (ncaa-eval-Mo2VJmcB-py3.14)

## Dev Notes

### Architecture Compliance

**CRITICAL — Follow these exactly:**

- **Project Structure:** Must match Architecture Section 9 precisely. The canonical structure is:

```text
ncaa-eval-platform/
├── src/
│   └── ncaa_eval/              # Core Logic Package
│       ├── __init__.py
│       ├── ingest/             # Data Ingestion logic
│       ├── transform/          # Feature Engineering and Data Transformation
│       ├── model/              # Model Implementations
│       ├── evaluation/         # Metrics & Simulation
│       └── utils/              # Shared utilities
├── dashboard/                  # Streamlit Frontend
│   ├── app.py                  # Entry point
│   ├── pages/
│   │   ├── 1_Lab.py
│   │   └── 2_Presentation.py
│   └── components/             # Reusable UI widgets
├── docs/                       # Documentation (already exists)
├── tests/                      # Test Suite
├── data/                       # Local Data Store (git-ignored)
├── pyproject.toml              # Poetry Config
├── noxfile.py                  # Session Management (Story 1.6)
└── README.md
```

- **Src Layout:** The package MUST use `src/ncaa_eval/` layout — NOT a flat `ncaa_eval/` at root. This is a deliberate architecture decision per Section 2.3 and Section 1.1.
- **Monolithic Package:** All logic lives in `ncaa_eval`. The dashboard is separate in `dashboard/`. [Source: docs/specs/05-architecture-fullstack.md#Section 2.5]
- **No Direct IO in UI:** Dashboard must call `ncaa_eval` functions — never read files directly. [Source: docs/specs/05-architecture-fullstack.md#Section 12]

### Technical Requirements

- **Python Version:** 3.12+ (strict). Set in `pyproject.toml` as `python = ">=3.12,<4.0"`.
- **Package Manager:** Poetry (latest). [Source: docs/specs/05-architecture-fullstack.md#Section 3]
- **Strict Typing:** `mypy --strict` compliance is mandatory from day one. All `__init__.py` files should have `py.typed` marker or appropriate typing setup. [Source: docs/specs/05-architecture-fullstack.md#Section 12]

### Library / Framework Requirements

All dependencies from PRD Section 4 and Architecture Section 3:

**Core Dependencies:**

| Package | Purpose | Source |
|---|---|---|
| pandas | DataFrame manipulation | PRD Section 4 |
| numpy | Vectorized operations (NFR1) | PRD Section 4 |
| xgboost | Gradient boosting models (FR6) | PRD Section 4 |
| scikit-learn | ML utilities, metrics (FR7) | PRD Section 4 |
| networkx | Graph features (FR5) | PRD Section 4 |
| joblib | Parallel execution (NFR2) | PRD Section 4 |
| plotly | Interactive visualization | PRD Section 4 |
| streamlit | Dashboard UI | PRD Section 4 |

**Dev Dependencies:**

| Package | Purpose | Source |
|---|---|---|
| pytest | Testing framework | PRD Section 4 |
| hypothesis | Property-based testing | PRD Section 4 |
| mutmut | Mutation testing | PRD Section 4 |
| ruff | Linting/formatting | PRD Section 4 |
| mypy | Strict type checking | PRD Section 4 |
| pre-commit | Automated checking | PRD Section 4 |
| nox | Session management | PRD Section 4 |
| commitizen | Versioning | PRD Section 4 |
| check-manifest | Manifest checking | PRD Section 4 |
| edgetest | Dependency management | PRD Section 4 |
| sphinx | Documentation | PRD Section 4 |
| furo | Sphinx theme | PRD Section 4 |

**Do NOT pin to specific versions** — use `latest` per Architecture Section 3. Let Poetry resolve compatible versions.

### File Structure Requirements

- Every subdirectory under `src/ncaa_eval/` MUST have an `__init__.py` (even if empty) for proper package discovery.
- `dashboard/pages/` files follow Streamlit multipage naming convention: `1_Lab.py`, `2_Presentation.py` (number prefix for ordering).
- `data/` should contain a `.gitkeep` file so the empty directory is tracked by git, while the `.gitignore` excludes everything else in `data/`.
- `tests/` directory should have `__init__.py` for pytest discovery.

### Testing Requirements

- After `poetry install`, verify:
  1. `poetry run python -c "import ncaa_eval"` exits 0
  2. All subdirectory packages are importable
- No unit tests required for this story — testing framework is configured in Story 1.5.

### Project Structure Notes

- This is a **greenfield project** — no existing code to worry about. [Source: docs/specs/05-architecture-fullstack.md#Section 1.1]
- The `docs/` directory already exists with specs. Do NOT overwrite existing content.
- The `_bmad/` and `_bmad-output/` directories are BMAD framework — leave them untouched.
- The existing `.gitignore` may need to be extended, not replaced.

### References

- [Source: docs/specs/05-architecture-fullstack.md#Section 1.1] — Greenfield project, Poetry, src layout
- [Source: docs/specs/05-architecture-fullstack.md#Section 2.3] — Repository structure (src layout)
- [Source: docs/specs/05-architecture-fullstack.md#Section 3] — Complete tech stack with versions
- [Source: docs/specs/05-architecture-fullstack.md#Section 9] — Unified project structure (canonical directory tree)
- [Source: docs/specs/05-architecture-fullstack.md#Section 12] — Coding standards (strict typing, no direct IO in UI)
- [Source: docs/specs/03-prd.md#Section 4] — Technical assumptions & constraints (all dependencies)
- [Source: docs/specs/03-prd.md#Section 6] — Success metric SM3: clone-to-pipeline in under 3 commands
- [Source: _bmad-output/planning-artifacts/epics.md#Story 1.1] — Story acceptance criteria

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6 (claude-opus-4-6)

### Debug Log References

- `sphinx-apidoc` is not a standalone PyPI package; it is bundled with `sphinx`. Removed from dev dependencies.
- System Python was 3.10.14 (conda default); switched Poetry to use system Python 3.14.2 via `poetry env use python`.
- Removed legacy setuptools artifacts: `setup.py`, `src/ncaa_eval/version.py`, `src/ncaa_eval/ncaa_eval.py`, `ncaa_eval.egg-info/`, `src/ncaa_eval.egg-info/`.
- **Code Review Fixes (AI-Review):** Fixed mypy strict mode configuration, added py.typed marker, updated tests/__init__.py, added .claude/ to .gitignore, updated File List to document all changes including sprint-status.yaml and story file.

### Completion Notes List

- Converted project from setuptools to Poetry with src layout
- All 8 core dependencies and 12 dev dependencies installed (sphinx-apidoc excluded as it's part of sphinx)
- Created full directory scaffolding matching Architecture Section 9
- All subpackages importable: ncaa_eval.{ingest, transform, model, evaluation, utils}
- Dashboard scaffolding with Streamlit multipage convention (1_Lab.py, 2_Presentation.py)
- Extended existing .gitignore with data/, .ruff_cache/, and .claude/ exclusions
- Preserved existing tool configurations (ruff, mypy, pytest, coverage, commitizen) during pyproject.toml rewrite
- **Code Review Fixes:** Configured mypy in strict mode per Architecture Section 12, added py.typed marker for PEP 561 compliance, updated tests/__init__.py with proper content, comprehensive File List documentation

### File List

- pyproject.toml (modified — rewritten from setuptools to Poetry, configured mypy strict mode)
- poetry.lock (new — generated by poetry install)
- .gitignore (modified — added data/, .ruff_cache/, and .claude/ exclusions)
- src/ncaa_eval/__init__.py (modified — added docstring and future annotations)
- src/ncaa_eval/py.typed (new — PEP 561 type marker for mypy strict compliance)
- src/ncaa_eval/ingest/__init__.py (new)
- src/ncaa_eval/transform/__init__.py (new)
- src/ncaa_eval/model/__init__.py (new)
- src/ncaa_eval/evaluation/__init__.py (new)
- src/ncaa_eval/utils/__init__.py (new)
- dashboard/app.py (new)
- dashboard/pages/1_Lab.py (new)
- dashboard/pages/2_Presentation.py (new)
- dashboard/components/__init__.py (new)
- tests/__init__.py (modified — added docstring and future annotations)
- data/.gitkeep (new)
- _bmad-output/implementation-artifacts/1-1-initialize-repository-package-structure.md (modified — added Dev Agent Record)
- _bmad-output/implementation-artifacts/sprint-status.yaml (modified — set story status to 'review')
- .claude/settings.local.json (new — IDE config, git-ignored)
- setup.py (deleted — replaced by Poetry)
- src/ncaa_eval/version.py (deleted — no longer needed)
- src/ncaa_eval/ncaa_eval.py (deleted — empty placeholder)
- ncaa_eval.egg-info/ (deleted — setuptools artifact)
- src/ncaa_eval.egg-info/ (deleted — setuptools artifact)

## Change Log

- 2026-02-15: Story 1.1 implemented — Poetry-managed project with src layout, full directory scaffolding, all dependencies installed
