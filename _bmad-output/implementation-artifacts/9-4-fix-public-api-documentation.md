# Story 9.4: Fix Public API Documentation

Status: ready-for-dev

## Story

As a **developer**,
I want **accurate documentation of import paths for the ncaa_eval package**,
so that **the Style Guide matches reality and I know how to import public symbols**.

## Acceptance Criteria

1. **Given** the Style Guide (Section 8, line 875-876) claims `from ncaa_eval import EloModel` should work
   **When** the developer reads the Style Guide
   **Then** documented import paths match actual importable paths
   **And** the Style Guide is updated to document the actual submodule import paths (e.g., `from ncaa_eval.model import get_model`)

2. **Given** the Style Guide (Section 3, lines 132-133) shows example imports `from ncaa_eval.features import rolling_efficiency` and `from ncaa_eval.models.base import ModelBase`
   **When** the developer copies these import examples
   **Then** the examples use real module paths (`ncaa_eval.transform`, `ncaa_eval.model.base`) and real symbol names

3. **Given** documentation files may reference import paths
   **When** the developer scans all docs for import examples
   **Then** every import example uses a path that actually resolves

## Tasks / Subtasks

- [ ] Task 1: Fix Style Guide Section 8 — `__init__.py` re-exports rule (AC: #1)
  - [ ] 1.1: Update line 875-876 in `docs/STYLE_GUIDE.md` — change `from ncaa_eval import EloModel` to the correct submodule import convention: `from ncaa_eval.model import Model` (base classes and registry functions are importable from `ncaa_eval.model`; concrete models use `get_model("elo")` or direct imports like `from ncaa_eval.model.elo import EloModel`)
  - [ ] 1.2: Clarify that `__init__.py` re-exports are at the *submodule* level (e.g., `ncaa_eval.model`, `ncaa_eval.ingest`, `ncaa_eval.evaluation`, `ncaa_eval.transform`, `ncaa_eval.utils`), not at the root `ncaa_eval` level

- [ ] Task 2: Fix Style Guide Section 3 — import ordering example (AC: #2)
  - [ ] 2.1: Replace `from ncaa_eval.features import rolling_efficiency` with a real import, e.g., `from ncaa_eval.transform import compute_rolling_stats`
  - [ ] 2.2: Replace `from ncaa_eval.models.base import ModelBase` with the real import `from ncaa_eval.model import Model`

- [ ] Task 3: Audit all docs for incorrect import paths (AC: #3)
  - [ ] 3.1: Scan all `.md` and `.rst` files in `docs/` for `from ncaa_eval` patterns
  - [ ] 3.2: Verify each documented import actually resolves — check against `__all__` exports in each `__init__.py`
  - [ ] 3.3: Fix any additional incorrect paths found
  - [ ] 3.4: Verify tutorials (`docs/tutorials/`) import paths are correct (based on prior analysis, these appear correct already)

- [ ] Task 4: Run tests to confirm no regressions (AC: all)
  - [ ] 4.1: Run `pytest` (full suite)
  - [ ] 4.2: Run `ruff check .`
  - [ ] 4.3: Run `mypy --strict src/ncaa_eval tests`

## Dev Notes

### Scope — Documentation-Only Changes

This story is **documentation-only**. No Python source code changes are needed. The `__init__.py` files and public API surface are correct — only the docs that describe them have errors.

### Specific Discrepancies Found

#### Discrepancy 1: Style Guide Section 8, Rule 3 (line 875-876)
- **Current text:** `__init__.py re-exports. Public symbols should be importable from the package level: from ncaa_eval import EloModel.`
- **Reality:** `ncaa_eval/__init__.py` is empty (contains only a docstring and `from __future__ import annotations`). There are no root-level re-exports. `EloModel` is not even in `ncaa_eval.model.__all__` — it auto-registers via `@register_model("elo")` and is accessed via the registry (`get_model("elo")`) or direct import (`from ncaa_eval.model.elo import EloModel`).
- **Fix:** Update to describe the actual convention — submodule-level re-exports. Example: `from ncaa_eval.model import Model, get_model`.

#### Discrepancy 2: Style Guide Section 3, Import Example (lines 132-133)
- **Current text:**
  ```python
  from ncaa_eval.features import rolling_efficiency
  from ncaa_eval.models.base import ModelBase
  ```
- **Reality:** `ncaa_eval.features` does not exist (the module is `ncaa_eval.transform`). `ncaa_eval.models` does not exist (singular: `ncaa_eval.model`). `ModelBase` does not exist (the class is `Model`). `rolling_efficiency` does not exist (closest: `compute_rolling_stats`).
- **Fix:** Replace with real imports:
  ```python
  from ncaa_eval.transform import compute_rolling_stats
  from ncaa_eval.model import Model
  ```

### Actual Import Architecture

The package uses **submodule-level re-exports** (NOT root-level):

| Submodule | `__all__` count | Example import |
|---|---|---|
| `ncaa_eval.model` | 7 | `from ncaa_eval.model import Model, get_model` |
| `ncaa_eval.ingest` | 14 | `from ncaa_eval.ingest import Game, ParquetRepository` |
| `ncaa_eval.evaluation` | 55 | `from ncaa_eval.evaluation import run_backtest, log_loss` |
| `ncaa_eval.transform` | 46 | `from ncaa_eval.transform import FeatureConfig` |
| `ncaa_eval.utils` | 6 | `from ncaa_eval.utils import get_logger` |

Concrete model classes (`EloModel`, `XGBoostModel`, `LogisticRegressionModel`) are NOT in `ncaa_eval.model.__all__`. They self-register on import and are accessed via:
- Registry: `get_model("elo")(config=...)`
- Direct: `from ncaa_eval.model.elo import EloModel`

### Files to Modify

- `docs/STYLE_GUIDE.md` — lines 132-133 (Section 3 import example) and lines 875-876 (Section 8, Rule 3)
- Possibly other doc files if audit (Task 3) reveals additional discrepancies

### Files NOT to Modify

- `src/ncaa_eval/__init__.py` — do NOT add re-exports to make the Style Guide's claim true; instead fix the docs to match reality
- `src/ncaa_eval/model/__init__.py` — do NOT add `EloModel` to `__all__`; the registry pattern is intentional
- Any other Python source files

### Previous Story Intelligence

Stories 9.1-9.3 established patterns:
- Code review feedback was systematically addressed
- Tests were comprehensive (13-18 new tests per story)
- Import paths in production code consistently use the submodule pattern (`from ncaa_eval.model import ...`)

Story 9.2 (Feature Config) noted an AI-Review follow-up: "Export FeatureConfig from public API" — this confirms the convention that public symbols are re-exported at the submodule level, not the root level.

### Testing Standards

- `pytest` — full suite must pass (currently ~980 tests)
- `ruff check .` — all linting must pass
- `mypy --strict src/ncaa_eval tests` — full type checking must pass
- Since this is docs-only, no new tests are needed, but existing tests must not regress

### Project Structure Notes

- Style Guide lives at `docs/STYLE_GUIDE.md`
- API reference RST files are auto-generated by Sphinx from docstrings in `docs/api/`
- Tutorials are at `docs/tutorials/` (getting-started, custom-model, custom-metric)
- Sphinx config at `docs/conf.py` uses autodoc + napoleon + myst_parser

### References

- [Source: docs/STYLE_GUIDE.md#Section-3 (lines 132-133)] — incorrect import example
- [Source: docs/STYLE_GUIDE.md#Section-8 (lines 875-876)] — incorrect `__init__.py` re-export claim
- [Source: src/ncaa_eval/__init__.py] — empty root package (no re-exports)
- [Source: src/ncaa_eval/model/__init__.py] — `__all__` has 7 items (no concrete model classes)
- [Source: _bmad-output/planning-artifacts/epics.md#Epic-9] — Story 9.4 requirements
- [Source: _bmad-output/implementation-artifacts/9-2-feature-config-as-model-level-concern.md] — AI-Review follow-up re: FeatureConfig public API export
- [Source: Audit item 2.18] — original finding

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
