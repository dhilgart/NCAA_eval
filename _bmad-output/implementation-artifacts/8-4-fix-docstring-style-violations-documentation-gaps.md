# Story 8.4: Fix Docstring Style Violations & Documentation Gaps

Status: ready-for-dev

## Story

As a developer,
I want all docstrings to follow the mandated Google style, all functions with 3+ operations to have detailed descriptions, and documentation gaps (troubleshooting, license, tutorial accuracy) to be resolved,
so that the codebase is internally consistent, Sphinx API docs render correctly, and users have accurate, complete documentation.

## Acceptance Criteria

### AC1: Convert NumPy-Style Docstrings to Google Style

1. All NumPy-style docstrings are converted to Google-style in these 5 modules:
   - `src/ncaa_eval/evaluation/metrics.py` — entire module
   - `src/ncaa_eval/transform/elo.py` — class docstring, `update_game`, `process_season`
   - `src/ncaa_eval/model/elo.py` — `set_state`, `load`
   - `src/ncaa_eval/model/base.py` — `_to_games`
   - `src/ncaa_eval/model/tracking.py` — `load_run`, `load_predictions`
2. Conversion is mechanical: `Parameters\n----------` → `Args:`, `Returns\n-------` → `Returns:`, `Raises\n------` → `Raises:`, remove dashed underlines, indent content under section headers
3. Type annotations are NOT duplicated in docstrings (types live in signatures only per STYLE_GUIDE Section 1)

### AC2: Add Missing `Returns:` Section

4. `_resolve_team_id` in `src/ncaa_eval/ingest/connectors/espn.py` has a `Returns:` section in its docstring

### AC3: Add Detailed Descriptions to Functions with 3+ Operations

5. All 28 functions listed in `_bmad-output/planning-artifacts/noncompliant-docstrings.md` are updated to include a detailed description paragraph (after the summary line) explaining *how* the function implements its purpose — not just restating the summary
6. The detailed description follows the STYLE_GUIDE Section 1 rule: "When a function performs 3 or more operations, the description paragraph must explain *how* the function implements its purpose"

### AC4: Fix All Ruff D-Rule Violations

7. All 25 `D107` violations are resolved — `__init__` methods that need docstrings get them; those that are trivial (just `self.x = x` assignments with class-level docs) get `D107` added to per-file ignores only if the class docstring already documents attributes
8. All 25 `D416` violations are auto-fixed (`ruff check --fix --select D416`)
9. All 13 `D102` violations are resolved — missing public method docstrings are added
10. All 2 `D411` violations are auto-fixed (`ruff check --fix --select D411`)
11. `ruff check src/ --select D` reports zero violations after all fixes

### AC5: Update Tutorial Expected Output

12. `docs/tutorials/getting-started.md` expected CLI output is updated to match actual `sync.py` and training CLI output
13. The tutorial is manually verified by reading the actual commands and comparing to documented output

### AC6: Add Troubleshooting Section to User Guide

14. A "Troubleshooting" section is added to `docs/user-guide.md` covering at minimum:
    - Kaggle authentication setup (`kaggle.json` placement, permissions)
    - ESPN rate limits and transient failures (retry behavior from Story 8.3)
    - Parquet version mismatches (arrow/pyarrow compatibility)
    - Common `poetry install` / conda env issues

### AC7: Add License Section to README

15. `README.md` has a `## License` section referencing GPL-3.0 (matching the `LICENSE` file in repo root)

### AC8: Fix Non-Existent Feature Documentation

16. The game theory sliders section in `docs/user-guide.md` (lines ~527-575) is prominently marked as "NOT YET IMPLEMENTED" with a visible admonition block — the current small `{note}` is insufficient
17. The inline reference to "Game Theory Sliders (planned feature)" at line ~266 in the reliability diagram table is updated to say "(not yet implemented)"

### AC9: Fix Dashboard Sidebar CLI Reference

18. `dashboard/app.py:69` is updated from `run 'python sync.py' first` to reference the canonical CLI command (e.g., `run 'ncaa-eval sync' first` or mention both options)

### AC10: Enable D-Rule Enforcement in pyproject.toml

19. The `"D1"` entry is removed from the Ruff `ignore` list in `pyproject.toml` — docstring presence is now enforced
20. `"D107"` is added to per-file ignores ONLY for files where `__init__` docstrings are genuinely unnecessary (class docstring covers the contract)
21. `"D415"` (first line punctuation) remains in the ignore list — this is a stylistic choice already documented

### AC11: Quality Gates

22. `ruff check .` passes (zero violations in `src/` and `tests/`)
23. `mypy --strict src/ncaa_eval tests` passes
24. All existing tests pass (`pytest`)
25. No behavioral code changes — this story is documentation-only for source code files

## Tasks / Subtasks

- [x] Task 1: Auto-fix trivial D-rule violations (AC: #8, #10)
  - [x] 1.1 Run `ruff check src/ --fix --select D416,D411` to auto-fix 27 violations
  - [x] 1.2 Verify fixes with `ruff check src/ --select D416,D411`

- [x] Task 2: Convert NumPy-style docstrings to Google style (AC: #1-3)
  - [x] 2.1 `src/ncaa_eval/evaluation/metrics.py` — convert all `Parameters`/`Returns`/`Raises` sections
  - [x] 2.2 `src/ncaa_eval/transform/elo.py` — convert class docstring, `update_game`, `process_season`
  - [x] 2.3 `src/ncaa_eval/model/elo.py` — convert `set_state`, `load`
  - [x] 2.4 `src/ncaa_eval/model/base.py` — convert `_to_games`
  - [x] 2.5 `src/ncaa_eval/model/tracking.py` — convert `load_run`, `load_predictions`
  - [x] 2.6 Verify no types are duplicated in docstrings (types in signatures only)

- [x] Task 3: Add missing `Returns:` section (AC: #4)
  - [x] 3.1 Add `Returns:` section to `_resolve_team_id` in `espn.py`

- [x] Task 4: Fix D107 violations — add `__init__` docstrings (AC: #7)
  - [x] 4.1 Run `ruff check src/ --select D107` to list all 25 violations
  - [x] 4.2 For each violation: if the class docstring documents constructor args → add per-file `D107` ignore; otherwise add `__init__` docstring
  - [x] 4.3 Verify with `ruff check src/ --select D107`

- [ ] Task 5: Fix D102 violations — add missing public method docstrings (AC: #9)
  - [ ] 5.1 Run `ruff check src/ --select D102` to list all 13 violations
  - [ ] 5.2 Add Google-style docstrings to each undocumented public method
  - [ ] 5.3 Verify with `ruff check src/ --select D102`

- [ ] Task 6: Add detailed descriptions to 28 noncompliant functions (AC: #5-6)
  - [ ] 6.1 Read `_bmad-output/planning-artifacts/noncompliant-docstrings.md` for the full list
  - [ ] 6.2 For each function: read the implementation, then add a description paragraph explaining *how* it works
  - [ ] 6.3 Focus on implementation approach, not restating the summary (e.g., "Iterates over games in date order, applies K-factor scaling, then updates both teams' ratings in-place" — not "Updates Elo ratings for a game")

- [ ] Task 7: Update pyproject.toml D-rule enforcement (AC: #19-21)
  - [ ] 7.1 Remove `"D1"` from the `ignore` list
  - [ ] 7.2 Add `"D107"` to per-file-ignores for files where class docstrings already cover `__init__`
  - [ ] 7.3 Keep `"D415"` in the ignore list
  - [ ] 7.4 Run `ruff check src/ --select D` to verify zero violations

- [ ] Task 8: Update tutorial expected output (AC: #12-13)
  - [ ] 8.1 Read `docs/tutorials/getting-started.md`
  - [ ] 8.2 Compare documented CLI output against actual command outputs
  - [ ] 8.3 Update expected output blocks to match reality

- [ ] Task 9: Add troubleshooting section to user guide (AC: #14)
  - [ ] 9.1 Add `## Troubleshooting` section to `docs/user-guide.md`
  - [ ] 9.2 Cover: Kaggle auth, ESPN rate limits, Parquet version issues, conda/poetry setup

- [ ] Task 10: Add license section to README (AC: #15)
  - [ ] 10.1 Add `## License` section referencing GPL-3.0 to `README.md`

- [ ] Task 11: Fix game theory sliders documentation (AC: #16-17)
  - [ ] 11.1 Replace the `{note}` admonition in `docs/user-guide.md` (~line 529) with a prominent `{warning}` or `{admonition}` block clearly stating "NOT YET IMPLEMENTED"
  - [ ] 11.2 Update the reliability diagram table reference (~line 266) from "(planned feature)" to "(not yet implemented)"

- [ ] Task 12: Fix dashboard sidebar CLI reference (AC: #18)
  - [ ] 12.1 Update `dashboard/app.py:69` to reference `ncaa-eval sync` instead of `python sync.py`

- [ ] Task 13: Quality gate validation (AC: #22-25)
  - [ ] 13.1 `ruff check .` — zero violations
  - [ ] 13.2 `mypy --strict src/ncaa_eval tests` — zero errors
  - [ ] 13.3 `pytest` — all tests pass
  - [ ] 13.4 Verify no behavioral code changes (documentation only in `.py` files)

## Dev Notes

### Key Principle

This story addresses **Category 3 findings** from the codebase audit — items that are obviously wrong and need fixing. The scope is documentation-only: no behavioral code changes, only docstrings, comments, markdown files, and one string literal in the dashboard.

### Docstring Convention: Google Style (Mandatory)

The project mandates Google-style docstrings per `docs/STYLE_GUIDE.md` Section 1 and `[tool.ruff.lint.pydocstyle] convention = "google"` in `pyproject.toml`.

**Google-style format:**
```python
def example(name: str, count: int) -> list[str]:
    """One-line summary (imperative mood).

    Detailed description explaining *how* this function works when
    it performs 3+ operations.

    Args:
        name: Description without type (type lives in annotation).
        count: Description without type.

    Returns:
        Description of return value without type.

    Raises:
        ValueError: When count is negative.
    """
```

**Common NumPy-to-Google conversion:**
- `Parameters\n----------\nparam : type\n    Description` → `Args:\n    param: Description`
- `Returns\n-------\ntype\n    Description` → `Returns:\n    Description`
- `Raises\n------\nExceptionType\n    Description` → `Raises:\n    ExceptionType: Description`
- Remove all `---` underlines
- Remove type info from docstring sections (types are in annotations)

### Known Gotcha: Dev Agents Default to NumPy Style

From `template-requirements.md` (2026-02-23): Dev agents default to NumPy docstring style when writing docstrings for data-science-oriented code. The Google convention in `[tool.ruff.lint.pydocstyle]` only activates when `D` rules are in `extend-select`. **All new docstrings in this story MUST use Google style.**

### D107 (`__init__`) Strategy

Not every `__init__` needs a docstring. The Ruff D107 rule requires docstrings on all `__init__` methods, but the Google style guide and project convention allow the class docstring to cover the constructor contract. Strategy:

1. If the class docstring documents `Args:` for constructor parameters → suppress D107 via per-file-ignores
2. If the class has no constructor docs and `__init__` has non-trivial logic → add `__init__` docstring
3. Pydantic models and dataclasses: `__init__` is auto-generated → always suppress D107

### Noncompliant Docstrings Reference

The full list of 28 functions needing detailed descriptions is in `_bmad-output/planning-artifacts/noncompliant-docstrings.md`. Key concentrations:
- `transform/feature_serving.py` — 8 functions (batch feature computation)
- `evaluation/simulation.py` — 5 functions (including nested `_traverse` closures)
- `ingest/connectors/` — 5 functions (parsing pipelines)
- `model/` — 4 functions (base, elo, tracking)

### Dashboard Sidebar Fix

The sidebar message in `dashboard/app.py:69` currently says:
```python
st.info("No data available — run `python sync.py` first")
```
Update to reference the canonical CLI:
```python
st.info("No data available — run `ncaa-eval sync` first")
```

### Game Theory Sliders Documentation

The user guide has ~50 lines (527-575) describing game theory sliders that don't exist. The current `{note}` admonition is easily missed. Replace with a prominent `{warning}`:
```markdown
```{warning}
**NOT YET IMPLEMENTED** — Game Theory Sliders are a planned feature based on
research from Story 7.7. The design below describes intended behavior that is
not yet available in the dashboard.
```
```

### Files to Modify

| File | Changes |
|------|---------|
| `src/ncaa_eval/evaluation/metrics.py` | NumPy → Google docstrings, add detailed descriptions |
| `src/ncaa_eval/transform/elo.py` | NumPy → Google docstrings |
| `src/ncaa_eval/model/elo.py` | NumPy → Google docstrings, add detailed descriptions |
| `src/ncaa_eval/model/base.py` | NumPy → Google docstrings, add detailed descriptions |
| `src/ncaa_eval/model/tracking.py` | NumPy → Google docstrings, add detailed descriptions |
| `src/ncaa_eval/ingest/connectors/espn.py` | Add Returns section, add detailed descriptions |
| `src/ncaa_eval/ingest/connectors/kaggle.py` | Add detailed descriptions |
| `src/ncaa_eval/evaluation/simulation.py` | Add detailed descriptions |
| `src/ncaa_eval/transform/feature_serving.py` | Add detailed descriptions |
| `src/ncaa_eval/transform/opponent.py` | Add detailed descriptions |
| `src/ncaa_eval/cli/main.py` | Add detailed description |
| ~20 other `src/` files | D107/D102 fixes (add `__init__` or method docstrings) |
| `pyproject.toml` | Remove `"D1"` from ignore, add per-file-ignores for D107 |
| `docs/user-guide.md` | Troubleshooting section, game theory sliders warning |
| `docs/tutorials/getting-started.md` | Update expected CLI output |
| `README.md` | Add License section |
| `dashboard/app.py` | Update sidebar CLI reference string |

### Files NOT Modified

| File | Reason |
|------|--------|
| `tests/` source files | Tests are not documentation targets (D-rules are for `src/` only) |
| `notebooks/` | EDA notebooks are exempt from docstring mandates |
| `template/` | Cookiecutter template files are excluded from all hooks |

### Previous Story Learnings (Stories 8.1–8.3)

- **Google-style docstrings enforced** — the project already has the convention configured, just not enforced. Story 8.4 completes the enforcement loop.
- **`# noqa: BLE001`** annotations: Keep them on all intentional broad exception handlers (from Story 8.3)
- **Dashboard files are NOT under `mypy --strict`** — don't add type annotations to dashboard files as part of this story
- **`from __future__ import annotations`** required in ALL Python files (Ruff-enforced) — already present in all src/ files

### Git Intelligence

Last 3 commits (Stories 8.1–8.3) were pure refactoring focused on code architecture and pipeline resilience. The codebase is stable — no active feature work. Story 8.4 is safe to modify docstrings across all modules without conflict risk.

### Source Document References

- [Source: `_bmad-output/planning-artifacts/codebase-audit-report.md` — Finding 3.16 (NumPy-style docstrings)]
- [Source: `_bmad-output/planning-artifacts/codebase-audit-report.md` — Finding 3.17 (tutorial inaccuracy)]
- [Source: `_bmad-output/planning-artifacts/codebase-audit-report.md` — Finding 3.18 (no troubleshooting)]
- [Source: `_bmad-output/planning-artifacts/codebase-audit-report.md` — Finding 3.19 (no license in README)]
- [Source: `_bmad-output/planning-artifacts/codebase-audit-report.md` — Finding 3.20 (missing Returns)]
- [Source: `_bmad-output/planning-artifacts/codebase-audit-pass2-addendum.md` — 2.19→3 (non-existent feature docs)]
- [Source: `_bmad-output/planning-artifacts/codebase-audit-pass2-addendum.md` — P2-8 (sidebar CLI reference)]
- [Source: `_bmad-output/planning-artifacts/noncompliant-docstrings.md` — 28 functions needing detailed descriptions]
- [Source: `_bmad-output/planning-artifacts/epic-8-codebase-improvements.md` — Story 8.4 section]
- [Source: `docs/STYLE_GUIDE.md` — Section 1 (Google-style docstring mandate)]
- [Source: `_bmad-output/planning-artifacts/template-requirements.md` — NumPy-style drift warning]

## Dev Agent Record

### Agent Model Used

{{agent_model_name_version}}

### Debug Log References

### Completion Notes List

### File List
