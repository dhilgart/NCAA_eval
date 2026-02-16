## PR Title
<!-- Use conventional commit format: type(scope): description -->
<!-- Examples: feat(model): add Elo rating engine, fix(ingest): handle missing season data -->

## PR Type
<!-- Keep only the types that apply, delete the rest -->

- **Bugfix**
- **New feature**
- **Refactoring**
- **Breaking change** (any change that would cause existing functionality to not work as expected)
- **Documentation update**
- **Performance improvement**
- **Other (please describe)**

## Description
<!-- What does this PR do and why? Link to the relevant story/issue if applicable. -->

## Expected behavior
<!--A clear and concise description of what you expected to happen-->

## Steps to reproduce the behavior
<!--
Steps to reproduce the behavior:
1. ...
2. ...
3. ...
-->

## Supporting Evidence
<!-- Where appropriate, include evidence that the change works as intended: -->
<!-- - Test output / coverage diff -->
<!-- - Benchmark results (before/after) for performance changes -->
<!-- - Screenshots for UI changes -->
<!-- - Regression test for bug fixes (describe the scenario that was broken) -->

## Pre-Commit Checks

- [ ] **Lint pass** — `ruff check .` reports no errors
- [ ] **Format pass** — `ruff format --check .` reports no changes needed
- [ ] **Type-check pass** — `mypy` reports no errors (strict mode)
- [ ] **Package manifest** — `check-manifest` reports no missing files
- [ ] **Smoke tests** — `pytest -m smoke` passes (imports, sanity, schema contracts)
- [ ] **Commit messages** — All commits use conventional format (`feat:`, `fix:`, `docs:`, etc.)

## Test Suite

- [ ] **All tests pass** — `pytest` exits with code 0, no failures or errors
- [ ] **No regressions** — Existing tests still pass after changes
- [ ] **New tests added** — New functionality has corresponding unit tests
- [ ] **Edge compatibility** — `edgetest` passes (dependency version boundaries verified)

## Code Quality

- [ ] **Docstrings** — New/changed public APIs have accurate Google-style docstrings
- [ ] **Inline comments** — Non-obvious logic has explanatory comments
- [ ] **Type annotations** — All function signatures, return types, and variables are annotated
- [ ] **Naming conventions** — Follows project standards ([STYLE_GUIDE.md](../docs/STYLE_GUIDE.md) Section 2)
- [ ] **Import ordering** — stdlib / third-party / local groups; `from __future__ import annotations` first
- [ ] **No vectorization violations** — No `for` loops over DataFrames for metric calculations
- [ ] **Line length** — Lines stay within 110 characters

## Architecture Compliance

- [ ] **Type sharing** — Shared data structures use Pydantic (boundaries) or TypedDict (internal)
- [ ] **No direct IO in UI** — Dashboard code calls `ncaa_eval` functions, never reads files directly
- [ ] **No unnecessary dependencies** — No new dependencies added without team discussion

## Documentation
<!-- User-facing documentation only. Style guide changes require a standalone PR. -->

- [ ] **README** — Updated if project setup, usage, or public-facing behavior changed
- [ ] **User guide** — Updated if end-user workflows or features changed (see `docs/`)
- [ ] **Tutorials** — Updated or added if new features need step-by-step walkthroughs
- [ ] **Sphinx API docs** — Rebuilt and verified (`sphinx-build` succeeds with no warnings)
- [ ] **Architecture docs** — Specs updated if system design or data flow changed
- [ ] **CHANGELOG** — Entry added for user-visible changes

## Related Issue
<!-- If applicable, reference the issue: Closes #123 -->

---

**Reference:** [`docs/TESTING_STRATEGY.md`](../docs/TESTING_STRATEGY.md) for the testing strategy
and quality gates | [`docs/STYLE_GUIDE.md`](../docs/STYLE_GUIDE.md) for full
coding standards
