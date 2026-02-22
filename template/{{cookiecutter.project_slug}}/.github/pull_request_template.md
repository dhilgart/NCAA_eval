## PR Title
<!-- Use conventional commit format: type(scope): description -->

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

## Supporting Evidence
<!-- Where appropriate, include evidence that the change works as intended: -->
<!-- - Test output / coverage diff -->
<!-- - Benchmark results (before/after) for performance changes -->
<!-- - Screenshots for UI changes -->

## Pre-Commit Checks

- [ ] **Lint pass** --- `ruff check .` reports no errors
- [ ] **Format pass** --- `ruff format --check .` reports no changes needed
- [ ] **Type-check pass** --- `mypy` reports no errors (strict mode)
- [ ] **Package manifest** --- `check-manifest` reports no missing files
- [ ] **Smoke tests** --- `pytest -m smoke` passes
- [ ] **Commit messages** --- All commits use conventional format (`feat:`, `fix:`, `docs:`, etc.)

## Test Suite

- [ ] **All tests pass** --- `pytest` exits with code 0, no failures or errors
- [ ] **No regressions** --- Existing tests still pass after changes
- [ ] **New tests added** --- New functionality has corresponding unit tests

## Code Quality

- [ ] **Docstrings** --- New/changed public APIs have accurate Google-style docstrings
- [ ] **Type annotations** --- All function signatures, return types, and variables are annotated
- [ ] **Naming conventions** --- Follows project standards (STYLE_GUIDE.md Section 2)
- [ ] **Import ordering** --- stdlib / third-party / local groups; `from __future__ import annotations` first
- [ ] **Line length** --- Lines stay within 110 characters

## Documentation

- [ ] **README** --- Updated if project setup, usage, or public-facing behavior changed
- [ ] **CHANGELOG** --- Entry added for user-visible changes

## Related Issue
<!-- If applicable, reference the issue: Closes #123 -->
