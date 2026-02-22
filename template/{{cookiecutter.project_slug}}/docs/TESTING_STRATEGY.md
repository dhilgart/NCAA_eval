# {{ cookiecutter.project_name }} Testing Strategy

## Overview

This project uses a **4-dimensional testing model** with **4-tier quality gates**.

---

## 1. Four Dimensions of Testing

Tests are organized across four **orthogonal dimensions**:

1. **Scope** (What): Unit vs Integration
2. **Approach** (How): Example-based vs Property-based vs Fuzz-based
3. **Purpose** (Why): Functional vs Performance vs Regression
4. **Execution** (When): Tier 1 (pre-commit) vs Tier 2 (PR/CI)

---

## 2. Test Markers

8 pytest markers defined in `pyproject.toml`:

| Marker | Description |
|---|---|
| `smoke` | Pre-commit tests (< 10s total) |
| `slow` | Excluded from pre-commit (> 5s each) |
| `integration` | I/O or external dependencies |
| `property` | Hypothesis property-based tests |
| `fuzz` | Hypothesis fuzz-based tests for crash resilience |
| `performance` | Speed/efficiency compliance |
| `regression` | Prevent bug recurrence |
| `mutation` | Mutation testing coverage |
| `no_mutation` | Tests incompatible with mutmut runner directory |

---

## 3. Quality Gates (4-Tier Model)

### Tier 1: Pre-commit (< 10 seconds)
- Ruff lint + format check
- Mypy strict type checking
- Smoke tests (`pytest -m smoke`)
- check-manifest

### Tier 2: PR/CI (minutes)
- Full test suite with coverage
- Integration tests
- Property-based tests (Hypothesis)
- Coverage report

### Tier 3: AI Code Review
- Architecture alignment
- Test quality assessment
- Docstring quality

### Tier 4: Owner Review
- Functional correctness
- Strategic alignment

---

## 4. Coverage Targets

| Scope | Line | Branch |
|---|---|---|
| Overall project | 80% | 75% |
| Critical modules | 95% | 90% |

Coverage is a **signal, not a gate** --- it identifies gaps but does not block PRs.

---

## 5. Mutation Testing

**Tool:** mutmut 3.x

Mutation testing verifies that tests actually detect code changes. Configure
target modules in `[tool.mutmut]` in `pyproject.toml`.

**Important notes:**
- mutmut 3.x is **Windows-incompatible** (requires `resource` stdlib module). Use WSL or Linux.
- mutmut creates `mutants/` directory and `.mutmut-cache` file (both in `.gitignore`).
- Tests using `Path(__file__)` for project root navigation break under mutmut. Mark with `@pytest.mark.no_mutation`.

---

## 6. Test Organization

```
tests/
  __init__.py
  conftest.py         # Shared fixtures
  fixtures/           # Test data files
  unit/               # Unit tests (fast, isolated)
    __init__.py
    test_package.py   # Smoke tests
  integration/        # Integration tests (I/O, external deps)
    __init__.py
```

### Conventions

- **Mirror `src/` in `tests/`**: `src/pkg/module.py` is tested by `tests/unit/test_module.py`
- **Use `return` not `yield`** in fixtures without teardown (Ruff PT022)
- **Test behavior, not internals**: Don't import private symbols from implementation
- **Vectorized assertions**: Use `np.testing.assert_allclose` or `pytest.approx` instead of loops

---

## 7. Property-Based Testing (Hypothesis)

Use Hypothesis for invariant testing of pure functions:

```python
from hypothesis import given
from hypothesis import strategies as st

@pytest.mark.property
@given(st.integers(-1000, 1000))
def test_result_always_bounded(value: int) -> None:
    result = compute(value)
    assert 0 <= result <= 1
```

---

## References

- `pyproject.toml` --- All test configuration
- `docs/STYLE_GUIDE.md` --- Coding standards
