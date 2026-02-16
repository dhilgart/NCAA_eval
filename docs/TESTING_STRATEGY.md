# Testing Strategy

**Quick Reference** for the `ncaa_eval` project testing approach. For detailed explanations and examples, see the [testing guides](testing/).

---

## Table of Contents

1. [Overview](#overview)
   - [Key Principles](#key-principles)
   - [Four Orthogonal Dimensions](#four-orthogonal-dimensions)
   - [Execution Tiers (When Checks Run)](#execution-tiers-when-checks-run)
     - [Tier 1: Pre-Commit](#tier-1-pre-commit--10s-total)
     - [Tier 2: PR/CI](#tier-2-prci-minutes)
     - [Tier 3: AI Code Review](#tier-3-ai-code-review)
     - [Tier 4: Owner Review](#tier-4-owner-review)
2. [Detailed Guides](#detailed-guides)
3. [Quick Decision Trees](#quick-decision-trees)
   - [Which test scope?](#which-test-scope)
   - [Which approach?](#which-approach)
   - [Which execution tier?](#which-execution-tier)
4. [Test Markers Reference](#test-markers-reference)
5. [Test Commands Reference](#test-commands-reference)
6. [Test Organization](#test-organization)
7. [Coverage Targets](#coverage-targets)
8. [Testing Tools](#testing-tools)
9. [Domain-Specific Testing](#domain-specific-testing)
   - [Performance Testing (NFR1: Vectorization)](#performance-testing-nfr1-vectorization)
   - [Data Leakage Prevention (NFR4: Temporal Boundaries)](#data-leakage-prevention-nfr4-temporal-boundaries)
10. [References](#references)

---

## Overview

### Key Principles

1. ✅ **Fast feedback** via Tier 1 (pre-commit, < 10s total)
2. ✅ **Thorough validation** via Tier 2 (PR/CI, complete suite)
3. ✅ **Four orthogonal dimensions** - choose appropriate combination
4. ✅ **Coverage is a signal, not a gate** - identify gaps, don't block
5. ✅ **Mutation testing** evaluates test quality (critical modules only)
6. ✅ **Vectorization compliance** via performance testing (NFR1)
7. ✅ **Temporal integrity** via data leakage testing (NFR4)
8. ✅ **4-tier execution model** - Tier 1 (pre-commit) → Tier 2 (PR/CI) → Tier 3 (AI review) → Tier 4 (owner review)

### Four Orthogonal Dimensions

This strategy separates **four independent dimensions** of testing. Choose the appropriate combination for each test case:

1. **Test Scope** - *What* you're testing → [Scope Guide](testing/test-scope-guide.md)
   - **Unit:** Single function/class in isolation
   - **Integration:** Multiple components working together

2. **Test Approach** - *How* you write the test → [Approach Guide](testing/test-approach-guide.md)
   - **Example-based:** Concrete inputs → expected outputs
   - **Property-based (Hypothesis):** Invariants that should hold for all inputs
   - **Fuzz-based (Hypothesis):** Random/mutated inputs to find crashes and error handling gaps

3. **Test Purpose** - *Why* you're writing the test → [Purpose Guide](testing/test-purpose-guide.md)
   - **Functional:** Correctness of behavior (default)
   - **Performance:** Speed/efficiency compliance (NFR1: vectorization)
   - **Regression:** Prevent previously fixed bugs from recurring

4. **Execution Scope** - *When* tests/checks run → [Execution Guide](testing/execution.md)
   - **Tier 1 (Pre-commit):** Smoke tests + fast checks (< 10s total)
   - **Tier 2 (PR/CI):** Complete suite + coverage + mutation
   - **Tier 3/4:** AI + Owner review

**Note:** Mutation testing and coverage are **not test types** - they're quality assurance tools. See [Quality Assurance Guide](testing/quality.md).

### Execution Tiers (When Checks Run)

The project uses a **4-tier execution model** that balances speed with thoroughness:

#### Tier 1: Pre-Commit (< 10s total)

Fast, local checks that run on every commit:

| Check | Tool | What It Catches |
|-------|------|-----------------|
| Lint | `ruff check .` | Style violations, import issues |
| Format | `ruff format --check .` | Inconsistent formatting |
| Type-check | `mypy` (strict) | Missing annotations, type errors |
| Smoke tests | `pytest -m smoke` | Broken imports, sanity failures |
| Package manifest | `check-manifest` | Missing distribution files |

**Rationale:** Catch 80% of issues in seconds before code leaves your machine.

#### Tier 2: PR/CI (minutes)

Comprehensive validation before merge:

| Check | Tool | What It Catches |
|-------|------|-----------------|
| Full test suite | `pytest` | All regressions, edge cases |
| Integration tests | `pytest -m integration` | Component interaction failures |
| Property-based | `pytest -m property` | Invariant violations |
| Performance | `pytest -m performance` | Vectorization violations, speed regressions |
| Coverage | `pytest-cov` | Untested code paths |
| Mutation (Tier 1 modules) | `mutmut` | Weak tests, coverage gaps |

**Rationale:** Catch remaining 20% requiring full project context.

#### Tier 3: AI Code Review

Docstring quality, vectorization compliance, architecture alignment, test quality, design intent.

#### Tier 4: Owner Review

Functional correctness, strategic alignment, complexity appropriateness, scope creep prevention.

See [Execution Guide](testing/execution.md) for complete details on each tier.

---

## Detailed Guides

For comprehensive explanations, examples, and best practices:

- **[Test Scope Guide](testing/test-scope-guide.md)** - Unit vs Integration tests
- **[Test Approach Guide](testing/test-approach-guide.md)** - Example-based vs Property-based
- **[Test Purpose Guide](testing/test-purpose-guide.md)** - Functional, Performance, Regression
- **[Execution Guide](testing/execution.md)** - When tests/checks run (4-tier model)
- **[Quality Assurance Guide](testing/quality.md)** - Mutation testing, coverage analysis
- **[Conventions Guide](testing/conventions.md)** - Fixtures, markers, organization, coverage targets
- **[Domain Testing Guide](testing/domain-testing.md)** - Performance testing, Data leakage prevention

---

## Quick Decision Trees

### Which test scope?

```
Does it interact with external systems (files, database, network)?
├─ YES → Integration test (@pytest.mark.integration, PR-time only)
└─ NO  → Unit test (fast, pre-commit eligible if smoke)
```

### Which approach?

```
Are you testing error handling / crash resilience?
├─ YES → Fuzz-based (@pytest.mark.fuzz, Hypothesis st.text()/st.binary())
└─ NO  → Do you have specific known scenarios to verify?
          ├─ YES → Example-based (parametrize for multiple cases)
          └─ NO  → Can you state an invariant?
                    ├─ YES → Property-based (@pytest.mark.property, Hypothesis)
                    └─ NO  → Example-based (test specific examples)
```

### Which execution tier?

```
Is the test fast (< 1 second)?
├─ NO  → Tier 2 only (@pytest.mark.slow, @pytest.mark.integration, etc.)
└─ YES → Is it an import/sanity/schema check OR critical regression?
          ├─ YES → Tier 1 eligible (@pytest.mark.smoke)
          └─ NO  → Tier 2 only (save pre-commit budget)
```

---

## Test Markers Reference

| Marker | Dimension | Command |
|--------|-----------|---------|
| `@pytest.mark.smoke` | Speed |  `pytest -m smoke` |
| `@pytest.mark.slow` | Speed |  `pytest -m "not slow"` |
| `@pytest.mark.integration` | Scope |  `pytest -m integration` |
| `@pytest.mark.property` | Approach |  `pytest -m property` |
| `@pytest.mark.fuzz` | Approach |  `pytest -m fuzz` |
| `@pytest.mark.performance` | Purpose |  `pytest -m performance` |
| `@pytest.mark.regression` | Purpose |  `pytest -m regression` |
| `@pytest.mark.mutation` | Quality |  `pytest -m mutation` |


**Combine markers across dimensions:**
```python
@pytest.mark.integration
@pytest.mark.property
@pytest.mark.regression
```

---

## Test Commands Reference

| Context | Command | What Runs |
|---------|---------|-----------|
| **Tier 1 (Pre-commit)** | `pytest -m smoke` | Smoke tests only (< 5s total) |
| **Tier 2 (PR/CI - full)** | `pytest` | All tests |
| **Tier 2 (PR/CI - coverage)** | `pytest --cov=src/ncaa_eval --cov-report=term-missing` | All + coverage report |
| **Tier 2 (exclude slow)** | `pytest -m "not slow"` | All except slow tests |
| **Filter by dimension** | `pytest -m integration` | Filter by marker |
| **Combined filters** | `pytest -m "integration and regression"` | Intersection |

---

## Test Organization

```
tests/
├── conftest.py                   # Shared fixtures
├── unit/                         # Unit tests
│   ├── test_metrics.py
│   ├── test_elo.py
│   └── test_features.py
├── integration/                  # Integration tests
│   ├── test_sync_pipeline.py
│   └── test_training_pipeline.py
└── fixtures/                     # Test data files
    └── sample_games.csv
```

**Naming conventions:**
- Test files: `test_<module_name>.py`
- Test functions: `test_<function>_<scenario>()`
- Fixtures: `<resource>_fixture()`

See [Conventions Guide](testing/conventions.md) for details.

---

## Coverage Targets

| Module | Line | Branch | Rationale |
|--------|------|--------|-----------|
| `evaluation/metrics.py` | 95% | 90% | Critical - errors invalidate all evaluations |
| `evaluation/simulation.py` | 90% | 85% | Monte Carlo simulator |
| `model/` | 90% | 85% | Core abstraction |
| `transform/` | 85% | 80% | Feature correctness, leakage prevention |
| `ingest/` | 80% | 75% | Data quality |
| `utils/` | 75% | 70% | Lower priority |
| **Overall** | **80%** | **75%** | Balanced |

**Coverage is a signal, not a gate.** Use to identify gaps, not block PRs.

See [Conventions Guide](testing/conventions.md#coverage-targets) for details.

---

## Testing Tools

| Tool | Purpose | Configuration |
|------|---------|---------------|
| **Pytest** | Testing framework | `pyproject.toml` `[tool.pytest.ini_options]` |
| **Hypothesis** | Property-based + Fuzz testing | Dev dependency |
| **Mutmut** | Mutation testing (quality) | Dev dependency |
| **pytest-cov** | Coverage reporting | `[tool.coverage.report]` |
| **Nox** | Session orchestration | `noxfile.py` (Story 1.6) |

---

## Domain-Specific Testing

### Performance Testing (NFR1: Vectorization)

- **Smoke:** Assertion-based vectorization checks (< 1s)
- **PR-time:** Performance benchmarks, 60-second backtest target

```python
@pytest.mark.smoke
@pytest.mark.performance
def test_metrics_are_vectorized():
    """Quick check: no .iterrows() in metrics."""
    # See domain-testing.md for example
```

### Data Leakage Prevention (NFR4: Temporal Boundaries)

- **Smoke:** API contract unit tests (fast)
- **PR-time:** End-to-end workflow tests, property-based invariants

```python
@pytest.mark.smoke
def test_api_enforces_cutoff():
    """Quick check: API rejects future data."""
    # See domain-testing.md for example
```

See [Domain Testing Guide](testing/domain-testing.md) for comprehensive examples.

---

## References

- [`STYLE_GUIDE.md`](STYLE_GUIDE.md) - Coding standards, vectorization rule
- [`docs/specs/05-architecture-fullstack.md`](specs/05-architecture-fullstack.md) - Architecture, nox workflow
- [`docs/specs/03-prd.md`](specs/03-prd.md) - Non-functional requirements (NFR1-NFR5)
- `pyproject.toml` - Pytest configuration
- `.github/pull_request_template.md` - PR checklist
