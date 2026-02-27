# Test Suite Quality Assurance

This guide explains **how to ensure test quality** through mutation testing and coverage analysis.

---

## Overview

Tools and techniques for **evaluating the quality and effectiveness of your test suite**. These are NOT test types - they are meta-testing practices that evaluate your existing tests.

---

## 1. Mutation Testing (Mutmut)

### What it is
A technique that evaluates test quality by mutating production code and verifying that tests fail.

### How it works
1. Mutmut introduces small changes to your code (e.g., `+` → `-`, `>` → `>=`, `True` → `False`)
2. Runs your existing test suite against each mutation
3. Verifies that at least one test fails for each mutation
4. **Surviving mutants** = mutations that didn't cause test failures = potential gaps in test coverage

### Purpose
Verify that your tests are effective at catching bugs, not just achieving high coverage.

### When to use
- **After initial test suite is written** - Mutation testing measures test quality, not code quality
- **Validating test quality for high-risk modules** - Ensure critical code has strong test coverage
- **Periodic quality audits** - Not every PR (too slow), but periodically to validate test effectiveness
- **Before major refactoring** - Ensure tests will catch regressions

### NOT a test type
Mutation testing doesn't write tests - it evaluates your existing tests.

### Characteristics
- **Extremely slow:** Mutates code and reruns entire test suite for each mutation
- **Quality signal:** Measures test effectiveness (do tests catch bugs?), not just coverage (are lines executed?)
- **Selective application:** Only run on designated high-priority modules
- **Orthogonal to test dimensions:** Works with any test scope, approach, or purpose

### Priority Tiers

| Tier | Modules | Frequency | Rationale |
|------|---------|-----------|-----------|
| **Tier 1 (Always)** | `evaluation/metrics.py`, `evaluation/simulation.py` | Every PR | Critical for correctness - errors invalidate all evaluations |
| **Tier 2 (Periodic)** | `model/`, `transform/` | Weekly or before release | Important but less critical than metrics |
| **Tier 3 (Rare)** | `ingest/`, `utils/` | Monthly or as needed | Lower priority, test coverage may suffice |

### Usage

```bash
# Run mutation testing on high-priority module
mutmut run --paths-to-mutate=src/ncaa_eval/evaluation/metrics.py

# Review results
mutmut results

# Show details of surviving mutants (gaps in test coverage)
mutmut show <mutant-id>

# Run only against fast tests (for quicker feedback)
mutmut run --paths-to-mutate=src/ncaa_eval/evaluation/metrics.py --runner="pytest -m smoke"
```

### Interpreting results

- **Mutation Score = (Killed Mutants / Total Mutants) × 100%**
- **Target:** 80%+ for Tier 1 modules, 70%+ for Tier 2
- **Surviving mutants** indicate:
  - Missing test cases
  - Weak assertions (e.g., `assert result is not None` instead of `assert result == expected`)
  - Dead code (code that can be changed without affecting behavior)

### Example surviving mutant (test gap)

```python
# Original code
def calculate_margin(home_score, away_score):
    return home_score - away_score

# Mutant (changed - to +)
def calculate_margin(home_score, away_score):
    return home_score + away_score  # Mutmut changed this

# If this mutant survives, it means no test verifies the margin calculation is correct!
# Solution: Add a test with known inputs/outputs
def test_calculate_margin_correctness():
    assert calculate_margin(100, 80) == 20  # Would catch the + mutation
```

### Marker
`@pytest.mark.mutation` - Tag tests that are good candidates for mutation testing

### Execution Tier
- **Tier 1 (Pre-commit):** ❌ NO (extremely slow)
- **Tier 2 (PR/CI):** ✅ YES, but only for designated Tier 1 modules

See [Execution Guide](execution.md) for details on execution tiers.

---

## 2. Coverage Analysis

### What it is
Measurement of what percentage of your code is executed during test runs.

### Types of coverage
- **Line coverage:** Percentage of code lines executed
- **Branch coverage:** Percentage of conditional branches (if/else) taken
- **Function coverage:** Percentage of functions called

### Purpose
Identify untested code paths, NOT a measure of test quality.

### Important distinction
- **High coverage ≠ Good tests:** You can have 100% coverage with weak assertions
- **Mutation testing** verifies test quality; **coverage** only verifies code execution

### Tool
`pytest-cov`

### Usage

```bash
# Generate coverage report
pytest --cov=src/ncaa_eval --cov-report=term-missing

# HTML report for detailed analysis
pytest --cov=src/ncaa_eval --cov-report=html
open htmlcov/index.html

# Branch coverage (more thorough than line coverage)
pytest --cov=src/ncaa_eval --cov-branch --cov-report=term-missing
```

### Coverage is a signal, not a gate
Use coverage to identify gaps, but don't block PRs solely for coverage numbers. A few well-written tests with 70% coverage are better than many weak tests with 100% coverage.

---

## 3. Combining Quality Assurance Tools

Use both mutation testing and coverage analysis together for comprehensive quality assessment:

### Workflow

1. **Write tests** for your feature
2. **Check coverage** - Identify untested code paths
3. **Add tests** to cover gaps
4. **Run mutation testing** - Verify tests actually catch bugs
5. **Fix surviving mutants** - Add stronger assertions or test cases

### Example - Comprehensive quality check

```bash
# Step 1: Run tests with coverage
pytest --cov=src/ncaa_eval/evaluation/metrics.py --cov-report=term-missing

# Coverage: 95% ✓ (high coverage)

# Step 2: Run mutation testing
mutmut run --paths-to-mutate=src/ncaa_eval/evaluation/metrics.py

# Mutation Score: 60% ✗ (many surviving mutants = weak tests!)

# Step 3: Review surviving mutants
mutmut results
mutmut show 42  # Mutation: changed `>` to `>=` and tests still passed

# Step 4: Add test to catch this mutation
@pytest.mark.mutation
def test_threshold_boundary():
    """Regression test: Ensure > is used, not >= (catches mutant #42)."""
    result = is_above_threshold(value=50, threshold=50)
    assert result is False  # Should be False because 50 is NOT > 50
```

---

## Key Principles

1. ✅ **Coverage shows what code runs** - It identifies untested code paths
2. ✅ **Mutation testing shows test effectiveness** - It verifies tests catch bugs
3. ✅ **Use both together** - Coverage finds gaps, mutation testing validates quality
4. ✅ **Selective mutation testing** - Only run on high-priority modules (too slow for everything)
5. ✅ **Coverage is a signal, not a gate** - Don't block PRs solely on coverage numbers
6. ✅ **Target 80%+ mutation score** - For critical modules (Tier 1)
7. ✅ **Weak assertions fail mutation tests** - Use specific assertions, not just `is not None`

---

## See Also

- [Test Scope Guide](test-scope-guide.md) - Unit vs Integration tests
- [Test Approach Guide](test-approach-guide.md) - Example-based vs Property-based
- [Test Purpose Guide](test-purpose-guide.md) - Functional, Performance, Regression
- [Execution Guide](execution.md) - When tests and checks run (4-tier model)
- [Conventions Guide](conventions.md) - Fixtures, markers, organization, coverage targets
- [Domain Testing Guide](domain-testing.md) - Performance and data leakage testing
