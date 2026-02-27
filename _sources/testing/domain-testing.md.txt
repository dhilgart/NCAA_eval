# Domain-Specific Testing

This guide covers testing approaches specific to the `ncaa_eval` project's domain requirements: **performance testing** (vectorization compliance) and **data leakage prevention**.

---

## Performance Testing Guidelines

### Context: Vectorization First Rule

The Architecture mandates **"Vectorization First"** (Section 12):

> **Reject PRs that use `for` loops over Pandas DataFrames for metric calculations.**

This project has a **60-second backtest target** (10-year Elo training + inference). NFR1 mandates vectorization; NFR2 mandates parallelism. Performance testing ensures these requirements are met.

---

### How to Test Vectorization

#### 1. Assertion-Based Unit Tests (Quick Smoke Check)

**Purpose:** Catch obvious violations quickly during pre-commit

```python
@pytest.mark.smoke
def test_calculate_brier_score_is_vectorized():
    """Verify Brier score uses vectorized operations (no Python loops)."""
    import inspect

    source = inspect.getsource(calculate_brier_score)

    # Check that implementation doesn't contain for loops over data
    # (Allow for loops in fixture setup, just not in calculation logic)
    assert "for " not in source or "# vectorized" in source, \
        "Brier score must use vectorized numpy/pandas operations"
```

**Pros:** Fast, catches obvious violations
**Cons:** Brittle (may trigger false positives on legitimate loops in comments)

**Pre-commit:** ✅ **YES** - Mark as `@pytest.mark.smoke`

---

#### 2. Performance Benchmark Integration Tests

**Purpose:** Verify actual performance meets targets

```python
import timeit
import pytest

@pytest.mark.slow
@pytest.mark.integration
def test_calculate_brier_score_performance():
    """Verify Brier score meets performance target."""
    predictions = np.random.rand(10_000)
    actuals = np.random.randint(0, 2, 10_000)

    time_taken = timeit.timeit(
        lambda: calculate_brier_score(predictions, actuals),
        number=100
    )

    # Should complete 100 iterations in < 100ms (1ms per call for 10k predictions)
    assert time_taken < 0.1, f"Brier score too slow: {time_taken:.3f}s for 100 iterations"
```

**Pros:** Verifies actual performance, catches regressions
**Cons:** Slow, results vary by machine (need generous thresholds)

**Pre-commit:** ❌ **NO** (too slow)
**PR-time:** ✅ **YES** - Mark as `@pytest.mark.slow` and `@pytest.mark.performance`

---

#### 3. pytest-benchmark (Optional, for Critical Paths)

> ⚠️ **NOT YET INSTALLED:** `pytest-benchmark` is optional and NOT currently in `pyproject.toml`.
> Use timeit-based benchmarks (Option 2) for now. Story 1.5 will evaluate pytest-benchmark for installation.

**Purpose:** Track performance over time with detailed statistics

```python
def test_calculate_brier_score_benchmark(benchmark):
    """Benchmark Brier score calculation."""
    predictions = np.random.rand(10_000)
    actuals = np.random.randint(0, 2, 10_000)

    result = benchmark(calculate_brier_score, predictions, actuals)

    # Benchmark provides detailed statistics (mean, stddev, percentiles)
    assert result is not None
```

**Pros:** Detailed statistics, integrated with pytest, tracks performance over time
**Cons:** Requires `pytest-benchmark` plugin (evaluate in Story 1.5)

**Pre-commit:** ❌ **NO**
**PR-time:** ✅ **YES** (if plugin is installed)

---

### Comprehensive Performance Test Example

```python
@pytest.mark.slow
@pytest.mark.performance
@pytest.mark.integration
def test_full_backtest_meets_60_second_target():
    """Verify 10-year Elo backtest completes within 60 seconds (NFR1)."""
    games = load_games_for_years(range(2013, 2023))

    start_time = timeit.default_timer()
    model = EloModel()
    model.fit(games)
    predictions = model.predict(games)
    elapsed = timeit.default_timer() - start_time

    assert elapsed < 60.0, f"Backtest too slow: {elapsed:.2f}s (target: < 60s)"
```

---

### Vectorization Smoke Test Pattern

```python
import inspect

@pytest.mark.smoke
@pytest.mark.performance
def test_metric_calculations_are_vectorized():
    """Verify critical metric calculations don't use Python loops (NFR1)."""
    from ncaa_eval.evaluation import metrics

    # Check that metric module doesn't contain for loops over DataFrames
    source = inspect.getsource(metrics)

    # This is a heuristic - manual review is still needed
    forbidden_patterns = [
        "for _ in df.iterrows()",
        ".iterrows()",
        ".itertuples()",
        "for row in df",
    ]

    for pattern in forbidden_patterns:
        assert pattern not in source, \
            f"Metrics module contains non-vectorized pattern: {pattern}"
```

---

### Recommendation

- Use **assertion-based tests** for quick smoke checks (mark as `@pytest.mark.smoke`)
- Use **performance benchmarks** for critical modules (`evaluation/metrics.py`, `evaluation/simulation.py`) as `@pytest.mark.slow` tests (PR-time only)
- Consider **pytest-benchmark** for performance regression tracking (optional, evaluate during Story 1.5)

---

## Data Leakage Prevention Testing

### Context: Temporal Boundary Enforcement

The Architecture mandates **"Data Safety: Temporal boundaries enforced strictly in the API to prevent data leakage"** (Section 11.2).

NFR4 requires strict temporal boundaries: **training data must never include information from future games**. This is critical for model validity.

---

### How to Test Temporal Integrity

#### 1. Unit Tests (API Contract)

**Purpose:** Verify API rejects requests for future data

```python
def test_get_chronological_season_enforces_cutoff():
    """Verify chronological API rejects requests for future data."""
    api = ChronologicalDataAPI()

    with pytest.raises(ValueError, match="Cannot access future data"):
        api.get_games_before(date="2025-12-31", cutoff_date="2025-01-01")
```

**Tests:** API raises error when requesting data beyond cutoff

**Pre-commit:** ✅ **YES** (fast, no I/O) - Mark as `@pytest.mark.smoke`

---

#### 2. Integration Tests (End-to-End Workflow)

**Purpose:** Verify end-to-end workflows prevent leakage

```python
@pytest.mark.integration
def test_walk_forward_validation_prevents_leakage(sample_games_fixture):
    """Verify walk-forward CV never trains on future data."""
    splitter = WalkForwardSplitter(years=range(2015, 2025))

    for train_data, test_data, year in splitter.split(sample_games_fixture):
        # Verify all training games occur before test games
        train_max_date = train_data['date'].max()
        test_min_date = test_data['date'].min()

        assert train_max_date < test_min_date, \
            f"Data leakage detected in {year} fold: train_max={train_max_date}, test_min={test_min_date}"
```

**Tests:** Cross-validation splitter never leaks future data into training

**Pre-commit:** ❌ **NO** (too slow)
**PR-time:** ✅ **YES** - Mark as `@pytest.mark.integration`

---

#### 3. Property-Based Tests (Invariants)

**Purpose:** Verify temporal boundary holds across all possible inputs

```python
from hypothesis import given, strategies as st

@pytest.mark.property
@given(cutoff_year=st.integers(2015, 2025))
def test_temporal_boundary_invariant(cutoff_year):
    """Verify no API call can access data beyond cutoff year."""
    api = ChronologicalDataAPI()
    games = api.get_games_before(cutoff_year=cutoff_year)

    # Invariant: ALL games must be from or before cutoff year
    assert all(game.season <= cutoff_year for game in games), \
        f"API returned games beyond cutoff year {cutoff_year}"
```

**Tests:** Temporal boundary holds across all possible cutoff years

**Pre-commit:** ❌ **NO** (Hypothesis is slow)
**PR-time:** ✅ **YES** - Mark as `@pytest.mark.property`

---

### Comprehensive Data Leakage Test Example

```python
@pytest.mark.regression
@pytest.mark.integration
def test_chronological_api_rejects_exact_cutoff_date():
    """Regression test: API allowed games ON cutoff date (data leakage).

    Bug: Issue #87 - get_games_before() used <= instead of < for date comparison.
    Fixed: 2026-02-10 - Changed to strict less-than comparison.
    """
    api = ChronologicalDataAPI()
    cutoff = "2023-03-15"

    # Load games before cutoff
    games = api.get_games_before(cutoff_date=cutoff)

    # NO game should have date >= cutoff (strict less-than)
    from datetime import datetime
    cutoff_dt = datetime.fromisoformat(cutoff)

    for game in games:
        game_date = datetime.fromisoformat(game.date)
        assert game_date < cutoff_dt, \
            f"Game on/after cutoff leaked through: {game.date} (cutoff: {cutoff})"
```

---

### Recommendation

- Add **unit tests** for API contract violations (fast, pre-commit safe)
- Add **integration tests** for end-to-end workflows like walk-forward CV (PR-time only)
- Add **property-based tests** for temporal boundary invariants (PR-time only)

---

## Summary

### Performance Testing Checklist

- ✅ Assertion-based smoke tests for vectorization compliance (`@pytest.mark.smoke`)
- ✅ Performance benchmarks for critical paths (`@pytest.mark.slow`, `@pytest.mark.performance`)
- ✅ Full backtest timing test (60-second target)
- ✅ Forbidden pattern detection (`.iterrows()`, etc.)

### Data Leakage Prevention Checklist

- ✅ Unit tests for API contract enforcement (`@pytest.mark.smoke`)
- ✅ Integration tests for end-to-end workflows (`@pytest.mark.integration`)
- ✅ Property-based tests for temporal boundary invariants (`@pytest.mark.property`)
- ✅ Regression tests for previously discovered leakage bugs (`@pytest.mark.regression`)

---

## See Also

- [Test Purpose Guide](test-purpose-guide.md) - Performance and Regression testing
- [Test Scope Guide](test-scope-guide.md) - Unit vs Integration tests
- [Test Approach Guide](test-approach-guide.md) - Example-based vs Property-based
- [Execution Guide](execution.md) - When tests/checks run (4-tier model)
- [Quality Assurance Guide](quality.md) - Mutation testing, coverage analysis
- [STYLE_GUIDE.md](../STYLE_GUIDE.md) Section 5 - Vectorization First rule
