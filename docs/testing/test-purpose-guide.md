# Test Purpose Guide: Functional, Performance, and Regression Testing

This guide explains **why you're writing a test** - the purpose or goal of the test.

## Overview

**Test purpose** is one of the four orthogonal dimensions of testing. These categories are orthogonal to both scope and approach - a single test can serve multiple purposes.

---

## Functional Testing

### What it is
Tests that verify the system behaves correctly according to functional requirements.

### Purpose
Ensure the code produces the correct output for given inputs and meets business/domain requirements.

### When to use
- **Always** - This is the primary purpose of most tests
- Verifying business logic correctness
- Validating API contracts and interfaces
- Ensuring data transformations produce expected results

### Characteristics
- Tests **correctness** of behavior
- Verifies outputs match specifications
- Can be any scope (unit, integration) and any approach (example-based, property-based)

### Examples

**Functional unit test (example-based):**

```python
def test_calculate_brier_score_correct_formula():
    """Verify Brier score calculation follows the correct mathematical formula."""
    predictions = np.array([0.8, 0.3, 0.6])
    actuals = np.array([1, 0, 1])

    # Brier = mean((prediction - actual)^2)
    expected = ((0.8-1)**2 + (0.3-0)**2 + (0.6-1)**2) / 3
    result = calculate_brier_score(predictions, actuals)

    assert abs(result - expected) < 1e-10
```

**Functional integration test (property-based):**

```python
@pytest.mark.integration
@pytest.mark.property
@given(season=st.integers(2015, 2025))
def test_games_loaded_have_required_fields(season):
    """Verify all loaded games contain required fields (functional correctness)."""
    games = load_games_for_season(season)

    required_fields = ["game_id", "date", "home_team", "away_team", "home_score", "away_score"]
    for game in games:
        for field in required_fields:
            assert hasattr(game, field), f"Game missing required field: {field}"
```

### Marker
No specific marker (default assumption for all tests). Combine with scope/approach markers.

---

## Performance Testing

### What it is
Tests that verify the system meets performance requirements (speed, memory, throughput).

### Purpose
Ensure code executes within acceptable time/resource bounds, especially for performance-critical operations.

### When to use
- **Vectorization compliance** - Verify critical paths don't use Python loops (NFR1)
- **Benchmark targets** - Ensure operations meet speed requirements (e.g., 60-second backtest)
- **Regression prevention** - Detect performance degradation over time
- **Resource limits** - Verify memory usage stays within bounds

### Characteristics
- Tests **speed/efficiency** of execution
- Often includes timing assertions or benchmarks
- Critical for this project due to 60-second backtest target
- Typically marked as `@pytest.mark.slow` (excluded from pre-commit)

### Examples

**Performance unit test (assertion-based):**

```python
import timeit
import pytest

@pytest.mark.slow
@pytest.mark.performance
def test_calculate_brier_score_vectorized_performance():
    """Verify Brier score calculation meets performance target (vectorized)."""
    predictions = np.random.rand(100_000)
    actuals = np.random.randint(0, 2, 100_000)

    # Should complete in < 10ms for 100k predictions (vectorized)
    time_taken = timeit.timeit(
        lambda: calculate_brier_score(predictions, actuals),
        number=10
    ) / 10  # Average per iteration

    assert time_taken < 0.01, f"Brier score too slow: {time_taken:.4f}s (target: < 0.01s)"
```

**Performance integration test (benchmark-based):**

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

**Vectorization compliance test:**

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

### Marker
`@pytest.mark.performance` (combine with scope markers like `@pytest.mark.integration`)

### Pre-commit
Generally ❌ **NO** (too slow), except for quick vectorization smoke checks

### PR-time
✅ **YES** - Essential for NFR1 compliance

---

## Regression Testing

### What it is
Tests written specifically to prevent previously discovered bugs from recurring.

### Purpose
Ensure fixed bugs stay fixed as the codebase evolves.

### When to use
- **After fixing a bug** - Always write a regression test that would have caught the bug
- **For critical bugs** - High-severity or hard-to-debug issues need regression coverage
- **Before refactoring** - Write regression tests for existing behavior before major changes

### Characteristics
- Tests **specific scenarios that previously failed**
- Often includes a reference to the bug/issue (e.g., "Regression test for #42")
- Can be any scope (unit, integration) and any approach (example-based, property-based)
- Usually example-based (tests the specific failing case)

### Examples

**Regression unit test (example-based):**

```python
@pytest.mark.regression
def test_elo_rating_never_negative_after_extreme_losses():
    """Regression test: Elo ratings went negative after 50+ consecutive losses.

    Bug: Issue #42 - Elo.update() didn't properly floor ratings at minimum value.
    Fixed: 2026-01-15 - Added max(rating, MIN_RATING) in update logic.
    """
    rating = 1200

    # Simulate 100 consecutive losses to much higher-rated opponent
    for _ in range(100):
        rating = update_elo_rating(rating, opponent_rating=2400, won=False, k_factor=32)

    # Should never go below minimum rating (e.g., 0 or 100)
    assert rating >= 0, f"Elo rating went negative: {rating}"
```

**Regression integration test (example-based):**

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

**Regression test with property-based approach:**

```python
@pytest.mark.regression
@pytest.mark.property
@given(k_factor=st.floats(1.0, 100.0))
def test_elo_update_stable_for_all_k_factors(k_factor):
    """Regression test: Elo exploded with large k_factors.

    Bug: Issue #56 - No upper bound on rating change led to overflow.
    Fixed: 2026-01-20 - Clamped rating changes to reasonable bounds.
    """
    rating = 1500
    result = update_elo_rating(rating, opponent_rating=1500, won=True, k_factor=k_factor)

    # Rating should stay within reasonable bounds
    assert 0 <= result <= 3000, f"Rating exploded with k_factor={k_factor}: {result}"
```

### Best Practice - Always include bug context

```python
@pytest.mark.regression
def test_specific_bug_fixed():
    """Regression test: [Brief description of bug]

    Bug: Issue #[number] or [Description of when it was discovered]
    Fixed: [Date] - [Brief description of fix]
    Test: [What this test verifies]
    """
    # Test code that would have failed before the fix
    pass
```

### Marker
`@pytest.mark.regression` (combine with scope markers)

### Pre-commit
✅ **YES** if fast (mark as `@pytest.mark.smoke`)
❌ **NO** if slow

### PR-time
✅ **YES** - Always run regression tests

---

## Combining Multiple Purposes

A single test can have multiple purposes:

**Functional + Performance:**

```python
@pytest.mark.performance
def test_brier_score_correct_and_fast():
    """Verify Brier score is correct AND meets performance target."""
    predictions = np.array([0.8, 0.3])
    actuals = np.array([1, 0])

    # Functional: Verify correctness
    expected = ((0.8-1)**2 + (0.3-0)**2) / 2
    result = calculate_brier_score(predictions, actuals)
    assert abs(result - expected) < 1e-10

    # Performance: Verify speed for large datasets
    large_preds = np.random.rand(100_000)
    large_actuals = np.random.randint(0, 2, 100_000)

    import timeit
    time_taken = timeit.timeit(
        lambda: calculate_brier_score(large_preds, large_actuals),
        number=10
    ) / 10

    assert time_taken < 0.01  # < 10ms for 100k predictions
```

**Functional + Regression:**

```python
@pytest.mark.regression
@pytest.mark.integration
def test_walk_forward_cv_prevents_leakage_bug():
    """Regression + Functional: Verify walk-forward CV doesn't leak data.

    Bug: Issue #92 - CV splitter leaked one day of overlap between train/test.
    Fixed: 2026-02-12 - Changed to strict < comparison for split boundaries.
    """
    splitter = WalkForwardSplitter(years=range(2015, 2020))
    games = load_games_for_years(range(2015, 2020))

    for train_data, test_data, year in splitter.split(games):
        train_max_date = train_data['date'].max()
        test_min_date = test_data['date'].min()

        # Functional: Verify no overlap
        assert train_max_date < test_min_date

        # Regression: Specific case that failed before fix
        if year == 2017:
            # This specific split had overlap before the fix
            assert train_max_date != test_min_date
```

**All three purposes:**

```python
@pytest.mark.integration
@pytest.mark.property
@pytest.mark.performance
@pytest.mark.regression
@given(season=st.integers(2015, 2025))
def test_game_loading_comprehensive(season):
    """Functional + Performance + Regression: Comprehensive game loading test.

    Functional: Verify all games have required fields
    Performance: Verify loading completes in reasonable time
    Regression: Issue #103 - Loading 2019 season caused memory overflow
    """
    import timeit

    # Performance: Time the loading
    start = timeit.default_timer()
    games = load_games_for_season(season)
    elapsed = timeit.default_timer() - start

    # Functional: Verify correctness
    assert len(games) > 0
    required_fields = ["game_id", "date", "home_team", "away_team"]
    for game in games:
        for field in required_fields:
            assert hasattr(game, field)

    # Performance: Should complete in < 5 seconds for any season
    assert elapsed < 5.0

    # Regression: 2019 season specifically caused issues
    if season == 2019:
        import sys
        memory_mb = sys.getsizeof(games) / (1024 * 1024)
        assert memory_mb < 100  # Should stay under 100MB
```

---

## See Also

- [Test Scope Guide](test-scope-guide.md) - Unit vs Integration tests
- [Test Approach Guide](test-approach-guide.md) - Example-based vs Property-based
- [Domain Testing Guide](domain-testing.md) - Performance and data leakage testing details
- [Execution Guide](execution.md) - When tests/checks run (4-tier model)
- [Quality Assurance Guide](quality.md) - Mutation testing, coverage analysis
