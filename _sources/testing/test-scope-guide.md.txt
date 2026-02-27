# Test Scope Guide: Unit vs Integration Tests

This guide explains **what you're testing** - the scope and boundaries of the test.

## Overview

**Test scope** is one of the four orthogonal dimensions of testing. It defines the boundaries of what's being tested - a single function in isolation, or multiple components working together.

---

## Unit Tests

### Purpose
Test individual functions or classes in isolation without external dependencies.

### Scope
Single function, method, or class in isolation.

### When to use
- Pure functions with clear input → output behavior
- Data transformations (e.g., `clean_team_name()`, `calculate_rolling_average()`)
- Single-responsibility classes (e.g., `EloRating.update()`)
- Mathematical calculations (e.g., `calculate_brier_score()`)

### Characteristics
- **Fast:** No I/O, database, or network access
- **Isolated:** Use mocks/stubs for external dependencies
- **Deterministic:** Same input always produces same output

### Example (example-based)

```python
def test_clean_team_name_normalizes_abbreviations():
    """Verify team name normalization handles common abbreviations."""
    assert clean_team_name("St. Mary's") == "Saint Mary's"
    assert clean_team_name("UNC-Chapel Hill") == "North Carolina"

def test_calculate_brier_score_perfect_prediction():
    """Verify Brier score is 0 for perfect predictions."""
    predictions = np.array([1.0, 0.0, 1.0])
    actuals = np.array([1, 0, 1])
    assert calculate_brier_score(predictions, actuals) == 0.0
```

### Example (property-based)

```python
from hypothesis import given, strategies as st

@pytest.mark.property
@given(prob=st.floats(0, 1))
def test_probability_always_bounded(prob):
    """Verify adjusted probabilities stay in [0, 1] regardless of input."""
    adjusted = adjust_probability(prob, home_advantage=0.05)
    assert 0 <= adjusted <= 1
```

### Pre-commit eligibility
✅ **YES** - Mark fast unit tests as `@pytest.mark.smoke`

---

## Integration Tests

### Purpose
Test interactions between components with real or mocked external dependencies.

### Scope
Multiple components, modules, or systems working together.

### When to use
- Repository interactions (database/file I/O)
- API workflows (data ingestion → storage → retrieval)
- Cross-module interactions (feature engineering → model training → evaluation)
- End-to-end pipelines

### Characteristics
- **Slower:** Involves I/O, setup/teardown of test data
- **Real interactions:** Tests actual component integration, not just interfaces
- **May use test doubles:** Can use in-memory databases or mock external APIs

### Example (example-based)

```python
@pytest.mark.integration
def test_sync_command_fetches_and_stores_games(temp_data_dir):
    """Verify sync command successfully ingests and stores game data."""
    sync_games(source="test_api", output_dir=temp_data_dir)

    games_df = load_games(temp_data_dir)
    assert len(games_df) > 0
    assert "game_id" in games_df.columns

@pytest.mark.integration
def test_end_to_end_training_pipeline(sample_games_fixture):
    """Verify complete pipeline from data to trained model."""
    features = engineer_features(sample_games_fixture)
    model = EloModel()
    model.fit(features)
    predictions = model.predict(features)

    assert len(predictions) == len(features)
    assert all(0 <= p <= 1 for p in predictions)
```

### Example (property-based)

```python
from hypothesis import given, strategies as st

@pytest.mark.integration
@pytest.mark.property
@given(cutoff_year=st.integers(2015, 2025))
def test_temporal_boundary_invariant_across_years(cutoff_year):
    """Verify chronological API enforces boundaries for any cutoff year."""
    api = ChronologicalDataAPI()
    games = api.get_games_before(cutoff_year=cutoff_year)

    # Invariant: ALL games must be from or before cutoff year
    assert all(game.season <= cutoff_year for game in games)
```

### Pre-commit eligibility
❌ **NO** - Too slow due to I/O

**PR-time:** ✅ **YES** - Mark as `@pytest.mark.integration`

---

## Decision Guide

```
Does the test interact with external systems (files, database, network)?
├─ YES → Integration test
│         Mark with: @pytest.mark.integration
│         Runs: PR-time only
│
└─ NO → Is it testing a single function/class in isolation?
        ├─ YES → Unit test
        │         Mark with: @pytest.mark.smoke (if fast)
        │         Runs: Pre-commit (if smoke) or PR-time
        │
        └─ NO → Probably integration (testing component interaction)
```

---

## Combining with Other Dimensions

Test scope is orthogonal to approach, purpose, and execution tier. Examples:

```python
# Unit + Example-based + Functional + Smoke
@pytest.mark.smoke
def test_clean_team_name_basic():
    """Verify team name normalization (fast sanity check)."""
    assert clean_team_name("St. Mary's") == "Saint Mary's"

# Unit + Property-based + Functional + Complete
@pytest.mark.property
@given(data=st.lists(st.integers(), min_size=1, max_size=100))
def test_rolling_average_length_invariant(data):
    """Verify rolling average output length matches input (invariant)."""
    series = pd.Series(data)
    result = calculate_rolling_average(series, window=5)
    assert len(result) == len(series)

# Integration + Example-based + Performance + Complete
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.performance
def test_full_backtest_meets_60_second_target():
    """Verify 10-year Elo backtest completes within 60 seconds (NFR1)."""
    games = load_games_for_years(range(2013, 2023))

    start_time = timeit.default_timer()
    model = EloModel()
    model.fit(games)
    predictions = model.predict(games)
    elapsed = timeit.default_timer() - start_time

    assert elapsed < 60.0, f"Backtest too slow: {elapsed:.2f}s (target: < 60s)"

# Integration + Property-based + Functional + Complete
@pytest.mark.integration
@pytest.mark.property
@given(cutoff_year=st.integers(2015, 2025))
def test_temporal_boundary_invariant(cutoff_year):
    """Verify API enforces temporal boundaries (property-based integration)."""
    api = ChronologicalDataAPI()
    games = api.get_games_before(cutoff_year=cutoff_year)
    assert all(game.season <= cutoff_year for game in games)
```

---

## See Also

- [Test Approach Guide](test-approach-guide.md) - Example-based vs Property-based testing
- [Test Purpose Guide](test-purpose-guide.md) - Functional, Performance, Regression
- [Execution Guide](execution.md) - When tests/checks run (4-tier model)
- [Quality Assurance Guide](quality.md) - Mutation testing, coverage analysis
