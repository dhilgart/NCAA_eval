# Test Approach Guide: Example-Based vs Property-Based Testing

This guide explains **how you write tests** - the approach used to specify test inputs and expected behavior.

## Overview

**Test approach** is one of the four orthogonal dimensions of testing. It's orthogonal to test scope - you can use either approach for unit tests, integration tests, or any other test type.

---

## Example-Based Testing (Standard Pytest)

### What it is
Tests that specify concrete input examples and their expected outputs.

### When to use
- Testing **specific known scenarios** (e.g., "log loss of [0.9] vs [1] is 0.105")
- Regression tests for previously discovered bugs
- Boundary conditions you want to verify explicitly
- Cases where you have domain knowledge of important edge cases

### Strengths
- **Easy to understand:** Concrete examples are easy to read and debug
- **Fast to write:** Just specify input → expected output
- **Precise:** You control exactly what scenarios are tested
- **Fast to run:** Tests a fixed number of cases

### Weaknesses
- **Limited coverage:** Only tests the scenarios you thought of
- **May miss edge cases:** Humans are bad at imagining all corner cases

### Techniques

**Simple assertions:** Basic input → output tests

```python
def test_elo_update_increases_winner_rating():
    """Verify Elo rating increases for the winning team."""
    initial_rating = 1500
    result = update_elo_rating(initial_rating, opponent_rating=1500, won=True, k_factor=32)
    assert result > initial_rating
```

**Parametrized tests:** `@pytest.mark.parametrize` to test multiple scenarios with same logic

```python
@pytest.mark.parametrize("predictions,actuals,expected", [
    ([1.0, 0.0, 1.0], [1, 0, 1], 0.0),       # Perfect predictions
    ([0.9, 0.1, 0.9], [1, 0, 1], 0.03),      # Near-perfect predictions
    ([0.5, 0.5, 0.5], [1, 0, 1], 0.25),      # Random guessing
])
def test_calculate_brier_score_known_cases(predictions, actuals, expected):
    """Verify Brier score for known prediction scenarios."""
    result = calculate_brier_score(np.array(predictions), np.array(actuals))
    assert abs(result - expected) < 0.01
```

### Pre-commit eligibility
✅ **YES** (if fast) - Mark as `@pytest.mark.smoke`

---

## Property-Based Testing (Hypothesis)

### What it is
Tests that specify **invariants** (properties that should always hold) and let Hypothesis automatically generate hundreds of test cases.

### When to use
- Testing **invariants** across many inputs (e.g., "output length always equals input length")
- **Mathematical properties** (e.g., "probabilities always in [0, 1]", "log loss always non-negative")
- **Discovering edge cases** you didn't think of
- Functions with **complex input spaces** where manual examples would be tedious

### Strengths
- **Comprehensive coverage:** Tests hundreds of generated cases automatically
- **Finds edge cases:** Often discovers bugs in corner cases humans miss
- **Self-documenting:** The invariant clearly states what property is being tested
- **Regression shrinking:** When a test fails, Hypothesis finds the minimal failing case

### Weaknesses
- **Slower:** Generates and runs many test cases (100+ by default)
- **Harder to debug:** Failures may involve unusual generated inputs
- **Requires invariant thinking:** Need to identify what properties should hold

### When to use Hypothesis vs. Parametrized Pytest

- Use **Hypothesis** when you can state an *invariant* (e.g., "output length = input length for all inputs")
- Use **Parametrized Pytest** when you have *specific scenarios* to verify (e.g., "for input X, output is Y")

### Examples

**Property-based unit test:**

```python
from hypothesis import given, strategies as st

@pytest.mark.property
@given(data=st.lists(st.integers(), min_size=1, max_size=100))
def test_rolling_average_length_invariant(data):
    """Verify rolling average output length matches input (invariant holds for all inputs)."""
    series = pd.Series(data)
    result = calculate_rolling_average(series, window=5)
    assert len(result) == len(series)  # Invariant: length preserved
```

**Property-based integration test:**

```python
@pytest.mark.integration
@pytest.mark.property
@given(cutoff_year=st.integers(2015, 2025))
def test_temporal_boundary_invariant(cutoff_year):
    """Verify API never returns data beyond cutoff (invariant holds for all years)."""
    api = ChronologicalDataAPI()
    games = api.get_games_before(cutoff_year=cutoff_year)

    # Invariant: all games before or at cutoff year
    assert all(game.season <= cutoff_year for game in games)
```

**Property-based test with complex strategy:**

```python
@pytest.mark.property
@given(
    predictions=st.lists(st.floats(0, 1), min_size=1, max_size=1000),
    actuals=st.lists(st.integers(0, 1), min_size=1, max_size=1000),
)
def test_brier_score_always_non_negative(predictions, actuals):
    """Verify Brier score is always >= 0 (mathematical property)."""
    # Make lists same length
    min_len = min(len(predictions), len(actuals))
    preds = np.array(predictions[:min_len])
    acts = np.array(actuals[:min_len])

    result = calculate_brier_score(preds, acts)
    assert result >= 0  # Invariant: Brier score cannot be negative
```

### Hypothesis Strategies (Common Patterns)

```python
from hypothesis import strategies as st

# Basic strategies
st.integers(min_value=0, max_value=100)        # Integers in range
st.floats(min_value=0.0, max_value=1.0)        # Floats in range
st.text(min_size=1, max_size=50)               # Random strings
st.booleans()                                   # True or False

# Collection strategies
st.lists(st.integers(), min_size=1, max_size=100)  # Lists of integers
st.sets(st.text(), min_size=0, max_size=10)         # Sets of strings
st.dictionaries(keys=st.text(), values=st.floats()) # Dictionaries

# Composite strategies
st.tuples(st.integers(), st.floats())           # Tuples with mixed types
st.one_of(st.integers(), st.none())             # Either int or None

# Data-driven strategies (most powerful)
st.data()  # Draw values dynamically within test
```

### Pre-commit eligibility
❌ **NO** - Hypothesis is slow (generates 100+ test cases)

Mark as `@pytest.mark.property` or `@pytest.mark.slow`

---

## Decision Tree: Example-Based vs. Property-Based

```
Does the test verify a specific known scenario?
├─ YES → Use Example-Based Testing
│         - Parametrize if testing multiple similar scenarios
│         - Examples: regression tests, boundary conditions, known edge cases
│
└─ NO → Can you state an invariant that should hold for all inputs?
         ├─ YES → Use Property-Based Testing (Hypothesis)
         │         - State the invariant clearly
         │         - Let Hypothesis generate test cases
         │         - Examples: "length preserved", "result always positive", "sum = 1.0"
         │
         └─ NO → Use Example-Based Testing
                   - If you can't state an invariant, test specific examples
                   - Consider if the function is too complex (may need refactoring)
```

---

## Combining Both Approaches

Many functions benefit from **both** example-based and property-based tests:

```python
# Example-based: Test specific known cases
@pytest.mark.parametrize("input,expected", [
    ("St. Mary's", "Saint Mary's"),
    ("UNC", "North Carolina"),
])
def test_clean_team_name_known_cases(input, expected):
    """Verify normalization for known abbreviations."""
    assert clean_team_name(input) == expected

# Property-based: Test invariants
@pytest.mark.property
@given(name=st.text(min_size=1))
def test_clean_team_name_never_empty(name):
    """Verify normalization never returns empty string."""
    result = clean_team_name(name)
    assert len(result) > 0  # Invariant: output is never empty
```

---

## See Also

- [Test Scope Guide](test-scope-guide.md) - Unit vs Integration tests
- [Test Purpose Guide](test-purpose-guide.md) - Functional, Performance, Regression
- [Execution Guide](execution.md) - When tests/checks run (4-tier model)
- [Quality Assurance Guide](quality.md) - Mutation testing, coverage analysis
