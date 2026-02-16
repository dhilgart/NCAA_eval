# Test Execution Tiers

This guide explains **when tests and quality checks run** across the 4-tier execution model.

---

## Test Execution Scope (When Things Run)

This section defines the **multi-tier approach** that balances development speed with thorough validation. Each tier answers: **What runs when?**

**Key question:** Is this fast enough to run without disrupting the development workflow?

---

### Tier 1: Pre-Commit (Smoke Tests + Fast Checks)

#### Overview
Fast, local checks that run automatically on every commit via pre-commit hooks.

#### Time budget
**< 10 seconds total** (all checks combined)

#### Design principle
Catch the most common issues before code ever leaves the developer's machine. Must complete in seconds to avoid disrupting flow.

---

#### What Runs: Complete Checklist

| Check | Tool | What It Catches | Command |
|-------|------|-----------------|---------|
| **Lint** | `ruff check .` | Style violations, import issues, anti-patterns | Auto-fix on commit |
| **Format** | `ruff format --check .` | Inconsistent formatting | Auto-fix on commit |
| **Type-check** | `mypy` (strict) | Missing annotations, type errors | `mypy src/ncaa_eval tests` |
| **Package manifest** | `check-manifest` | Missing files from distribution (MANIFEST.in drift) | `check-manifest` |
| **Smoke tests** | `pytest -m smoke` | Broken imports, basic sanity failures, schema contract breaks | `pytest -m smoke` |
| **Commit message** | Commitizen | Non-conventional commit format | Automatic validation |

---

#### Smoke Tests: What to Include

Smoke tests are a **curated subset** of tests designed for speed. Individual smoke tests should be < 1 second each, **< 5 seconds total**.

**Include:**
- ✅ **Import checks** - Verify package imports work (catches circular imports, missing dependencies, broken `__init__.py`)
- ✅ **Core function sanity** - Critical functions accept valid input without crashing (not full correctness, just "doesn't blow up")
- ✅ **Schema/contract tests** - Pydantic models and TypedDicts validate with representative sample data (catches accidental field renames or type changes)
- ✅ **Quick invariant checks** - Fast property tests that verify basic invariants
- ✅ **Fast regression tests** - Regression tests for critical bugs that can be verified quickly

**Exclude:**
- ❌ Anything touching disk, network, or external services
- ❌ Tests that process large DataFrames or datasets
- ❌ Full correctness / edge-case tests (save for complete suite)
- ❌ Integration tests with I/O
- ❌ Performance benchmarks
- ❌ Property-based tests that generate many examples (Hypothesis is slow)

---

#### Smoke Test Examples

```python
# Smoke-eligible: Fast unit test (import check)
@pytest.mark.smoke
def test_import_package():
    """Verify package can be imported without errors."""
    import ncaa_eval
    assert ncaa_eval.__version__

# Smoke-eligible: Fast unit test (sanity check)
@pytest.mark.smoke
def test_calculate_brier_score_accepts_valid_input():
    """Verify Brier score accepts valid input without crashing."""
    predictions = np.array([0.8, 0.3])
    actuals = np.array([1, 0])
    result = calculate_brier_score(predictions, actuals)
    assert result is not None  # Just verify it doesn't crash

# Smoke-eligible: Schema contract test
@pytest.mark.smoke
def test_game_schema_validates():
    """Verify Game TypedDict validates with sample data."""
    game: Game = {
        "game_id": 1,
        "season": 2023,
        "home_team": "Duke",
        "away_team": "UNC",
        "home_score": 75,
        "away_score": 70,
    }
    # If this compiles with mypy --strict, schema is correct
    assert game["game_id"] == 1

# Smoke-eligible: Fast regression test
@pytest.mark.smoke
@pytest.mark.regression
def test_elo_rating_never_negative_quick():
    """Regression test: Elo ratings should never go negative (fast check)."""
    rating = update_elo_rating(1200, opponent_rating=2400, won=False, k_factor=32)
    assert rating >= 0

# NOT smoke-eligible: Integration test with I/O
@pytest.mark.integration
def test_load_games_from_disk(temp_data_dir):
    """Verify games can be loaded from disk (too slow for smoke)."""
    games = load_games(temp_data_dir)
    assert len(games) > 0

# NOT smoke-eligible: Property-based test (Hypothesis is slow)
@pytest.mark.property
@given(prob=st.floats(0, 1))
def test_probability_always_bounded(prob):
    """Verify adjusted probabilities stay in [0, 1]."""
    adjusted = adjust_probability(prob, home_advantage=0.05)
    assert 0 <= adjusted <= 1
```

---

#### Commands
- **Marker:** `@pytest.mark.smoke`
- **Run:** `pytest -m smoke`
- **When:** Every commit (pre-commit hook)

#### Rationale
Fast feedback loop - catches broken imports, basic sanity failures, schema contract breaks before code is committed.

---

### Tier 2: PR / CI (Complete Suite)

#### Overview
Thorough validation that runs when a pull request is opened or updated.

#### Time budget
**Minutes** (acceptable for PR review, not for pre-commit)

#### Design principle
Complete testing ensures nothing is broken before code reaches the main branch. Time is not a constraint - thoroughness is the priority.

---

#### What Runs: Complete Checklist

| Check | Tool | What It Catches | Command |
|-------|------|-----------------|---------|
| **Unit tests** | `pytest` | Logic regressions, broken contracts | `pytest` or `pytest -m "not slow"` |
| **Integration tests** | `pytest -m integration` | Component interaction failures | `pytest -m integration` |
| **Property-based tests** | `pytest -m property` | Invariant violations, edge cases | `pytest -m property` |
| **Performance tests** | `pytest -m performance` | Vectorization violations, speed regressions | `pytest -m performance` |
| **Edge compatibility** | `edgetest` | Dependency compatibility issues at version boundaries | `edgetest` |
| **Coverage** | `pytest-cov` | Untested code paths | `pytest --cov=src/ncaa_eval --cov-report=term-missing` |
| **Mutation testing (selective)** | `mutmut` | Weak tests, gaps in test coverage | `mutmut run --paths-to-mutate=src/ncaa_eval/evaluation/metrics.py` |

---

#### Complete Tests: What to Include

The full test suite including **all tests** - smoke tests plus everything too slow for pre-commit.

**Include:**
- ✅ **All smoke tests** - Fast tests run again as part of complete suite
- ✅ **Integration tests** - Tests with I/O, database, or external dependencies
- ✅ **Property-based tests** - Hypothesis tests that generate hundreds of examples
- ✅ **Performance tests** - Benchmarks and timing assertions
- ✅ **Mutation testing** - Quality verification for high-priority modules
- ✅ **Full edge-case coverage** - Comprehensive correctness testing

---

#### Complete Test Examples

```python
# Complete-only: Integration test (too slow for smoke)
@pytest.mark.integration
def test_sync_command_fetches_and_stores_games(temp_data_dir):
    """Verify sync command successfully ingests and stores game data."""
    sync_games(source="test_api", output_dir=temp_data_dir)
    games_df = load_games(temp_data_dir)
    assert len(games_df) > 0
    assert "game_id" in games_df.columns

# Complete-only: Property-based test (Hypothesis is slow)
@pytest.mark.property
@given(data=st.lists(st.integers(), min_size=1, max_size=100))
def test_rolling_average_length_invariant(data):
    """Verify rolling average output length matches input."""
    series = pd.Series(data)
    result = calculate_rolling_average(series, window=5)
    assert len(result) == len(series)

# Complete-only: Performance test
@pytest.mark.slow
@pytest.mark.performance
def test_calculate_brier_score_vectorized_performance():
    """Verify Brier score meets performance target (vectorized)."""
    predictions = np.random.rand(100_000)
    actuals = np.random.randint(0, 2, 100_000)

    time_taken = timeit.timeit(
        lambda: calculate_brier_score(predictions, actuals),
        number=10
    ) / 10

    assert time_taken < 0.01, f"Too slow: {time_taken:.4f}s"

# Complete-only: Comprehensive edge-case test
@pytest.mark.parametrize("predictions,actuals,expected", [
    ([1.0, 0.0, 1.0], [1, 0, 1], 0.0),       # Perfect
    ([0.9, 0.1, 0.9], [1, 0, 1], 0.03),      # Near-perfect
    ([0.5, 0.5, 0.5], [1, 0, 1], 0.25),      # Random
    ([0.0, 1.0, 0.0], [1, 0, 1], 1.0),       # Worst case
])
def test_calculate_brier_score_edge_cases(predictions, actuals, expected):
    """Verify Brier score for known edge cases (comprehensive)."""
    result = calculate_brier_score(np.array(predictions), np.array(actuals))
    assert abs(result - expected) < 0.01
```

---

#### Commands
- **Marker:** No specific marker (all tests run by default)
- **Run:** `pytest` (no filter)
- **When:** Pull request / CI

#### Rationale
Comprehensive validation before merge - catches regressions, verifies test quality, ensures performance targets.

---

### Tier 3: Code Review (AI Agent)

#### Overview
Code review performed by an AI agent via the `code-review` workflow (ideally using a different LLM than the one that implemented the story). Automated tooling and CI cannot catch everything.

#### What the AI agent reviews

| Focus Area | What to Check |
|------------|---------------|
| **Docstring quality** | Are public APIs documented with clear Google-style docstrings? |
| **Vectorization compliance** | No `for` loops over DataFrames for calculations (see [STYLE_GUIDE.md](../STYLE_GUIDE.md) Section 5) |
| **Architecture compliance** | Type sharing, no direct IO in UI, appropriate use of Pydantic vs TypedDict |
| **Supporting evidence** | Performance claims backed by benchmarks, bug fixes accompanied by regression tests |
| **Design intent** | Does the implementation match the story's acceptance criteria and architectural intent? |
| **Test quality** | Are tests comprehensive? Do they test the right things? Are invariants verified? |

#### Rationale
Higher-level concerns that automated tools can't evaluate. Requires understanding of project architecture, domain knowledge, and design intent.

---

### Tier 4: Owner Review

#### Overview
Final approval rests with the project owner. Focus areas beyond what automated tools and AI review cover.

#### What the owner reviews

| Focus Area | Questions to Ask |
|------------|------------------|
| **Functional correctness** | Does this actually solve the problem from a domain perspective? |
| **Strategic alignment** | Is the approach what I expected? Does it align with project direction? |
| **Complexity appropriateness** | Is the solution appropriately complex (not over-engineered, not under-engineered)? |
| **Naming and clarity** | Are names intuitive? Is the code self-explanatory? |
| **Scope creep** | Does this PR do only what was requested, or does it include unrelated changes? |
| **Gut check** | Anything feel off? Trust your instincts. |

#### Rationale
Human judgment on strategic decisions, domain correctness, and alignment with project vision.

---

### Decision Tree: Smoke vs. Complete

```
Is this test fast (< 1 second)?
├─ NO → Complete suite only
│        - Mark with appropriate scope/purpose markers (@pytest.mark.integration, @pytest.mark.slow, etc.)
│        - Examples: I/O tests, property-based tests, performance benchmarks
│
└─ YES → Could be smoke-eligible, check purpose:
         │
         ├─ Is it an import check, sanity check, or schema contract test?
         │  └─ YES → Add @pytest.mark.smoke (pre-commit eligible)
         │
         ├─ Is it a critical regression test for a high-severity bug?
         │  └─ YES → Add @pytest.mark.smoke + @pytest.mark.regression
         │
         └─ Is it a comprehensive correctness test with many edge cases?
            └─ YES → Complete suite only (save pre-commit time for sanity checks)
```

**Important:** If the smoke suite grows beyond 5 seconds, demote the slowest tests to complete-only. Pre-commit speed is critical for developer productivity.

---

### Why This Multi-Tier Approach Matters

**Fast feedback where it matters:**
- Pre-commit catches 80% of issues in seconds
- Developers get immediate feedback without context switching
- Reduces wasted time on broken code

**Thorough validation before merge:**
- PR/CI catches the remaining 20% that requires full context
- Comprehensive testing ensures quality without slowing development

**Human judgment for strategic decisions:**
- AI and human review catch design issues that tools can't
- Ensures alignment with architectural intent and project vision

**Avoiding common anti-patterns:**
- ❌ Putting slow checks in pre-commit → developers use `--no-verify`
- ❌ Putting fast checks only in CI → issues caught too late
- ❌ No code review → missing architectural and design issues

**The multi-tier approach keeps the feedback loop tight where it matters most.**

---

## See Also

- [Test Scope Guide](test-scope-guide.md) - Unit vs Integration tests
- [Test Approach Guide](test-approach-guide.md) - Example-based vs Property-based
- [Test Purpose Guide](test-purpose-guide.md) - Functional, Performance, Regression
- [Quality Assurance Guide](quality.md) - Mutation testing, coverage analysis
- [Conventions Guide](conventions.md) - Fixtures, markers, organization
- [Domain Testing Guide](domain-testing.md) - Performance and data leakage testing
