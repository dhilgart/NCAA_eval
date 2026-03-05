# Testing Conventions

This guide covers test organization, fixtures, markers, and naming conventions.

---

## Test Organization

### Directory Structure

```
tests/
├── __init__.py
├── conftest.py                          # Shared fixtures
├── fixtures/
│   ├── .gitkeep
│   └── kaggle/
│       ├── MNCAATourneyCompactResults.csv
│       ├── MRegularSeasonCompactResults.csv
│       ├── MSeasons.csv
│       └── MTeams.csv
├── integration/
│   ├── __init__.py
│   ├── test_documented_commands.py      # E2E CLI documentation tests
│   ├── test_elo_integration.py          # Integration: Elo pipeline
│   ├── test_feature_serving_integration.py  # Integration: feature serving
│   └── test_sync.py                     # Integration: ingest → storage
└── unit/
    ├── __init__.py
    ├── test_bracket_page.py
    ├── test_bracket_renderer.py
    ├── test_calibration.py
    ├── test_chronological_serving.py
    ├── test_cli_train.py
    ├── test_connector_base.py
    ├── test_dashboard_app.py
    ├── test_dashboard_filters.py
    ├── test_deep_dive_page.py
    ├── test_elo.py
    ├── test_espn_connector.py
    ├── test_evaluation_backtest.py
    ├── test_evaluation_metrics.py
    ├── test_evaluation_plotting.py
    ├── test_evaluation_simulation.py
    ├── test_evaluation_splitter.py
    ├── test_feature_serving.py
    ├── test_framework_validation.py
    ├── test_fuzzy.py
    ├── test_graph.py
    ├── test_home_page.py
    ├── test_imports.py
    ├── test_kaggle_connector.py
    ├── test_leaderboard_page.py
    ├── test_logger.py
    ├── test_model_base.py
    ├── test_model_elo.py
    ├── test_model_logistic_regression.py
    ├── test_model_registry.py
    ├── test_model_tracking.py
    ├── test_model_xgboost.py
    ├── test_normalization.py
    ├── test_opponent.py
    ├── test_package_structure.py
    ├── test_pool_scorer_page.py
    ├── test_repository.py
    ├── test_run_store_metrics.py
    ├── test_schema.py
    └── test_sequential.py
```

### Naming Conventions

| Entity | Convention | Example |
|---|---|---|
| **Test files** | `test_<module_name>.py` | `test_metrics.py` for `src/ncaa_eval/evaluation/metrics.py` |
| **Test functions** | `test_<function>_<scenario>()` | `test_brier_score_perfect_prediction()` |
| **Fixture functions** | Descriptive name (no suffix) | `sample_teams()`, `elo_config()`, `temp_data_dir()` |
| **Test classes** | `Test<ClassName>` | `class TestEloModel:` |

### Pytest Discovery

Pytest automatically discovers:
- Files matching `test_*.py` or `*_test.py`
- Functions matching `test_*()`
- Classes matching `Test*`

No custom discovery configuration is needed (already set in `pyproject.toml`).

---

## Fixture Conventions

### Fixture Scope

| Scope | Lifetime | Use Case | Example |
|---|---|---|---|
| `function` | Per test function (default) | Independent test data, reset each test | `@pytest.fixture def sample_game(): ...` |
| `class` | Per test class | Shared setup for class methods | `@pytest.fixture(scope="class") def db_connection(): ...` |
| `module` | Per test file | Expensive setup, reused across file | `@pytest.fixture(scope="module") def trained_model(): ...` |
| `session` | Per test session | One-time setup for all tests | `@pytest.fixture(scope="session") def test_database(): ...` |

### Fixture Organization

- **`tests/conftest.py`:** Project-wide fixtures (e.g., `sample_teams()`, `elo_config()`, `temp_data_dir()`)
- **Inline fixtures:** Simple fixtures can be defined in test files if not reused elsewhere

### Fixture Best Practices

**1. Type annotations:** All fixtures must have return type annotations (mypy --strict compliance)

```python
from __future__ import annotations

import pytest
from ncaa_eval.ingest.schema import Game

@pytest.fixture
def sample_game() -> Game:
    """Provide a sample game for testing."""
    return Game(season=2023, day_num=100, w_team_id=1234, l_team_id=5678, w_score=75, l_score=70)
```

**2. Parametrized fixtures:** Use `@pytest.fixture(params=[...])` for testing multiple scenarios

```python
@pytest.fixture(params=[1500, 1800, 2100])
def elo_rating(request: pytest.FixtureRequest) -> int:
    """Provide different Elo ratings for testing."""
    return request.param
```

**3. Teardown with yield:** Use `yield` for setup/teardown patterns

```python
from typing import Iterator
from pathlib import Path
import shutil

@pytest.fixture
def temp_data_dir() -> Iterator[Path]:
    """Create temporary directory for test data."""
    temp_dir = Path("test_data_temp")
    temp_dir.mkdir(exist_ok=True)
    yield temp_dir
    # Teardown: clean up after test
    shutil.rmtree(temp_dir)
```

---

## Test Markers

Pytest markers enable selective test execution for pre-commit vs. PR-time distinction. **Markers can be combined** across all dimensions.

### Marker Definitions

| Marker | Dimension | Purpose | Command |
|---|---|---|---|
| `@pytest.mark.smoke` | Speed | Fast smoke tests for pre-commit (< 1s each, < 5s total) | `pytest -m smoke` |
| `@pytest.mark.slow` | Speed | Slow tests excluded from pre-commit (> 5 seconds each) | `pytest -m "not slow"` |
| `@pytest.mark.unit` | Scope | Pure unit tests with no I/O or external dependencies | `pytest -m unit` |
| `@pytest.mark.integration` | Scope | Integration tests (I/O, database) | `pytest -m integration` |
| `@pytest.mark.property` | Approach | Property-based tests (Hypothesis) | `pytest -m property` |
| `@pytest.mark.performance` | Purpose | Performance/benchmark tests | `pytest -m performance` |
| `@pytest.mark.regression` | Purpose | Regression tests (prevent bug recurrence) | `pytest -m regression` |
| `@pytest.mark.no_mutation` | Quality | Tests incompatible with mutmut runner directory (`Path(__file__)`-dependent) | N/A |

### Marker Configuration

Markers are configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "smoke: Fast smoke tests for pre-commit (< 10 seconds total)",
    "slow: Slow tests excluded from pre-commit (> 5 seconds each)",
    "integration: Integration tests with I/O or external dependencies",
    "property: Hypothesis property-based tests",
    "performance: Performance and benchmark tests",
    "regression: Regression tests to prevent bug recurrence",
    "no_mutation: Tests incompatible with mutmut runner directory (Path(__file__)-dependent structural tests)",
    "unit: Pure unit tests with no I/O or external dependencies",
]
```

### Marker Usage Examples

```python
import pytest
from hypothesis import given, strategies as st

# Speed marker only (fast unit test)
@pytest.mark.smoke
def test_import_package():
    """Verify package can be imported without errors."""
    import ncaa_eval
    assert ncaa_eval.__version__

# Scope + Speed markers (slow integration test, example-based)
@pytest.mark.integration
@pytest.mark.slow
def test_full_season_processing(large_dataset_fixture):
    """Process a full season of games (slow due to data volume)."""
    result = process_season(large_dataset_fixture, season=2023)
    assert len(result) > 1000

# Approach marker only (property-based unit test)
@pytest.mark.property
@given(probs=st.lists(st.floats(0.0, 1.0), min_size=1))
def test_brier_score_is_bounded(probs):
    """Verify Brier score is always in [0, 1] (invariant)."""
    preds = np.array(probs)
    actuals = np.ones(len(probs))
    score = brier_score(preds, actuals)
    assert 0.0 <= score <= 1.0

# Scope + Approach markers (property-based integration test)
@pytest.mark.integration
@pytest.mark.property
@given(cutoff_year=st.integers(2015, 2025))
def test_temporal_boundary_invariant(cutoff_year):
    """Verify API enforces temporal boundaries (integration + invariant)."""
    api = ChronologicalDataServer()
    games = api.get_games_before(cutoff_year=cutoff_year)
    assert all(game.season <= cutoff_year for game in games)

# Purpose markers (regression test)
@pytest.mark.regression
def test_elo_never_negative(elo_config, sample_games):
    """Regression test: Prevent Issue #42 (negative Elo ratings)."""
    engine = EloFeatureEngine(elo_config)
    for game in sample_games:
        engine.update_game(game)
    assert all(r >= 0 for r in engine.ratings.values())

# All dimensions combined (integration + property + performance)
@pytest.mark.integration
@pytest.mark.property
@pytest.mark.performance
@pytest.mark.slow
@given(season=st.integers(2015, 2025))
def test_game_loading_fast_and_correct(season):
    """Verify game loading is correct AND performant (all dimensions)."""
    import timeit
    start = timeit.default_timer()
    games = load_games_for_season(season)
    elapsed = timeit.default_timer() - start

    # Functional correctness
    assert len(games) > 0
    # Performance requirement
    assert elapsed < 5.0
```

---

## Test Execution Commands

| Context | Command | What Runs |
|---|---|---|
| **Pre-commit** | `pytest -m smoke` | Smoke tests only (< 5s) |
| **Local full suite** | `pytest` | All tests (no filter) |
| **Local with coverage** | `pytest --cov=src/ncaa_eval --cov-report=html` | All tests + HTML coverage report |
| **Exclude slow tests** | `pytest -m "not slow"` | All except slow tests |
| **Integration only** | `pytest -m integration` | Integration tests only (scope filter) |
| **Property-based only** | `pytest -m property` | Property-based tests only (approach filter) |
| **Performance only** | `pytest -m performance` | Performance tests only (purpose filter) |
| **Regression only** | `pytest -m regression` | Regression tests only (purpose filter) |
| **Combined filters** | `pytest -m "integration and regression"` | Integration regression tests |
| **CI/PR** | `pytest --cov=src/ncaa_eval --cov-report=term-missing` | All tests + terminal coverage |

---

## Coverage Targets

Coverage is a **quality signal, not a binary gate**. Targets guide development but are not enforced as strict gates.

### Module-Specific Targets

| Module | Line Coverage | Branch Coverage | Rationale |
|---|---|---|---|
| `evaluation/metrics.py` | 95% | 90% | Critical for correctness (LogLoss, Brier, ECE). Errors invalidate all model evaluations. |
| `evaluation/simulation.py` | 90% | 85% | Monte Carlo simulator (Epic 6). Errors affect tournament strategy. |
| `model/` (Model ABC) | 90% | 85% | Core abstraction for all models. Errors cascade to all implementations. |
| `transform/` (Features) | 85% | 80% | Feature correctness impacts model quality. Data leakage prevention critical. |
| `ingest/` (Data Ingestion) | 80% | 75% | Data quality impacts everything downstream. |
| `utils/` (Utilities) | 75% | 70% | Lower priority than core logic but still important. |
| **Overall Project** | **80%** | **75%** | Balanced target: rigorous without being burdensome. |

### Enforcement Approach

- **Pre-commit:** NO coverage enforcement (would slow development loop)
- **PR-time:** Coverage report generated (`pytest --cov`) but NOT enforced as gate (informational only)
- **Rationale:** Coverage highlights gaps but shouldn't block PRs if tests are high-quality. Manual review of coverage reports is more valuable than automated enforcement.

### Coverage Tooling

- **Tool:** `pytest-cov` plugin (configured in `pyproject.toml`)
- **HTML reports:** `pytest --cov --cov-report=html` (local debugging)
- **Terminal reports:** `pytest --cov --cov-report=term-missing` (CI)
- **Branch coverage:** `pytest --cov=src/ncaa_eval --cov-branch` (measures both line and branch coverage)

---

## Development Workflow Integration

### Nox Workflow

The testing strategy integrates into the **nox-orchestrated development pipeline**:

**Command:** `nox`
**Workflow:** Ruff (lint/format) → Mypy (strict) → Pytest (full suite)

```python
# Actual noxfile.py tests session (python=False uses active conda env)
@nox.session(python=False)
def tests(session):
    """Run the full pytest test suite."""
    session.run("pytest", "--tb=short", *session.posargs)
```

### Pre-commit Hook Integration

Pre-commit hooks are configured in `.pre-commit-config.yaml`:

```yaml
# .pre-commit-config.yaml (excerpt — pytest-smoke hook)
  - repo: local
    hooks:
      - id: pytest-smoke
        name: pytest-smoke
        entry: poetry run pytest -m smoke --tb=short -q
        language: system
        types: [python]
        pass_filenames: false
        stages: [commit]
```

---

## See Also

- [Test Scope Guide](test-scope-guide.md) - Unit vs Integration tests
- [Test Approach Guide](test-approach-guide.md) - Example-based vs Property-based
- [Test Purpose Guide](test-purpose-guide.md) - Functional, Performance, Regression
- [Execution Guide](execution.md) - When tests/checks run (4-tier model)
- [Quality Assurance Guide](quality.md) - Mutation testing, coverage analysis
- [Domain Testing Guide](domain-testing.md) - Performance and data leakage testing
