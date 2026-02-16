# NCAA Eval Style Guide

This guide documents the coding standards and conventions for the `ncaa_eval` project.
All rules here reflect decisions already configured in `pyproject.toml`. Tooling
enforcement is handled by separate stories (1.4-1.6); this document captures the
**what** and **why**.

---

## 1. Docstring Convention (Google Style)

**Choice:** Google-style docstrings.
**Configured in:** `[tool.ruff.lint.pydocstyle] convention = "google"`

### Rationale

- Google style reads naturally in source code and renders well in Sphinx via
  `napoleon`.
- With PEP 484 type annotations present everywhere (mypy --strict), repeating
  types inside docstrings is redundant. Google style keeps the prose concise.

### Rules

1. **Do not duplicate types** in `Args:` / `Returns:` sections. Type information
   lives in annotations only.
2. Include `Args:`, `Returns:` (unless `None`), and `Raises:` (if applicable).
3. Add an `Examples:` section for complex public functions.
4. Docstrings are **encouraged but not enforced** on every entity. When pydocstyle
   (`D`) rules are enabled (Story 1.4), `D1` (missing docstring) and `D415`
   (first-line punctuation) will be suppressed via the Ruff `ignore` list.
   **Note:** `D` rules are *not currently active* — the `convention = "google"`
   setting and ignore entries in `pyproject.toml` are preparatory config that
   takes effect once `D` is added to `extend-select` in Story 1.4.
   At minimum, every public module and every public class should have a docstring.

### Example

```python
from __future__ import annotations


def rolling_efficiency(
    scores: pd.Series,
    window: int,
    min_periods: int = 1,
) -> pd.Series:
    """Compute rolling offensive efficiency over a sliding window.

    Applies an exponentially-weighted mean to per-game point totals,
    normalised by possessions.

    Args:
        scores: Raw per-game point totals.
        window: Number of games in the rolling window.
        min_periods: Minimum observations required for a valid result.

    Returns:
        Rolling efficiency values aligned to the input index.

    Raises:
        ValueError: If *window* is less than 1.

    Examples:
        >>> rolling_efficiency(pd.Series([70, 80, 90]), window=2)
        0      NaN
        1    75.0
        2    85.0
        dtype: float64
    """
```

---

## 2. Naming Conventions

| Entity | Convention | Example |
|---|---|---|
| **Modules / packages** | `snake_case` | `feature_pipeline.py`, `ncaa_eval/` |
| **Classes** | `PascalCase` | `EloModel`, `TeamStats` |
| **Functions / methods** | `snake_case` | `compute_margin`, `get_team_id` |
| **Variables** | `snake_case` | `home_score`, `season_df` |
| **Constants** | `UPPER_SNAKE_CASE` | `DEFAULT_K_FACTOR`, `MIN_GAMES` |
| **Type aliases** | `PascalCase` (use `type` statement) | `type TeamId = int` |
| **Private members** | Single leading underscore | `_cache`, `_validate_input()` |
| **Test files** | `test_<module>.py` | `test_elo_model.py` |
| **Test functions** | `test_<behaviour>` | `test_margin_positive_when_home_wins` |

### Additional Rules

- **No abbreviations** in public APIs unless universally understood (`id`, `df`,
  `url`).
- **Boolean names** should read as predicates: `is_valid`, `has_played`,
  `should_normalize`.
- **DataFrame variables** should indicate their grain: `game_df`, `season_stats_df`,
  `team_features_df`.

---

## 3. Import Ordering

**Configured in:** `[tool.ruff.lint.isort]`

Imports are ordered in three groups separated by blank lines:

1. **Standard library** (`os`, `pathlib`, `typing`, ...)
2. **Third-party** (`pandas`, `numpy`, `xgboost`, ...)
3. **Local / first-party** (`ncaa_eval`, `tests`)

### Mandatory Rules

| Rule | Detail | Config Reference |
|---|---|---|
| Future annotations | Every file must start with `from __future__ import annotations` | `required-imports` |
| Combine as-imports | `from foo import bar, baz as B` on one line | `combine-as-imports = true` |
| First-party detection | `tests` is known first-party | `known-first-party = ["tests"]` |

### Example

```python
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from ncaa_eval.features import rolling_efficiency
from ncaa_eval.models.base import ModelBase
```

### Active Ruff Rules

Beyond isort, `pyproject.toml` extends the default Ruff rules (`E`, `F`) with:

| Rule | Category | What It Enforces |
|---|---|---|
| `I` | isort | Import ordering (see above) |
| `UP` | pyupgrade | Modern Python syntax — e.g., `list[int]` not `List[int]`, `X \| None` not `Optional[X]` |
| `PT` | flake8-pytest-style | Pytest best practices — e.g., `@pytest.fixture` conventions, parametrize style |
| `TID25` | tidy-imports | Import hygiene — bans relative imports from parent packages |

### Suppressed Rules

| Rule | Why Suppressed |
|---|---|
| `E501` | Line length is enforced by the Ruff **formatter** (110 chars), not the linter. Suppressing `E501` avoids double-reporting. |
| `D1` | Missing-docstring warnings. Not yet active (see Section 1 note). |
| `D415` | First-line punctuation. Not yet active (see Section 1 note). |

---

## 4. Type Annotation Standards

**Configured in:** `[tool.mypy] strict = true`

### What `--strict` Means for Developers

Every function signature, variable, and return type must be annotated. The strict
flag enables all of the following (as of mypy 1.x — the exact set may change with
future versions; run `mypy --help` to see the current list):

- `--disallow-untyped-defs` — every function needs annotations
- `--disallow-any-generics` — no bare `list`, `dict`; use `list[int]`, `dict[str, float]`
- `--warn-return-any` — functions must not silently return `Any`
- `--no-implicit-reexport` — explicit `__all__` required for re-exports
- `--strict-bytes` — distinguishes `bytes` from `str`

### Practical Guidelines

1. Use `from __future__ import annotations` (already enforced) so all annotations
   are strings and forward references work.
2. Use modern syntax: `list[int]` not `List[int]`, `X | None` not `Optional[X]`.
3. Use the `type` statement for aliases: `type TeamId = int`.
4. For third-party libraries with incomplete stubs (numpy, pandas, xgboost), the
   config uses `follow_imports = "silent"` to suppress errors from untyped
   dependencies. Add `# type: ignore[<code>]` only when truly necessary and always
   include the specific error code.
5. The `py.typed` marker at `src/ncaa_eval/py.typed` signals PEP 561 compliance.

---

## 5. Vectorization First

> **Reject PRs that use `for` loops over Pandas DataFrames for metric
> calculations.**
>
> — Architecture Section 12

This is a **hard rule**. Scientific computation in this project must use vectorized
operations.

### Why

- Vectorized pandas/numpy operations are 10-100x faster than Python loops.
- They are easier to reason about for correctness.
- They compose cleanly with `.pipe()` chains.

### Correct Pattern

```python
# Vectorized — computes margin for every game in one shot
game_df["margin"] = game_df["home_score"] - game_df["away_score"]

# Vectorized rolling mean
game_df["rolling_ppg"] = game_df.groupby("team_id")["points"].transform(
    lambda s: s.rolling(window=5, min_periods=1).mean()
)
```

### Incorrect Pattern

```python
# WRONG — for loop over DataFrame rows
margins = []
for _, row in game_df.iterrows():
    margins.append(row["home_score"] - row["away_score"])
game_df["margin"] = margins
```

### Exceptions

Explicit `for` loops are acceptable only when:

1. Operating on a **small, fixed collection** (e.g., iterating over 5 model
   configs, not 10,000 game rows).
2. Performing **graph traversal** (e.g., NetworkX operations that have no
   vectorized equivalent).
3. The loop body involves **side effects** that cannot be vectorized (e.g.,
   writing files per team).

In these cases, add a brief comment explaining why the loop is necessary.

---

## 6. Design Philosophy (PEP 20 - The Zen of Python)

> "Beautiful is better than ugly. Explicit is better than implicit. Simple is better than complex."
>
> — PEP 20

While PEP 20 can't be fully automated, we enforce its core principles through code review and tooling where possible.

### Enforced Principles

#### Simple is Better Than Complex
- **Tooling:** Ruff complexity checks (see Section 6.1)
- **Review:** Max cyclomatic complexity = 10, max function length = 50 lines
- **Guideline:** If a function is hard to name, it's doing too much — split it

```python
# GOOD: Simple, single-purpose function
def calculate_margin(home_score: int, away_score: int) -> int:
    """Calculate point margin (positive = home win)."""
    return home_score - away_score


# BAD: Too complex, doing multiple things
def process_game(game_data: dict) -> dict:
    """Process game... but what does this actually do?"""
    # 80 lines of nested logic with unclear responsibilities
    ...
```

#### Explicit is Better Than Implicit
- **Tooling:** Mypy strict mode enforces explicit types
- **Review:** No magic numbers, no implicit state changes, clear function names
- **Guideline:** Code should read like prose — the reader shouldn't have to guess

```python
# GOOD: Explicit parameters and behavior
def calculate_elo_change(
    rating: int,
    opponent_rating: int,
    won: bool,
    k_factor: int = 32,
) -> int:
    """Calculate Elo rating change after a game."""
    expected = 1 / (1 + 10 ** ((opponent_rating - rating) / 400))
    actual = 1 if won else 0
    return int(k_factor * (actual - expected))


# BAD: Implicit behavior, magic numbers
def adjust_rating(r: int, o: int, w: bool) -> int:
    return int(32 * (w - 1 / (1 + 10 ** ((o - r) / 400))))
```

#### Readability Counts
- **Tooling:** Ruff formatting (110 char lines), naming conventions
- **Review:** Variable names reflect domain concepts, not abbreviations
- **Guideline:** Write for humans first, computers second

```python
# GOOD: Clear domain language
team_offensive_efficiency = total_points / total_possessions

# BAD: Abbreviations require mental translation
off_eff = pts / poss
```

#### Flat is Better Than Nested
- **Tooling:** Ruff detects excessive nesting
- **Review:** Max nesting depth = 3
- **Guideline:** Use early returns to reduce nesting

```python
# GOOD: Flat structure with early returns
def validate_game(game: Game) -> None:
    """Validate game data."""
    if game.home_score < 0:
        raise ValueError("Home score cannot be negative")
    if game.away_score < 0:
        raise ValueError("Away score cannot be negative")
    if game.date > datetime.now():
        raise ValueError("Game cannot be in the future")


# BAD: Nested structure
def validate_game(game: Game) -> None:
    if game.home_score >= 0:
        if game.away_score >= 0:
            if game.date <= datetime.now():
                return
            else:
                raise ValueError("Game cannot be in the future")
        else:
            raise ValueError("Away score cannot be negative")
    else:
        raise ValueError("Home score cannot be negative")
```

#### There Should Be One Obvious Way to Do It
- **Review:** Follow established project patterns (e.g., vectorization for calculations)
- **Guideline:** Check existing code before inventing new approaches

```python
# GOOD: Follows project pattern (vectorized operations)
game_df["margin"] = game_df["home_score"] - game_df["away_score"]

# BAD: Invents custom approach (loops)
for idx in range(len(game_df)):
    game_df.loc[idx, "margin"] = (
        game_df.loc[idx, "home_score"] - game_df.loc[idx, "away_score"]
    )
```

### Code Review Checklist for PEP 20

During code review, verify:

- [ ] **Simplicity:** Functions have single, clear responsibilities (McCabe complexity ≤ 10)
- [ ] **Explicitness:** No magic numbers, parameters have clear names, behavior is obvious
- [ ] **Readability:** Domain concepts use full words, not abbreviations
- [ ] **Flatness:** Nesting depth ≤ 3, early returns preferred
- [ ] **Consistency:** Follows existing project patterns (vectorization, type sharing, etc.)

---

### 6.1 Complexity Gates (Ruff Configuration)

**Configured in:** `pyproject.toml` → `[tool.ruff.lint.mccabe]`

| Metric | Limit | Enforced By |
|---|---|---|
| **McCabe Cyclomatic Complexity** | 10 | Ruff `C901` (pre-commit) |
| **Max Function Length** | 50 lines | Manual review |
| **Max Nesting Depth** | 3 | Manual review |
| **Max Arguments** | 5 | Ruff `PLR0913` (pre-commit) |

See `pyproject.toml` for exact configuration.

---

### 6.2 Pure Functions vs Side Effects

> **Pure functions are easier to test, faster to run, and easier to reason about.**

**Design guideline:** Keep business logic pure, push side effects to edges.

#### Pure Functions (Preferred)

**Definition:** Same input always produces same output, no side effects.

**Characteristics:**
- Deterministic (predictable)
- No external dependencies (no I/O, no database, no network, no time/randomness)
- No state mutation
- Easy to test (just input → output, no mocking)
- Perfect for property-based testing (Hypothesis)
- Fast (no I/O)

```python
# PURE: Always deterministic, easy to test
def calculate_win_probability(rating_diff: int, k_factor: float = 32.0) -> float:
    """Calculate win probability from rating difference."""
    return 1 / (1 + 10 ** (-rating_diff / 400))


# PURE: Data transformation, no side effects
def normalize_team_names(names: pd.Series) -> pd.Series:
    """Normalize team names to standard format (vectorized)."""
    return names.str.strip().str.title().str.replace("St.", "Saint")


# PURE: Mathematical calculation (vectorized)
def calculate_margins(home_scores: np.ndarray, away_scores: np.ndarray) -> np.ndarray:
    """Calculate point margins (vectorized, pure)."""
    return home_scores - away_scores
```

**Testing pure functions:**

```python
# Unit test: Simple input → output
def test_win_probability_equal_ratings():
    """Verify win probability is 50% for equal ratings."""
    assert calculate_win_probability(rating_diff=0) == 0.5


# Property test: Perfect for pure functions
@pytest.mark.property
@given(rating_diff=st.integers(-1000, 1000))
def test_win_probability_always_bounded(rating_diff):
    """Verify win probability always in [0, 1] (invariant)."""
    prob = calculate_win_probability(rating_diff)
    assert 0 <= prob <= 1
```

---

#### Side-Effect Functions (Push to Edges)

**Definition:** Functions that interact with the outside world or modify state.

**Characteristics:**
- Non-deterministic (may vary based on external state)
- External dependencies (files, database, network, time, randomness)
- Modifies state (mutates objects, writes files, updates database)
- Harder to test (requires mocks, stubs, fixtures)
- Requires integration tests

```python
# SIDE-EFFECT: Reads from file system
def load_games(path: Path) -> pd.DataFrame:
    """Load games from CSV file (I/O operation)."""
    return pd.read_csv(path)


# SIDE-EFFECT: Depends on current time (non-deterministic)
def is_game_started(game_start: datetime) -> bool:
    """Check if game has started."""
    return datetime.now() > game_start  # Changes over time


# SIDE-EFFECT: Mutates external state
def update_team_rating(team_id: int, new_rating: int) -> None:
    """Update team rating in database."""
    db.execute("UPDATE teams SET rating = ? WHERE id = ?", new_rating, team_id)
```

**Testing side-effect functions:**

```python
# Integration test: Requires fixtures
@pytest.mark.integration
def test_load_games_returns_valid_dataframe(temp_data_dir):
    """Verify games can be loaded from CSV."""
    # Setup: Create test file
    test_file = temp_data_dir / "games.csv"
    test_file.write_text("game_id,home_team,away_team\n1,Duke,UNC\n")

    # Execute
    games = load_games(test_file)

    # Assert
    assert len(games) == 1
    assert "game_id" in games.columns
```

---

#### Good Separation: Pure Core + Side-Effect Shell

**Pattern:** Keep calculations pure, orchestrate I/O at edges.

```python
# PURE: Core business logic (easy to test, vectorized)
def calculate_win_probabilities(
    home_ratings: np.ndarray,
    away_ratings: np.ndarray,
) -> np.ndarray:
    """Calculate win probabilities (pure, vectorized)."""
    rating_diff = home_ratings - away_ratings
    return 1 / (1 + 10 ** (-rating_diff / 400))


# SIDE-EFFECT: Orchestration at the edge
def simulate_tournament(games_path: Path, ratings_path: Path) -> pd.DataFrame:
    """Simulate tournament (orchestrates pure logic + I/O)."""
    # Side effects: Load data
    games = pd.read_csv(games_path)
    ratings = pd.read_csv(ratings_path, index_col="team")

    # Side effects: Data prep
    games = games.merge(ratings, left_on="home_team", right_index=True)
    games = games.merge(
        ratings, left_on="away_team", right_index=True, suffixes=("_home", "_away")
    )

    # PURE: Core calculation (vectorized, easy to test separately)
    games["win_prob"] = calculate_win_probabilities(
        games["rating_home"].values,
        games["rating_away"].values,
    )

    return games
```

**Why this is better:**

1. **Pure function (`calculate_win_probabilities`):**
   - Fast unit tests (no I/O)
   - Property-based tests (invariants)
   - Reusable in different contexts
   - Vectorized (meets NFR1)

2. **Side-effect function (`simulate_tournament`):**
   - Thin orchestration layer
   - Easy to see where I/O happens
   - Core logic testable independently

---

#### Bad: Mixing Pure Logic with Side Effects

```python
# BAD: Pure logic buried inside side effects
def simulate_tournament_bad() -> pd.DataFrame:
    """Simulate tournament (mixed design - hard to test)."""
    games = pd.read_csv("data/games.csv")  # Side effect

    results = []
    for _, game in games.iterrows():  # Side effect + non-vectorized!
        # Side effects: Database calls
        home_rating = db.query("SELECT rating FROM teams WHERE id = ?", game.home_team)
        away_rating = db.query("SELECT rating FROM teams WHERE id = ?", game.away_team)

        # Pure logic buried inside (can't test without database!)
        rating_diff = home_rating - away_rating
        prob = 1 / (1 + 10 ** (-rating_diff / 400))

        results.append(prob)

    return pd.DataFrame(results)
```

**Problems:**
- Can't test calculation logic without database
- Can't use property-based tests
- Slow (I/O in loop)
- Violates vectorization requirement
- Hard to debug (mixed concerns)

---

#### Testing Strategy by Function Type

| Function Type | Test Type | Characteristics | Example |
|---------------|-----------|-----------------|---------|
| **Pure** | Unit test, Property-based | Fast, no mocking, deterministic | `calculate_win_probability()` |
| **Side-effect** | Integration test | Slower, requires fixtures/mocks | `load_games()`, `update_team_rating()` |
| **Mixed** | ❌ AVOID | Hard to test, refactor! | `simulate_tournament_bad()` |

---

#### Code Review Checklist for Pure vs Side-Effect

During code review, verify:

- [ ] **Pure logic separated:** Business calculations are pure functions
- [ ] **Side effects at edges:** I/O, database, network calls in orchestration layer
- [ ] **No mixing:** Pure functions don't contain I/O operations
- [ ] **Vectorization:** Pure functions use numpy/pandas operations (not loops)
- [ ] **Property tests:** Pure functions have property-based tests for invariants
- [ ] **Integration tests:** Side-effect functions have integration tests with fixtures

---

## 7. PR Checklist (Summary)

Every pull request must pass the following gates. The actual PR template is at
[`.github/pull_request_template.md`](../.github/pull_request_template.md). For
the philosophy behind this two-tier approach, see
[`docs/TESTING_STRATEGY.md`](TESTING_STRATEGY.md) (Section: Quality Gates and Execution Tiers).

| Gate | Tool | Timing |
|---|---|---|
| Lint pass | Ruff | Pre-commit (fast) |
| Type-check pass | Mypy (`--strict`) | Pre-commit (fast) |
| Test pass | Pytest | PR / CI |
| Docstring coverage | Manual review | PR review |
| No vectorization violations | Manual review | PR review |
| Conventional commit messages | Commitizen | Pre-commit |
| PEP 20 compliance | Manual review | PR review |
| SOLID principles | Manual review | PR review |
| Pure functions / functional design | Manual review | PR review |

### Review Criteria

- Code follows naming conventions (Section 2).
- Imports are ordered correctly (Section 3).
- New public APIs have docstrings (Section 1).
- No `for` loops over DataFrames for calculations (Section 5).
- Type annotations are complete (Section 4).
- PEP 20 design principles respected (Section 6).
- Pure functions used for business logic, side effects at edges (Section 6.2).
- SOLID principles applied for testability (Section 10).
- Data structures shared between Logic and UI use Pydantic models or TypedDicts.
- Dashboard code never reads files directly — it calls `ncaa_eval` functions.

---

## 8. File & Module Organization

### Project Layout

```
src/
  ncaa_eval/
    __init__.py          # Package root — re-exports public API
    py.typed             # PEP 561 marker
    ingest/              # Data source connectors
    transform/           # Feature engineering
    model/               # Model ABC and implementations
    evaluation/          # Metrics, CV, simulation
    utils/               # Shared helpers (logging, assertions)
tests/
  __init__.py
  test_<module>.py       # Mirror src/ structure
  conftest.py            # Shared fixtures
dashboard/               # Streamlit UI (imports ncaa_eval — no direct IO)
data/                    # Local data store (git-ignored)
```

### Rules

1. **One concept per module.** A module should do one thing. If it grows beyond
   ~300 lines, consider splitting.
2. **Mirror `src/` in `tests/`.** `src/ncaa_eval/models/elo.py` is tested by
   `tests/test_elo.py` (or `tests/models/test_elo.py`).
3. **`__init__.py` re-exports.** Public symbols should be importable from the
   package level: `from ncaa_eval import EloModel`.
4. **No circular imports.** If two modules need each other, extract shared types
   into a third module.
5. **Configuration lives in `pyproject.toml`.** Do not create separate config files
   for tools that support `pyproject.toml`.
6. **Line length is 110.** Not the default 88. Configured in
   `[tool.ruff] line-length = 110`.

---

## 9. Additional Architecture Rules

These rules come from the project architecture and apply across all code:

| Rule | Detail |
|---|---|
| **Type sharing** | All data structures shared between Logic and UI must use Pydantic models or TypedDicts. |
| **No direct IO in UI** | The Streamlit dashboard must call `ncaa_eval` library functions — never read files directly. |
| **Commit messages** | Use conventional commits format (`feat:`, `fix:`, `docs:`, etc.) enforced by Commitizen. |
| **Python version** | `>=3.12,<4.0`. Use modern syntax (`match`, `type` statement, `X \| None`). |

---

## 10. SOLID Principles for Testability

> **SOLID principles make code testable.** Violating SOLID = hard-to-test code.

These five object-oriented design principles improve maintainability and testability. While not automatically enforced, they're checked during code review.

### S - Single Responsibility Principle (SRP)

> "A class should have only one reason to change."

**What it means:** Each class/function does ONE thing well.

**Already enforced by:** PEP 20 complexity checks (Section 6)

```python
# GOOD: Single responsibility
class GameLoader:
    """Loads games from CSV files."""

    def load(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path)


class GameValidator:
    """Validates game data."""

    def validate(self, games: pd.DataFrame) -> None:
        if games["home_score"].min() < 0:
            raise ValueError("Scores cannot be negative")


# BAD: Multiple responsibilities
class GameManager:
    """Does everything - loading, validating, processing, saving."""

    def do_everything(self, path: Path) -> None:
        # Too many reasons to change!
        pass
```

**Testing impact:** Easy to test (one thing = one test class)

---

### O - Open/Closed Principle (OCP)

> "Open for extension, closed for modification."

**What it means:** Add new features without changing existing code.

```python
# GOOD: Open for extension via inheritance
class RatingModel(ABC):
    @abstractmethod
    def predict(self, game: Game) -> float:
        pass


class EloModel(RatingModel):
    def predict(self, game: Game) -> float:
        return self._calculate_elo(game)


class GlickoModel(RatingModel):  # New model, no changes to existing code
    def predict(self, game: Game) -> float:
        return self._calculate_glicko(game)


# BAD: Must modify existing code for each new model
def predict(game: Game, model_type: str) -> float:
    if model_type == "elo":
        return calculate_elo(game)
    elif model_type == "glicko":  # Must modify this function
        return calculate_glicko(game)
```

**Testing impact:** Easy to mock/stub new implementations

---

### L - Liskov Substitution Principle (LSP)

> "Subtypes must be substitutable for their base types."

**What it means:** If `Dog` inherits from `Animal`, you can use `Dog` anywhere you use `Animal` without breaking things.

**Already enforced by:** Property-based tests verify contracts (Section: Test Purpose Guide)

```python
# GOOD: Subclass honors parent contract
class RatingModel(ABC):
    @abstractmethod
    def predict(self, game: Game) -> float:
        """Return probability in [0, 1]."""
        pass


class EloModel(RatingModel):
    def predict(self, game: Game) -> float:
        # Always returns [0, 1] as promised
        return 1 / (1 + 10 ** ((opp_rating - rating) / 400))


# BAD: Subclass violates parent contract
class BrokenModel(RatingModel):
    def predict(self, game: Game) -> float:
        # Returns values > 1.0, breaking the contract!
        return rating_difference * 100
```

**Testing impact:** Property tests verify contracts hold across all implementations

---

### I - Interface Segregation Principle (ISP)

> "Many specific interfaces are better than one general-purpose interface."

**What it means:** Don't force classes to implement methods they don't need.

**Already enforced by:** MyPy strict mode with Protocols (Section 4)

```python
# GOOD: Small, focused interfaces
class Predictable(Protocol):
    def predict(self, game: Game) -> float: ...


class Trainable(Protocol):
    def fit(self, games: pd.DataFrame) -> None: ...


class EloModel:
    # Only implements what it needs
    def predict(self, game: Game) -> float: ...


# BAD: Fat interface forces unused methods
class Model(ABC):
    @abstractmethod
    def predict(self, game: Game) -> float: ...

    @abstractmethod
    def fit(self, games: pd.DataFrame) -> None: ...

    @abstractmethod
    def cross_validate(self, games: pd.DataFrame) -> dict: ...

    # Simple models forced to implement methods they don't need!
```

**Testing impact:** Less mocking needed (small interfaces)

---

### D - Dependency Inversion Principle (DIP)

> "Depend on abstractions, not concretions."

**What it means:** High-level code shouldn't depend on low-level details.

**Already enforced by:** Architecture rule "Type sharing: use Pydantic models or TypedDicts" (Section 9)

```python
# GOOD: Depends on abstraction (Protocol)
class Simulator:
    def __init__(self, model: Predictable):  # Any model works
        self.model = model

    def simulate_tournament(self, games: list[Game]) -> pd.DataFrame:
        for game in games:
            pred = self.model.predict(game)  # Works with any Predictable
            ...


# BAD: Depends on concrete implementation
class Simulator:
    def __init__(self, elo_model: EloModel):  # Locked to EloModel
        self.model = elo_model

    def simulate_tournament(self, games: list[Game]) -> pd.DataFrame:
        # Can only use EloModel, not flexible
        pass
```

**Testing impact:** Easy to inject test doubles

---

### SOLID Review Checklist

During code review, verify:

- [ ] **SRP:** Classes/functions have single, clear responsibility (covered by PEP 20 complexity)
- [ ] **OCP:** New features added via extension (inheritance, composition), not modification
- [ ] **LSP:** Subtypes honor parent contracts (property tests verify this)
- [ ] **ISP:** Interfaces are small and focused (use Protocols, not fat abstract classes)
- [ ] **DIP:** Depends on abstractions (Protocols, TypedDicts), not concrete classes

### Summary: What's Already Covered

| SOLID Principle | Already Enforced By |
|-----------------|---------------------|
| **SRP** | PEP 20: "Simple is better than complex" (complexity ≤ 10) |
| **OCP** | Manual review (can't automate) |
| **LSP** | Property tests: "probabilities in [0, 1]" invariants |
| **ISP** | MyPy strict mode: Protocols preferred over abstract classes |
| **DIP** | Architecture: "Type sharing via Pydantic/TypedDicts" |

**Result:** SOLID compliance is mostly automated or already covered by existing standards. Code review adds a final check for OCP and overall SOLID adherence.

---

## References

- `pyproject.toml` — Single source of truth for all tool configurations
- `docs/specs/05-architecture-fullstack.md` Section 12 — Coding standards
- `docs/specs/05-architecture-fullstack.md` Section 10 — Development workflow
- `docs/specs/03-prd.md` Section 4 — Technical assumptions & constraints
