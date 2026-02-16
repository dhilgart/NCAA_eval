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
    game_df.loc[idx, "margin"] = game_df.loc[idx, "home_score"] - game_df.loc[idx, "away_score"]
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

### Review Criteria

- Code follows naming conventions (Section 2).
- Imports are ordered correctly (Section 3).
- New public APIs have docstrings (Section 1).
- No `for` loops over DataFrames for calculations (Section 5).
- Type annotations are complete (Section 4).
- PEP 20 design principles respected (Section 6).
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

## References

- `pyproject.toml` — Single source of truth for all tool configurations
- `docs/specs/05-architecture-fullstack.md` Section 12 — Coding standards
- `docs/specs/05-architecture-fullstack.md` Section 10 — Development workflow
- `docs/specs/03-prd.md` Section 4 — Technical assumptions & constraints
