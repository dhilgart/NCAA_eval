# {{ cookiecutter.project_name }} Style Guide

This guide documents the coding standards and conventions for the `{{ cookiecutter.project_slug }}` project.
All rules here reflect decisions already configured in `pyproject.toml`. Tooling
enforcement is handled by pre-commit hooks and Nox sessions.

---

## 1. Docstring Convention (Google Style)

**Choice:** Google-style docstrings.
**Configured in:** `[tool.ruff.lint.pydocstyle] convention = "google"`

### Rules

1. **Do not duplicate types** in `Args:` / `Returns:` sections. Type information
   lives in annotations only.
2. Include `Args:`, `Returns:` (unless `None`), and `Raises:` (if applicable).
3. Add an `Examples:` section for complex public functions.
4. At minimum, every public module and every public class should have a docstring.

### Example

```python
from __future__ import annotations

def compute_result(
    values: pd.Series,
    window: int,
    min_periods: int = 1,
) -> pd.Series:
    """Compute rolling result over a sliding window.

    Args:
        values: Raw per-item values.
        window: Number of items in the rolling window.
        min_periods: Minimum observations required for a valid result.

    Returns:
        Rolling values aligned to the input index.

    Raises:
        ValueError: If *window* is less than 1.
    """
```

---

## 2. Naming Conventions

| Entity | Convention | Example |
|---|---|---|
| **Modules / packages** | `snake_case` | `feature_pipeline.py` |
| **Classes** | `PascalCase` | `DataProcessor`, `TeamStats` |
| **Functions / methods** | `snake_case` | `compute_result`, `get_item` |
| **Variables** | `snake_case` | `total_count`, `raw_data` |
| **Constants** | `UPPER_SNAKE_CASE` | `DEFAULT_THRESHOLD`, `MAX_RETRIES` |
| **Type aliases** | `PascalCase` (use `type` statement) | `type ItemId = int` |
| **Private members** | Single leading underscore | `_cache`, `_validate()` |
| **Test files** | `test_<module>.py` | `test_processor.py` |
| **Test functions** | `test_<behaviour>` | `test_result_positive_when_valid` |

### Additional Rules

- **No abbreviations** in public APIs unless universally understood (`id`, `df`, `url`).
- **Boolean names** should read as predicates: `is_valid`, `has_data`, `should_normalize`.

---

## 3. Import Ordering

**Configured in:** `[tool.ruff.lint.isort]`

Imports are ordered in three groups separated by blank lines:

1. **Standard library** (`os`, `pathlib`, `typing`, ...)
2. **Third-party** (`pandas`, `numpy`, ...)
3. **Local / first-party** (`{{ cookiecutter.project_slug }}`, `tests`)

### Mandatory Rules

| Rule | Detail |
|---|---|
| Future annotations | Every file must start with `from __future__ import annotations` |
| Combine as-imports | `from foo import bar, baz as B` on one line |
| First-party detection | `tests` is known first-party |

---

## 4. Type Annotation Standards

**Configured in:** `[tool.mypy] strict = true`

### Practical Guidelines

1. Use `from __future__ import annotations` (already enforced) so all annotations
   are strings and forward references work.
2. Use modern syntax: `list[int]` not `List[int]`, `X | None` not `Optional[X]`.
3. Use the `type` statement for aliases: `type ItemId = int`.
4. For third-party libraries with incomplete stubs, the config uses
   `follow_imports = "silent"` to suppress errors from untyped dependencies.
   Add `# type: ignore[<code>]` only when truly necessary and always include
   the specific error code.
5. The `py.typed` marker at `src/{{ cookiecutter.project_slug }}/py.typed` signals PEP 561 compliance.

---

## 5. Vectorization First (Data Science Projects)

> **Reject PRs that use `for` loops over Pandas DataFrames for metric calculations.**

This is a **hard rule** for data-intensive projects. Scientific computation should
use vectorized operations.

### Why

- Vectorized pandas/numpy operations are 10-100x faster than Python loops.
- They are easier to reason about for correctness.
- They compose cleanly with `.pipe()` chains.

### Exceptions

Explicit `for` loops are acceptable only when:

1. Operating on a **small, fixed collection** (e.g., iterating over 5 configs).
2. Performing **graph traversal** (e.g., NetworkX operations).
3. The loop body involves **side effects** that cannot be vectorized.

---

## 6. Design Philosophy (PEP 20)

### Enforced Principles

- **Simple is better than complex:** Max cyclomatic complexity = 10 (Ruff C901)
- **Explicit is better than implicit:** Mypy strict enforces explicit types
- **Readability counts:** 110-char line length, clear naming
- **Flat is better than nested:** Max nesting depth = 3

### Complexity Gates

| Metric | Limit | Enforced By |
|---|---|---|
| McCabe Cyclomatic Complexity | 10 | Ruff C901 |
| Max Function Length | 50 lines | Manual review |
| Max Arguments | 5 | Ruff PLR0913 |
| Max Branches | 12 | Ruff PLR0912 |
| Max Returns | 6 | Ruff PLR0911 |

### Pure Functions vs Side Effects

Keep business logic pure, push side effects to edges.

- **Pure functions:** Same input always produces same output, no side effects.
  Fast to test, perfect for property-based testing.
- **Side-effect functions:** Interact with the outside world. Push to orchestration
  layer, test with integration tests.

---

## 7. PR Checklist (Summary)

| Gate | Tool | Timing |
|---|---|---|
| Lint pass | Ruff | Pre-commit |
| Type-check pass | Mypy (`--strict`) | Pre-commit |
| Test pass | Pytest | PR / CI |
| Docstring coverage | Manual review | PR review |
| Conventional commits | Commitizen | Pre-commit |

---

## References

- `pyproject.toml` --- Single source of truth for all tool configurations
