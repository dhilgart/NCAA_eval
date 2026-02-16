# Quality Gates

This document explains **what runs when** and **why** in the ncaa_eval quality
pipeline. For the actual PR checklist, see
[`.github/pull_request_template.md`](../.github/pull_request_template.md).

---

## Pre-Commit (Fast, Local)

These checks run automatically on every commit via pre-commit hooks (configured in
Story 1.4). They are fast enough to run on each commit without disrupting flow.

| Check | Tool | What It Catches |
|---|---|---|
| Lint | `ruff check .` | Style violations, import issues, anti-patterns |
| Format | `ruff format --check .` | Inconsistent formatting |
| Type-check | `mypy` (strict) | Missing annotations, type errors |
| Package manifest | `check-manifest` | Missing files from distribution (MANIFEST.in drift) |
| Smoke tests | `pytest -m smoke` | Broken imports, basic sanity failures, schema contract breaks |
| Commit message | Commitizen | Non-conventional commit format |

**Design principle:** Pre-commit gates must complete in seconds. They catch the
most common issues before code ever leaves the developer's machine.

### Smoke Tests

A curated subset of the test suite runs at pre-commit time via the `smoke` pytest
marker. The total smoke suite must stay **under 5 seconds** — if it gets slower,
tests should be demoted to the full PR suite.

**What belongs in smoke:**

- **Import checks** — can the package be imported without errors? (catches circular
  imports, missing dependencies, broken `__init__.py`)
- **Core function sanity** — do critical public functions accept valid input without
  crashing? (not full correctness, just "doesn't blow up")
- **Schema/contract tests** — do Pydantic models and TypedDicts validate with
  representative sample data? (catches accidental field renames or type changes)

**What stays out of smoke:**

- Anything touching disk, network, or large DataFrames
- Full correctness / edge-case tests
- Integration and end-to-end tests
- Performance benchmarks

Mark a test as smoke: `@pytest.mark.smoke`. The marker and pre-commit hook
configuration happen in Stories 1.4-1.5.

## PR / CI (Full Suite)

These checks run when a pull request is opened or updated. They are slower but
more thorough.

| Check | Tool | What It Catches |
|---|---|---|
| Unit tests | `pytest` | Logic regressions, broken contracts |
| Integration tests | `pytest` | Component interaction failures |
| Edge compatibility | `edgetest` | Dependency compatibility issues at version boundaries |
| Coverage | `pytest-cov` | Untested code paths |

**Design principle:** PR gates catch issues that require running the full project.
They ensure nothing is broken before code reaches the main branch.

## Code Review (AI Agent)

Code review is performed by an AI agent via the `code-review` workflow (ideally
using a different LLM than the one that implemented the story). Automated tooling
and CI cannot catch everything — the review agent evaluates higher-level concerns:

- **Docstring quality** — Are public APIs documented with clear descriptions?
- **Vectorization compliance** — No `for` loops over DataFrames for calculations (see [STYLE_GUIDE.md](STYLE_GUIDE.md) Section 5)
- **Architecture compliance** — Type sharing, no direct IO in UI, appropriate use of Pydantic vs TypedDict
- **Supporting evidence** — Performance claims backed by benchmarks, bug fixes accompanied by regression tests
- **Design intent** — Does the implementation match the story's acceptance criteria and architectural intent?

## Owner Review

Final approval rests with the project owner. Focus areas beyond what automated
tools and AI review cover:

- **Does this actually solve the problem?** — Functional correctness from a domain perspective
- **Is the approach what I expected?** — Strategic alignment with project direction
- **Anything feel off?** — Gut-check on complexity, naming, or scope creep

## Why This Separation Matters

Putting slow checks in pre-commit annoys developers and leads to `--no-verify`.
Putting fast checks only in CI means issues are caught too late. The two-tier
approach keeps the feedback loop tight where it matters most.

---

**Reference:** [`docs/STYLE_GUIDE.md`](STYLE_GUIDE.md) for full coding standards
and rationale.
