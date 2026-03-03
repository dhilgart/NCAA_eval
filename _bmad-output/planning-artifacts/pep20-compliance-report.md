# PEP 20 Compliance Report — `src/ncaa_eval/`

**Date:** 2026-03-03 (updated)
**Story:** 8.9 — Add PEP 20, SOLID & Pure Function Gates
**Scope:** All source files in `src/ncaa_eval/` (excludes tests, dashboard, template)
**Methodology:** Automated Ruff analysis (C901, PLR0911, PLR0912, PLR0913, PLR2004) + manual review of exception handlers, nesting depth, and naming patterns

---

## Executive Summary

The NCAA_eval codebase demonstrates **moderate PEP 20 compliance**. Automated complexity checks (C901, PLR0911, PLR0912) found **1 function violating multiple rules** (`run_training()` in `cli/train.py`). **16 inline `# noqa` suppressions** for complexity-class codes (PLR0911, PLR0912, PLR0913, C901) were audited and annotated with descriptive rationales per the updated Lint Suppression Policy (PO approval required; inline `# noqa` preferred over `per-file-ignores` for visibility). Functions tracked for refactoring carry `— REFACTOR Story 8.1` annotations; domain-justified suppressions carry explicit rationales. The primary area of concern is **PEP 20 #3 ("Simple is better than complex")** — multiple functions exceed the 5-argument limit, and `run_training()` is a God Function that should be decomposed.

| Principle | Automated Check | Result |
|-----------|-----------------|--------|
| #2 Explicit > Implicit | PLR2004 (suppressed) | 18 magic number instances — see Section 2 |
| #3 Simple > Complex | C901 (max 10) | **1 violation** (`run_training` complexity 11) |
| #3 Simple > Complex | PLR0913 (max 5 args) | **7 violations** in `src/` (see Section 3) |
| #5 Flat > Nested | PLR0912 (max 12 branches) | **1 violation** (`run_training` has 13 branches) |
| #10 Errors pass silently | Manual review | **3 violations** (2 fixed in 8.9, 1 deferred) |
| #12 Don't guess | mypy --strict | **0 violations** (all types annotated) |

---

## 1. PEP 20 #10: Errors Should Never Pass Silently

### Fixed in Story 8.9

#### 1a. `evaluation/backtest.py:183` — Silent NaN Substitution
- **Before:** `except Exception: metrics[name] = float("nan")` — no logging
- **After:** Added `logger.warning("Metric '%s' computation failed; substituting NaN", name, exc_info=True)` before the NaN assignment
- **Impact:** Metric computation failures are now visible in logs

#### 1b. `ingest/connectors/espn.py:142` — Per-Team Fetch at DEBUG Level
- **Before:** `logger.debug("espn: get_team_schedule('%s', %d) failed", ...)` — invisible at normal log levels
- **After:** Changed to `logger.warning(...)` with `exc_info=True`
- **Impact:** Individual team schedule fetch failures are now visible without debug logging

### Deferred to Story 8.3

#### 1c. `ingest/connectors/espn.py:240` — `_parse_date()` Returns None Silently
- **Location:** `_parse_date()` static method, `except Exception: return None`
- **Issue:** Cannot distinguish "field was missing" from "field had unparseable format"
- **Why deferred:** Story 8.3 is scoped for ESPN connector resilience refactoring, including retry logic and error handling. The `_parse_date` fix requires broader changes to how date parsing failures propagate through the pipeline.

---

## 2. PEP 20 #2: Explicit is Better Than Implicit (Magic Numbers)

PLR2004 is intentionally suppressed project-wide. The following analysis documents each
instance and assesses whether a named constant is warranted.

### Tier 1: Should Define Named Constants

| File | Line | Value | Represents | Recommendation |
|------|------|-------|-----------|----------------|
| `evaluation/simulation.py` | 193 | `64` | NCAA tournament bracket size | Define `N_TEAMS = 64` (matches existing `N_ROUNDS`, `N_GAMES` pattern) |
| `evaluation/simulation.py` | 940, 1145 | `0.5` | Win probability threshold (favorite determination) | Define `WIN_PROBABILITY_THRESHOLD = 0.5` — DRY: appears at 2 locations |
| `evaluation/simulation.py` | 1073 | `100` | Minimum MC simulations for validity | Define `MIN_SIMULATIONS = 100` |
| `cli/train.py` | 150 | `0.95` / `0.05` | Class imbalance warning thresholds | Define `LABEL_IMBALANCE_HI = 0.95`, `LABEL_IMBALANCE_LO = 0.05` |

**Action:** Deferred — these are documentation-only findings per Story 8.9 scope. Story 8.1 (simulation module split) is the appropriate story for extracting simulation constants. Training constants can be addressed when `cli/train.py` is refactored (also Story 8.1).

### Tier 2: Acceptable Inline (No Action Needed)

| File | Line | Value | Rationale |
|------|------|-------|-----------|
| `evaluation/simulation.py` | 1118, 1156 | `10_000` | Logging threshold — presentational, not algorithmic |
| `evaluation/metrics.py` | 158 | `2` | Binary classification requirement — self-documenting from context |
| `evaluation/splitter.py` | 70 | `2` | Minimum seasons for walk-forward — inherent to the split concept |
| `evaluation/plotting.py` | 294 | `2` | Histogram bin count check — edge case handling |
| `ingest/connectors/espn.py` | 43, 46 | `2` | ESPN result string parsing (Team + Score) — format-specific |
| `ingest/sync.py` | 104 | `5` | Log output truncation threshold — presentational |
| `transform/normalization.py` | 119, 124 | `3` | NCAA seed format (Region + Seed) — fixed domain format |
| `transform/serving.py` | 185 | `2025` | Season-specific deduplication — documented in MEMORY.md |
| `cli/train.py` | 208 | `2` | Minimum seasons for backtest — self-documenting from context |

---

## 3. PEP 20 #3/#5: Simple > Complex / Flat > Nested

### Inline Suppression Audit (PO Directive)

All **16 inline `# noqa` suppressions** for PLR0911, PLR0912, PLR0913, and C901 were
audited and annotated with descriptive rationales. Per the updated Lint Suppression
Policy, these codes require PO approval. Inline `# noqa` is the preferred form (over
`per-file-ignores` in `pyproject.toml`) because it makes suppressions visible at the
point of use.

Each suppression now carries one of:
- **`— REFACTOR Story 8.1`**: Function is too complex; tracked for decomposition
- **`— <domain rationale>`**: Suppression justified by inherent domain dimensionality
- **`— Typer CLI options dictate arg count`**: CLI framework constraint
- **`— mirrors Game schema fields`**: Test helper mirrors data model
- **`— @patch mock injection`**: Test parameterization by pytest/mock

**16 inline suppressions annotated** across `src/`, `tests/`, and `dashboard/`:

#### Source Files (`src/ncaa_eval/`)

| File | Line | Function | Codes | Args/Complexity | Annotation |
|------|------|----------|-------|-----------------|------------|
| `cli/train.py` | 73 | `run_training()` | PLR0913, C901, PLR0912 | 7 args, complexity 11, 13 branches | `— REFACTOR Story 8.1` |
| `cli/main.py` | 44 | `train()` | PLR0913 | 6 args (Typer CLI surface) | `— Typer CLI options dictate arg count` |
| `evaluation/backtest.py` | 203 | `run_backtest()` | PLR0913 | 8 args | `— REFACTOR Story 8.1` |
| `evaluation/simulation.py` | 1042 | `simulate_tournament_mc()` | PLR0913 | 7 args | `— REFACTOR Story 8.1` |
| `evaluation/simulation.py` | 1218 | `simulate_tournament()` | PLR0913 | 8 args | `— REFACTOR Story 8.1` |
| `transform/elo.py` | 98 | `update_game()` | PLR0913 | 7 args (game dimensions) | `— game data has inherent dimensionality` |
| `transform/graph.py` | 243 | `add_game_to_graph()` | PLR0913 | 6 args (graph dimensions) | `— graph construction has inherent dimensionality` |

#### Dashboard Files

| File | Line | Function | Codes | Args | Annotation |
|------|------|----------|-------|------|------------|
| `dashboard/lib/filters.py` | 371 | `run_bracket_simulation()` | PLR0913 | 6 args | `— REFACTOR Story 8.1` |
| `dashboard/lib/filters.py` | 589 | `_game_win_probability()` | PLR0913 | 6 args | `— REFACTOR Story 8.1` |
| `dashboard/lib/bracket_renderer.py` | 192 | `_team_cell()` | PLR0913 | 6 args | `— REFACTOR Story 8.1` |
| `dashboard/lib/bracket_renderer.py` | 216 | `_render_region_html()` | PLR0913 | 6 args | `— REFACTOR Story 8.1` |

#### Test Files

| File | Line | Function | Codes | Args | Annotation |
|------|------|----------|-------|------|------------|
| `tests/unit/test_elo.py` | 16 | `_make_game()` | PLR0913 | 10 args | `— mirrors Game schema fields` |
| `tests/unit/test_feature_serving.py` | 120 | `_make_game()` | PLR0913 | 10 args | `— mirrors Game schema fields` |
| `tests/unit/test_dashboard_filters.py` | 614 | `test_returns_result_for_elo_model()` | PLR0913 | 9 args | `— @patch mock injection` |
| `tests/integration/test_elo_integration.py` | 25 | `_make_game()` | PLR0913 | 10 args | `— mirrors Game schema fields` |
| `tests/integration/test_feature_serving_integration.py` | 26 | `_make_game()` | PLR0913 | 10 args | `— mirrors Game schema fields` |

### Current Suppression Status

| Rule | Suppressions | Category Breakdown |
|------|-------------|-------------------|
| C901 (McCabe complexity > 10) | **1** | `run_training()` — REFACTOR Story 8.1 |
| PLR0911 (returns > 6) | **0** | |
| PLR0912 (branches > 12) | **1** | `run_training()` — REFACTOR Story 8.1 |
| PLR0913 (args > 5) | **14** | 8 REFACTOR Story 8.1, 3 domain-justified, 1 CLI constraint, 1 schema mirror, 1 mock injection |

All suppressions carry inline `# noqa` with descriptive annotations. PO approval has been
granted for all 16 current suppressions. Any new suppression of these codes requires
separate PO approval per the Lint Suppression Policy.

### Manual Nesting Review

20 files contain code at 4+ indentation levels. All instances were reviewed and found
to be **justified by domain logic** (parsing ESPN game data, feature block construction,
DataFrame operations, error handling). No refactoring needed.

**Notable well-structured complex modules:**
- `evaluation/simulation.py` — Inherently complex (bracket traversal, Monte Carlo sampling) but well-organized with clear interfaces
- `transform/feature_serving.py` — Batch feature computation with nested conditional blocks, each handling a distinct feature family
- `ingest/connectors/espn.py` — Multi-step parsing pipeline, each indentation level handles a distinct concern

---

## 4. PEP 20 #4: Complex > Complicated

The following modules are legitimately complex (not merely complicated):
- `evaluation/simulation.py` — Monte Carlo tournament simulation with bracket traversal, probability propagation, and seeded randomness
- `cli/train.py` — Training orchestration with model type dispatch, walk-forward backtesting, and Rich progress display

`cli/train.py:run_training()` is flagged as a **God Function** (complexity 11, 13 branches, 7 args). It should be decomposed into an orchestrator calling focused helper functions. **Tracked for refactoring in Story 8.1.**

---

## 5. PEP 20 #12: Refuse the Temptation to Guess

All files in `src/ncaa_eval/` use `from __future__ import annotations` and pass `mypy --strict`. No `Any` types escape into public APIs. Third-party library stubs (`numpy`, `pandas`, `xgboost`, `joblib`) use targeted `# type: ignore[import-untyped]` suppressions — acceptable per the Lint Suppression Policy.

---

## 6. Summary of Actions

| Action | Count | Details |
|--------|-------|---------|
| **Fixed in Story 8.9** | 2 | `backtest.py` NaN logging, `espn.py` debug→warning |
| **Inline noqa annotated** | 16 | All PLR0911/PLR0912/PLR0913/C901 suppressions carry descriptive rationale annotations |
| **Tracked for refactoring** | 9 | Functions with `— REFACTOR Story 8.1` annotation (God Function decomposition, simulation split) |
| **Domain-justified** | 5 | Functions with inherent dimensionality (game data, graph construction, CLI surface, schema mirrors) |
| **Deferred to Story 8.3** | 1 | `espn.py _parse_date()` silent None return |
| **Named constants deferred** | 5 | Should become constants in Stories 8.1/8.3 |
| **Acceptable as-is** | 13 | Inline domain constants with clear context |
| **Policy updated** | — | PLR0911/PLR0912/PLR0913/C901 now require PO approval; inline `# noqa` preferred over `per-file-ignores` |
