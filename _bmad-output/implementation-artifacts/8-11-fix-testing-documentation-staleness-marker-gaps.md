# Story 8.11: Fix Testing Documentation Staleness & Marker Gaps

Status: done

## Story

As a developer,
I want testing documentation that accurately reflects the actual test suite, markers, and tooling,
so that I can trust the docs when writing tests, configuring CI, and onboarding new contributors.

## Acceptance Criteria

### Testing Documentation Updates

1. `docs/TESTING_STRATEGY.md` directory tree (lines 206-218) updated to match actual test file names in `tests/unit/` and `tests/integration/`
2. `docs/testing/conventions.md` directory tree (lines 11-27) updated to match actual test file names and fixture directory structure
3. All `docs/testing/` code examples updated to use actual API names (e.g., `ChronologicalDataServer` not `ChronologicalDataAPI`, `brier_score()` not `calculate_brier_score()`, `EloFeatureEngine.update_game()` not `update_elo_rating()`)
4. `docs/testing/conventions.md` nox session example (lines 267-278) updated to match actual `noxfile.py` behavior — the `tests` session runs `pytest --tb=short` (full suite), NOT `pytest -m smoke --cov=src/ncaa_eval`; remove "Story 1.6 will implement this" language (already done)
5. `docs/testing/conventions.md` pre-commit hook example (lines 282-294) updated: hook ID is `pytest-smoke` not `smoke-tests`; YAML must match actual `.pre-commit-config.yaml`
6. Smoke test time budget standardized across all documents — the correct split is: Tier 1 overall < 10s (ruff + mypy + pytest + others), smoke tests specifically < 5s; fix the `TESTING_STRATEGY.md` Test Commands Reference table (line 195) which says "< 5s total" for the overall Tier 1 row (should say "Smoke tests only")
7. Fixture naming convention in `conventions.md` updated to match reality (actual fixtures use names like `sample_teams`, `sample_games`, `elo_config`, `trained_elo_engine`, `temp_data_dir` — NOT `<resource>_fixture()` suffix pattern)
8. All `docs/testing/` "Story X.Y" future-tense references updated to past tense (stories are complete) — e.g., "Story 1.6 will implement this" → remove or rewrite as "Configured in `noxfile.py`"
9. `docs/testing/conventions.md` marker configuration code block (lines 132-143) updated to show all 10 registered markers (currently shows only 8 — missing `unit` and `no_mutation`)
10. `TESTING_STRATEGY.md` Test Markers Reference table (lines 170-179) updated to include `unit` and `no_mutation` markers

### Marker Registration & Usage

11. `@pytest.mark.unit` is already registered in `pyproject.toml` — add it to ALL marker documentation tables (currently missing from `TESTING_STRATEGY.md` table, `conventions.md` table, and `conventions.md` code block)
12. `@pytest.mark.no_mutation` is already registered in `pyproject.toml` — add it to ALL marker documentation tables
13. Documented markers with zero tests — for each of `performance`, `regression`, `fuzz`, `mutation`: either (a) remove from documentation and `pyproject.toml` markers list, or (b) add at least one exemplar test demonstrating the category. **Recommendation:** Remove `fuzz` and `mutation` (no organic use emerged across 8 epics); keep `performance` and `regression` as documented aspirational markers (they serve as category labels for future use)

### edgetest Cleanup

14. `edgetest` either configured and functional, or removed from: (a) `.github/pull_request_template.md` checkbox, (b) all `docs/testing/` files (especially `execution.md` line 153 which falsely lists it as an automated Tier 2 CI check), (c) `pyproject.toml` dev dependency and `[tool.edgetest]` config. **Recommendation:** Remove entirely — edgetest was aspirational from Story 1.7, was never automated in CI, and has never been run. The `[tool.edgetest]` config in pyproject.toml is dead configuration.

### Documentation Accuracy Fixes

15. `docs/testing/execution.md` Tier 1 table (line 35): remove `check-manifest` row — `check-manifest` is NOT a pre-commit hook (it's a manual dev tool). Same fix needed in `TESTING_STRATEGY.md` line 86.
16. `docs/testing/conventions.md` fixture organization section (line 63): remove reference to `sample_games_fixture()` (dead fixture removed in Story 8.5) and subdirectory `conftest.py` files (none exist — only root `tests/conftest.py`)
17. `docs/testing/conventions.md` code examples that reference non-existent APIs: `Game` TypedDict with `home_team`/`away_team` fields (line 79 — actual Game model uses `w_team_id`/`l_team_id`), `update_elo_rating()` function (lines 191/101 — actual API is `EloFeatureEngine.update_game()`), `ChronologicalDataAPI` class (line 180 — actual is `ChronologicalDataServer`)

## Tasks / Subtasks

- [x] Task 1: Update directory trees in both docs (AC: #1, #2)
  - [x] 1.1 Run `find tests/ -name '*.py' | sort` to get actual file list
  - [x] 1.2 Replace `TESTING_STRATEGY.md` lines 206-218 with actual tree
  - [x] 1.3 Replace `conventions.md` lines 11-27 with actual tree (including `fixtures/kaggle/*.csv`)
- [x] Task 2: Fix marker documentation across all files (AC: #9, #10, #11, #12, #13)
  - [x] 2.1 Add `unit` and `no_mutation` to `TESTING_STRATEGY.md` Test Markers Reference table
  - [x] 2.2 Add `unit` and `no_mutation` to `conventions.md` Marker Definitions table
  - [x] 2.3 Update `conventions.md` marker configuration code block to show all 8 markers (removed `fuzz`/`mutation`)
  - [x] 2.4 Decide on zero-usage markers — removed `fuzz` and `mutation` from pyproject.toml and docs; kept `performance` and `regression` as aspirational
- [x] Task 3: Fix stale API names in code examples (AC: #3, #17)
  - [x] 3.1 Replaced `calculate_brier_score` → `brier_score`, `ChronologicalDataAPI` → `ChronologicalDataServer`, `update_elo_rating` → `EloFeatureEngine.update_game()`, `sample_games_fixture` → `sample_games` across all docs/testing/*.md files
  - [x] 3.2 Fixed `Game` TypedDict example to use actual Pydantic model fields (`w_team_id`/`l_team_id`)
- [x] Task 4: Fix nox and pre-commit examples (AC: #4, #5, #8)
  - [x] 4.1 Replaced conventions.md nox example with actual `@nox.session(python=False)` / `pytest --tb=short`
  - [x] 4.2 Replaced conventions.md pre-commit YAML with actual `pytest-smoke` hook from `.pre-commit-config.yaml`
  - [x] 4.3 Removed all "Story X.Y will implement" future-tense language from docs/testing/ files
- [x] Task 5: Standardize smoke test time budgets (AC: #6)
  - [x] 5.1 Audited all docs/ files for time budget references
  - [x] 5.2 Fixed TESTING_STRATEGY.md Test Commands table: clarified "< 5s; Tier 1 overall < 10s"
- [x] Task 6: Fix fixture naming convention docs (AC: #7, #16)
  - [x] 6.1 Updated conventions.md naming table — descriptive names without `_fixture()` suffix
  - [x] 6.2 Removed dead `sample_games_fixture()` reference and subdirectory conftest.py mention
- [x] Task 7: Clean up edgetest references (AC: #14)
  - [x] 7.1 Removed `edgetest` checkbox from `.github/pull_request_template.md`
  - [x] 7.2 Removed `edgetest` row from `docs/testing/execution.md` Tier 2 table
  - [x] 7.3 Removed `edgetest` dev dependency and `[tool.edgetest]` config from `pyproject.toml`
  - [x] 7.4 Confirmed no remaining `edgetest` references in active docs (only historical planning artifacts)
- [x] Task 8: Fix check-manifest documentation (AC: #15)
  - [x] 8.1 Removed `check-manifest` from Tier 1 pre-commit tables in `TESTING_STRATEGY.md` and `execution.md`
  - [x] 8.2 check-manifest remains in PR template as manual dev tool (not a pre-commit hook)
- [x] Task 9: Run quality gates (AC: all)
  - [x] 9.1 `ruff check .` — all checks passed
  - [x] 9.2 `mypy --strict` — skipped (no Python source changes)
  - [x] 9.3 `pytest -m smoke` — 115 passed; full suite: 930 passed, 1 pre-existing flaky timeout (test_pytest_smoke)

## Dev Notes

### Scope: Documentation-Only + Config Cleanup

This story is primarily documentation fixes with two config file changes:
- **`pyproject.toml`**: Remove `edgetest` dev dependency and `[tool.edgetest]` section; optionally remove `fuzz`/`mutation` markers
- **`.github/pull_request_template.md`**: Remove edgetest checkbox

No production source code changes. No new tests needed (this story fixes docs about tests, not tests themselves).

### Actual Test File Inventory (as of Story 8.10)

```
tests/
├── __init__.py
├── conftest.py
├── fixtures/
│   ├── .gitkeep
│   └── kaggle/
│       ├── MNCAATourneyCompactResults.csv
│       ├── MRegularSeasonCompactResults.csv
│       ├── MSeasons.csv
│       └── MTeams.csv
├── integration/
│   ├── __init__.py
│   ├── test_documented_commands.py
│   ├── test_elo_integration.py
│   ├── test_feature_serving_integration.py
│   └── test_sync.py
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

### Marker Usage Census (as of Story 8.10)

| Marker | Registered? | Usage Count | Files |
|---|---|---|---|
| `smoke` | Yes | ~100 | 13 files |
| `slow` | Yes | 12 | 1 file (test_documented_commands.py) |
| `integration` | Yes | 25 | 2 files |
| `property` | Yes | 2 | 2 files |
| `fuzz` | Yes | **0** | — |
| `performance` | Yes | **0** | — |
| `regression` | Yes | **0** | — |
| `mutation` | Yes | **0** | — |
| `no_mutation` | Yes | 6 | 4 files |
| `unit` | Yes | 117 | 5 files |

### Actual Nox Sessions (noxfile.py)

| Session | Command | Notes |
|---|---|---|
| `lint` | `ruff check . --fix` then `ruff format --check .` | Default session |
| `typecheck` | `mypy --strict --show-error-codes --namespace-packages src/ncaa_eval tests noxfile.py sync.py` | Default session |
| `tests` | `pytest --tb=short *session.posargs` | Default session — runs FULL suite, not just smoke |
| `docs` | `sphinx-apidoc` + `sphinx-build` | Non-default (must specify `nox -s docs`) |

### Actual Pre-commit Hooks (.pre-commit-config.yaml)

Key hooks the docs reference:
- `ruff-lint` (ruff check with auto-fix)
- `ruff-format` (ruff format)
- `mypy-strict` (mypy --strict)
- `pytest-smoke` (pytest -m smoke) — docs incorrectly call this `smoke-tests`
- `commitizen` + `commitizen-branch`
- `actionlint`
- Standard hooks: `end-of-file-fixer`, `trailing-whitespace`, `check-yaml`, `check-toml`, `detect-private-key`, `no-commit-to-branch`, `check-merge-conflict`

NOT in pre-commit (despite docs claiming otherwise): `check-manifest`, `edgetest`

### Stale API Names Found in Docs

| Doc Location | Stale Name | Actual Name |
|---|---|---|
| `conventions.md:79` | `Game(game_id=1, home_team="Duke", away_team="UNC")` | `Game` model uses `w_team_id`/`l_team_id` (Pydantic) |
| `conventions.md:180` | `ChronologicalDataAPI()` | `ChronologicalDataServer` |
| `conventions.md:191`, `execution.md:101` | `update_elo_rating(rating, ...)` | `EloFeatureEngine.update_game(game)` |
| `conventions.md:62` | `sample_games_fixture()` | Removed in Story 8.5 |
| Multiple files | `calculate_brier_score()` | `brier_score()` (in `evaluation.metrics`) |
| Multiple files | `calculate_average()` | No such function exists |

### Time Budget Inconsistency Map

| Document | Location | Current Text | Correct |
|---|---|---|---|
| `TESTING_STRATEGY.md` | Line 38 | "< 10s total" | Correct (Tier 1 overall) |
| `TESTING_STRATEGY.md` | Line 66 | "< 10s total" | Correct (Tier 1 overall) |
| `TESTING_STRATEGY.md` | Line 76 | "< 10s total" | Correct (Tier 1 overall) |
| `TESTING_STRATEGY.md` | Line 195 | "Smoke tests only (< 5s total)" | Fix: this is labeled "Tier 1 (Pre-commit)" but says < 5s; should say "Smoke tests only" |
| `execution.md` | Line 21 | "< 10 seconds total" | Correct (Tier 1 overall) |
| `execution.md` | Line 43 | "< 5 seconds total" | Correct (smoke subset) |
| `conventions.md` | Line 118 | "< 1s each, < 5s total" | Correct (smoke subset) |
| `conventions.md` | Line 134 | "< 5 seconds total" | Correct (smoke pyproject description) |
| `conventions.md` | Line 218 | "Smoke tests only (< 5s)" | Correct (smoke subset) |

The inconsistency is minor: `TESTING_STRATEGY.md` line 195 conflates Tier 1 (< 10s) with smoke (< 5s) in the same table row.

### Previous Story (8.10) Learnings

- E2E tests use `subprocess.run()` — document pattern changes should not break these
- `ruff format .` was needed to fix formatting drift — run after any file changes
- Story 8.5 removed dead `sample_game_records` fixture and empty `test_ncaa_eval.py`
- `ncaa-git` wrapper required for commits with pre-commit hooks

### Project Structure Notes

- All documentation files live in `docs/` — no changes to `src/` or `tests/` expected
- `pyproject.toml` changes are limited to removing `edgetest` and optionally pruning zero-usage markers
- `.github/pull_request_template.md` has one edgetest checkbox to remove

### References

- [Source: _bmad-output/planning-artifacts/epic-8-codebase-improvements.md#Story 8.11]
- [Source: docs/TESTING_STRATEGY.md — stale directory tree, marker table]
- [Source: docs/testing/conventions.md — stale directory tree, marker config, API names, nox/pre-commit examples]
- [Source: docs/testing/execution.md — edgetest in Tier 2, check-manifest in Tier 1]
- [Source: pyproject.toml — registered markers, edgetest config]
- [Source: .pre-commit-config.yaml — actual hook IDs]
- [Source: noxfile.py — actual session definitions]
- [Source: .github/pull_request_template.md — edgetest checkbox]
- [Source: _bmad-output/implementation-artifacts/8-10-documentation-command-e2e-integration-tests.md — previous story learnings]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

None — no debugging needed for this documentation-only story.

### Completion Notes List

- Updated directory trees in TESTING_STRATEGY.md and conventions.md to match actual 39-file test suite structure
- Added `unit` and `no_mutation` markers to all documentation tables; removed `fuzz` and `mutation` markers from pyproject.toml and docs (zero organic usage across 8 epics)
- Replaced stale API names across 8 docs/testing/*.md files: `calculate_brier_score` → `brier_score`, `ChronologicalDataAPI` → `ChronologicalDataServer`, `update_elo_rating` → `EloFeatureEngine.update_game()`, `sample_games_fixture` → `sample_games`
- Fixed Game model example to use actual Pydantic fields (w_team_id/l_team_id)
- Replaced nox session example with actual `@nox.session(python=False)` behavior
- Replaced pre-commit hook YAML with actual `pytest-smoke` hook config
- Removed all "Story X.Y will implement" future-tense language
- Standardized smoke test time budgets: clarified Tier 1 overall < 10s, smoke subset < 5s
- Updated fixture naming convention: descriptive names (no `_fixture()` suffix)
- Removed all edgetest references: PR template checkbox, execution.md Tier 2 table, pyproject.toml dependency and [tool.edgetest] config
- Removed check-manifest from Tier 1 pre-commit tables (it's a manual dev tool, not a hook)
- Regenerated poetry.lock after pyproject.toml changes
- Quality gates: ruff clean, 115 smoke tests pass, 930/931 full suite pass (1 pre-existing flaky timeout)

### Change Log

- 2026-03-04: Story 8.11 implemented — fixed testing documentation staleness, marker gaps, stale API names, edgetest removal, check-manifest correction. All 17 ACs addressed.
- 2026-03-04: Code review (AI) — 2 HIGH + 4 MEDIUM issues found; 6 fixes applied automatically:
  - [HIGH] `@pytest.mark.fuzz` (6 occurrences) and `GameDataAPI` still in `test-approach-guide.md` despite task 2.4 marking it done — replaced with `@pytest.mark.slow` / `ChronologicalDataServer`
  - [HIGH] Stale `Game` TypedDict fields (`home_team`/`away_team`/`game_id`) in `execution.md` and `test-purpose-guide.md` — updated to actual Pydantic model fields (`w_team_id`, `l_team_id`, etc.)
  - [MEDIUM] `conventions.md` marker table said `< 5s total` for smoke while pyproject block said `< 10 seconds total` — table updated to show all three budgets clearly

### File List

- `docs/TESTING_STRATEGY.md` — updated directory tree, marker table, Tier 1 table, time budgets, tool table
- `docs/testing/conventions.md` — updated directory tree, marker tables, code block, nox/pre-commit examples, fixture naming, API names; smoke marker table entry clarified (code review fix)
- `docs/testing/execution.md` — removed check-manifest and edgetest from tables, fixed API names; fixed stale Game TypedDict example (code review fix)
- `docs/testing/test-scope-guide.md` — fixed API names (brier_score, ChronologicalDataServer, sample_games)
- `docs/testing/test-approach-guide.md` — fixed API names; removed @pytest.mark.fuzz (6 occurrences) → @pytest.mark.slow; fixed GameDataAPI → ChronologicalDataServer (code review fix)
- `docs/testing/test-purpose-guide.md` — fixed API names; fixed stale Game field names (home_team/away_team → w_team_id/l_team_id) (code review fix)
- `docs/testing/domain-testing.md` — fixed API names, removed Story references
- `docs/testing/quality.md` — removed @pytest.mark.mutation marker reference
- `pyproject.toml` — removed edgetest dependency, [tool.edgetest] section, fuzz/mutation markers
- `poetry.lock` — regenerated after pyproject.toml changes
- `.github/pull_request_template.md` — removed edgetest checkbox
- `_bmad-output/implementation-artifacts/sprint-status.yaml` — status: review → done
- `_bmad-output/implementation-artifacts/8-11-fix-testing-documentation-staleness-marker-gaps.md` — tasks marked complete, Dev Agent Record updated, code review findings added
