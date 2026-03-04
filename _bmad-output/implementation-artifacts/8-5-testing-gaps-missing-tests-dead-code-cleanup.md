# Story 8.5: Testing Gaps — Missing Tests & Dead Code Cleanup

Status: done

## Story

As a developer,
I want to close testing gaps for exported functions, test the CLI with all registered model types, and remove dead test code,
so that the test suite provides comprehensive coverage of production code paths and contains no misleading dead artifacts.

## Acceptance Criteria

1. `scoring_from_config` has parameterized tests covering all 5 dispatch branches + unknown-type error
2. CLI training pipeline tested with XGBoost model type (in addition to existing logistic_regression tests)
3. CLI training pipeline tested with Elo model type (stateful path)
4. Dead `sample_game_records` fixture removed from `tests/conftest.py`
5. Empty `tests/test_ncaa_eval.py` file removed
6. Fibonacci scoring test added asserting actual point values match documented values
7. All tests pass; no regressions

## Tasks / Subtasks

- [x] Task 1: Verify `scoring_from_config` coverage (AC: #1)
  - [x] Read `tests/unit/test_evaluation_simulation.py` and confirm all 5 branches of `scoring_from_config` are tested
  - [x] If any branch is missing, add parameterized test covering it
  - [x] Confirm unknown-type error path is tested
  - [x] **Expected result:** AC #1 may already be satisfied — Story 8.4 agent reported 11 tests covering all 5 branches. Verify and document.

- [x] Task 2: Add CLI training tests for XGBoost model (AC: #2)
  - [x] Read `tests/unit/test_cli_train.py` to understand existing test patterns
  - [x] Read `src/ncaa_eval/cli/train.py` to understand model dispatch logic
  - [x] Add `test_train_xgboost()` using the same `_mock_serve_season_features` helper
  - [x] Ensure the test exercises the stateless model path (XGBoost uses `Model.fit()` directly)
  - [x] Use `@pytest.mark.smoke` if test runs in <1s, otherwise `@pytest.mark.unit`

- [x] Task 3: Add CLI training tests for Elo model (AC: #3)
  - [x] Add `test_train_elo()` exercising the stateful model path
  - [x] Elo's `fit()` is inherited from `StatefulModel` — it reconstructs `Game` objects from X and calls `update()` per game
  - [x] The mock feature data must include columns that `StatefulModel.fit()` expects for Game reconstruction
  - [x] Read `src/ncaa_eval/model/base.py` `StatefulModel.fit()` to understand required columns
  - [x] Use `@pytest.mark.smoke` if test runs in <1s, otherwise `@pytest.mark.unit`

- [x] Task 4: Remove dead `sample_game_records` fixture (AC: #4)
  - [x] Open `tests/conftest.py`, locate `sample_game_records` fixture (lines ~34-75)
  - [x] Verify it is truly unused: grep all test files for `sample_game_records`
  - [x] Remove the fixture
  - [x] Ensure no imports break

- [x] Task 5: Remove empty `tests/test_ncaa_eval.py` (AC: #5)
  - [x] Delete `tests/test_ncaa_eval.py` (confirmed empty — 0 lines)
  - [x] Verify no other file imports from it

- [x] Task 6: Add Fibonacci scoring point-value test (AC: #6)
  - [x] Read `src/ncaa_eval/evaluation/scoring.py` to find `FibonacciScoringRule` and its actual point values
  - [x] Add `test_fibonacci_scoring_point_values()` in `tests/unit/test_evaluation_simulation.py`
  - [x] Assert each round's exact points match the implemented values
  - [x] Note: The Fibonacci values may be (1,1,2,3,5,8) or (2,3,5,8,13,21) — PO decision pending in Story 8.13. Test whatever the code currently implements.

- [x] Task 7: Run quality gates (AC: #7)
  - [x] Run `pytest` — all tests pass
  - [x] Run `ruff check .` — no violations
  - [x] Run `mypy --strict src/ncaa_eval tests` — no type errors

## Dev Notes

### Key Code Locations

| Purpose | File Path |
|---|---|
| CLI train module | `src/ncaa_eval/cli/train.py` |
| CLI train tests | `tests/unit/test_cli_train.py` |
| Scoring module | `src/ncaa_eval/evaluation/scoring.py` |
| Scoring tests | `tests/unit/test_evaluation_simulation.py` |
| Model ABC | `src/ncaa_eval/model/base.py` |
| Elo model | `src/ncaa_eval/model/elo.py` |
| XGBoost model | `src/ncaa_eval/model/xgboost_model.py` |
| Test conftest | `tests/conftest.py` |
| Empty test file | `tests/test_ncaa_eval.py` |
| Model registry | `src/ncaa_eval/model/registry.py` |

### Critical Implementation Details

**CLI Test Pattern (existing):** All CLI tests in `test_cli_train.py` use a `_mock_serve_season_features()` helper that produces synthetic feature DataFrames. Reuse this pattern for XGBoost and Elo tests — do NOT create a new fixture pattern.

**StatefulModel.fit() Column Requirements:** `StatefulModel.fit(X, y)` reconstructs `Game` objects from X columns. Read `StatefulModel.fit()` in `model/base.py` to identify which columns (e.g., `season`, `day_num`, `w_team_id`, `l_team_id`, `w_score`, `l_score`, `loc`, `num_ot`) are required. The mock data for Elo tests must include these columns or the test will fail with a KeyError.

**XGBoost Test Isolation:** XGBoost tests should mock or use minimal data to keep test runtime fast. Use `n_estimators=2, max_depth=1` or similar minimal config to prevent slow training.

**scoring_from_config Branches:** The 5 dispatch branches are: `"standard"`, `"fibonacci"`, `"seed_diff_bonus"`, `"dict"` (custom point dict), and `"custom"` (callable). The explore agent confirmed 11 existing tests cover all 5 + error cases — verify this before writing duplicate tests.

**Fibonacci Values:** Check the actual implementation in `FibonacciScoringRule`. The test should assert the exact round-by-round point values (whatever the code returns for rounds 1-6), not just that the object instantiates.

### Anti-Patterns to Avoid

- Do NOT add tests for `scoring_from_config` if they already exist and cover all branches — verify first
- Do NOT use `iterrows()` in any test data construction — use vectorized DataFrame construction
- Do NOT add `from __future__ import annotations` exemptions — it IS required in test files per project convention
- Do NOT add new markers without registering them in `pyproject.toml` `[tool.pytest.ini_options]` markers list
- Do NOT mock the model registry — import real model classes to ensure registry wiring is tested end-to-end
- Do NOT add integration test markers to these tests — they are unit tests (no I/O, no external deps)

### Previous Story Intelligence

**From Story 8.3:** Established the pattern of using `tenacity` for retry logic and centralized `fuzzy_match_team()` — no impact on this story.

**From Story 8.4:** Fixed all docstring style violations and enabled Ruff D-rule enforcement. All new test code must use Google-style docstrings for any module/class/function docstrings.

**From Story 8.9:** Added PEP 20, SOLID, and pure-function gates to PR template. New tests should follow these principles — no silent exception swallowing in test helpers, single responsibility per test function.

### Project Structure Notes

- All test files live under `tests/unit/` or `tests/integration/`
- Test file naming: `test_<module>.py`
- The dead `tests/test_ncaa_eval.py` is at the top level — NOT in `tests/unit/`
- `conftest.py` is at `tests/conftest.py` (shared fixtures for all tests)

### References

- [Source: _bmad-output/planning-artifacts/epic-8-codebase-improvements.md#Story 8.5]
- [Source: _bmad-output/planning-artifacts/codebase-audit-report.md] — Audit items 3.21-3.24, 1.8
- [Source: docs/STYLE_GUIDE.md] — Google-style docstrings, Vectorization First rule
- [Source: docs/testing/conventions.md] — Test marker taxonomy, fixture naming

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

- Elo CLI test initially failed due to `_make_synthetic_season` generating `w_score <= l_score` which violates the `Game` Pydantic validator. Fixed by adjusting score ranges: `w_score` starts at 75 (min 75), `l_score` caps at 69 (55+14).

### Completion Notes List

- **AC #1 (scoring_from_config):** Verified — 10 existing tests in `TestDictScoring` already cover all 5 dispatch branches (standard, fibonacci, seed_diff_bonus, dict, custom) plus error paths (unknown type, missing type key, missing seed_map, missing points, missing callable). No new tests needed.
- **AC #2 (XGBoost CLI test):** Added `test_train_xgboost()` — exercises stateless model path with minimal config (`n_estimators=2, max_depth=1, early_stopping_rounds=1`) for fast execution. Verifies `model_type == "xgboost"` in persisted run.
- **AC #3 (Elo CLI test):** Added `test_train_elo()` — exercises stateful model path through `StatefulModel.fit()` which reconstructs `Game` objects. Fixed `_make_synthetic_season` to guarantee `w_score > l_score`. Verifies `model_type == "elo"` in persisted run.
- **AC #4 (dead fixture):** Removed `sample_game_records` fixture from `tests/conftest.py` — grep confirmed zero usages across all test files.
- **AC #5 (empty test file):** Deleted `tests/test_ncaa_eval.py` — was empty (0 lines), no imports from it.
- **AC #6 (Fibonacci point values):** Verified — existing `test_fibonacci_scoring_values` already asserts exact point values (2.0, 3.0, 5.0, 8.0, 13.0, 21.0) for all 6 rounds. `test_fibonacci_perfect_bracket` confirms total of 231.
- **AC #7 (quality gates):** All pass — 886 tests passed (1 skipped), mypy --strict clean. `ruff check .` is clean for `src/` and `tests/`; 21 pre-existing notebook violations in `notebooks/eda/` are tracked separately (pre-exist on `main`, not introduced by this story).

### Change Log

- 2026-03-03: Story 8.5 implementation complete — added XGBoost + Elo CLI tests, removed dead fixture + empty test file, verified existing coverage for scoring_from_config and Fibonacci point values.
- 2026-03-03: Code review fixes (round 1) — added `@pytest.mark.unit` to `test_train_xgboost` and `test_train_elo`; added `import pytest`; changed module docstring from "Integration" to "Unit"; fixed fragile random `team_a_won` labels to deterministic alternating pattern; added `model.ubj` artifact assertion to XGBoost test; added `start_year`/`end_year` assertions to Elo test.
- 2026-03-03: Code review fixes (round 2) — registered `unit` marker in `pyproject.toml` (was used across 4 test files without being formally registered); added `model/` directory + `feature_names.json` artifact assertions to `test_train_elo` to prove stateful fit() path completed.

### File List

- `tests/unit/test_cli_train.py` — modified (added `test_train_xgboost`, `test_train_elo`; fixed `w_score`/`l_score` ranges in `_make_synthetic_season`; added Elo model artifact assertions)
- `tests/conftest.py` — modified (removed dead `sample_game_records` fixture)
- `tests/test_ncaa_eval.py` — deleted (was empty)
- `pyproject.toml` — modified (registered `unit` marker in `[tool.pytest.ini_options]` markers list)
