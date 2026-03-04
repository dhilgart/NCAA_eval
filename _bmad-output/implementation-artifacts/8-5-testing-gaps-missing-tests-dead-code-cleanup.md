# Story 8.5: Testing Gaps — Missing Tests & Dead Code Cleanup

Status: ready-for-dev

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

- [ ] Task 1: Verify `scoring_from_config` coverage (AC: #1)
  - [ ] Read `tests/unit/test_evaluation_simulation.py` and confirm all 5 branches of `scoring_from_config` are tested
  - [ ] If any branch is missing, add parameterized test covering it
  - [ ] Confirm unknown-type error path is tested
  - [ ] **Expected result:** AC #1 may already be satisfied — Story 8.4 agent reported 11 tests covering all 5 branches. Verify and document.

- [ ] Task 2: Add CLI training tests for XGBoost model (AC: #2)
  - [ ] Read `tests/unit/test_cli_train.py` to understand existing test patterns
  - [ ] Read `src/ncaa_eval/cli/train.py` to understand model dispatch logic
  - [ ] Add `test_train_xgboost()` using the same `_mock_serve_season_features` helper
  - [ ] Ensure the test exercises the stateless model path (XGBoost uses `Model.fit()` directly)
  - [ ] Use `@pytest.mark.smoke` if test runs in <1s, otherwise `@pytest.mark.unit`

- [ ] Task 3: Add CLI training tests for Elo model (AC: #3)
  - [ ] Add `test_train_elo()` exercising the stateful model path
  - [ ] Elo's `fit()` is inherited from `StatefulModel` — it reconstructs `Game` objects from X and calls `update()` per game
  - [ ] The mock feature data must include columns that `StatefulModel.fit()` expects for Game reconstruction
  - [ ] Read `src/ncaa_eval/model/base.py` `StatefulModel.fit()` to understand required columns
  - [ ] Use `@pytest.mark.smoke` if test runs in <1s, otherwise `@pytest.mark.unit`

- [ ] Task 4: Remove dead `sample_game_records` fixture (AC: #4)
  - [ ] Open `tests/conftest.py`, locate `sample_game_records` fixture (lines ~34-75)
  - [ ] Verify it is truly unused: grep all test files for `sample_game_records`
  - [ ] Remove the fixture
  - [ ] Ensure no imports break

- [ ] Task 5: Remove empty `tests/test_ncaa_eval.py` (AC: #5)
  - [ ] Delete `tests/test_ncaa_eval.py` (confirmed empty — 0 lines)
  - [ ] Verify no other file imports from it

- [ ] Task 6: Add Fibonacci scoring point-value test (AC: #6)
  - [ ] Read `src/ncaa_eval/evaluation/scoring.py` to find `FibonacciScoringRule` and its actual point values
  - [ ] Add `test_fibonacci_scoring_point_values()` in `tests/unit/test_evaluation_simulation.py`
  - [ ] Assert each round's exact points match the implemented values
  - [ ] Note: The Fibonacci values may be (1,1,2,3,5,8) or (2,3,5,8,13,21) — PO decision pending in Story 8.13. Test whatever the code currently implements.

- [ ] Task 7: Run quality gates (AC: #7)
  - [ ] Run `pytest` — all tests pass
  - [ ] Run `ruff check .` — no violations
  - [ ] Run `mypy --strict src/ncaa_eval tests` — no type errors

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

{{agent_model_name_version}}

### Debug Log References

### Completion Notes List

### File List
