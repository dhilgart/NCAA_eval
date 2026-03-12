# Story 9.12: Implement st.progress for Simulation

Status: done

## Story

As a **bracket pool participant**,
I want to **see a numeric progress bar during Monte Carlo simulation instead of a spinner**,
so that **I know how long the simulation will take and can see it progressing**.

## Acceptance Criteria

1. **Given** the user triggers a Monte Carlo simulation on the Pool Scorer page,
   **When** the simulation engine runs 10k+ iterations,
   **Then** a `st.progress()` bar displays the current iteration count.

2. **Given** the simulation engine is running,
   **When** each round completes,
   **Then** the progress bar updates to reflect `(round_completed / total_rounds)` as a fraction.

3. **And** the simulation engine exposes an iteration callback for progress reporting.

4. **And** the `st.spinner()` is replaced with `st.progress()` for the simulation step on both the Pool Scorer page (`4_Pool_Scorer.py`) and the Presentation page (`2_Presentation.py`).

5. **Given** the simulation uses the analytical method (not Monte Carlo),
   **When** the computation runs,
   **Then** the existing `st.spinner()` behavior is preserved (analytical is near-instant, no progress bar needed).

## Tasks / Subtasks

- [x] Task 1: Add `progress_callback` parameter to simulation engine (AC: #3)
  - [x] 1.1: Add `progress_callback: Callable[[int, int], None] | None = None` parameter to `simulate_tournament_mc()` in `src/ncaa_eval/evaluation/simulation.py`
  - [x] 1.2: Call `progress_callback(round_completed, total_rounds)` after each round in the MC loop (alongside existing `tqdm` support)
  - [x] 1.3: Add `progress_callback` parameter to `simulate_tournament()` orchestrator and pass through to MC path
  - [x] 1.4: Update docstrings for both functions

- [x] Task 2: Create uncached simulation wrapper with progress support (AC: #1, #2)
  - [x] 2.1: In `dashboard/lib/simulation_helpers.py`, create a new function `run_bracket_simulation_with_progress()` that duplicates the MC path of `run_bracket_simulation()` but is NOT decorated with `@st.cache_data`
  - [x] 2.2: The new function accepts an `st.progress` bar object and passes a lambda callback to `simulate_tournament()` that updates the progress bar
  - [x] 2.3: Reuse all existing bracket-building, provider-creation, and scoring logic from the cached function (call shared helpers, do NOT duplicate)

- [x] Task 3: Replace `st.spinner` with `st.progress` on Pool Scorer page (AC: #1, #4)
  - [x] 3.1: In `dashboard/pages/4_Pool_Scorer.py` `_run_simulation()`, replace `st.spinner("Running Monte Carlo simulation...")` with an `st.progress(0, text="Running Monte Carlo simulation...")` bar
  - [x] 3.2: Call the new uncached simulation function, passing the progress bar
  - [x] 3.3: Set progress to 1.0 and clear after simulation completes

- [x] Task 4: Replace `st.spinner` with `st.progress` on Presentation page (AC: #4, #5)
  - [x] 4.1: In `dashboard/pages/2_Presentation.py`, replace the MC spinner with `st.progress` only when `method == "monte_carlo"`
  - [x] 4.2: Keep `st.spinner` for analytical method (near-instant, no progress needed)

- [x] Task 5: Update tests (AC: all)
  - [x] 5.1: Add unit tests for `progress_callback` in `test_evaluation_simulation.py` — verify callback is called once per round with correct `(round, total)` args
  - [x] 5.2: Update `test_pool_scorer_page.py` — verify `st.progress` is called instead of `st.spinner` for MC simulation path
  - [x] 5.3: Verify `st.spinner` is still used for analytical path on Presentation page

### Review Follow-ups (AI)

- [ ] [AI-Review][MEDIUM] `run_bracket_simulation_with_progress` catches `(OSError, ValueError, KeyError, TypeError)` but MC path can also raise `RuntimeError` (numpy shape mismatch, malformed bracket tree) or `AttributeError` (bad provider) — add to catch list or broaden to `Exception` [dashboard/lib/simulation_helpers.py:378]
- [ ] [AI-Review][MEDIUM] `run_bracket_simulation_with_progress` passes `n_simulations` to the cached `run_bracket_simulation(method="analytical")` call — for analytical, this param is ignored but still pollutes the cache key. Pass a fixed value (e.g., `n_simulations=10_000`) or omit it to prevent duplicate analytical cache entries [dashboard/lib/simulation_helpers.py:297-306]

## Dev Notes

### Architecture & Design Decisions

**Callback pattern over `tqdm` replacement:** The existing `simulate_tournament_mc()` uses `tqdm` for console progress (`progress: bool = False`). Rather than replacing `tqdm`, add a parallel `progress_callback` parameter. This keeps CLI `tqdm` working and adds a clean hook for Streamlit's `st.progress`. The callback signature `Callable[[int, int], None]` passes `(round_completed, total_rounds)` — simple, testable, UI-agnostic.

**Caching incompatibility:** `run_bracket_simulation()` is decorated with `@st.cache_data(ttl=None)`. Cached functions cannot display progress because Streamlit skips the function body on cache hit. The solution: create a thin uncached wrapper that handles the progress bar, delegates bracket-building to shared helpers, and calls `simulate_tournament()` directly. The cached path remains for repeated renders within the same session.

**Round-based granularity:** The MC engine iterates 6 rounds (log2(64) = 6). With 10k simulations, each round takes ~0.1-0.5s. Progress updates at round boundaries (1/6, 2/6, ..., 6/6) provide meaningful feedback. Per-simulation callbacks would be too granular and add overhead.

**`st.progress` API:** `st.progress(value, text=None)` where `value` is a float 0.0–1.0. The `text` parameter (added in Streamlit 1.18) displays a message above the bar. Use `text=f"Simulating round {r+1}/{n_rounds}..."` for user feedback. After completion, call `progress_bar.empty()` to remove the bar from the UI.

### Codebase Context

**Simulation call chain:**
1. `dashboard/pages/4_Pool_Scorer.py:_run_simulation()` → `run_bracket_simulation()` (cached)
2. `dashboard/pages/2_Presentation.py` (line 335) → `run_bracket_simulation()` (cached)
3. `dashboard/lib/simulation_helpers.py:run_bracket_simulation()` → `simulate_tournament()` (core lib)
4. `src/ncaa_eval/evaluation/simulation.py:simulate_tournament()` → `simulate_tournament_mc()` (MC path)
5. `simulate_tournament_mc()` — round loop at line 603

**Existing `progress` parameter:** Both `simulate_tournament_mc()` (line 537) and `simulate_tournament()` (line 714) already accept `progress: bool = False` for `tqdm`. The new `progress_callback` parameter coexists with this.

**Dashboard `st.spinner` locations to replace:**
- `dashboard/pages/4_Pool_Scorer.py:170` — `with st.spinner("Running Monte Carlo simulation...")`
- `dashboard/pages/2_Presentation.py:335` — `with st.spinner(spinner_msg)` (only when `method == "monte_carlo"`)

### Key Constraints

- **Do NOT remove `@st.cache_data` from `run_bracket_simulation()`** — the cached function must remain for subsequent renders that don't need re-simulation.
- **Do NOT break the existing `progress: bool` / `tqdm` path** — CLI users and tests rely on it.
- **Dashboard files are NOT under `mypy --strict`** — but `src/ncaa_eval/` files ARE. The `progress_callback` type must be fully typed in `simulation.py`.
- **`from __future__ import annotations`** required in all Python files.
- **Ruff complexity limits:** McCabe ≤10, returns ≤6, branches ≤12, args ≤5. The `simulate_tournament_mc` function already has a `noqa: PLR0913` — adding one more parameter is fine.

### Project Structure Notes

- Core library change: `src/ncaa_eval/evaluation/simulation.py` — add `progress_callback` parameter
- Dashboard lib change: `dashboard/lib/simulation_helpers.py` — add uncached wrapper
- Dashboard page changes: `dashboard/pages/4_Pool_Scorer.py`, `dashboard/pages/2_Presentation.py` — replace spinners
- Test changes: `tests/unit/test_evaluation_simulation.py`, `tests/unit/test_pool_scorer_page.py`
- No new files needed — modifications only

### Previous Story Intelligence (Story 9.11)

- **TypedDict safe access pattern:** `.get()` for `TypedDict` with `total=False` is the correct pattern
- **Mock updates:** When changing from one Streamlit pattern to another, update test mocks to match (Story 9.11 changed MagicMock attribute mocks to dict mocks)
- **`use_container_width=True` is deprecated** in Streamlit 1.54.0+ → use `width="stretch"` instead
- **All 1114 tests pass** as of Story 9.11 completion — maintain this baseline

### References

- [Source: `src/ncaa_eval/evaluation/simulation.py:530-681`] — `simulate_tournament_mc()` with round loop
- [Source: `src/ncaa_eval/evaluation/simulation.py:706-779`] — `simulate_tournament()` orchestrator
- [Source: `dashboard/lib/simulation_helpers.py:121-256`] — `run_bracket_simulation()` cached wrapper
- [Source: `dashboard/pages/4_Pool_Scorer.py:162-193`] — `_run_simulation()` with `st.spinner`
- [Source: `dashboard/pages/2_Presentation.py:334-345`] — Presentation page spinner
- [Source: `_bmad-output/planning-artifacts/epics.md`] — Story 9.12 AC (Audit item 2.16)
- [Source: `_bmad-output/planning-artifacts/architecture.md`] — §7.1 Frontend Architecture, §11.1 Performance

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

No debugging issues encountered.

### Completion Notes List

- **Task 1**: Added `progress_callback: Callable[[int, int], None] | None = None` parameter to both `simulate_tournament_mc()` and `simulate_tournament()`. Callback is invoked after each MC round with `(round_completed, total_rounds)`. Existing `tqdm` progress path is preserved.
- **Task 2**: Created `run_bracket_simulation_with_progress()` in `simulation_helpers.py` — uncached wrapper that delegates bracket setup to the cached `run_bracket_simulation(method="analytical")`, then runs MC with progress callback. Supports game-theory slider values (`upset_aggression`, `seed_weight_pct`).
- **Task 3**: Pool Scorer page `_run_simulation()` now uses `st.progress(0, text="Running Monte Carlo simulation...")` instead of `st.spinner`. Progress bar is cleared via `.empty()` after simulation.
- **Task 4**: Presentation page now branches: MC path uses `st.progress` + `run_bracket_simulation_with_progress()`, analytical path keeps `st.spinner("Computing bracket...")`.
- **Task 5**: Added 3 new tests for `progress_callback` (direct MC, via orchestrator, default None). Updated 3 existing pool scorer tests to mock `run_bracket_simulation_with_progress`. Updated 2 bracket page MC tests. Added new `TestAnalyticalUsesSpinner` test class. All 1122 tests pass (up from 1114 baseline — 8 new tests added).

### File List

- `src/ncaa_eval/evaluation/simulation.py` — added `progress_callback` param to `simulate_tournament_mc()` and `simulate_tournament()`
- `dashboard/lib/simulation_helpers.py` — added `run_bracket_simulation_with_progress()` uncached wrapper, imported `simulate_tournament_mc`
- `dashboard/pages/4_Pool_Scorer.py` — replaced `st.spinner` with `st.progress` in `_run_simulation()`
- `dashboard/pages/2_Presentation.py` — split MC/analytical paths: `st.progress` for MC, `st.spinner` for analytical
- `tests/unit/test_evaluation_simulation.py` — 3 new `progress_callback` tests in `TestMonteCarlo`
- `tests/unit/test_pool_scorer_page.py` — updated 3 tests to mock `run_bracket_simulation_with_progress`, added `test_progress_bar_used_instead_of_spinner`
- `tests/unit/test_bracket_page.py` — updated 2 MC tests to mock `run_bracket_simulation_with_progress`, added `TestAnalyticalUsesSpinner`
- `_bmad-output/implementation-artifacts/sprint-status.yaml` — status: in-progress → review
- `_bmad-output/implementation-artifacts/9-12-implement-st-progress-simulation.md` — story file updates

## Senior Developer Review (AI) — Pass 1

**Reviewer:** Claude Sonnet 4.6 — 2026-03-11

**Verdict:** APPROVED with fixes applied

**Findings and fixes (4 Medium, 2 Low — all resolved):**

- 🟡 MEDIUM (FIXED): Progress bar leaked on exception — `progress_bar.empty()` was never called if `run_bracket_simulation_with_progress` raised. Fixed with `try/finally` in both `4_Pool_Scorer.py:_run_simulation()` and `2_Presentation.py` MC path.
- 🟡 MEDIUM (FIXED): `run_bracket_simulation_with_progress` had no exception handler — unlike the cached function's `except (OSError, ValueError, KeyError, TypeError)`, exceptions propagated raw. Added matching `except` block returning `None` on failure.
- 🟡 MEDIUM (FIXED): `test_button_triggers_simulation_with_progress` in `test_pool_scorer_page.py` was missing `mock_progress_bar.empty.assert_called_once()` — the bar could be removed without test failure. Fixed by adding the assertion and `mock_progress_bar` setup.
- 🟡 MEDIUM (FIXED): `run_bracket_simulation_with_progress` docstring did not explain why provider re-loading is required despite having a cached result (unperturbed vs perturbed matrix). Fixed with expanded docstring.
- 🟢 LOW (ACCEPTED): Pool Scorer intentionally passes `upset_aggression=0, seed_weight_pct=0` — no sliders on that page. Already documented via comment in page code.
- 🟢 LOW (ACCEPTED): Orchestrator `test_progress_callback_via_orchestrator` checks only `calls[-1]` — the direct `test_progress_callback_called_each_round` test already covers all intermediate calls. Not a meaningful gap.

**AC verification:** All 5 ACs confirmed implemented. All 1122 tests pass post-fix.

## Senior Developer Review (AI) — Pass 2

**Reviewer:** Claude Sonnet 4.6 — 2026-03-11

**Verdict:** APPROVED with fixes applied

**Findings and fixes (1 High fixed, 1 Medium fixed, 2 Medium → action items, 2 Low accepted):**

- 🔴 HIGH (FIXED): `try/finally` for `progress_bar.empty()` was untested on the exception path — all existing tests mocked the helper to return a result, never to raise. Added `test_progress_bar_cleared_on_exception` that patches `run_bracket_simulation_with_progress` to `side_effect=RuntimeError` and asserts `progress_bar.empty.assert_called_once()`. [test_pool_scorer_page.py]
- 🟡 MEDIUM (FIXED): `test_progress_callback_via_orchestrator` only asserted `calls[-1] == (2, 2)`, leaving intermediate values unchecked. Tightened to `assert calls == [(1, 2), (2, 2)]` to match the direct test's strictness. [test_evaluation_simulation.py:1230]
- 🟡 MEDIUM (ACTION ITEM): `run_bracket_simulation_with_progress` exception clause misses `RuntimeError`/`AttributeError` — numpy/bracket errors propagate as unhandled crashes. Added to Review Follow-ups.
- 🟡 MEDIUM (ACTION ITEM): Analytical cached call receives `n_simulations` (ignored but pollutes cache key) — different n_sim values create duplicate analytical cache entries. Added to Review Follow-ups.
- 🟢 LOW (ACCEPTED): `_update_progress` closure has no guard for `total_rounds == 0` — impossible in production (64-team bracket → 6 rounds) but defensively weak.
- 🟢 LOW (FIXED): Stale test count in Pass 1 review block said "1075 tests" — corrected to 1122.

**AC verification:** All 5 ACs confirmed implemented. All 1123 tests pass post-fix (1 new test added).

## Change Log

- **2026-03-11**: Implemented st.progress for Monte Carlo simulation (Story 9.12). Added `progress_callback` parameter to simulation engine, created uncached dashboard wrapper, replaced spinners with progress bars on Pool Scorer and Presentation pages, added/updated 8 tests. All 1122 tests pass.
- **2026-03-11**: Code review fixes — added `try/finally` around progress bar usage in both pages, added exception handler to `run_bracket_simulation_with_progress`, strengthened pool scorer test assertion, improved docstring. 4 Medium issues resolved.
- **2026-03-11**: Code review pass 2 — added `test_progress_bar_cleared_on_exception` test (H1 fix), tightened orchestrator callback assertion to full list equality (M3 fix), corrected stale test count, added 2 Medium action items. 1123 tests pass.
