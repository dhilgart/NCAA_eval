# Story 9.12: Implement st.progress for Simulation

Status: ready-for-dev

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

- [ ] Task 1: Add `progress_callback` parameter to simulation engine (AC: #3)
  - [ ] 1.1: Add `progress_callback: Callable[[int, int], None] | None = None` parameter to `simulate_tournament_mc()` in `src/ncaa_eval/evaluation/simulation.py`
  - [ ] 1.2: Call `progress_callback(round_completed, total_rounds)` after each round in the MC loop (alongside existing `tqdm` support)
  - [ ] 1.3: Add `progress_callback` parameter to `simulate_tournament()` orchestrator and pass through to MC path
  - [ ] 1.4: Update docstrings for both functions

- [ ] Task 2: Create uncached simulation wrapper with progress support (AC: #1, #2)
  - [ ] 2.1: In `dashboard/lib/simulation_helpers.py`, create a new function `run_bracket_simulation_with_progress()` that duplicates the MC path of `run_bracket_simulation()` but is NOT decorated with `@st.cache_data`
  - [ ] 2.2: The new function accepts an `st.progress` bar object and passes a lambda callback to `simulate_tournament()` that updates the progress bar
  - [ ] 2.3: Reuse all existing bracket-building, provider-creation, and scoring logic from the cached function (call shared helpers, do NOT duplicate)

- [ ] Task 3: Replace `st.spinner` with `st.progress` on Pool Scorer page (AC: #1, #4)
  - [ ] 3.1: In `dashboard/pages/4_Pool_Scorer.py` `_run_simulation()`, replace `st.spinner("Running Monte Carlo simulation...")` with an `st.progress(0, text="Running Monte Carlo simulation...")` bar
  - [ ] 3.2: Call the new uncached simulation function, passing the progress bar
  - [ ] 3.3: Set progress to 1.0 and clear after simulation completes

- [ ] Task 4: Replace `st.spinner` with `st.progress` on Presentation page (AC: #4, #5)
  - [ ] 4.1: In `dashboard/pages/2_Presentation.py`, replace the MC spinner with `st.progress` only when `method == "monte_carlo"`
  - [ ] 4.2: Keep `st.spinner` for analytical method (near-instant, no progress needed)

- [ ] Task 5: Update tests (AC: all)
  - [ ] 5.1: Add unit tests for `progress_callback` in `test_evaluation_simulation.py` — verify callback is called once per round with correct `(round, total)` args
  - [ ] 5.2: Update `test_pool_scorer_page.py` — verify `st.progress` is called instead of `st.spinner` for MC simulation path
  - [ ] 5.3: Verify `st.spinner` is still used for analytical path on Presentation page

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

### Debug Log References

### Completion Notes List

### File List
