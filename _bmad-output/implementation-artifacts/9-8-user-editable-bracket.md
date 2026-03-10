# Story 9.8: User-Editable Bracket

Status: ready-for-dev

## Story

As a **data scientist**,
I want to **click matchups in the bracket view to override the model's predicted winner**,
So that **I can score my own picks against historical results and evaluate the model's guidance relative to my own judgment**.

## Acceptance Criteria

1. **Given** the Bracket Visualizer dashboard page
   **When** the user clicks a team in a matchup to override the predicted winner
   **Then** the bracket downstream of that matchup updates to reflect the user's pick
   **And** overridden matchups are visually distinct from model predictions (e.g., highlighted border or icon)
   **And** complementarity is preserved — downstream matchups re-resolve using the overridden winner

2. **Given** a user-edited bracket
   **When** the user navigates to the Pool Scorer page
   **Then** the Pool Scorer scores the **user-edited** bracket (not the model's most-likely bracket)
   **And** the score distribution reflects the user's picks against MC simulations

3. **Given** user overrides in the bracket
   **When** the user clicks a "Reset to Model Predictions" button
   **Then** all overrides are cleared and the bracket reverts to the model's most-likely picks

4. **Given** user overrides in session state
   **When** the user changes model run, year, scoring format, or game-theory slider values
   **Then** overrides are cleared (since the bracket structure/probabilities changed)
   **And** the user sees an info message that overrides were reset

5. **Given** user overrides persist in `st.session_state`
   **When** the user navigates between dashboard pages within the same session
   **Then** overrides are preserved (standard Streamlit session state behavior)

## Tasks / Subtasks

- [ ] Task 1: Create bracket override state management module (AC: #1, #3, #4, #5)
  - [ ] 1.1: Create `dashboard/lib/bracket_overrides.py` with:
    - `BracketOverrides` — a dict-like container mapping `game_index → winner_team_index` stored in `st.session_state["bracket_overrides"]`
    - `get_overrides() -> dict[int, int]` — return current overrides from session state (empty dict if none)
    - `set_override(game_index: int, winner_index: int) -> None` — add/update a single override
    - `clear_overrides() -> None` — remove all overrides from session state
    - `apply_overrides(most_likely: MostLikelyBracket, overrides: dict[int, int], bracket: BracketStructure, prob_matrix: npt.NDArray[np.float64]) -> MostLikelyBracket` — produce a new `MostLikelyBracket` that incorporates user overrides and cascades downstream effects
  - [ ] 1.2: In `apply_overrides()`, implement downstream cascade logic:
    - Start from the model's `most_likely.winners` tuple
    - For each override, replace the winner at that game_index
    - Cascade: if an override changes game G's winner, recompute all downstream games that depended on G's winner (the feeder game for a later round)
    - Recompute `champion_team_id` and `log_likelihood` from the final bracket
  - [ ] 1.3: Add override invalidation key: store `(run_id, year, scoring, upset_aggression, seed_weight_pct)` alongside overrides in session state. When any of these change, auto-clear overrides.

- [ ] Task 2: Make bracket tree interactive with clickable matchups (AC: #1)
  - [ ] 2.1: Modify `dashboard/lib/bracket_renderer.py` — `render_bracket_html()` to support an interactive mode:
    - Add optional parameter `interactive: bool = False` and `overrides: dict[int, int] | None = None`
    - When `interactive=True`, each team cell in the bracket tree becomes a clickable element
    - Use Streamlit's `components.html()` with bidirectional communication via `streamlit-component-lib` JavaScript, OR use a simpler approach: render the bracket with `st.form` / `st.button` grid, OR use `st.html` + JavaScript `postMessage` to communicate clicks back to Streamlit
    - **IMPORTANT DESIGN DECISION**: The current bracket renderer produces a self-contained HTML string rendered via `components.html()`. Making this interactive requires either:
      - **Option A (Recommended)**: Replace the HTML bracket with a Streamlit-native widget layout using `st.columns` and `st.button` for each matchup. This is simpler but may look different.
      - **Option B**: Use `streamlit-component-lib` JavaScript bidirectional messaging within the `components.html()` iframe to send click events back to Streamlit via `Streamlit.setComponentValue()`. The page would re-render with the new override.
      - **Option C**: Use `st.selectbox` per matchup inside expanders/containers. Less visual but fully native.
    - See Dev Notes for recommendation.
  - [ ] 2.2: Visually distinguish overridden matchups (e.g., golden/yellow border, "USER" badge, or different background color) so the user can see which picks are their own vs. the model's.
  - [ ] 2.3: Show the model's original pick alongside the user's override (e.g., strikethrough on the model's pick or a small "Model: [team]" label).

- [ ] Task 3: Integrate overrides into Bracket Visualizer page (AC: #1, #3, #4)
  - [ ] 3.1: Modify `dashboard/pages/2_Presentation.py` — `_render_bracket_page()`:
    - After `run_bracket_simulation()` returns, check for override invalidation (compare current params against stored invalidation key)
    - If invalidated, clear overrides and show `st.info("Bracket parameters changed — user overrides have been reset.")`
    - Get current overrides via `get_overrides()`
    - Apply overrides to produce a user-edited `MostLikelyBracket` via `apply_overrides()`
    - Pass the user-edited bracket to `_render_results()` instead of the model's `most_likely`
  - [ ] 3.2: Add "Reset to Model Predictions" button (AC: #3):
    - Only show when overrides exist (non-empty dict)
    - On click: `clear_overrides()` + `st.rerun()`
  - [ ] 3.3: Update `_render_results()` to accept and display override status:
    - Show count of user overrides (e.g., "3 of 63 picks overridden")
    - Pass overrides to bracket renderer for visual distinction

- [ ] Task 4: Integrate user-edited bracket into Pool Scorer (AC: #2)
  - [ ] 4.1: Modify `dashboard/pages/4_Pool_Scorer.py` — `_run_simulation()`:
    - After simulation, apply user overrides from session state to produce an edited bracket
    - Store the edited `MostLikelyBracket` in session state alongside `pool_sim_data`
  - [ ] 4.2: Modify `dashboard/lib/filters.py` — `score_chosen_bracket()`:
    - Currently uses `sim_data.most_likely.winners` as the chosen bracket
    - Change to accept an optional `chosen_bracket` parameter (or pass the user-edited `MostLikelyBracket` directly)
    - When user overrides exist, score the **user's** bracket against MC sims, not the model's most-likely bracket
  - [ ] 4.3: Modify `_render_results()` in `4_Pool_Scorer.py`:
    - Pass the user-edited bracket to `export_bracket_csv()` so the CSV export reflects user picks

- [ ] Task 5: Add comprehensive tests (AC: all)
  - [ ] 5.1: Create `tests/unit/test_bracket_overrides.py`:
    - `get_overrides()` returns empty dict when no overrides set
    - `set_override()` stores override correctly
    - `clear_overrides()` removes all overrides
    - `apply_overrides()` with no overrides returns original bracket unchanged
    - `apply_overrides()` with single override replaces correct game winner
    - `apply_overrides()` cascades downstream — override in R64 forces re-evaluation through R32, S16, etc.
    - `apply_overrides()` recomputes champion_team_id when championship game overridden
    - `apply_overrides()` recomputes log_likelihood correctly
    - Override invalidation: changing run_id/year/sliders clears overrides
  - [ ] 5.2: Update `tests/unit/test_bracket_page.py`:
    - Test that overrides appear in rendered output (override count display)
    - Test "Reset to Model Predictions" button behavior
    - Test override invalidation on parameter change
    - Test override flow: click → bracket updates → re-render
  - [ ] 5.3: Update `tests/unit/test_pool_scorer_page.py`:
    - Test that user-edited bracket is scored (not model's most-likely)
    - Test CSV export uses user-edited bracket
  - [ ] 5.4: Run full quality gates: `pytest`, `ruff check .`, `mypy --strict src/ncaa_eval tests`

- [ ] Task 6: Update user guide documentation (AC: #1)
  - [ ] 6.1: Add a "User-Editable Bracket" section to `docs/user-guide.md` under the Bracket Visualizer section
  - [ ] 6.2: Document: how to override picks, how downstream cascades work, how overrides integrate with Pool Scorer, when overrides are reset

## Dev Notes

### Recommended Interactive Approach (Option A — Streamlit-Native)

**The current HTML bracket renderer (`bracket_renderer.py`) is a pure function that outputs a static HTML string.** Making it interactive with JavaScript bidirectional messaging (Option B) requires:
1. `streamlit-component-lib` npm package
2. Custom JavaScript event handling
3. `postMessage` / `Streamlit.setComponentValue()` integration
4. Significant complexity for iframe↔Streamlit communication

**Recommendation: Option A — Replace the bracket display with Streamlit-native widgets for the interactive portion**, while keeping the static HTML renderer as a "read-only" view option.

**Implementation approach:**
- Below the static HTML bracket (keep it for the visual overview), add an **interactive "Edit Picks" section** using `st.expander` or `st.tabs`
- In the interactive section, render matchups round-by-round using `st.columns` and `st.selectbox` (or `st.radio`) per game
- Each matchup shows: "Game N (Round): Team A vs Team B" with a selectbox to pick the winner
- When the user changes a pick, the page re-renders with the cascade applied
- The static HTML bracket at the top updates to reflect the user's picks

**Why this is better:**
- No JavaScript / npm dependencies
- No iframe bidirectional messaging complexity
- Full Streamlit state management (session state "just works")
- Testable with standard `mock_st` patterns
- Consistent with the project's Streamlit-native approach (no custom components exist in the codebase)

**Alternative (if PO wants direct bracket clicking):** Use `streamlit-elements` or write a minimal `st.components.declare_component()` with JS `postMessage`. This is significantly more complex and the project has no precedent for custom Streamlit components.

### Cascade Logic in `apply_overrides()`

The `most_likely.winners` tuple is in **round-major order**: indices 0–31 = R64, 32–47 = R32, 48–55 = S16, 56–59 = E8, 60–61 = F4, 62 = Championship.

**Cascade algorithm:**
```python
def apply_overrides(
    most_likely: MostLikelyBracket,
    overrides: dict[int, int],   # game_index → winner_team_index
    bracket: BracketStructure,
    prob_matrix: npt.NDArray[np.float64],
) -> MostLikelyBracket:
    winners = list(most_likely.winners)  # mutable copy

    # Apply overrides in round order (earlier rounds first)
    for game_idx in sorted(overrides):
        winners[game_idx] = overrides[game_idx]

    # Cascade: for each round after R64, the participants in game G
    # are the winners of feeder games (2G and 2G+1 from previous round).
    # If a feeder game's winner changed, re-pick using either the
    # existing override (if one exists for this downstream game) or
    # the model's argmax from prob_matrix.
    n_games = len(winners)  # 63
    n_teams = n_games + 1   # 64
    offset = 0
    games_in_round = n_teams // 2  # 32

    for round_idx in range(6):
        next_offset = offset + games_in_round
        next_games = games_in_round // 2

        for g in range(next_games):
            downstream_idx = next_offset + g
            feeder_a = offset + g * 2
            feeder_b = offset + g * 2 + 1

            participant_a = winners[feeder_a] if round_idx > 0 else ...
            participant_b = winners[feeder_b] if round_idx > 0 else ...

            # If downstream game has explicit override, keep it (if valid)
            if downstream_idx in overrides:
                # Validate: the overridden winner must be one of the two participants
                if overrides[downstream_idx] in (participant_a, participant_b):
                    winners[downstream_idx] = overrides[downstream_idx]
                else:
                    # Override is stale (participant changed upstream), use model
                    winners[downstream_idx] = _pick_model_winner(
                        participant_a, participant_b, prob_matrix
                    )
            else:
                # No override: use model prediction (argmax)
                winners[downstream_idx] = _pick_model_winner(
                    participant_a, participant_b, prob_matrix
                )

        offset = next_offset
        games_in_round = next_games
```

**Key insight**: When an upstream override changes a participant in a downstream game, the downstream game must be re-resolved. If the downstream game ALSO has an override, validate that the override's winner is still a valid participant. If not, fall back to model prediction.

### R64 Game Participants

For Round of 64 (game indices 0–31), participants are determined by bracket position, NOT by previous game winners:
- Game 0: team_index 0 vs team_index 1
- Game 1: team_index 2 vs team_index 3
- Game G: team_index 2G vs team_index 2G+1

For Round of 32+ (game indices 32+), participants are winners of the two feeder games from the previous round.

### Session State Design

```python
# Override storage in session state
st.session_state["bracket_overrides"] = {
    5: 11,     # Game 5 (R64): user picked team_index 11 instead of model's pick
    35: 3,     # Game 35 (R32): user picked team_index 3
}

# Invalidation key — clear overrides when any of these change
st.session_state["bracket_override_key"] = (run_id, year, scoring, upset_aggression, seed_weight_pct)
```

### Pool Scorer Integration

Currently `score_chosen_bracket()` in `dashboard/lib/filters.py` uses `sim_data.most_likely.winners`:
```python
chosen = np.array(sim_data.most_likely.winners, dtype=np.int32)
raw_scores = score_bracket_against_sims(chosen, sim_winners, _scoring_rules)
```

**Change required**: When user overrides exist, replace `sim_data.most_likely` with the user-edited bracket:
```python
# In 4_Pool_Scorer.py or filters.py
overrides = get_overrides()
if overrides:
    edited = apply_overrides(sim_data.most_likely, overrides, sim_data.bracket, sim_data.prob_matrix)
    chosen = np.array(edited.winners, dtype=np.int32)
else:
    chosen = np.array(sim_data.most_likely.winners, dtype=np.int32)
```

**CRITICAL**: The `score_chosen_bracket` function is `@st.cache_data`. If the chosen bracket changes due to overrides, the cache key must reflect this. Options:
1. Add an `override_key: str` parameter (hash of overrides dict) to the cache function
2. Move override application outside the cached function and pass `chosen` as parameter
3. Make `score_chosen_bracket` accept an explicit `chosen_winners` array instead of extracting from `sim_data`

Option 3 is cleanest — it separates "which bracket to score" from "what simulations to score against."

### File Locations

| File | Action |
|---|---|
| `dashboard/lib/bracket_overrides.py` | **NEW** — override state management and cascade logic |
| `dashboard/lib/bracket_renderer.py` | **MODIFY** — add override visual distinction (highlighted cells) |
| `dashboard/pages/2_Presentation.py` | **MODIFY** — add interactive pick editing UI, reset button, override display |
| `dashboard/pages/4_Pool_Scorer.py` | **MODIFY** — score user-edited bracket instead of model's |
| `dashboard/lib/filters.py` | **MODIFY** — accept explicit chosen bracket for scoring |
| `dashboard/lib/export.py` | **MODIFY** — accept explicit bracket for CSV export |
| `docs/user-guide.md` | **MODIFY** — add user-editable bracket documentation |
| `tests/unit/test_bracket_overrides.py` | **NEW** — unit tests for override logic |
| `tests/unit/test_bracket_page.py` | **MODIFY** — add override interaction tests |
| `tests/unit/test_pool_scorer_page.py` | **MODIFY** — test user-edited bracket scoring |

### Testing Strategy

1. **Unit tests** (`test_bracket_overrides.py`): Pure logic tests for `apply_overrides()` cascade — no Streamlit mocking needed. Test: identity (no overrides), single override, multi-round cascade, stale override invalidation, champion change.

2. **Page tests** (`test_bracket_page.py`): Mock `st.session_state` with overrides, verify `_render_bracket_page()` applies them. Use existing `patch.object` pattern. Test: override count display, reset button visibility, invalidation message.

3. **Integration test** (`test_pool_scorer_page.py`): Verify that when `bracket_overrides` exists in session state, the Pool Scorer scores the user's bracket. Mock `score_bracket_against_sims` and check `chosen_bracket` argument.

4. **Quality gates**: `mypy --strict`, `ruff check`, full `pytest` suite.

### Existing Patterns to Follow

- **`_render_xxx()` wrapper pattern**: All page logic inside a function, single call at module bottom (mandatory per `template-requirements.md`)
- **`patch.object(module, ...)` for tests**: Numeric-prefixed page modules require `importlib.import_module()` + `patch.object()`
- **Multi-call selectbox mocking**: Use `side_effect` lists for multiple `st.selectbox` calls per page
- **`from __future__ import annotations`** required in all Python files (enforced by Ruff)
- **`@st.cache_data` unhashable Protocol workaround**: Prefix Protocol params with `_`, use string key for cache discrimination
- **Session state `setdefault()` pattern**: Never overwrite existing session state values

### Previous Story Intelligence

**From Story 9.7 (Game Theory Sliders):**
- `run_bracket_simulation()` already accepts `upset_aggression` and `seed_weight_pct` as cache key parameters
- `BracketSimulationResult` contains `most_likely: MostLikelyBracket` and `prob_matrix`
- Slider changes trigger cache miss and re-computation — override invalidation should follow the same pattern
- `_render_results()` receives `sim_data` and `scoring` — needs to also receive override info
- Test pattern: `mock_st.columns.side_effect = lambda *a, **kw: [MagicMock() for _ in range(a[0] if isinstance(a[0], int) else len(a[0]))]`
- 3-column layout for Game Theory Sliders section

**From Story 7.5 (Bracket Visualizer):**
- `render_bracket_html()` is a pure function (no Streamlit imports) — keep it pure
- `components.html(bracket_html, height=750, scrolling=True)` renders the bracket
- `_team_cell()` renders individual team cells with seed, name, and probability

**From Story 7.6 (Pool Scorer):**
- `score_chosen_bracket()` in `filters.py` uses `sim_data.most_likely.winners` — this is the key integration point
- `export_bracket_csv()` in `export.py` takes `most_likely` parameter — needs to accept user-edited bracket

### Architecture Compliance

- **Dashboard files** (`dashboard/`) are NOT subject to `mypy --strict` — but follow type hints for readability
- **Tests** (`tests/unit/`) ARE subject to `mypy --strict`
- **No custom Streamlit components** exist in the project — prefer native Streamlit widgets
- **`@st.cache_data`** for all expensive computations; override logic is cheap (O(63) games)
- **Session state** is the canonical state management — no external stores

### Project Structure Notes

- New module `bracket_overrides.py` lives in `dashboard/lib/` alongside `simulation_helpers.py`, `bracket_renderer.py`, `export.py`
- No changes to `src/ncaa_eval/` — overrides are a dashboard-only concern
- Follows the decomposition pattern: pure logic in `dashboard/lib/`, UI in `dashboard/pages/`

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 9.8] — Acceptance criteria
- [Source: _bmad-output/planning-artifacts/po-decision-log-epic8.md#1.2] — PO decision: implement user-editable bracket
- [Source: dashboard/lib/simulation_helpers.py] — `BracketSimulationResult`, `run_bracket_simulation()`
- [Source: dashboard/pages/2_Presentation.py] — Bracket Visualizer page (integration point)
- [Source: dashboard/pages/4_Pool_Scorer.py] — Pool Scorer page (scoring integration)
- [Source: dashboard/lib/bracket_renderer.py] — `render_bracket_html()` (visual update)
- [Source: dashboard/lib/filters.py] — `score_chosen_bracket()` (scoring integration)
- [Source: dashboard/lib/export.py] — `export_bracket_csv()` (CSV export integration)
- [Source: src/ncaa_eval/evaluation/simulation.py] — `MostLikelyBracket`, `score_bracket_against_sims()`
- [Source: _bmad-output/implementation-artifacts/9-7-game-theory-slider-implementation.md] — Previous story patterns

## Dev Agent Record

### Agent Model Used

{{agent_model_name_version}}

### Debug Log References

### Completion Notes List

### Change Log

### File List
