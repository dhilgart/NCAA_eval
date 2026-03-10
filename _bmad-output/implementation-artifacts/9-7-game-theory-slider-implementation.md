# Story 9.7: Game Theory Slider Implementation

Status: ready-for-dev

## Story

As a **data scientist**,
I want to **adjust Upset Aggression and Seed-Weight sliders in the dashboard sidebar to perturb the model's base probabilities in real time**,
so that **I can explore bracket outcomes under different risk strategies without retraining the model**.

## Acceptance Criteria

1. **Given** the dashboard Presentation page
   **When** the user moves the Upset Aggression slider (range: −5 to +5, default 0)
   **Then** the model's base win probabilities are perturbed using the power/temperature transform: `p' = p^(1/T) / (p^(1/T) + (1-p)^(1/T))` where `T = 2^(slider_value / 3)`
   **And** the bracket visualization, advancement heatmap, expected points table, and pairwise probability selector all update to reflect the perturbed probabilities
   **And** the Monte Carlo score distribution does NOT re-run (it uses the original unperturbed P)

2. **Given** the dashboard Presentation page
   **When** the user moves the Seed-Weight slider (range: 0% to 100%, default 0%)
   **Then** the perturbed probabilities are further blended with historical seed priors: `p'' = (1-w)*p' + w*p_seed_prior`
   **And** the bracket visualization updates to reflect the blended probabilities
   **And** complementarity is preserved: `P'[i,j] + P'[j,i] = 1` for all team pairs

3. **Given** both sliders at neutral positions (Upset Aggression = 0, Seed-Weight = 0%)
   **When** the page loads
   **Then** the bracket visualization is identical to the unperturbed model output (identity property)

4. **Given** the user guide (`docs/user-guide.md`)
   **When** the implementation is complete
   **Then** the "NOT YET IMPLEMENTED" banner (lines 529–533) is removed
   **And** the slider descriptions in the user guide are updated to reflect the actual 2-slider configuration (Upset Aggression + Seed-Weight, not the original 3-slider spec)

## Tasks / Subtasks

- [ ] Task 1: Create `src/ncaa_eval/evaluation/perturbation.py` module (AC: #1, #2, #3)
  - [ ] 1.1: Define `FIRST_ROUND_SEED_PRIORS: dict[int, float]` mapping `seed_diff → P(higher seed wins)` using historical data: `{15: 0.993, 13: 0.938, 11: 0.854, 9: 0.792, 7: 0.646, 5: 0.625, 3: 0.604, 1: 0.521}`
  - [ ] 1.2: Implement `slider_to_temperature(slider_value: int) -> float` — maps `[-5, +5]` to `T` via `T = 2^(slider_value / 3)`. Validates input range.
  - [ ] 1.3: Implement `power_transform(P: npt.NDArray[np.float64], temperature: float) -> npt.NDArray[np.float64]` — applies `p' = p^(1/T) / (p^(1/T) + (1-p)^(1/T))` element-wise. Must preserve diagonal (zeros) and handle `p=0` / `p=1` correctly.
  - [ ] 1.4: Implement `build_seed_prior_matrix(seed_map: dict[int, int], team_ids: Sequence[int]) -> npt.NDArray[np.float64]` — constructs `(n, n)` seed prior matrix. Uses `FIRST_ROUND_SEED_PRIORS` with linear interpolation for even seed differences. Same-seed matchups get 0.5.
  - [ ] 1.5: Implement `perturb_probability_matrix(P, seed_map, team_ids, temperature=1.0, seed_weight=0.0) -> npt.NDArray[np.float64]` — applies temperature transform first, then seed blend. Must preserve complementarity, range [0,1], and diagonal zeros. Returns P unchanged when both params are neutral (T=1.0, w=0.0).
  - [ ] 1.6: Add `from __future__ import annotations` and full type annotations for `mypy --strict` compliance.

- [ ] Task 2: Export perturbation API from `evaluation` package (AC: #1, #2)
  - [ ] 2.1: Add imports of `perturb_probability_matrix`, `slider_to_temperature`, `build_seed_prior_matrix`, `FIRST_ROUND_SEED_PRIORS`, and `power_transform` to `src/ncaa_eval/evaluation/__init__.py`
  - [ ] 2.2: Add all five names to `__all__`

- [ ] Task 3: Integrate sliders into dashboard Presentation page (AC: #1, #2, #3)
  - [ ] 3.1: Add two slider controls to `dashboard/pages/2_Presentation.py` in a new "Game Theory Sliders" subsection below the existing "Simulation Settings":
    - Upset Aggression: `st.slider("Upset Aggression", min_value=-5, max_value=5, value=0, step=1, help="Chalk ← → Chaos")` with session key `"bracket_upset_aggression"`
    - Seed-Weight: `st.slider("Seed-Weight", min_value=0, max_value=100, value=0, step=5, format="%d%%", help="Model ← → Seeds")` with session key `"bracket_seed_weight"`
  - [ ] 3.2: Modify `dashboard/lib/simulation_helpers.py`:
    - Add `upset_aggression: int = 0` and `seed_weight_pct: int = 0` parameters to `run_bracket_simulation()` (these become part of the `@st.cache_data` cache key)
    - Build `prob_matrix` via `build_probability_matrix()`, then compute `perturbed_matrix` via `perturb_probability_matrix(prob_matrix, bracket.seed_map, bracket.team_ids, temperature=slider_to_temperature(upset_aggression), seed_weight=seed_weight_pct / 100.0)`
    - Run analytical `simulate_tournament()` with a `MatrixProvider(perturbed_matrix, ...)` to get perturbed `advancement_probs` and `expected_points`
    - If MC is selected, run a separate `simulate_tournament()` with the ORIGINAL provider (method="monte_carlo") for `sim_winners` and `bracket_distributions` — MC represents "true reality" unchanged by slider adjustments
    - Merge the MC fields (`sim_winners`, `bracket_distributions`) into the analytical result, or restructure `BracketSimulationResult` to hold both
    - Use `perturbed_matrix` for `compute_most_likely_bracket()` and as `prob_matrix` in the returned `BracketSimulationResult`
    - **CRITICAL**: The `prob_matrix` stored in `BracketSimulationResult` must be the PERTURBED matrix (P'), since all rendering (bracket tree, heatmap, pairwise selector) uses it
  - [ ] 3.3: Pass the new slider values from `2_Presentation.py` to `run_bracket_simulation()` call
  - [ ] 3.4: Add a "Reset Sliders" button that sets both sliders back to neutral (Upset Aggression=0, Seed-Weight=0%)

- [ ] Task 4: Update user guide documentation (AC: #4)
  - [ ] 4.1: Remove the "NOT YET IMPLEMENTED" warning banner from `docs/user-guide.md` (lines 529–533)
  - [ ] 4.2: Update the slider documentation section to describe the actual 2-slider configuration (Upset Aggression + Seed-Weight) instead of the original 3-slider spec. Reference the mathematical formulas.
  - [ ] 4.3: Add a brief explanation of what each slider does: "Upset Aggression: Negative = favorites reinforced; Positive = upsets more likely" and "Seed-Weight: 0% = pure model; 100% = pure historical seed rates"

- [ ] Task 5: Add comprehensive tests (AC: all)
  - [ ] 5.1: Create `tests/unit/test_perturbation.py` with tests for:
    - `slider_to_temperature()` — boundary values (-5, 0, +5), neutral returns 1.0
    - `power_transform()` — identity at T=1.0, T>1 compresses toward 0.5, T<1 sharpens, preserves complementarity, handles p=0/1/0.5 edge cases, preserves diagonal zeros
    - `build_seed_prior_matrix()` — correct lookups for standard matchups, same-seed returns 0.5, interpolates even seed_diff values
    - `perturb_probability_matrix()` — identity at neutral (T=1, w=0), combined transform produces correct output, complementarity preserved under all slider combinations, diagonal remains zero
  - [ ] 5.2: Add parametrized tests for the worked examples from the spike research (Section 4 of `specs/research/game-theory-slider-mechanism.md`): verify computed values match documented values for 1v16, 5v12, 8v9, 6v11 at T=0.5/1.0/1.5/2.0/3.0 and w=0.0/0.25/0.5/0.75/1.0
  - [ ] 5.3: Run full quality gates: `pytest`, `ruff check .`, `mypy --strict src/ncaa_eval tests`

## Dev Notes

### Core Mathematical Specification (from Story 7.7 Spike)

**Two independent transformations applied in sequence:**

1. **Temperature (Upset Aggression):**
   ```
   T = 2^(slider_value / 3)         # slider [-5, +5] → T [0.315, 3.17]
   p' = p^(1/T) / (p^(1/T) + (1-p)^(1/T))
   ```
   - T=1.0 (slider=0): identity
   - T>1 (slider>0): compresses probabilities toward 0.5 (more upsets)
   - T<1 (slider<0): sharpens probabilities away from 0.5 (more chalk)
   - Fixed point: p=0.5 is unchanged for all T

2. **Seed Blend (Seed-Weight):**
   ```
   p'' = (1-w) * p' + w * p_seed_prior
   ```
   - w=0 (slider=0%): identity
   - w=1 (slider=100%): pure seed prior (model ignored)
   - Seed prior from `FIRST_ROUND_SEED_PRIORS` keyed by `|seed_a - seed_b|`

**Order of operations:** Temperature FIRST, then seed blend. This is the correct order because temperature adjusts the model's confidence level (property of model output), while seed blend anchors toward historical priors (property of the matchup).

**Complementarity preserved at each step:**
- After temperature: `P'[i,j] + P'[j,i] = 1` (by construction of power normalization)
- After seed blend: `P''[i,j] + P''[j,i] = (1-w)(1) + w(1) = 1`

### Two-Slider vs Three-Slider Decision

The spike research (Story 7.7, Section 3.4) recommended collapsing the original 3-slider UX spec (Upset Aggression + Chalk Bias + Seed-Weight) into 2 sliders, because Upset Aggression and Chalk Bias control the SAME mathematical axis (temperature — one raises T, the other lowers T). PO approved this recommendation as part of item 1.1 in `po-decision-log-epic8.md`.

**The AC in `epics.md` says "Upset Aggression, Chalk Bias, or Seed-Weight" — but per the spike recommendation and PO approval, Chalk Bias is collapsed into Upset Aggression as a single bidirectional slider.** The story implements 2 sliders.

### Pipeline Integration Architecture

**Current pipeline** (in `dashboard/lib/simulation_helpers.py:run_bracket_simulation`, lines 188–198):
```python
sim_result = simulate_tournament(bracket, provider, context, ...)  # builds matrix internally
prob_matrix = build_probability_matrix(provider, bracket.team_ids, context)
most_likely = compute_most_likely_bracket(bracket, prob_matrix)
```

Note: `simulate_tournament()` internally builds its own probability matrix and computes `advancement_probs` and `expected_points`. When sliders are non-neutral, the returned `sim_result.advancement_probs` and `sim_result.expected_points` will reflect the UNPERTURBED model. This needs careful handling.

**Proposed pipeline:**
```python
# 1. Build original probability matrix
prob_matrix = build_probability_matrix(provider, bracket.team_ids, context)

# 2. Perturb the matrix (NEW)
perturbed_matrix = perturb_probability_matrix(
    prob_matrix, bracket.seed_map, bracket.team_ids,
    temperature=slider_to_temperature(upset_aggression),
    seed_weight=seed_weight_pct / 100.0,
)

# 3. MC simulation uses ORIGINAL provider (unperturbed "reality")
#    Only run MC if method == "monte_carlo"
sim_result = simulate_tournament(bracket, provider, context, ...)

# 4. For analytical path outputs (advancement, EP), re-compute from PERTURBED matrix
#    Use MatrixProvider(perturbed_matrix, bracket.team_ids) as the perturbed provider,
#    then call compute_advancement_probs() and compute_expected_points() separately.
#    OR: create a second simulate_tournament() call with a perturbed MatrixProvider
#    for method="analytical" to get correct advancement_probs/EP from perturbed probs.
perturbed_provider = MatrixProvider(perturbed_matrix, list(bracket.team_ids))
perturbed_sim = simulate_tournament(
    bracket, perturbed_provider, context, scoring_rules=..., method="analytical"
)

# 5. Most-likely bracket from perturbed matrix
most_likely = compute_most_likely_bracket(bracket, perturbed_matrix)
```

**CRITICAL architectural note:** The current `simulate_tournament()` computes `advancement_probs` and `expected_points` internally using whatever provider is passed. When sliders are non-neutral:
- The **MC simulation** (sim_winners, score distributions) should use the ORIGINAL provider (unperturbed "reality")
- The **advancement heatmap** and **EP table** should reflect PERTURBED probabilities (what your bracket strategy looks like)
- The simplest approach: always run an analytical simulation with the perturbed `MatrixProvider` for advancement/EP, and optionally run MC with the original provider for score distributions. Combine results.
- Alternative: run `simulate_tournament()` once with the perturbed provider for method="analytical", and separately with the original provider for method="monte_carlo" only when MC is selected. Merge `sim_winners`/`bracket_distributions` from the MC run into the analytical result.

**Re-render scope when sliders change:**

| Component | Re-renders? | Why |
|---|---|---|
| Bracket tree (HTML) | Yes | Uses `most_likely` picks which depend on P' |
| Advancement heatmap | Yes | Uses `advancement_probs` computed from P' |
| EP table | Yes | Uses `expected_points` computed from P' advancement probs |
| Pairwise probability selector | Yes | Displays P'[i,j] values directly |
| MC score distribution | **No** | Represents "true" simulation outcomes under original model |

**Rationale for NOT re-running MC:** The sliders adjust the user's BRACKET STRATEGY (which teams to pick). The MC simulation represents possible tournament outcomes under the model's true beliefs. Scoring the perturbed bracket against original simulations answers: "If I pick upsets, how would my bracket score against reality?"

### Caching Design

The `run_bracket_simulation()` function uses `@st.cache_data(ttl=None)`. Adding `upset_aggression` and `seed_weight_pct` as parameters automatically makes them part of the cache key. Different slider values will produce different cache entries.

**Important:** The perturbation must happen INSIDE the cached function so that the perturbed `prob_matrix` is stored in the `BracketSimulationResult`. This means slider changes trigger a cache miss and re-computation of the perturbation + analytical path, which is cheap (< 10ms for a 64×64 matrix).

### Seed Prior Matrix Construction

The `build_seed_prior_matrix()` function builds an (n×n) seed prior matrix:

```python
FIRST_ROUND_SEED_PRIORS: dict[int, float] = {
    15: 0.993,  # 1 vs 16
    13: 0.938,  # 2 vs 15
    11: 0.854,  # 3 vs 14
    9:  0.792,  # 4 vs 13
    7:  0.646,  # 5 vs 12
    5:  0.625,  # 6 vs 11
    3:  0.604,  # 7 vs 10
    1:  0.521,  # 8 vs 9
}
```

For seed differences NOT in the table (even values), use linear interpolation between adjacent tabulated values. For `seed_diff = 0` (same seed), use `0.5`. For `seed_diff > 15`, clamp to 0.993.

**Limitation note:** The seed_diff-based lookup uses 8v9 first-round rates for `seed_diff=1`, which understates how similar top-seed pairs actually are in later rounds. Treat the seed prior as a coarse anchor for non-first-round matchups.

### File Locations

| File | Action |
|---|---|
| `src/ncaa_eval/evaluation/perturbation.py` | **NEW** — core perturbation functions |
| `src/ncaa_eval/evaluation/__init__.py` | **MODIFY** — add perturbation exports |
| `dashboard/lib/simulation_helpers.py` | **MODIFY** — add slider params, call perturbation |
| `dashboard/pages/2_Presentation.py` | **MODIFY** — add slider UI controls |
| `docs/user-guide.md` | **MODIFY** — remove banner, update slider docs |
| `tests/unit/test_perturbation.py` | **NEW** — unit tests for perturbation module |

### Testing Strategy

1. **Unit tests** cover mathematical correctness of each function:
   - Identity property (neutral sliders = unchanged output)
   - Complementarity preservation (`P[i,j] + P[j,i] = 1`)
   - Edge cases (`p=0`, `p=1`, `p=0.5`, diagonal entries)
   - Numerical agreement with worked examples from the spike research

2. **Integration**: The Streamlit caching + slider parameter flow is verified via the existing dashboard test infrastructure. No separate E2E test needed — the critical path is the mathematical module, not the UI wiring.

3. **Quality gates**: `mypy --strict`, `ruff check`, full `pytest` suite.

### Previous Story Intelligence

**From Story 9.5 (Post-Sync Data Validation) — most recent completed story:**
- Pattern: new module in existing subpackage (`ingest/validation.py`), exported from `__init__.py`
- Test pattern: `tests/unit/test_validation.py` with controlled data fixtures
- Function-based API (not class-based) for stateless operations
- All functions typed for `mypy --strict`

**From Story 7.7 (Spike Research):**
- Comprehensive mathematical specification in `specs/research/game-theory-slider-mechanism.md`
- Verified worked examples with numpy (corrected 8 values where hand calculations were imprecise)
- Proposed function signatures, file locations, module structure — follow them

**From Story 7.5 (Bracket Visualizer):**
- `run_bracket_simulation()` orchestrator pattern in `simulation_helpers.py`
- `BracketSimulationResult` dataclass holds `prob_matrix`, `most_likely`, `bracket`
- `@st.cache_data(ttl=None)` caching decorator
- `BracketStructure.seed_map` already provides team_id → seed_num mapping

### Project Structure Notes

- New module `perturbation.py` lives in `src/ncaa_eval/evaluation/` alongside `simulation.py`, `providers.py`, `bracket.py` — same domain
- Subject to `mypy --strict` type checking
- Importable from both dashboard and Jupyter notebooks
- `from __future__ import annotations` required (enforced by Ruff)

### References

- [Source: specs/research/game-theory-slider-mechanism.md] — Full mathematical specification (§1-§9)
- [Source: specs/research/game-theory-slider-mechanism.md#§6] — Slider specifications and ranges
- [Source: specs/research/game-theory-slider-mechanism.md#§8] — UI integration design and function signatures
- [Source: _bmad-output/planning-artifacts/po-decision-log-epic8.md#1.1] — PO decision: implement sliders in Epic 9
- [Source: _bmad-output/planning-artifacts/epics.md#Story 9.7] — Acceptance criteria
- [Source: _bmad-output/implementation-artifacts/7-7-research-game-theory-slider-mechanism.md] — Story 7.7 spike story with code review notes
- [Source: docs/user-guide.md#529-533] — "NOT YET IMPLEMENTED" banner to remove
- [Source: src/ncaa_eval/evaluation/providers.py] — ProbabilityProvider protocol and build_probability_matrix()
- [Source: dashboard/lib/simulation_helpers.py] — run_bracket_simulation() orchestrator
- [Source: dashboard/pages/2_Presentation.py] — Bracket Visualizer page (slider insertion point)
- [Source: src/ncaa_eval/evaluation/__init__.py] — Package exports (add perturbation symbols)

## Dev Agent Record

### Agent Model Used

{{agent_model_name_version}}

### Debug Log References

### Completion Notes List

### File List
