# Story 7.7: Research Game Theory Slider Mechanism

Status: done

## Story

As a data scientist,
I want a documented analysis of how Game Theory sliders (Upset Aggression, Chalk Bias, Seed-Weight) should mathematically transform a model's base win probabilities,
so that the Bracket Visualizer (Story 7.5) can implement real-time probability perturbation with a sound mathematical foundation.

## Acceptance Criteria

1. **Candidate Transformations Evaluated**: At least three candidate mathematical transformations are evaluated (e.g., logit-space additive adjustments, multiplicative scaling, Bayesian prior blending). Each is assessed for intuitive user behavior (slider up = more upsets), numerical stability (probabilities remain valid 0-1), and reversibility (slider at neutral = original probabilities).

2. **Recommended Approach Documented**: A single recommended approach is documented with formula, worked examples for representative matchups (1v16, 5v12, 8v9), and edge case analysis (p=0.0, p=1.0, p=0.5).

3. **Slider Parameter Ranges Specified**: Default values and sensible ranges are defined for each slider (Upset Aggression, Chalk Bias, Seed-Weight). Neutral/identity values are clearly identified.

4. **Findings Committed as Project Document**: The spike research document is committed to `specs/research/` as a standalone reference for the implementation story that follows.

## Tasks / Subtasks

- [x] Task 1: Survey candidate mathematical transformations (AC: #1)
  - [x] 1.1: Research and document logit-space additive adjustment (`logit(p') = logit(p) + alpha * f(seed_diff)`)
  - [x] 1.2: Research and document temperature/power scaling (`p' = p^(1/T) / (p^(1/T) + (1-p)^(1/T))`)
  - [x] 1.3: Research and document linear blend with seed-prior (`p' = (1-w)*p + w*seed_prior(a, b)`)
  - [x] 1.4: Research and document entropy-based lambda parameterization (chalk-to-uniform interpolation)
  - [x] 1.5: Evaluate each candidate on the three assessment criteria (intuition, stability, reversibility)
- [x] Task 2: Select and fully specify the recommended approach (AC: #2)
  - [x] 2.1: Choose the best candidate (or hybrid) with rationale
  - [x] 2.2: Write the full mathematical specification with formulas
  - [x] 2.3: Compute worked examples for representative matchups: 1v16 (p=0.99), 5v12 (p=0.65), 8v9 (p=0.52), and 11v6 (p=0.40)
  - [x] 2.4: Analyze edge cases: p=0.0, p=1.0, p=0.5, extreme slider values
  - [x] 2.5: Document the relationship between the three sliders (independent vs. coupled)
- [x] Task 3: Define slider parameter specifications (AC: #3)
  - [x] 3.1: Specify Upset Aggression slider: range, default, step, UI label, mathematical effect
  - [x] 3.2: Specify Chalk Bias slider: range, default, step, UI label, mathematical effect
  - [x] 3.3: Specify Seed-Weight slider: range, default, step, UI label, mathematical effect
  - [x] 3.4: Document interaction effects when multiple sliders are non-default simultaneously
- [x] Task 4: Document UI integration design for future implementation (AC: #2, #3)
  - [x] 4.1: Specify where perturbation hooks into the existing simulation pipeline
  - [x] 4.2: Specify what needs to re-render when sliders change (bracket tree, heatmap, EP table) vs. what does not (MC simulation)
  - [x] 4.3: Propose function signature for the perturbation function
  - [x] 4.4: Propose file location and module structure
- [x] Task 5: Write and commit the spike research document (AC: #4)
  - [x] 5.1: Write `specs/research/game-theory-slider-mechanism.md`
  - [x] 5.2: Commit the document

## Dev Notes

### Spike Context — What This Research Must Answer

The UX spec (Section 4.1) defines three `st.sidebar.slider` controls — **Upset Aggression**, **Chalk Bias**, and **Seed-Weight** — that "perturb the model's base probabilities." The spec intentionally leaves the mathematical mechanism undefined. Story 7.5 (Bracket Visualizer, `done`) deferred slider implementation pending this spike.

**The core open question**: Given a model's base pairwise win probability `p(team_a beats team_b)` and three slider values, what is the formula for `p'(team_a beats team_b)` — the perturbed probability?

### Existing Simulation Pipeline Hook Point

The perturbation must operate on the 64x64 probability matrix (`prob_matrix`) that `build_probability_matrix()` produces in `src/ncaa_eval/evaluation/simulation.py`. The matrix is symmetric: `P[i][j] + P[j][i] = 1.0`.

**Insertion point** (from Story 7.5 `run_bracket_simulation()` in `dashboard/lib/filters.py`):
```
build_probability_matrix(provider, bracket.team_ids, context)
  → P  (64x64 numpy float64)
  → [PERTURBATION GOES HERE]  ← apply perturb(P, sliders, seed_map)
  → compute_most_likely_bracket(bracket, P_perturbed)
  → simulate_tournament(bracket, perturbed_provider, ...)
```

**Key constraint from UX spec**: "slider adjustments update the bracket visualization **without altering the underlying model data**." This means perturbation must not modify the cached model or original probability matrix — it produces a new matrix.

**Key constraint from architecture**: "Dashboard must never read files directly; must call `ncaa_eval` functions." The perturbation function should live in the `ncaa_eval` library (not `dashboard/lib/`) so it's testable with `mypy --strict` and usable from both Streamlit and Jupyter.

### Candidate Transformations to Evaluate

**1. Temperature Scaling (Logit-Space)**
The most established approach from ML calibration literature. The transformation:
```
logit(p) = log(p / (1-p))
logit(p') = logit(p) / T
p' = sigmoid(logit(p) / T)
```
Where `T` = temperature. `T=1` is identity, `T<1` sharpens (more chalk), `T>1` softens (more upsets → probabilities compress toward 0.5).

Properties:
- Preserves (0,1) range and symmetry: `p' + (1-p)' = 1`
- Reversible: T=1 is identity
- Single parameter controls chalk/upset spectrum
- Well-understood from neural network calibration (Guo et al., 2017)
- Edge: `p=0` and `p=1` require clipping to `[eps, 1-eps]`

**2. Power Transform (Generalized Temperature)**
```
p' = p^alpha / (p^alpha + (1-p)^alpha)
```
Where `alpha > 0`. `alpha=1` is identity, `alpha<1` compresses toward 0.5 (more upsets), `alpha>1` sharpens (more chalk).

Properties:
- Equivalent to temperature scaling in logit space
- Preserves symmetry: `p' + (1-p)' = 1`
- More numerically stable than logit at extremes (no log(0) issue)
- Single parameter, same chalk/upset spectrum as temperature

**3. Linear Blend with Seed Prior**
```
p' = (1-w)*p_model + w*p_seed_prior
```
Where `p_seed_prior` is the historical win probability for the given seed matchup (e.g., 1v16 historically ~0.99, 5v12 ~0.64). `w=0` is identity (pure model), `w=1` is pure seed prior.

Properties:
- Intuitive: blends model confidence with seed-based expectation
- Requires a seed prior lookup table (historical data)
- Does NOT compress/expand probabilities — it shifts them toward a fixed prior
- `w ∈ [0, 1]` range
- Distinct from temperature: doesn't affect the chalk/upset spectrum globally

**4. Entropy-Based Lambda (PMC11354004)**
```
Q_ij = lambda * P_ij + (1-lambda) * 0.5   [for lambda in [0,1]]
```
Interpolates between model probabilities and maximum entropy (50/50). `lambda=1` is identity, `lambda=0` is pure coin-flip. This is a simplified version of the entropy-based approach from Clair & Letscher (2024).

Properties:
- Very simple formula
- Compresses all probabilities toward 0.5
- Reversible at lambda=1
- But: does NOT distinguish between upset-favoring and chalk-favoring — it only has one direction (toward randomness)

### Three Sliders — Relationship Design

The three UX-spec sliders must be mapped to mathematical parameters. Key question: are they independent or coupled?

**Recommended mapping** (to be validated by spike):
- **Upset Aggression** → Temperature parameter (T). Higher T = probabilities compress toward 0.5 = more upsets in the most-likely bracket. Lower T = probabilities sharpen = more chalk.
- **Chalk Bias** → May be the inverse of Upset Aggression (redundant), OR a separate "sharpening" that only applies to high-probability matchups (p > threshold). Needs investigation.
- **Seed-Weight** → Linear blend weight (w). Higher w = model probabilities pull toward seed-based historical priors. This is genuinely independent of temperature.

If Chalk Bias is simply the inverse of Upset Aggression, the spike should recommend collapsing them into a single bidirectional slider or redefining Chalk Bias as something distinct (e.g., only sharpening p when the favorite is the higher seed).

### Available Data for Seed Priors

`BracketStructure.seed_map` (dict[int, int]: team_id → seed_num) is already computed in the simulation pipeline. Historical seed-vs-seed win rates are well-established:
- 1v16: 0.993 (151-1 through 2023)
- 2v15: 0.938
- 3v14: 0.854
- 4v13: 0.792
- 5v12: 0.646
- 6v11: 0.625
- 7v10: 0.604
- 8v9: 0.521

These can serve as the `p_seed_prior` for the Seed-Weight slider.

### Performance Constraint

The UX spec requires `< 500ms` interaction response for diagnostic plots and bracket updates. Perturbation of a 64x64 matrix is O(n^2) = 4,096 operations — negligible. However, if the perturbed matrix triggers a full `simulate_tournament()` re-run (MC mode with 10K sims), that takes 1-5s.

**Recommended approach**: When sliders change, only re-run `compute_most_likely_bracket()` and `compute_advancement_probs()` (analytical mode) from the perturbed matrix. Do NOT re-run MC simulation. The MC score distribution should use the original (unperturbed) probabilities as the "ground truth" simulation — sliders affect the bracket pick strategy, not the simulated outcomes.

### Project Structure Notes

- Spike output: `specs/research/game-theory-slider-mechanism.md` (new file)
- Future implementation story will add: `src/ncaa_eval/evaluation/perturbation.py` (library module) and slider controls to `dashboard/pages/2_Presentation.py`
- The perturbation module must pass `mypy --strict`

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 7.7] — AC definitions
- [Source: specs/04-front-end-spec.md#4.1] — "Simulation Sliders: Upset Aggression, Chalk Bias, Seed-Weight"
- [Source: specs/04-front-end-spec.md#3.1] — "User interactively toggles Game Theory sliders"
- [Source: dashboard/pages/2_Presentation.py] — Current bracket visualizer (no sliders yet)
- [Source: dashboard/lib/filters.py:362-453] — `run_bracket_simulation()` orchestrator (hook point)
- [Source: src/ncaa_eval/evaluation/simulation.py:218-260] — `ProbabilityProvider` Protocol
- [Source: src/ncaa_eval/evaluation/simulation.py:263-341] — `MatrixProvider` / `EloProvider` implementations
- [Source: src/ncaa_eval/evaluation/simulation.py:344-376] — `build_probability_matrix()` (insertion point for perturbation)
- [Source: PMC11354004] — Entropy-based bracket optimization (Clair & Letscher 2024)
- [Source: Guo et al. 2017] — "On Calibration of Modern Neural Networks" (temperature scaling)
- [Source: _bmad-output/implementation-artifacts/7-5-build-presentation-page-bracket-visualizer.md#Game Theory Sliders — OUT OF SCOPE] — Story 7.5 explicit deferral
- [Source: _bmad-output/implementation-artifacts/7-6-build-presentation-page-pool-scorer-point-outcome-analysis.md] — Pool Scorer page patterns (recent story)

### Previous Story Intelligence (7.5 + 7.6)

**From Story 7.5 (Bracket Visualizer):**
- `run_bracket_simulation()` already produces `prob_matrix` (64x64 numpy) — the perturbation target
- `compute_most_likely_bracket()` greedy traversal must re-run on perturbed matrix
- `plot_advancement_heatmap()` can accept any probability matrix — works with perturbed version
- `render_bracket_html()` takes `most_likely` results — works with perturbed bracket
- `n_simulations` and `scoring_name` are cache keys — slider values must also become cache keys when sliders are implemented
- `@st.cache_data(ttl=None)` is the correct caching decorator for simulation results
- `inspect.signature(scoring_cls)` pattern detects `seed_map` parameter on scoring constructors
- `BracketStructure.seed_map` is already available in `BracketSimulationResult.bracket.seed_map`
- Deferred architectural issue: `n_simulations` is a cache key for analytical mode causing spurious cache misses

**From Story 7.6 (Pool Scorer):**
- `score_bracket_against_sims()` scores YOUR picks against simulated outcomes — the spike must clarify whether perturbed picks should be scored against original or perturbed simulations
- `build_custom_scoring()` wraps `DictScoring` — same pattern could wrap perturbation params
- `BracketDistribution` dataclass holds all score distribution stats — no changes needed for perturbation
- Breadcrumb pattern, empty-state handling, and `_render_results()` extraction for C901 are well-established conventions

### Git Intelligence

Recent commits show Epic 7 is nearly complete (7.1-7.6 all done). Patterns:
- Story branches: `story/7-X-slug`
- Commit style: `feat(dashboard): <description> (Story 7.X)`
- All dashboard pages follow the same structure: breadcrumbs, session state, data loading, empty-state guards, render functions
- Quality gates: `mypy --strict src/ncaa_eval tests dashboard`, `ruff check .`, `pytest`
- Test count at last story: 865 passed, 1 skipped

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

- Verified all worked examples programmatically (Python numpy) — corrected 8 values where hand calculations were imprecise

### Completion Notes List

- **Task 1**: Surveyed four candidate transformations: logit-space additive, power/temperature scaling, linear blend with seed prior, and entropy-based lambda. Documented formulas, properties, and behavior for each. Built 7-criteria assessment matrix.
- **Task 2**: Selected hybrid approach (power transform + seed blend) as recommended. Justified orthogonality of the two parameters. Computed and verified worked examples for 4 representative matchups at 5 temperature levels and 5 seed weight levels, plus combined examples. Analyzed 7 edge cases including p=0, p=1, p=0.5, extreme sliders, diagonal entries, and simultaneous extremes.
- **Task 3**: Specified two-slider configuration (Upset Aggression [-5,+5] mapping to T=2^(v/3); Seed Weight 0-100%). Also documented three-slider alternative where Chalk Bias is a threshold-gated sharpener for top seeds. Recommended two-slider as default with three-slider as optional extension.
- **Task 4**: Documented pipeline insertion point (post-matrix, pre-analytical), re-render scope (bracket/heatmap/EP: yes; MC simulation: no), proposed `perturb_probability_matrix()` function signature with full type annotations, and proposed `src/ncaa_eval/evaluation/perturbation.py` module with helper functions.
- **Task 5**: Wrote comprehensive 9-section spike research document at `specs/research/game-theory-slider-mechanism.md` with quick-navigation table, verified numerical examples, and full reference list.

### Senior Developer Review (AI)

**Reviewer:** Claude Sonnet 4.6 (Code Review Agent)
**Date:** 2026-02-24
**Outcome:** ✅ APPROVED — all ACs satisfied; 2 Medium + 4 Low issues fixed in-place

**AC Validation:**
- AC1 (≥3 candidates): ✅ 4 candidates evaluated on 7 criteria
- AC2 (recommended approach + worked examples + edge cases): ✅ Hybrid power+blend; 4 matchups verified numerically; 7 edge cases analyzed
- AC3 (slider ranges/defaults/neutrals for all 3): ✅ Section 6.1 (Upset Aggression, Seed-Weight) + Section 6.2 (Chalk Bias alternative)
- AC4 (document committed to specs/research/): ✅ Committed in 18ab069

**Issues Fixed (6 total — 0 Critical/High, 2 Medium, 4 Low):**
- [M1 FIXED] Section 8.4 module outline corrected: `SEED_PRIOR_TABLE: dict[tuple,float]` → `FIRST_ROUND_SEED_PRIORS: dict[int, float]` (matches actual Section 8.5 code)
- [M2 FIXED] Chalk Bias Section 6.2 comment corrected: `slider=5 → threshold=0` → `threshold=1` (max(1,5-5)=1 not 0); effect description updated to match formula
- [L1 FIXED] Reference attribution corrected: "Clair & Letscher (2024)" → "Brown, Caro & Sullivan (2024)" for PMC11354004
- [L2 FIXED] Slider naming aligned to UX spec: "Seed Weight" → "Seed-Weight" (hyphenated) in Section 6.1 heading and UI Label
- [L3 FIXED] Seed prior limitation note added to Section 8.5 (later-round matchups use coarse 8v9 approximation for top-seed pairs)
- [L4 FIXED] PO approval gate added to Section 3.4 for the two-slider vs three-slider UX spec change recommendation

### Change Log

- 2026-02-24: Created spike research document — all 5 tasks complete, all 4 ACs satisfied
- 2026-02-24: Code review complete — 6 issues fixed; status set to done

### File List

- `specs/research/game-theory-slider-mechanism.md` (new) — Spike research document
- `_bmad-output/implementation-artifacts/7-7-research-game-theory-slider-mechanism.md` (modified) — Story file updates
- `_bmad-output/implementation-artifacts/sprint-status.yaml` (modified) — Status: ready-for-dev → review
