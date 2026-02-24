# Story 6.4: Research Tournament Simulation Confidence

Status: ready-for-dev

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a data scientist,
I want a documented analysis of how to improve confidence in tournament simulation predictions given limited historical data,
So that I can make informed decisions about simulation methodology and result interpretation for Stories 6.5 and 6.6.

## Acceptance Criteria

1. **Given** the tournament only happens once per year, limiting the historical dataset (~40 tournaments 1985–2025), **When** the data scientist reviews the spike findings document, **Then** statistical approaches for improving simulation confidence are evaluated (bootstrapping, Bayesian methods, ensemble simulations, analytical computation).

2. The impact of sample size on simulation stability is quantified — specifically, how many Monte Carlo simulations are needed for stable Expected Points estimates in a 64-team single-elimination bracket, with concrete error bounds at N=1,000 / 10,000 / 100,000.

3. Methods for computing confidence intervals on simulation outputs are documented — including standard MC confidence intervals, bootstrap methods (BCa), Bayesian credible intervals, and analytical uncertainty propagation via the delta method.

4. The analytical alternative to Monte Carlo simulation is evaluated: conditional probability tree traversal (Phylourny / Felsenstein pruning algorithm) that computes exact per-team advancement probabilities in O(n^2) time without simulation noise. A clear recommendation on when to use analytical vs. Monte Carlo is provided.

5. Recommendations for the simulation implementation (Story 6.5) are provided, covering: bracket representation, probability input contract, scoring rule interface, performance targets, and integration with the existing evaluation pipeline.

6. The findings are committed as a project document at `specs/research/tournament-simulation-confidence.md`.

## Tasks / Subtasks

- [ ] Task 1: Research simulation convergence and sample-size requirements (AC: #1, #2)
  - [ ] 1.1 Document the 1/sqrt(N) convergence rate and its implications for tournament simulation
  - [ ] 1.2 Compute concrete error bounds: for a team with P(championship)=0.05, calculate 95% CI widths at N=100, 1K, 10K, 100K
  - [ ] 1.3 Survey industry/competition practice: FiveThirtyEight (100K), Kaggle competitors (10K), BracketOdds, Dimers, SportsLine
  - [ ] 1.4 Document the compound variance issue: Expected Points requires stable per-round advancement probabilities for all 64 teams across 6 rounds

- [ ] Task 2: Research analytical alternatives to Monte Carlo (AC: #4)
  - [ ] 2.1 Evaluate Phylourny (Bettisworth & Jordan 2023, *Statistics and Computing*): O(n^2) exact computation via Felsenstein pruning; 599us for 64-team bracket vs. 15ms for 1K MC runs
  - [ ] 2.2 Evaluate "Stop Simulating!" (Brandes, Marmulla & Smokovic 2025, *Journal of Sports Analytics*): exact computation of tournament winning probabilities
  - [ ] 2.3 Document FiveThirtyEight's approach: conditional probability tree traversal (not MC) for their March Madness model
  - [ ] 2.4 Assess feasibility for ncaa_eval: can we implement analytical computation and fall back to MC only when needed (e.g., score-distribution modeling)?

- [ ] Task 3: Research confidence interval methods (AC: #3)
  - [ ] 3.1 Standard MC confidence intervals: p_hat +/- 1.96 * sqrt(p*(1-p)/N)
  - [ ] 3.2 Bootstrap methods: nonparametric, BCa (bias-corrected and accelerated), parametric bootstrap
  - [ ] 3.3 Two-layer bootstrap: perturb model params (outer loop, B=500–1K) x analytical EP computation (inner loop) — separates model uncertainty from simulation noise
  - [ ] 3.4 Bayesian credible intervals: sample from posterior, compute advancement probs per draw, report quantiles
  - [ ] 3.5 Delta method / analytical uncertainty propagation through the probability computation tree

- [ ] Task 4: Research small-sample mitigation strategies (AC: #1)
  - [ ] 4.1 Game-level modeling (~200K observations) vs. tournament-level modeling (~40 observations)
  - [ ] 4.2 Ensemble / composite ratings (FiveThirtyEight's 6-system blend)
  - [ ] 4.3 Parametric assumptions: BracketOdds' truncated geometric distribution for seed advancement
  - [ ] 4.4 Bayesian logistic regression with informative priors (Duquesne thesis, won MMLM 2015 & 2017)
  - [ ] 4.5 Conformal prediction (*The American Statistician*, 2023): distribution-free calibration with fewer assumptions

- [ ] Task 5: Produce implementation recommendations for Story 6.5 (AC: #5)
  - [ ] 5.1 Bracket representation: tree vs. DataFrame vs. array — recommend approach for 64-team single-elimination
  - [ ] 5.2 Probability input contract: how to generate P(A beats B) for hypothetical matchups; stateful vs. stateless model dispatch
  - [ ] 5.3 Scoring rule interface: Standard (1-2-4-8-16-32), Fibonacci, Seed-Difference Bonus, custom callable — plugin-registry compatible
  - [ ] 5.4 Performance targets: exact analytical computation for fixed brackets; MC only for score-distribution or bracket-count analysis
  - [ ] 5.5 Integration with existing evaluation pipeline: separate `SimulationResult` vs. extension of `BacktestResult`
  - [ ] 5.6 Data requirements: tournament seeds (from `MNCAATourneySeeds.csv` already ingested in Story 4.3), bracket structure (hardcoded or data-driven)

- [ ] Task 6: Write and commit findings document (AC: #6)
  - [ ] 6.1 Write `specs/research/tournament-simulation-confidence.md` with all findings
  - [ ] 6.2 Include a decision matrix: analytical vs. MC for each use case
  - [ ] 6.3 Include concrete pseudocode for the recommended simulation approach
  - [ ] 6.4 Commit the findings document

## Dev Notes

### This is a Spike (Research) Story

This story produces a **research document** (`specs/research/tournament-simulation-confidence.md`), not production code. The output directly informs Stories 6.5 (Monte Carlo Tournament Simulator) and 6.6 (Tournament Scoring).

### Key Research Questions to Answer

1. **Analytical vs. Monte Carlo**: Should ncaa_eval implement exact conditional-probability computation (O(n^2), zero simulation noise) and only fall back to MC for scenarios where analytical computation is infeasible?

2. **Simulation count**: If MC is used, what is the minimum N for model comparison (likely 10K–50K) vs. production bracket analysis (likely 50K–100K)?

3. **Confidence intervals**: The two-layer bootstrap (perturb model params x analytical EP) cleanly separates model uncertainty from simulation noise — is this the recommended approach?

4. **Bracket representation**: The tournament is a fixed 64-team single-elimination bracket (post-First Four). What data structure best supports both analytical computation and MC simulation?

5. **Probability generation for hypothetical matchups**: Stateful models (Elo) can generate P(A beats B) from ratings alone. Stateless models (XGBoost) require synthetic feature rows. How should the simulator handle both?

### Prior Art Survey Inputs

**Academic:**
- Phylourny (Bettisworth & Jordan 2023): exact tournament probability computation, 599us for 64-team bracket
- "Stop Simulating!" (Brandes et al. 2025): exact computation faster than simulation
- Conformal win probabilities (*American Statistician*, 2023): distribution-free calibration
- Bayesian logistic regression for NCAA (Duquesne thesis): uncertainty quantification with priors
- BracketOdds (U. Illinois, 2007–present): truncated geometric seed distributions, INFORMS 2013 award

**Industry:**
- FiveThirtyEight: conditional probability tree (analytical), not MC; blends 6 rating systems
- BracketOdds: power model + 10K–100K simulations
- Dimers/SportsLine: 10K simulations standard

**Competition (Kaggle MMLM):**
- Matthews & Lopez (2014 winners): 10K MC simulations, estimated ~12% probability of winning even with optimal model
- Competitors submit per-matchup probabilities (not bracket picks) — MC is used for internal evaluation only
- XGBoost dominant for calibrated game-level probabilities
- Probability clipping as post-processing (reduce <0.35 by 0.05, increase >0.65 by 0.05)

### Existing Codebase Context

**What exists (DO NOT reimplement):**
- `evaluation/metrics.py`: log_loss, brier_score, roc_auc, expected_calibration_error — vectorized numpy
- `evaluation/splitter.py`: walk_forward_splits, CVFold — temporal boundary enforcement
- `evaluation/backtest.py`: run_backtest, FoldResult, BacktestResult — parallel joblib orchestration
- `model/base.py`: Model ABC with predict_proba(X) -> pd.Series; StatefulModel with _predict_one(team_a_id, team_b_id) -> float
- `ingest/schema.py`: Game Pydantic model with is_tournament flag
- `transform/feature_serving.py`: StatefulFeatureServer with serve_season_features()
- Tournament seeds: already ingested via `MNCAATourneySeeds.csv` in Story 4.3 (seed_num, region, is_play_in)

**What does NOT exist yet (Story 6.5+ will create):**
- Bracket structure definition (64-team single-elimination tree)
- Tournament seed lookup table (separate from feature matrix)
- Matchup probability generator for hypothetical pairings
- Monte Carlo / analytical simulation engine
- Scoring rule specification (point schedules per round)
- SimulationResult data model

### Architecture Constraints

- `TournamentBracket` entity defined in Architecture: `SimulationID`, `BracketStructure` (JSON/Graph), `TotalScore`
- Evaluation Engine interface: `simulate_tournament(probs)` — function signature from architecture spec
- **NFR1 (Vectorization)**: per-simulation probability lookups should be vectorized with numpy
- **NFR2 (Parallelism)**: parallel execution across N simulations if MC is used
- **NFR3 (Extensibility)**: scoring rules should use plugin-registry pattern
- Performance: PRD explicitly excludes MC simulation time from the 60-second backtest target

### Critical Implementation Constraints (for the research document)

1. **`from __future__ import annotations`** required in all Python files.
2. **`mypy --strict`** mandatory.
3. **Google docstring style** (Args:, Returns:, Raises:).
4. **Library-First Rule**: evaluate `phylourny` PyPI package, `numpy` batch sampling, `scipy.stats` for confidence intervals. Do NOT reimplement from scratch if a library handles it.
5. The research document should be written as a self-contained reference that Story 6.5's dev agent can use without additional context.

### Previous Story Learnings (Stories 6.1–6.3)

- **Google docstring style**: NOT NumPy style. This was the #1 code review fix across all Epic 6 stories.
- **Frozen dataclasses**: Use for immutable result containers (FoldResult, BacktestResult pattern).
- **Mode validation at entry point**: Public APIs must validate at entry point, not delegate.
- **Library-First**: Use joblib (parallelism), scikit-learn (metrics), numpy (vectorization) — already in pyproject.toml.
- **METADATA_COLS**: Defined in `evaluation/backtest.py` (moved from `cli/train.py` in Story 6.3).
- **Exception guards**: Broad `except Exception` for per-fold/per-metric safety.

### Git Intelligence

Recent commits show Epic 6 in active development:
- `293abd1` Story 6.3: Parallel cross-validation backtest
- `484786f` Story 6.2: Walk-forward CV splitter
- `f301fd6` Story 6.1: Evaluation metric library

All three stories follow the same pattern: frozen dataclasses for results, vectorized numpy operations, joblib for parallelism, Google docstring style, and comprehensive unit tests.

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Epic 6, Story 6.4 — acceptance criteria]
- [Source: specs/03-prd.md line 54 — FR9 Monte Carlo Tournament Simulator]
- [Source: specs/03-prd.md line 162–163 — Stories 5.3 and 5.4 simulation spike and implementation]
- [Source: specs/03-prd.md line 173 — 60-second backtest target excludes MC simulation time]
- [Source: specs/05-architecture-fullstack.md lines 160–162 — TournamentBracket entity definition]
- [Source: specs/05-architecture-fullstack.md lines 186–187 — evaluate/simulate_tournament interface]
- [Source: src/ncaa_eval/evaluation/metrics.py — existing metric functions]
- [Source: src/ncaa_eval/evaluation/backtest.py — parallel backtest orchestrator, FoldResult/BacktestResult]
- [Source: src/ncaa_eval/model/base.py — Model ABC, StatefulModel._predict_one()]
- [Source: _bmad-output/implementation-artifacts/6-3-implement-parallel-cross-validation-execution.md — previous story learnings]
- [Source: Phylourny — Bettisworth & Jordan 2023, Statistics and Computing (PMC10186292)]
- [Source: "Stop Simulating!" — Brandes et al. 2025, Journal of Sports Analytics]
- [Source: FiveThirtyEight — How Our March Madness Predictions Work methodology page]
- [Source: BracketOdds — bracketodds.cs.illinois.edu, University of Illinois]

## Dev Agent Record

### Agent Model Used

{{agent_model_name_version}}

### Debug Log References

### Completion Notes List

### File List
