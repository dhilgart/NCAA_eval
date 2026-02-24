# Tournament Simulation Confidence — Research Findings

**Date:** 2026-02-23
**Story:** 6.4 (Spike) — Research Tournament Simulation Confidence
**Status:** Complete — Awaiting Review
**Author:** Dev Agent (Claude Opus 4.6)

---

## Quick-Navigation Table

| Section | Topic | Key Takeaway |
|:---|:---|:---|
| [1. Simulation Convergence](#1-simulation-convergence-and-sample-size-requirements) | MC convergence rate and error bounds | 1/sqrt(N) convergence; 10K sims gives +/-0.4 pp for p=0.05 |
| [2. Analytical Alternatives](#2-analytical-alternatives-to-monte-carlo) | Phylourny, "Stop Simulating!", FiveThirtyEight | Exact O(n^2) computation eliminates simulation noise entirely |
| [3. Confidence Intervals](#3-confidence-interval-methods) | Standard MC, bootstrap, Bayesian, delta method | Two-layer bootstrap (perturb params x analytical EP) is recommended |
| [4. Small-Sample Mitigation](#4-small-sample-mitigation-strategies) | Game-level modeling, ensembles, Bayesian priors, conformal | Model at game level (200K obs), not tournament level (40 obs) |
| [5. Implementation Recommendations](#5-implementation-recommendations-for-story-65) | Bracket repr, probability contract, scoring, perf targets | Analytical first; MC only for score-distribution analysis |
| [6. Decision Matrix](#6-decision-matrix-analytical-vs-monte-carlo) | When to use each approach | Analytical for EP/advancement; MC for bracket-count/score-distribution |
| [7. Pseudocode](#7-pseudocode-recommended-simulation-approach) | Analytical EP, two-layer bootstrap CI, MC fallback | provider_factory pattern for model-agnostic CI; batch prob matrix |
| [8. Recommendations Summary](#8-key-recommendations-summary) | 8 concrete decisions for Story 6.5 | START HERE for Story 6.5 implementation |

---

## 1. Simulation Convergence and Sample-Size Requirements

### 1.1 The 1/sqrt(N) Convergence Rate

Monte Carlo simulation converges at O(1/sqrt(N)), a consequence of the Central Limit Theorem. For a Bernoulli proportion p estimated from N independent simulation runs:

```
SE(p_hat) = sqrt(p * (1 - p) / N)
```

This means:
- **Halving the error requires 4x the simulations.**
- **Reducing error by 10x requires 100x the simulations.**
- The rate is dimension-independent — the only advantage MC has over grid methods for high-dimensional problems.

### 1.2 Concrete Error Bounds

For a team with P(championship) = 0.05, the 95% confidence interval width is:

```
CI_width = 2 * 1.96 * sqrt(0.05 * 0.95 / N) = 0.854 / sqrt(N)
```

| N | SE | 95% CI Width | CI as [low, high] | Interpretation |
|---:|-----:|-----:|:---|:---|
| 100 | 0.0218 | 0.0854 | [0.7%, 9.3%] | Useless — can't distinguish 3% from 7% |
| 1,000 | 0.0069 | 0.0270 | [3.7%, 6.4%] | Wide — +/-1.4 percentage points |
| 10,000 | 0.0022 | 0.0085 | [4.6%, 5.4%] | Reasonable — +/-0.4 percentage points |
| 100,000 | 0.0007 | 0.0027 | [4.9%, 5.1%] | Precise — +/-0.14 percentage points |

**Relative error** (from Phylourny benchmarks, NCAA 64-team bracket):

| MC Simulations | Median Relative Error |
|---:|---:|
| 100 | 72.7% |
| 1,000 | 23.2% |
| 10,000 | 7.3% |
| 100,000 | 2.3% |

### 1.3 Industry Practice Survey

| Source | Simulation Count | Method | Context |
|:---|---:|:---|:---|
| FiveThirtyEight | 0 (analytical) | Conditional probability tree | Production forecasting |
| BracketOdds (U. Illinois) | 50,000 per batch | Power model + MC | Academic research |
| Dimers / SportsLine | ~10,000 | MC simulation | Commercial forecasting |
| KenPom | ~10,000 | MC simulation | Rating system evaluation |
| Kaggle MMLM competitors | 10,000 (internal) | MC for bracket analysis | Competition evaluation |
| Matthews & Lopez (2014) | 10,000 | MC simulation | Luck vs. skill analysis |

### 1.4 The Compound Variance Problem

A 64-team single-elimination bracket requires estimating **384 advancement probabilities** (64 teams × 6 rounds), plus 64 team-level Expected Points and 1 bracket-level Expected Points — **449 total estimates**.

The compound variance issue arises because later-round estimates depend on earlier rounds:

```
P(A reaches Final Four) = P(A wins R1) × P(A wins R2 | A's R2 opponent)
                        × P(A wins Sweet 16 | A's S16 opponent)
```

Each factor depends on the bracket outcomes of other games. The 384 probabilities are **not independent** — exactly one team advances per matchup. Variance compounds multiplicatively through the dependency chain.

**Practical consequence:** At N=10,000, while any single championship probability has a CI width of ~0.85 percentage points (for p=0.05), rare advancement events (e.g., a 16-seed reaching the Elite Eight, p ≈ 0.001) may have only ~10 occurrences across 10,000 simulations, giving CI widths comparable to the estimate itself.

**This is the primary motivation for analytical computation** — it eliminates simulation noise for all 384 advancement probabilities simultaneously.

---

## 2. Analytical Alternatives to Monte Carlo

### 2.1 Phylourny (Bettisworth & Jordan 2023)

**Paper:** "Phylourny: efficiently calculating elimination tournament win probabilities via phylogenetic methods," *Statistics and Computing* 33(4):80, 2023.

**Core insight:** A single-elimination tournament bracket is structurally isomorphic to a phylogenetic (bifurcating) tree. Teams map to leaf nodes, matches to internal nodes, and the final to the root. The Felsenstein Pruning Algorithm — designed for phylogenetic likelihood computation — computes exact per-team advancement probabilities.

**Algorithm:** Post-order (bottom-up) traversal of the tournament tree, computing a Win Probability Vector (WPV) at each node:

```
R = V ⊙ (W · Pᵀ) + W ⊙ (V · Pᵀ)
```

Where:
- R is the n-dimensional win probability vector at the parent node
- V, W are WPVs from left and right children
- P is the n×n pairwise win probability matrix
- ⊙ is element-wise (Hadamard) product
- · is matrix-vector multiplication

**Leaf initialization:** Each leaf is a canonical unit vector (team i gets eᵢ).
**Root result:** The WPV at the root contains exact championship probabilities for all n teams.

**Complexity:** O(n²) — each of n internal nodes performs O(n) operations. This replaces the naive O(n · 2ⁿ) approach, which is infeasible for n=64.

**Performance benchmarks:**

| Tournament | Teams | Phylourny | Naive | MC (1K sims) |
|:---|---:|---:|---:|---:|
| UEFA EURO 2020 | 16 | **15 μs** | 1,413 μs | 2,111 μs |
| NCAA 2022 | 64 | **599 μs** | >2 hours | 15,010 μs |

Phylourny is **25x faster than 1K MC runs** and produces **exact results with zero noise**.

**PyPI package:** Does not exist. Phylourny is implemented in C++ (95.2% of codebase). However, the algorithm is straightforward to reimplement in Python/NumPy — the core is a recursive WPV formula applied via post-order traversal, with the inner operation being element-wise products and matrix-vector multiplications.

**MCMC integration:** The paper demonstrates integrating Phylourny into a Bayesian workflow — 100K MCMC samples with Phylourny evaluation took 51 seconds for 16 teams and ~1.5 hours for 64 teams.

### 2.2 "Stop Simulating!" (Brandes, Marmulla & Smokovic 2025)

**Paper:** "Efficient computation of tournament winning probabilities," *Journal of Sports Analytics* 11, 2025. (ArXiv: 2307.10411, July 2023.)

**Core result:** Same fundamental insight as Phylourny — exact computation via bottom-up bracket traversal — extended to handle **complex formats with group stages** (e.g., FIFA World Cup). For pure single-elimination brackets, the algorithm is equivalent to Phylourny.

**Performance:** Exact computation takes time equivalent to **a few hundred MC simulation runs**, representing a 2-orders-of-magnitude improvement over any reasonably accurate MC approximation.

**Key differences from Phylourny:**

| Aspect | Phylourny | Stop Simulating! |
|:---|:---|:---|
| Tournament format | Pure single-elimination only | Group stage + knockout (FIFA) |
| Algorithm origin | Phylogenetics (Felsenstein pruning) | Graph theory / operations research |
| Group stage handling | Not addressed | Core contribution |
| Implementation | C++ command-line tool (GitHub) | Research paper (no public code) |
| Application | NCAA, UEFA EURO | FIFA World Cup |

**For ncaa_eval:** Both papers confirm that exact computation is the right approach for NCAA March Madness (a fixed 64-team single-elimination bracket). The group-stage extensions in "Stop Simulating!" are not needed.

### 2.3 FiveThirtyEight's Approach

FiveThirtyEight explicitly stated they **"directly calculate"** the chance of teams advancing to a given round using conditional probability tree traversal — the same conceptual approach as Phylourny, implemented before the Phylourny paper was published.

**Per-game win probability formula:**

```
P(A beats B) = 1 / (1 + 10^(-diff × 30.464 / 400))
```

Where `diff` is the travel-adjusted composite power rating difference.

**Rating blend:** 6 computer systems (KenPom, Sagarin, Moore, LRMC, ESPN BPI, Elo) weighted 75% + 2 human rankings (NCAA S-curve, preseason polls) weighted 25%.

### 2.4 Feasibility Assessment for ncaa_eval

**Recommendation: Implement analytical computation as the primary method; MC only as fallback.**

| Use Case | Method | Rationale |
|:---|:---|:---|
| Per-team advancement probabilities | Analytical | Exact, zero noise, 599μs |
| Expected Points (EP) computation | Analytical | EP = sum(advancement_prob × round_points) — arithmetic on exact values |
| EP confidence intervals | Two-layer bootstrap + analytical inner | Captures model uncertainty, eliminates simulation noise |
| Score-distribution analysis | MC simulation | Analytical methods compute probabilities, not score distributions |
| Bracket-count analysis ("how many brackets have X winning?") | MC simulation | Requires sampling from the joint distribution of all games |

**Implementation path:**
1. Implement the Phylourny algorithm in Python/NumPy (the WPV recursion is ~30 lines of vectorized code).
2. Use it as the inner loop for all EP and advancement probability computations.
3. Implement MC simulation as a separate path for score-distribution and bracket-count use cases.
4. The MC path can reuse the same probability matrix P that feeds the analytical path.

---

## 3. Confidence Interval Methods

### 3.1 Standard MC Confidence Intervals

```
p_hat ± z_{α/2} × sqrt(p_hat × (1 - p_hat) / N)
```

For 95% CI: z = 1.96.

**When sufficient:** Large N, probabilities not near 0 or 1, model parameters treated as fixed (quantifying simulation noise only).

**Limitations:**
- Does not capture **model uncertainty**.
- Poor for rare events: a 16-seed championship (p ≈ 0.0001) needs N > 50,000 just for Np ≥ 5.
- Assumes symmetric intervals — problematic near boundaries.

**Verdict for ncaa_eval:** Insufficient as the primary CI method because it ignores model uncertainty. If we use analytical computation, there is no simulation noise to quantify in the first place.

### 3.2 Bootstrap Methods

#### Nonparametric Bootstrap

Resample B times (with replacement) from original MC output, compute statistic per resample, use quantiles for CI.

**Percentile method:** Use α/2 and (1 − α/2) quantiles of bootstrap distribution.

#### BCa (Bias-Corrected and Accelerated) Bootstrap

Adjusts percentile intervals for bias and skewness (Efron, 1987):

```
α₁ = Φ(ẑ₀ + (ẑ₀ + z_{α/2}) / (1 − â(ẑ₀ + z_{α/2})))
α₂ = Φ(ẑ₀ + (ẑ₀ + z_{1−α/2}) / (1 − â(ẑ₀ + z_{1−α/2})))
```

Where ẑ₀ is the bias-correction parameter and â is the acceleration parameter (estimated via jackknife).

**Achieves second-order accuracy** (error O(1/N)) vs. first-order (O(1/sqrt(N))) for standard percentile intervals. Particularly valuable for probabilities near 0 or 1 (common in tournament brackets).

#### Parametric Bootstrap

Draw B samples of model parameters from their estimated distribution (e.g., MLE's asymptotic normal), compute statistic per draw. More efficient when the model is approximately correct.

### 3.3 Two-Layer Bootstrap (Recommended)

This is the key architecture for cleanly separating **model uncertainty** from **simulation noise**.

**Outer loop (B = 500–1,000 iterations):** Perturbs model parameters.
- Draw θ_b from the posterior or bootstrap distribution of model parameters.
- Captures **epistemic uncertainty** (uncertainty in the true model).

**Inner loop:** For each θ_b, compute **analytical** Expected Points.
- Given fixed θ_b, advancement probabilities are deterministic via the Phylourny algorithm.
- **Zero simulation noise** in the inner computation.

```python
expected_points: list[float] = []
for b in range(B):  # B = 500–1000
    theta_b = draw_from_posterior(model)        # Perturb parameters
    P_b = build_probability_matrix(theta_b)     # n×n pairwise probs
    adv_probs = phylourny_compute(bracket, P_b) # Exact analytical
    ep_b = (adv_probs * scoring_vector).sum()   # Expected Points
    expected_points.append(ep_b)

ci_lower = np.percentile(expected_points, 2.5)
ci_upper = np.percentile(expected_points, 97.5)
```

**Why this is the recommended approach:**
- CI width reflects **only genuine model uncertainty**, not artificial MC noise.
- Computationally cheap: 1,000 Phylourny evaluations at 599μs each = 0.6 seconds total.
- Straightforward to implement with any model that can produce parameter draws.
- Composes naturally with both Bayesian (posterior draws) and frequentist (bootstrap draws) models.

### 3.4 Bayesian Credible Intervals

**Procedure:**
1. Fit Bayesian model (e.g., Bayesian logistic regression) to game-level data.
2. Draw S samples {θ₁, ..., θ_S} from posterior p(θ | data).
3. For each θ_s, compute analytical advancement probabilities and EP.
4. Report quantiles.

**Two types:**
- **Equal-Tailed Interval (ETI):** α/2 and (1 − α/2) quantiles. Simple, may include low-density regions for skewed posteriors.
- **Highest Density Interval (HDI):** Narrowest interval containing (1−α) of posterior mass. Preferred for skewed posteriors.

**Advantage:** Direct probability interpretation — "95% probability the true value lies in this interval" (given model and prior). Natural fit for tournament prediction questions.

**Practical note:** If using PyMC or NumPyro for Bayesian model fitting, posterior samples are automatically available, making this approach seamless with the two-layer architecture above.

### 3.5 Delta Method / Analytical Uncertainty Propagation

Propagates uncertainty through a differentiable function g(θ):

```
Var(g(θ̂)) ≈ ∇g^T · Σ · ∇g
```

Where Σ is the parameter covariance matrix.

**Limitations for tournament brackets:**
- The probability tree is highly nonlinear (products and sums across 6 rounds).
- Jacobian has 384 outputs × hundreds of model parameters.
- First-order approximation poor when probabilities are near 0 or 1.

**Verdict for ncaa_eval:** Not recommended. The two-layer bootstrap handles nonlinearity naturally and is simpler to implement.

### 3.6 Method Comparison

| Method | Captures Model Uncertainty | Simulation Noise | Distribution-Free | Cost | Best For |
|:---|:---:|:---:|:---:|:---|:---|
| Standard MC CI | No | Yes | No | Low | Quick checks, large N |
| Nonparametric Bootstrap | Partial | Yes | Yes | Medium | Unknown distributions |
| BCa Bootstrap | Partial | Yes | Yes | Medium-High | Skewed/biased estimates |
| **Two-Layer Bootstrap** | **Yes** | **Eliminated** | **Depends on outer** | **Medium** | **Recommended for EP CIs** |
| Bayesian Credible Interval | Yes | Eliminated | No | High (MCMC) | Full probabilistic inference |
| Delta Method | Yes | N/A | No | Low | Quick approximate variance |

---

## 4. Small-Sample Mitigation Strategies

### 4.1 Game-Level vs. Tournament-Level Modeling

The NCAA tournament has only ~40 historical tournaments (1985–2025), but the underlying game-level data spans **~200,000+ regular-season and tournament games** over the same period.

**Why game-level modeling provides more statistical power:**

1. **Observation count:** ~200K game outcomes vs. ~40 tournament brackets.
2. **Transferability:** A game-level model ("given team A's and B's features, what is P(A wins)?") applies equally to regular-season and tournament games.
3. **Feature richness:** Each game provides efficiency ratings, pace, strength-of-schedule signals.
4. **Reduced overfitting:** "P(1-seed beats 16-seed in R1)" has only ~160 observations (4 matchups × 40 years). A game-level model uses thousands of games between similarly-ranked teams.
5. **Continuous updating:** Game-level ratings (Elo, efficiency) evolve throughout the season.

**This is the consensus approach across FiveThirtyEight, KenPom, BracketOdds, and Kaggle MMLM winners.** All model at the game level, then feed per-game probabilities into the tournament structure.

### 4.2 Ensemble / Composite Ratings

FiveThirtyEight's approach: blend **6 independent computer rating systems** plus 2 human rankings.

**The 6 computer systems:** KenPom, Sagarin, Moore, LRMC (Sokol), ESPN BPI, FiveThirtyEight Elo.
**Human rankings (25% weight):** NCAA S-curve, preseason polls.

**Why blending mitigates small-sample issues:**
- Each system uses different methodology, features, and assumptions.
- Blending reduces variance (ensemble averaging).
- Partially uncorrelated errors make the blend more robust.
- Human component adds information not captured by algorithms (injuries, eye-test).

**Kaggle pattern:** Many top competitors use the mean of the top 10 historically-accurate Massey Ordinal systems as their primary feature.

### 4.3 BracketOdds' Truncated Geometric Distribution

**Reference:** Jacobson, Nikolaev, King & Lee (2011), "Seed distributions for the NCAA men's basketball tournament," *Omega* 39(2):116–125.

Rather than modeling individual game outcomes, BracketOdds models the **distribution of seeds advancing to each round** using a truncated geometric distribution:

```
P(X = k) = k × q × (1 - q)^{k-1},  for k = 1, 2, ..., k_max
```

Where q is estimated from historical data as the inverse of the average seed number reaching that round.

**Goodness of fit:** χ² = 18.321 (p = 0.246 at 15 df) for Final Four seed distributions 1985–2019.

**Advantage:** Instead of 384 individual advancement probabilities, estimates a single parameter per round — dramatically reducing parameters and leveraging the structured nature of seeded tournaments.

**For ncaa_eval:** This is a useful validation tool (do our simulated seed distributions match the historical truncated geometric?), not a replacement for game-level modeling.

### 4.4 Bayesian Logistic Regression with Informative Priors

**Track record:** Won Kaggle MMLM in 2015 (Bradshaw), 2017 (Landgraf — 1st place with rstanarm).

**Why informative priors help:**
- Regular-season data (200K+ games) provides strong prior information on team strength parameters.
- Priors regularize, preventing overfitting to the small tournament sample.
- Posterior distribution naturally quantifies parameter uncertainty.
- Prior knowledge like "home court ≈ 3–4 points" stabilizes estimates when tournament data is sparse.

**Landgraf's 2017 innovation:** Built a second mixed-effects model (lme4) to predict competitor submissions, then optimized his own submission to maximize expected rank against the predicted field — a meta-strategy that goes beyond model accuracy.

### 4.5 Conformal Prediction

**Reference:** Johnstone & Nettleton (2023), "Using Conformal Win Probability to Predict the Winners of the Canceled 2020 NCAA Basketball Tournaments," *The American Statistician* 78(3).

**What it is:** A framework for uncertainty quantification with **distribution-free, finite-sample coverage guarantees** under only an exchangeability assumption:

```
P(Y_{n+1} ∈ C(X_{n+1})) ≥ 1 − α
```

This holds exactly in finite samples — no asymptotic argument needed.

**Key findings:**
- Conformal win probabilities were **well-calibrated** across the full probability range.
- For **underdogs** (low win probability), conformal prediction was **much better calibrated** than logistic or linear regression.
- Validated on 2011–2023 seasons.
- Code: https://github.com/chancejohnstone/marchmadnessconformal

**For ncaa_eval:** Conformal prediction is best positioned as a **calibration diagnostic** — regardless of the primary model, conformal methods can verify that output probabilities are well-calibrated. It could also be used as a post-processing wrapper around any model's raw probabilities.

---

## 5. Implementation Recommendations for Story 6.5

### 5.1 Bracket Representation

**Recommendation: NumPy array-based tree with region structure.**

A 64-team single-elimination bracket has 63 games across 6 rounds. The bracket has 4 regions of 16 teams each; regions merge at the Final Four.

```python
# Bracket as array: seeds[region][position] → team_id
# 4 regions × 16 seeds = 64 teams
bracket_seeds: np.ndarray  # shape (4, 16), dtype=int

# Game results as array: results[round][game_index] → winning_team_id
# Round 0 (R64): 32 games, Round 1 (R32): 16, ..., Round 5 (NCG): 1
# Total: 32 + 16 + 8 + 4 + 2 + 1 = 63 games
```

**Why array over tree or DataFrame:**
- **Tree:** Natural for Phylourny traversal but overhead for Python object creation; a flat array with index arithmetic is equivalent and faster.
- **DataFrame:** Too heavy for 63 games; array indexing is simpler and vectorizes better.
- **Array:** Index arithmetic maps parent/child relationships: game i's children are games 2i+1 and 2i+2 (0-indexed). Compatible with both analytical traversal and MC batch simulation.

**Bracket structure** can be hardcoded (the NCAA bracket structure is fixed: 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15 in each region). Data-driven structure is unnecessary for NCAA; the matchup order hasn't changed since 1985.

### 5.2 Probability Input Contract

**The simulator needs:** For any two teams (A, B), produce P(A beats B).

**Stateful models (Elo):**
```python
# Already have: StatefulModel._predict_one(team_a_id, team_b_id) -> float
p = model._predict_one(team_a, team_b)
```

**Stateless models (XGBoost):**
```python
# Need to generate a synthetic feature row for a hypothetical matchup.
# NOTE: StatefulFeatureServer does NOT currently have this method.
# Story 6.5 must add serve_matchup_features_batch() to StatefulFeatureServer.
# The batch method (used via build_probability_matrix) generates a DataFrame
# with one row per (team_a, team_b) pair and feeds it to model.predict_proba()
# in a single call (NFR1-compliant).
features_batch = feature_server.serve_matchup_features_batch(
    team_a_ids, team_b_ids, context
)
probs = model.predict_proba(features_batch)  # pd.Series of P(a wins)
```

**Recommended contract:**

```python
class ProbabilityProvider(Protocol):
    """Generate P(A beats B) for hypothetical matchups."""

    def matchup_probability(
        self, team_a_id: int, team_b_id: int, context: MatchupContext
    ) -> float: ...

    def batch_matchup_probabilities(
        self,
        team_a_ids: Sequence[int],
        team_b_ids: Sequence[int],
        context: MatchupContext,
    ) -> np.ndarray: ...
```

Where `MatchupContext` carries season, day_num, and neutral-court flag. This protocol is implementable by both stateful (Elo — ignore context, use internal ratings) and stateless (XGBoost — use context to generate a batch feature matrix) models.

**Batch interface for the probability matrix (NFR1-compliant):**

For stateless models (XGBoost), the probability matrix must be built using **batched inference**, not per-pair prediction loops, to comply with NFR1 (Vectorization):

```python
def build_probability_matrix(
    provider: ProbabilityProvider,
    team_ids: Sequence[int],
    context: MatchupContext,
) -> np.ndarray:
    """Build n×n matrix P where P[i,j] = P(team_i beats team_j).

    Uses batch inference for stateless models (single predict_proba call
    over all n*(n-1)/2 pairs) and scalar dispatch for stateful models.
    Complies with NFR1 (Vectorization).
    """
    n = len(team_ids)
    # Generate upper-triangle indices: 2,016 pairs for n=64
    rows, cols = np.triu_indices(n, k=1)
    a_ids = [team_ids[i] for i in rows]
    b_ids = [team_ids[j] for j in cols]

    # Single batched call — provider builds feature matrix internally
    probs = provider.batch_matchup_probabilities(a_ids, b_ids, context)

    P = np.zeros((n, n))
    P[rows, cols] = probs
    P[cols, rows] = 1 - probs
    return P
```

**Implementation note for Story 6.5:** `batch_matchup_probabilities` for a stateless model (XGBoost) should call `feature_server.serve_matchup_features_batch(a_ids, b_ids, context)` — a **new method** Story 6.5 must add to `StatefulFeatureServer` — that returns a DataFrame with one row per (a, b) pair, then passes the whole matrix to `model.predict_proba()` in a single call.

For Elo (stateful): the batch call simply loops over `_predict_one()` pairs since Elo is already O(1) per pair — no vectorization needed.

For 64 teams: 64×63/2 = 2,016 probability computations. Batched XGBoost: ~50ms total (single inference). Elo: ~2ms total.

### 5.3 Scoring Rule Interface

**Recommendation: Plugin-registry compatible scoring rules.**

```python
class ScoringRule(Protocol):
    """Map round number (0-indexed) to points for a correct pick."""

    def points_per_round(self, round_idx: int) -> float: ...

    @property
    def name(self) -> str: ...
```

**Built-in rules:**

| Name | R1 | R2 | R3 | R4 | R5 | R6 | Total (perfect) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Standard (ESPN) | 1 | 2 | 4 | 8 | 16 | 32 | 192 |
| Fibonacci | 2 | 3 | 5 | 8 | 13 | 21 | 164 |
| Seed-Diff Bonus | 1+Δ | 2+Δ | 4+Δ | 8+Δ | 16+Δ | 32+Δ | Variable |

The Seed-Diff Bonus rule adds the absolute seed difference as a bonus when the lower seed wins (rewarding correct upset picks).

**Custom callable:** Allow `Callable[[int], float]` as a scoring rule for user-defined schedules.

### 5.4 Performance Targets

| Operation | Method | Target | Rationale |
|:---|:---|:---|:---|
| Advancement probabilities (64 teams) | Analytical | < 1 ms | Phylourny: 599μs in C++; Python/NumPy should be <1ms |
| Expected Points (64 teams, 1 scoring rule) | Analytical | < 1 ms | Arithmetic on advancement probs |
| EP confidence intervals (B=1000) | Two-layer bootstrap | < 5 s | 1000 × (param draw + prob matrix + Phylourny) |
| MC simulation (N=10,000) | MC | < 2 s | 10K bracket traversals, vectorizable |
| Full bracket analysis (prob matrix + analytical EP) | Analytical | < 3 s | Dominated by prob matrix construction for XGBoost |

**The PRD explicitly excludes MC simulation time from the 60-second backtest target.** These targets are internal performance goals for the simulation engine itself.

### 5.5 Integration with Existing Evaluation Pipeline

**Recommendation: Separate `SimulationResult` rather than extending `BacktestResult`.**

Rationale:
- `BacktestResult` contains fold-level metrics (log_loss, brier_score, etc.) from walk-forward CV.
- Tournament simulation produces fundamentally different outputs: per-team advancement probabilities, expected points, bracket-level scores.
- The two can be composed at a higher level (e.g., a "full evaluation" that runs both backtest + simulation), but their internal structures should remain distinct.

```python
@dataclass(frozen=True)
class SimulationResult:
    """Result of tournament simulation for one season.

    Both the analytical path and the MC path produce a SimulationResult.
    The difference is in method, n_simulations, and confidence_intervals.

    expected_points: computed externally for each desired ScoringRule, then
    passed in as a dict. compute_expected_points() returns a single np.ndarray
    per ScoringRule; the caller computes EP for all desired rules and assembles
    the dict before constructing SimulationResult.

    Example (analytical path):
        adv_probs = compute_advancement_probs(bracket, P)
        ep_dict = {
            rule.name: compute_expected_points(adv_probs, rule)
            for rule in scoring_rules
        }
        result = SimulationResult(
            season=context.season,
            advancement_probs=adv_probs,
            expected_points=ep_dict,
            method="analytical",
            n_simulations=None,
            confidence_intervals=None,
        )
    """

    season: int
    advancement_probs: np.ndarray    # shape (n_teams, n_rounds)
    expected_points: dict[str, np.ndarray]  # scoring_rule_name → per-team EP, shape (n_teams,)
    method: str                       # "analytical" or "monte_carlo"
    n_simulations: int | None         # None for analytical, N for MC
    confidence_intervals: dict[str, tuple[np.ndarray, np.ndarray]] | None
    # ^ scoring_rule_name → (ep_lower, ep_upper), each shape (n_teams,)
    # Populated only when compute_ep_confidence_intervals() is called
```

### 5.6 Data Requirements

**Tournament seeds:** Already ingested in Story 4.3 via `MNCAATourneySeeds.csv`. Available fields: `season`, `seed_str` (e.g., "W01"), parsed into `seed_num`, `region`, `is_play_in`.

**Bracket structure:** Hardcoded. The NCAA bracket matchup structure has been unchanged since 1985:
- Each region: 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
- Region winners meet in the Final Four (fixed pairing by bracket position)
- Final Four → Championship game

**First Four (play-in) handling:** The First Four (introduced 2011) determines the last 4 teams entering the 64-team bracket. The simulator should operate on the **post-First-Four 64-team field**. First Four results are treated as given inputs, not simulated.

---

## 6. Decision Matrix: Analytical vs. Monte Carlo

| Use Case | Analytical | MC | Recommended | Rationale |
|:---|:---:|:---:|:---|:---|
| Per-team advancement probabilities | ✅ | ❌ | Analytical | Exact, zero noise, 599μs |
| Expected Points (single bracket) | ✅ | ❌ | Analytical | EP = adv_prob · scoring_vector |
| EP confidence intervals | ✅ (inner) | ❌ | Two-layer bootstrap | Outer: param perturbation; inner: analytical |
| Model comparison (which model is better?) | ✅ | ❌ | Analytical | Exact EPs are directly comparable |
| Score-distribution analysis | ❌ | ✅ | MC (N≥10K) | Need to sample actual scores, not just probabilities |
| Bracket-count analysis | ❌ | ✅ | MC (N≥10K) | "How many brackets have Team X winning?" |
| Pool strategy optimization | ❌ | ✅ | MC (N≥50K) | Need joint distribution of all 63 game outcomes |

**Summary:** For ncaa_eval's primary use cases (model evaluation, EP computation, confidence intervals), **analytical computation is strictly superior**. MC simulation is only needed for secondary use cases involving score distributions or bracket-count analysis.

---

## 7. Pseudocode: Recommended Simulation Approach

### 7.1 Analytical Expected Points Computation

```python
def compute_advancement_probs(
    bracket: BracketStructure,
    P: np.ndarray,  # n×n pairwise probability matrix
) -> np.ndarray:
    """Compute exact advancement probabilities via Phylourny algorithm.

    Uses post-order traversal of the bracket tree, computing
    Win Probability Vectors (WPVs) at each internal node.

    round_index convention (0-indexed from first playable round):
      0 = Round of 64  (R64, 32 games)
      1 = Round of 32  (R32, 16 games)
      2 = Sweet 16     (S16,  8 games)
      3 = Elite Eight  (E8,   4 games)
      4 = Final Four   (FF,   2 games)
      5 = Championship (NCG,  1 game)

    Each internal BracketNode stores its round_index. In a perfect binary
    tree with n=64 leaves, the accumulation adv_probs[:, r] += wpv is safe
    because each round has exactly one game slot per bracket half (disjoint
    subtrees). A team appears in at most one internal node per round.

    Args:
        bracket: Tournament bracket structure (matchup tree).
            Must have exactly n leaf nodes where n is a power of 2.
            n=64 for NCAA post-First-Four bracket.
        P: Pairwise win probability matrix, shape (n, n).

    Returns:
        adv_probs, shape (n, n_rounds).
        adv_probs[i, r] = P(team i advances past round r).
    """
    n = P.shape[0]
    assert n > 0 and (n & (n - 1)) == 0, f"n must be a power of 2, got {n}"
    n_rounds = int(np.log2(n))  # 6 for 64 teams

    # Store per-round advancement probs
    adv_probs = np.zeros((n, n_rounds))

    def traverse(node: BracketNode) -> np.ndarray:
        """Post-order traversal returning WPV at this node.

        The WPV at an internal node is the probability vector over all n
        teams of being the team that wins this particular match slot.
        For a leaf, it is the unit vector for that team.
        """
        if node.is_leaf:
            wpv = np.zeros(n)
            wpv[node.team_index] = 1.0
            return wpv

        left_wpv = traverse(node.left)
        right_wpv = traverse(node.right)

        # Phylourny core: R = V ⊙ (P^T · W) + W ⊙ (P^T · V)
        # R[i] = P(team i wins this match) given the left/right sub-brackets
        wpv = left_wpv * (P.T @ right_wpv) + right_wpv * (P.T @ left_wpv)

        # Accumulate advancement probs: safe because each round writes to
        # disjoint game slots (binary tree, no overlapping subtrees).
        adv_probs[:, node.round_index] += wpv
        return wpv

    traverse(bracket.root)
    return adv_probs


def compute_expected_points(
    adv_probs: np.ndarray,
    scoring_rule: ScoringRule,
) -> np.ndarray:
    """Compute Expected Points per team.

    Args:
        adv_probs: Advancement probabilities, shape (n, n_rounds).
        scoring_rule: Maps round index to points.

    Returns:
        Expected Points per team, shape (n,).
    """
    n, n_rounds = adv_probs.shape
    points = np.array([scoring_rule.points_per_round(r) for r in range(n_rounds)])
    return adv_probs @ points
```

### 7.2 Two-Layer Bootstrap for EP Confidence Intervals

```python
def compute_ep_confidence_intervals(
    bracket: BracketStructure,
    provider_factory: Callable[[], ProbabilityProvider],
    team_ids: Sequence[int],
    context: MatchupContext,
    scoring_rule: ScoringRule,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute EP confidence intervals via two-layer bootstrap.

    Outer loop: draw a perturbed ProbabilityProvider (captures model
    uncertainty). Inner loop: analytical EP computation (zero simulation
    noise).

    Args:
        bracket: Tournament bracket structure.
        provider_factory: Callable that returns a new ProbabilityProvider
            with independently perturbed model parameters on each call.
            Story 6.5 must implement this for each model type:
            - Elo (stateful): resample rating perturbation noise from the
              Elo model's estimated rating uncertainty.
            - XGBoost (stateless): refit on a bootstrap resample of the
              training data (parametric bootstrap).
            - Bayesian models: draw from the posterior directly.
        team_ids: Team IDs in bracket order.
        context: Matchup context (season, day_num, neutral court).
        scoring_rule: Scoring rule for EP computation.
        n_bootstrap: Number of outer-loop draws (500–1000).
        alpha: Significance level (default 0.05 for 95% CI).

    Returns:
        Tuple of (ep_median, ep_lower, ep_upper), each shape (n,).
    """
    n = len(team_ids)
    ep_samples = np.zeros((n_bootstrap, n))

    for b in range(n_bootstrap):
        # Outer loop: obtain a perturbed probability provider
        # (does NOT call model.sample_from_posterior() — that method
        #  does not exist on the Model ABC; provider_factory encapsulates
        #  the perturbation strategy for each model type)
        perturbed_provider = provider_factory()

        # Build probability matrix with perturbed params
        P = build_probability_matrix(perturbed_provider, team_ids, context)

        # Inner loop: exact analytical computation (zero simulation noise)
        adv_probs = compute_advancement_probs(bracket, P)
        ep_samples[b] = compute_expected_points(adv_probs, scoring_rule)

    ep_median = np.median(ep_samples, axis=0)
    ep_lower = np.percentile(ep_samples, 100 * alpha / 2, axis=0)
    ep_upper = np.percentile(ep_samples, 100 * (1 - alpha / 2), axis=0)

    return ep_median, ep_lower, ep_upper
```

**Implementation note for Story 6.5:** The `provider_factory` pattern decouples CI computation from model internals. Story 6.5 must implement a `BootstrapProviderFactory` for each model type:
- **Elo:** Store a snapshot of ratings; add Gaussian noise scaled to the Elo model's expected rating uncertainty (±50–100 Elo points).
- **XGBoost:** Refit the XGBoost model on a resampled training dataset (expensive — use B=50–100 not 1000); or use XGBoost's internal `predict_leaf` for parametric variance estimation.
- **Bayesian (PyMC/NumPyro):** Draw from the posterior sample cache directly — the cheapest outer loop.

The `Model` ABC itself does **not** need a `sample_from_posterior()` method. The `provider_factory` callable captures the perturbation logic externally.

### 7.3 Monte Carlo Simulation (Fallback)

```python
def simulate_tournament_mc(
    bracket: BracketStructure,
    P: np.ndarray,
    scoring_rule: ScoringRule,
    season: int,  # Required: populates SimulationResult.season
    n_simulations: int = 10_000,
    rng: np.random.Generator | None = None,
) -> SimulationResult:
    """Monte Carlo tournament simulation.

    Used only when score distributions or bracket counts are needed.
    For advancement probabilities and EP, use analytical computation.

    Args:
        bracket: Tournament bracket structure.
        P: Pairwise win probability matrix, shape (n, n).
        scoring_rule: Scoring rule for EP computation.
        season: Tournament season year (e.g. 2024). Stored in result.
        n_simulations: Number of MC runs (≥10K recommended).
        rng: NumPy random generator for reproducibility.

    Returns:
        SimulationResult with per-team advancement counts and scores.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = P.shape[0]
    assert n > 0 and (n & (n - 1)) == 0, f"n must be a power of 2, got {n}"
    n_rounds = int(np.log2(n))

    # Batch-sample all game outcomes at once for vectorization
    # Pre-generate uniform random numbers: shape (n_simulations, n_games)
    n_games = n - 1  # 63 for 64 teams
    randoms = rng.random((n_simulations, n_games))

    # Simulate brackets (vectorized across simulations)
    advancement_counts = np.zeros((n, n_rounds), dtype=int)
    scores = np.zeros((n_simulations, n))

    for sim in range(n_simulations):
        # Traverse bracket, using randoms[sim] for game outcomes
        # ... (implementation details in Story 6.5)
        pass

    return SimulationResult(
        season=season,
        advancement_probs=advancement_counts / n_simulations,
        expected_points={scoring_rule.name: scores.mean(axis=0)},
        method="monte_carlo",
        n_simulations=n_simulations,
        confidence_intervals=None,
    )
```

---

## 8. Key Recommendations Summary

1. **Primary method: Analytical computation** via the Phylourny algorithm (post-order bracket traversal with Win Probability Vectors). This gives exact per-team advancement probabilities in O(n²) time (~599μs for 64 teams) with zero simulation noise.

2. **Confidence intervals: Two-layer bootstrap** — perturb model parameters (B=500–1,000) in the outer loop, compute analytical EP in the inner loop. This cleanly separates model uncertainty from simulation noise and takes <5 seconds.

3. **MC simulation as fallback only** — for score-distribution and bracket-count analysis (N≥10,000 for meaningful results, N≥50,000 for production).

4. **No PyPI package to wrap** — the Phylourny algorithm must be reimplemented in Python/NumPy. The core is ~30 lines of vectorized code (the WPV recursion formula). This is a justified custom implementation per the Library-First Rule.

5. **Bracket representation: NumPy arrays** with index arithmetic for parent/child relationships. Bracket structure is hardcoded (unchanged since 1985).

6. **Probability contract: `ProbabilityProvider` protocol** — works for both stateful (Elo) and stateless (XGBoost) models via a unified `matchup_probability()` interface.

7. **Scoring rules: Plugin-registry pattern** — `ScoringRule` protocol with built-in Standard/Fibonacci/Seed-Diff-Bonus implementations plus custom callable support.

8. **Separate `SimulationResult`** data model rather than extending `BacktestResult` — the outputs are fundamentally different in structure and semantics.

---

## References

### Academic Papers

- Bettisworth, B., Jordan, A. I., & Stamatakis, A. (2023). Phylourny: efficiently calculating elimination tournament win probabilities via phylogenetic methods. *Statistics and Computing*, 33(4), 80. [PMC10186292]
- Brandes, U., Marmulla, G., & Smokovic, I. (2025). Efficient computation of tournament winning probabilities. *Journal of Sports Analytics*, 11. [arXiv: 2307.10411]
- Johnstone, C., & Nettleton, D. (2023). Using conformal win probability to predict the winners of the canceled 2020 NCAA basketball tournaments. *The American Statistician*, 78(3).
- Jacobson, S., Nikolaev, A., King, D., & Lee, A. (2011). Seed distributions for the NCAA men's basketball tournament. *Omega*, 39(2), 116–125.
- Nelson, J. (2012). Modeling the NCAA tournament through Bayesian logistic regression. M.S. thesis, Duquesne University.
- Efron, B. (1987). Better bootstrap confidence intervals. *Journal of the American Statistical Association*, 82(397), 171–185.

### Industry / Competition

- FiveThirtyEight. How Our March Madness Predictions Work. [methodology page]
- Lopez, M. & Matthews, G. (2014). Building an NCAA men's basketball predictive model and quantifying its success. [arXiv: 1412.0248]
- Landgraf, A. (2017). Kaggle March Machine Learning Mania 1st Place Winner Interview. [Kaggle blog]
- BracketOdds. University of Illinois. [bracketodds.cs.illinois.edu]

### Existing Codebase

- `src/ncaa_eval/evaluation/metrics.py` — log_loss, brier_score, roc_auc, expected_calibration_error
- `src/ncaa_eval/evaluation/splitter.py` — walk_forward_splits, CVFold
- `src/ncaa_eval/evaluation/backtest.py` — run_backtest, FoldResult, BacktestResult
- `src/ncaa_eval/model/base.py` — Model ABC, StatefulModel._predict_one()
- `src/ncaa_eval/ingest/schema.py` — Game Pydantic model (is_tournament flag)
- `src/ncaa_eval/transform/feature_serving.py` — StatefulFeatureServer
