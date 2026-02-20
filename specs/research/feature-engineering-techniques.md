# Feature Engineering Techniques — NCAA Tournament Prediction

**Date:** 2026-02-20
**Story:** 4.1 (Spike) — Research Feature Engineering Techniques
**Status:** Complete — Pending Product Owner Approval (AC 8)
**Author:** Dev Agent (Claude Sonnet 4.6)

---

## Quick-Navigation Table

| Section | Topic | Key Recommendation |
|:---|:---|:---|
| [1. Data Context](#1-data-context) | What EDA confirmed; open questions | Pre-read before implementation |
| [2. Opponent Adjustment](#2-opponent-adjustment-techniques) | SRS, Ridge, Massey, Colley | Ridge > SRS for sparse schedules; Massey for full-season final ratings |
| [3. Sequential / Momentum](#3-sequential--momentum-features) | Rolling windows, EWMA, streaks | EWMA alpha 0.15–0.20; validate 5/10/20-game windows empirically |
| [4. Graph Features](#4-graph-based-features) | PageRank, betweenness, HITS | PageRank validated; betweenness/HITS marginal |
| [5. Massey Ordinals](#5-massey-ordinal-systems) | 100+ ranking systems | SAG + POM + MOR + WLK average; 23-season coverage |
| [6. Community Techniques](#6-kaggle-mmlm-community-techniques) | Kaggle MMLM top solutions 2014–2025 | Same raddar/538 signal dominated 2018–2023; Elo + ordinals + seed diff dominate from-scratch solutions |
| [7. Prioritized Plan](#7-prioritized-implementation-plan) | Mapping to Stories 4.2–4.7 | See MVP vs. Post-MVP recommendations |

---

## 1. Data Context

### What EDA Already Confirmed (Do Not Reopen)

All values below are from `notebooks/eda/statistical_exploration_findings.md` and `eda_findings_synthesis.md`. Epic 4 implementations must cite these references — do not recompute.

**Top correlation signals with tournament advancement** (Pearson r, 2003–2024, n=196,716 deduplicated games):

| Feature | r | Notes |
|:---|:---|:---|
| Strength of Schedule (SoS) | **+0.2970** | Highest single-stat r; mean opponent regular-season win rate |
| FGM | +0.2628 | Field goals made per game |
| Score | +0.2349 | Points scored per game |
| FGPct | +0.2269 | Field goal percentage; tournament differential: +0.078 |
| PF | –0.1574 | Personal fouls (negative predictor) |
| TO_rate | –0.1424 | `TO / (FGA + 0.44×FTA + TO)` |

**Normalization — baseline established, further investigation open** (`statistical_exploration_findings.md` Section 7):

The EDA established a working baseline using standard transforms (logit, sqrt, none) paired with StandardScaler or RobustScaler. These are a reasonable starting point but were not exhaustively compared against advanced distribution families (Beta, Gamma, Weibull, Log-Normal). Story 4.7 should treat these as the default, not the final answer.

| Stat Group | Baseline Transform | Baseline Scaler | Notes |
|:---|:---|:---|:---|
| Bounded rates (FGPct, 3PPct, FTPct, TO_rate) | logit | StandardScaler | Beta distribution fit may be superior — validate in Story 4.7 |
| Mildly right-skewed counts (Blk, Stl, TO, FTM, OR, FGM3, FTA) | sqrt | RobustScaler | Gamma/Weibull may fit better than sqrt-Normal assumption |
| High-volume counting (Score, FGM, FGA, DR, Ast, FGA3, PF) | none | StandardScaler | Likely near-Normal at scale; validate with Q-Q plots |

New features not in this table (graph centrality, Elo ratings, Massey ordinals) require separate normalization assessment — addressed in their respective sections below.

**Baselines any new technique must beat** to justify its implementation complexity:

| Baseline | r | Source |
|:---|:---|:---|
| Raw SoS (opponent win rate) | 0.2970 | `statistical_exploration_findings.md` Section 4 |
| FGM | 0.2628 | Section 5 |
| FGPct | 0.2269 | Section 5 |
| Score | 0.2349 | Section 5 |

### Open Questions (Priority Research Questions for This Spike)

| Open Question | Priority | Resolution in This Document |
|:---|:---|:---|
| Do graph centrality metrics add signal beyond naive SoS (r=0.2970)? | **#1** | See Section 4 — literature suggests yes for PageRank; betweenness/HITS marginal |
| Which Massey Ordinal systems best complement box-score features? | **#2** | See Section 5 — SAG + POM + MOR + WLK composite is community-validated |
| What rolling window size is optimal for sequential features? | **#3** | See Section 3 — no strong evidence; recommend empirical validation of 5/10/20 |

### Data Coverage Caveats

- **Box-score features (FGM, FGPct, etc.) available 2003–2025 only.** Pre-2003 sequential features are limited to compact stats (score, margin, loc).
- **Tournament detailed results available through 2024 only.** 2025 tournament box scores unavailable until tournament completion.
- **2025 deduplication required.** 4,545 games stored twice (Kaggle + ESPN IDs). Deduplicate by `(w_team_id, l_team_id, day_num)`; prefer ESPN records for `loc` and `num_ot`.
- **2020 COVID: no tournament.** Include 2020 regular season in training; exclude from evaluation.

---

## 2. Opponent Adjustment Techniques

**Story 4.6 scope.** Validation baseline: opponent-adjusted efficiency must exceed raw SoS (r=0.2970) to justify implementation.

### 2.1 Simple Rating System (SRS)

**Description:** Iterative least-squares system that simultaneously solves for all team ratings such that each team's rating equals its average point differential plus its average opponent rating. Used by Sports Reference and closely related to KenPom's AdjEM.

**Matrix formulation:** Solve `Gᵀ G r = Gᵀ S` where `G` is the team-game indicator matrix (+1 home, −1 away per game row), `r` is the ratings vector, and `S` is the margin-of-victory vector. Fixed-point iteration: `r_i(k+1) = avg_margin_i + avg(r_j for all opponents j)`. Convergence is guaranteed for connected schedules (spectral radius < 1); approximately 3,000–5,000 iterations to full convergence.

**KenPom AdjEM relationship:** KenPom solves offense and defense separately on a per-100-possession basis. The iterative formula is `AdjO_A = RawO_A × (national_avg_D / AdjD_opponent)`. Conceptually identical to SRS but normalized by possessions.

**Data requirements:** Box-score results 2003+. This project always computes ratings on full-season data (pre-tournament snapshot), so convergence is well-conditioned with 30+ games per team.

**Computational complexity:** O(k × |E|) iterative, or O(n³) direct solve. For NCAA (n=350 teams, |E|=5,800 games/season): trivial.

**Expected predictive value:** SRS (full-season) comparable to SoS baseline; opponent-adjusted version should exceed r=0.2970 baseline. KenPom AdjEM historically achieves ~75%+ accuracy on regular-season game outcomes, substantially above raw SoS.

| Factor | Assessment |
|:---|:---|
| Feasibility | ✅ High — well-documented linear algebra; no new dependencies |
| Complexity | Medium — iterative convergence; numerical stability required |
| Expected value | High — KenPom-equivalent metric; should exceed SoS baseline |

---

### 2.2 Ridge Regression / Penalized Least Squares

**Description:** Frames the team-rating problem as regularized regression: minimize `‖Gr − S‖² + λ‖r‖²`. The regularized normal equations `(GᵀG + λI) r = GᵀS` are always full-rank and invertible, shrinking team ratings toward the league mean (zero).

**Relevance for full-season ratings:** Since this project always uses full-season data, regularization's primary benefit (stabilizing sparse early-season ratings) is not a concern. Ridge still provides a benefit for teams in structurally isolated conference subgraphs — where all intra-conference games form a near-singular sub-block — but this is a minor edge case with 30+ games per team.

**Lambda selection:**
- Cross-validation on prior seasons: minimize log loss on game outcomes across the prior 3–5 seasons.
- Typical range: λ = 10–100 (in points-squared units for margin data).
- With full-season data, small λ (10–20) is appropriate — heavy shrinkage is unnecessary.

**Lasso (L1):** Produces sparse solutions; less useful for ratings (all teams should have non-zero ratings). Elastic net (L1 + L2) may help if the schedule graph has near-collinear subgraphs (isolated conferences playing only among themselves).

**Data requirements:** Same as SRS — box-score results 2003+ for per-possession normalization; compact results available 1985+.

**Implementation:** `scipy.linalg.solve` or `sklearn.linear_model.Ridge`. Both are readily available.

| Factor | Assessment |
|:---|:---|
| Feasibility | ✅ High — scikit-learn Ridge is trivial |
| Complexity | Low-Medium — single matrix solve; lambda tuning adds one CV loop |
| Expected value | High — equivalent to SRS at full season; minor stability benefit for isolated conferences |

---

### 2.3 Massey's Rating Method

**Description:** Least-squares rating system solving `Mr = p` where M is the n×n Massey matrix (diagonal = games played; off-diagonal = −games between team pair) and `p` is the vector of cumulative point differentials. Standard fix for singularity: replace last row with all-ones and set `p[n] = 0`, enforcing `Σr = 0`.

**Relationship to SRS:** Algebraically equivalent to SRS for the same objective function; difference is the matrix formulation (Massey solves the full n×n system directly; SRS uses the game-team indicator matrix and iterates). Both minimize sum of squared prediction errors on game margins.

**Handling ties / OT:** Overtime games contribute to the diagonal (games played) and to `p` (full final margin, including OT). The standard Massey formulation naturally handles OT — no special case required.

**Available via MMasseyOrdinals.csv:** The "MAS" system in the Kaggle dataset is Massey's own system. Full historical ordinals are available for 2003–2024, providing an alternative to re-implementing the solver.

| Factor | Assessment |
|:---|:---|
| Feasibility | ✅ High — n×n matrix solve; O(n³) but trivial for n=350 |
| Complexity | Low — equivalent to SRS in a different formulation |
| Expected value | Same as SRS; available pre-computed via MMasseyOrdinals |

---

### 2.4 Colley Matrix Method

**Description:** Win/loss-only rating system using a Bayesian interpretation. The Colley rating for team i starts at 0.5 (prior) and is updated by: `r_i = (1 + w_i) / (2 + t_i)` with opponents' win rates replaced by full Colley ratings. Solved via the Colley matrix `C` (symmetric positive definite): `C[i,i] = 2 + t_i`, `C[i,j] = −n_ij` (games between i and j), right-hand side `b[i] = 1 + (w_i − l_i)/2`.

**Key difference from Massey/SRS:** Colley ignores margin of victory entirely. This is both its strength (no score-running incentive) and its weakness (less predictive for tournament outcomes where efficiency matters).

**Available via MMasseyOrdinals.csv:** The "COL" system is the Colley Matrix. Pre-computed ordinals available for 2003–2024.

**Predictive value relative to SRS:** Colley is strictly less predictive than SRS/Massey because it discards margin information. Expected predictive value below the SoS baseline (r=0.2970) for tournament advancement.

| Factor | Assessment |
|:---|:---|
| Feasibility | ✅ High — Cholesky solve via scipy |
| Complexity | Low — same as Massey |
| Expected value | Lower than SRS/Massey — no margin signal |

**Recommendation:** Prefer as a feature from MMasseyOrdinals (pre-computed) rather than re-implementing. Do not implement as a standalone solver for Story 4.6.

---

### 2.5 Feasibility Summary and Recommendation for Story 4.6

**Context:** This project always computes ratings on full-season data (pre-tournament snapshot, 30+ games per team). Sparse-schedule considerations are not a factor — all methods are well-conditioned at full season.

| Method | Full-Season Accuracy | Implementation Cost | Margin Used | Recommendation |
|:---|:---|:---|:---|:---|
| SRS / Massey iterative | High | Low — no tuning required | Yes | **Recommended for MVP** — simplest to implement; deterministic; no hyperparameter |
| Massey direct solve | Same as SRS | Low — Cholesky; no tuning | Yes | **Recommended for MVP** — equivalent to SRS; pre-computed via MMasseyOrdinals as fallback |
| Ridge regression | Equivalent to SRS | Medium — lambda tuning via CV | Yes | Secondary — adds lambda CV complexity with no meaningful accuracy gain at full season |
| Colley Matrix | Lower | Low | No | **Not Recommended** — discards margin signal |
| Elo (dynamic) | Comparable | Medium — K-factor design | Via K-factor | Best for *in-season* tracking; see Section 6 |

**Revised recommendation for Story 4.6:** Implement SRS (iterative solve) as the primary opponent-adjustment method — it is the simplest, requires no hyperparameter tuning, and is well-conditioned on full-season data. Massey (direct Cholesky solve) is an equivalent alternative if a direct solver is preferred. Ridge adds lambda tuning complexity without meaningful accuracy benefit given full-season data.

**Validation gate for Story 4.6:** Any opponent-adjustment implementation must be validated to exceed the SoS baseline (r=0.2970). If opponent-adjusted efficiency does not exceed this baseline, the adjustment adds no value over the raw SoS feature already planned for Story 4.4.

**Normalization for adjusted efficiency features:** Adjusted efficiency metrics (AdjO, AdjD, AdjEM equivalents) are approximately normally distributed (similar to Score/FGM). Recommend: no transform + StandardScaler. Validate distribution before committing.

---

## 3. Sequential / Momentum Features

**Story 4.4 scope.** Normalization for box-score rolling statistics is already resolved (Section 1). This section covers window size selection, EWMA parameterization, streak encoding, and momentum/trajectory features.

### 3.1 Rolling Window Sizes

**Candidates:** 5, 10, 20 games (from story scope). No strong empirical consensus exists in the academic literature for NCAA basketball specifically. The EDA found no strong evidence for any particular window size.

**What is known:**
- **5-game windows:** Highly volatile, sensitive to single-game anomalies. Used in betting models (short-term form). For tournament prediction (a March snapshot of season-long performance), 5-game windows may over-fit to recent noise.
- **10-game windows:** Balance between recency and stability. Roughly corresponds to the final month of regular-season play (~30-35 game seasons; 10 games ≈ last 25–30% of season). Appears in documented Kaggle MMLM solutions as a "late-season form" feature.
- **20-game windows:** Covers roughly 55–65% of a college basketball season. Approaches a "second half of season" metric. Less sensitive to single-game volatility.
- **Full-season:** KenPom and SRS compute full-season aggregates but weight recent games more heavily in their iterative solvers (implicitly encoding recency).

**Academic reference:** A 2025 deep learning paper on NCAA basketball (Habib, M.I., "Forecasting NCAA Basketball Outcomes with Deep Learning: A Comparative Study of LSTM and Transformer Models," arXiv:2508.02725) explicitly noted that rolling windows were not used and that "static features lack patterns such as recent form, late-season momentum." No paper in the literature provided a rigorous optimal-window-size comparison for NCAA tournament prediction specifically.

**Recommendation for Story 4.4:** Implement all three window sizes (5, 10, 20) as parallel feature columns. Empirically validate via correlation with tournament advancement and XGBoost feature importance during Story 5.4 model development. Include full-season aggregate as a fourth reference column.

**Implementation note:** For walk-forward compatibility (Story 4.7), rolling windows must respect temporal boundaries — a game on day N can only use games from days < N.

---

### 3.2 Exponential Weighted Moving Average (EWMA)

**Formula:** `value_t = α × observation_t + (1 − α) × value_{t-1}`

**Equivalent window sizes by alpha:**

| α | Equivalent N (games) | Half-life (games) | Notes |
|:---|:---|:---|:---|
| 0.10 | ~19 | 6.6 | Heavy smoothing; ~full second-half of season |
| 0.15 | ~12.3 | 4.3 | Moderate smoothing; recommended starting point |
| 0.20 | ~9 | 3.1 | Standard "fast" decay |
| 0.30 | ~5.7 | 1.9 | Aggressive; nearly equivalent to 5-game rolling |

**Alpha selection guidance:** Sports analytics literature (NBA Elo systems, KenPom) generally uses effective windows in the 10–20 game range for season-long efficiency tracking. For college basketball (shorter seasons, larger variance, significant roster turnover):
- Recommended starting range: α = 0.15–0.20 (effective window 9–12 games)
- α = 0.15 maps to the equivalent of a 12-game uniform rolling window — a reasonable trade-off between responsiveness and stability
- Tune via cross-validated log loss across multiple prior seasons

**BartTorvik recency weighting (reference):** BartTorvik applies explicit time-decay: games > 40 days old lose 1% emphasis per day, reaching 60% weight at 80+ days. This is equivalent to a piecewise linear decay rather than exponential. Could be adapted as an alternative to pure EWMA.

**Implementation note:** EWMA is more memory-efficient than rolling windows (no stored window of observations required) and handles irregular game spacing (gaps from schedule breaks) more gracefully. Recommend `pandas.DataFrame.ewm(alpha=α).mean()` per team per season.

---

### 3.3 Streak Features

**Encoding options:**
1. **Signed integer:** `current_streak = +N` for winning streak of N, `−N` for losing streak of N. Zero-indexed (0 = just broke a streak).
2. **Separate columns:** `win_streak_len`, `loss_streak_len` (one is always 0 given the other is active).
3. **Binary flag:** `entered_tournament_on_5plus_win_streak` (1/0).

**Statistical evidence:** Mixed. A study of elite basketball found winning teams had significantly higher "momentum occurrences" than losing teams (p < 0.001), measured as within-game scoring runs. However, pre-tournament win/loss streaks are a coarser measure. The distribution of winning/losing streaks in sports often arises from random statistical fluctuations — simple streak counts may have limited predictive power over and above underlying efficiency metrics.

**Recommendation:** Include as a low-complexity feature (signed integer encoding). Validate feature importance during Story 5.4 — if importance is near zero, drop. Frame as "Requires Validation" in the prioritized plan.

---

### 3.4 Performance Trajectory (Momentum)

**Definition:** Rate of change of a rolling efficiency metric, capturing whether a team is improving or declining as the tournament approaches.

**Candidate formulations:**
- `momentum_N = ewma(adj_eff, α=0.20) − ewma(adj_eff, α=0.10)` — positive values indicate improving recent form over medium-term trend
- `trajectory = slope of OLS regression over last-N games' margins`
- `form_ratio = last_10_adj_eff / full_season_adj_eff` — ratio > 1 indicates better recent than average performance

**Practical note:** KenPom weights recent games more heavily in its iterative efficiency solve, implicitly encoding momentum. Teams with strong recent form will have slightly elevated ratings compared to a pure full-season average. Sagarin publishes a "Recent" sub-rating that is an explicit implementation of this concept.

**Recommendation:** Include `ewma_fast − ewma_slow` (momentum) as a feature. Low implementation cost once EWMA is built. Validate during Story 5.4.

---

### 3.5 Home Court Advantage Encoding

**From EDA** (`eda_findings_synthesis.md` Section 2, Tier 2 Feature #8):
- Home team wins 65.8% of non-neutral regular-season games
- Home margin advantage: +2.2 pts over neutral sites
- Declining trend: slope = −0.00048/season (p=0.0006)

**Encoding:** `loc` should be encoded as a numeric feature: H=+1, A=−1, N=0. For tree-based models (XGBoost), one-hot encoding (is_home, is_away, is_neutral) is equivalent. The declining trend suggests this should be treated as a time-varying feature (include `season` interaction or rolling home-advantage estimate).

**Normalization:** Categorical-to-numeric mapping; no distribution transform required.

---

## 4. Graph-Based Features

**Story 4.5 scope.** NetworkX is already in the tech stack (`specs/05-architecture-fullstack.md` Section 3) — no new dependency required. Validation baseline: graph centrality features must exceed SoS (r=0.2970) baseline to justify the additional algorithmic complexity.

### 4.1 PageRank on Win/Loss Directed Graph

**Graph construction:**
- **Nodes:** Teams (one per team per season)
- **Edges:** Directed W→L (loser "votes for" winner's quality, following the web PageRank metaphor where L links to W)
- **Edge weight:** Margin of victory (cap at 25–30 points to prevent extreme-blowout distortion; use `min(margin, 25)`)
- **Date weighting (optional):** Multiply weight by a recency factor (e.g., games in last 20 days get 1.5× weight) to emphasize tournament run-up form

**What PageRank captures:** Transitivity of strength. If Team A beat Team B who beat Team C who beat many strong teams, Team A receives credit for the full chain. This is the primary advantage over naive SoS (which only looks one level deep at opponent win rates).

**Validated performance on NCAA basketball:** A 2021 study in the *Journal of Sports Analytics* (Matthews et al., "PageRank as a Method for Ranking NCAA Division I Men's Basketball Teams") applied PageRank to 2014–2018 NCAA D1 basketball:
- Standard PageRank: 65.7% predictive accuracy on tournament game outcomes
- Modified PageRank (with margin and date weighting): 71.6% accuracy
- Win ratio (naive): 64.2%
- PageRank **outperformed naive SoS** across all five seasons tested

**Expected r correlation:** Based on the study results, PageRank (modified) is expected to achieve accuracy of ~71%, which implies a correlation substantially above the SoS r=0.2970 baseline. However, direct r-value comparison requires local implementation — the study used accuracy % not Pearson r.

**Implementation via NetworkX:**
```python
import networkx as nx

# Vectorized edge construction — do NOT use iterrows (project mandate)
games_df = games_df.assign(weight=games_df["w_score"].sub(games_df["l_score"]).clip(upper=25))
G = nx.from_pandas_edgelist(
    games_df,
    source="l_team_id",
    target="w_team_id",
    edge_attr="weight",
    create_using=nx.DiGraph,
)

pr = nx.pagerank(G, alpha=0.85, weight="weight")
```
> ⚠️ **Story 4.5 implementation note:** Use `nx.from_pandas_edgelist()` as shown above. Never use `iterrows()` — the project's no-iterrows mandate applies to all source code in `src/ncaa_eval/`.

**Normalization for PageRank scores:** PageRank values are non-negative and approximately power-law distributed (few teams have very high scores). Recommend: log transform + StandardScaler. Validate distribution empirically.

| Factor | Assessment |
|:---|:---|
| Feasibility | ✅ High — NetworkX already in stack; 10 lines of code |
| Complexity | Low-Medium — O(k × \|E\|) per iteration; fast for n=350 |
| Expected value | **High** — validated literature shows improvement over SoS |
| Walk-forward compatibility | ⚠️ Requires incremental update strategy (see Section 4.4) |

---

### 4.2 Betweenness Centrality

**Definition:** For team node v, betweenness centrality = sum over all team pairs (s,t) of `(shortest paths through v) / (total shortest paths s→t)`. Measures how often a team lies on the shortest path between any other pair of teams in the schedule network.

**Sports interpretation:** A team with high betweenness is a "bridge" between otherwise separate competitive clusters (e.g., a mid-major that beat a top-10 team and was beaten by a bottom-25 team, bridging two otherwise unconnected conference bubbles).

**Computational complexity:** Brandes algorithm: O(V×E) for unweighted, O(V×E + V² log V) for weighted. For NCAA (V=350, E=5,800 games), this runs in milliseconds.

**Validated performance:** No peer-reviewed study found validating betweenness centrality specifically for NCAA tournament prediction. The metric captures structural position rather than raw strength — a useful supplementary measure of schedule connectivity but unlikely to independently exceed the SoS baseline. Most useful as a complementary feature alongside PageRank.

**Normalization:** Betweenness values are bounded [0,1] but with extreme right skew (most teams near 0, few "bridge" teams much higher). Recommend: sqrt transform + RobustScaler, similar to Blk/Stl stats.

| Factor | Assessment |
|:---|:---|
| Feasibility | ✅ High — `nx.betweenness_centrality()` |
| Complexity | Low — single function call |
| Expected value | **Low** — no NCAA validation; structural rather than strength metric |
| Walk-forward compatibility | ⚠️ Must recompute each time step; warm-start not possible for betweenness |

---

### 4.3 HITS (Hub/Authority) Algorithm

**Mechanics:** Computes two scores per node:
- **Authority score:** High if many high-hub teams have lost to you (you beat teams that beat many others)
- **Hub score:** High if you have lost to many high-authority teams (you lost to good teams)

**Sports interpretation:** Authority ≈ "best team" (won against many quality opponents). Hub ≈ "difficult schedule" (played many good opponents even with losses).

**Correlation with PageRank:** HITS authority scores correlate ~0.908 with PageRank scores for typical sports networks. The incremental signal over PageRank is minimal. Hub scores capture "quality schedule despite losses" — a distinct signal partially covered by Colley/Massey SoS.

**Assessment:** HITS authority adds marginal incremental value over PageRank. The hub score is a somewhat novel "schedule connectivity" metric. No peer-reviewed paper found validating HITS specifically for NCAA basketball. Given the high correlation with PageRank and absence of NCAA validation, HITS is low priority.

| Factor | Assessment |
|:---|:---|
| Feasibility | ✅ High — `nx.hits()` |
| Complexity | Low |
| Expected value | **Low** — 0.908 correlation with PageRank; marginal incremental value |
| Walk-forward compatibility | Same issues as PageRank; warm-start possible |

---

### 4.4 Incremental Graph Update Strategy

**Problem:** Walk-forward feature serving (Story 4.7) requires computing graph features as of each game date without future data leakage. Naive approach: rebuild the full season graph from scratch after each game. For a 35-game season with 350 teams = 35 × O(k × 5,800) full recomputes — expensive at scale.

**Solution — Power Iteration Warm Start:**
Adding one edge to a 5,800-edge graph changes the PageRank solution by < 1% for most teams. Initialize the new iteration with the previous solution vector; convergence requires 2–5 iterations instead of 30–50.

```python
# Maintain sparse adjacency matrix incrementally
# After adding edge (l_team_id, w_team_id, weight):
A[l_team_id, w_team_id] += weight  # update sparse CSR matrix
# Re-run PageRank with warm start:
r_new = pagerank_power_iteration(A, r_init=r_prev, tol=1e-6, max_iter=5)
```

**Alternative — scipy sparse directly:** The `fast-pagerank` package (separate PyPI package) implements power iteration on scipy sparse matrices, avoiding NetworkX's format-conversion overhead. If performance becomes an issue, `fast-pagerank` is the recommended upgrade path.

**Betweenness centrality:** Does not have a warm-start strategy — full recomputation required after each edge addition. For 35-game seasonal schedules and 350 teams, a single betweenness computation takes ~50–100ms (acceptable). For walk-forward over 40 seasons × 5,000+ regular-season games, pre-computing and storing historical betweenness values per game date is the most practical approach.

---

### 4.5 Clustering Coefficient

**Definition:** Fraction of a team's network neighbors (teams connected by a win or loss) that also played each other. High clustering = tight local competitive cluster (single-conference team); low clustering = schedule spanning many separate competitive regions.

**Predictive value:** Functions as a "schedule diversity" metric. Teams with low clustering (non-conference scheduling across many regions) tend to be strong programs seeking quality opponents — correlated with tournament advancement but likely captured by SoS/PageRank already. No NCAA validation found.

**Assessment:** Low priority. Implement only if PageRank and SoS features leave a significant unexplained variance that clustering might address.

---

### 4.6 Graph Feature Summary and Validation Gate

| Graph Feature | Expected Value | Implementation Cost | Validation Requirement |
|:---|:---|:---|:---|
| PageRank (weighted) | **High** — peer-reviewed validation | Low (10 lines) | Must exceed SoS r=0.2970 in local validation |
| Betweenness centrality | Low — no NCAA validation | Low (1 line) | Should improve on SoS; if not, drop |
| HITS authority | Low — 0.908 correlation with PR | Low (1 line) | Drop if PageRank already included |
| Clustering coefficient | Very Low | Low (1 line) | Drop unless strong residual signal |

**Recommendation for Story 4.5:** Implement PageRank with margin-weighted directed graph as the primary graph feature. Include betweenness centrality as a secondary feature. Skip HITS and clustering coefficient unless PageRank alone leaves meaningful unexplained variance.

---

## 5. Massey Ordinal Systems

**Story 4.3 scope** (integration into data pipeline); also directly relevant to Stories 4.2, 4.4, and 4.7. Massey Ordinals are pre-computed from `MMasseyOrdinals.csv` — no custom rating solver required for this feature type.

### 5.1 Coverage and System Availability

**From `eda_findings_synthesis.md` Section 2 (already confirmed):**
- 100+ ranking systems available in `MMasseyOrdinals.csv`
- Top 5 systems by season coverage (all 23 seasons, 2003–2025): **AP, DOL, COL, MOR, POM**
- File structure: `Season, RankingDayNum, SystemName, TeamID, OrdinalRank`

**Key system identifications:**

| Code | System | Methodology | Margin Used |
|:---|:---|:---|:---|
| AP | Associated Press Poll | Human media poll | No |
| COL | Colley Matrix | Win/loss, Bayesian, Cholesky solve | No |
| DOL | Dokter Odds | Odds/margin-based rating | Yes |
| MOR | Sonny Moore | Empirical power ratings | Yes |
| MAS | Massey | Least-squares, Massey matrix | Yes |
| POM | Pomeroy (KenPom) | Adjusted efficiency, per-possession | Yes |
| SAG | Sagarin | Three-component: Predictor + Golden Mean + Recent | Yes |
| WLK | Whitlock | Margin-based power ratings | Yes |

**Coverage alert:** The confirmed top-5 systems with all 23 seasons of coverage (2003–2025) are AP, DOL, COL, MOR, POM. Of these, AP and COL do not use margin of victory — leaving MOR, POM, and DOL as the confirmed full-coverage margin-based systems. SAG (Sagarin) and WLK (Whitlock) appear in community-validated ensembles as the most predictive composite, but their full 23-season coverage in the local `MMasseyOrdinals.csv` is **unconfirmed** — Story 4.3 must verify before committing to SAG+POM+MOR+WLK. If SAG or WLK have gaps, fall back to MOR+POM+DOL as the confirmed margin-based composite.

---

### 5.2 Community-Validated Composite: SAG + POM + MOR + WLK

**Most predictive ensemble (from documented Kaggle MMLM solutions 2019–2024):**

A composite average of the four most predictive individual Massey systems:
```
massey_composite = (SAG_rank + POM_rank + MOR_rank + WLK_rank) / 4
```

Logistic regression using `delta_massey_composite = team1_massey_composite - team2_massey_composite` achieved log loss ~−0.540–0.543 in documented solutions (Conor Dewey methodology, 2019+).

**⚠️ Coverage verification required (Story 4.3 gate):** SAG and WLK are NOT confirmed in the top-5 by 23-season coverage (that list is AP, DOL, COL, MOR, POM). Story 4.3 must verify SAG and WLK span before committing to this composite. If coverage gaps are found:
- **Primary fallback:** `(MOR_rank + POM_rank + DOL_rank) / 3` — all three confirmed for 23 seasons, all margin-based
- The SAG+POM+MOR+WLK composite remains the MVP target if coverage is confirmed

**Which systems to include/exclude:**
- **Include (if coverage confirmed):** SAG (Sagarin), POM (KenPom), MOR (Sonny Moore), WLK (Whitlock) — consistently validated in community solutions
- **Confirmed-coverage fallback:** DOL, MOR, POM — all 23 seasons, all margin-based
- **Neutral:** MAS, COL — less frequently cited in top-solution feature lists
- **Exclude from efficiency composite:** AP — human poll; less predictive than computer systems for game outcomes

**Preseason note:** Preseason AP polls have been found to predict bowl game outcomes better than final pre-game polls (capturing program history/prestige rather than recency). This is counterintuitive but suggests AP as a preseason prior feature may add value distinct from efficiency-based end-of-season rankings.

---

### 5.3 Composite Ranking Approaches

**Option A — Simple Average (Recommended for MVP):**
Average the ordinal ranks of the 4–8 most predictive systems. Compute `delta_composite` between two teams as the matchup feature. Achieves log loss ~−0.540, competitive with more complex approaches.

**Option B — Weighted Ensemble:**
Derive system-specific weights from prior-season cross-validated log loss. Individual system performances are similar (roughly equal weights after optimization), so Option A and Option B typically produce similar results in practice.

**Option C — PCA Reduction (Post-MVP):**
Apply PCA to reduce the full set of available Massey ordinal columns (potentially 30–50 systems with >20 seasons coverage) to 10–15 principal components capturing 90%+ of variance. First principal component ≈ consensus "overall quality" factor. Avoids multicollinearity among correlated systems. Requires more preprocessing infrastructure; recommend deferring to Post-MVP.

**Option D — Tournament-Specific Ensemble:**
Use only Massey rankings from the specific pre-tournament snapshot date (day 128 or equivalent) rather than in-season temporal rankings. For Kaggle MMLM tournament prediction, the final snapshot immediately before tournament selection is the most relevant.

---

### 5.4 End-of-Season vs. In-Season Temporal Rankings

**Recommendation:** For tournament prediction (predicting the tournament bracket before it begins), use the final pre-tournament Massey ordinals (last available `RankingDayNum` ≤ 128 for each system). This already incorporates full-season performance with the most recent games having the highest weight.

**For in-season walk-forward feature serving (Story 4.7):** The `RankingDayNum` field enables temporal slicing — use only ordinals published as of the feature computation date. This is critical for preventing data leakage during walk-forward backtesting.

**Temporal snapshot dates:** Massey systems publish at varying frequencies (weekly for some, daily for others near tournament). Verify actual publication dates in the local `MMasseyOrdinals.csv` before assuming daily availability.

---

### 5.5 Do Massey Ordinals Add Signal Beyond Box Scores?

**EDA context:** Box-score features (FGM, FGPct, Score) achieve r=0.23–0.26 with tournament advancement individually. SoS achieves r=0.2970. Massey ordinals (especially POM/SAG) incorporate both margin-adjusted efficiency AND schedule strength — they should subsume the individual box-score signals while adding cross-team comparative information.

**Community validation:** Documented Kaggle MMLM solutions consistently use Massey ordinals as the primary feature set (not raw box scores). This strongly suggests ordinals add signal beyond raw box scores. However, direct empirical validation against box-score-only models would be valuable — Story 5.4 should include this comparison.

**Key gap:** The 13 ranked features in `eda_findings_synthesis.md` Section 2 are correlated with tournament outcomes computed from raw game data. Massey ordinals capture the *relative team strength* perspective — this is a different angle that is likely additive, not redundant, with the direct box-score correlation analysis.

---

### 5.6 Normalization for Massey Ordinal Features

**Ordinal rank distribution:** Ranks are integers 1–N (N ≈ 350 for full D1 field). The rank distribution is approximately uniform (not normally distributed). Feature engineering should convert ranks to either:
1. **Delta (difference):** `team1_rank − team2_rank` — this delta is approximately normally distributed and is the most natural matchup-level feature.
2. **Percentile:** `(N − rank) / N` — converts to bounded [0,1] scale; apply logit + StandardScaler for consistency with other bounded features.
3. **z-score across the season:** `(rank − mean_rank) / std_rank` — equivalent to StandardScaler applied to the rank column.

**Recommendation:** Use delta ranks (`team1 − team2`) as the primary feature for matchup predictions. For team-level features (not matchup-specific), z-score standardization per season is appropriate.

---

## 6. Kaggle MMLM Community Techniques

**Story 4.1 research scope.** This section synthesizes year-by-year findings from Kaggle MMLM competition solutions (2014–2025), recovered via Kaggle winner interviews, mlcontests.com reports, Medium writeups, and public GitHub repositories. **Note:** The Kaggle competition discussion boards themselves require authentication and returned only JavaScript boilerplate when fetched directly. The findings below are sourced from publicly accessible writeups and should be treated as representative, not exhaustive, of the full board discussion content.

### 6.1 Year-by-Year Solution Summary (2014–2025)

| Year | 1st Place | Core Signal | Model | Key Feature Insight |
|:---|:---|:---|:---|:---|
| 2014 | Lopez & Matthews | Vegas point spreads + KenPom | Logistic regression | Market efficiency > complex ML |
| 2015 | Zach Bradshaw | KenPom + Bayesian domain priors | Bayesian logistic regression | Domain priors from NBA analytics |
| 2016 | Miguel Alomar | Massey ordinals + power ratings | Unknown | Boards mention ordinals as dominant signal |
| 2017 | Andrew Landgraf | Team efficiency ratings + travel distance | Bayesian logistic (rstanarm) | **Game-theoretic**: modeled competitors' submissions |
| 2017 2nd | Scott Kellert | Opponent-adjusted point differential | Unknown | "Preprocessing is the most important part" |
| 2018 | raddar (Barušauskas) | FiveThirtyEight 538 team ratings | Team Quality transformation | FTE already the best public signal — use it directly |
| 2019 | Unknown | KenPom + ORB/TO margin | Likely Bayesian logistic | Silver medal: goto_conversion bias correction on 538 |
| 2020 | — | — | — | Tournament cancelled (COVID-19) |
| 2021 | Unknown (raddar approach continued by others) | 538 Team Quality | Same raddar approach | 65 Massey systems + XGBoost documented in a parallel top-ranked solution (Edwards 2021); actual 1st-place attribution unconfirmed |
| 2022 | Amir Ghazi | 538 Team Quality | raddar code verbatim | Original raddar finished 593rd with *new* approach |
| 2023 | RustyB | 538 Team Quality + Brier fix | raddar code, minor tweaks | Top-1% gold: 10 best Massey systems + XGBoost |
| 2024 | Unknown | External ratings | Monte Carlo simulation (R) | Probabilistic simulation; no ML model in conventional sense |
| 2025 | Mohammad Odeh | Seed diff + Team Quality + efficiency | XGBoost (Optuna-tuned) | Cubic-spline calibration in-fold; ~23 features |

---

### 6.2 The Dominant Finding: FiveThirtyEight (538) Ratings Signal (2018–2023)

The most striking cross-year pattern is that the **same core solution** — raddar's (Darius Barušauskas) FiveThirtyEight 538-based "Team Quality" approach — won or nearly won the competition across **at least 4 consecutive years (2021–2023)** and was originally created for the 2018 women's competition.

**Core insight:** FiveThirtyEight's NCAA team ratings already incorporate adjusted efficiency, schedule strength, and prior-season information. Rather than building complex ML pipelines from raw box scores, the top approach was to use 538 ratings directly with a simple logistic transformation.

**Practical implication for this project:** The `ncaa-men-538-team-ratings` dataset (available on Kaggle via raddar, covering 2016–2023) is a high-value external signal. However, the 538 NCAA model was discontinued after the 2023 competition season (FiveThirtyEight shut down). This data source is available for historical backtesting (2016–2023) but **cannot be extended to 2024–2025**. The project must fall back to Massey ordinals (POM/SAG/MOR/WLK composite) as the equivalent signal for full-history backtesting.

---

### 6.3 The goto_conversion Finding (Multiple Silver Medals: 2019, 2021)

The `goto_conversion` adjustment corrects for the "favourite-longshot bias" inherent in probability estimates derived from regular-season performance:

- **Problem:** FiveThirtyEight ratings underestimate stronger teams because they play more "garbage time" minutes in regular-season blowouts, suppressing their measured efficiency.
- **Correction:** `goto_conversion` adjusts win probabilities toward favorites by correcting the favourite-longshot bias present in betting-odds-derived probabilities.
- **Result:** Won silver medal in both 2019 and 2021 using this approach on top of 538 ratings.

**Implication:** Any efficiency-based rating system (including our SRS / ridge regression implementation in Story 4.6) may systematically underrate dominant teams. Calibration of probability outputs is critical — plain logistic regression on rating differences may need isotonic regression or spline calibration (as used by the 2025 winner: cubic-spline calibration in-fold).

---

### 6.4 Most Detailed Non-Winning Solution: John Edwards 2021

The 2021 solution by John Edwards (mid-pack final rank) is the most comprehensively documented open-source solution found and represents what a thorough feature engineering pipeline looks like. It used:

**Rating systems computed from scratch (all implemented in R, all available in Python):**
- Elo ratings with home-field advantage + margin-of-victory scaling
- LRMC (Logistic Regression Markov Chain)
- SRS (Simple Rating System)
- RPI (Rating Percentage Index)
- TrueSkill ratings
- Mixed-effects model ratings (via lme4 equivalent)
- Colley Matrix method
- Win percentage and margin of victory

**Box-score features (per possession):**
All 11 box-score stats normalized by possession count: `possessions = FGA − OR + TO + (0.44 × FTA)`. Features: FGA, 3PA, FTA, OR, DR, Ast, TO, Stl, Blk, PF, Score — all divided by possession count (team stats) or opponent possession count (defensive stats).

**Massey Ordinals:** Pre-tournament rankings from **65 different rating systems** with KNN imputation for missing values across systems with incomplete coverage.

**Model:** XGBoost, tuned via Bayesian optimization (1,000 iterations), minimizing log loss. Feature selection via Boruta algorithm (~40 confirmed important features). Validated on 2015–2019 tournament holdout (log loss ~0.50 on 2019 data).

**Key preprocessing notes from Edwards:**
- Overtime games rescaled to points-per-40-minutes equivalent before all calculations
- Home court advantage estimated via logistic regression on all non-neutral regular-season games (then used as a fixed adjustment)
- Tournament seeding and seed differential included

---

### 6.5 Feature Patterns Consistent Across Top Solutions (2014–2025)

**Tier 1 — Appear in virtually every top solution regardless of year:**

| Feature | Source | Typical Implementation |
|:---|:---|:---|
| **Seed difference** | `MNCAATourneySeeds.csv` | `re.search(r"\d+", seed_str)` → int; delta between teams |
| **External adjusted efficiency** | 538 ratings (2016–2023) or KenPom/POM | Pre-computed ratings used directly; not recomputed from box scores |
| **Composite Massey ordinals** | `MMasseyOrdinals.csv` | Avg of SAG + POM + MOR + WLK; pre-tournament snapshot |

**Tier 2 — Common in top-10 solutions:**

| Feature | Source | Notes |
|:---|:---|:---|
| **Elo rating difference** | Computed from game results | K=38–42; season mean-reversion 20–35% toward conference mean |
| **Per-possession box scores (Four Factors)** | `MRegularSeasonDetailedResults.csv` | eFG%, TO%, ORB%, FTR; 2003+ only |
| **Win percentage** | Game results | Full-season + conference-only split |
| **Opponent-adjusted point differential** | Game results | SRS or ridge regression output |

**Tier 3 — Appear in top-25–50 solutions:**

| Feature | Source | Notes |
|:---|:---|:---|
| **Recent form** | Rolling window on box-score stats | Last 10–15 games efficiency or EWMA |
| **Conference quality** | Inferred from seeding or `MTeamConferences.csv` | Handle realignment by year |
| **Vegas point spreads / market ratings** | External (The Prediction Tracker) | Used by 2014 winner; requires external data not in Kaggle MMLM |

---

### 6.6 Model Architecture Patterns

- **Logistic regression** (with composite Massey ordinal deltas) is a competitive simple baseline — achieves log loss ~−0.540, wins competitions when the right external signal is used (2014–2018 winners).
- **XGBoost / LightGBM / CatBoost** dominated 2019–2025 when custom-feature pipelines were built from raw box scores. The 2025 winner used XGBoost with Optuna hyperparameter tuning.
- **Neural networks / deep learning** have been attempted but generally do not outperform well-tuned GBDT ensembles. The primary bottleneck is the small training set (~2,000 tournament games since 2003).
- **Monte Carlo tournament simulation** (2024 MMLM winner): simulate tournament outcomes probabilistically from pairwise win probabilities rather than predicting individual games in isolation. This is planned for Story 6.5.
- **Game-theoretic competition strategy** (2017 winner Landgraf): modeled competitors' expected submissions, then found the optimal submission to maximize probability of a top-5 finish — not just the submission with highest accuracy. This is a competition-specific meta-strategy, not a feature engineering technique.

---

### 6.7 Novel Techniques Not Captured in EDA Tier List

**Elo Rating System (in-season dynamic ratings):**
Standard Elo update: `r_new = r_old + K × (actual − expected)`, where `expected = 1 / (1 + 10^((r_opponent − r_team)/400))`.

Enhanced Elo variants validated across competition years:
- **Margin-of-victory scaling:** `K_eff = K × min(margin, max_margin)^0.85` — diminishing returns on blowouts (Silver/SBCB formula; Edwards' 2021 solution)
- **Variable K-factor:** K=56 (first 20 games, high uncertainty) → K=38 (regular season) → K=47.5 (tournament)
- **Season mean-reversion:** Between seasons, regress 20–35% of rating toward conference mean. Captures roster turnover.
- **Home-court adjustment:** Subtract 3–4 Elo points from home team's effective rating before computing expected outcome

**goto_conversion / Probability Calibration:**
Corrects favourite-longshot bias in efficiency-based win probabilities. The 2025 winner applied cubic-spline calibration in-fold. The goto_conversion approach provides a ready-made implementation.

**LRMC (Logistic Regression Markov Chain):**
Models tournament outcomes as a Markov chain where each team's win probability against any opponent is derived from game-by-game outcomes via logistic regression. Results in a steady-state probability distribution over tournament outcomes. Documented in Edwards 2021.

**Time-Decay Features:**
- BartTorvik: games > 40 days old lose 1% emphasis per day; minimum weight 60% at 80+ days
- Applicable as a game-level weight in rolling aggregations: `weight = max(0.6, 1 − 0.01 × max(0, days_ago − 40))`

**Per-Possession Normalization (Four Factors):**
Normalizing all box-score stats by possession count is the standard in modern CBB analytics:
- `possessions = FGA − OR + TO + 0.44 × FTA`
- `eFG% = (FGM + 0.5 × FGM3) / FGA`
- `ORB% = OR / (OR + opponent_DR)`
- `FTR = FTA / FGA`
- Used consistently in top-10 solutions (Edwards 2021 built a full per-possession pipeline)

**Overtime Rescaling:**
Edwards 2021 rescaled overtime game scores to a points-per-40-minutes equivalent before any aggregation — treating a 5-OT game the same as a regulation game. This prevents outlier inflation of scoring statistics for teams that happened to play many overtime games.

---

### 6.8 Features to Add Beyond EDA Tier List

| Technique | Assessment for Epic 4 |
|:---|:---|
| **Composite Massey ordinals (SAG + POM + MOR + WLK)** | **Recommended for MVP** — validated across 2014–2025; primary signal in top solutions |
| **Elo rating difference** | **Recommended for MVP** — validated in Edwards 2021 and many other solutions |
| **Four Factors (eFG%, ORB%, FTR)** | **Recommended for MVP** — eFG% extends FGPct; ORB% is new; 2003+ required |
| **Per-possession normalization (possession denominator)** | **Recommended MVP** — use `possessions = FGA − OR + TO + 0.44×FTA` as denominator for counting stats (distinct from Four Factors derived metrics — see below) |
| **Probability calibration** | **Recommended MVP** — apply isotonic or spline calibration to model outputs; prevents overconfidence |
| **LRMC** | Post-MVP — higher implementation complexity; validated but marginal vs. Elo |
| **Overtime rescaling** | **Recommended** — low-cost preprocessing; prevents OT game outliers from distorting aggregations |
| **TrueSkill / Glicko-2** | Post-MVP — marginal improvement over Elo |
| **Vegas point spreads** | Post-MVP — requires external data source not in Kaggle MMLM dataset |
| **goto_conversion calibration** | Post-MVP — useful for calibration; implement after base model is validated |

---

## 7. Prioritized Implementation Plan

**DECISION-GATE NOTE:** The rankings below represent proposals with trade-off assessments, not final MVP-scope selections. The product owner makes the final scope decisions per AC 8 before downstream Stories 4.2–4.7 begin. Features marked "Recommended for MVP" are proposals based on expected predictive value, implementation cost, and community validation.

**VALIDATION GATE NOTE:** All MVP recommendations are **conditional** on passing the validation gates specified in the table. Any feature that does not exceed the SoS baseline (r=0.2970) during Story 4.4–4.6 implementation should be demoted to Post-MVP regardless of its ranking here, and the product owner should be notified before that story is marked complete.

### 7.1 Story Mapping

| Story | Scope | MVP Features | Post-MVP |
|:---|:---|:---|:---|
| **4.2** | Chronological Data Serving API | Walk-forward game iterator with date guards; 2025 deduplication; 2020 COVID handling | — |
| **4.3** | Canonical ID Mapping & Data Cleaning | `MTeamSpellings.csv` canonical name map; `MNCAATourneySeeds.csv` (seed_num, region, is_play_in); `MTeamConferences.csv` (season, team_id → conf); Massey Ordinal ingestion (SAG + POM + MOR + WLK temporal slices) | Full Massey coverage (50+ systems); PCA composite |
| **4.4** | Sequential Transformations | Rolling FGPct (10-game), FGM, Score, TO_rate, PF, DR; EWMA (α=0.20); loc encoding (H=1, A=−1, N=0); momentum feature (ewma_fast − ewma_slow); OT game rescaling (pts/40min equivalent); per-possession normalization (`possessions = FGA − OR + TO + 0.44×FTA`) | 5-game and 20-game windows; streak features; time-decay weights; eFG%, ORB%, FTR (Four Factors beyond FGPct/TO_rate) |
| **4.5** | Graph Builders & Centrality | PageRank (directed, margin-weighted, warm-start incremental); betweenness centrality | HITS authority; clustering coefficient |
| **4.6** | Opponent Adjustments | Ridge regression efficiency ratings (AdjO, AdjD, AdjEM equivalent); Elo rating with K=38, margin scaling, season mean-reversion | Glicko-2 / TrueSkill; LRMC; goto_conversion calibration |
| **4.7** | Stateful Feature Serving | Feature composition from all 4.2–4.6 outputs; `gender_scope` and `dataset_scope` configurability; temporal slicing for walk-forward; probability calibration (isotonic or spline) | Matrix completion as alternative rating approach |

### 7.2 MVP vs. Post-MVP Summary

**Recommended for MVP (in priority order within each story):**

| # | Feature | Story | Rationale | Validation Requirement |
|:---|:---|:---|:---|:---|
| 1 | Massey composite (SAG + POM + MOR + WLK avg, **if coverage confirmed**; else MOR + POM + DOL) | 4.3 | Community-validated; pre-computed; highest log-loss performance in Kaggle MMLM | **GATE:** Verify SAG and WLK 23-season coverage in local `MMasseyOrdinals.csv` before implementation. Fallback: MOR+POM+DOL (all confirmed full-coverage margin systems). |
| 2 | Rolling FGPct (10-game) | 4.4 | Highest raw r=0.2269; tournament diff +0.078 | Should improve on season-average FGPct |
| 3 | Rolling FGM / Scoring (10-game) | 4.4 | r=0.2628 / 0.2349 | Standard feature from EDA Tier 1 |
| 4 | Elo rating (K=38, margin scaling, mean reversion) | 4.6 | Tier 2 in every validated Kaggle MMLM solution; captures recency | Must exceed SoS baseline (r=0.2970) |
| 5 | PageRank (margin-weighted, warm-start) | 4.5 | Peer-reviewed validation; should exceed SoS baseline | Must exceed SoS r=0.2970 in local validation |
| 6 | SRS / Massey efficiency ratings (AdjO, AdjD, AdjEM equivalent) | 4.6 | KenPom-equivalent metric; simplest full-season solver; no hyperparameter tuning | Must exceed SoS baseline (r=0.2970) |
| 7 | Rolling TO_rate, PF, DR (10-game) | 4.4 | EDA Tier 1 negative predictors; r=−0.1424, −0.1574 | Standard from EDA Tier 1 |
| 8 | EWMA (α=0.20) momentum feature | 4.4 | Low cost; addresses recency without window-size tuning | Feature importance validation in Story 5.4 |
| 9 | loc encoding (H/A/N) | 4.4 | EDA Tier 2; declining home advantage is time-varying | Encode as numeric + include season interaction |
| 10 | Probability calibration (isotonic or cubic-spline) | 4.7 | 2019 and 2021 silver medals; 2025 winner used in-fold spline calibration; corrects favourite-longshot bias in all efficiency-based ratings | Apply after base model validated; in-fold calibration prevents leakage |

**Recommended for Post-MVP (defer unless MVP underperforms):**

| # | Feature | Story | Rationale for Deferral |
|:---|:---|:---|:---|
| 10 | 5-game and 20-game rolling windows | 4.4 | Validate 10-game window first; add alternatives only if feature importance warrants |
| 11 | Streak features | 4.4 | Mixed statistical evidence; low expected incremental value |
| 12 | Betweenness centrality | 4.5 | No NCAA validation; likely low incremental value over PageRank |
| 13 | Four Factors (eFG%, ORB%, FTR) | 4.4/4.6 | Subsumes FGPct/TO_rate; higher complexity; validate after MVP |
| 14 | Full Four Factors feature set (eFG%, ORB%, FTR) via per-possession arithmetic | 4.4/4.6 | Distinct from MVP possession denominator (which is simpler). Full Four Factors require per-possession arithmetic applied to derived metrics — higher pipeline complexity; validate after MVP rolling stats are implemented. |
| 15 | Glicko-2 / TrueSkill | 4.6 | Marginal improvement over Elo; uncertainty quantification only useful for early-season; not needed for pre-tournament snapshot |
| 16 | PCA composite (full Massey set) | 4.3 | Complexity of PCA pipeline; Simple average (MVP option) captures most of the signal |
| 17 | HITS / clustering coefficient | 4.5 | No NCAA validation; high PageRank correlation for HITS |
| 18 | Time-decay game weighting | 4.4 | Low cost but complex to implement correctly for walk-forward |
| 19 | Travel distance / elevation | 4.4 | Marginal expected value; requires external geocoding data |
| 20 | Matrix completion | 4.7 | High implementation cost; validate only if rating-based approaches underperform |

---

### 7.3 Open Questions Deferred to Implementation Stories

These questions cannot be answered definitively without empirical validation during Stories 4.4–4.6:

1. **Optimal rolling window size (5 / 10 / 20 games):** Implement all three; select via XGBoost feature importance in Story 5.4.
2. **Optimal EWMA alpha:** Start at 0.20; tune via cross-validated log loss across 3 prior seasons.
3. **Ridge lambda for efficiency ratings:** Tune via cross-validation; start at λ=20.
4. **PageRank vs. SoS baseline validation:** Must be confirmed empirically on local data. Literature suggests improvement, but the correlation depends on the specific graph construction (edge weights, damping factor).
5. **Massey ordinal system coverage:** Verify SAG and WLK are available for the full 23-season span in the local `MMasseyOrdinals.csv` before committing to the SAG + POM + MOR + WLK composite.
6. **Normalization for new features (Elo, PageRank, Massey delta):** Approximately normal for deltas — validate empirically; apply no transform + StandardScaler as baseline.

---

## Normalization Reference for New Features

**New features not in `statistical_exploration_findings.md` Section 7:**

| Feature | Expected Distribution | Proposed Transform | Proposed Scaler |
|:---|:---|:---|:---|
| Elo rating (single team) | Approximately normal | none | StandardScaler |
| Elo delta (matchup feature) | Approximately normal | none | StandardScaler |
| PageRank score | Right-skewed (power law) | log | StandardScaler |
| Betweenness centrality | Extreme right skew | sqrt | RobustScaler |
| Massey ordinal rank | Approximately uniform (integers 1–N) | none (use delta between teams) | StandardScaler after delta |
| Massey ordinal delta | Approximately normal | none | StandardScaler |
| Ridge AdjEM (single team) | Approximately normal | none | StandardScaler |
| EWMA efficiency | Same as underlying stat | Same as underlying stat | Same as underlying stat |

*All recommendations are empirically unvalidated for this dataset — verify distributions locally before committing.*

---

## References

### Internal Project Documents
- `notebooks/eda/eda_findings_synthesis.md` — All 4 sections; primary EDA context
- `notebooks/eda/statistical_exploration_findings.md` — Section 7 (normalization), Section 4 (SoS baseline), Section 5 (correlations)
- `specs/05-architecture-fullstack.md` — Section 3 (tech stack: networkx, vectorization mandate), Section 9 (project structure: `transform/` module)
- `specs/research/data-source-evaluation.md` — Document format convention and research methodology
- `_bmad-output/planning-artifacts/template-requirements.md` — Epic 4 normalization design requirements; decision-gate pattern from Story 2.1 code review

### Academic Papers
- Matthews, M., et al. (2021). "PageRank as a Method for Ranking NCAA Division I Men's Basketball Teams." *Journal of Sports Analytics*, 7(2). DOI: 10.3233/JSA-200425
- Massey, K. (1997). "Statistical Models Applied to the Rating of Sports Teams." masseyratings.com/theory/massey97.pdf
- Colley, W.N. (2002). "Colley's Bias Free College Football Rankings Method." colleyrankings.com/matrate.pdf
- Lopez, M. & Matthews, G. (2015). "Building an NCAA mens basketball predictive model and quantifying its success." arXiv:1412.0248

### Kaggle MMLM Competition Writeups (2014–2025)
- Lopez (2014 winner): Skidmore College news article — skidmore.edu/news/2015/032315-lopez-wins-prize-for-best-ncaa-picks.php
- Bradshaw (2015 winner): Kaggle Blog / Medium — medium.com/kaggle-blog/predicting-march-madness-1st-place-winner-zach-bradshaw
- Landgraf (2017 winner): medium.com/kaggle-blog/march-machine-learning-mania-1st-place-winners-interview-andrew-landgraf-f18214efc659
- Kellert (2017 2nd): medium.com/kaggle-blog/march-machine-learning-mania-2017-2nd-place-winners-interview-scott-kellert-f9f272194bd3
- Forseth (2017 4th): medium.com/kaggle-blog/march-machine-learning-mania-4th-place-winners-interview-erik-forseth-8d915d8cea57
- Edwards (2021 documented solution): johnbedwards.io/blog/march_madness_2021/
- maze508 (2023 top-1% gold): medium.com/@maze508/top-1-gold-kaggle-march-machine-learning-mania-2023-solution-writeup-2c0273a62a78
- Odeh (2025 winner): kaggle.com/competitions/march-machine-learning-mania-2025/writeups/mohammad-odeh-first-place-solution
- mlcontests.com annual reports (2022–2025): mlcontests.com/state-of-machine-learning-competitions-[year]

### Methodology References
- Dewey, C. (2019). "Machine Learning Madness." conordewey.com/blog/machine-learning-madness-predicting-every-ncaa-tournament-matchup/
- Silver, N. (2024). SBCB Methodology. natesilver.net/p/sbcb-methodology
- Sports Reference (2015). "SRS Calculation Details." sports-reference.com/blog/2015/03/srs-calculation-details/
- goto_conversion Python package: github.com/gotoConversion/goto_conversion
- raddar's NCAA Men 538 Team Ratings Dataset: kaggle.com/datasets/raddar/ncaa-men-538-team-ratings
