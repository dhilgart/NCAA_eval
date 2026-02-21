# Feature Engineering Techniques — NCAA Tournament Prediction

**Date:** 2026-02-20
**Story:** 4.1 (Spike) — Research Feature Engineering Techniques
**Status:** Complete — Pending Product Owner Approval (AC 8)
**Author:** Dev Agent (Claude Sonnet 4.6)

---

## Quick-Navigation Table

| Section | Topic | Library Notes |
|:---|:---|:---|
| [1. Data Context](#1-data-context) | What EDA confirmed; open questions | Pre-read before implementation |
| [2. Opponent Adjustment](#2-opponent-adjustment-techniques) | SRS, Ridge, Massey, Colley, Elo | SRS / Massey / Ridge form one equivalence group (margin-adjusted batch ratings); Colley is distinct (win/loss-only); Elo is distinct (dynamic) |
| [3. Sequential / Momentum](#3-sequential--momentum-features) | Rolling windows, EWMA, streaks | Window size and alpha are modeler-configurable parameters of distinct building blocks |
| [4. Graph Features](#4-graph-based-features) | PageRank, betweenness, HITS | PageRank captures transitive strength; betweenness captures structural position; HITS hub captures opponent quality from the loss side |
| [5. Massey Ordinals](#5-massey-ordinal-systems) | 100+ ranking systems | Pre-computed multi-system consensus; covers systems the project cannot replicate; SAG + POM + MOR + WLK composite (verify coverage) |
| [6. Community Techniques](#6-kaggle-mmlm-community-techniques) | Kaggle MMLM top solutions 2014–2025 | 538-based approach dominated 2018–2023; open alternatives: Elo + ordinals + seed diff; goto_conversion for calibration |
| [7. Library Catalog](#7-library-building-blocks-catalog) | Mapping to Stories 4.2–4.7 | Equivalence groups (implement one representative) and distinct building blocks (implement all) |

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

**EDA reference correlations** (contextual reference for implementation stories — not a gate):

| Feature | r | Source |
|:---|:---|:---|
| Raw SoS (opponent win rate) | 0.2970 | `statistical_exploration_findings.md` Section 4 |
| FGM | 0.2628 | Section 5 |
| Score | 0.2349 | Section 5 |
| FGPct | 0.2269 | Section 5 |

### Open Questions (Research Questions Addressed by This Spike)

| Open Question | Resolution in This Document |
|:---|:---|
| What distinct information do graph centrality metrics provide vs. SoS? | See Section 4 — PageRank captures win-chain transitivity (2 hops vs SoS 1 hop); betweenness captures structural bridge position; HITS hub captures opponent quality from the loss side |
| Which Massey Ordinal systems should form the pre-computed composite? | See Section 5 — SAG + POM + MOR + WLK community-validated; fallback MOR + POM + DOL if SAG/WLK coverage unconfirmed |
| Are rolling window sizes distinct building blocks or parameters of the same block? | See Section 3 — parameters of the same building block; all three (5/10/20) should be implemented as configurable options for the modeler |

### Data Coverage Caveats

- **Box-score features (FGM, FGPct, etc.) available 2003–2025 only.** Pre-2003 sequential features are limited to compact stats (score, margin, loc).
- **Tournament detailed results available through 2024 only.** 2025 tournament box scores unavailable until tournament completion.
- **2025 deduplication required.** 4,545 games stored twice (Kaggle + ESPN IDs). Deduplicate by `(w_team_id, l_team_id, day_num)`; prefer ESPN records for `loc` and `num_ot`.
- **2020 COVID: no tournament.** Include 2020 regular season in training; exclude from evaluation.

---

## 2. Opponent Adjustment Techniques

**Story 4.6 scope.** These techniques form an equivalence group (Section 7.1) — SRS, Massey, and Ridge all capture the same signal (full-season margin-adjusted batch ratings) via different formulations. Colley and Elo are distinct building blocks that capture different information.

### 2.1 Simple Rating System (SRS)

**Description:** Iterative least-squares system that simultaneously solves for all team ratings such that each team's rating equals its average point differential plus its average opponent rating. Used by Sports Reference and closely related to KenPom's AdjEM.

**Matrix formulation:** Solve `Gᵀ G r = Gᵀ S` where `G` is the team-game indicator matrix (+1 home, −1 away per game row), `r` is the ratings vector, and `S` is the margin-of-victory vector. Fixed-point iteration: `r_i(k+1) = avg_margin_i + avg(r_j for all opponents j)`. Convergence is guaranteed for connected schedules (spectral radius < 1); approximately 3,000–5,000 iterations to full convergence.

**KenPom AdjEM relationship:** KenPom solves offense and defense separately on a per-100-possession basis. The iterative formula is `AdjO_A = RawO_A × (national_avg_D / AdjD_opponent)`. Conceptually identical to SRS but normalized by possessions. KenPom AdjEM historically achieves ~75%+ accuracy on regular-season game outcomes.

**Data requirements:** Box-score results 2003+. This project always computes ratings on full-season data (pre-tournament snapshot), so convergence is well-conditioned with 30+ games per team.

**Computational complexity:** O(k × |E|) iterative, or O(n³) direct solve. For NCAA (n=350 teams, |E|=5,800 games/season): trivial.

| Factor | Assessment |
|:---|:---|
| Feasibility | ✅ High — well-documented linear algebra; no new dependencies |
| Complexity | Medium — iterative convergence; numerical stability required |
| Distinct signal | Margin-adjusted opponent strength accounting for schedule quality — the canonical representative of the equivalence group (Section 7.1 Group A) |

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
| Distinct signal | Same margin-adjusted signal as SRS (Group A equivalence); adds λ as a modeler-exposed shrinkage parameter — not a new signal but a tuning knob on the same building block |

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
| Distinct signal | Same margin-adjusted signal as SRS (Group A equivalence); distinct formulation only; pre-computed as "MAS" in MMasseyOrdinals as an alternative to re-implementing |

---

### 2.4 Colley Matrix Method

**Description:** Win/loss-only rating system using a Bayesian interpretation. The Colley rating for team i starts at 0.5 (prior) and is updated by: `r_i = (1 + w_i) / (2 + t_i)` with opponents' win rates replaced by full Colley ratings. Solved via the Colley matrix `C` (symmetric positive definite): `C[i,i] = 2 + t_i`, `C[i,j] = −n_ij` (games between i and j), right-hand side `b[i] = 1 + (w_i − l_i)/2`.

**Key difference from Massey/SRS:** Colley ignores margin of victory entirely — this is both its defining characteristic (no score-running incentive; win/loss purity) and what makes it a distinct building block from the Group A methods. A modeler can use Colley to test whether margin-free ratings add independent information on top of margin-adjusted ratings.

**Available via MMasseyOrdinals.csv:** The "COL" system is the Colley Matrix. Pre-computed ordinals available for 2003–2024.

| Factor | Assessment |
|:---|:---|
| Feasibility | ✅ High — Cholesky solve via scipy |
| Complexity | Low — same as Massey |
| Distinct signal | Win/loss-only rating (Group B); explicitly omits margin — provides a pure win-percentage-adjusted signal that a modeler can combine with or contrast against Group A margin-based ratings |

**Implementation note:** Available pre-computed as "COL" in MMasseyOrdinals — using the pre-computed value avoids re-implementing the solver, which is a viable option for Story 4.3/4.6.

---

### 2.5 Equivalence and Distinctness Summary for Story 4.6

**Context:** This project always computes ratings on full-season data (pre-tournament snapshot, 30+ games per team). All methods are well-conditioned at full season.

**Equivalence Group A — Margin-Adjusted Batch Ratings:** SRS, Massey, and Ridge are algebraically near-equivalent for full-season data. They capture the same underlying signal; the differences are formulation and hyperparameter exposure:

| Method | Formulation | Margin Used | Hyperparameter | Implementation Cost |
|:---|:---|:---|:---|:---|
| SRS / Massey iterative | Fixed-point iteration | Yes | None | Low |
| Massey direct solve | Cholesky n×n solve | Yes | None | Low; pre-computed as "MAS" in MMasseyOrdinals |
| Ridge regression | Regularized SRS | Yes | λ (shrinkage) | Medium — λ tuning via CV; exposes regularization as a modeling knob |

**Equivalence Group B — Win/Loss-Only Ratings (distinct from Group A):**

| Method | Formulation | Margin Used | Notes |
|:---|:---|:---|:---|
| Colley Matrix | Bayesian Cholesky | No | Provides a margin-free counterpart; available pre-computed as "COL" |

**Distinct building block — Dynamic Ratings (distinct from both groups above):**

| Method | Notes |
|:---|:---|
| Elo (in-season) | Stateful, game-by-game updates; weights recent games more heavily than a full-season solve; captures in-season progression — see Section 6 for full specification |

**Normalization for adjusted efficiency features:** Adjusted efficiency metrics (AdjO, AdjD, AdjEM equivalents) are approximately normally distributed (similar to Score/FGM). Baseline: no transform + StandardScaler. Validate distribution empirically before committing.

---

## 3. Sequential / Momentum Features

**Story 4.4 scope.** This section covers the sequential and temporal building blocks available to the modeler: rolling window stats, EWMA, streak encoding, momentum/trajectory features, and per-game preprocessing (loc encoding, OT rescaling). Normalization for box-score rolling statistics follows Section 1 baselines. Window sizes and alpha values are modeler-configurable parameters, not fixed design decisions.

### 3.1 Rolling Window Sizes

**Candidates:** 5, 10, 20 games (from story scope). No strong empirical consensus exists in the academic literature for NCAA basketball specifically. The EDA found no strong evidence for any particular window size.

**What is known:**
- **5-game windows:** Highly volatile, sensitive to single-game anomalies. Used in betting models (short-term form). For tournament prediction (a March snapshot of season-long performance), 5-game windows may over-fit to recent noise.
- **10-game windows:** Balance between recency and stability. Roughly corresponds to the final month of regular-season play (~30-35 game seasons; 10 games ≈ last 25–30% of season). Appears in documented Kaggle MMLM solutions as a "late-season form" feature.
- **20-game windows:** Covers roughly 55–65% of a college basketball season. Approaches a "second half of season" metric. Less sensitive to single-game volatility.
- **Full-season:** KenPom and SRS compute full-season aggregates but weight recent games more heavily in their iterative solvers (implicitly encoding recency).

**Academic reference:** A 2025 deep learning paper on NCAA basketball (Habib, M.I., "Forecasting NCAA Basketball Outcomes with Deep Learning: A Comparative Study of LSTM and Transformer Models," arXiv:2508.02725) explicitly noted that rolling windows were not used and that "static features lack patterns such as recent form, late-season momentum." No paper in the literature provided a rigorous optimal-window-size comparison for NCAA tournament prediction specifically.

**Implementation note for Story 4.4:** Implement all three window sizes (5, 10, 20) as parallel feature columns — they are parameters of the same building block, not competing features. Include full-season aggregate as a fourth reference column. The modeler selects which window sizes to include in any given model.

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

**Implementation note:** Include signed integer encoding as the standard representation. Streak features are a distinct building block — they capture pure win/loss sequence dynamics independent of efficiency magnitude. The modeler decides whether to include them in a given model based on their goals.

---

### 3.4 Performance Trajectory (Momentum)

**Definition:** Rate of change of a rolling efficiency metric, capturing whether a team is improving or declining as the tournament approaches.

**Candidate formulations:**
- `momentum_N = ewma(adj_eff, α=0.20) − ewma(adj_eff, α=0.10)` — positive values indicate improving recent form over medium-term trend
- `trajectory = slope of OLS regression over last-N games' margins`
- `form_ratio = last_10_adj_eff / full_season_adj_eff` — ratio > 1 indicates better recent than average performance

**Practical note:** KenPom weights recent games more heavily in its iterative efficiency solve, implicitly encoding momentum. Teams with strong recent form will have slightly elevated ratings compared to a pure full-season average. Sagarin publishes a "Recent" sub-rating that is an explicit implementation of this concept.

**Implementation note:** Include `ewma_fast − ewma_slow` (momentum) as a standard output of the EWMA building block — low implementation cost once EWMA is built. Momentum captures the *rate of change* of efficiency (a distinct signal from the efficiency level itself).

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

**Story 4.5 scope.** NetworkX is already in the tech stack (`specs/05-architecture-fullstack.md` Section 3) — no new dependency required. Graph features capture *structural* information about how teams are connected through their schedule — distinct from the direct efficiency and schedule-strength measures in Sections 2–3.

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
| Distinct signal | Transitive win-chain strength (2 hops): credit propagates through win chains; distinct from SoS (1-hop opponent win rate); peer-reviewed NCAA validation (Matthews et al. 2021: 71.6% vs 64.2% naive win-ratio) |
| Walk-forward compatibility | ⚠️ Requires incremental update strategy (see Section 4.4) |

---

### 4.2 Betweenness Centrality

**Definition:** For team node v, betweenness centrality = sum over all team pairs (s,t) of `(shortest paths through v) / (total shortest paths s→t)`. Measures how often a team lies on the shortest path between any other pair of teams in the schedule network.

**Sports interpretation:** A team with high betweenness is a "bridge" between otherwise separate competitive clusters (e.g., a mid-major that beat a top-10 team and was beaten by a bottom-25 team, bridging two otherwise unconnected conference bubbles).

**Computational complexity:** Brandes algorithm: O(V×E) for unweighted, O(V×E + V² log V) for weighted. For NCAA (V=350, E=5,800 games), this runs in milliseconds.

**No peer-reviewed NCAA validation found.** The metric captures *structural position* rather than raw strength — a team with high betweenness is a "bridge" between otherwise separate competitive clusters, regardless of its win-loss record. This is a conceptually distinct building block from both PageRank (strength) and SoS (direct opponent quality).

**Normalization:** Betweenness values are bounded [0,1] but with extreme right skew (most teams near 0, few "bridge" teams much higher). Recommend: sqrt transform + RobustScaler, similar to Blk/Stl stats.

| Factor | Assessment |
|:---|:---|
| Feasibility | ✅ High — `nx.betweenness_centrality()` |
| Complexity | Low — single function call |
| Distinct signal | Structural "bridge" position in schedule network — conceptually distinct from both strength metrics (PageRank, SRS) and schedule quality (SoS); measures cross-cluster connectivity independent of win/loss record |
| Walk-forward compatibility | ⚠️ Must recompute each time step; warm-start not possible for betweenness |

---

### 4.3 HITS (Hub/Authority) Algorithm

**Mechanics:** Computes two scores per node:
- **Authority score:** High if many high-hub teams have lost to you (you beat teams that beat many others)
- **Hub score:** High if you have lost to many high-authority teams (you lost to good teams)

**Sports interpretation:** Authority ≈ "best team" (won against many quality opponents). Hub ≈ "difficult schedule" (played many good opponents even with losses).

**Correlation with PageRank:** HITS authority scores correlate ~0.908 with PageRank scores for typical sports networks. The incremental signal over PageRank is minimal. Hub scores capture "quality schedule despite losses" — a distinct signal partially covered by Colley/Massey SoS.

**Distinctness assessment:** The two HITS scores have different distinctness profiles:
- **Authority score**: r≈0.908 correlation with PageRank for sports networks — captures nearly the same information. Largely redundant as a library building block if PageRank is already implemented.
- **Hub score**: captures "quality of schedule despite losses" — a team that repeatedly lost to high-authority opponents scores high. This is a distinct signal not captured by PageRank or betweenness, partially complementary to Colley (win/loss-only) ratings.

No peer-reviewed paper found validating HITS specifically for NCAA basketball.

| Factor | Assessment |
|:---|:---|
| Feasibility | ✅ High — `nx.hits()` |
| Complexity | Low |
| Distinct signal | **Authority**: largely redundant with PageRank (r≈0.908 in sports networks) — low novelty over PageRank. **Hub**: "quality schedule despite losses" — distinct signal; captures opponent authority from the losing side, not captured by PageRank. |
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

**Distinct signal:** Functions as a "schedule diversity" metric — distinct from SoS (opponent quality), PageRank (strength transitivity), and betweenness (structural bridging). Low clustering indicates a team that played across many separate competitive regions rather than within a tight conference bubble. No NCAA peer-reviewed validation found.

**Implementation note:** Low implementation cost (`nx.clustering()`) — implement as part of the graph feature building block suite in Story 4.5 alongside PageRank and betweenness.

---

### 4.6 Graph Feature Library Summary

| Graph Feature | Distinct Information | Novelty vs. Other Building Blocks | Implementation Cost |
|:---|:---|:---|:---|
| PageRank (weighted) | Transitive strength through win chains (2 hops) | Distinct from SoS (1 hop) and SRS (global batch solve); peer-reviewed NCAA validation | Low (10 lines) |
| Betweenness centrality | Structural "bridge" position across schedule clusters | Distinct from strength (PageRank) and quality (SoS) — measures positional connectivity | Low (1 line) |
| HITS authority | Same underlying signal as PageRank | Largely redundant with PageRank (r≈0.908); low novelty if PageRank is implemented | Low (1 line) |
| HITS hub | "Lost to strong opponents" — opponent authority from the loss side | Distinct from PageRank and betweenness; complementary to win/loss-based ratings | Low (same `nx.hits()` call as authority) |
| Clustering coefficient | Schedule diversity (tight vs broad conference spread) | Distinct from all strength/position metrics | Low (1 line) |

**Story 4.5 implementation scope:** Implement all five graph building blocks. The modeler selects which to include in any given model.

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

**Community-validated composite (from documented Kaggle MMLM solutions 2019–2024):**

A composite average of the four most consistently cited Massey systems:
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

These are distinct building block options the modeler can choose from — they are not competing "best" approaches but alternative formulations each with different trade-offs:

**Option A — Simple Average:**
Average the ordinal ranks of the 4–8 systems. Compute `delta_composite` between two teams as the matchup feature. Community validation shows log loss ~−0.540. Simplest to implement and interpret.

**Option B — Weighted Ensemble:**
Derive system-specific weights from prior-season cross-validated log loss. Individual system performances are similar (roughly equal weights after optimization), so Option A and Option B typically produce similar results in practice. Adds a cross-validation infrastructure requirement.

**Option C — PCA Reduction:**
Apply PCA to reduce the full set of available Massey ordinal columns (potentially 30–50 systems with >20 seasons coverage) to 10–15 principal components capturing 90%+ of variance. First principal component ≈ consensus "overall quality" factor. Avoids multicollinearity among correlated systems. Requires more preprocessing infrastructure.

**Option D — Tournament-Specific Snapshot:**
Use only Massey rankings from the specific pre-tournament snapshot date (day 128 or equivalent) rather than in-season temporal rankings. For Kaggle MMLM tournament prediction, the final snapshot immediately before tournament selection is the most relevant. This is an implementation choice applicable to any of Options A–C.

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

### 6.8 Building Blocks Surfaced by Community Research

These techniques appear in the Kaggle MMLM top solutions but were not in the EDA tier list. Each is a distinct building block:

| Technique | Community Validation | Distinct Signal | Notes |
|:---|:---|:---|:---|
| **Composite Massey ordinals (SAG + POM + MOR + WLK)** | Primary signal in top solutions 2014–2025; log loss ~−0.540 | Pre-computed multi-system consensus including systems the project cannot replicate | See Section 5 for full specification |
| **Elo rating difference** | Tier 2 across 2014–2025; Edwards 2021 and many others | Dynamic/stateful ratings capturing in-season progression — distinct from batch SRS/Massey | K-factor, margin scaling, mean reversion are modeler-configurable parameters |
| **Four Factors (eFG%, ORB%, FTR)** | Top-10 solutions; Edwards 2021 full per-possession pipeline | Derived efficiency ratios beyond raw box scores; ORB% is not in EDA tier list; eFG% extends FGPct | 2003+ only; requires per-possession arithmetic |
| **Per-possession normalization** | Consistently used in top solutions 2019–2025 | Removes pace confound from counting stats — distinct from Four Factors (simpler denominator normalization) | `possessions = FGA − OR + TO + 0.44×FTA` |
| **Probability calibration** | 2019 and 2021 silver medals; 2025 winner used in-fold cubic-spline | Output-level correction for favourite-longshot bias — not a feature but a post-processing step | Apply to model outputs; must be in-fold to prevent leakage |
| **LRMC** | Edwards 2021; documented in original Markov chain basketball literature | Markov-chain-derived win probabilities — distinct approach from SRS/Elo | Higher implementation complexity |
| **Overtime rescaling** | Edwards 2021 | Preprocessing step — normalizes per-40-minute equivalent for OT games | Low cost; prevents scoring-stat distortion for OT-heavy teams |
| **TrueSkill / Glicko-2** | Occasional top-25 solutions | Uncertainty-quantified Elo variant — distinct in that it explicitly models rating uncertainty per team | Marginal signal over Elo for pre-tournament snapshot |
| **Vegas point spreads** | 2014 winner (Lopez & Matthews) | Market efficiency signal not derivable from box scores | Requires external data source not in Kaggle MMLM dataset |
| **goto_conversion calibration** | 2019 and 2021 silver medals | Alternative calibration method to isotonic/spline — same functional role | Ready-made Python package; see Section 6.3 |

---

## 7. Library Building Blocks Catalog

The Epic 4 feature engineering suite provides **building blocks** from which the modeler selects combinations to experiment with. This section organizes all building blocks identified in this research into two structural categories:

1. **Equivalence groups** — techniques that capture the same underlying signal via different formulations. The library needs one representative implementation per group; implementing multiple members of the same group does not add distinct information.
2. **Distinct building blocks** — techniques that each capture genuinely different information. Combining any two of these provides the modeler with independent signals.

**Note:** The product owner makes final scope selections per AC 8 before downstream Stories 4.2–4.7 begin.

---

### 7.1 Equivalence Groups

**Group A — Full-Season Margin-Adjusted Batch Ratings (Story 4.6)**

All three methods below are algebraically near-equivalent for full-season data — they capture the same underlying signal:

| Method | Implementation | Hyperparameter | Notes |
|:---|:---|:---|:---|
| SRS (Simple Rating System) | Iterative fixed-point solve | None | Simplest; deterministic |
| Massey direct solve | Cholesky n×n solve | None | Equivalent to SRS; pre-computed as "MAS" in MMasseyOrdinals |
| Ridge regression | Regularized SRS (`sklearn.Ridge`) | λ (shrinkage) | Same signal as SRS with an exposed λ tuning knob for modelers who want regularization; λ=10–100 range |

*Implement SRS as the canonical Group A representative. Expose Ridge as a parameterized variant (λ-configurable). Massey is available pre-computed for modelers who want to skip the solver.*

**Group B — Win/Loss-Only Batch Ratings (Story 4.3 / 4.6)**

| Method | Implementation | Notes |
|:---|:---|:---|
| Colley Matrix | Cholesky solve | Win/loss purity; no margin; available pre-computed as "COL" in MMasseyOrdinals |

*Distinct from Group A because it explicitly discards margin. Provides the modeler a margin-free counterpart to SRS/Massey.*

**Group C — Graph Centrality (Strength Dimension) (Story 4.5)**

| Method | Notes |
|:---|:---|
| PageRank | Primary representative — transitive win-chain strength |
| HITS authority | r≈0.908 with PageRank in sports networks — largely redundant; low novelty over PageRank |

*Implement PageRank as the Group C representative. HITS authority adds minimal distinct information over PageRank.*

---

### 7.2 Distinct Building Blocks

Each technique below provides genuinely different information from all others in this table and from the equivalence groups above:

| Building Block | Distinct Information Captured | Story | Configuration Parameters |
|:---|:---|:---|:---|
| **Elo ratings (dynamic)** | Stateful, game-by-game team strength; weights recent games more heavily; captures in-season trajectory | 4.6 | K-factor; margin scaling exponent; season mean-reversion fraction; home court adjustment |
| **Massey Ordinals composite** | Pre-computed multi-system consensus from 100+ raters; includes proprietary/human systems the project cannot replicate | 4.3 | Which systems to include; snapshot date; composite method (Options A–D from Section 5.3) |
| **Betweenness centrality** | Structural "bridge" position in schedule network — how often a team lies on the path between other pairs | 4.5 | Weighted vs. unweighted; directed vs. undirected |
| **HITS hub score** | "Quality schedule despite losses" — how often a team lost to high-authority opponents | 4.5 | Standard HITS parameters |
| **Clustering coefficient** | Schedule diversity — how tightly clustered are a team's opponents? Low = broad cross-conference scheduling | 4.5 | None (computed from graph structure) |
| **Rolling window stats** | Temporal dynamics of recent performance vs season-long aggregate | 4.4 | Window size (5 / 10 / 20-game); all three are distinct parameter choices of the same building block |
| **EWMA** | Recency-weighted smoothing; alternative to discrete rolling windows | 4.4 | Alpha (α=0.10–0.30); all alpha values are parameters of the same building block |
| **Momentum / trajectory** | Rate of change of efficiency (`ewma_fast − ewma_slow`); improving vs. declining into tournament | 4.4 | Fast and slow alpha values |
| **Streak features** | Pure win/loss count sequences; categorical "hot/cold" dynamics independent of efficiency magnitude | 4.4 | Signed integer vs. separate columns; minimum streak length threshold |
| **Per-possession normalization** | Removes pace confound from all counting stats | 4.4 | `possessions = FGA − OR + TO + 0.44×FTA` (denominator fixed; not a tuning parameter) |
| **Four Factors (eFG%, ORB%, FTR, TO%)** | Derived per-possession efficiency ratios: shooting, rebounding, turnover, free-throw efficiency | 4.4 | None — formulas are fixed by Dean Oliver's definition |
| **Home court encoding (loc)** | Game-level venue context (H/A/N); +2.2pt EDA-confirmed advantage | 4.4 | Numeric (H=+1, A=−1, N=0) or one-hot |
| **Probability calibration** | Output-level correction for favourite-longshot bias in win probability estimates | 4.7 | Method (isotonic / cubic-spline / goto_conversion); in-fold vs. post-hoc |
| **LRMC** | Markov-chain-derived win probabilities — alternative to point-differential-based ratings | 4.6 | Logistic regression hyperparameters |
| **TrueSkill / Glicko-2** | Uncertainty-quantified Elo variant; explicitly models rating variance per team | 4.6 | β (performance variance); τ (rating volatility) |

**Preprocessing steps** (not features themselves, but affect all features they touch):

| Step | Effect | Story |
|:---|:---|:---|
| Overtime rescaling | Normalize OT games to pts/40min equivalent before aggregation | 4.4 |
| Time-decay game weighting | Weight games by recency before rolling aggregations (BartTorvik: 1% per day > 40 days old, floor 60%) | 4.4 |

---

### 7.3 Building Blocks by Story

| Story | Scope | Building Blocks to Implement |
|:---|:---|:---|
| **4.2** | Chronological Data Serving API | Walk-forward game iterator with date guards; 2025 deduplication by `(w_team_id, l_team_id, day_num)`; 2020 COVID flag; OT rescaling preprocessing |
| **4.3** | Canonical ID Mapping & Data Cleaning | `MTeamSpellings.csv` canonical name map; `MNCAATourneySeeds.csv` (seed_num, region, is_play_in); `MTeamConferences.csv` (season, team_id → conf); Massey Ordinals ingestion (all systems with temporal slices); composite building blocks (Options A–D) |
| **4.4** | Sequential Transformations | Rolling windows (5/10/20-game) for all EDA Tier 1 stats; EWMA (configurable α); momentum (`ewma_fast − ewma_slow`); streak features (signed int); loc encoding; per-possession normalization; Four Factors (eFG%, ORB%, FTR, TO%); time-decay weighting |
| **4.5** | Graph Builders & Centrality | PageRank (directed, margin-weighted, warm-start incremental); betweenness centrality; HITS (hub + authority); clustering coefficient |
| **4.6** | Opponent Adjustments / Rating Systems | SRS (Group A representative); Ridge (Group A with λ knob); Colley (Group B, or use pre-computed COL); Elo (dynamic, configurable K/margin/mean-reversion); LRMC; TrueSkill/Glicko-2 |
| **4.7** | Stateful Feature Serving | Feature composition from all 4.2–4.6 outputs; `gender_scope` and `dataset_scope` configurability; temporal slicing for walk-forward backtesting; probability calibration (isotonic or spline, in-fold) |

---

### 7.4 Open Questions for Implementation Stories

These questions cannot be resolved by research alone — they require implementation:

1. **SAG and WLK coverage in local `MMasseyOrdinals.csv`:** Story 4.3 must verify 23-season coverage before committing to the SAG+POM+MOR+WLK composite. Fallback: MOR+POM+DOL (all confirmed full-coverage, all margin-based).
2. **Rolling window sizes:** All three (5/10/20-game) should be implemented as configurable columns — they are parameters of the same building block, not competing features.
3. **EWMA alpha parameter range:** Implement as a configurable parameter. Recommended starting range α=0.15–0.20 (effective window 9–12 games); expose full range α=0.10–0.30 to modelers.
4. **Ridge lambda range:** Implement as a configurable parameter. Recommended starting point λ=20 for full-season data; expose range λ=10–100.
5. **Normalization for new features (Elo, PageRank, Massey delta):** See Normalization Reference table. Validate distributions empirically — approximate normals for deltas → no transform + StandardScaler; PageRank → log + StandardScaler.
6. **Overtime rescaling formula:** `adjusted_score = raw_score × 40 / (40 + 5 × num_ot)`. Apply before any aggregation step in Story 4.2/4.4.

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
