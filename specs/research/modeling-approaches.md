# Modeling Approaches — NCAA Tournament Prediction

**Date:** 2026-02-22
**Story:** 5.1 (Spike) — Research Modeling Approaches
**Status:** Complete — Awaiting Product Owner Review (AC 7)
**Author:** Dev Agent (Claude Opus 4.6)

---

## Quick-Navigation Table

| Section | Topic | Key Takeaway |
|:---|:---|:---|
| [1. Kaggle MMLM Solution Survey](#1-kaggle-mmlm-solution-survey-2014-2025) | Community solutions 2014–2025 | Elo/ratings + gradient boosting most consistent; 538 dominated 2018–2023 |
| [2. Stateful Models](#2-stateful-model-catalogue) | Elo, Glicko-2, TrueSkill | Elo is the proven baseline; Glicko-2/TrueSkill add uncertainty but marginal signal |
| [3. Stateless Models](#3-stateless-model-catalogue) | XGBoost, LightGBM, logistic regression, neural nets | XGBoost is the tabular-data standard; neural nets underperform GBDT on small NCAA data |
| [4. Hybrid Approaches](#4-hybrid-approaches) | Elo-as-feature, ordinal composites, stacking | Elo features → XGBoost is the dominant community pattern |
| [5. Model ABC Interface](#5-model-abc-interface-requirements) | Interface specification for Story 5.2 | Dual-contract ABC: `StatefulModel` (per-game) + `StatelessModel` (batch) |
| [6. Reference Model Recommendations](#6-reference-model-recommendations) | Story 5.3 (Elo) and 5.4 (XGBoost) | Elo as stateful reference, XGBoost as stateless reference |
| [7. Equivalence Groups](#7-model-equivalence-groups) | Which models are redundant vs. distinct | 3 distinct model families; GBDT variants form one equivalence group |
| [8. Scope Recommendation](#8-scope-recommendation-for-po-decision) | MVP options for PO approval | Recommended: 2 reference models + ABC; LightGBM/neural nets deferred |

---

## 1. Kaggle MMLM Solution Survey (2014–2025)

### 1.1 Year-by-Year Top Solution Summary

This survey builds on the community techniques documented in `specs/research/feature-engineering-techniques.md` Section 6 (Story 4.1 spike). That document covered **feature engineering** patterns; this section focuses on **model architectures** and **modeling strategies**.

| Year | Winner | Core Model | Core Signal | Key Modeling Insight |
|:---|:---|:---|:---|:---|
| 2014 | Lopez & Matthews | Logistic regression | Vegas point spreads + KenPom | Market efficiency signal > complex ML; simple model + great features wins |
| 2015 | Zach Bradshaw | Bayesian logistic regression | KenPom + Bayesian domain priors | Informative priors from NBA analytics stabilized small-sample estimates |
| 2016 | Miguel Alomar | Unknown (logistic variant) | Massey ordinals + power ratings | Ordinals as dominant signal; model type mattered less than feature quality |
| 2017 | Andrew Landgraf | Bayesian logistic (rstanarm) | Team efficiency + travel distance | **Meta-strategy**: modeled competitors' submissions to optimize relative placement |
| 2018 | raddar (Barušauskas) | Logistic transformation | FiveThirtyEight 538 ratings | 538 ratings are already a near-optimal feature; a simple model sufficed |
| 2019 | Unknown | Bayesian logistic | KenPom + ORB/TO margin | goto_conversion bias correction earned silver; calibration > model complexity |
| 2020 | — | — | — | Tournament cancelled (COVID-19) |
| 2021 | Unknown (raddar approach) | Logistic / XGBoost | 538 Team Quality | Edwards (mid-pack): 65 Massey systems + XGBoost with Boruta feature selection |
| 2022 | Amir Ghazi | Logistic (raddar code) | 538 Team Quality | Original raddar finished 593rd with a *new* approach; old approach still won |
| 2023 | RustyB / maze508 (gold) | XGBoost (maze508) | 10 Massey systems + seed diff | **Metric change**: Brier Score (2023+), men's + women's combined. maze508: RFE feature selection |
| 2024 | Unknown | Monte Carlo simulation (R) | External ratings + intuitions | Not a pure ML approach; probabilistic tournament simulation |
| 2025 | Mohammad Odeh | XGBoost (Optuna-tuned) | Seed diff + Team Quality + efficiency | ~23 features; cubic-spline in-fold calibration; XGBoost outperformed CatBoost and LightGBM |

### 1.2 Recurring Patterns Across Winning Solutions

**Pattern 1: Feature quality dominates model complexity.**
The most consistent finding across 11 years of competition is that the quality of input signals (538 ratings, Massey ordinals, Elo ratings, seed difference) matters far more than the choice of model architecture. Simple logistic regression with great features won or nearly won in 2014, 2015, 2017, 2018, 2019, 2021, and 2022.

**Pattern 2: Elo/rating features + gradient boosting = most consistent approach.**
When features are constructed from raw game data (rather than pre-computed external ratings like 538), the dominant pattern is:
1. Compute team-level ratings (Elo, SRS, Massey ordinals)
2. Construct matchup-level deltas (team_A rating − team_B rating)
3. Feed deltas + seed info into XGBoost/LightGBM
4. Calibrate output probabilities

**Pattern 3: Probability calibration is a first-class concern.**
Multiple silver medals were won via calibration improvements alone (goto_conversion in 2019, 2021). The 2025 winner used cubic-spline in-fold calibration. Since the competition metric is a proper scoring rule (LogLoss 2014–2022, Brier 2023+), well-calibrated probabilities directly improve placement.

**Pattern 4: The 538 era is over.**
FiveThirtyEight's NCAA ratings powered winning solutions from 2018–2023, but the 538 NCAA model was discontinued. From 2024+, competitors must build their own rating systems or use Massey ordinals as the equivalent consensus signal. This project's feature pipeline (Epic 4) already provides the building blocks.

**Pattern 5: Luck plays a significant role in tournament prediction.**
Matthews & Lopez (2014 winners) simulated the tournament 10,000 times and estimated their winning entry had only ~12% probability of winning the competition even under the most optimistic modeling assumptions. This highlights that single-tournament evaluation is noisy — multi-year backtesting (Leave-One-Tournament-Out, Story 6.2) is essential for reliable model comparison.

**Pattern 6: Competition-specific quirks shape strategy.**
- **2023+: Brier Score** replaced LogLoss, changing optimal calibration targets
- **2023+: Men's + Women's combined** doubled the submission size and added a gender dimension
- **2017: Meta-strategy** (Landgraf modeled competitors' submissions) — a competition-specific trick, not transferable to pure prediction

### 1.3 Model Type Frequency in Top Solutions

| Model Type | Frequency in Top-10 Solutions (est.) | Best Year Placement |
|:---|:---|:---|
| Logistic regression (standard/Bayesian) | Very high (2014–2022) | 1st (2014, 2015, 2017, 2018) |
| XGBoost | High (2019–2025) | 1st (2025), Gold (2023) |
| LightGBM / CatBoost | Moderate (2021–2025) | Top-10 (various) |
| Monte Carlo simulation | Low (2024) | 1st (2024) |
| Neural networks (LSTM, Transformer) | Low (research, not competition winners) | N/A |
| Random forest / SVM | Low (early years) | Top-25 |
| Ensemble / stacking | Moderate (2019+) | Top-10 (various) |

### 1.4 Competition-Specific Quirks

| Year | Quirk | Impact on Modeling |
|:---|:---|:---|
| 2014–2022 | Metric: LogLoss | Optimize for log-likelihood; penalizes overconfident wrong predictions |
| 2023+ | Metric: Brier Score | Optimize for calibration; less harsh on confident predictions vs. LogLoss |
| 2023+ | Men's + Women's combined | Need two sets of features/models or gender-aware unified model |
| 2024+ | FiveThirtyEight NCAA ratings discontinued | Must build own rating systems (Elo, SRS) or use Massey ordinals; this project's Epic 4 pipeline already provides equivalent building blocks |
| 2020 | COVID — no tournament | Training data gap; models must handle missing evaluation year |
| 2017 | Game-theoretic meta-strategy | Competition-only insight; Landgraf modeled other participants' submissions |

---

## 2. Stateful Model Catalogue

Stateful models maintain internal state (team ratings) that evolves game-by-game across a season. They process games sequentially and produce predictions from the accumulated state.

### 2.1 Elo Rating System

**Status in this project:** Already implemented as a *feature building block* (`EloFeatureEngine` in `transform.elo`, Story 4.8). Story 5.3 wraps it as a *predictive model* via the Model ABC.

**Core mechanics:**
- Update rule: `r_new = r_old + K × (actual − expected)`
- Expected score: `expected = 1 / (1 + 10^((r_opponent − r_team) / 400))`
- **Prediction**: Win probability = `expected_score(r_team_a, r_team_b)` directly from rating difference

**Variants implemented (EloConfig parameters):**

| Parameter | Default | Range | Description |
|:---|:---|:---|:---|
| `k_early` | 56.0 | 40–70 | K-factor for first 20 games (high uncertainty) |
| `k_regular` | 38.0 | 25–50 | K-factor for regular season |
| `k_tournament` | 47.5 | 35–60 | K-factor for tournament games |
| `margin_exponent` | 0.85 | 0.6–1.0 | Silver/SBCB margin-of-victory scaling |
| `max_margin` | 25 | 15–35 | Cap on margin before scaling |
| `home_advantage_elo` | 3.5 | 2.0–5.0 | Elo points subtracted for home-court |
| `mean_reversion_fraction` | 0.25 | 0.15–0.40 | Between-season regression toward conference mean |

**Competition validation:** Elo ratings (or close variants) appear as Tier 2 features in virtually every top-10 MMLM solution (see feature-engineering-techniques.md Section 6.5). K=38–42, season mean-reversion 20–35%.

**Distinction: Feature Elo vs. Model Elo:**

| Aspect | Feature Elo (Story 4.8) | Model Elo (Story 5.3) |
|:---|:---|:---|
| Purpose | Compute ratings as INPUT features for other models | Use ratings directly as a PREDICTIVE MODEL |
| Interface | `EloFeatureEngine.update_game()` → before-ratings | `Model.predict(team_a, team_b)` → win probability |
| Output | `elo_w_before`, `elo_l_before` columns in feature DataFrame | Calibrated probability `P(team_a wins)` |
| ABC conformance | No — it's a transform building block | Yes — implements `StatefulModel` ABC (Story 5.2) |
| Calibration | N/A (raw ratings) | May need post-hoc calibration (isotonic/sigmoid) |

### 2.2 Glicko-2 Rating System

**Description:** Extension of Elo with explicit uncertainty quantification. Each team has three parameters: rating (μ), rating deviation (RD), and volatility (σ).

**Core mechanics:**
- Rating update is Elo-like but weighted by opponent's RD (higher RD = less informative game)
- RD increases during inactivity (off-season), decreases with more games played
- Volatility captures how consistently a team performs relative to their rating

**What Glicko-2 adds beyond Elo:**
- **Rating deviation (RD):** A confidence interval on the team's true strength. New teams or teams that haven't played recently have high RD (high uncertainty). This is genuinely distinct information — it tells you *how much you trust* a rating, not just the rating itself.
- **Volatility (σ):** Measures how erratic a team's performance is. A team with high volatility produces more upsets. This could be a useful feature for tournament prediction (volatile teams are higher-variance bets).

**Implementation options:**
- Python `glicko2` package (PyPI) — pure Python, minimal dependencies
- Custom implementation — Glicko-2 algorithm is well-documented (Glickman 2012)

**Competition evidence:** Appears occasionally in top-25 MMLM solutions (Edwards 2021 documented TrueSkill, a cousin). No documented case of Glicko-2 outperforming well-tuned Elo in MMLM specifically. The marginal signal (RD, volatility) over Elo is expected to be small for full-season snapshots where all teams have played 30+ games (RD converges, reducing its informational value).

**Assessment:**

| Factor | Assessment |
|:---|:---|
| Distinct signal beyond Elo | Low-Medium — RD and volatility are genuinely new parameters, but converge for full-season data |
| Implementation cost | Low — `glicko2` PyPI package or ~100 lines of Python |
| Competition validation | Weak — not in winning solutions; occasional top-25 |
| **Recommendation** | **Post-MVP** — defer to Post-MVP Backlog; implement as a Model ABC plugin when time permits |

### 2.3 TrueSkill Rating System

**Description:** Microsoft's Bayesian rating system using factor graphs and Gaussian belief propagation. Each team has a Gaussian belief (μ, σ²) for their skill level.

**What TrueSkill adds beyond Elo:**
- Handles draws natively (not relevant for basketball)
- Convergence is mathematically guaranteed (factor graph inference)
- σ² provides a confidence estimate similar to Glicko-2's RD
- Originally designed for Xbox Live matchmaking; adapted for sports analytics

**Implementation:** `trueskill` PyPI package (maintained, Python 3 compatible).

**Competition evidence:** Edwards 2021 documented TrueSkill in his comprehensive feature pipeline but it did not stand out as a top feature. Occasional appearances in top-25 solutions.

**Assessment:**

| Factor | Assessment |
|:---|:---|
| Distinct signal beyond Elo | Low — similar to Glicko-2; uncertainty parameter converges for full-season data |
| Implementation cost | Low — `trueskill` PyPI package |
| Competition validation | Weak — not in winning solutions |
| **Recommendation** | **Post-MVP** — same rationale as Glicko-2 |

### 2.4 Custom Rating Systems Found in Community Solutions

**LRMC (Logistic Regression Markov Chain):**
Models tournament outcomes as a Markov chain where transition probabilities are derived from game-by-game logistic regression. Documented in Edwards 2021. Distinct from Elo (Bayesian updating) and SRS (batch solve). Higher implementation complexity. Deferred to Post-MVP.

**Mixed-Effects Model Ratings:**
Edwards 2021 used mixed-effects models (lme4 equivalent) to compute team ratings. This treats team strength as a random effect and game outcomes as observations. Similar signal to SRS but with a proper statistical framework for uncertainty quantification. Python implementation via `statsmodels` or custom. Deferred to Post-MVP.

### 2.5 Stateful Model Comparison Summary

| Model | Distinct Signal | Competition Evidence | Implementation Cost | MVP? |
|:---|:---|:---|:---|:---|
| **Elo** | Dynamic ratings, in-season trajectory, configurable K/margin/HCA | Tier 2 across all years; proven baseline | Already implemented (4.8) | **Yes — Reference (Story 5.3)** |
| Glicko-2 | Rating uncertainty (RD), performance volatility (σ) | Occasional top-25 | Low (PyPI package) | Post-MVP |
| TrueSkill | Skill uncertainty (σ²), factor graph inference | Occasional top-25 | Low (PyPI package) | Post-MVP |
| LRMC | Markov chain win probabilities | Edwards 2021 | Medium | Post-MVP |
| Mixed-effects | Random-effect team strength with uncertainty | Edwards 2021 | Medium | Post-MVP |

---

## 3. Stateless Model Catalogue

Stateless models perform batch training on a feature matrix `(X, y)` and batch prediction. They do not maintain per-game state.

### 3.1 XGBoost (Gradient Boosted Decision Trees)

**Competition dominance:** XGBoost is the most successful model architecture in Kaggle MMLM from 2019–2025:
- 2025 1st place (Odeh): XGBoost with Optuna hyperparameter tuning
- 2023 gold (maze508): XGBoost with Recursive Feature Elimination
- 2021 (Edwards, documented): XGBoost with Boruta feature selection

**Why XGBoost works for NCAA prediction:**
- Tabular data with heterogeneous features (continuous ratings, ordinal seeds, categorical conference) — this is XGBoost's strongest domain
- Handles missing values natively (important for pre-2003 seasons lacking detailed box scores)
- Feature importance provides interpretability
- Regularization (L1, L2, max_depth) prevents overfitting on small tournament datasets
- Native handling of imbalanced data (though NCAA matchups are not imbalanced by construction)

**Hyperparameter ranges from competition solutions:**

| Parameter | Range | Notes |
|:---|:---|:---|
| `n_estimators` | 100–1000 | Tune via early stopping on validation fold |
| `max_depth` | 3–8 | Deeper trees risk overfitting on ~2,000 tournament games |
| `learning_rate` | 0.01–0.1 | Lower rate + more estimators = better generalization |
| `subsample` | 0.6–0.9 | Row sampling per tree |
| `colsample_bytree` | 0.5–0.9 | Feature sampling per tree |
| `min_child_weight` | 1–10 | Regularization against sparse splits |
| `reg_alpha` (L1) | 0–1.0 | Sparsity-inducing regularization |
| `reg_lambda` (L2) | 0.5–5.0 | Ridge regularization |
| `scale_pos_weight` | 1.0 | NCAA matchups are balanced (always one winner, one loser) |
| `objective` | `binary:logistic` | Outputs probability directly |

**Feature selection approaches used with XGBoost in MMLM:**
- Boruta algorithm (Edwards 2021) — shadow features as baseline
- Recursive Feature Elimination (maze508 2023) — iterative removal
- Optuna-based feature pruning (Odeh 2025) — Bayesian search
- Manual selection based on domain knowledge + feature importance plots

**Calibration:** XGBoost with `binary:logistic` objective outputs probabilities, but they may not be perfectly calibrated. The 2025 winner applied cubic-spline in-fold calibration. Our project already implements `IsotonicCalibrator` and `SigmoidCalibrator` in `transform.calibration` (Story 4.7).

**Assessment:**

| Factor | Assessment |
|:---|:---|
| Distinct signal | Standard GBDT — captures non-linear feature interactions |
| Competition validation | **Strong** — 1st place 2025, gold 2023, top-10 consistently |
| Implementation cost | Low — `xgboost.XGBClassifier` already in dependencies |
| **Recommendation** | **Yes — Reference stateless model (Story 5.4)** |

### 3.2 LightGBM

**How it differs from XGBoost:**
- Leaf-wise tree growth (vs. XGBoost's level-wise) — faster training, potentially deeper trees
- Gradient-based one-side sampling (GOSS) — reduces training time by focusing on high-gradient samples
- Exclusive Feature Bundling (EFB) — reduces effective feature count for sparse data
- Native categorical feature support (no one-hot encoding needed)

**Competition evidence:** Used alongside XGBoost in many top solutions. The 2025 winner explicitly noted XGBoost outperformed both CatBoost and LightGBM in his experiments.

**When LightGBM is preferred over XGBoost:**
- Very large datasets (>100K samples) — LightGBM's leaf-wise growth is faster
- Many categorical features — native categorical handling avoids encoding overhead
- For NCAA (~5,000 games/year, ~2,000 tournament games total): the dataset is small enough that XGBoost and LightGBM perform similarly

**Assessment:**

| Factor | Assessment |
|:---|:---|
| Distinct signal beyond XGBoost | Low — same GBDT family; minor algorithmic differences |
| Competition validation | Moderate — top-10 but outperformed by XGBoost in 2025 |
| Implementation cost | Low — `lightgbm.LGBMClassifier` |
| **Recommendation** | **Post-MVP** — implement as a third Model ABC plugin; low priority since XGBoost covers the GBDT family |

### 3.3 CatBoost

**How it differs from XGBoost:**
- Ordered boosting — uses permutation-driven approach to prevent target leakage
- Native categorical feature handling with target statistics
- Symmetric tree structure — faster inference

**Competition evidence:** Odeh (2025 winner) tested CatBoost and found it underperformed XGBoost.

**Assessment:**

| Factor | Assessment |
|:---|:---|
| Distinct signal beyond XGBoost | Low — same GBDT family |
| Competition validation | Moderate — tested but not winning |
| **Recommendation** | **Post-MVP** — same rationale as LightGBM |

### 3.4 Logistic Regression

**Competition dominance (2014–2022):** Logistic regression (standard or Bayesian) won more MMLM competitions than any other model type, primarily because the winning signal was external pre-computed ratings (538, KenPom) that were already near-optimal. A simple model that doesn't overfit works better than a complex model when features are strong and data is limited.

**Variants used in MMLM:**

| Variant | Description | Competition Example |
|:---|:---|:---|
| Standard LR | `sklearn.linear_model.LogisticRegression` with L2 penalty | 2014 (Lopez) |
| Bayesian LR | `rstanarm` / `pymc` with informative priors | 2015 (Bradshaw), 2017 (Landgraf) |
| Regularized LR | L1 (Lasso) or Elastic Net for feature selection | Various top-25 |

**When logistic regression is best:**
- Few, strong features (seed diff, ordinal delta, Elo delta) — LR captures the linear relationship without overfitting
- Very small training sets (tournament-only data: ~2,000 games)
- As a baseline to compare against more complex models

**Bayesian logistic regression specifics (Landgraf 2017):**
- Used `rstanarm::stan_glm()` with Student-t priors
- Priors informed by NBA analytics literature
- Posterior predictive distributions provide natural uncertainty quantification
- Python equivalent: `pymc` or `bambi` (Bayesian model-building interface for PyMC)

**Assessment:**

| Factor | Assessment |
|:---|:---|
| Distinct signal | Linear relationships only — complementary to XGBoost's non-linear interactions |
| Competition validation | **Strong** — won 2014–2018 competitions |
| Implementation cost | Trivial — `sklearn.linear_model.LogisticRegression` |
| **Recommendation** | **Not a separate reference model** — include as a built-in option in Story 5.4 or as a minimal Model ABC example in Story 5.2 |

### 3.5 Neural Networks (LSTM, Transformer)

**Academic research (arXiv:2508.02725, Habib 2025):**
A 2025 arXiv paper compared LSTM and Transformer architectures for NCAA tournament prediction.

> ⚠️ **Verification note:** arXiv:2508.02725 is dated August 2025, near the LLM knowledge cutoff. The specific AUC values, architecture details, and feature ablation numbers cited below are from training knowledge and have **not been live-verified** against the actual paper PDF. Before using these numbers as authoritative benchmarks in downstream implementation or evaluation, the implementer should retrieve the paper at `https://arxiv.org/abs/2508.02725` and confirm the reported values.

| Architecture | Loss Function | AUC | Calibration | Notes |
|:---|:---|:---|:---|:---|
| Transformer | BCE | **0.8473** (highest) | Weaker | Highest discriminative accuracy but poor calibration |
| LSTM | Brier | 0.8320 | **Best** | Superior probability calibration; better for scoring rules |
| LSTM | BCE | 0.8392 | Moderate | Standard cross-entropy training |
| Transformer | Brier | 0.8156 | Good | Lower accuracy than BCE variant |

**Features used:** GLM team quality metrics, Elo ratings, seed differences, box-score stats — similar to the tabular features this project already computes.

**Architecture details (arXiv:2508.02725):**
- LSTM: 32 hidden units, dropout 0.5, dense 16 ReLU units, Adam optimizer (lr=1e-3), batch size 128, early stopping (patience=10)
- Transformer: multi-head attention (2 heads, d=64), 2 feedforward layers (64 units), dropout 0.5, Adam (lr=1e-4), 100 epochs
- Feature ablation: GLM quality most impactful (removing costs −0.049 AUC), Elo second (−0.045 AUC), box scores moderate (−0.021 AUC), seed diff smallest (−0.012 AUC)

**Deep learning in competition (non-winning):**
- **Forseth (2017, 4th place):** Used a neural network (Theano) trained on raw data alongside a logistic regression on modified Massey ratings. Averaged both model predictions. Unperturbed predictions would have placed ~25th (top 5%), proving both model quality and strategy matter.
- **Combinatorial Fusion Analysis (Alfatemi et al., 2024):** Four diverse neural architectures (CNN, RNN, feedforward, residual) combined via CFA to merge predictions.
- **GitHub implementations:** Multiple educational repos using simple feedforward and LSTM networks.

**Why neural networks underperform for NCAA prediction in competitions:**
1. **Small data:** ~2,000 tournament games since 2003. Neural networks excel with millions of samples; GBDT is the standard for small tabular datasets.
2. **Tabular features:** The input is structured tabular data (ratings, seeds, box scores), not sequential images or text. GBDT is the established winner for tabular data (see "Why do tree-based models still outperform deep learning on tabular data?" — Grinsztajn et al. 2022, NeurIPS).
3. **Training complexity:** Neural networks require careful architecture design, learning rate scheduling, and regularization. XGBoost with default parameters is already competitive.
4. **No competition wins:** No MMLM competition has been won by a neural network approach.

**Assessment:**

| Factor | Assessment |
|:---|:---|
| Distinct signal | Sequential temporal patterns (LSTM) — potentially captures game-sequence dynamics not in tabular features |
| Competition validation | **Weak** — academic research only; no competition wins |
| Implementation cost | Medium-High — requires PyTorch/TensorFlow, custom training loop, architecture design |
| **Recommendation** | **Post-MVP** — LSTM as an experimental model for users who want to explore temporal patterns |

### 3.6 Random Forest / SVM

**Random Forest:** Occasionally used in early MMLM competitions (2014–2016). Outperformed by XGBoost in nearly all documented comparisons. Not recommended as a reference model; trivially implementable as a Model ABC plugin via `sklearn.ensemble.RandomForestClassifier` if desired.

**SVM:** Rare in MMLM. Kernel SVM does not scale well and provides no probability output without Platt scaling. Not recommended.

**Assessment:** Both are **Post-MVP** — easy to add as Model ABC plugins but do not provide distinct signal or competitive advantage.

### 3.7 Stateless Model Comparison Summary

| Model | Distinct Signal | Competition Evidence | Cost | MVP? |
|:---|:---|:---|:---|:---|
| **XGBoost** | Non-linear feature interactions, native missing values | **1st** (2025), Gold (2023), top consistently | Low | **Yes — Reference (Story 5.4)** |
| LightGBM | Same GBDT; faster on large data | Top-10; outperformed by XGBoost in 2025 | Low | Post-MVP |
| CatBoost | Same GBDT; better categorical handling | Top-10; outperformed by XGBoost in 2025 | Low | Post-MVP |
| Logistic Regression | Linear relationships; regularization | **1st** (2014–2018) | Trivial | Story 5.2 example, not separate reference |
| LSTM / Transformer | Temporal sequence patterns | Academic only; no competition wins | High | Post-MVP |
| Random Forest | Bagged decision trees | Early years; outperformed by XGBoost | Low | Post-MVP |

---

## 4. Hybrid Approaches

Hybrid approaches combine stateful and stateless models, using the output of one as input to another.

### 4.1 Elo-as-Feature Pattern (Primary Hybrid)

**This is the single most important hybrid pattern for this project.**

**Architecture:**
1. `EloFeatureEngine` (Story 4.8) processes games chronologically → produces `elo_w_before`, `elo_l_before` per game
2. `StatefulFeatureServer` (Story 4.7) includes Elo ratings in the feature matrix alongside sequential stats, graph features, batch ratings, ordinals, and seeds
3. XGBoost (Story 5.4) trains on the full feature matrix, with `delta_elo` as one of many matchup-level features

**Already implemented in this project:** The `StatefulFeatureServer.serve_season_features()` output includes `delta_elo` as a matchup feature column. The XGBoost model in Story 5.4 will consume this naturally.

**Competition validation:** This is exactly the pattern used by the most successful MMLM solutions. Edwards 2021, maze508 2023, and Odeh 2025 all used Elo (or similar rating) deltas as features in XGBoost.

### 4.2 Ordinal Composite Features + Gradient Boosting

**Pattern:** Compute deltas from multiple external rating systems → feed into XGBoost.

**maze508 (2023 gold) approach:**
1. Select top-10 most predictive Massey ordinal systems
2. Compute `delta_system = team_a_rank − team_b_rank` for each system
3. Add box-score aggregates and win rates
4. Train XGBoost with Recursive Feature Elimination

**Already in this project:** `StatefulFeatureServer` produces `delta_ordinal_composite`, `delta_srs`, `delta_ridge`, `delta_colley` — all matchup-level deltas that XGBoost can consume.

### 4.3 Stacking / Blending Architectures

**Description:** Use outputs of multiple base models as meta-features for a second-level model.

**Typical stacking architecture for MMLM:**
```text
Level 0 (base models):
  - Elo model → P(win)_elo
  - XGBoost model → P(win)_xgb
  - Logistic regression → P(win)_lr

Level 1 (meta-model):
  - Input: [P(win)_elo, P(win)_xgb, P(win)_lr, seed_diff, ...]
  - Model: Logistic regression (simple meta-learner)
  - Output: P(win)_final
```

**Competition evidence:** Used in top-10 solutions occasionally. Not a consistent winner — the marginal benefit of stacking is small when the base models are trained on the same features and same data. Most useful when base models use genuinely different feature sets or modeling paradigms.

**Model ABC implication:** The ABC must support:
- Multiple model instances running independently
- Extracting predictions from each model
- A meta-model that consumes other models' predictions

This is naturally supported by the proposed dual-contract ABC — any `Model` can consume the predictions of other `Model` instances as features.

### 4.4 Game-Theory / Meta-Modeling

**Landgraf 2017 approach:** Instead of maximizing prediction accuracy, Landgraf modeled *what other competitors would submit* via a mixed-effects model. He then found the submission that maximized his probability of finishing in the top-5, rather than the submission with the lowest expected loss.

**This is a competition-specific meta-strategy, not a modeling technique.** It does not improve the quality of predictions; it optimizes for relative placement in a leaderboard. Not applicable to this project's goals (building the best possible NCAA prediction system).

**Assessment:** Document for completeness but **exclude from Model ABC scope**. This is a competition strategy, not a model architecture.

---

## 5. Model ABC Interface Requirements

### 5.1 Design Principles

The Model ABC must support the following requirements derived from the survey:

1. **Unified fit interface, specialised lifecycle:** Both model types expose `fit(X, y)` — the standard sklearn-style entry point. `StatefulModel` provides a *concrete* template implementation that reconstructs `Game` objects from `X` and iterates them sequentially via an abstract `update()` hook; `StatelessModel` leaves `fit(X, y)` abstract for batch training. The two-subclass split is justified by lifecycle methods specific to stateful models (`start_season`, `get_state`, `set_state`) that have no meaning for stateless models. **Use two abstract base classes** under a common parent.

2. **Calibrated probability output:** All models must output calibrated probabilities, not raw scores. This is required because the evaluation metrics (Brier Score, LogLoss) are proper scoring rules that reward calibration. Models can either:
   - Output calibrated probabilities directly (e.g., Elo expected score)
   - Output raw predictions that the caller calibrates via `transform.calibration`
   - The ABC should NOT enforce a specific calibration approach — leave it to the implementation.

3. **Persistence:** All models must support `save()` and `load()`. The format varies by model type:
   - Stateful models: serialize rating state (dict of team_id → rating) + config
   - Stateless models: serialize trained model artifact (XGBoost native JSON, joblib for sklearn)
   - Hyperparameters: always serializable to JSON (Pydantic `BaseModel`)

4. **Plugin registry:** Models register by name for runtime discovery. This enables:
   - CLI: `--model elo` or `--model xgboost`
   - Configuration files that reference models by name
   - Dynamic model loading without import chains

5. **Scikit-learn compatibility (partial):** The ABC should follow sklearn conventions where practical:
   - `fit(X, y)` / `predict(X)` naming convention
   - `get_params()` / `set_params()` for hyperparameter introspection
   - But: NOT full sklearn `BaseEstimator` inheritance — NCAA-specific requirements (walk-forward, tournament filtering, per-game state) don't fit sklearn's assumptions

### 5.2 Proposed Interface Specification

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel as PydanticBaseModel

from ncaa_eval.ingest.schema import Game  # noqa: TC001 — used in StatefulModel.update() signature


class ModelConfig(PydanticBaseModel):
    """Base config validated by Pydantic. Subclassed per model."""
    model_name: str


class Model(ABC):
    """Common parent for all NCAA prediction models."""

    @abstractmethod
    def predict(self, team_a_id: int, team_b_id: int) -> float:
        """Return P(team_a wins) as a calibrated probability in [0, 1]."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model state/artifacts to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "Model":
        """Restore a model from disk."""

    @abstractmethod
    def get_config(self) -> ModelConfig:
        """Return the model's configuration (Pydantic-validated)."""


class StatefulModel(Model):
    """Model that maintains per-team state updated game-by-game.

    Exposes a concrete fit(X, y) that reconstructs Game objects from X and
    iterates them in chronological order via the update() hook. X must contain
    raw game columns (not pre-engineered features) so that Game objects can be
    faithfully reconstructed — see _to_games() for required columns.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Concrete template: reconstruct Games from X, iterate sequentially.

        Calls start_season() at each season boundary. X rows must be sorted
        chronologically (ascending by date/day_num).
        """
        games = self._to_games(X, y)
        current_season: int | None = None
        for game in games:
            if game.season != current_season:
                self.start_season(game.season)
                current_season = game.season
            self.update(game)

    def _to_games(self, X: pd.DataFrame, y: pd.Series) -> list[Game]:
        """Reconstruct ordered Game objects from raw game DataFrame.

        Required columns in X: w_team_id, l_team_id, season, day_num, date,
        loc, num_ot (and any score columns needed by the concrete model).
        Rows must already be sorted chronologically.
        """
        ...  # concrete implementation — reconstruct from standard schema

    @abstractmethod
    def update(self, game: Game) -> None:
        """Process one game and update internal state."""

    @abstractmethod
    def start_season(self, season: int) -> None:
        """Prepare for a new season (e.g., mean reversion)."""

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Return serializable snapshot of internal state."""

    @abstractmethod
    def set_state(self, state: dict[str, Any]) -> None:
        """Restore internal state from a snapshot."""


class StatelessModel(Model):
    """Model that trains on a batch feature matrix."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit model on feature matrix X and labels y."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Return P(team_a wins) for each row in X."""

    # NOTE: StatelessModel satisfies the Model.predict() abstract method
    # by providing a concrete delegation that builds a single-row DataFrame.
    # Implementations that need per-matchup prediction without a full feature
    # matrix should override this method; the default is a convenience wrapper.
    def predict(self, team_a_id: int, team_b_id: int) -> float:
        """Not the primary API for stateless models — use predict_proba(X).

        This implementation raises NotImplementedError by default; stateless
        models are designed for batch prediction via predict_proba(). Override
        if your model supports single-matchup lookup without a feature matrix.
        """
        raise NotImplementedError(
            "StatelessModel.predict() requires a feature matrix. "
            "Call predict_proba(X) with a pre-built feature row instead."
        )
```

### 5.3 Interface Design Rationale

**Why two ABC subclasses (not one):**

| Consideration | Single ABC | Dual ABC (proposed) |
|:---|:---|:---|
| `fit(X, y)` | Same signature — a single ABC *could* work | `StatefulModel.fit()` is concrete (template method); `StatelessModel.fit()` is abstract — both work cleanly |
| Stateful lifecycle hooks (`start_season`, `get_state`, `set_state`) | Must raise `NotImplementedError` in stateless models | Only on `StatefulModel`; stateless models don't see them |
| `update(game)` hook | Must raise `NotImplementedError` in stateless models | Only on `StatefulModel`; stateless models don't see it |
| Type safety | Caller must guard against missing lifecycle methods | Type system prevents calling stateful lifecycle methods on stateless models |
| Evaluation pipeline `fit` call | Single dispatch — both use `fit(X, y)`, no branching needed | Both use `fit(X, y)` — no branching needed for training |
| **Verdict** | Viable for `fit`, but lifecycle methods still pollute the interface | **Preferred** — lifecycle methods properly isolated |

**Why NOT full sklearn BaseEstimator:**

| sklearn Convention | NCAA Requirement | Conflict? |
|:---|:---|:---|
| `fit(X, y)` | Stateful models process games one at a time, not as a matrix | **Resolved** — `StatefulModel.fit(X, y)` reconstructs `Game` objects from `X` internally and iterates sequentially |
| `predict(X)` returns labels | We need probabilities, not labels | **Minor** — use `predict_proba` convention |
| `get_params()`/`set_params()` | Need Pydantic-validated configs | **Compatible** — implement via Pydantic model |
| `clone()` via constructor introspection | We have Pydantic configs that can reconstruct | **Compatible** — implement via config |
| Pipeline integration | Walk-forward temporal logic doesn't fit in `Pipeline.fit()` | **Yes** — custom evaluation loop required |

**Conclusion:** Adopt sklearn **naming conventions** (`fit`, `predict`, `predict_proba`, `get_params`) but NOT sklearn `BaseEstimator` inheritance. The `fit(X, y)` interface is unified across both model types — `StatefulModel` handles game reconstruction internally. The evaluation pipeline (Epic 6) will handle walk-forward logic.

**How the evaluation pipeline (Epic 6) dispatches across both model types:**

The evaluation pipeline must generate predictions for every matchup in a test season. Since `StatefulModel` and `StatelessModel` have different prediction APIs, the pipeline dispatches on ABC type:

```python
from __future__ import annotations

import pandas as pd

from ncaa_eval.transform.feature_serving import StatefulFeatureServer


def generate_predictions(
    model: Model,
    test_games: pd.DataFrame,
    feature_server: StatefulFeatureServer,
    season: int,
) -> pd.Series:
    """Return P(team_a wins) for every game in test_games."""
    if isinstance(model, StatefulModel):
        # Stateful: predict per-game from in-memory ratings (no feature matrix needed)
        return pd.Series(
            [model.predict(row.team_a_id, row.team_b_id) for row in test_games.itertuples()],
            index=test_games.index,
        )
    elif isinstance(model, StatelessModel):
        # Stateless: build feature matrix, predict in batch
        X = feature_server.serve_season_features(season, mode="tournament")
        X_test = X.loc[test_games.index]
        return model.predict_proba(X_test)
    else:
        raise TypeError(f"Unknown model type: {type(model)}")
```

*Note: The `itertuples()` call in the stateful branch is acceptable here — this is evaluation/orchestration code (a side-effect boundary), not business logic. Stateful models inherently require sequential per-game access, so vectorization is not applicable at this layer.*

### 5.4 Plugin Registry Requirements

**Registry design:**

```python
from __future__ import annotations

from collections.abc import Callable

_MODEL_REGISTRY: dict[str, type[Model]] = {}


def register_model(name: str) -> Callable[[type[Model]], type[Model]]:
    """Decorator to register a model class by name."""
    def decorator(cls: type[Model]) -> type[Model]:
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name: str) -> type[Model]:
    """Look up a registered model class by name."""
    ...


def list_models() -> list[str]:
    """Return all registered model names."""
    ...
```

**Usage pattern:**
```python
@register_model("elo")
class EloModel(StatefulModel):
    ...

@register_model("xgboost")
class XGBoostModel(StatelessModel):
    ...

# Runtime discovery
model_cls = get_model("elo")
model = model_cls(config=EloModelConfig(...))
```

**Auto-registration:** Models in `src/ncaa_eval/model/` are auto-registered when the package is imported. No manual registry maintenance required.

### 5.5 Hyperparameter Configuration Schema

All model configs extend `ModelConfig` (Pydantic `BaseModel`):

```python
class EloModelConfig(ModelConfig):
    model_name: str = "elo"
    initial_rating: float = 1500.0
    k_early: float = 56.0
    k_regular: float = 38.0
    k_tournament: float = 47.5
    early_game_threshold: int = 20  # Games before K transitions from k_early → k_regular
    margin_exponent: float = 0.85
    max_margin: int = 25
    home_advantage_elo: float = 3.5
    mean_reversion_fraction: float = 0.25

class XGBoostModelConfig(ModelConfig):
    model_name: str = "xgboost"
    n_estimators: int = 500
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3  # Regularization for small NCAA datasets
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50
```

Benefits:
- JSON serialization/deserialization via Pydantic
- Automatic validation (type checking, value ranges)
- Schema documentation via Pydantic's JSON Schema export
- CLI integration: configs can be loaded from YAML/JSON files

### 5.6 Prediction Output Contract

All models output **calibrated probabilities**:

| Output | Type | Range | Description |
|:---|:---|:---|:---|
| `predict(team_a, team_b)` | `float` | `[0.0, 1.0]` | P(team_a wins) |
| `predict_proba(X)` (stateless) | `pd.Series` | `[0.0, 1.0]` per row | P(team_a wins) for each matchup row |

**Calibration responsibility:**
- Models MAY output raw probabilities and rely on the evaluation pipeline to calibrate
- Models MAY include built-in calibration (e.g., CalibratedClassifierCV wrapping XGBoost)
- The evaluation pipeline (Epic 6) applies isotonic/sigmoid calibration as a configurable post-processing step

### 5.7 Persistence Format

| Model Type | Format | Library | Rationale |
|:---|:---|:---|:---|
| Stateful (Elo) | JSON | `json` (stdlib) | Ratings are a simple dict; JSON is human-readable and version-stable |
| Stateless (XGBoost) | UBJSON | `xgboost.save_model()` | XGBoost native format; backward-compatible across versions |
| Stateless (sklearn) | joblib | `joblib.dump()` | sklearn convention; efficient for numpy arrays |
| Config (all models) | JSON | Pydantic `.model_dump_json()` | Human-readable; schema-validated on load |

**XGBoost persistence notes (from XGBoost docs):**
- `save_model()` saves trees + objective — stable across versions
- Pickle/joblib saves full training state — NOT stable across versions
- **Recommendation:** Use `xgboost.Booster.save_model()` for production persistence; joblib for short-term checkpointing only
- Always save config separately as JSON alongside the model artifact

---

## 6. Reference Model Recommendations

### 6.1 Stateful Reference: Elo (Story 5.3)

**Rationale:**
1. **Already implemented as a feature engine** (Story 4.8) — wrapping as a Model ABC plugin is minimal additional work
2. **Proven competition baseline** — Elo ratings are Tier 2 features in virtually every top MMLM solution
3. **Well-understood mathematics** — the expected score formula directly produces win probabilities
4. **Configurable** — K-factor, margin scaling, home-court, mean-reversion are all tunable hyperparameters
5. **Template value** — demonstrates the stateful model contract (per-game update, season management, state persistence)

**Implementation approach for Story 5.3:**
- Create `EloModel(StatefulModel)` that wraps `EloFeatureEngine` from `transform.elo`
- `predict(team_a, team_b)` → `EloFeatureEngine.expected_score(r_a, r_b)`
- `update(game)` → `EloFeatureEngine.update_game(...)`
- `start_season(season)` → `EloFeatureEngine.start_new_season(season)`
- `save(path)` → JSON dump of ratings dict + config to `path`
- `load(path)` → reconstruct from JSON at `path` (class method)

### 6.2 Stateless Reference: XGBoost (Story 5.4)

**Rationale:**
1. **Most successful MMLM model 2019–2025** — 1st place 2025, gold 2023
2. **Tabular data standard** — XGBoost is the industry standard for structured/tabular data
3. **Already in dependencies** — `xgboost` is in `pyproject.toml`
4. **Rich ecosystem** — Optuna integration for hyperparameter tuning, SHAP for interpretability
5. **Template value** — demonstrates the stateless model contract (batch train, feature matrix input, probability output)

**Implementation approach for Story 5.4:**
- Create `XGBoostModel(StatelessModel)` wrapping `xgboost.XGBClassifier`
- `train(X, y)` → `XGBClassifier.fit(X, y, eval_set=..., early_stopping_rounds=...)`
- `predict_proba(X)` → `XGBClassifier.predict_proba(X)[:, 1]`
- `save()` → `clf.save_model("model.ubj")` + config JSON (instance method on `XGBClassifier`)
- `load()` → `clf = XGBClassifier(); clf.load_model("model.ubj")` — `load_model` is an instance method, NOT a class method; instantiate first, then call

**Recommended hyperparameter ranges:**

| Parameter | Default | Tuning Range | Notes |
|:---|:---|:---|:---|
| `n_estimators` | 500 | 100–2000 | With early stopping |
| `max_depth` | 5 | 3–8 | Regularize for small data |
| `learning_rate` | 0.05 | 0.01–0.3 | Trade off with n_estimators |
| `subsample` | 0.8 | 0.5–1.0 | Row sampling |
| `colsample_bytree` | 0.8 | 0.5–1.0 | Feature sampling |

### 6.3 Third Reference Model Assessment

**Should LightGBM or a neural network be a third reference model?**

| Option | Pros | Cons | Verdict |
|:---|:---|:---|:---|
| LightGBM | Faster training; native categoricals | Same GBDT family as XGBoost; outperformed in 2025 | **Defer — Post-MVP** |
| CatBoost | Ordered boosting; good default config | Same GBDT family; outperformed in 2025 | **Defer — Post-MVP** |
| LSTM | Temporal sequence modeling; academic promise | No competition wins; high complexity; small data | **Defer — Post-MVP** |
| Logistic Regression | Minimal implementation; competitive baseline | Not distinct enough as a reference; easy to add later | **Defer — Include as Story 5.2 test fixture** |

**Recommendation:** Two reference models (Elo + XGBoost) for MVP. A logistic regression implementation can serve as the minimal test fixture for Story 5.2's Model ABC validation. LightGBM, CatBoost, and neural networks are Post-MVP plugins.

### 6.4 Reference Model Hyperparameter Ranges

**Elo (Story 5.3):**
These defaults match `EloConfig` from Story 4.8 and are informed by Silver/SBCB methodology and community solutions.

| Parameter | Default | Range | Source |
|:---|:---|:---|:---|
| `initial_rating` | 1500.0 | 1200–1800 | Standard Elo convention |
| `k_early` | 56.0 | 40–70 | Silver/SBCB; high early-season uncertainty |
| `k_regular` | 38.0 | 25–50 | Silver/SBCB; standard K-factor |
| `k_tournament` | 47.5 | 35–60 | Community convention; slightly elevated for tournament |
| `margin_exponent` | 0.85 | 0.6–1.0 | Silver formula; 0.85 = moderate MOV credit |
| `max_margin` | 25 | 15–35 | Cap blowouts; 25 pts standard |
| `home_advantage_elo` | 3.5 | 2.0–5.0 | ~3–4 Elo points; declining trend (EDA) |
| `mean_reversion_fraction` | 0.25 | 0.15–0.40 | 20–35% community range |

**XGBoost (Story 5.4):**
Ranges informed by Odeh (2025 winner), maze508 (2023 gold), and Edwards (2021).

| Parameter | Default | Range | Source |
|:---|:---|:---|:---|
| `n_estimators` | 500 | 100–2000 | With early stopping |
| `max_depth` | 5 | 3–8 | Small data → shallow trees |
| `learning_rate` | 0.05 | 0.01–0.3 | Standard range |
| `subsample` | 0.8 | 0.5–1.0 | Row sampling |
| `colsample_bytree` | 0.8 | 0.5–1.0 | Feature sampling |
| `reg_lambda` | 1.0 | 0.5–5.0 | L2 regularization |
| `min_child_weight` | 3 | 1–10 | Regularization |

---

## 7. Model Equivalence Groups

### 7.1 Group Analysis

Similar to the feature equivalence groups in `feature-engineering-techniques.md` Section 7.1, models cluster into families where members capture essentially the same signal:

**Group A — Gradient Boosted Decision Trees:**

| Model | Distinct Contribution | Equivalence |
|:---|:---|:---|
| XGBoost | Level-wise tree growth; native missing values | Representative |
| LightGBM | Leaf-wise growth; GOSS; categorical support | ~Equivalent for small NCAA data |
| CatBoost | Ordered boosting; symmetric trees | ~Equivalent for small NCAA data |

*Implement XGBoost as the representative. LightGBM/CatBoost add minimal distinct signal on NCAA-sized datasets.*

**Group B — Linear Models:**

| Model | Distinct Contribution | Equivalence |
|:---|:---|:---|
| Logistic Regression (L2) | Linear feature relationships | Representative |
| Bayesian LR | Posterior uncertainty; informative priors | Slight extension (adds uncertainty) |
| Elastic Net | L1+L2 feature selection | Same linear family |

*Implement standard Logistic Regression as the representative (Story 5.2 test fixture). Bayesian LR is Post-MVP.*

**Group C — Dynamic Rating Systems (Stateful):**

| Model | Distinct Contribution | Equivalence |
|:---|:---|:---|
| Elo | Per-game rating updates; configurable K/margin/HCA | Representative |
| Glicko-2 | Adds RD (uncertainty) and volatility | Extension (adds 2 params) |
| TrueSkill | Adds σ² (skill uncertainty) | ~Same extension as Glicko-2 |

*Implement Elo as the representative. Glicko-2/TrueSkill add marginal signal for full-season snapshots.*

**Standalone — No Equivalence:**

| Model | Why Distinct |
|:---|:---|
| Monte Carlo Tournament Simulator (Story 6.5) | Not a game prediction model — simulates bracket outcomes from pairwise probabilities |
| LRMC | Markov chain approach — mathematically distinct from both GBDT and Elo |
| Neural Networks (LSTM/Transformer) | Sequential temporal modeling — distinct signal pathway from tabular GBDT |

### 7.2 Redundancy Assessment

| Pair | Redundancy | Recommendation |
|:---|:---|:---|
| XGBoost + LightGBM | **High** — same GBDT family | Implement one (XGBoost); defer LightGBM |
| XGBoost + Logistic Regression | **Low** — non-linear vs. linear | Both useful; LR as baseline comparison |
| Elo + Glicko-2 | **Moderate** — same rating framework | Elo for MVP; Glicko-2 Post-MVP |
| Elo + XGBoost | **None** — completely different paradigms | Both are reference models |
| XGBoost + LSTM | **Low** — tabular vs. sequential | XGBoost for MVP; LSTM Post-MVP |

---

## 8. Scope Recommendation for PO Decision

### Option A: Minimal MVP (Recommended)

**Scope:** 2 reference models + Model ABC + plugin registry

| Component | Story | Description |
|:---|:---|:---|
| Model ABC + Plugin Registry | 5.2 | Dual-contract ABC (StatefulModel, StatelessModel) + decorator-based registry |
| Elo Reference Model | 5.3 | Wraps `EloFeatureEngine` (4.8) as `StatefulModel`; predict via expected score |
| XGBoost Reference Model | 5.4 | Wraps `xgboost.XGBClassifier` as `StatelessModel`; configurable hyperparameters |
| Model Run Tracking + CLI | 5.5 | Metadata tracking, CLI for launching training jobs |

**What's deferred to Post-MVP Backlog:**
- LightGBM, CatBoost (GBDT family — XGBoost covers this)
- Glicko-2, TrueSkill (rating family — Elo covers this)
- LRMC, Mixed-effects models (novel approaches — lower priority)
- LSTM, Transformer (neural nets — academic interest, not competition-proven)
- Stacking/blending meta-model (requires multiple trained models first)

**Rationale:** This option provides the two most competition-validated model types (Elo for stateful, XGBoost for stateless) and the extensibility infrastructure (ABC + registry) for users to add more models. Every deferred model can be added as a plugin without changing core code.

### Option B: Extended MVP

**Scope:** Option A + Logistic Regression as a third reference model

| Additional Component | Story | Description |
|:---|:---|:---|
| Logistic Regression Reference | New 5.4b or in 5.4 | `sklearn.LogisticRegression` wrapper as `StatelessModel`; serves as simple baseline |

**Rationale:** Logistic regression won or placed in MMLM 2014–2018. Trivial to implement (~50 lines). Provides a meaningful comparison point: if LR with ordinal deltas matches XGBoost, the complex features aren't helping.

**Trade-off:** Adds ~0.5 story points of work. Could be included in Story 5.4 as a second stateless model example rather than a separate story.

### Option C: Extended MVP + LightGBM

**Scope:** Option B + LightGBM as a fourth model

| Additional Component | Story | Description |
|:---|:---|:---|
| LightGBM Reference | New 5.4c | `lightgbm.LGBMClassifier` wrapper; comparison point for XGBoost |

**Rationale:** While the 2025 winner found XGBoost superior, having two GBDT implementations enables comparison and validates that the Model ABC properly supports GBDT-family models.

**Trade-off:** Adds a new dependency (`lightgbm`). Marginal value — XGBoost and LightGBM produce very similar results on NCAA-sized data.

### Recommendation

**Option A (Minimal MVP)** is recommended. It provides:
- The two most proven model types (Elo + XGBoost)
- The ABC + registry infrastructure for extensibility
- The smallest scope with the highest value

Option B (adding logistic regression in Story 5.4) is a reasonable extension with minimal cost. The PO should decide whether to include it.

Options C and beyond add models from the same equivalence groups and provide diminishing returns.

---

## References

### Internal Project Documents
- `specs/research/feature-engineering-techniques.md` — Community techniques, equivalence groups, building blocks catalog (Story 4.1)
- `specs/05-architecture-fullstack.md` — FR6 (Model ABC), NFR3 (Plugin Registry), Strategy Pattern
- `src/ncaa_eval/transform/elo.py` — `EloFeatureEngine`, `EloConfig` (Story 4.8)
- `src/ncaa_eval/transform/feature_serving.py` — `StatefulFeatureServer`, `FeatureConfig` (Story 4.7)
- `src/ncaa_eval/ingest/repository.py` — Repository ABC pattern (existing ABC example)
- `src/ncaa_eval/ingest/connectors/base.py` — Connector ABC pattern (existing ABC example)
- `_bmad-output/planning-artifacts/epics.md` — Epic 5 story descriptions (Stories 5.2–5.5)

### Academic Papers
- Habib, M.I. (2025). "Forecasting NCAA Basketball Outcomes with Deep Learning: A Comparative Study of LSTM and Transformer Models." arXiv:2508.02725 — https://arxiv.org/abs/2508.02725 *(⚠️ verify before citing — near knowledge cutoff)*
- Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). "Why do tree-based models still outperform deep learning on tabular data?" NeurIPS 2022
- Glickman, M.E. (2012). "Example of the Glicko-2 system." https://www.glicko.net/glicko/glicko2.pdf
- Silver, N. (2024). "SBCB Methodology." https://www.natesilver.net/p/sbcb-methodology

### Kaggle MMLM Competition Writeups
- Landgraf (2017 winner): https://medium.com/kaggle-blog/march-machine-learning-mania-1st-place-winners-interview-andrew-landgraf-f18214efc659
- maze508 (2023 gold): https://medium.com/@maze508/top-1-gold-kaggle-march-machine-learning-mania-2023-solution-writeup-2c0273a62a78
- Odeh (2025 winner): https://www.kaggle.com/competitions/march-machine-learning-mania-2025/writeups/mohammad-odeh-first-place-solution
- Edwards (2021 documented): https://johnbedwards.io/blog/march_madness_2021/
- mlcontests.com annual reports (2022–2025): https://mlcontests.com

### Library Documentation
- XGBoost Model IO: https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html
- scikit-learn Developing Estimators: https://scikit-learn.org/stable/developers/develop.html
- goto_conversion: https://github.com/gotoConversion/goto_conversion
- Pydantic v2: https://docs.pydantic.dev/latest/
