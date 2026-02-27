# User Guide

## Getting Started

This guide picks up where the `README.md` (project root) left off.
Once you have installed the project and synced data, the typical workflow is:

1. **Sync data** — download NCAA game results from Kaggle (and optionally ESPN):

   ```bash
   python sync.py --source all --dest data/
   ```

2. **Train a model** — fit a prediction model on historical seasons:

   ```bash
   python -m ncaa_eval.cli train --model elo
   python -m ncaa_eval.cli train --model xgboost
   ```

   Common options:

   | Flag | Default | Description |
   |------|---------|-------------|
   | `--model` | *(required)* | Registered model name (`elo`, `xgboost`, or custom) |
   | `--start-year` | `2015` | First training season (inclusive) |
   | `--end-year` | `2025` | Last training season (inclusive) |
   | `--data-dir` | `data/` | Path to synced Parquet files |
   | `--output-dir` | `data/` | Where to write run artifacts |
   | `--config` | `None` | JSON file overriding model hyperparameters |

3. **Explore results in the dashboard** — launch the Streamlit app:

   ```bash
   streamlit run dashboard/app.py
   ```

   The sidebar lets you select a tournament year, model run, and scoring format.
   All pages update automatically when you change these filters.

4. **Iterate** — retrain with different hyperparameters, compare on the Leaderboard,
   inspect calibration in Model Deep Dive, and use the Bracket Visualizer and
   Pool Scorer to turn predictions into bracket picks.


## Evaluation Metrics

The platform evaluates models on four complementary metrics.
Each captures a different aspect of prediction quality.

### Log Loss

**What it measures:** How well predicted probabilities match actual outcomes, with
heavy penalties for confident wrong predictions.

**Formula:**

$$\text{Log Loss} = -\frac{1}{N} \sum_{i=1}^{N} \bigl[y_i \ln(p_i) + (1-y_i) \ln(1-p_i)\bigr]$$

**Interpretation:**

| Value | Meaning |
|------:|---------|
| 0.0 | Perfect — every prediction was 0% or 100% and correct |
| ~0.50 | Good — a well-calibrated model typically lands here |
| 0.693 | Random baseline (equivalent to predicting 50% for every game) |
| > 0.693 | Worse than guessing — the model is actively misleading |

```{tip}
Log Loss is the primary ranking metric in the Kaggle March Machine Learning Mania
competition.  A score of 0.55 means your model is meaningfully better than random
but has room to improve.
```

```{warning}
Log Loss punishes confident wrong predictions exponentially.  A single game where
you predicted 99% and the other team won adds roughly 4.6 to your loss — far more
than 100 correct predictions at 60% confidence save.
```


### Brier Score

**What it measures:** Mean squared error of probability predictions — a gentler
alternative to Log Loss.

**Formula:**

$$\text{Brier Score} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2$$

**Interpretation:**

| Value | Meaning |
|------:|---------|
| 0.0 | Perfect |
| ~0.20 | Good model |
| 0.25 | Random baseline (predicting 50% for every game) |
| > 0.25 | Worse than guessing |

Brier Score is more forgiving of confident wrong predictions than Log Loss because
it uses squared error instead of logarithmic error.  A 99%-confident wrong
prediction adds 0.98 to Brier (vs. 4.6 to Log Loss).


### ROC-AUC

**What it measures:** Discrimination — can the model distinguish winners from
losers?  Equivalently: if you pick a random winning team and a random losing team,
what is the probability the model assigns a higher win probability to the winner?

**Formula:** Area under the Receiver Operating Characteristic curve.

**Interpretation:**

| Value | Meaning |
|------:|---------|
| 1.0 | Perfect discrimination |
| ~0.75 | Good model |
| 0.5 | Random — no discrimination ability |
| < 0.5 | Inversely correlated (predicting losers as winners) |

```{warning}
ROC-AUC does **not** measure calibration.  A model can have perfect AUC (1.0) but
terrible calibration — e.g., predicting 99% for every game that the favored team
wins and 1% for every upset.  Always pair AUC with calibration metrics (ECE,
reliability diagrams).
```


### Expected Calibration Error (ECE)

**What it measures:** How well predicted probabilities correspond to actual win
rates.  If a model says "70% win probability" for 100 games, about 70 should
actually be wins.

**Formula:**

$$\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{N} \left| \text{acc}(b) - \text{conf}(b) \right|$$

where predictions are binned into $B$ equal-width bins, $n_b$ is the count in bin
$b$, $\text{acc}(b)$ is the observed win rate, and $\text{conf}(b)$ is the mean
predicted probability.

**Interpretation:**

| Value | Meaning |
|------:|---------|
| 0.0 | Perfectly calibrated |
| < 0.03 | Excellent calibration |
| 0.03–0.08 | Reasonable calibration |
| > 0.10 | Poor calibration — predictions are systematically off |

```{tip}
ECE is the single best number for answering "can I trust these probabilities at
face value?"  A low ECE means you can use the model's probability outputs directly
for bet sizing, pool strategy, and expected-value calculations.
```


## Model Types

NCAA_eval supports two model paradigms through a common abstract base class.

### Stateful Models

Stateful models maintain internal ratings that update game-by-game through a season.
They process games **sequentially** — the order matters.

**Reference implementation: Elo** (`elo`)

The built-in Elo model tracks a rating for every team.  After each game, ratings
shift based on the outcome and margin of victory:

- Winners gain rating points; losers lose them
- Upset victories produce larger rating swings
- Ratings mean-revert between seasons (configurable fraction)
- Separate K-factors for early-season, regular-season, and tournament games

**Best for:** Capturing in-season trajectory and momentum.  Elo is simple,
interpretable, and requires no feature engineering.

**Key hyperparameters:**

| Parameter | Default | Description |
|-----------|--------:|-------------|
| `initial_rating` | 1500 | Starting rating for new teams |
| `k_early` | 56.0 | K-factor for the first 20 games |
| `k_regular` | 38.0 | K-factor for regular-season games |
| `k_tournament` | 47.5 | K-factor for tournament games |
| `mean_reversion_fraction` | 0.25 | Fraction pulled toward mean between seasons |

```{tip}
You can override any hyperparameter via a JSON config file:
`python -m ncaa_eval.cli train --model elo --config my_elo_config.json`
```


### Stateless Models

Stateless models are standard batch-trained classifiers.  They take a feature
matrix as input and produce probability predictions — game order does not matter.

**Reference implementation: XGBoost** (`xgboost`)

The built-in XGBoost model uses gradient-boosted decision trees trained on feature
snapshots.  Features are computed by the feature engineering pipeline (Epic 4) and
include team statistics, strength-of-schedule metrics, and graph-based centrality
measures.

**Best for:** Combining many feature dimensions for maximum predictive accuracy.
XGBoost typically outperforms Elo when strong features are available.

**Key hyperparameters:**

| Parameter | Default | Description |
|-----------|--------:|-------------|
| `n_estimators` | 500 | Maximum number of boosting rounds |
| `max_depth` | 5 | Maximum tree depth |
| `learning_rate` | 0.05 | Step size shrinkage |
| `early_stopping_rounds` | 50 | Stop if validation loss doesn't improve |


### Plugin Registry

Models register themselves via the `@register_model("name")` decorator.  To create
a custom model:

1. Subclass `Model` (stateless) or `StatefulModel` (stateful)
2. Implement the required methods:
   - **Stateless (`Model`):** `fit`, `predict_proba`, `save`, `load`, `get_config`
   - **Stateful (`StatefulModel`):** `_predict_one`, `update`, `start_season`, `get_state`,
     `set_state`, `save`, `load`, `get_config` — `fit` and `predict_proba` are provided by
     the template (`_predict_one` is the per-pair hook that `predict_proba` calls)
3. Decorate with `@register_model("my_model")`
4. Import the module before training so the decorator fires

The CLI discovers all registered models automatically:

```bash
# List available models
python -c "from ncaa_eval.model import list_models; print(list_models())"
```

For implementation details, see the [API Reference](api/modules.rst).


## Interpreting Results

### Reliability Diagrams

A reliability diagram is the visual counterpart to ECE.  It plots predicted
probabilities against observed win rates:

- **X-axis:** Predicted probability (grouped into bins, e.g., 0–10%, 10–20%, …)
- **Y-axis:** Actual win rate within each bin
- **Perfect calibration line:** The 45° diagonal — if your model says 70%, 70% of
  those games should be wins

**How to read the diagram:**

| Pattern | Meaning | Action |
|---------|---------|--------|
| Points on the diagonal | Well-calibrated | No action needed |
| Points **above** the diagonal | Under-confident — actual win rates exceed predictions | Model could be sharper |
| Points **below** the diagonal | Over-confident — predictions overstate win likelihood | Model needs calibration |
| S-shaped curve | Probabilities are too extreme on both ends | Retrain with calibration regularization; temperature scaling via Game Theory Sliders (planned feature) |
| Flat line near 0.5 | Model lacks discrimination | Improve features or model architecture |

```{tip}
The Model Deep Dive page in the dashboard shows reliability diagrams with per-year
drill-down.  Compare diagrams across years to check whether calibration is stable
or drifts.
```


### Calibration in Plain Language

A **well-calibrated** model is one you can take at face value.  When it says
"Duke has a 65% chance of beating UNC," that means that in a large sample of
similar matchups, Duke would win about 65% of the time.

**Why calibration matters for bracket pools:**

- **Pool strategy** depends on knowing *how likely* outcomes are, not just *which
  team is favored*
- Expected-point calculations multiply advancement probabilities by scoring weights
  — if probabilities are wrong, the strategy is wrong
- An over-confident model will undercount upsets, leading you to pick too much chalk


### Over-Confidence vs. Under-Confidence

| Type | Symptom | Reliability Diagram | Impact on Brackets |
|------|---------|--------------------|--------------------|
| **Over-confident** | Predictions are too extreme (90% when reality is 70%) | Points below diagonal at high probabilities | Too much chalk; undervalues upsets |
| **Under-confident** | Predictions are too moderate (55% when reality is 70%) | Points above diagonal at high probabilities | Picks too many upsets; misses value in favorites |
| **Well-calibrated** | Predictions match reality | Points on or near diagonal | Bracket strategy reflects true odds |


## Tournament Simulation

### Monte Carlo Methodology

The platform simulates the full 64-team NCAA tournament bracket using two methods:

**Analytical (Phylourny algorithm):**
Computes exact advancement probabilities via a post-order tree traversal.  Fast and
deterministic — no random sampling needed.  Best for expected-point calculations
where you want precise values.

**Monte Carlo simulation:**
Runs thousands of independent tournament simulations (default 10,000).  Each
simulation randomly resolves every game using the model's pairwise win
probabilities.  Produces:

- **Advancement probabilities** — fraction of simulations each team reaches each
  round
- **Score distributions** — histogram of total bracket points across all
  simulations
- **Confidence intervals** — percentile-based ranges for expected outcomes


### Bracket Distribution

When you run a Monte Carlo simulation, the platform computes a full score
distribution for the "chalk bracket" (picking the pre-game favorite in every
matchup).  This answers: "If the model's probabilities are correct, what range of
scores should I expect?"

Key statistics shown on the Pool Scorer page:

| Statistic | What It Tells You |
|-----------|-------------------|
| **Median** | The score you'd most typically get |
| **Mean** | Average expected score (weighted by probability) |
| **5th percentile** | Worst-case scenario (lots of upsets) |
| **95th percentile** | Best-case scenario (mostly chalk) |
| **Std Dev** | How much scores vary across simulated outcomes |


### Expected Points

Expected Points (EP) combines advancement probabilities with a scoring rule to
answer: "How many bracket points is each team *worth*?"

$$\text{EP}_i = \sum_{r=0}^{5} P(\text{team } i \text{ wins round } r) \times \text{points}(r)$$

Teams with high EP are valuable picks — they are likely to advance far *and* those
rounds are worth many points.  The Bracket Visualizer's Expected Points table ranks
all 64 teams by EP under your chosen scoring rule.


## Tournament Scoring

The platform supports three built-in scoring systems and lets you define custom
rules.

### Standard Scoring (ESPN-style)

The most common pool format.  Points double each round:

| Round | Abbrev. | Games | Points | Max Points |
|-------|---------|------:|-------:|-----------:|
| Round of 64 | R64 | 32 | 1 | 32 |
| Round of 32 | R32 | 16 | 2 | 32 |
| Sweet 16 | S16 | 8 | 4 | 32 |
| Elite Eight | E8 | 4 | 8 | 32 |
| Final Four | F4 | 2 | 16 | 32 |
| Championship | NCG | 1 | 32 | 32 |
| **Total** | | **63** | | **192** |

**Worked example:** You correctly pick 20 R64 games, 10 R32 games, 4 S16 games,
2 E8 games, 1 F4 game, and the champion:

`20×1 + 10×2 + 4×4 + 2×8 + 1×16 + 1×32 = 20 + 20 + 16 + 16 + 16 + 32 = **120 points**`


### Fibonacci Scoring

Rewards later rounds more steeply than Standard:

| Round | Points |
|-------|-------:|
| R64 | 2 |
| R32 | 3 |
| S16 | 5 |
| E8 | 8 |
| F4 | 13 |
| NCG | 21 |
| **Total (perfect)** | **231** |

Fibonacci scoring gives more credit for getting later rounds right.  Picking the
champion is worth 21 points (vs. 32 in Standard), but the ratio of
late-round-to-early-round points is higher.


### Seed-Difference Bonus

Standard base points plus an upset bonus equal to the seed difference when the
lower-seeded team wins:

| Round | Base Points | Upset Bonus |
|-------|------------:|-------------|
| R64 | 1 | + \|seed_winner − seed_loser\| if upset |
| R32 | 2 | + \|seed_winner − seed_loser\| if upset |
| S16 | 4 | + \|seed_winner − seed_loser\| if upset |
| E8 | 8 | + \|seed_winner − seed_loser\| if upset |
| F4 | 16 | + \|seed_winner − seed_loser\| if upset |
| NCG | 32 | + \|seed_winner − seed_loser\| if upset |

**Worked example:** A 12-seed beats a 5-seed in the R64.  You get
1 (base) + 7 (seed diff) = **8 points** for that single game — the same as
getting an Elite Eight pick right under Standard scoring.

```{tip}
Seed-Difference Bonus rewards contrarian picks.  If you are in a large pool where
most people pick chalk, this scoring format lets you differentiate by picking
well-chosen upsets.
```


### Custom Scoring

The Pool Scorer page lets you define custom per-round point values.  Check
"Use custom scoring" and enter your pool's specific point schedule.  This is useful
for pools with non-standard formats (e.g., 1-2-3-5-8-13 or 10-20-40-80-160-320).


## Dashboard Guide

The dashboard has four main pages organized into two sections.

### Lab: Backtest Leaderboard

**Purpose:** Compare all trained models side-by-side.

**What you see:**

- **KPI cards** at the top showing the best score for each metric across all runs,
  with delta indicators comparing best vs. worst
- **Sortable table** with every model run's Log Loss, Brier Score, ROC-AUC, and ECE
- Color-coded cells (red-yellow-green gradient) for quick visual comparison
- If a year filter is active, metrics are shown for that year only; otherwise,
  metrics are averaged across all evaluated years

**How to use it:**

1. Select a tournament year in the sidebar (or leave unset for aggregate view)
2. Click any row to navigate to the Model Deep Dive for that run


### Lab: Model Deep Dive

**Purpose:** Diagnose a single model's calibration, accuracy, and feature behavior.

**What you see:**

- **Reliability diagram** — calibration visualization (see
  [Interpreting Results](#interpreting-results) above)
- **Year drill-down** — select a specific fold year to see how calibration varies
  over time
- **Per-year metric summary** — table of Log Loss, Brier, AUC, and ECE broken out
  by year, with gradient coloring
- **Feature importance** (XGBoost only) — horizontal bar chart showing which
  features contribute most to predictions
- **Hyperparameters** — JSON view of the model's configuration

**How to use it:**

1. Click a model run on the Leaderboard, or select a run in the sidebar
2. Use the year dropdown to compare calibration across different seasons
3. For XGBoost models, check feature importance to understand what drives predictions


### Presentation: Bracket Visualizer

**Purpose:** Turn model predictions into a visual bracket with advancement
probabilities.

**What you see:**

- **Most-likely bracket** — interactive HTML bracket tree showing predicted winners
  at every matchup, with win probabilities displayed
- **Advancement heatmap** — color-coded grid showing each team's probability of
  reaching each round
- **Pairwise win probabilities** — expandable section where you can pick any two
  teams and see the head-to-head probability
- **Expected Points table** — all 64 teams ranked by expected points under the
  selected scoring rule
- **Score distribution** (Monte Carlo only) — histogram of possible bracket scores

**How to use it:**

1. Select a model run, tournament year, and scoring format in the sidebar
2. Choose "Analytical (exact)" for speed or "Monte Carlo" for score distributions
3. If using Monte Carlo, adjust the number of simulations (more = more precise, but
   slower)
4. Use the Expected Points table to identify high-value picks
5. Expand "Pairwise Win Probabilities" to investigate specific matchups


### Presentation: Pool Scorer

**Purpose:** Score your bracket against thousands of simulated tournament outcomes
to understand your expected point distribution.

**What you see:**

- **Scoring configuration** — choose a built-in rule or define custom per-round
  points
- **Outcome summary** — median, mean, standard deviation, min/max, and percentile
  metrics for your bracket's point total
- **Score distribution histogram** — visual distribution of how your bracket would
  score across all simulated outcomes
- **CSV export** — download your bracket as a CSV file for submission to your pool

**How to use it:**

1. Select a model run and tournament year in the sidebar
2. Configure your pool's scoring rules (or use the default Standard scoring)
3. Adjust the number of Monte Carlo simulations (10,000 is a good default)
4. Click "Analyze Outcomes" to run the simulation
5. Review the outcome summary to understand your expected score range
6. Download the bracket CSV for your pool submission


### Game Theory Sliders

```{note}
Game Theory Sliders are a planned feature based on research from Story 7.7.
They are not yet implemented in the dashboard.  This section describes the
intended design.
```

Two sliders will allow you to adjust the bracket strategy without retraining:

**Upset Aggression** (range: −5 to +5, default: 0)

Controls whether your bracket picks favor chalk (favorites) or chaos (upsets):

| Setting | Temperature | Effect |
|--------:|:-----------:|--------|
| −5 | 0.31 | Extreme chalk — nearly every favorite wins |
| −3 | 0.50 | Strong chalk — favorites heavily reinforced |
| 0 | 1.00 | Neutral — model probabilities unchanged |
| +3 | 2.00 | Strong chaos — probabilities compress toward 50/50 |
| +5 | 3.17 | Extreme chaos — nearly every game is a coin flip |

Mathematically, this applies a power transform to every win probability:

$$p' = \frac{p^{1/T}}{p^{1/T} + (1-p)^{1/T}} \quad \text{where } T = 2^{v/3}$$

A probability of exactly 50% is never moved.  Favorites remain favorites — the
transform preserves the ranking of all probabilities.

**Seed-Weight** (range: 0% to 100%, default: 0%)

Blends the model's predictions with historical seed-vs-seed win rates:

| Setting | Effect |
|--------:|--------|
| 0% | Pure model predictions |
| 25% | 75% model + 25% historical seed performance |
| 50% | Equal blend of model and seed history |
| 100% | Ignore the model entirely; use historical seed win rates |

This is useful when you believe the tournament seeding committee has information
your model doesn't capture, or when your model makes predictions that diverge
significantly from historical seed performance.

```{tip}
In a **small pool** (< 10 people), pick chalk — you want the most likely
bracket.  In a **large pool** (50+ people), increase Upset Aggression to
differentiate your bracket from the crowd.
```
