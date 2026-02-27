# Getting Started Tutorial

This tutorial walks you through the full NCAA_eval pipeline — from syncing data
to viewing model predictions in the interactive dashboard.

**Prerequisites:** You have already installed the project (`poetry install`) and
configured your Kaggle API credentials.  See the [README](../../README.md) for
setup instructions.

## Step 1: Sync Data

Download NCAA game data from Kaggle (historical seasons 1985–2025) and ESPN
(current-season scores):

```bash
python sync.py --source all --dest data/
```

Expected output (first run):

```text
Downloading Kaggle competition data...
  ✓ MRegularSeasonCompactResults.csv (150,000+ games)
  ✓ MNCAATourneyCompactResults.csv
  ✓ MTeams.csv
  ...
Fetching ESPN current-season data...
  ✓ 2025 regular season games
Writing Parquet files to data/
  ✓ data/games.parquet
  ✓ data/teams.parquet
  ✓ data/seeds.parquet
Sync complete.
```

```{tip}
Subsequent runs skip already-cached files automatically.  Use `--force-refresh`
to re-download everything.
```

## Step 2: Train an Elo Model

The Elo model is a stateful rating system — it maintains per-team ratings that
update game-by-game.  It requires no feature engineering and is a good first
model to train:

```bash
python -m ncaa_eval.cli train --model elo
```

Expected output:

```text
Training model: elo
  Seasons: 2015–2025
  Data dir: data/
  ...
Walk-forward backtest (10 folds):
  2016: Log Loss=0.560, Brier=0.208, AUC=0.740, ECE=0.031
  2017: Log Loss=0.548, Brier=0.204, AUC=0.753, ECE=0.028
  ...
Run artifacts saved to data/runs/<run_id>/
  ✓ model/   (trained model files)
  ✓ predictions.parquet
  ✓ metrics.json
  ✓ run_metadata.json
```

The `--model` flag selects from registered model plugins.  To see all available
models:

```bash
python -c "from ncaa_eval.model import list_models; print(list_models())"
```

```text
['elo', 'logistic_regression', 'xgboost']
```

### Customize Hyperparameters

Override any Elo hyperparameter via a JSON config file:

```json
{
  "k_regular": 40.0,
  "mean_reversion_fraction": 0.30
}
```

```bash
python -m ncaa_eval.cli train --model elo --config my_elo_config.json
```

See the [User Guide — Stateful Models](../user-guide.md#stateful-models) for the
full list of Elo hyperparameters.

## Step 3: Train an XGBoost Model

XGBoost is a stateless model — it takes a feature matrix as input and learns
which features best predict game outcomes:

```bash
python -m ncaa_eval.cli train --model xgboost
```

Expected output:

```text
Training model: xgboost
  Seasons: 2015–2025
  Building feature matrix...
  Features: 85 columns (sequential, graph, batch ratings, seeds)
  Training with early stopping (50 rounds patience)...
  Best iteration: 312/500
Walk-forward backtest (10 folds):
  2016: Log Loss=0.525, Brier=0.195, AUC=0.775, ECE=0.025
  ...
```

XGBoost typically outperforms Elo when the feature engineering pipeline provides
strong signal.  See the [User Guide — Stateless Models](../user-guide.md#stateless-models)
for hyperparameter details.

### Adjust the Training Window

Train on more (or fewer) seasons using `--start-year` and `--end-year`:

```bash
python -m ncaa_eval.cli train --model xgboost --start-year 2010 --end-year 2024
```

## Step 4: Launch the Dashboard

Start the Streamlit dashboard to explore your model results:

```bash
streamlit run dashboard/app.py
```

The dashboard opens in your browser at `http://localhost:8501`.

### Dashboard Navigation

The dashboard has four pages organized into two sections:

**Lab** (model analysis):

- **Backtest Leaderboard** — Compare all trained models side-by-side on
  Log Loss, Brier Score, ROC-AUC, and ECE.  Color-coded cells highlight the best
  and worst performers.
- **Model Deep Dive** — Inspect a single model's calibration via reliability
  diagrams, per-year metric breakdowns, and feature importance (XGBoost only).

**Presentation** (tournament predictions):

- **Bracket Visualizer** — View the model's predicted bracket with advancement
  probabilities, pairwise win probabilities, and expected points per team under
  your chosen scoring rule.
- **Pool Scorer** — Score your bracket against thousands of simulated tournament
  outcomes to see your expected point distribution.  Export the bracket as CSV.

### Sidebar Filters

Use the sidebar to control what you see:

| Filter | Options | Effect |
|--------|---------|--------|
| **Tournament Year** | Any year with tournament data | Filters all pages to that year |
| **Model Run** | Any completed training run | Selects which model's predictions to display |
| **Scoring Format** | Standard, Fibonacci, Seed-Diff Bonus, Custom | Changes how bracket points are calculated |

```{tip}
All pages update automatically when you change sidebar filters.  Start on the
Leaderboard to compare models, then click a model run to dive into its details.
```

## Step 5: Interpret Your Results

### Compare Models on the Leaderboard

The Leaderboard shows key metrics for every training run:

| Metric | Better When | Random Baseline |
|--------|-------------|-----------------|
| Log Loss | Lower | 0.693 |
| Brier Score | Lower | 0.25 |
| ROC-AUC | Higher | 0.5 |
| ECE | Lower | — |

```{tip}
For a detailed explanation of each metric (formulas, interpretation, worked
examples), see the [User Guide — Evaluation Metrics](../user-guide.md#evaluation-metrics).
```

### Check Calibration in Model Deep Dive

Select a model run and navigate to the Deep Dive page.  The reliability diagram
shows whether your model's predicted probabilities match reality:

- **Points on the diagonal** = well-calibrated
- **Points above** = under-confident (predicts 60% but wins 70%)
- **Points below** = over-confident (predicts 80% but wins 65%)

Use the year dropdown to check calibration stability across seasons.

### Build a Bracket in the Bracket Visualizer

1. Select a model run and tournament year
2. Choose "Analytical (exact)" for fast expected points, or "Monte Carlo" for
   score distributions
3. Review the **Expected Points table** — teams at the top are the most valuable
   bracket picks under your scoring rule
4. Check the **Advancement Heatmap** to see each team's probability of reaching
   each round

## Step 6: Iterate and Improve

The typical workflow loop is:

1. **Train** a new model (or retrain with different hyperparameters)
2. **Compare** on the Leaderboard — did metrics improve?
3. **Inspect** calibration on the Deep Dive page
4. **Build a bracket** using the Bracket Visualizer
5. **Score** the bracket on the Pool Scorer page
6. Repeat

```{tip}
Try training a Logistic Regression model as a simple baseline:
`python -m ncaa_eval.cli train --model logistic_regression`
```

## Next Steps

- **Create a custom model** — See the [Custom Model Tutorial](custom-model.md)
- **Add a custom metric** — See the [Custom Metric Tutorial](custom-metric.md)
- **Deep dive into metrics** — See the [User Guide](../user-guide.md)
