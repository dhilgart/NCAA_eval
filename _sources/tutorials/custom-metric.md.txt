# How to Add a Custom Metric

This tutorial shows how to extend NCAA_eval's evaluation engine with custom
metrics and custom tournament scoring rules.

## Prerequisites

- Project installed (`poetry install`)
- Data synced and at least one model trained (see the
  [Getting Started Tutorial](getting-started.md))

## Part 1: Custom Metric Function

The evaluation engine uses metric functions with a simple contract:

```python
def my_metric(y_true: np.ndarray, y_prob: np.ndarray) -> float
```

- `y_true` — binary labels (0 or 1), shape `(n_games,)`
- `y_prob` — predicted probabilities in [0, 1], shape `(n_games,)`
- Returns a single `float` score

### Step 1: Write the Metric

Here is an example that computes the average absolute error (a simpler
alternative to Brier Score):

```python
import numpy as np
import numpy.typing as npt


def mean_absolute_error(
    y_true: npt.NDArray[np.float64],
    y_prob: npt.NDArray[np.float64],
) -> float:
    """Mean Absolute Error between predictions and outcomes.

    Lower is better. Random baseline (predicting 0.5): 0.5.
    """
    return float(np.mean(np.abs(y_prob - y_true)))
```

Another example — a "surprise" metric that measures how often the model was
confidently wrong:

```python
def surprise_rate(
    y_true: npt.NDArray[np.float64],
    y_prob: npt.NDArray[np.float64],
    threshold: float = 0.75,
) -> float:
    """Fraction of games where model was confident but wrong.

    A game is a "surprise" if model predicted > threshold for one team
    but the other team won. Lower is better.
    """
    confident_team_a = y_prob >= threshold
    confident_team_b = y_prob <= (1.0 - threshold)
    confident = confident_team_a | confident_team_b

    if not np.any(confident):
        return 0.0

    confident_wrong = (
        (confident_team_a & (y_true == 0))
        | (confident_team_b & (y_true == 1))
    )
    return float(np.sum(confident_wrong) / np.sum(confident))
```

```{note}
Metric functions with extra parameters (like `threshold` above) need a wrapper
to match the `(y_true, y_prob) -> float` signature when passed to the backtest.
See Step 2 below.
```

### Step 2: Use in a Backtest

Pass your custom metrics to `run_backtest` via the `metric_fns` parameter:

```python
from functools import partial
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ncaa_eval.evaluation.backtest import run_backtest, DEFAULT_METRICS
from ncaa_eval.evaluation.metrics import log_loss, brier_score
from ncaa_eval.ingest import ParquetRepository
from ncaa_eval.model import get_model
from ncaa_eval.transform.feature_serving import FeatureConfig, StatefulFeatureServer
from ncaa_eval.transform.serving import ChronologicalDataServer


def mean_absolute_error(
    y_true: npt.NDArray[np.float64],
    y_prob: npt.NDArray[np.float64],
) -> float:
    return float(np.mean(np.abs(y_prob - y_true)))


def surprise_rate(
    y_true: npt.NDArray[np.float64],
    y_prob: npt.NDArray[np.float64],
    threshold: float = 0.75,
) -> float:
    confident_a = y_prob >= threshold
    confident_b = y_prob <= (1.0 - threshold)
    confident = confident_a | confident_b
    if not np.any(confident):
        return 0.0
    wrong = (confident_a & (y_true == 0)) | (confident_b & (y_true == 1))
    return float(np.sum(wrong) / np.sum(confident))


# Build metric dictionary — include built-in metrics plus custom ones
my_metrics = {
    "log_loss": log_loss,
    "brier_score": brier_score,
    "mae": mean_absolute_error,
    "surprise_75": partial(surprise_rate, threshold=0.75),
    "surprise_90": partial(surprise_rate, threshold=0.90),
}

# Set up model and feature server
model_cls = get_model("elo")
model = model_cls()
repo = ParquetRepository(base_path=Path("data/"))
data_server = ChronologicalDataServer(repo)
server = StatefulFeatureServer(config=FeatureConfig(), data_server=data_server)

# Run backtest with custom metrics
result = run_backtest(
    model=model,
    feature_server=server,
    seasons=list(range(2015, 2026)),
    mode="stateful",
    metric_fns=my_metrics,
)

# View per-year results
print(result.summary)
```

Expected output:

```text
      log_loss  brier_score    mae  surprise_75  surprise_90  elapsed_seconds
2016    0.5601      0.2082  0.412        0.182        0.091            1.23
2017    0.5483      0.2041  0.405        0.175        0.085            1.18
...
```

````{tip}
To use the default metrics *plus* your custom ones, merge with `DEFAULT_METRICS`:

```python
from ncaa_eval.evaluation.backtest import DEFAULT_METRICS

my_metrics = {**DEFAULT_METRICS, "mae": mean_absolute_error}
```
````

### Step 3: Verify Your Metric

Test your metric with known inputs to make sure it behaves correctly:

```python
import numpy as np

# Perfect predictions
y_true = np.array([1.0, 0.0, 1.0, 0.0])
y_prob = np.array([1.0, 0.0, 1.0, 0.0])
assert mean_absolute_error(y_true, y_prob) == 0.0

# Random predictions
y_true = np.array([1.0, 0.0, 1.0, 0.0])
y_prob = np.array([0.5, 0.5, 0.5, 0.5])
assert mean_absolute_error(y_true, y_prob) == 0.5

# Worst-case predictions
y_true = np.array([1.0, 0.0, 1.0, 0.0])
y_prob = np.array([0.0, 1.0, 0.0, 1.0])
assert mean_absolute_error(y_true, y_prob) == 1.0
```

## Part 2: Custom Tournament Scoring Rule

Scoring rules define how bracket points are awarded per round.  The platform
uses a `ScoringRule` protocol:

```python
class ScoringRule(Protocol):
    @property
    def name(self) -> str: ...

    def points_per_round(self, round_idx: int) -> float: ...
```

`round_idx` maps to tournament rounds:

| `round_idx` | Round |
|:-----------:|-------|
| 0 | Round of 64 |
| 1 | Round of 32 |
| 2 | Sweet 16 |
| 3 | Elite Eight |
| 4 | Final Four |
| 5 | Championship |

### Step 1: Create a Scoring Rule

Here is an example scoring rule where each round is worth 10x the previous:

```python
class ExponentialScoring:
    """Exponential scoring: 1-10-100-1000-10000-100000."""

    _POINTS = (1.0, 10.0, 100.0, 1000.0, 10_000.0, 100_000.0)

    @property
    def name(self) -> str:
        return "exponential"

    def points_per_round(self, round_idx: int) -> float:
        return self._POINTS[round_idx]
```

Alternatively, use the built-in `DictScoring` helper:

```python
from ncaa_eval.evaluation.simulation import DictScoring

my_pool_scoring = DictScoring(
    points={0: 2, 1: 3, 2: 5, 3: 10, 4: 15, 5: 25},
    scoring_name="my_pool",
)
```

### Step 2: Use in Simulation

Pass your scoring rule to the tournament simulator:

```python
from pathlib import Path

from ncaa_eval.evaluation.simulation import (
    EloProvider,
    MatchupContext,
    StandardScoring,
    build_bracket,
    simulate_tournament,
)
from ncaa_eval.model.tracking import RunStore
from ncaa_eval.transform.normalization import TourneySeedTable

# Replace with the run ID printed when you trained the model
run_id = "<your-run-id>"

# Load tournament seeds from the Kaggle CSV in your data directory
seed_table = TourneySeedTable.from_csv(Path("data/kaggle/MNCAATourneySeeds.csv"))
seeds = seed_table.all_seeds(season=2024)  # list[TourneySeed]

# Build the 64-team bracket tree (play-in teams excluded automatically)
bracket = build_bracket(seeds, season=2024)

# Load the trained model via RunStore
store = RunStore(Path("data/"))
model = store.load_model(run_id)
assert model is not None, f"No model artifacts found for run_id={run_id!r}"

# Create probability provider (wraps the model's _predict_one method)
# NOTE: EloProvider requires a StatefulModel (one with _predict_one).
# Use the run_id of an Elo or other stateful model training run.
provider = EloProvider(model)
context = MatchupContext(season=2024, day_num=136, is_neutral=True)

# Simulate with both built-in and custom scoring
result = simulate_tournament(
    bracket=bracket,
    probability_provider=provider,
    context=context,
    scoring_rules=[StandardScoring(), ExponentialScoring()],
    method="monte_carlo",
    n_simulations=10_000,
)

# result.expected_points maps rule name → per-team expected-points array.
# bracket.team_ids[i] gives the team ID for bracket position i.
for rule_name, ep_array in result.expected_points.items():
    print(f"\n{rule_name}:")
    for i, ep in enumerate(ep_array):
        team_id = bracket.team_ids[i]
        print(f"  Team {team_id}: {ep:.1f} EP")
```

### Step 3: Register for CLI Use (Optional)

To make your scoring rule available in the dashboard and CLI, register it:

```python
from ncaa_eval.evaluation.simulation import register_scoring


@register_scoring("exponential")
class ExponentialScoring:
    """Exponential scoring: 1-10-100-1000-10000-100000."""

    _POINTS = (1.0, 10.0, 100.0, 1000.0, 10_000.0, 100_000.0)

    @property
    def name(self) -> str:
        return "exponential"

    def points_per_round(self, round_idx: int) -> float:
        return self._POINTS[round_idx]
```

Verify registration:

```python
from ncaa_eval.evaluation.simulation import list_scorings

print(list_scorings())
# ['fibonacci', 'seed_diff_bonus', 'standard', 'exponential']
```

## Summary

| Extension Point | Contract | Where to Use |
|----------------|----------|--------------|
| Custom metric | `(y_true, y_prob) -> float` | `run_backtest(metric_fns=...)` |
| Custom scoring rule | `ScoringRule` protocol (`.name`, `.points_per_round()`) | `simulate_tournament(scoring_rules=...)` |
| Dict-based scoring | `DictScoring(points={0: ..., 5: ...})` | Quick custom point schedules |

## Next Steps

- **Build a custom model** — See the [Custom Model Tutorial](custom-model.md)
- **Explore the built-in metrics** — See
  `src/ncaa_eval/evaluation/metrics.py` and the
  [User Guide — Evaluation Metrics](../user-guide.md#evaluation-metrics)
- **Tournament scoring details** — See the
  [User Guide — Tournament Scoring](../user-guide.md#tournament-scoring)
