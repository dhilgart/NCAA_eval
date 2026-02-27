# How to Create a Custom Model

This tutorial walks you through building and registering a custom prediction
model.  By the end, you will have a working model that integrates with the CLI,
evaluation engine, and dashboard.

NCAA_eval supports two model paradigms:

- **Stateless** — batch-trained classifiers (like XGBoost or Logistic Regression)
- **Stateful** — sequential-update models (like Elo) that maintain per-team
  ratings updated game-by-game

This tutorial covers both.

## Prerequisites

- Project installed (`poetry install`)
- Data synced (`python sync.py --source all --dest data/`)
- At least one model trained (see the [Getting Started Tutorial](getting-started.md))

## Part 1: Stateless Model (Feature-Based)

A stateless model receives a feature matrix `X` and binary labels `y`, and
produces win probabilities.  This is the simpler paradigm — if you have a
standard ML classifier, wrap it here.

### Step 1: Define the Config

Every model needs a Pydantic config class that extends `ModelConfig`:

```python
# my_model.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Self

import numpy as np
import pandas as pd
from pydantic import Field

from ncaa_eval.model.base import Model, ModelConfig
from ncaa_eval.model.registry import register_model


class WeightedAverageConfig(ModelConfig):
    """Hyperparameters for the weighted-average model."""

    model_name: Literal["weighted_avg"] = "weighted_avg"
    home_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    recency_decay: float = Field(default=0.95, ge=0.0, le=1.0)
```

The `model_name` field must match the name you will use with `@register_model`.

### Step 2: Implement the Model ABC

Subclass `Model` and implement all five abstract methods:

```python
@register_model("weighted_avg")
class WeightedAverageModel(Model):
    """A simple model that predicts based on weighted feature averages."""

    def __init__(self, config: WeightedAverageConfig | None = None) -> None:
        self._config = config or WeightedAverageConfig()
        self._weights: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Learn feature weights from training data."""
        # Simple example: correlation between each feature and outcome
        correlations = X.corrwith(y).fillna(0.0)
        self._weights = correlations.values

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Return P(team_a wins) for each row."""
        if self._weights is None:
            msg = "Model must be fit() before predict_proba()"
            raise RuntimeError(msg)
        # Weighted sum → sigmoid → probability
        raw = X.values @ self._weights
        probs = 1.0 / (1.0 + np.exp(-raw))
        return pd.Series(probs, index=X.index)

    def save(self, path: Path) -> None:
        """Persist model to directory."""
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(self._config.model_dump_json())
        if self._weights is not None:
            np.save(path / "weights.npy", self._weights)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Restore model from directory."""
        config = WeightedAverageConfig.model_validate_json(
            (path / "config.json").read_text()
        )
        instance = cls(config)
        weights_path = path / "weights.npy"
        if weights_path.exists():
            instance._weights = np.load(weights_path)
        return instance

    def get_config(self) -> WeightedAverageConfig:
        """Return the model's configuration."""
        return self._config
```

**Key contract:**
- `fit(X, y)` — `X` is a pandas DataFrame of numeric features, `y` is a binary
  Series (1 = team_a won, 0 = team_b won)
- `predict_proba(X)` — returns a Series of probabilities in [0, 1]
- `save(path)` / `load(path)` — persist to and restore from a directory
- `get_config()` — return the Pydantic config instance

```{note}
The backtest pipeline automatically strips metadata columns (game_id, season,
day_num, etc.) before passing `X` to stateless models.  Your `fit()` and
`predict_proba()` only see numeric feature columns.
```

### Step 3: Register and Use

The `@register_model("weighted_avg")` decorator handles registration.  To use
your model, ensure the module is imported before the CLI runs.

**Option A:** Place `my_model.py` in `src/ncaa_eval/model/` and add an import
in `src/ncaa_eval/model/__init__.py`:

```python
# In src/ncaa_eval/model/__init__.py, add:
import ncaa_eval.model.my_model  # noqa: F401
```

**Option B:** Import it in a script:

```python
import ncaa_eval.model.my_model  # registers "weighted_avg"
from ncaa_eval.model import list_models

print(list_models())
# ['elo', 'logistic_regression', 'weighted_avg', 'xgboost']
```

Then train via the CLI:

```bash
python -m ncaa_eval.cli train --model weighted_avg
```

### Step 4: Save and Load

The training pipeline calls `save()` automatically.  To manually save and load:

```python
from pathlib import Path

model = WeightedAverageModel()
# ... fit the model ...
model.save(Path("data/runs/my_run/model"))

# Later, restore it:
restored = WeightedAverageModel.load(Path("data/runs/my_run/model"))
```

```{tip}
Look at `src/ncaa_eval/model/logistic_regression.py` for a minimal (~30 lines)
reference implementation of a stateless model.
```

## Part 2: Stateful Model (Rating-Based)

A stateful model processes games sequentially and maintains internal state
(e.g., per-team ratings).  The `StatefulModel` base class provides concrete
`fit()` and `predict_proba()` implementations — you implement five hooks.

### Step 1: Define Config and Model

```python
# my_stateful_model.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Self

from ncaa_eval.ingest.schema import Game
from ncaa_eval.model.base import ModelConfig, StatefulModel
from ncaa_eval.model.registry import register_model


class SimpleRatingConfig(ModelConfig):
    """Hyperparameters for a simple win-percentage rating model."""

    model_name: Literal["simple_rating"] = "simple_rating"
    initial_rating: float = 0.5
    learning_rate: float = 0.1
    mean_reversion: float = 0.3


@register_model("simple_rating")
class SimpleRatingModel(StatefulModel):
    """A minimal rating model: tracks team win percentages with smoothing."""

    def __init__(self, config: SimpleRatingConfig | None = None) -> None:
        self._config = config or SimpleRatingConfig()
        self._ratings: dict[int, float] = {}

    def start_season(self, season: int) -> None:
        """Mean-revert ratings at the start of each season."""
        mean = self._config.initial_rating
        frac = self._config.mean_reversion
        self._ratings = {
            tid: mean * frac + rating * (1 - frac)
            for tid, rating in self._ratings.items()
        }

    def update(self, game: Game) -> None:
        """Update ratings based on game outcome."""
        lr = self._config.learning_rate
        init = self._config.initial_rating

        w_rating = self._ratings.get(game.w_team_id, init)
        l_rating = self._ratings.get(game.l_team_id, init)

        # Winner's rating increases, loser's decreases
        self._ratings[game.w_team_id] = w_rating + lr * (1.0 - w_rating)
        self._ratings[game.l_team_id] = l_rating + lr * (0.0 - l_rating)

    def _predict_one(self, team_a_id: int, team_b_id: int) -> float:
        """Return P(team_a wins) based on rating difference."""
        init = self._config.initial_rating
        a = self._ratings.get(team_a_id, init)
        b = self._ratings.get(team_b_id, init)
        # Simple sigmoid of rating difference
        diff = a - b
        return 1.0 / (1.0 + 10.0 ** (-diff / 0.2))

    def get_state(self) -> dict[str, Any]:
        """Snapshot current ratings for serialization."""
        return {"ratings": dict(self._ratings)}

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore ratings from a snapshot."""
        self._ratings = dict(state["ratings"])

    def save(self, path: Path) -> None:
        """Persist model config and state."""
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(self._config.model_dump_json())
        (path / "state.json").write_text(
            json.dumps(self.get_state())
        )

    @classmethod
    def load(cls, path: Path) -> Self:
        """Restore model from saved files."""
        config = SimpleRatingConfig.model_validate_json(
            (path / "config.json").read_text()
        )
        instance = cls(config)
        state = json.loads((path / "state.json").read_text())
        instance.set_state(state)
        return instance

    def get_config(self) -> SimpleRatingConfig:
        return self._config
```

### Step 2: Understand the Hooks

The `StatefulModel` base class calls your hooks in this order:

1. **`start_season(season)`** — Called before the first game of each season.
   Use this to mean-revert ratings or reset accumulators.

2. **`update(game)`** — Called once per game, in chronological order.  The
   `Game` object contains:
   - `w_team_id` / `l_team_id` — winner and loser team IDs
   - `w_score` / `l_score` — final scores
   - `loc` — "H" (home), "A" (away), or "N" (neutral)
   - `num_ot` — number of overtime periods
   - `is_tournament` — True for NCAA tournament games

3. **`_predict_one(team_a_id, team_b_id)`** — Return P(team_a wins) using
   current internal ratings.  The base class calls this for each game in the
   test set.

4. **`get_state()` / `set_state(state)`** — Serialize and restore internal
   state.  Used for model persistence and for the evaluation engine to snapshot
   state between folds.

```{warning}
The `fit()` and `predict_proba()` methods are provided by `StatefulModel` —
do **not** override them.  They handle the game reconstruction, season
iteration, and per-row prediction logic automatically.
```

### Step 3: Train and Evaluate

Register and train just like a stateless model:

```bash
python -m ncaa_eval.cli train --model simple_rating
```

```{tip}
See `src/ncaa_eval/model/elo.py` for the full reference implementation of a
stateful model with margin-of-victory adjustments, variable K-factors, and
home-court advantage.
```

## Running Evaluation with a Custom Model

Once trained, your model's run artifacts appear in `data/runs/<run_id>/`.  The
dashboard automatically picks them up:

```bash
streamlit run dashboard/app.py
```

Select your model run in the sidebar to see its metrics on the Leaderboard and
Deep Dive pages.

To run a backtest programmatically:

```python
from ncaa_eval.evaluation.backtest import run_backtest, DEFAULT_METRICS
from ncaa_eval.transform.feature_serving import StatefulFeatureServer, FeatureConfig

# Create feature server
config = FeatureConfig()
server = StatefulFeatureServer(data_dir=Path("data/"), config=config)

# Run backtest
result = run_backtest(
    model=my_model,
    feature_server=server,
    seasons=list(range(2015, 2026)),
    mode="batch",   # "stateful" for StatefulModel
)

# Print per-year metrics
print(result.summary)
```

## Summary

| Step | Stateless (`Model`) | Stateful (`StatefulModel`) |
|------|--------------------|-----------------------------|
| Config | Extend `ModelConfig` | Extend `ModelConfig` |
| Core methods | `fit`, `predict_proba` | `update`, `_predict_one`, `start_season` |
| State mgmt | N/A | `get_state`, `set_state` |
| Persistence | `save`, `load` | `save`, `load` |
| Config access | `get_config` | `get_config` |
| Register | `@register_model("name")` | `@register_model("name")` |
| Train | `python -m ncaa_eval.cli train --model name` | Same |

## Next Steps

- **Add a custom metric** — See the [Custom Metric Tutorial](custom-metric.md)
- **Compare models** — Train multiple models and use the Leaderboard to compare
- **Explore the reference implementations:**
  - `src/ncaa_eval/model/logistic_regression.py` — minimal stateless model
  - `src/ncaa_eval/model/elo.py` — full stateful model
  - `src/ncaa_eval/model/xgboost_model.py` — production stateless model
