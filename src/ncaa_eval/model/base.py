"""Model abstract base classes and configuration.

Defines the ``Model`` ABC, the ``StatefulModel`` template subclass for
sequential-update models, and the ``ModelConfig`` Pydantic base used by
every model's hyperparameter schema.
"""

from __future__ import annotations

import abc
import datetime
from pathlib import Path
from typing import Any, Literal, Self

import pandas as pd  # type: ignore[import-untyped]
from pydantic import BaseModel

from ncaa_eval.ingest.schema import Game


class ModelConfig(BaseModel):
    """Base configuration shared by all model implementations.

    Subclasses add model-specific hyperparameters as additional fields.
    """

    model_name: str


class Model(abc.ABC):
    """Abstract base class for all NCAA prediction models.

    Every model — stateful or stateless — must implement these five
    methods so that the training CLI, evaluation engine, and persistence
    layer can treat all models uniformly.
    """

    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model on feature matrix *X* and labels *y*."""
        ...

    @abc.abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Return P(team_a wins) in [0, 1] for each row of *X*."""
        ...

    @abc.abstractmethod
    def save(self, path: Path) -> None:
        """Persist the trained model to *path*."""
        ...

    @classmethod
    @abc.abstractmethod
    def load(cls, path: Path) -> Self:
        """Load a previously-saved model from *path*."""
        ...

    @abc.abstractmethod
    def get_config(self) -> ModelConfig:
        """Return the Pydantic-validated configuration for this model."""
        ...


# ---------------------------------------------------------------------------
# Location encoding helpers
# ---------------------------------------------------------------------------
_LOC_FROM_ENCODING: dict[int, Literal["H", "A", "N"]] = {1: "H", -1: "A", 0: "N"}


class StatefulModel(Model):
    """Template base for models that process games sequentially.

    Concrete methods ``fit`` and ``predict_proba`` are provided as
    template methods.  Subclasses implement the abstract hooks:

    * ``update(game)`` — absorb a single game result
    * ``_predict_one(team_a_id, team_b_id)`` — return P(team_a wins)
    * ``start_season(season)`` — reset / prepare for a new season
    * ``get_state()`` / ``set_state(state)`` — snapshot / restore ratings
    """

    # ------------------------------------------------------------------
    # Concrete template methods
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Reconstruct games from *X*/*y* and update sequentially."""
        games = self._to_games(X, y)
        current_season: int | None = None
        for game in games:
            if game.season != current_season:
                self.start_season(game.season)
                current_season = game.season
            self.update(game)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Call ``_predict_one`` per row using ``itertuples``."""
        preds: list[float] = [self._predict_one(row.team_a_id, row.team_b_id) for row in X.itertuples()]
        return pd.Series(preds, index=X.index)

    # ------------------------------------------------------------------
    # Concrete helper
    # ------------------------------------------------------------------

    @staticmethod
    def _to_games(X: pd.DataFrame, y: pd.Series) -> list[Game]:
        """Reconstruct :class:`Game` objects from the feature DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with columns: ``team_a_id``, ``team_b_id``,
            ``season``, ``day_num``, ``date``, ``loc_encoding``,
            ``game_id``, ``is_tournament``.  Optionally ``w_score``,
            ``l_score``, ``num_ot``.
        y : pd.Series
            Binary label — ``1`` (or ``True``) means team_a won.
        """
        # Hoist column-existence checks outside the loop (O(1) each, not O(n))
        has_scores = "w_score" in X.columns and "l_score" in X.columns
        has_num_ot = "num_ot" in X.columns

        games: list[Game] = []
        for row in X.itertuples():
            idx = row.Index
            team_a_won = bool(y.loc[idx])

            team_a_id = int(row.team_a_id)
            team_b_id = int(row.team_b_id)

            if team_a_won:
                w_team_id, l_team_id = team_a_id, team_b_id
            else:
                w_team_id, l_team_id = team_b_id, team_a_id

            loc_enc = int(row.loc_encoding)
            if loc_enc not in _LOC_FROM_ENCODING:
                msg = f"Unknown loc_encoding {loc_enc!r}; expected one of {sorted(_LOC_FROM_ENCODING)}"
                raise ValueError(msg)
            loc = _LOC_FROM_ENCODING[loc_enc]

            # Scores: use real values if present, else dummy
            w_score = int(row.w_score) if has_scores else 1
            l_score = int(row.l_score) if has_scores else 0
            num_ot = int(row.num_ot) if has_num_ot else 0

            # Date handling: pd.isna covers None, float NaN, and pd.NaT uniformly
            raw_date = row.date
            date_val: datetime.date | None = None
            if not pd.isna(raw_date):
                date_val = pd.Timestamp(raw_date).date()

            games.append(
                Game(
                    game_id=str(row.game_id),
                    season=int(row.season),
                    day_num=int(row.day_num),
                    date=date_val,
                    w_team_id=w_team_id,
                    l_team_id=l_team_id,
                    w_score=w_score,
                    l_score=l_score,
                    loc=loc,
                    num_ot=num_ot,
                    is_tournament=bool(row.is_tournament),
                )
            )
        return games

    # ------------------------------------------------------------------
    # Abstract hooks for subclasses
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _predict_one(self, team_a_id: int, team_b_id: int) -> float:
        """Return P(team_a wins) given team IDs."""
        ...

    @abc.abstractmethod
    def update(self, game: Game) -> None:
        """Absorb the result of a single game."""
        ...

    @abc.abstractmethod
    def start_season(self, season: int) -> None:
        """Called before the first game of each season."""
        ...

    @abc.abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Return a serialisable snapshot of internal ratings."""
        ...

    @abc.abstractmethod
    def set_state(self, state: dict[str, Any]) -> None:
        """Restore internal ratings from a snapshot."""
        ...
