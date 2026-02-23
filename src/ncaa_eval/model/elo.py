"""Elo rating model — reference stateful model for NCAA tournament prediction.

Thin wrapper around :class:`~ncaa_eval.transform.elo.EloFeatureEngine`.
All Elo math is delegated to the engine; this module adds :class:`StatefulModel`
ABC conformance, Pydantic configuration, JSON persistence, and plugin
registration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Self

from ncaa_eval.ingest.schema import Game
from ncaa_eval.model.base import ModelConfig, StatefulModel
from ncaa_eval.model.registry import register_model
from ncaa_eval.transform.elo import EloConfig, EloFeatureEngine


class EloModelConfig(ModelConfig):
    """Pydantic configuration for the Elo model.

    Fields and defaults mirror :class:`~ncaa_eval.transform.elo.EloConfig`.
    """

    model_name: Literal["elo"] = "elo"
    initial_rating: float = 1500.0
    k_early: float = 56.0
    k_regular: float = 38.0
    k_tournament: float = 47.5
    early_game_threshold: int = 20
    margin_exponent: float = 0.85
    max_margin: int = 25
    home_advantage_elo: float = 3.5
    mean_reversion_fraction: float = 0.25


@register_model("elo")
class EloModel(StatefulModel):
    """Elo rating model wrapping :class:`EloFeatureEngine`."""

    def __init__(self, config: EloModelConfig | None = None) -> None:
        self._config = config or EloModelConfig()
        self._engine = EloFeatureEngine(self._to_elo_config(self._config))

    # ------------------------------------------------------------------
    # StatefulModel abstract hooks
    # ------------------------------------------------------------------

    def update(self, game: Game) -> None:
        """Delegate game processing to the engine."""
        self._engine.update_game(
            w_team_id=game.w_team_id,
            l_team_id=game.l_team_id,
            w_score=game.w_score,
            l_score=game.l_score,
            loc=game.loc,
            is_tournament=game.is_tournament,
            num_ot=game.num_ot,
        )

    def start_season(self, season: int) -> None:
        """Delegate season transition to the engine."""
        self._engine.start_new_season(season)

    def _predict_one(self, team_a_id: int, team_b_id: int) -> float:
        """Return P(team_a wins) using the Elo expected-score formula."""
        r_a = self._engine.get_rating(team_a_id)
        r_b = self._engine.get_rating(team_b_id)
        return EloFeatureEngine.expected_score(r_a, r_b)

    def get_state(self) -> dict[str, Any]:
        """Return ratings and game counts as a serialisable snapshot."""
        return {
            "ratings": self._engine.get_all_ratings(),
            "game_counts": dict(self._engine._game_counts),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore ratings and game counts from a snapshot."""
        self._engine._ratings = dict(state["ratings"])
        self._engine._game_counts = dict(state["game_counts"])

    # ------------------------------------------------------------------
    # Model ABC: persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """JSON-dump config and state to *path* directory."""
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(self._config.model_dump_json())
        state = self.get_state()
        # JSON keys must be strings
        serialisable = {
            "ratings": {str(k): v for k, v in state["ratings"].items()},
            "game_counts": {str(k): v for k, v in state["game_counts"].items()},
        }
        (path / "state.json").write_text(json.dumps(serialisable))

    @classmethod
    def load(cls, path: Path) -> Self:
        """Reconstruct an EloModel from a saved directory."""
        config = EloModelConfig.model_validate_json((path / "config.json").read_text())
        instance = cls(config)
        raw = json.loads((path / "state.json").read_text())
        state = {
            "ratings": {int(k): v for k, v in raw["ratings"].items()},
            "game_counts": {int(k): v for k, v in raw["game_counts"].items()},
        }
        instance.set_state(state)
        return instance

    # ------------------------------------------------------------------
    # Model ABC: config
    # ------------------------------------------------------------------

    def get_config(self) -> EloModelConfig:
        """Return the Pydantic-validated configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_elo_config(config: EloModelConfig) -> EloConfig:
        """Convert Pydantic config to the frozen dataclass the engine expects."""
        return EloConfig(
            initial_rating=config.initial_rating,
            k_early=config.k_early,
            k_regular=config.k_regular,
            k_tournament=config.k_tournament,
            early_game_threshold=config.early_game_threshold,
            margin_exponent=config.margin_exponent,
            max_margin=config.max_margin,
            home_advantage_elo=config.home_advantage_elo,
            mean_reversion_fraction=config.mean_reversion_fraction,
        )
