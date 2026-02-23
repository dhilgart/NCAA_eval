"""Elo rating model — reference stateful model for NCAA tournament prediction.

Thin wrapper around :class:`~ncaa_eval.transform.elo.EloFeatureEngine`.
All Elo math is delegated to the engine; this module adds :class:`StatefulModel`
ABC conformance, Pydantic configuration, JSON persistence, and plugin
registration.
"""

from __future__ import annotations

import dataclasses
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
        """Restore ratings and game counts from a snapshot.

        Parameters
        ----------
        state
            Must contain ``"ratings"`` (``dict[int, float]``) and
            ``"game_counts"`` (``dict[int, int]``) keys, as returned by
            :meth:`get_state`.  Keys may be ``int`` or ``str``; string keys
            are coerced to ``int`` so that JSON-decoded dicts (where all keys
            are strings) work correctly without silent rating loss.

        Raises
        ------
        KeyError
            If ``"ratings"`` or ``"game_counts"`` keys are absent.
        TypeError
            If either value is not a ``dict``.
        """
        if "ratings" not in state or "game_counts" not in state:
            missing = {"ratings", "game_counts"} - state.keys()
            msg = f"set_state() state dict missing required keys: {missing}"
            raise KeyError(msg)
        ratings = state["ratings"]
        game_counts = state["game_counts"]
        if not isinstance(ratings, dict) or not isinstance(game_counts, dict):
            msg = "set_state() 'ratings' and 'game_counts' must be dicts"
            raise TypeError(msg)
        # Coerce string keys to int so JSON-decoded dicts (all keys are str)
        # work correctly — without coercion, get_rating(team_id_int) would
        # silently return initial_rating for every team.
        # EloFeatureEngine has no public setter — direct attribute assignment is
        # intentional here.  If the engine later adds validation, these lines
        # should be replaced with the appropriate public API.
        self._engine._ratings = {int(k): float(v) for k, v in ratings.items()}
        self._engine._game_counts = {int(k): int(v) for k, v in game_counts.items()}

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
        """Reconstruct an EloModel from a saved directory.

        Raises
        ------
        FileNotFoundError
            If either ``config.json`` or ``state.json`` is missing.  A missing
            file indicates an incomplete :meth:`save` (e.g., interrupted write).
        """
        config_path = path / "config.json"
        state_path = path / "state.json"
        missing = [p for p in (config_path, state_path) if not p.exists()]
        if missing:
            missing_names = ", ".join(p.name for p in missing)
            msg = (
                f"Incomplete save at {path!r}: missing {missing_names}. "
                "The save may have been interrupted."
            )
            raise FileNotFoundError(msg)
        config = EloModelConfig.model_validate_json(config_path.read_text())
        instance = cls(config)
        raw = json.loads(state_path.read_text())
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
        """Convert Pydantic config to the frozen dataclass the engine expects.

        Uses :func:`dataclasses.fields` to derive the argument set from
        ``EloConfig`` at runtime, so any new field added to ``EloConfig`` is
        automatically included — without requiring a manual update here.
        ``EloModelConfig`` must keep its fields in sync with ``EloConfig``.
        """
        elo_field_names = {f.name for f in dataclasses.fields(EloConfig)}
        kwargs = {k: v for k, v in config.model_dump().items() if k in elo_field_names}
        return EloConfig(**kwargs)
