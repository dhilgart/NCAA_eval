"""Unit tests for model base classes: ModelConfig, Model ABC, StatefulModel."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Self

import pandas as pd  # type: ignore[import-untyped]
import pytest
from pydantic import ValidationError

from ncaa_eval.ingest.schema import Game
from ncaa_eval.model.base import Model, ModelConfig, StatefulModel

# ---------------------------------------------------------------------------
# Task 1: ModelConfig tests (AC #5)
# ---------------------------------------------------------------------------


class TestModelConfig:
    """Test ModelConfig Pydantic validation and serialization."""

    def test_basic_creation(self) -> None:
        cfg = ModelConfig(model_name="test")
        assert cfg.model_name == "test"

    def test_json_round_trip(self) -> None:
        cfg = ModelConfig(model_name="round_trip_test")
        json_str = cfg.model_dump_json()
        restored = ModelConfig.model_validate_json(json_str)
        assert restored == cfg

    def test_missing_model_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Task 2: Model ABC tests (AC #1, #2, #3, #4)
# ---------------------------------------------------------------------------


class TestModelABC:
    """Test that Model ABC enforces abstract methods."""

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError, match="abstract method"):
            Model()  # type: ignore[abstract]

    def test_requires_all_abstract_methods(self) -> None:
        """A partial implementation should still be abstract."""

        class PartialModel(Model):
            def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
                pass

        with pytest.raises(TypeError, match="abstract method"):
            PartialModel()  # type: ignore[abstract]

    def test_complete_implementation_can_instantiate(self) -> None:
        """A complete implementation should be instantiable."""

        class ConcreteModel(Model):
            def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
                pass

            def predict_proba(self, X: pd.DataFrame) -> pd.Series:
                return pd.Series(dtype=float)

            def save(self, path: Path) -> None:
                pass

            @classmethod
            def load(cls, path: Path) -> Self:
                return cls()

            def get_config(self) -> ModelConfig:
                return ModelConfig(model_name="concrete")

        model = ConcreteModel()
        assert model.get_config().model_name == "concrete"


# ---------------------------------------------------------------------------
# Task 3: StatefulModel tests (AC #6)
# ---------------------------------------------------------------------------


class _DummyStatefulModel(StatefulModel):
    """Minimal concrete StatefulModel for testing template methods."""

    def __init__(self) -> None:
        self.seasons_started: list[int] = []
        self.games_updated: list[Game] = []
        self._ratings: dict[int, float] = {}

    def _predict_one(self, team_a_id: int, team_b_id: int) -> float:
        return 0.5

    def update(self, game: Game) -> None:
        self.games_updated.append(game)

    def start_season(self, season: int) -> None:
        self.seasons_started.append(season)

    def save(self, path: Path) -> None:
        pass

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls()

    def get_config(self) -> ModelConfig:
        return ModelConfig(model_name="dummy_stateful")

    def get_state(self) -> dict[str, Any]:
        return {"ratings": self._ratings}

    def set_state(self, state: dict[str, Any]) -> None:
        self._ratings = state["ratings"]


def _make_game_dataframe(
    *,
    n_games: int = 3,
    seasons: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create a minimal feature DataFrame and label Series for testing."""
    if seasons is None:
        seasons = [2020] * n_games
    assert len(seasons) == n_games

    data = {
        "team_a_id": [100 + i for i in range(n_games)],
        "team_b_id": [200 + i for i in range(n_games)],
        "season": seasons,
        "day_num": list(range(10, 10 + n_games)),
        "date": [datetime.date(2020, 1, 10 + i) for i in range(n_games)],
        "loc_encoding": [1, -1, 0][:n_games],
        "game_id": [f"game_{i}" for i in range(n_games)],
        "is_tournament": [False] * n_games,
        "w_score": [80 + i for i in range(n_games)],
        "l_score": [70 + i for i in range(n_games)],
        "num_ot": [0] * n_games,
    }
    X = pd.DataFrame(data)
    y = pd.Series([True, False, True][:n_games])
    return X, y


class TestStatefulModel:
    """Test StatefulModel template methods and abstract hook enforcement."""

    def test_cannot_instantiate_without_hooks(self) -> None:
        with pytest.raises(TypeError, match="abstract method"):
            StatefulModel()  # type: ignore[abstract]

    def test_to_games_reconstructs_correctly(self) -> None:
        X, y = _make_game_dataframe(n_games=3)
        games = StatefulModel._to_games(X, y)

        assert len(games) == 3

        # Game 0: team_a_won=True, so w=team_a, l=team_b
        assert games[0].w_team_id == 100
        assert games[0].l_team_id == 200
        assert games[0].loc == "H"  # loc_encoding=1

        # Game 1: team_a_won=False, so w=team_b, l=team_a
        assert games[1].w_team_id == 201
        assert games[1].l_team_id == 101
        assert games[1].loc == "A"  # loc_encoding=-1

        # Game 2: team_a_won=True
        assert games[2].w_team_id == 102
        assert games[2].l_team_id == 202
        assert games[2].loc == "N"  # loc_encoding=0

    def test_to_games_dummy_scores_when_absent(self) -> None:
        X, y = _make_game_dataframe(n_games=1)
        X = X.drop(columns=["w_score", "l_score", "num_ot"])
        games = StatefulModel._to_games(X, y)
        assert games[0].w_score == 1
        assert games[0].l_score == 0
        assert games[0].num_ot == 0

    def test_fit_calls_start_season_at_boundaries(self) -> None:
        model = _DummyStatefulModel()
        X, y = _make_game_dataframe(n_games=3, seasons=[2019, 2019, 2020])
        model.fit(X, y)

        assert model.seasons_started == [2019, 2020]
        assert len(model.games_updated) == 3

    def test_fit_calls_update_per_game(self) -> None:
        model = _DummyStatefulModel()
        X, y = _make_game_dataframe(n_games=3)
        model.fit(X, y)
        assert len(model.games_updated) == 3

    def test_predict_proba_dispatches_per_row(self) -> None:
        model = _DummyStatefulModel()
        X, y = _make_game_dataframe(n_games=3)
        preds = model.predict_proba(X)
        assert len(preds) == 3
        assert all(p == 0.5 for p in preds)
        assert list(preds.index) == list(X.index)

    def test_get_set_state_round_trip(self) -> None:
        model = _DummyStatefulModel()
        model._ratings = {1: 1500.0, 2: 1600.0}
        state = model.get_state()
        new_model = _DummyStatefulModel()
        new_model.set_state(state)
        assert new_model._ratings == {1: 1500.0, 2: 1600.0}
