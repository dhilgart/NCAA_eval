"""Unit tests for the EloModel reference stateful model."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest

from ncaa_eval.model import get_model, list_models
from ncaa_eval.model.base import StatefulModel
from ncaa_eval.model.elo import EloModel, EloModelConfig
from ncaa_eval.transform.elo import EloFeatureEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_game_dataframe(
    *,
    n_games: int = 3,
    seasons: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create a minimal feature DataFrame and label Series for Elo tests."""
    if seasons is None:
        seasons = [2020] * n_games
    assert len(seasons) == n_games

    data = {
        "team_a_id": [100 + i for i in range(n_games)],
        "team_b_id": [200 + i for i in range(n_games)],
        "season": seasons,
        "day_num": list(range(10, 10 + n_games)),
        "date": [datetime.date(s, 1, 10 + i) for i, s in enumerate(seasons)],
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


# ---------------------------------------------------------------------------
# Task 5.1: EloModelConfig creation and JSON round-trip
# ---------------------------------------------------------------------------


class TestEloModelConfig:
    def test_defaults(self) -> None:
        cfg = EloModelConfig()
        assert cfg.model_name == "elo"
        assert cfg.initial_rating == 1500.0
        assert cfg.k_early == 56.0
        assert cfg.k_regular == 38.0
        assert cfg.k_tournament == 47.5
        assert cfg.early_game_threshold == 20
        assert cfg.margin_exponent == 0.85
        assert cfg.max_margin == 25
        assert cfg.home_advantage_elo == 3.5
        assert cfg.mean_reversion_fraction == 0.25

    def test_custom_values(self) -> None:
        cfg = EloModelConfig(k_early=40.0, initial_rating=1600.0)
        assert cfg.k_early == 40.0
        assert cfg.initial_rating == 1600.0

    def test_json_round_trip(self) -> None:
        cfg = EloModelConfig(k_early=40.0, max_margin=30)
        json_str = cfg.model_dump_json()
        restored = EloModelConfig.model_validate_json(json_str)
        assert restored == cfg

    def test_model_name_literal(self) -> None:
        cfg = EloModelConfig()
        assert cfg.model_name == "elo"


# ---------------------------------------------------------------------------
# Task 5.2: update(game) delegates to engine and changes ratings
# ---------------------------------------------------------------------------


class TestEloModelUpdate:
    def test_update_changes_ratings(self) -> None:
        model = EloModel()
        X, y = _make_game_dataframe(n_games=1)
        model.start_season(2020)
        games = StatefulModel._to_games(X, y)
        game = games[0]

        state_before = model.get_state()
        assert len(state_before["ratings"]) == 0

        model.update(game)

        state_after = model.get_state()
        assert len(state_after["ratings"]) == 2
        # Winner rating should increase, loser should decrease
        r_winner = state_after["ratings"][game.w_team_id]
        r_loser = state_after["ratings"][game.l_team_id]
        assert r_winner > 1500.0
        assert r_loser < 1500.0


# ---------------------------------------------------------------------------
# Task 5.3: _predict_one returns correct probability
# ---------------------------------------------------------------------------


class TestPredictOne:
    def test_equal_ratings_give_50_percent(self) -> None:
        model = EloModel()
        prob = model._predict_one(1, 2)
        assert prob == pytest.approx(0.5, abs=1e-10)

    def test_higher_rated_team_favored(self) -> None:
        model = EloModel()
        # Manually set different ratings
        model._engine._ratings[1] = 1600.0
        model._engine._ratings[2] = 1400.0
        prob = model._predict_one(1, 2)
        assert prob > 0.5

    def test_lower_rated_team_unfavored(self) -> None:
        model = EloModel()
        model._engine._ratings[1] = 1400.0
        model._engine._ratings[2] = 1600.0
        prob = model._predict_one(1, 2)
        assert prob < 0.5

    def test_predict_one_matches_expected_score(self) -> None:
        """_predict_one should delegate to EloFeatureEngine.expected_score."""
        model = EloModel()
        model._engine._ratings[10] = 1550.0
        model._engine._ratings[20] = 1450.0
        prob = model._predict_one(10, 20)
        expected = EloFeatureEngine.expected_score(1550.0, 1450.0)
        assert prob == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# Task 5.4: start_season triggers mean reversion
# ---------------------------------------------------------------------------


class TestStartSeason:
    def test_start_season_applies_mean_reversion(self) -> None:
        model = EloModel()
        model._engine._ratings[1] = 1600.0
        model._engine._ratings[2] = 1400.0
        model._engine._game_counts[1] = 30
        model._engine._game_counts[2] = 30

        model.start_season(2021)

        # Ratings should move toward the mean (1500)
        assert model._engine.get_rating(1) < 1600.0
        assert model._engine.get_rating(2) > 1400.0
        # Game counts should be reset
        assert model._engine._game_counts == {}


# ---------------------------------------------------------------------------
# Task 5.5: get_state / set_state round-trip
# ---------------------------------------------------------------------------


class TestGetSetState:
    def test_round_trip(self) -> None:
        model = EloModel()
        model._engine._ratings = {1: 1550.0, 2: 1450.0}
        model._engine._game_counts = {1: 10, 2: 15}

        state = model.get_state()
        assert state["ratings"] == {1: 1550.0, 2: 1450.0}
        assert state["game_counts"] == {1: 10, 2: 15}

        new_model = EloModel()
        new_model.set_state(state)
        assert new_model.get_state() == state

    def test_state_is_independent_copy(self) -> None:
        """Modifying returned state should not affect model."""
        model = EloModel()
        model._engine._ratings = {1: 1550.0}
        model._engine._game_counts = {1: 5}

        state = model.get_state()
        state["ratings"][1] = 9999.0
        assert model._engine.get_rating(1) == 1550.0

    def test_set_state_missing_key_raises(self) -> None:
        """set_state() rejects dicts missing required keys."""
        model = EloModel()
        with pytest.raises(KeyError, match="missing required keys"):
            model.set_state({"ratings": {1: 1500.0}})  # missing game_counts

    def test_set_state_wrong_type_raises(self) -> None:
        """set_state() rejects non-dict values for ratings/game_counts."""
        model = EloModel()
        with pytest.raises(TypeError, match="must be dicts"):
            model.set_state({"ratings": [1, 2, 3], "game_counts": {}})


# ---------------------------------------------------------------------------
# Task 5.6: save / load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_round_trip(self, tmp_path: Path) -> None:
        config = EloModelConfig(k_early=40.0, initial_rating=1600.0)
        model = EloModel(config)
        model._engine._ratings = {1: 1650.0, 2: 1550.0}
        model._engine._game_counts = {1: 10, 2: 12}

        save_dir = tmp_path / "elo_model"
        model.save(save_dir)

        loaded = EloModel.load(save_dir)
        assert loaded.get_config() == config
        assert loaded.get_state() == model.get_state()

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        model = EloModel()
        save_dir = tmp_path / "nested" / "elo_model"
        model.save(save_dir)
        assert (save_dir / "config.json").exists()
        assert (save_dir / "state.json").exists()

    def test_state_json_has_string_keys(self, tmp_path: Path) -> None:
        """JSON spec requires string keys."""
        model = EloModel()
        model._engine._ratings = {1: 1550.0}
        model._engine._game_counts = {1: 5}
        save_dir = tmp_path / "elo_model"
        model.save(save_dir)

        raw = json.loads((save_dir / "state.json").read_text())
        assert "1" in raw["ratings"]
        assert "1" in raw["game_counts"]

    def test_load_restores_predictions(self, tmp_path: Path) -> None:
        """Predictions from loaded model should match original."""
        model = EloModel()
        model._engine._ratings = {1: 1600.0, 2: 1400.0}
        model._engine._game_counts = {1: 10, 2: 10}
        pred_before = model._predict_one(1, 2)

        save_dir = tmp_path / "elo_model"
        model.save(save_dir)
        loaded = EloModel.load(save_dir)
        pred_after = loaded._predict_one(1, 2)

        assert pred_before == pytest.approx(pred_after, abs=1e-12)

    def test_load_missing_state_json_raises(self, tmp_path: Path) -> None:
        """load() raises FileNotFoundError with a clear message if state.json is missing."""
        model = EloModel()
        save_dir = tmp_path / "elo_model"
        model.save(save_dir)
        (save_dir / "state.json").unlink()

        with pytest.raises(FileNotFoundError, match="state.json"):
            EloModel.load(save_dir)

    def test_load_missing_config_json_raises(self, tmp_path: Path) -> None:
        """load() raises FileNotFoundError with a clear message if config.json is missing."""
        model = EloModel()
        save_dir = tmp_path / "elo_model"
        model.save(save_dir)
        (save_dir / "config.json").unlink()

        with pytest.raises(FileNotFoundError, match="config.json"):
            EloModel.load(save_dir)


# ---------------------------------------------------------------------------
# Task 5.7: Full fit → predict_proba end-to-end
# ---------------------------------------------------------------------------


class TestFitPredictProba:
    def test_fit_then_predict(self) -> None:
        model = EloModel()
        X, y = _make_game_dataframe(n_games=3)
        model.fit(X, y)

        preds = model.predict_proba(X)
        assert len(preds) == 3
        assert all(0.0 <= p <= 1.0 for p in preds)

    def test_fit_across_seasons(self) -> None:
        model = EloModel()
        X, y = _make_game_dataframe(n_games=3, seasons=[2019, 2019, 2020])
        model.fit(X, y)

        state = model.get_state()
        assert len(state["ratings"]) > 0

    def test_fit_twice_accumulates_state(self) -> None:
        """fit() accumulates ratings — it does NOT reset state between calls.

        StatefulModel.fit() is designed for sequential accumulation.  Callers
        who need a clean slate should instantiate a new EloModel.
        """
        model = EloModel()
        X, y = _make_game_dataframe(n_games=1)
        model.fit(X, y)
        ratings_after_first = dict(model.get_state()["ratings"])

        # Second fit on the same data — ratings change further (not reset)
        model.fit(X, y)
        ratings_after_second = dict(model.get_state()["ratings"])

        # Ratings differ: second fit accumulates on top of first
        assert ratings_after_first != ratings_after_second


# ---------------------------------------------------------------------------
# Task 5.8: Plugin registration
# ---------------------------------------------------------------------------


class TestPluginRegistration:
    def test_elo_in_list_models(self) -> None:
        assert "elo" in list_models()

    def test_get_model_returns_elo_class(self) -> None:
        cls = get_model("elo")
        assert cls is EloModel


# ---------------------------------------------------------------------------
# Task 5.9: Known numeric calculation verification
# ---------------------------------------------------------------------------


class TestKnownNumericCalculation:
    def test_known_rating_update(self) -> None:
        """Verify exact numeric outcomes from a hand-calculated example.

        Game: Team 100 (home) beats Team 200, score 80-70, neutral site
        Both start at 1500, both are early-season (game count < 20).

        With loc="N" (neutral), no home adjustment:
        - expected_100 = 1 / (1 + 10^((1500-1500)/400)) = 0.5
        - margin = 10, mult = 10^0.85 ≈ 7.0795
        - K_eff = 56 × 7.0795 ≈ 396.452
        - r_100_new = 1500 + 396.452 × (1 - 0.5) ≈ 1698.23
        - r_200_new = 1500 + 396.452 × (0 - 0.5) ≈ 1301.77
        """
        model = EloModel()
        # Create a single-game DataFrame: team 100 wins over team 200 at neutral
        X = pd.DataFrame(
            {
                "team_a_id": [100],
                "team_b_id": [200],
                "season": [2020],
                "day_num": [10],
                "date": [datetime.date(2020, 1, 10)],
                "loc_encoding": [0],  # Neutral
                "game_id": ["test_game"],
                "is_tournament": [False],
                "w_score": [80],
                "l_score": [70],
                "num_ot": [0],
            }
        )
        y = pd.Series([True])  # team_a (100) won

        model.fit(X, y)
        state = model.get_state()

        # Compute expected values
        margin = 10
        mult = margin**0.85  # ≈ 7.0795
        k_eff = 56.0 * mult
        expected_winner = 1500.0 + k_eff * (1.0 - 0.5)
        expected_loser = 1500.0 + k_eff * (0.0 - 0.5)

        assert state["ratings"][100] == pytest.approx(expected_winner, abs=0.01)
        assert state["ratings"][200] == pytest.approx(expected_loser, abs=0.01)

    def test_predict_one_known_probability(self) -> None:
        """Verify P(A wins) for a known rating difference."""
        model = EloModel()
        model._engine._ratings[1] = 1600.0
        model._engine._ratings[2] = 1400.0

        prob = model._predict_one(1, 2)
        # expected = 1 / (1 + 10^((1400 - 1600)/400)) = 1 / (1 + 10^(-0.5))
        expected = 1.0 / (1.0 + 10.0 ** (-0.5))
        assert prob == pytest.approx(expected, abs=1e-10)

    def test_home_advantage_adjustment(self) -> None:
        """Verify home-court adjustment affects rating updates."""
        model_home = EloModel()
        model_neutral = EloModel()

        # Game at home: loc_encoding=1, team_a wins
        X_home = pd.DataFrame(
            {
                "team_a_id": [100],
                "team_b_id": [200],
                "season": [2020],
                "day_num": [10],
                "date": [datetime.date(2020, 1, 10)],
                "loc_encoding": [1],  # Home
                "game_id": ["test_game"],
                "is_tournament": [False],
                "w_score": [80],
                "l_score": [70],
                "num_ot": [0],
            }
        )
        y_home = pd.Series([True])

        # Same game at neutral
        X_neutral = X_home.copy()
        X_neutral["loc_encoding"] = 0
        y_neutral = pd.Series([True])

        model_home.fit(X_home, y_home)
        model_neutral.fit(X_neutral, y_neutral)

        # Home winner gets a smaller rating boost (expected was higher due to home advantage)
        r_home_winner = model_home.get_state()["ratings"][100]
        r_neutral_winner = model_neutral.get_state()["ratings"][100]
        assert r_home_winner != r_neutral_winner


class TestGetConfig:
    def test_returns_config(self) -> None:
        cfg = EloModelConfig(k_early=40.0)
        model = EloModel(cfg)
        assert model.get_config() is cfg

    def test_default_config(self) -> None:
        model = EloModel()
        cfg = model.get_config()
        assert cfg.model_name == "elo"
        assert cfg.initial_rating == 1500.0
