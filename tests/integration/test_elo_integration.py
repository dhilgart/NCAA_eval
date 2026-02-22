"""Integration tests for the Elo feature engine.

Validates walk-forward temporal integrity, multi-season continuity,
batch/stateful mode equivalence, and feature serving round-trip.
"""

from __future__ import annotations

import datetime
from typing import Literal
from unittest.mock import MagicMock

import pandas as pd  # type: ignore[import-untyped]
import pytest

from ncaa_eval.ingest.schema import Game
from ncaa_eval.transform.elo import EloConfig, EloFeatureEngine
from ncaa_eval.transform.feature_serving import FeatureConfig, StatefulFeatureServer
from ncaa_eval.transform.normalization import ConferenceLookup
from ncaa_eval.transform.serving import ChronologicalDataServer, SeasonGames

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_game(  # noqa: PLR0913
    game_id: str = "1",
    season: int = 2023,
    day_num: int = 10,
    w_team_id: int = 101,
    l_team_id: int = 102,
    w_score: int = 75,
    l_score: int = 60,
    loc: Literal["H", "A", "N"] = "N",
    num_ot: int = 0,
    is_tournament: bool = False,
) -> Game:
    return Game(
        game_id=game_id,
        season=season,
        day_num=day_num,
        date=datetime.date(season - 1, 11, 1) + datetime.timedelta(days=day_num),
        w_team_id=w_team_id,
        l_team_id=l_team_id,
        w_score=w_score,
        l_score=l_score,
        loc=loc,
        num_ot=num_ot,
        is_tournament=is_tournament,
    )


def _mock_data_server(games: list[Game], year: int = 2023) -> MagicMock:
    server = MagicMock(spec=ChronologicalDataServer)
    server.get_chronological_season.return_value = SeasonGames(year=year, games=games, has_tournament=True)
    return server


def _elo_feature_config() -> FeatureConfig:
    """Minimal config with only Elo enabled."""
    return FeatureConfig(
        sequential_windows=(),
        ewma_alphas=(),
        graph_features_enabled=False,
        batch_rating_types=(),
        ordinal_composite=None,
        matchup_deltas=True,
        calibration_method=None,
        elo_enabled=True,
    )


# ── 6.1: Walk-forward temporal integrity ─────────────────────────────────────


class TestEloTemporalIntegrity:
    """Verify Elo rating for game at day_num D reflects only games with day_num < D."""

    def test_elo_before_rating_uses_only_prior_games(self) -> None:
        """Each game's Elo feature reflects only earlier games in the season."""
        games = [
            _make_game(game_id="1", day_num=10, w_team_id=101, l_team_id=102, w_score=75, l_score=60),
            _make_game(game_id="2", day_num=20, w_team_id=101, l_team_id=103, w_score=70, l_score=65),
            _make_game(game_id="3", day_num=30, w_team_id=102, l_team_id=103, w_score=80, l_score=55),
        ]
        engine = EloFeatureEngine(EloConfig())
        result = engine.process_season(games, season=2023)

        # Game 1: both teams start at 1500 (no prior games)
        assert result.iloc[0]["elo_w_before"] == 1500.0
        assert result.iloc[0]["elo_l_before"] == 1500.0

        # Game 2: team 101 has played 1 game (won), so rating > 1500
        assert result.iloc[1]["elo_w_before"] > 1500.0
        # Team 103 is new → still at initial_rating
        assert result.iloc[1]["elo_l_before"] == 1500.0

        # Game 3: team 102 lost game 1, so rating < 1500
        assert result.iloc[2]["elo_w_before"] < 1500.0
        # Team 103 lost game 2, so rating < 1500
        assert result.iloc[2]["elo_l_before"] < 1500.0

    def test_rating_never_incorporates_current_game(self) -> None:
        """The before-rating for a game does NOT include that game's result."""
        games = [
            _make_game(game_id="1", day_num=10, w_team_id=101, l_team_id=102, w_score=90, l_score=50),
        ]
        engine = EloFeatureEngine(EloConfig())
        result = engine.process_season(games, season=2023)

        # Before-rating for the first game is the initial rating
        assert result.iloc[0]["elo_w_before"] == 1500.0
        assert result.iloc[0]["elo_l_before"] == 1500.0

        # After processing, the engine has been updated
        assert engine.get_rating(101) != 1500.0


# ── 6.2: Multi-season continuity ─────────────────────────────────────────────


class TestEloMultiSeasonContinuity:
    """Verify ratings carry forward across seasons with mean-reversion."""

    def test_ratings_carry_forward_with_reversion(self) -> None:
        """Season 2 start reflects mean-reverted season 1 final ratings."""
        s1_games = [
            _make_game(
                game_id="1", season=2023, day_num=10, w_team_id=101, l_team_id=102, w_score=85, l_score=60
            ),
            _make_game(
                game_id="2", season=2023, day_num=20, w_team_id=101, l_team_id=103, w_score=80, l_score=55
            ),
        ]
        s2_games = [
            _make_game(game_id="3", season=2024, day_num=10, w_team_id=101, l_team_id=102),
        ]

        engine = EloFeatureEngine(EloConfig(mean_reversion_fraction=0.25))
        engine.process_season(s1_games, season=2023)

        # Capture post-season-1 ratings before reversion
        r101_s1_end = engine.get_rating(101)
        r102_s1_end = engine.get_rating(102)

        result_s2 = engine.process_season(s2_games, season=2024)

        # Season 2 ratings reflect mean-reversion, not raw season 1 end
        assert result_s2.iloc[0]["elo_w_before"] != r101_s1_end
        assert result_s2.iloc[0]["elo_l_before"] != r102_s1_end

        # After reversion: old + 0.25*(mean - old) = old*0.75 + mean*0.25
        # The winner (101) should be pulled DOWN toward mean
        assert result_s2.iloc[0]["elo_w_before"] < r101_s1_end

    def test_conference_mean_reversion(self) -> None:
        """With ConferenceLookup, teams regress toward conference mean."""
        lookup = ConferenceLookup(
            {
                (2024, 101): "ACC",
                (2024, 102): "ACC",
                (2024, 201): "B10",
                (2024, 202): "B10",
            }
        )
        engine = EloFeatureEngine(EloConfig(mean_reversion_fraction=0.25), conference_lookup=lookup)

        # Simulate season 1 with different conferences
        s1_games = [
            _make_game(
                game_id="1", season=2023, day_num=10, w_team_id=101, l_team_id=102, w_score=80, l_score=60
            ),
            _make_game(
                game_id="2", season=2023, day_num=20, w_team_id=201, l_team_id=202, w_score=80, l_score=60
            ),
        ]
        engine.process_season(s1_games, season=2023)

        r101_end = engine.get_rating(101)
        r102_end = engine.get_rating(102)

        s2_games = [
            _make_game(game_id="3", season=2024, day_num=10, w_team_id=101, l_team_id=201),
        ]
        result_s2 = engine.process_season(s2_games, season=2024)

        # After reversion, ACC teams should regress toward ACC mean
        acc_mean = (r101_end + r102_end) / 2
        expected_101 = r101_end + 0.25 * (acc_mean - r101_end)
        assert result_s2.iloc[0]["elo_w_before"] == pytest.approx(expected_101)


# ── 6.3: Batch/stateful equivalence ──────────────────────────────────────────


class TestEloBatchStatefulEquivalence:
    """Verify batch and stateful modes produce identical delta_elo values."""

    def test_delta_elo_matches_between_modes(self) -> None:
        """Both modes produce identical delta_elo for the same season."""
        games = [
            _make_game(game_id="1", day_num=10, w_team_id=101, l_team_id=102, w_score=75, l_score=60),
            _make_game(game_id="2", day_num=20, w_team_id=103, l_team_id=104, w_score=80, l_score=55),
            _make_game(game_id="3", day_num=30, w_team_id=101, l_team_id=103, w_score=70, l_score=65),
        ]
        ds = _mock_data_server(games)

        # Fresh engines for each mode (they mutate state)
        engine_batch = EloFeatureEngine(EloConfig())
        engine_stateful = EloFeatureEngine(EloConfig())

        cfg = _elo_feature_config()

        server_batch = StatefulFeatureServer(config=cfg, data_server=ds, elo_engine=engine_batch)
        batch = server_batch.serve_season_features(2023, mode="batch")

        server_stateful = StatefulFeatureServer(config=cfg, data_server=ds, elo_engine=engine_stateful)
        stateful = server_stateful.serve_season_features(2023, mode="stateful")

        assert list(batch["delta_elo"]) == pytest.approx(list(stateful["delta_elo"]))

    def test_elo_a_elo_b_match_between_modes(self) -> None:
        """Raw elo_a and elo_b columns match between batch and stateful modes."""
        games = [
            _make_game(game_id="1", day_num=10, w_team_id=101, l_team_id=102, w_score=75, l_score=60),
            _make_game(game_id="2", day_num=20, w_team_id=101, l_team_id=102, w_score=70, l_score=65),
        ]
        ds = _mock_data_server(games)

        engine_batch = EloFeatureEngine(EloConfig())
        engine_stateful = EloFeatureEngine(EloConfig())

        cfg = _elo_feature_config()

        server_batch = StatefulFeatureServer(config=cfg, data_server=ds, elo_engine=engine_batch)
        batch = server_batch.serve_season_features(2023, mode="batch")

        server_stateful = StatefulFeatureServer(config=cfg, data_server=ds, elo_engine=engine_stateful)
        stateful = server_stateful.serve_season_features(2023, mode="stateful")

        assert list(batch["elo_a"]) == pytest.approx(list(stateful["elo_a"]))
        assert list(batch["elo_b"]) == pytest.approx(list(stateful["elo_b"]))


# ── 6.4: Feature serving round-trip ──────────────────────────────────────────


class TestEloFeatureServingRoundTrip:
    """StatefulFeatureServer with Elo enabled produces non-NaN delta_elo."""

    def test_non_nan_delta_elo(self) -> None:
        """delta_elo should not be NaN when Elo is enabled."""
        games = [
            _make_game(game_id="1", day_num=10, w_team_id=101, l_team_id=102),
            _make_game(game_id="2", day_num=20, w_team_id=103, l_team_id=104),
        ]
        ds = _mock_data_server(games)
        engine = EloFeatureEngine(EloConfig())
        cfg = _elo_feature_config()

        server = StatefulFeatureServer(config=cfg, data_server=ds, elo_engine=engine)
        result = server.serve_season_features(2023, mode="batch")

        assert not result["delta_elo"].isna().any()

    def test_elo_disabled_still_has_nan_delta_elo(self) -> None:
        """When Elo is disabled, delta_elo remains NaN."""
        games = [_make_game(game_id="1", day_num=10)]
        ds = _mock_data_server(games)
        cfg = FeatureConfig(
            sequential_windows=(),
            ewma_alphas=(),
            graph_features_enabled=False,
            batch_rating_types=(),
            ordinal_composite=None,
            matchup_deltas=True,
            calibration_method=None,
            elo_enabled=False,
        )
        server = StatefulFeatureServer(config=cfg, data_server=ds)
        result = server.serve_season_features(2023, mode="batch")
        assert pd.isna(result.iloc[0]["delta_elo"])

    def test_empty_season_returns_empty_df_with_elo(self) -> None:
        """Empty season with Elo enabled returns proper empty DataFrame."""
        ds = _mock_data_server([])
        engine = EloFeatureEngine(EloConfig())
        cfg = _elo_feature_config()
        server = StatefulFeatureServer(config=cfg, data_server=ds, elo_engine=engine)
        result = server.serve_season_features(2023, mode="batch")
        assert len(result) == 0
        assert "delta_elo" in result.columns

    def test_delta_elo_equals_elo_a_minus_elo_b(self) -> None:
        """delta_elo must equal elo_a − elo_b."""
        games = [
            _make_game(game_id="1", day_num=10, w_team_id=101, l_team_id=102, w_score=85, l_score=60),
            _make_game(game_id="2", day_num=20, w_team_id=101, l_team_id=103, w_score=70, l_score=65),
        ]
        ds = _mock_data_server(games)
        engine = EloFeatureEngine(EloConfig())
        cfg = _elo_feature_config()
        server = StatefulFeatureServer(config=cfg, data_server=ds, elo_engine=engine)
        result = server.serve_season_features(2023, mode="batch")

        expected_deltas = (result["elo_a"] - result["elo_b"]).tolist()
        actual_deltas = result["delta_elo"].tolist()
        assert actual_deltas == pytest.approx(expected_deltas)


# ── 6.5: Double-reversion guard ──────────────────────────────────────────────


class TestEloDoubleReversionGuard:
    """Verify that start_new_season is called exactly once per season transition.

    Scenario: stateful mode for season N updates engine state; then stateful
    mode for season N+1 should call start_new_season once, not twice.
    """

    def test_stateful_mode_two_seasons_no_double_reversion(self) -> None:
        """serve_season_features() in stateful mode across two seasons applies
        mean-reversion exactly once at the start of season 2."""
        s1_games = [
            _make_game(
                game_id="1", season=2023, day_num=10, w_team_id=101, l_team_id=102, w_score=85, l_score=60
            ),
        ]
        s2_games = [
            _make_game(game_id="2", season=2024, day_num=10, w_team_id=101, l_team_id=102),
        ]

        engine = EloFeatureEngine(EloConfig(mean_reversion_fraction=0.25))
        cfg = _elo_feature_config()

        ds_s1 = _mock_data_server(s1_games, year=2023)
        server = StatefulFeatureServer(config=cfg, data_server=ds_s1, elo_engine=engine)
        server.serve_season_features(2023, mode="stateful")

        r101_after_s1 = engine.get_rating(101)
        r102_after_s1 = engine.get_rating(102)
        global_mean = (r101_after_s1 + r102_after_s1) / 2

        # Expected after one reversion:
        expected_101 = r101_after_s1 + 0.25 * (global_mean - r101_after_s1)
        expected_102 = r102_after_s1 + 0.25 * (global_mean - r102_after_s1)

        ds_s2 = _mock_data_server(s2_games, year=2024)
        server2 = StatefulFeatureServer(config=cfg, data_server=ds_s2, elo_engine=engine)
        result_s2 = server2.serve_season_features(2024, mode="stateful")

        # The before-ratings for season 2 game 1 should reflect exactly one reversion
        assert result_s2.iloc[0]["elo_a"] == pytest.approx(expected_101)
        assert result_s2.iloc[0]["elo_b"] == pytest.approx(expected_102)

    def test_process_season_after_stateful_applies_single_reversion(self) -> None:
        """process_season() after stateful mode applies mean-reversion exactly once."""
        s1_games = [
            _make_game(
                game_id="1", season=2023, day_num=10, w_team_id=101, l_team_id=102, w_score=85, l_score=60
            ),
        ]
        s2_games = [
            _make_game(game_id="2", season=2024, day_num=10, w_team_id=101, l_team_id=102),
        ]

        engine = EloFeatureEngine(EloConfig(mean_reversion_fraction=0.25))
        cfg = _elo_feature_config()

        # Stateful mode for season 1: internally calls start_new_season if prior ratings
        # (none here, since first season)
        ds_s1 = _mock_data_server(s1_games, year=2023)
        server = StatefulFeatureServer(config=cfg, data_server=ds_s1, elo_engine=engine)
        server.serve_season_features(2023, mode="stateful")

        r101_after_s1 = engine.get_rating(101)
        r102_after_s1 = engine.get_rating(102)
        global_mean = (r101_after_s1 + r102_after_s1) / 2

        expected_101_after_one_reversion = r101_after_s1 + 0.25 * (global_mean - r101_after_s1)

        # Now call process_season directly (batch-style) for season 2
        result_s2 = engine.process_season(s2_games, season=2024)

        # process_season calls start_new_season once (ratings non-empty)
        assert result_s2.iloc[0]["elo_w_before"] == pytest.approx(expected_101_after_one_reversion)


# ── 6.6: Multi-season stateful mode via serve_season_features() ───────────────


class TestEloMultiSeasonStatefulViaServer:
    """Verify StatefulFeatureServer handles multi-season stateful calls correctly."""

    def test_multi_season_stateful_ratings_accumulate(self) -> None:
        """Ratings from season 1 persist into season 2 in stateful mode."""
        s1_games = [
            _make_game(
                game_id="1", season=2023, day_num=10, w_team_id=101, l_team_id=102, w_score=80, l_score=60
            ),
        ]
        s2_games = [
            _make_game(game_id="2", season=2024, day_num=10, w_team_id=101, l_team_id=103),
        ]

        engine = EloFeatureEngine(EloConfig(mean_reversion_fraction=0.0))  # No reversion for clarity
        cfg = _elo_feature_config()

        ds_s1 = _mock_data_server(s1_games, year=2023)
        server1 = StatefulFeatureServer(config=cfg, data_server=ds_s1, elo_engine=engine)
        server1.serve_season_features(2023, mode="stateful")
        r101_after_s1 = engine.get_rating(101)

        # With 0% reversion, season 2 start should use exact season 1 end ratings
        ds_s2 = _mock_data_server(s2_games, year=2024)
        server2 = StatefulFeatureServer(config=cfg, data_server=ds_s2, elo_engine=engine)
        result_s2 = server2.serve_season_features(2024, mode="stateful")

        # team 101 carried forward (no reversion)
        assert result_s2.iloc[0]["elo_a"] == pytest.approx(r101_after_s1)

    def test_multi_season_stateful_delta_elo_non_nan(self) -> None:
        """delta_elo is non-NaN for all games across multi-season stateful run."""
        games = [
            _make_game(game_id="1", season=2023, day_num=10, w_team_id=101, l_team_id=102),
            _make_game(game_id="2", season=2023, day_num=20, w_team_id=103, l_team_id=104),
        ]
        engine = EloFeatureEngine(EloConfig())
        cfg = _elo_feature_config()
        ds = _mock_data_server(games, year=2023)
        server = StatefulFeatureServer(config=cfg, data_server=ds, elo_engine=engine)
        result = server.serve_season_features(2023, mode="stateful")
        assert not result["delta_elo"].isna().any()
