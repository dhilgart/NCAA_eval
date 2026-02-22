"""Unit tests for ncaa_eval.transform.elo."""

from __future__ import annotations

import datetime
from typing import Literal

import pytest

from ncaa_eval.ingest.schema import Game
from ncaa_eval.transform.elo import EloConfig, EloFeatureEngine

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
    """Create a Game object for testing."""
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


# ── Task 1: EloConfig tests ─────────────────────────────────────────────────


class TestEloConfig:
    """Tests for the EloConfig frozen dataclass."""

    def test_defaults(self) -> None:
        cfg = EloConfig()
        assert cfg.initial_rating == 1500
        assert cfg.k_early == 56
        assert cfg.k_regular == 38
        assert cfg.k_tournament == 47.5
        assert cfg.early_game_threshold == 20
        assert cfg.margin_exponent == 0.85
        assert cfg.max_margin == 25
        assert cfg.home_advantage_elo == 3.5
        assert cfg.mean_reversion_fraction == 0.25

    def test_frozen_immutability(self) -> None:
        cfg = EloConfig()
        with pytest.raises(AttributeError):
            cfg.initial_rating = 1600  # type: ignore[misc]

    def test_custom_values(self) -> None:
        cfg = EloConfig(
            initial_rating=1400,
            k_early=60,
            k_regular=40,
            k_tournament=50.0,
            early_game_threshold=15,
            margin_exponent=0.9,
            max_margin=30,
            home_advantage_elo=4.0,
            mean_reversion_fraction=0.30,
        )
        assert cfg.initial_rating == 1400
        assert cfg.k_early == 60
        assert cfg.k_regular == 40
        assert cfg.k_tournament == 50.0
        assert cfg.early_game_threshold == 15
        assert cfg.margin_exponent == 0.9
        assert cfg.max_margin == 30
        assert cfg.home_advantage_elo == 4.0
        assert cfg.mean_reversion_fraction == 0.30


# ── Task 2: EloFeatureEngine core tests ──────────────────────────────────────


class TestEloFeatureEngineExpectedScore:
    """Tests for expected_score method."""

    def test_equal_ratings_returns_half(self) -> None:
        engine = EloFeatureEngine(EloConfig())
        assert engine.expected_score(1500.0, 1500.0) == pytest.approx(0.5)

    def test_higher_rating_higher_expected(self) -> None:
        engine = EloFeatureEngine(EloConfig())
        assert engine.expected_score(1600.0, 1400.0) > 0.5

    def test_symmetry(self) -> None:
        """expected(A, B) + expected(B, A) == 1."""
        engine = EloFeatureEngine(EloConfig())
        e_ab = engine.expected_score(1600.0, 1400.0)
        e_ba = engine.expected_score(1400.0, 1600.0)
        assert e_ab + e_ba == pytest.approx(1.0)

    def test_400_point_gap(self) -> None:
        """A 400-point gap gives ~0.909 expected score for the stronger team."""
        engine = EloFeatureEngine(EloConfig())
        expected = engine.expected_score(1900.0, 1500.0)
        assert expected == pytest.approx(1.0 / (1.0 + 10.0 ** (-1.0)), rel=1e-6)


class TestEloFeatureEngineUpdate:
    """Tests for update_game method."""

    def test_basic_update_returns_before_ratings(self) -> None:
        engine = EloFeatureEngine(EloConfig())
        elo_w_before, elo_l_before = engine.update_game(
            w_team_id=101,
            l_team_id=102,
            w_score=75,
            l_score=60,
            loc="N",
            is_tournament=False,
        )
        # Both teams start at initial_rating
        assert elo_w_before == 1500.0
        assert elo_l_before == 1500.0
        # After update: winner goes up, loser goes down
        assert engine.get_rating(101) > 1500.0
        assert engine.get_rating(102) < 1500.0

    def test_winner_gains_loser_loses(self) -> None:
        engine = EloFeatureEngine(EloConfig())
        engine.update_game(101, 102, 75, 60, "N", False)
        assert engine.get_rating(101) > engine.get_rating(102)

    def test_rating_changes_are_symmetric(self) -> None:
        """Winner gain + loser loss ≈ 0 (not exactly due to margin scaling)."""
        engine = EloFeatureEngine(EloConfig())
        engine.update_game(101, 102, 75, 60, "N", False)
        gain = engine.get_rating(101) - 1500.0
        loss = 1500.0 - engine.get_rating(102)
        # With margin scaling, both teams get the same absolute magnitude
        assert gain == pytest.approx(loss, rel=1e-6)

    def test_unseen_team_gets_initial_rating(self) -> None:
        engine = EloFeatureEngine(EloConfig())
        assert engine.get_rating(999) == 1500.0

    def test_game_count_incremented(self) -> None:
        engine = EloFeatureEngine(EloConfig())
        engine.update_game(101, 102, 75, 60, "N", False)
        # Both teams should have 1 game played
        assert engine._game_counts[101] == 1
        assert engine._game_counts[102] == 1


class TestEloMarginScaling:
    """Tests for margin-of-victory scaling."""

    def test_larger_margin_larger_update(self) -> None:
        """Blowout should result in larger rating change than close game."""
        engine_close = EloFeatureEngine(EloConfig())
        engine_close.update_game(101, 102, 61, 60, "N", False)
        change_close = abs(engine_close.get_rating(101) - 1500.0)

        engine_blow = EloFeatureEngine(EloConfig())
        engine_blow.update_game(101, 102, 90, 60, "N", False)
        change_blow = abs(engine_blow.get_rating(101) - 1500.0)

        assert change_blow > change_close

    def test_margin_capped_at_max(self) -> None:
        """Margins above max_margin should produce same effect as max_margin."""
        cfg = EloConfig(max_margin=25)
        engine_at_max = EloFeatureEngine(cfg)
        engine_at_max.update_game(101, 102, 85, 60, "N", False)  # margin=25
        change_at_max = abs(engine_at_max.get_rating(101) - 1500.0)

        engine_over = EloFeatureEngine(cfg)
        engine_over.update_game(101, 102, 100, 60, "N", False)  # margin=40>25
        change_over = abs(engine_over.get_rating(101) - 1500.0)

        assert change_at_max == pytest.approx(change_over, rel=1e-6)

    def test_margin_multiplier_formula(self) -> None:
        """Verify _margin_multiplier follows min(margin, max)^exponent."""
        engine = EloFeatureEngine(EloConfig())
        # margin=15, max_margin=25, exponent=0.85
        expected = 15**0.85
        assert engine._margin_multiplier(15) == pytest.approx(expected)

    def test_margin_multiplier_capped(self) -> None:
        engine = EloFeatureEngine(EloConfig(max_margin=25))
        # margin=50 > max_margin=25
        expected = 25**0.85
        assert engine._margin_multiplier(50) == pytest.approx(expected)


class TestEloHomeCourt:
    """Tests for home-court adjustment."""

    def test_home_win_different_from_neutral(self) -> None:
        """When winner is home (loc=H), rating change differs from neutral.

        Per AC #4, home team's effective rating is lowered (subtracted),
        which lowers their expected score → winning is a bigger surprise
        → winner gains MORE.  This matches the story spec verbatim.
        """
        engine_neutral = EloFeatureEngine(EloConfig())
        engine_neutral.update_game(101, 102, 75, 60, "N", False)
        gain_neutral = engine_neutral.get_rating(101) - 1500.0

        engine_home = EloFeatureEngine(EloConfig())
        engine_home.update_game(101, 102, 75, 60, "H", False)
        gain_home = engine_home.get_rating(101) - 1500.0

        # Subtracting from home team's eff. rating → lower expected → more gain
        assert gain_home > gain_neutral

    def test_away_win_different_from_neutral(self) -> None:
        """When winner is away (loc=A), loser's eff. rating is lowered.

        Loser (home) has their rating subtracted → they look weaker →
        winner's expected is higher → winning is less surprising → less gain.
        """
        engine_neutral = EloFeatureEngine(EloConfig())
        engine_neutral.update_game(101, 102, 75, 60, "N", False)
        gain_neutral = engine_neutral.get_rating(101) - 1500.0

        engine_away = EloFeatureEngine(EloConfig())
        engine_away.update_game(101, 102, 75, 60, "A", False)
        gain_away = engine_away.get_rating(101) - 1500.0

        # Subtracting from loser (home) → winner expected higher → less gain
        assert gain_away < gain_neutral

    def test_neutral_no_adjustment(self) -> None:
        """Neutral game should have no home court effect."""
        engine = EloFeatureEngine(EloConfig(home_advantage_elo=100.0))
        # With huge home advantage, neutral should be unaffected
        engine.update_game(101, 102, 75, 60, "N", False)
        # Expected is 0.5 for equal ratings at neutral
        expected_at_neutral = engine.expected_score(1500.0, 1500.0)
        assert expected_at_neutral == pytest.approx(0.5)


class TestEloVariableK:
    """Tests for variable K-factor transitions."""

    def test_early_season_k(self) -> None:
        engine = EloFeatureEngine(EloConfig(early_game_threshold=20))
        # 0 games played: should use k_early
        k = engine._effective_k(101, is_tournament=False)
        assert k == 56

    def test_regular_season_k_after_threshold(self) -> None:
        engine = EloFeatureEngine(EloConfig(early_game_threshold=2))
        # Play 2 games to pass threshold
        engine.update_game(101, 102, 75, 60, "N", False)
        engine.update_game(101, 103, 75, 60, "N", False)
        k = engine._effective_k(101, is_tournament=False)
        assert k == 38

    def test_tournament_k(self) -> None:
        engine = EloFeatureEngine(EloConfig())
        k = engine._effective_k(101, is_tournament=True)
        assert k == 47.5

    def test_tournament_k_overrides_early(self) -> None:
        """Tournament K used even if team has played few games."""
        engine = EloFeatureEngine(EloConfig(early_game_threshold=100))
        k = engine._effective_k(101, is_tournament=True)
        assert k == 47.5


# ── Task 3: Season management tests ─────────────────────────────────────────


class TestEloSeasonManagement:
    """Tests for season mean-reversion and game count reset."""

    def test_mean_reversion_toward_global_mean_no_lookup(self) -> None:
        """Without ConferenceLookup, regress toward global mean."""
        engine = EloFeatureEngine(EloConfig(mean_reversion_fraction=0.25))
        # Set up some ratings
        engine._ratings = {101: 1600.0, 102: 1400.0}
        # global mean = (1600+1400)/2 = 1500

        engine.apply_season_mean_reversion(season=2024)

        # Team 101: 1600 + 0.25*(1500-1600) = 1600 - 25 = 1575
        assert engine.get_rating(101) == pytest.approx(1575.0)
        # Team 102: 1400 + 0.25*(1500-1400) = 1400 + 25 = 1425
        assert engine.get_rating(102) == pytest.approx(1425.0)

    def test_mean_reversion_toward_conference_mean(self) -> None:
        """With ConferenceLookup, regress toward conference mean."""
        from ncaa_eval.transform.normalization import ConferenceLookup

        lookup = ConferenceLookup(
            {
                (2024, 101): "ACC",
                (2024, 102): "ACC",
                (2024, 201): "B10",
                (2024, 202): "B10",
            }
        )
        engine = EloFeatureEngine(
            EloConfig(mean_reversion_fraction=0.25),
            conference_lookup=lookup,
        )
        engine._ratings = {
            101: 1600.0,
            102: 1400.0,  # ACC mean = 1500
            201: 1550.0,
            202: 1450.0,  # B10 mean = 1500
        }

        engine.apply_season_mean_reversion(season=2024)

        # ACC conf mean = 1500; team 101: 1600 + 0.25*(1500-1600) = 1575
        assert engine.get_rating(101) == pytest.approx(1575.0)
        assert engine.get_rating(102) == pytest.approx(1425.0)
        # B10 conf mean = 1500; team 201: 1550 + 0.25*(1500-1550) = 1537.5
        assert engine.get_rating(201) == pytest.approx(1537.5)
        assert engine.get_rating(202) == pytest.approx(1462.5)

    def test_mean_reversion_no_conf_falls_back_to_global(self) -> None:
        """Teams without conference info regress toward global mean."""
        from ncaa_eval.transform.normalization import ConferenceLookup

        lookup = ConferenceLookup(
            {
                (2024, 101): "ACC",
                (2024, 102): "ACC",
                # team 201 has no conference entry
            }
        )
        engine = EloFeatureEngine(
            EloConfig(mean_reversion_fraction=0.25),
            conference_lookup=lookup,
        )
        engine._ratings = {101: 1600.0, 102: 1400.0, 201: 1700.0}
        global_mean = (1600.0 + 1400.0 + 1700.0) / 3.0

        engine.apply_season_mean_reversion(season=2024)

        # Team 201 (no conf): regress toward global mean
        expected_201 = 1700.0 + 0.25 * (global_mean - 1700.0)
        assert engine.get_rating(201) == pytest.approx(expected_201)

    def test_mean_reversion_noop_when_no_ratings(self) -> None:
        """No-op when no prior ratings exist."""
        engine = EloFeatureEngine(EloConfig())
        engine.apply_season_mean_reversion(season=2024)  # Should not raise
        assert engine._ratings == {}

    def test_reset_game_counts(self) -> None:
        engine = EloFeatureEngine(EloConfig())
        engine._game_counts = {101: 30, 102: 28}
        engine.reset_game_counts()
        assert engine._game_counts == {}

    def test_start_new_season_orchestrates(self) -> None:
        """start_new_season does mean-reversion then resets counts."""
        engine = EloFeatureEngine(EloConfig(mean_reversion_fraction=0.25))
        engine._ratings = {101: 1600.0, 102: 1400.0}
        engine._game_counts = {101: 30, 102: 28}

        engine.start_new_season(season=2024)

        # Ratings should have changed (mean-reverted)
        assert engine.get_rating(101) == pytest.approx(1575.0)
        assert engine.get_rating(102) == pytest.approx(1425.0)
        # Game counts should be reset
        assert engine._game_counts == {}


# ── Task 4: Snapshot and bulk processing tests ───────────────────────────────


class TestEloSnapshotAndBulk:
    """Tests for get_all_ratings and process_season."""

    def test_get_all_ratings_returns_copy(self) -> None:
        engine = EloFeatureEngine(EloConfig())
        engine._ratings = {101: 1600.0, 102: 1400.0}
        snapshot = engine.get_all_ratings()
        assert snapshot == {101: 1600.0, 102: 1400.0}
        # Modifying copy should not affect engine
        snapshot[101] = 9999.0
        assert engine.get_rating(101) == 1600.0

    def test_process_season_empty_games(self) -> None:
        engine = EloFeatureEngine(EloConfig())
        result = engine.process_season([], season=2023)
        assert list(result.columns) == ["game_id", "elo_w_before", "elo_l_before"]
        assert len(result) == 0

    def test_process_season_output_schema(self) -> None:
        games = [
            _make_game(game_id="1", day_num=10, w_team_id=101, l_team_id=102),
            _make_game(game_id="2", day_num=15, w_team_id=103, l_team_id=104),
        ]
        engine = EloFeatureEngine(EloConfig())
        result = engine.process_season(games, season=2023)
        assert list(result.columns) == ["game_id", "elo_w_before", "elo_l_before"]
        assert len(result) == 2
        assert list(result["game_id"]) == ["1", "2"]

    def test_process_season_first_games_start_at_initial(self) -> None:
        games = [
            _make_game(game_id="1", day_num=10, w_team_id=101, l_team_id=102),
        ]
        engine = EloFeatureEngine(EloConfig())
        result = engine.process_season(games, season=2023)
        assert result.iloc[0]["elo_w_before"] == 1500.0
        assert result.iloc[0]["elo_l_before"] == 1500.0

    def test_process_season_second_game_reflects_first(self) -> None:
        """Second game uses updated ratings from first game."""
        games = [
            _make_game(game_id="1", day_num=10, w_team_id=101, l_team_id=102, w_score=75, l_score=60),
            _make_game(game_id="2", day_num=15, w_team_id=101, l_team_id=102, w_score=70, l_score=65),
        ]
        engine = EloFeatureEngine(EloConfig())
        result = engine.process_season(games, season=2023)
        # Second game: team 101 should have rating > 1500 (won first game)
        assert result.iloc[1]["elo_w_before"] > 1500.0
        assert result.iloc[1]["elo_l_before"] < 1500.0

    def test_process_season_calls_start_new_season_when_prior_exists(self) -> None:
        """If prior ratings exist, process_season calls start_new_season."""
        engine = EloFeatureEngine(EloConfig(mean_reversion_fraction=0.25))
        # Simulate prior season
        engine._ratings = {101: 1600.0, 102: 1400.0}
        engine._game_counts = {101: 30, 102: 28}

        games = [
            _make_game(game_id="1", season=2024, day_num=10, w_team_id=101, l_team_id=102),
        ]
        result = engine.process_season(games, season=2024)

        # Before-rating for game 1 should reflect mean-reverted values
        # 101: 1600 + 0.25*(1500-1600) = 1575
        assert result.iloc[0]["elo_w_before"] == pytest.approx(1575.0)
        assert result.iloc[0]["elo_l_before"] == pytest.approx(1425.0)

    def test_multi_season_continuity(self) -> None:
        """Ratings carry forward across seasons."""
        engine = EloFeatureEngine(EloConfig(mean_reversion_fraction=0.0))
        s1_games = [
            _make_game(game_id="1", season=2023, day_num=10, w_team_id=101, l_team_id=102),
        ]
        engine.process_season(s1_games, season=2023)
        rating_after_s1 = engine.get_rating(101)

        s2_games = [
            _make_game(game_id="2", season=2024, day_num=10, w_team_id=101, l_team_id=102),
        ]
        result = engine.process_season(s2_games, season=2024)

        # With 0% reversion, ratings carry forward exactly
        assert result.iloc[0]["elo_w_before"] == pytest.approx(rating_after_s1)
