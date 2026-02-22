"""Unit tests for ncaa_eval.transform.feature_serving."""

from __future__ import annotations

import datetime
from typing import Literal
from unittest.mock import MagicMock

import pandas as pd  # type: ignore[import-untyped]
import pytest

from ncaa_eval.ingest.schema import Game
from ncaa_eval.transform.feature_serving import (
    FeatureBlock,
    FeatureConfig,
    StatefulFeatureServer,
)
from ncaa_eval.transform.serving import ChronologicalDataServer, SeasonGames

# ── FeatureBlock enum tests ──────────────────────────────────────────────────


class TestFeatureBlock:
    """Tests for the FeatureBlock enum."""

    def test_members(self) -> None:
        assert FeatureBlock.SEQUENTIAL.value == "sequential"
        assert FeatureBlock.GRAPH.value == "graph"
        assert FeatureBlock.BATCH_RATING.value == "batch_rating"
        assert FeatureBlock.ORDINAL.value == "ordinal"
        assert FeatureBlock.SEED.value == "seed"
        assert FeatureBlock.ELO.value == "elo"

    def test_member_count(self) -> None:
        assert len(FeatureBlock) == 6


# ── FeatureConfig tests ─────────────────────────────────────────────────────


class TestFeatureConfig:
    """Tests for the FeatureConfig dataclass."""

    def test_defaults(self) -> None:
        cfg = FeatureConfig()
        assert cfg.sequential_windows == (5, 10, 20)
        assert cfg.ewma_alphas == (0.15, 0.20)
        assert cfg.graph_features_enabled is True
        assert cfg.batch_rating_types == ("srs", "ridge", "colley")
        assert cfg.ordinal_systems is None  # use coverage-gate default
        assert cfg.ordinal_composite == "simple_average"
        assert cfg.matchup_deltas is True
        assert cfg.gender_scope == "M"
        assert cfg.dataset_scope == "kaggle"
        assert cfg.calibration_method == "isotonic"

    def test_frozen(self) -> None:
        cfg = FeatureConfig()
        with pytest.raises(AttributeError):
            cfg.gender_scope = "W"  # type: ignore[misc]

    def test_custom_values(self) -> None:
        cfg = FeatureConfig(
            sequential_windows=(3, 7),
            ewma_alphas=(0.10,),
            graph_features_enabled=False,
            batch_rating_types=("srs",),
            ordinal_systems=("POM", "SAG"),
            ordinal_composite="weighted",
            matchup_deltas=True,
            gender_scope="W",
            dataset_scope="all",
            calibration_method="sigmoid",
        )
        assert cfg.sequential_windows == (3, 7)
        assert cfg.ewma_alphas == (0.10,)
        assert cfg.graph_features_enabled is False
        assert cfg.batch_rating_types == ("srs",)
        assert cfg.ordinal_systems == ("POM", "SAG")
        assert cfg.ordinal_composite == "weighted"
        assert cfg.gender_scope == "W"
        assert cfg.dataset_scope == "all"
        assert cfg.calibration_method == "sigmoid"

    def test_calibration_method_none(self) -> None:
        cfg = FeatureConfig(calibration_method=None)
        assert cfg.calibration_method is None

    def test_active_blocks_all(self) -> None:
        cfg = FeatureConfig()
        blocks = cfg.active_blocks()
        assert FeatureBlock.SEQUENTIAL in blocks
        assert FeatureBlock.GRAPH in blocks
        assert FeatureBlock.BATCH_RATING in blocks
        assert FeatureBlock.ORDINAL in blocks
        assert FeatureBlock.SEED in blocks
        # ELO excluded by default (elo_enabled=False)
        assert FeatureBlock.ELO not in blocks

    def test_active_blocks_graph_disabled(self) -> None:
        cfg = FeatureConfig(graph_features_enabled=False)
        blocks = cfg.active_blocks()
        assert FeatureBlock.GRAPH not in blocks
        assert FeatureBlock.SEQUENTIAL in blocks

    def test_active_blocks_no_batch_ratings(self) -> None:
        cfg = FeatureConfig(batch_rating_types=())
        blocks = cfg.active_blocks()
        assert FeatureBlock.BATCH_RATING not in blocks

    def test_active_blocks_no_ordinals(self) -> None:
        cfg = FeatureConfig(ordinal_composite=None)
        blocks = cfg.active_blocks()
        assert FeatureBlock.ORDINAL not in blocks


# ── Helpers / Fixtures ──────────────────────────────────────────────────────


def _make_game(  # noqa: PLR0913
    game_id: str = "1",
    season: int = 2023,
    day_num: int = 10,
    w_team_id: int = 101,
    l_team_id: int = 102,
    w_score: int = 75,
    l_score: int = 60,
    loc: Literal["H", "A", "N"] = "H",
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


def _make_games(n: int = 5, season: int = 2023) -> list[Game]:
    """Create n test games with unique teams."""
    games: list[Game] = []
    for i in range(n):
        games.append(
            _make_game(
                game_id=str(i + 1),
                season=season,
                day_num=10 + i * 5,
                w_team_id=101 + i * 2,
                l_team_id=102 + i * 2,
                w_score=70 + i,
                l_score=60 + i,
            )
        )
    return games


def _minimal_config() -> FeatureConfig:
    """Config with all optional blocks disabled for fast testing."""
    return FeatureConfig(
        sequential_windows=(),
        ewma_alphas=(),
        graph_features_enabled=False,
        batch_rating_types=(),
        ordinal_composite=None,
        matchup_deltas=False,
        calibration_method=None,
    )


def _mock_data_server(games: list[Game], year: int = 2023) -> MagicMock:
    """Create a mock ChronologicalDataServer returning the given games."""
    server = MagicMock(spec=ChronologicalDataServer)
    server.get_chronological_season.return_value = SeasonGames(
        year=year,
        games=games,
        has_tournament=True,
    )
    return server


# ── StatefulFeatureServer tests ─────────────────────────────────────────────


class TestStatefulFeatureServerConstruction:
    """Tests for the StatefulFeatureServer constructor."""

    def test_accepts_config_and_data_server(self) -> None:
        server = StatefulFeatureServer(
            config=FeatureConfig(),
            data_server=_mock_data_server([]),
        )
        assert server.config == FeatureConfig()

    def test_accepts_optional_lookup_objects(self) -> None:
        seed_table = MagicMock()
        ordinals_store = MagicMock()
        server = StatefulFeatureServer(
            config=FeatureConfig(),
            data_server=_mock_data_server([]),
            seed_table=seed_table,
            ordinals_store=ordinals_store,
        )
        assert server._seed_table is seed_table
        assert server._ordinals_store is ordinals_store


class TestStatefulFeatureServerBatchMode:
    """Tests for batch mode of serve_season_features."""

    def test_returns_dataframe(self) -> None:
        games = _make_games(3)
        ds = _mock_data_server(games)
        server = StatefulFeatureServer(
            config=_minimal_config(),
            data_server=ds,
        )
        result = server.serve_season_features(2023, mode="batch")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_has_game_metadata_columns(self) -> None:
        games = _make_games(3)
        ds = _mock_data_server(games)
        server = StatefulFeatureServer(
            config=_minimal_config(),
            data_server=ds,
        )
        result = server.serve_season_features(2023, mode="batch")
        for col in (
            "game_id",
            "season",
            "day_num",
            "date",
            "team_a_id",
            "team_b_id",
            "is_tournament",
            "loc_encoding",
            "team_a_won",
        ):
            assert col in result.columns, f"Missing column: {col}"

    def test_empty_season_returns_empty_df(self) -> None:
        ds = _mock_data_server([])
        server = StatefulFeatureServer(
            config=_minimal_config(),
            data_server=ds,
        )
        result = server.serve_season_features(2023, mode="batch")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_loc_encoding_values(self) -> None:
        """H→+1, A→-1, N→0 from team_a perspective."""
        games = [
            _make_game(game_id="1", loc="H", day_num=10),
            _make_game(game_id="2", loc="A", day_num=15),
            _make_game(game_id="3", loc="N", day_num=20),
        ]
        ds = _mock_data_server(games)
        server = StatefulFeatureServer(
            config=_minimal_config(),
            data_server=ds,
        )
        result = server.serve_season_features(2023, mode="batch")
        locs = result.set_index("game_id")["loc_encoding"]
        assert locs["1"] == 1  # H → +1 (winner is home)
        assert locs["2"] == -1  # A → -1 (winner is away)
        assert locs["3"] == 0  # N → 0

    def test_team_a_is_winner(self) -> None:
        """team_a_id = w_team_id, team_b_id = l_team_id by convention."""
        game = _make_game(w_team_id=101, l_team_id=202)
        ds = _mock_data_server([game])
        server = StatefulFeatureServer(
            config=_minimal_config(),
            data_server=ds,
        )
        result = server.serve_season_features(2023, mode="batch")
        assert result.iloc[0]["team_a_id"] == 101
        assert result.iloc[0]["team_b_id"] == 202
        assert bool(result.iloc[0]["team_a_won"]) is True


class TestStatefulFeatureServerStatefulMode:
    """Tests for stateful mode of serve_season_features."""

    def test_returns_dataframe(self) -> None:
        games = _make_games(3)
        ds = _mock_data_server(games)
        server = StatefulFeatureServer(
            config=_minimal_config(),
            data_server=ds,
        )
        result = server.serve_season_features(2023, mode="stateful")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_same_columns_as_batch(self) -> None:
        games = _make_games(3)
        ds = _mock_data_server(games)
        cfg = _minimal_config()
        server = StatefulFeatureServer(config=cfg, data_server=ds)
        batch = server.serve_season_features(2023, mode="batch")
        stateful = server.serve_season_features(2023, mode="stateful")
        assert set(batch.columns) == set(stateful.columns)

    def test_invalid_mode_raises(self) -> None:
        ds = _mock_data_server([])
        server = StatefulFeatureServer(
            config=_minimal_config(),
            data_server=ds,
        )
        with pytest.raises(ValueError, match="mode"):
            server.serve_season_features(2023, mode="invalid")


# ── Task 3: Ordinal temporal slicing tests ──────────────────────────────────


def _mock_ordinals_store() -> MagicMock:
    """Create a mock MasseyOrdinalsStore with temporal filtering behavior."""
    from ncaa_eval.transform.normalization import MasseyOrdinalsStore

    store = MagicMock(spec=MasseyOrdinalsStore)

    def _composite_simple_average(season: int, day_num: int, systems: list[str]) -> pd.Series:
        """Return ordinal composite scores that vary by day_num."""
        # Early-season: team 101=10, 102=20, etc.
        # Later: team 101=5, 102=15, etc. (lower is better)
        offset = 0 if day_num < 50 else -5
        data: dict[int, float] = {}
        for i in range(10):
            tid = 101 + i
            data[tid] = float(10 + i * 10 + offset)
        return pd.Series(data, name="ordinal_composite")

    store.composite_simple_average.side_effect = _composite_simple_average

    gate_result = MagicMock()
    gate_result.recommended_systems = ("SAG", "POM", "MOR", "WLK")
    store.run_coverage_gate.return_value = gate_result

    return store


class TestOrdinalTemporalSlicing:
    """Tests for Massey ordinal temporal slicing (Task 3 / AC #5)."""

    def test_ordinal_features_use_game_day_num(self) -> None:
        """Each game gets ordinals sliced at its own day_num, not global."""
        game_early = _make_game(game_id="1", day_num=30, w_team_id=101, l_team_id=102)
        game_late = _make_game(game_id="2", day_num=60, w_team_id=101, l_team_id=102)
        ds = _mock_data_server([game_early, game_late])
        ordinals = _mock_ordinals_store()

        cfg = FeatureConfig(
            sequential_windows=(),
            ewma_alphas=(),
            graph_features_enabled=False,
            batch_rating_types=(),
            ordinal_composite="simple_average",
            matchup_deltas=False,
            calibration_method=None,
        )
        server = StatefulFeatureServer(
            config=cfg,
            data_server=ds,
            ordinals_store=ordinals,
        )
        server.serve_season_features(2023, mode="batch")

        # Verify composite_simple_average was called with each game's day_num
        calls = ordinals.composite_simple_average.call_args_list
        day_nums_called = [c[0][1] for c in calls]
        assert 30 in day_nums_called
        assert 60 in day_nums_called

    def test_ordinal_features_present_when_enabled(self) -> None:
        games = [_make_game(day_num=50, w_team_id=101, l_team_id=102)]
        ds = _mock_data_server(games)
        ordinals = _mock_ordinals_store()

        cfg = FeatureConfig(
            sequential_windows=(),
            ewma_alphas=(),
            graph_features_enabled=False,
            batch_rating_types=(),
            ordinal_composite="simple_average",
            matchup_deltas=False,
            calibration_method=None,
        )
        server = StatefulFeatureServer(
            config=cfg,
            data_server=ds,
            ordinals_store=ordinals,
        )
        result = server.serve_season_features(2023, mode="batch")
        assert "ordinal_composite_a" in result.columns
        assert "ordinal_composite_b" in result.columns

    def test_ordinal_disabled_no_columns(self) -> None:
        games = [_make_game()]
        ds = _mock_data_server(games)
        cfg = _minimal_config()  # ordinal_composite=None
        server = StatefulFeatureServer(config=cfg, data_server=ds)
        result = server.serve_season_features(2023, mode="batch")
        assert "ordinal_composite_a" not in result.columns


# ── Task 4: Matchup-level feature tests ─────────────────────────────────────


class TestMatchupDeltas:
    """Tests for matchup-level delta computation (Task 4 / AC #6)."""

    def test_seed_diff_tournament_game(self) -> None:
        """seed_diff = seed_num_A − seed_num_B for tournament games."""
        from ncaa_eval.transform.normalization import TourneySeed, TourneySeedTable

        game = _make_game(
            w_team_id=101,
            l_team_id=102,
            is_tournament=True,
            day_num=135,
        )
        ds = _mock_data_server([game])

        # team 101 = 1 seed, team 102 = 8 seed
        seeds = TourneySeedTable(
            {
                (2023, 101): TourneySeed(2023, 101, "W01", "W", 1, False),
                (2023, 102): TourneySeed(2023, 102, "W08", "W", 8, False),
            }
        )

        cfg = FeatureConfig(
            sequential_windows=(),
            ewma_alphas=(),
            graph_features_enabled=False,
            batch_rating_types=(),
            ordinal_composite=None,
            matchup_deltas=True,
            calibration_method=None,
        )
        server = StatefulFeatureServer(
            config=cfg,
            data_server=ds,
            seed_table=seeds,
        )
        result = server.serve_season_features(2023, mode="batch")
        assert "seed_diff" in result.columns
        assert result.iloc[0]["seed_diff"] == 1 - 8  # = -7

    def test_seed_diff_nan_for_regular_season(self) -> None:
        """seed_diff is NaN for non-tournament games."""
        game = _make_game(is_tournament=False)
        ds = _mock_data_server([game])
        cfg = FeatureConfig(
            sequential_windows=(),
            ewma_alphas=(),
            graph_features_enabled=False,
            batch_rating_types=(),
            ordinal_composite=None,
            matchup_deltas=True,
            calibration_method=None,
        )
        server = StatefulFeatureServer(config=cfg, data_server=ds)
        result = server.serve_season_features(2023, mode="batch")
        assert pd.isna(result.iloc[0]["seed_diff"])

    def test_elo_delta_always_nan(self) -> None:
        """delta_elo is NaN until Story 4.8."""
        game = _make_game()
        ds = _mock_data_server([game])
        cfg = _minimal_config()
        server = StatefulFeatureServer(config=cfg, data_server=ds)
        result = server.serve_season_features(2023, mode="batch")
        assert "delta_elo" in result.columns
        assert pd.isna(result.iloc[0]["delta_elo"])

    def test_ordinal_delta_when_matchup_enabled(self) -> None:
        """Ordinal deltas computed when both ordinals and matchup_deltas enabled."""
        game = _make_game(day_num=50, w_team_id=101, l_team_id=102)
        ds = _mock_data_server([game])
        ordinals = _mock_ordinals_store()

        cfg = FeatureConfig(
            sequential_windows=(),
            ewma_alphas=(),
            graph_features_enabled=False,
            batch_rating_types=(),
            ordinal_composite="simple_average",
            matchup_deltas=True,
            calibration_method=None,
        )
        server = StatefulFeatureServer(
            config=cfg,
            data_server=ds,
            ordinals_store=ordinals,
        )
        result = server.serve_season_features(2023, mode="batch")
        assert "delta_ordinal_composite" in result.columns
        # ordinal for 101 = 5 (after offset), 102 = 15
        # delta = 5 - 15 = -10
        assert result.iloc[0]["delta_ordinal_composite"] == pytest.approx(-10.0)

    def test_batch_rating_deltas(self) -> None:
        """Batch rating deltas are computed when both are enabled."""
        # Create games involving teams 101, 102 (enough for SRS/Ridge/Colley)
        games = [
            _make_game(game_id="1", day_num=10, w_team_id=101, l_team_id=102),
            _make_game(game_id="2", day_num=15, w_team_id=102, l_team_id=103),
            _make_game(game_id="3", day_num=20, w_team_id=101, l_team_id=103),
        ]
        ds = _mock_data_server(games)

        cfg = FeatureConfig(
            sequential_windows=(),
            ewma_alphas=(),
            graph_features_enabled=False,
            batch_rating_types=("srs", "ridge", "colley"),
            ordinal_composite=None,
            matchup_deltas=True,
            calibration_method=None,
        )
        server = StatefulFeatureServer(config=cfg, data_server=ds)
        result = server.serve_season_features(2023, mode="batch")

        for col in ("delta_srs", "delta_ridge", "delta_colley"):
            assert col in result.columns, f"Missing column: {col}"


# ── Task 6: Scope filtering tests ───────────────────────────────────────────


class TestScopeFiltering:
    """Tests for gender_scope and dataset_scope parameters (Task 6 / AC #8)."""

    def test_gender_scope_stored(self) -> None:
        cfg = FeatureConfig(gender_scope="M")
        assert cfg.gender_scope == "M"
        cfg_w = FeatureConfig(gender_scope="W")
        assert cfg_w.gender_scope == "W"

    def test_dataset_scope_stored(self) -> None:
        cfg = FeatureConfig(dataset_scope="kaggle")
        assert cfg.dataset_scope == "kaggle"
        cfg_all = FeatureConfig(dataset_scope="all")
        assert cfg_all.dataset_scope == "all"

    def test_scope_passed_to_server(self) -> None:
        """Scope parameters are accessible on the server's config."""
        ds = _mock_data_server([])
        server = StatefulFeatureServer(
            config=FeatureConfig(gender_scope="W", dataset_scope="all"),
            data_server=ds,
        )
        assert server.config.gender_scope == "W"
        assert server.config.dataset_scope == "all"


# ── Task 5 (Story 4.8): Elo feature serving integration tests ───────────────


class TestEloFeatureConfig:
    """Tests for elo_enabled and elo_config fields on FeatureConfig."""

    def test_elo_disabled_by_default(self) -> None:
        cfg = FeatureConfig()
        assert cfg.elo_enabled is False
        assert cfg.elo_config is None

    def test_elo_enabled_in_active_blocks(self) -> None:
        cfg = FeatureConfig(elo_enabled=True)
        blocks = cfg.active_blocks()
        assert FeatureBlock.ELO in blocks

    def test_elo_disabled_not_in_active_blocks(self) -> None:
        cfg = FeatureConfig(elo_enabled=False)
        blocks = cfg.active_blocks()
        assert FeatureBlock.ELO not in blocks

    def test_elo_config_stored(self) -> None:
        from ncaa_eval.transform.elo import EloConfig

        ecfg = EloConfig(initial_rating=1400)
        cfg = FeatureConfig(elo_enabled=True, elo_config=ecfg)
        assert cfg.elo_config is not None
        assert cfg.elo_config.initial_rating == 1400


class TestEloFeatureServing:
    """Tests for StatefulFeatureServer with Elo enabled."""

    def _elo_config(self) -> FeatureConfig:
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

    def test_batch_mode_has_elo_columns(self) -> None:
        from ncaa_eval.transform.elo import EloConfig, EloFeatureEngine

        games = _make_games(3)
        ds = _mock_data_server(games)
        engine = EloFeatureEngine(EloConfig())
        server = StatefulFeatureServer(
            config=self._elo_config(),
            data_server=ds,
            elo_engine=engine,
        )
        result = server.serve_season_features(2023, mode="batch")
        assert "elo_a" in result.columns
        assert "elo_b" in result.columns
        assert "delta_elo" in result.columns

    def test_batch_mode_delta_elo_is_numeric(self) -> None:
        from ncaa_eval.transform.elo import EloConfig, EloFeatureEngine

        games = _make_games(3)
        ds = _mock_data_server(games)
        engine = EloFeatureEngine(EloConfig())
        server = StatefulFeatureServer(
            config=self._elo_config(),
            data_server=ds,
            elo_engine=engine,
        )
        result = server.serve_season_features(2023, mode="batch")
        # delta_elo should be non-NaN (actual Elo values, not placeholder)
        assert not pd.isna(result.iloc[0]["delta_elo"])

    def test_stateful_mode_has_elo_columns(self) -> None:
        from ncaa_eval.transform.elo import EloConfig, EloFeatureEngine

        games = _make_games(3)
        ds = _mock_data_server(games)
        engine = EloFeatureEngine(EloConfig())
        server = StatefulFeatureServer(
            config=self._elo_config(),
            data_server=ds,
            elo_engine=engine,
        )
        result = server.serve_season_features(2023, mode="stateful")
        assert "elo_a" in result.columns
        assert "elo_b" in result.columns
        assert "delta_elo" in result.columns
        assert not pd.isna(result.iloc[0]["delta_elo"])

    def test_elo_disabled_delta_elo_is_nan(self) -> None:
        """When elo_enabled=False, delta_elo should still be NaN."""
        games = _make_games(3)
        ds = _mock_data_server(games)
        cfg = _minimal_config()
        server = StatefulFeatureServer(config=cfg, data_server=ds)
        result = server.serve_season_features(2023, mode="batch")
        assert "delta_elo" in result.columns
        assert pd.isna(result.iloc[0]["delta_elo"])

    def test_elo_disabled_stateful_delta_elo_is_nan(self) -> None:
        games = _make_games(3)
        ds = _mock_data_server(games)
        cfg = _minimal_config()
        server = StatefulFeatureServer(config=cfg, data_server=ds)
        result = server.serve_season_features(2023, mode="stateful")
        assert "delta_elo" in result.columns
        assert pd.isna(result.iloc[0]["delta_elo"])

    def test_first_game_elo_is_initial(self) -> None:
        """First game: both teams start at initial_rating → delta_elo = 0."""
        from ncaa_eval.transform.elo import EloConfig, EloFeatureEngine

        game = _make_game(game_id="1", w_team_id=101, l_team_id=102)
        ds = _mock_data_server([game])
        engine = EloFeatureEngine(EloConfig())
        server = StatefulFeatureServer(
            config=self._elo_config(),
            data_server=ds,
            elo_engine=engine,
        )
        result = server.serve_season_features(2023, mode="batch")
        # Both teams start at 1500 → delta = 0
        assert result.iloc[0]["delta_elo"] == pytest.approx(0.0)
