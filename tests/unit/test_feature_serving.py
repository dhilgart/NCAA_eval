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
        # ELO is always excluded (placeholder for 4.8)
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
