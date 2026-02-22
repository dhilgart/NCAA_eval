"""Declarative feature serving layer for NCAA basketball prediction.

Combines sequential, graph, batch-rating, ordinal, seed, and Elo feature
building blocks into a temporally-safe, matchup-level feature matrix.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ncaa_eval.transform.serving import ChronologicalDataServer

if TYPE_CHECKING:
    from ncaa_eval.ingest.schema import Game
    from ncaa_eval.transform.normalization import (
        MasseyOrdinalsStore,
        TourneySeedTable,
    )

logger = logging.getLogger(__name__)

# Location encoding: H→+1, A→-1, N→0 (from team_a / winner perspective)
_LOC_ENCODING: dict[str, int] = {"H": 1, "A": -1, "N": 0}


# ── Feature Block Enum ───────────────────────────────────────────────────────


class FeatureBlock(enum.Enum):
    """Individual feature building blocks that can be activated."""

    SEQUENTIAL = "sequential"
    GRAPH = "graph"
    BATCH_RATING = "batch_rating"
    ORDINAL = "ordinal"
    SEED = "seed"
    ELO = "elo"


# ── Feature Configuration ───────────────────────────────────────────────────


@dataclass(frozen=True)
class FeatureConfig:
    """Declarative specification of which feature blocks and parameters to use.

    Parameters
    ----------
    sequential_windows
        Rolling window sizes for sequential features (e.g., ``(5, 10, 20)``).
    ewma_alphas
        EWMA smoothing factors for sequential features (e.g., ``(0.15, 0.20)``).
    graph_features_enabled
        Whether to compute graph centrality features (PageRank, etc.).
    batch_rating_types
        Which batch rating systems to include (``"srs"``, ``"ridge"``, ``"colley"``).
    ordinal_systems
        Massey ordinal systems to use; ``None`` means use coverage-gate defaults.
    ordinal_composite
        Composite method: ``"simple_average"``, ``"weighted"``, ``"pca"``, or ``None`` to disable.
    matchup_deltas
        Whether to compute team_A − team_B deltas for matchup features.
    gender_scope
        ``"M"`` for men's, ``"W"`` for women's.
    dataset_scope
        ``"kaggle"`` for Kaggle-only games, ``"all"`` for Kaggle + ESPN enrichment.
    calibration_method
        ``"isotonic"``, ``"sigmoid"``, or ``None`` to skip calibration.
    """

    sequential_windows: tuple[int, ...] = (5, 10, 20)
    ewma_alphas: tuple[float, ...] = (0.15, 0.20)
    graph_features_enabled: bool = True
    batch_rating_types: tuple[str, ...] = ("srs", "ridge", "colley")
    ordinal_systems: tuple[str, ...] | None = None
    ordinal_composite: str | None = "simple_average"
    matchup_deltas: bool = True
    gender_scope: str = field(default="M")
    dataset_scope: str = field(default="kaggle")
    calibration_method: str | None = "isotonic"

    def active_blocks(self) -> frozenset[FeatureBlock]:
        """Return the set of feature blocks that are currently enabled.

        ELO is always excluded (placeholder until Story 4.8).
        """
        blocks: set[FeatureBlock] = set()

        if self.sequential_windows:
            blocks.add(FeatureBlock.SEQUENTIAL)
        if self.graph_features_enabled:
            blocks.add(FeatureBlock.GRAPH)
        if self.batch_rating_types:
            blocks.add(FeatureBlock.BATCH_RATING)
        if self.ordinal_composite is not None:
            blocks.add(FeatureBlock.ORDINAL)
        # Seed is always active (NaN for non-tournament games)
        blocks.add(FeatureBlock.SEED)

        return frozenset(blocks)


# ── Metadata columns present in every output row ────────────────────────────

_META_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "day_num",
    "date",
    "team_a_id",
    "team_b_id",
    "is_tournament",
    "loc_encoding",
    "team_a_won",
)


# ── StatefulFeatureServer ────────────────────────────────────────────────────


class StatefulFeatureServer:
    """Combines feature building blocks into a single feature matrix.

    Supports two consumption modes:

    * **batch** — compute all features for an entire season at once
      (suitable for stateless models like XGBoost).
    * **stateful** — iterate game-by-game, accumulating state incrementally
      (suitable for Elo-style models; placeholder until Story 4.8).

    Parameters
    ----------
    config
        Declarative specification of which feature blocks to activate.
    data_server
        Chronological data serving layer wrapping the Repository.
    seed_table
        Tournament seed lookup table (optional; needed for seed features).
    ordinals_store
        Massey ordinals store (optional; needed for ordinal features).
    """

    def __init__(
        self,
        config: FeatureConfig,
        data_server: ChronologicalDataServer,
        *,
        seed_table: TourneySeedTable | None = None,
        ordinals_store: MasseyOrdinalsStore | None = None,
    ) -> None:
        self.config = config
        self._data_server = data_server
        self._seed_table = seed_table
        self._ordinals_store = ordinals_store

    # ── Public API ───────────────────────────────────────────────────────

    def serve_season_features(
        self,
        year: int,
        mode: str = "batch",
    ) -> pd.DataFrame:
        """Build the feature matrix for a full season.

        Parameters
        ----------
        year
            Season year (e.g. 2023 for the 2022-23 season).
        mode
            ``"batch"`` or ``"stateful"``.

        Returns
        -------
        pd.DataFrame
            One row per game with metadata, feature deltas, and the target label.
        """
        if mode not in ("batch", "stateful"):
            msg = f"mode must be 'batch' or 'stateful', got {mode!r}"
            raise ValueError(msg)

        season_data = self._data_server.get_chronological_season(year)
        games = season_data.games

        if not games:
            return self._empty_frame()

        if mode == "batch":
            return self._serve_batch(year, games)
        return self._serve_stateful(year, games)

    # ── Internal: batch mode ─────────────────────────────────────────────

    def _serve_batch(self, year: int, games: list[Game]) -> pd.DataFrame:
        """Compute features for all games at once (batch mode)."""
        rows = self._build_game_metadata(games)
        df = pd.DataFrame(rows)

        # Placeholder: feature columns added by Tasks 3-6
        # Add Elo placeholder column
        df["delta_elo"] = np.nan

        return df

    # ── Internal: stateful mode ──────────────────────────────────────────

    def _serve_stateful(self, year: int, games: list[Game]) -> pd.DataFrame:
        """Compute features game-by-game, accumulating state (stateful mode)."""
        result_rows: list[dict[str, object]] = []
        for game in games:
            row = self._game_to_metadata_dict(game)
            # Placeholder: stateful feature accumulation (Tasks 3-6)
            row["delta_elo"] = np.nan
            result_rows.append(row)

        return pd.DataFrame(result_rows)

    # ── Internal: metadata extraction ────────────────────────────────────

    def _build_game_metadata(self, games: list[Game]) -> list[dict[str, object]]:
        """Extract metadata columns from a list of games."""
        return [self._game_to_metadata_dict(g) for g in games]

    @staticmethod
    def _game_to_metadata_dict(game: Game) -> dict[str, object]:
        """Convert a Game to metadata dict (team_a=winner convention)."""
        return {
            "game_id": game.game_id,
            "season": game.season,
            "day_num": game.day_num,
            "date": game.date,
            "team_a_id": game.w_team_id,
            "team_b_id": game.l_team_id,
            "is_tournament": game.is_tournament,
            "loc_encoding": _LOC_ENCODING.get(game.loc, 0),
            "team_a_won": True,
        }

    def _empty_frame(self) -> pd.DataFrame:
        """Return an empty DataFrame with the correct column set."""
        cols = list(_META_COLUMNS) + ["delta_elo"]
        return pd.DataFrame(columns=cols)
