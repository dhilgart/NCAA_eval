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

# Batch rating type → module function name
_BATCH_RATING_FUNCS: dict[str, str] = {
    "srs": "compute_srs_ratings",
    "ridge": "compute_ridge_ratings",
    "colley": "compute_colley_ratings",
}

# Batch rating type → column name in the returned DataFrame
_BATCH_RATING_COLS: dict[str, str] = {
    "srs": "srs_rating",
    "ridge": "ridge_rating",
    "colley": "colley_rating",
}


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

        active = self.config.active_blocks()

        # Batch ratings (season-level, from regular-season games only)
        batch_ratings: dict[str, pd.DataFrame] = {}
        if FeatureBlock.BATCH_RATING in active:
            batch_ratings = self._compute_batch_ratings(games)

        # Per-game features
        for idx, game in enumerate(games):
            # Ordinal features
            if FeatureBlock.ORDINAL in active:
                ord_a, ord_b = self._get_ordinal_values(game)
                df.at[idx, "ordinal_composite_a"] = ord_a
                df.at[idx, "ordinal_composite_b"] = ord_b

            # Seed features
            if FeatureBlock.SEED in active:
                seed_a, seed_b = self._get_seed_nums(game)
                df.at[idx, "seed_num_a"] = seed_a
                df.at[idx, "seed_num_b"] = seed_b

            # Batch rating per-team values
            if FeatureBlock.BATCH_RATING in active:
                for rating_type in self.config.batch_rating_types:
                    col = _BATCH_RATING_COLS.get(rating_type, f"{rating_type}_rating")
                    rating_df = batch_ratings.get(rating_type)
                    val_a, val_b = self._lookup_rating(rating_df, game.w_team_id, game.l_team_id, col)
                    df.at[idx, f"{rating_type}_a"] = val_a
                    df.at[idx, f"{rating_type}_b"] = val_b

        # Compute matchup deltas
        if self.config.matchup_deltas:
            df = self._compute_matchup_deltas(df, active)

        # Elo placeholder (always present)
        df["delta_elo"] = np.nan

        return df

    # ── Internal: stateful mode ──────────────────────────────────────────

    def _serve_stateful(self, year: int, games: list[Game]) -> pd.DataFrame:
        """Compute features game-by-game, accumulating state (stateful mode)."""
        active = self.config.active_blocks()
        result_rows: list[dict[str, object]] = []

        # Batch ratings are pre-computed once for stateful mode too
        batch_ratings: dict[str, pd.DataFrame] = {}
        if FeatureBlock.BATCH_RATING in active:
            batch_ratings = self._compute_batch_ratings(games)

        for game in games:
            row = self._game_to_metadata_dict(game)

            # Ordinal features
            if FeatureBlock.ORDINAL in active:
                ord_a, ord_b = self._get_ordinal_values(game)
                row["ordinal_composite_a"] = ord_a
                row["ordinal_composite_b"] = ord_b

            # Seed features
            if FeatureBlock.SEED in active:
                seed_a, seed_b = self._get_seed_nums(game)
                row["seed_num_a"] = seed_a
                row["seed_num_b"] = seed_b

            # Batch rating per-team values
            if FeatureBlock.BATCH_RATING in active:
                for rating_type in self.config.batch_rating_types:
                    col = _BATCH_RATING_COLS.get(rating_type, f"{rating_type}_rating")
                    rating_df = batch_ratings.get(rating_type)
                    val_a, val_b = self._lookup_rating(rating_df, game.w_team_id, game.l_team_id, col)
                    row[f"{rating_type}_a"] = val_a
                    row[f"{rating_type}_b"] = val_b

            row["delta_elo"] = np.nan
            result_rows.append(row)

        df = pd.DataFrame(result_rows)

        # Compute matchup deltas
        if self.config.matchup_deltas and not df.empty:
            df = self._compute_matchup_deltas(df, active)

        return df

    # ── Internal: ordinal features (Task 3) ──────────────────────────────

    def _get_ordinal_values(self, game: Game) -> tuple[float, float]:
        """Get ordinal composite values for both teams at game's day_num.

        Returns (ordinal_a, ordinal_b) where a=w_team_id, b=l_team_id.
        """
        if self._ordinals_store is None:
            return (np.nan, np.nan)

        systems = self._resolve_ordinal_systems()
        composite = self._ordinals_store.composite_simple_average(game.season, game.day_num, systems)

        val_a = composite.get(game.w_team_id, np.nan)
        val_b = composite.get(game.l_team_id, np.nan)
        return (float(val_a), float(val_b))

    def _resolve_ordinal_systems(self) -> list[str]:
        """Determine which ordinal systems to use."""
        if self.config.ordinal_systems is not None:
            return list(self.config.ordinal_systems)
        # Use coverage-gate recommended systems
        if self._ordinals_store is not None:
            gate = self._ordinals_store.run_coverage_gate()
            return list(gate.recommended_systems)
        return []

    # ── Internal: seed features (Task 4) ─────────────────────────────────

    def _get_seed_nums(self, game: Game) -> tuple[float, float]:
        """Get seed numbers for both teams. NaN if not in tournament or unseeded."""
        if self._seed_table is None:
            return (np.nan, np.nan)
        seed_a = self._seed_table.get(game.season, game.w_team_id)
        seed_b = self._seed_table.get(game.season, game.l_team_id)
        return (
            float(seed_a.seed_num) if seed_a is not None else np.nan,
            float(seed_b.seed_num) if seed_b is not None else np.nan,
        )

    # ── Internal: batch ratings ──────────────────────────────────────────

    def _compute_batch_ratings(self, games: list[Game]) -> dict[str, pd.DataFrame]:
        """Compute batch ratings from regular-season games only."""
        from ncaa_eval.transform.opponent import (
            compute_colley_ratings,
            compute_ridge_ratings,
            compute_srs_ratings,
        )

        # Filter to regular-season games for batch rating computation
        reg_games = [g for g in games if not g.is_tournament]
        if not reg_games:
            return {}

        # Build DataFrame in the format batch solvers expect
        games_df = pd.DataFrame(
            [
                {
                    "w_team_id": g.w_team_id,
                    "l_team_id": g.l_team_id,
                    "w_score": g.w_score,
                    "l_score": g.l_score,
                }
                for g in reg_games
            ]
        )

        funcs = {
            "srs": compute_srs_ratings,
            "ridge": compute_ridge_ratings,
            "colley": compute_colley_ratings,
        }

        results: dict[str, pd.DataFrame] = {}
        for rating_type in self.config.batch_rating_types:
            func = funcs.get(rating_type)
            if func is not None:
                results[rating_type] = func(games_df)  # type: ignore[operator]
            else:
                logger.warning("Unknown batch rating type: %s", rating_type)

        return results

    @staticmethod
    def _lookup_rating(
        rating_df: pd.DataFrame | None,
        team_a_id: int,
        team_b_id: int,
        col: str,
    ) -> tuple[float, float]:
        """Look up rating values for two teams from a rating DataFrame."""
        if rating_df is None or rating_df.empty:
            return (np.nan, np.nan)
        indexed = rating_df.set_index("team_id")
        val_a = indexed[col].get(team_a_id, np.nan)
        val_b = indexed[col].get(team_b_id, np.nan)
        return (float(val_a), float(val_b))

    # ── Internal: matchup deltas (Task 4) ────────────────────────────────

    def _compute_matchup_deltas(self, df: pd.DataFrame, active: frozenset[FeatureBlock]) -> pd.DataFrame:
        """Compute team_A − team_B deltas for all active features."""
        # Seed differential
        if FeatureBlock.SEED in active and "seed_num_a" in df.columns:
            df["seed_diff"] = df["seed_num_a"] - df["seed_num_b"]
        elif self.config.matchup_deltas:
            df["seed_diff"] = np.nan

        # Ordinal composite delta
        if FeatureBlock.ORDINAL in active and "ordinal_composite_a" in df.columns:
            df["delta_ordinal_composite"] = df["ordinal_composite_a"] - df["ordinal_composite_b"]

        # Batch rating deltas
        if FeatureBlock.BATCH_RATING in active:
            for rating_type in self.config.batch_rating_types:
                col_a = f"{rating_type}_a"
                col_b = f"{rating_type}_b"
                if col_a in df.columns:
                    df[f"delta_{rating_type}"] = df[col_a] - df[col_b]

        return df

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
