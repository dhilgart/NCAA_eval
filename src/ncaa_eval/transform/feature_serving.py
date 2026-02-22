"""Declarative feature serving layer for NCAA basketball prediction.

Combines sequential, graph, batch-rating, ordinal, seed, and Elo feature
building blocks into a temporally-safe, matchup-level feature matrix.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


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
