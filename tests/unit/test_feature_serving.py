"""Unit tests for ncaa_eval.transform.feature_serving."""

from __future__ import annotations

import pytest

from ncaa_eval.transform.feature_serving import (
    FeatureBlock,
    FeatureConfig,
)

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
