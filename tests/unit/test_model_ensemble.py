"""Unit tests for the StackedEnsemble dataclass and OOF training pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from ncaa_eval.model import list_models
from ncaa_eval.model.base import StatefulModel
from ncaa_eval.model.ensemble import StackedEnsemble, StackedEnsembleConfig
from ncaa_eval.model.logistic_regression import LogisticRegressionModel
from ncaa_eval.transform.feature_serving import FeatureConfig

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_lr(
    *,
    batch_rating_types: tuple[str, ...] = ("srs",),
    graph_features_enabled: bool = False,
    ordinal_composite: str | None = None,
    elo_enabled: bool = False,
) -> LogisticRegressionModel:
    """Create a LogisticRegressionModel with the given feature config."""
    model = LogisticRegressionModel(
        batch_rating_types=batch_rating_types,  # type: ignore[arg-type]
        graph_features_enabled=graph_features_enabled,
        ordinal_composite=ordinal_composite,  # type: ignore[arg-type]
    )
    if elo_enabled:
        object.__setattr__(  # FeatureConfig is frozen
            model,
            "feature_config",
            FeatureConfig(
                batch_rating_types=batch_rating_types,  # type: ignore[arg-type]
                graph_features_enabled=graph_features_enabled,
                ordinal_composite=ordinal_composite,  # type: ignore[arg-type]
                elo_enabled=True,
            ),
        )
    return model


def _make_train_data(n: int = 100) -> tuple[pd.DataFrame, pd.Series]:
    """Generate simple linearly separable data."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"feat_a": rng.standard_normal(n), "feat_b": rng.standard_normal(n)})
    y = pd.Series((X["feat_a"] + X["feat_b"] > 0).astype(int))
    return X, y


# ── Test: StackedEnsemble construction ───────────────────────────────────────


class TestStackedEnsembleConstruction:
    """Tests for __post_init__ validation."""

    def test_min_two_base_models(self) -> None:
        """ValueError raised with fewer than 2 base models."""
        with pytest.raises(ValueError, match="at least 2 base models"):
            StackedEnsemble(
                base_models=[_make_lr()],
                meta_learner=_make_lr(),
            )

    def test_meta_learner_stateful_rejected(self) -> None:
        """TypeError raised when meta_learner is StatefulModel."""
        # Create a mock StatefulModel by using a real Elo model
        from ncaa_eval.model.elo import EloModel

        elo = EloModel()
        assert isinstance(elo, StatefulModel)

        with pytest.raises(TypeError, match="stateless Model, not StatefulModel"):
            StackedEnsemble(
                base_models=[_make_lr(), _make_lr()],
                meta_learner=elo,
            )

    def test_valid_construction(self) -> None:
        """Two LR models and LR meta-learner is valid."""
        ensemble = StackedEnsemble(
            base_models=[_make_lr(), _make_lr()],
            meta_learner=_make_lr(),
        )
        assert len(ensemble.base_models) == 2

    def test_default_contextual_features(self) -> None:
        """Default contextual features match spec."""
        ensemble = StackedEnsemble(
            base_models=[_make_lr(), _make_lr()],
            meta_learner=_make_lr(),
        )
        assert ensemble.contextual_features == [
            "seed_diff",
            "is_tournament",
            "loc_encoding",
        ]


# ── Test: feature_config union logic ─────────────────────────────────────────


class TestFeatureConfigUnion:
    """AC #8: feature_config returns the union of all base model configs."""

    def test_batch_rating_types_union(self) -> None:
        """batch_rating_types are merged across base models."""
        m1 = _make_lr(batch_rating_types=("srs",))
        m2 = _make_lr(batch_rating_types=("ridge", "colley"))
        ensemble = StackedEnsemble(
            base_models=[m1, m2],
            meta_learner=_make_lr(),
        )
        fc = ensemble.feature_config
        assert set(fc.batch_rating_types) == {"srs", "ridge", "colley"}

    def test_graph_features_or(self) -> None:
        """graph_features_enabled is True if ANY base model enables it."""
        m1 = _make_lr(graph_features_enabled=False)
        m2 = _make_lr(graph_features_enabled=True)
        ensemble = StackedEnsemble(
            base_models=[m1, m2],
            meta_learner=_make_lr(),
        )
        assert ensemble.feature_config.graph_features_enabled is True

    def test_elo_enabled_or(self) -> None:
        """elo_enabled is True if ANY base model enables it."""
        m1 = _make_lr(elo_enabled=False)
        m2 = _make_lr(elo_enabled=True)
        ensemble = StackedEnsemble(
            base_models=[m1, m2],
            meta_learner=_make_lr(),
        )
        assert ensemble.feature_config.elo_enabled is True

    def test_ordinal_composite_first_non_none(self) -> None:
        """ordinal_composite takes the first non-None value."""
        m1 = _make_lr(ordinal_composite=None)
        m2 = _make_lr(ordinal_composite="simple_average")
        ensemble = StackedEnsemble(
            base_models=[m1, m2],
            meta_learner=_make_lr(),
        )
        assert ensemble.feature_config.ordinal_composite == "simple_average"

    def test_all_none_ordinal_composite(self) -> None:
        """ordinal_composite is None when all base models have None."""
        m1 = _make_lr(ordinal_composite=None)
        m2 = _make_lr(ordinal_composite=None)
        ensemble = StackedEnsemble(
            base_models=[m1, m2],
            meta_learner=_make_lr(),
        )
        assert ensemble.feature_config.ordinal_composite is None

    def test_sequential_windows_union(self) -> None:
        """sequential_windows are union-merged and sorted."""
        m1 = LogisticRegressionModel()
        m2 = LogisticRegressionModel()
        # Default windows are (5, 10, 20) — same for both, union = same
        ensemble = StackedEnsemble(
            base_models=[m1, m2],
            meta_learner=_make_lr(),
        )
        assert ensemble.feature_config.sequential_windows == (5, 10, 20)


# ── Test: config agreement ───────────────────────────────────────────────────


class TestFeatureConfigAgreement:
    """Verify ValueError when base models disagree on scope fields."""

    def test_gender_scope_mismatch_raises(self) -> None:
        m1 = _make_lr()
        m2 = _make_lr()
        # Forcefully set different gender_scope
        object.__setattr__(
            m2,
            "feature_config",
            FeatureConfig(gender_scope="W"),
        )
        ensemble = StackedEnsemble(
            base_models=[m1, m2],
            meta_learner=_make_lr(),
        )
        with pytest.raises(ValueError, match="gender_scope"):
            _ = ensemble.feature_config

    def test_matchup_deltas_mismatch_raises(self) -> None:
        m1 = _make_lr()
        m2 = _make_lr()
        object.__setattr__(
            m2,
            "feature_config",
            FeatureConfig(matchup_deltas=False),
        )
        ensemble = StackedEnsemble(
            base_models=[m1, m2],
            meta_learner=_make_lr(),
        )
        with pytest.raises(ValueError, match="matchup_deltas"):
            _ = ensemble.feature_config

    def test_dataset_scope_mismatch_raises(self) -> None:
        m1 = _make_lr()
        m2 = _make_lr()
        # Default dataset_scope is "kaggle"; set m2 to "all" to force mismatch
        object.__setattr__(
            m2,
            "feature_config",
            FeatureConfig(dataset_scope="all"),
        )
        ensemble = StackedEnsemble(
            base_models=[m1, m2],
            meta_learner=_make_lr(),
        )
        with pytest.raises(ValueError, match="dataset_scope"):
            _ = ensemble.feature_config


# ── Test: StackedEnsembleConfig ──────────────────────────────────────────────


class TestStackedEnsembleConfig:
    def test_get_config(self) -> None:
        ensemble = StackedEnsemble(
            base_models=[_make_lr(), _make_lr()],
            meta_learner=_make_lr(),
        )
        cfg = ensemble.get_config()
        assert isinstance(cfg, StackedEnsembleConfig)
        assert cfg.model_name == "ensemble"
        assert len(cfg.base_model_types) == 2
        assert all(t == "logistic_regression" for t in cfg.base_model_types)


# ── Test: ensemble registered ────────────────────────────────────────────────


class TestEnsembleRegistered:
    """AC: 'ensemble' is in list_models()."""

    def test_ensemble_in_list_models(self) -> None:
        assert "ensemble" in list_models()


# ── Test: save/load round-trip ───────────────────────────────────────────────


class TestSaveLoadRoundTrip:
    """AC #7: save/load round-trip preserves manifest and base model count."""

    def test_round_trip(self, tmp_path: Path) -> None:
        m1 = _make_lr()
        m2 = _make_lr(batch_rating_types=("srs", "ridge"))
        meta = _make_lr()

        # Fit all models so they can be saved
        X, y = _make_train_data()
        m1.fit(X, y)
        m2.fit(X, y)
        meta.fit(X, y)

        ensemble = StackedEnsemble(
            base_models=[m1, m2],
            meta_learner=meta,
            contextual_features=["seed_diff", "is_tournament"],
        )

        save_dir = tmp_path / "ensemble"
        ensemble.save(save_dir)

        # Check files exist
        assert (save_dir / "manifest.json").exists()
        assert (save_dir / "feature_config.json").exists()
        assert (save_dir / "base_models" / "base_0").exists()
        assert (save_dir / "base_models" / "base_1").exists()
        assert (save_dir / "meta_learner").exists()

        # Load and verify
        loaded = StackedEnsemble.load(save_dir)
        assert len(loaded.base_models) == 2
        assert loaded.contextual_features == ["seed_diff", "is_tournament"]

        # Verify predictions match after round-trip
        preds_before = meta.predict_proba(X)
        preds_after = loaded.meta_learner.predict_proba(X)
        pd.testing.assert_series_equal(preds_before, preds_after)

    def test_manifest_contents(self, tmp_path: Path) -> None:
        m1 = _make_lr()
        m2 = _make_lr()
        meta = _make_lr()
        X, y = _make_train_data()
        m1.fit(X, y)
        m2.fit(X, y)
        meta.fit(X, y)

        ensemble = StackedEnsemble(
            base_models=[m1, m2],
            meta_learner=meta,
        )
        save_dir = tmp_path / "ensemble"
        ensemble.save(save_dir)

        manifest = json.loads((save_dir / "manifest.json").read_text())
        assert manifest["base_model_count"] == 2
        assert manifest["base_model_types"] == [
            "logistic_regression",
            "logistic_regression",
        ]
        assert "contextual_features" in manifest
        assert manifest["meta_learner_type"] == "logistic_regression"


# ── Test: OOF alignment ─────────────────────────────────────────────────────


class TestOofAlignment:
    """AC #3: OOF alignment inner join + >5% drop warning."""

    def test_inner_join_alignment(self) -> None:
        """Overlapping but not identical game_id sets produce correct join."""
        from ncaa_eval.cli.train import _align_oof_predictions

        df1 = pd.DataFrame(
            {
                "game_id": ["g1", "g2", "g3"],
                "pred_base_0": [0.5, 0.6, 0.7],
                "team_a_won": [1, 0, 1],
            }
        )
        df2 = pd.DataFrame(
            {
                "game_id": ["g2", "g3", "g4"],
                "pred_base_1": [0.4, 0.5, 0.6],
                "team_a_won": [0, 1, 0],
            }
        )
        aligned = _align_oof_predictions([df1, df2])
        assert len(aligned) == 2
        assert set(aligned["game_id"].tolist()) == {"g2", "g3"}
        assert "pred_base_0" in aligned.columns
        assert "pred_base_1" in aligned.columns

    def test_warning_on_large_drop(self, caplog: pytest.LogCaptureFixture) -> None:
        """Warning logged when >5% of games are dropped by alignment."""
        from ncaa_eval.cli.train import _align_oof_predictions

        # Create frames where only 1 out of 20 overlaps (95% drop)
        ids_a = [f"g{i}" for i in range(20)]
        ids_b = [f"g{i}" for i in range(19, 40)]  # only g19 overlaps

        df1 = pd.DataFrame(
            {
                "game_id": ids_a,
                "pred_base_0": [0.5] * 20,
                "team_a_won": [1] * 20,
            }
        )
        df2 = pd.DataFrame(
            {
                "game_id": ids_b,
                "pred_base_1": [0.5] * 21,
                "team_a_won": [1] * 21,
            }
        )

        # Temporarily enable propagation so caplog (root-level handler) sees
        # records from the ncaa_eval hierarchy (propagate=False set by
        # configure_logging may be active from prior tests).
        ncaa_logger = logging.getLogger("ncaa_eval")
        orig_propagate = ncaa_logger.propagate
        ncaa_logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING):
                aligned = _align_oof_predictions([df1, df2])
        finally:
            ncaa_logger.propagate = orig_propagate

        assert len(aligned) == 1
        assert any("dropped" in record.message.lower() for record in caplog.records)

    def test_no_warning_on_small_drop(self, caplog: pytest.LogCaptureFixture) -> None:
        """No warning when drop is well below 5%."""
        from ncaa_eval.cli.train import _align_oof_predictions

        # 100 games, 98 overlap = 2% drop (well under 5%)
        ids_common = [f"g{i}" for i in range(98)]
        df1 = pd.DataFrame(
            {
                "game_id": ids_common + ["g_extra_a1", "g_extra_a2"],
                "pred_base_0": [0.5] * 100,
                "team_a_won": [1] * 100,
            }
        )
        df2 = pd.DataFrame(
            {
                "game_id": ids_common + ["g_extra_b1", "g_extra_b2"],
                "pred_base_1": [0.5] * 100,
                "team_a_won": [1] * 100,
            }
        )

        ncaa_logger = logging.getLogger("ncaa_eval")
        orig_propagate = ncaa_logger.propagate
        ncaa_logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING):
                _align_oof_predictions([df1, df2])
        finally:
            ncaa_logger.propagate = orig_propagate

        drop_warnings = [r for r in caplog.records if "dropped" in r.message.lower()]
        assert len(drop_warnings) == 0


# ── Test: meta-training set construction ─────────────────────────────────────


class TestMetaTrainingSet:
    """AC #4: meta-training DataFrame has correct columns."""

    def test_shape_and_columns(self) -> None:
        from ncaa_eval.cli.train import _build_meta_training_set

        aligned = pd.DataFrame(
            {
                "game_id": ["g1", "g2"],
                "pred_base_0": [0.5, 0.6],
                "pred_base_1": [0.4, 0.5],
                "seed_diff": [1.0, -2.0],
                "is_tournament": [True, False],
                "loc_encoding": [1, 0],
                "team_a_won": [1, 0],
            }
        )
        meta_X, meta_y = _build_meta_training_set(
            aligned,
            contextual_features=["seed_diff", "is_tournament", "loc_encoding"],
        )
        assert list(meta_X.columns) == [
            "pred_base_0",
            "pred_base_1",
            "seed_diff",
            "is_tournament",
            "loc_encoding",
        ]
        assert len(meta_y) == 2
        assert list(meta_y) == [1, 0]


# ── Test: run_training dispatches to ensemble ────────────────────────────────


class TestRunTrainingDispatch:
    """AC #1: run_training detects StackedEnsemble and dispatches."""

    def test_dispatches_to_ensemble_training(self) -> None:
        """Calling run_training with StackedEnsemble calls _run_ensemble_training."""
        ensemble = StackedEnsemble(
            base_models=[_make_lr(), _make_lr()],
            meta_learner=_make_lr(),
        )
        with patch("ncaa_eval.cli.train._run_ensemble_training") as mock_fn:
            from ncaa_eval.cli.train import run_training

            mock_fn.return_value = None  # won't actually run
            try:
                run_training(
                    ensemble,
                    start_year=2015,
                    end_year=2020,
                    data_dir=Path("/tmp/fake"),
                    output_dir=Path("/tmp/fake_out"),
                    model_name="ensemble",
                )
            except Exception:  # noqa: BLE001
                pass  # We just need to verify dispatch was called

            mock_fn.assert_called_once()


# ── Test: meta_column_order persistence ──────────────────────────────────────


class TestMetaColumnOrder:
    """AC #3: meta_column_order persisted in manifest and loaded correctly."""

    def test_default_meta_column_order_empty(self) -> None:
        """Default meta_column_order is an empty list."""
        ensemble = StackedEnsemble(
            base_models=[_make_lr(), _make_lr()],
            meta_learner=_make_lr(),
        )
        assert ensemble.meta_column_order == []

    def test_meta_column_order_save_and_load(self, tmp_path: Path) -> None:
        """meta_column_order round-trips through save/load via manifest."""
        m1 = _make_lr()
        m2 = _make_lr()
        meta = _make_lr()
        X, y = _make_train_data()
        m1.fit(X, y)
        m2.fit(X, y)
        meta.fit(X, y)

        ensemble = StackedEnsemble(
            base_models=[m1, m2],
            meta_learner=meta,
        )
        expected_order = ["pred_base_0", "pred_base_1", "seed_diff", "is_tournament", "loc_encoding"]
        ensemble.meta_column_order = expected_order

        save_dir = tmp_path / "ensemble"
        ensemble.save(save_dir)

        # Verify manifest contains meta_column_order
        manifest = json.loads((save_dir / "manifest.json").read_text())
        assert manifest["meta_column_order"] == expected_order

        # Verify load restores it
        loaded = StackedEnsemble.load(save_dir)
        assert loaded.meta_column_order == expected_order

    def test_load_without_meta_column_order_defaults_empty(self, tmp_path: Path) -> None:
        """Loading an ensemble saved before meta_column_order gives empty list."""
        m1 = _make_lr()
        m2 = _make_lr()
        meta = _make_lr()
        X, y = _make_train_data()
        m1.fit(X, y)
        m2.fit(X, y)
        meta.fit(X, y)

        ensemble = StackedEnsemble(
            base_models=[m1, m2],
            meta_learner=meta,
        )
        save_dir = tmp_path / "ensemble"
        ensemble.save(save_dir)

        # Remove meta_column_order from manifest to simulate old format
        manifest_path = save_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.pop("meta_column_order", None)
        manifest_path.write_text(json.dumps(manifest))

        loaded = StackedEnsemble.load(save_dir)
        assert loaded.meta_column_order == []
