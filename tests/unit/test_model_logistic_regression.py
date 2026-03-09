"""Unit tests for the LogisticRegressionModel test fixture."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ncaa_eval.model import list_models
from ncaa_eval.model.logistic_regression import (
    LogisticRegressionConfig,
    LogisticRegressionModel,
)
from ncaa_eval.transform.feature_serving import FeatureConfig


def _make_train_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate simple linearly separable data for testing."""
    rng = np.random.default_rng(42)
    n = 100
    X = pd.DataFrame({"feat_a": rng.standard_normal(n), "feat_b": rng.standard_normal(n)})
    y = pd.Series((X["feat_a"] + X["feat_b"] > 0).astype(int))
    return X, y


class TestAutoRegistration:
    """AC8: built-in models auto-register on package import."""

    def test_logistic_regression_registered_on_import(self) -> None:
        """Importing ncaa_eval.model must register 'logistic_regression'."""
        assert "logistic_regression" in list_models()


class TestLogisticRegressionConfig:
    def test_defaults(self) -> None:
        cfg = LogisticRegressionConfig()
        assert cfg.model_name == "logistic_regression"
        assert cfg.C == 1.0
        assert cfg.max_iter == 200

    def test_custom_values(self) -> None:
        cfg = LogisticRegressionConfig(C=0.5, max_iter=500)
        assert cfg.C == 0.5
        assert cfg.max_iter == 500


class TestLogisticRegressionModel:
    def test_fit_predict_proba(self) -> None:
        X, y = _make_train_data()
        model = LogisticRegressionModel()
        model.fit(X, y)
        preds = model.predict_proba(X)
        assert len(preds) == len(X)
        assert all(0.0 <= p <= 1.0 for p in preds)

    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        X, y = _make_train_data()
        model = LogisticRegressionModel(LogisticRegressionConfig(C=0.1, max_iter=300))
        model.fit(X, y)
        preds_before = model.predict_proba(X)

        save_dir = tmp_path / "lr_model"
        model.save(save_dir)

        loaded = LogisticRegressionModel.load(save_dir)
        preds_after = loaded.predict_proba(X)

        pd.testing.assert_series_equal(preds_before, preds_after)
        assert loaded.get_config().C == 0.1
        assert loaded.get_config().max_iter == 300

    def test_get_config(self) -> None:
        cfg = LogisticRegressionConfig(C=2.0)
        model = LogisticRegressionModel(cfg)
        assert model.get_config() is cfg

    def test_predictions_in_valid_range(self) -> None:
        X, y = _make_train_data()
        model = LogisticRegressionModel()
        model.fit(X, y)
        preds = model.predict_proba(X)
        assert (preds >= 0.0).all()
        assert (preds <= 1.0).all()


# ---------------------------------------------------------------------------
# Story 9.2: feature_config as model-level concern
# ---------------------------------------------------------------------------


class TestFeatureConfig:
    """Story 9.2: feature_config attribute on LogisticRegressionModel."""

    def test_default_feature_config(self) -> None:
        """AC1: Model exposes feature_config derived from constructor kwargs."""
        model = LogisticRegressionModel()
        assert isinstance(model.feature_config, FeatureConfig)
        assert model.feature_config.batch_rating_types == ("srs",)
        assert model.feature_config.graph_features_enabled is False
        assert model.feature_config.ordinal_composite is None

    def test_custom_feature_kwargs(self) -> None:
        """AC2: Feature-relevant kwargs threaded into FeatureConfig."""
        model = LogisticRegressionModel(
            batch_rating_types=("srs", "ridge"),
            graph_features_enabled=True,
            ordinal_composite="simple_average",
        )
        assert model.feature_config.batch_rating_types == ("srs", "ridge")
        assert model.feature_config.graph_features_enabled is True
        assert model.feature_config.ordinal_composite == "simple_average"

    def test_feature_names_set_after_fit(self) -> None:
        """AC6: feature_names_ contains ordered list of feature columns after fit."""
        X, y = _make_train_data()
        model = LogisticRegressionModel()
        assert model.feature_names_ == []
        model.fit(X, y)
        assert model.feature_names_ == ["feat_a", "feat_b"]


class TestFeatureConfigSaveLoad:
    """Story 9.2: feature_config sidecar serialization for LogisticRegressionModel."""

    def test_save_writes_feature_config_json(self, tmp_path: Path) -> None:
        """AC4: save() writes feature_config.json sidecar."""
        X, y = _make_train_data()
        model = LogisticRegressionModel()
        model.fit(X, y)
        save_dir = tmp_path / "lr_model"
        model.save(save_dir)
        assert (save_dir / "feature_config.json").exists()
        data = json.loads((save_dir / "feature_config.json").read_text())
        assert data["batch_rating_types"] == ["srs"]

    def test_load_reads_feature_config(self, tmp_path: Path) -> None:
        """AC5: load() reconstructs FeatureConfig from sidecar."""
        X, y = _make_train_data()
        model = LogisticRegressionModel(
            batch_rating_types=("srs", "ridge"),
        )
        model.fit(X, y)
        save_dir = tmp_path / "lr_model"
        model.save(save_dir)

        loaded = LogisticRegressionModel.load(save_dir)
        assert loaded.feature_config.batch_rating_types == ("srs", "ridge")

    def test_load_without_sidecar_uses_defaults(self, tmp_path: Path) -> None:
        """AC5 backward compat: load() without sidecar uses default FeatureConfig."""
        X, y = _make_train_data()
        model = LogisticRegressionModel()
        model.fit(X, y)
        save_dir = tmp_path / "lr_model"
        model.save(save_dir)
        (save_dir / "feature_config.json").unlink()

        loaded = LogisticRegressionModel.load(save_dir)
        assert loaded.feature_config.batch_rating_types == ("srs",)

    def test_load_preserves_feature_names(self, tmp_path: Path) -> None:
        """AC6: feature_names_ survives save/load round-trip."""
        X, y = _make_train_data()
        model = LogisticRegressionModel()
        model.fit(X, y)
        save_dir = tmp_path / "lr_model"
        model.save(save_dir)

        loaded = LogisticRegressionModel.load(save_dir)
        assert loaded.feature_names_ == ["feat_a", "feat_b"]
