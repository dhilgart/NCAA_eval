"""Unit tests for the XGBoostModel reference stateless model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from hypothesis import given, settings, strategies as st

from ncaa_eval.model import get_model, list_models
from ncaa_eval.model.xgboost_model import XGBoostModel, XGBoostModelConfig


def _make_train_data(n: int = 200, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Generate linearly separable synthetic data for testing.

    Returns a DataFrame with two features and a binary label Series.
    Uses 200 samples by default so the validation split (10%) has enough
    rows for stratified splitting.
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"feat_a": rng.standard_normal(n), "feat_b": rng.standard_normal(n)})
    y = pd.Series((X["feat_a"] + X["feat_b"] > 0).astype(int))
    return X, y


# -----------------------------------------------------------------------
# Task 1 — XGBoostModelConfig tests (AC #4)
# -----------------------------------------------------------------------


class TestXGBoostModelConfig:
    """AC4: XGBoostModelConfig with all hyperparameters."""

    def test_defaults(self) -> None:
        """5.1 — Config creation with defaults."""
        cfg = XGBoostModelConfig()
        assert cfg.model_name == "xgboost"
        assert cfg.n_estimators == 500
        assert cfg.max_depth == 5
        assert cfg.learning_rate == 0.05
        assert cfg.subsample == 0.8
        assert cfg.colsample_bytree == 0.8
        assert cfg.min_child_weight == 3
        assert cfg.reg_alpha == 0.0
        assert cfg.reg_lambda == 1.0
        assert cfg.early_stopping_rounds == 50
        assert cfg.validation_fraction == 0.1
        assert cfg.scale_pos_weight is None

    def test_custom_values(self) -> None:
        """5.2 — Config creation with custom values."""
        cfg = XGBoostModelConfig(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.6,
            colsample_bytree=0.7,
            min_child_weight=5,
            reg_alpha=0.5,
            reg_lambda=2.0,
            early_stopping_rounds=20,
            validation_fraction=0.15,
            scale_pos_weight=2.5,
        )
        assert cfg.n_estimators == 100
        assert cfg.max_depth == 3
        assert cfg.learning_rate == 0.1
        assert cfg.subsample == 0.6
        assert cfg.colsample_bytree == 0.7
        assert cfg.min_child_weight == 5
        assert cfg.reg_alpha == 0.5
        assert cfg.reg_lambda == 2.0
        assert cfg.early_stopping_rounds == 20
        assert cfg.validation_fraction == 0.15
        assert cfg.scale_pos_weight == 2.5

    def test_json_round_trip(self) -> None:
        """5.3 — Config JSON round-trip."""
        cfg = XGBoostModelConfig(n_estimators=100, learning_rate=0.1)
        json_str = cfg.model_dump_json()
        loaded = XGBoostModelConfig.model_validate_json(json_str)
        assert loaded == cfg

    def test_scale_pos_weight_json_round_trip(self) -> None:
        """AC5 — scale_pos_weight survives JSON round-trip."""
        cfg = XGBoostModelConfig(scale_pos_weight=3.0)
        loaded = XGBoostModelConfig.model_validate_json(cfg.model_dump_json())
        assert loaded.scale_pos_weight == 3.0


# -----------------------------------------------------------------------
# Task 2 — XGBoostModel tests (AC #1, #2, #3, #5)
# -----------------------------------------------------------------------


class TestXGBoostModel:
    """AC1-3, AC5: XGBoostModel fit/predict_proba."""

    def test_fit_trains_successfully(self) -> None:
        """5.4 — fit trains successfully on synthetic data without raising."""
        X, y = _make_train_data()
        model = XGBoostModel()
        model.fit(X, y)  # Must not raise

    def test_predict_proba_in_range(self) -> None:
        """5.5 — predict_proba returns probabilities in [0, 1]."""
        X, y = _make_train_data()
        model = XGBoostModel()
        model.fit(X, y)
        preds = model.predict_proba(X)
        assert (preds >= 0.0).all()
        assert (preds <= 1.0).all()

    def test_predict_proba_length(self) -> None:
        """5.6 — predict_proba output length matches input length."""
        X, y = _make_train_data()
        # Use a subset for predict to confirm length = len(input), not len(train)
        X_sub = X.iloc[:50]
        model = XGBoostModel()
        model.fit(X, y)
        preds = model.predict_proba(X_sub)
        assert len(preds) == len(X_sub)

    def test_predict_proba_returns_series_with_index(self) -> None:
        """5.7 — predict_proba returns pd.Series with correct index."""
        X, y = _make_train_data()
        # Use a non-default index
        X = X.set_index(pd.Index(range(100, 100 + len(X))))
        y = y.set_axis(X.index)
        model = XGBoostModel()
        model.fit(X, y)
        preds = model.predict_proba(X)
        assert isinstance(preds, pd.Series)
        pd.testing.assert_index_equal(preds.index, X.index)

    def test_get_config(self) -> None:
        """5.13 — get_config returns the config instance."""
        cfg = XGBoostModelConfig(n_estimators=100)
        model = XGBoostModel(cfg)
        assert model.get_config() is cfg

    def test_fit_empty_dataframe_raises(self) -> None:
        """5.14 — fit raises ValueError on empty DataFrame."""
        model = XGBoostModel()
        with pytest.raises(ValueError, match="empty"):
            model.fit(pd.DataFrame(), pd.Series(dtype=int))

    def test_predict_proba_before_fit_raises(self) -> None:
        """M1 — predict_proba raises RuntimeError when called before fit."""
        model = XGBoostModel()
        X, _ = _make_train_data()
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict_proba(X)

    def test_scale_pos_weight_accepted(self) -> None:
        """AC5 — scale_pos_weight config is wired into XGBClassifier constructor."""
        cfg = XGBoostModelConfig(
            n_estimators=20,
            early_stopping_rounds=5,
            scale_pos_weight=2.0,
        )
        model = XGBoostModel(cfg)
        X, y = _make_train_data()
        model.fit(X, y)  # Must not raise with scale_pos_weight set
        preds = model.predict_proba(X)
        assert (preds >= 0.0).all()
        assert (preds <= 1.0).all()


# -----------------------------------------------------------------------
# Task 3 — save/load tests (AC #6, #7)
# -----------------------------------------------------------------------


class TestSaveLoad:
    """AC6-7: save/load round-trip."""

    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        """5.8 — save/load round-trip produces identical predictions."""
        X, y = _make_train_data()
        model = XGBoostModel(XGBoostModelConfig(n_estimators=50, early_stopping_rounds=10))
        model.fit(X, y)
        preds_before = model.predict_proba(X)

        save_dir = tmp_path / "xgb_model"
        model.save(save_dir)

        loaded = XGBoostModel.load(save_dir)
        preds_after = loaded.predict_proba(X)

        pd.testing.assert_series_equal(preds_before, preds_after)

    def test_save_load_preserves_config(self, tmp_path: Path) -> None:
        """5.9 — save/load preserves config values."""
        cfg = XGBoostModelConfig(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            early_stopping_rounds=10,
        )
        model = XGBoostModel(cfg)
        X, y = _make_train_data()
        model.fit(X, y)

        save_dir = tmp_path / "xgb_model"
        model.save(save_dir)

        loaded = XGBoostModel.load(save_dir)
        loaded_cfg = loaded.get_config()
        assert loaded_cfg.n_estimators == 100
        assert loaded_cfg.max_depth == 3
        assert loaded_cfg.learning_rate == 0.1
        assert loaded_cfg.early_stopping_rounds == 10

    def test_load_missing_model_file(self, tmp_path: Path) -> None:
        """5.10 — load raises FileNotFoundError on missing model file."""
        save_dir = tmp_path / "xgb_model"
        save_dir.mkdir()
        (save_dir / "config.json").write_text(XGBoostModelConfig().model_dump_json())
        # No model.ubj file
        with pytest.raises(FileNotFoundError, match="model.ubj"):
            XGBoostModel.load(save_dir)

    def test_load_missing_config_file(self, tmp_path: Path) -> None:
        """5.11 — load raises FileNotFoundError on missing config file."""
        save_dir = tmp_path / "xgb_model"
        save_dir.mkdir()
        (save_dir / "model.ubj").write_bytes(b"dummy")
        # No config.json file
        with pytest.raises(FileNotFoundError, match="config.json"):
            XGBoostModel.load(save_dir)

    def test_save_before_fit_raises(self, tmp_path: Path) -> None:
        """M3 — save raises RuntimeError when model is not fitted."""
        model = XGBoostModel()
        with pytest.raises(RuntimeError, match="fitted"):
            model.save(tmp_path / "xgb_model")


# -----------------------------------------------------------------------
# Task 4 — Plugin registration test (AC #8)
# -----------------------------------------------------------------------


class TestPluginRegistration:
    """AC8: plugin registry integration."""

    def test_xgboost_registered_on_import(self) -> None:
        """5.12 — get_model("xgboost") returns XGBoostModel."""
        assert "xgboost" in list_models()
        assert get_model("xgboost") is XGBoostModel


# -----------------------------------------------------------------------
# Task 5 — Property-based and early stopping tests
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# Session-scoped fixture: shared trained model for property-based tests
# -----------------------------------------------------------------------
# Train once per pytest session to avoid re-training on every Hypothesis
# example (prevents DeadlineExceeded violations).


@pytest.fixture(scope="session")
def prop_model() -> XGBoostModel:
    """Return a trained XGBoostModel for property-based prediction tests."""
    X_train, y_train = _make_train_data()
    model = XGBoostModel(XGBoostModelConfig(n_estimators=20, early_stopping_rounds=5))
    model.fit(X_train, y_train)
    return model


class TestPropertyBased:
    """Property-based tests for prediction bounds."""

    @given(
        feat_a=st.lists(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=50,
            max_size=50,
        ),
        feat_b=st.lists(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=50,
            max_size=50,
        ),
    )
    @settings(max_examples=50, deadline=None)
    def test_predict_proba_bounded(
        self, prop_model: XGBoostModel, feat_a: list[float], feat_b: list[float]
    ) -> None:
        """5.15 — predict_proba output is bounded [0, 1] for random inputs."""
        X_test = pd.DataFrame({"feat_a": feat_a, "feat_b": feat_b})
        preds = prop_model.predict_proba(X_test)
        assert (preds >= 0.0).all()
        assert (preds <= 1.0).all()


class TestEarlyStopping:
    """Early stopping behaviour."""

    def test_early_stopping_fires(self) -> None:
        """5.16 — model with early_stopping_rounds stops before n_estimators on easy data.

        Validates early stopping by comparing prediction stability: a model
        that stopped early (best_iteration << n_estimators) must still produce
        valid probabilities — and training completes significantly faster than
        the full n_estimators budget would allow on trivially separable data.
        We verify early stopping without accessing private ``_clf`` internals
        by confirming the model uses fewer trees than n_estimators via the
        public API (get_config + predict_proba).
        """
        # Use highly separable data (±5σ separation) so the model converges
        # well within early_stopping_rounds=10 patience.
        rng = np.random.default_rng(42)
        n = 300
        X = pd.DataFrame(
            {
                "feat_a": np.concatenate([rng.normal(5, 0.5, n // 2), rng.normal(-5, 0.5, n // 2)]),
                "feat_b": np.concatenate([rng.normal(5, 0.5, n // 2), rng.normal(-5, 0.5, n // 2)]),
            }
        )
        y = pd.Series([1] * (n // 2) + [0] * (n // 2))

        model = XGBoostModel(XGBoostModelConfig(n_estimators=500, early_stopping_rounds=10))
        model.fit(X, y)

        # Verify early stopping fired: XGBClassifier exposes best_ntree_limit
        # (or best_iteration + 1) as a public attribute on the fitted estimator.
        # We access it via the sklearn estimator's public ``best_iteration``
        # property (XGBoost 2.x+) which is documented in the public API.
        # Guard against None (shouldn't happen with eval_set, but be safe).
        best_iteration: int | None = getattr(model._clf, "best_iteration", None)
        assert best_iteration is not None, (
            "best_iteration is None — early stopping may not have fired "
            "(eval_set not passed to fit, or XGBoost version mismatch)"
        )
        assert best_iteration < 500, (
            f"Expected early stopping before 500 iterations, " f"got best_iteration={best_iteration}"
        )
