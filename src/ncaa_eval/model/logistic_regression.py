"""Minimal logistic regression model — test fixture for the Model contract.

This is NOT a production model.  It exists solely to demonstrate and
test the stateless ``Model`` interface in ~30 lines of logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Self

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from ncaa_eval.model._feature_config_io import load_feature_config, save_feature_config
from ncaa_eval.model.base import Model, ModelConfig
from ncaa_eval.model.registry import register_model
from ncaa_eval.transform.feature_serving import (
    BatchRatingType,
    FeatureConfig,
    OrdinalCompositeMethod,
)


class LogisticRegressionConfig(ModelConfig):
    """Hyperparameters for the logistic regression test fixture."""

    model_name: Literal["logistic_regression"] = "logistic_regression"
    C: float = 1.0  # noqa: N815 — sklearn convention
    max_iter: int = 200


@register_model("logistic_regression")
class LogisticRegressionModel(Model):
    """Thin wrapper around sklearn ``LogisticRegression``."""

    def __init__(
        self,
        config: LogisticRegressionConfig | None = None,
        *,
        batch_rating_types: tuple[BatchRatingType, ...] = ("srs",),
        graph_features_enabled: bool = False,
        ordinal_composite: OrdinalCompositeMethod | None = None,
    ) -> None:
        """Initialize logistic regression model with optional configuration.

        Args:
            config: Pydantic config; defaults to
                :class:`LogisticRegressionConfig` when ``None``.
            batch_rating_types: Which batch rating systems to include.
            graph_features_enabled: Whether to compute graph centrality features.
            ordinal_composite: Composite method for ordinal systems.
        """
        self._config = config or LogisticRegressionConfig()
        self._clf = LogisticRegression(C=self._config.C, max_iter=self._config.max_iter)
        self.feature_config = FeatureConfig(
            batch_rating_types=batch_rating_types,
            graph_features_enabled=graph_features_enabled,
            ordinal_composite=ordinal_composite,
        )
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model on feature matrix *X* and labels *y*."""
        self.feature_names_ = list(X.columns)
        self._clf.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Return P(team_a wins) in [0, 1] for each row of *X*."""
        probs = self._clf.predict_proba(X)[:, 1]
        return pd.Series(probs, index=X.index)

    def save(self, path: Path) -> None:
        """Persist the trained classifier, config, and feature config to *path*."""
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._clf, path / "model.joblib")
        (path / "config.json").write_text(self._config.model_dump_json())
        (path / "feature_names.json").write_text(json.dumps(self.feature_names_))
        save_feature_config(self.feature_config, path)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load a previously-saved model from *path*."""
        config = LogisticRegressionConfig.model_validate_json((path / "config.json").read_text())
        instance = cls(config)
        instance._clf = joblib.load(path / "model.joblib")
        feature_names_path = path / "feature_names.json"
        if feature_names_path.exists():
            instance.feature_names_ = json.loads(feature_names_path.read_text())
        loaded_fc = load_feature_config(path)
        if loaded_fc is not None:
            instance.feature_config = loaded_fc
        return instance

    def get_feature_importances(self) -> list[tuple[str, float]] | None:
        """Return absolute coefficient values as feature importance."""
        if not self.feature_names_:
            return None
        coefs = np.abs(self._clf.coef_[0])
        return list(zip(self.feature_names_, coefs.tolist()))

    def get_config(self) -> LogisticRegressionConfig:
        """Return the Pydantic-validated configuration."""
        return self._config
