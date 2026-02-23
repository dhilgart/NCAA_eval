"""Minimal logistic regression model — test fixture for the Model contract.

This is NOT a production model.  It exists solely to demonstrate and
test the stateless ``Model`` interface in ~30 lines of logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import joblib  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from ncaa_eval.model.base import Model, ModelConfig
from ncaa_eval.model.registry import register_model


class LogisticRegressionConfig(ModelConfig):
    """Hyperparameters for the logistic regression test fixture."""

    model_name: Literal["logistic_regression"] = "logistic_regression"
    C: float = 1.0  # noqa: N815 — sklearn convention
    max_iter: int = 200


@register_model("logistic_regression")
class LogisticRegressionModel(Model):
    """Thin wrapper around sklearn ``LogisticRegression``."""

    def __init__(self, config: LogisticRegressionConfig | None = None) -> None:
        self._config = config or LogisticRegressionConfig()
        self._clf = LogisticRegression(C=self._config.C, max_iter=self._config.max_iter)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._clf.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        probs = self._clf.predict_proba(X)[:, 1]
        return pd.Series(probs, index=X.index)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._clf, path / "model.joblib")
        (path / "config.json").write_text(self._config.model_dump_json())

    @classmethod
    def load(cls, path: Path) -> Self:
        config = LogisticRegressionConfig.model_validate_json((path / "config.json").read_text())
        instance = cls(config)
        instance._clf = joblib.load(path / "model.joblib")
        return instance

    def get_config(self) -> LogisticRegressionConfig:
        return self._config
