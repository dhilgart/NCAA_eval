"""XGBoost gradient-boosting model — reference stateless model.

Wraps :class:`xgboost.XGBClassifier` behind the :class:`Model` ABC,
providing ``fit`` / ``predict_proba`` / ``save`` / ``load`` with XGBoost's
native UBJSON persistence format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import pandas as pd  # type: ignore[import-untyped]
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]
from xgboost import XGBClassifier

from ncaa_eval.model.base import Model, ModelConfig
from ncaa_eval.model.registry import register_model


class XGBoostModelConfig(ModelConfig):
    """Hyperparameters for the XGBoost gradient-boosting model.

    Defaults from ``specs/research/modeling-approaches.md`` §5.5 and §6.4.
    """

    model_name: Literal["xgboost"] = "xgboost"
    n_estimators: int = 500
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50
    validation_fraction: float = 0.1


@register_model("xgboost")
class XGBoostModel(Model):
    """XGBoost binary classifier wrapping :class:`XGBClassifier`.

    This is a *stateless* model — it implements :class:`Model` directly
    (no ``StatefulModel`` lifecycle hooks).

    **Label balance convention:** The feature server typically assigns
    ``team_a = w_team_id`` (the winner), so ``y`` may be heavily biased
    toward 1.  Callers should either randomise team assignment before
    training (recommended) or set ``scale_pos_weight`` in the config to
    ``count(y==0) / count(y==1)``.  The default ``scale_pos_weight`` is
    ``None`` (XGBoost default = 1.0), appropriate when team assignment is
    randomised.
    """

    def __init__(self, config: XGBoostModelConfig | None = None) -> None:
        self._config = config or XGBoostModelConfig()
        self._clf = XGBClassifier(
            n_estimators=self._config.n_estimators,
            max_depth=self._config.max_depth,
            learning_rate=self._config.learning_rate,
            subsample=self._config.subsample,
            colsample_bytree=self._config.colsample_bytree,
            min_child_weight=self._config.min_child_weight,
            reg_alpha=self._config.reg_alpha,
            reg_lambda=self._config.reg_lambda,
            objective="binary:logistic",
            eval_metric="logloss",
            early_stopping_rounds=self._config.early_stopping_rounds,
            random_state=42,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train on feature matrix *X* and binary labels *y*.

        Automatically splits *X* into train/validation sets using
        ``validation_fraction`` from the config.  The validation set is
        used for early stopping via ``eval_set``.

        **Label balance convention:** ``team_a`` assignment in the feature
        server is typically ``w_team_id`` (the winner).  If labels are
        imbalanced, either randomise team assignment upstream or set
        ``scale_pos_weight`` = ``count(y==0) / count(y==1)`` in the
        ``XGBoostModelConfig``.

        Raises
        ------
        ValueError
            If *X* is empty.
        """
        if X.empty:
            msg = "Cannot fit on an empty DataFrame"
            raise ValueError(msg)

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self._config.validation_fraction,
            random_state=42,
            stratify=y,
        )
        self._clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Return P(team_a wins) for each row of *X*."""
        probs: pd.Series[float] = pd.Series(self._clf.predict_proba(X)[:, 1], index=X.index)
        return probs

    def save(self, path: Path) -> None:
        """Persist the trained model to *path* directory.

        Writes two files:
        - ``model.ubj`` — XGBoost native UBJSON format (stable across versions)
        - ``config.json`` — Pydantic-serialised hyperparameter config
        """
        path.mkdir(parents=True, exist_ok=True)
        self._clf.save_model(str(path / "model.ubj"))
        (path / "config.json").write_text(self._config.model_dump_json())

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load a previously-saved XGBoost model from *path*.

        Raises
        ------
        FileNotFoundError
            If either ``config.json`` or ``model.ubj`` is missing.
        """
        config_path = path / "config.json"
        model_path = path / "model.ubj"
        missing = [p for p in (config_path, model_path) if not p.exists()]
        if missing:
            missing_names = ", ".join(p.name for p in missing)
            msg = (
                f"Incomplete save at {path!r}: missing {missing_names}. "
                "The save may have been interrupted."
            )
            raise FileNotFoundError(msg)
        config = XGBoostModelConfig.model_validate_json(config_path.read_text())
        instance = cls(config)
        instance._clf.load_model(str(model_path))
        return instance

    def get_config(self) -> XGBoostModelConfig:
        """Return the Pydantic-validated configuration for this model."""
        return self._config
