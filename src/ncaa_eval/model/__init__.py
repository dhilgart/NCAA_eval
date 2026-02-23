"""Model implementations module."""

from __future__ import annotations

# Auto-register built-in models (import triggers @register_model decorator)
from ncaa_eval.model import (
    elo as _elo,  # noqa: F401
    logistic_regression as _lr,  # noqa: F401
    xgboost_model as _xgb,  # noqa: F401
)
from ncaa_eval.model.base import Model, ModelConfig, StatefulModel
from ncaa_eval.model.registry import ModelNotFoundError, get_model, list_models, register_model

__all__ = [
    "Model",
    "ModelConfig",
    "ModelNotFoundError",
    "StatefulModel",
    "get_model",
    "list_models",
    "register_model",
]
