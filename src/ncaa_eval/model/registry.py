"""Plugin registry for NCAA prediction models.

Provides decorator-based registration so that built-in and external
models are discoverable at runtime without modifying core code.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from ncaa_eval.model.base import Model

_T = TypeVar("_T", bound=type[Model])

_MODEL_REGISTRY: dict[str, type[Model]] = {}


class ModelNotFoundError(KeyError):
    """Raised when a requested model name is not in the registry."""


def register_model(name: str) -> Callable[[_T], _T]:
    """Class decorator that registers a :class:`Model` subclass.

    Usage::

        @register_model("elo")
        class EloModel(StatefulModel):
            ...
    """

    def decorator(cls: _T) -> _T:
        if name in _MODEL_REGISTRY:
            msg = f"Model name {name!r} is already registered to {_MODEL_REGISTRY[name].__name__}"
            raise ValueError(msg)
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model(name: str) -> type[Model]:
    """Return the model class registered under *name*.

    Raises :class:`ModelNotFoundError` if not found.
    """
    try:
        return _MODEL_REGISTRY[name]
    except KeyError:
        msg = f"No model registered with name {name!r}. Available: {list_models()}"
        raise ModelNotFoundError(msg) from None


def list_models() -> list[str]:
    """Return all registered model names (sorted)."""
    return sorted(_MODEL_REGISTRY)
