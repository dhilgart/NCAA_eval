"""Unit tests for the model plugin registry."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Self

import pandas as pd  # type: ignore[import-untyped]
import pytest

from ncaa_eval.model.base import Model, ModelConfig
from ncaa_eval.model.registry import (
    _MODEL_REGISTRY,
    ModelNotFoundError,
    get_model,
    list_models,
    register_model,
)


@pytest.fixture(autouse=True)
def _clean_registry() -> Generator[None, None, None]:
    """Snapshot and restore the global registry around each test."""
    snapshot = dict(_MODEL_REGISTRY)
    yield
    _MODEL_REGISTRY.clear()
    _MODEL_REGISTRY.update(snapshot)


def _make_dummy_model(name: str = "dummy") -> type[Model]:
    """Create and return a minimal Model subclass (not registered)."""

    class _Dummy(Model):
        def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
            pass

        def predict_proba(self, X: pd.DataFrame) -> pd.Series:
            return pd.Series(dtype=float)

        def save(self, path: Path) -> None:
            pass

        @classmethod
        def load(cls, path: Path) -> Self:
            return cls()

        def get_config(self) -> ModelConfig:
            return ModelConfig(model_name=name)

    _Dummy.__name__ = f"Dummy_{name}"
    _Dummy.__qualname__ = f"Dummy_{name}"
    return _Dummy


class TestRegisterModel:
    def test_decorator_registers(self) -> None:
        cls = _make_dummy_model("test_reg")
        decorated = register_model("test_reg")(cls)
        assert decorated is cls
        assert get_model("test_reg") is cls

    def test_duplicate_name_raises(self) -> None:
        cls1 = _make_dummy_model("dup")
        register_model("dup")(cls1)
        cls2 = _make_dummy_model("dup2")
        with pytest.raises(ValueError, match="already registered"):
            register_model("dup")(cls2)


class TestGetModel:
    def test_returns_registered_class(self) -> None:
        cls = _make_dummy_model("lookup")
        register_model("lookup")(cls)
        assert get_model("lookup") is cls

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ModelNotFoundError, match="no_such_model"):
            get_model("no_such_model")


class TestListModels:
    def test_returns_sorted_names(self) -> None:
        register_model("beta")(_make_dummy_model("b"))
        register_model("alpha")(_make_dummy_model("a"))
        names = list_models()
        assert "alpha" in names
        assert "beta" in names
        assert names.index("alpha") < names.index("beta")
