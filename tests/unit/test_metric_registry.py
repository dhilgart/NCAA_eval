"""Unit tests for the metric plugin registry."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Self
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]
import pytest
from rich.console import Console

from ncaa_eval.evaluation.metrics import (
    _METRIC_REGISTRY,
    MetricNotFoundError,
    get_metric,
    list_metrics,
    log_loss,
    register_metric,
)


@pytest.fixture(autouse=True)
def _clean_registry() -> Generator[None, None, None]:
    """Snapshot and restore the global metric registry around each test."""
    snapshot = dict(_METRIC_REGISTRY)
    yield
    _METRIC_REGISTRY.clear()
    _METRIC_REGISTRY.update(snapshot)


# ---------------------------------------------------------------------------
# Built-in registration (AC #5)
# ---------------------------------------------------------------------------


class TestBuiltInMetrics:
    def test_list_metrics_returns_four_builtins(self) -> None:
        names = list_metrics()
        assert set(names) == {"brier_score", "ece", "log_loss", "roc_auc"}

    def test_list_metrics_is_sorted(self) -> None:
        names = list_metrics()
        assert names == sorted(names)

    def test_get_metric_log_loss_returns_function(self) -> None:
        fn = get_metric("log_loss")
        assert fn is log_loss


# ---------------------------------------------------------------------------
# get_metric (AC #8)
# ---------------------------------------------------------------------------


class TestGetMetric:
    def test_nonexistent_raises_metric_not_found(self) -> None:
        with pytest.raises(MetricNotFoundError, match="nonexistent"):
            get_metric("nonexistent")

    def test_error_lists_available_metrics(self) -> None:
        with pytest.raises(MetricNotFoundError, match="log_loss"):
            get_metric("nonexistent")


# ---------------------------------------------------------------------------
# register_metric (AC #7)
# ---------------------------------------------------------------------------


class TestRegisterMetric:
    def test_decorator_registers_and_returns_function(self) -> None:
        @register_metric("custom_test")
        def my_metric(
            y_true: npt.NDArray[np.float64],
            y_prob: npt.NDArray[np.float64],
        ) -> float:
            return 0.0

        assert get_metric("custom_test") is my_metric
        assert "custom_test" in list_metrics()

    def test_duplicate_registration_raises_value_error(self) -> None:
        @register_metric("dup_test")
        def first(
            y_true: npt.NDArray[np.float64],
            y_prob: npt.NDArray[np.float64],
        ) -> float:
            return 0.0

        with pytest.raises(ValueError, match="already registered"):

            @register_metric("dup_test")
            def second(
                y_true: npt.NDArray[np.float64],
                y_prob: npt.NDArray[np.float64],
            ) -> float:
                return 1.0

    def test_duplicate_builtin_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="already registered"):

            @register_metric("log_loss")
            def fake_log_loss(
                y_true: npt.NDArray[np.float64],
                y_prob: npt.NDArray[np.float64],
            ) -> float:
                return 0.0


# ---------------------------------------------------------------------------
# Backtest integration (AC #1, #6)
# ---------------------------------------------------------------------------


class TestBacktestIntegration:
    def test_default_metrics_includes_custom(self) -> None:
        """Registering a custom metric before calling default_metrics() includes it."""
        from ncaa_eval.evaluation.backtest import default_metrics

        @register_metric("custom_bt")
        def custom_bt(
            y_true: npt.NDArray[np.float64],
            y_prob: npt.NDArray[np.float64],
        ) -> float:
            return 42.0

        dm = default_metrics()
        assert "custom_bt" in dm
        assert dm["custom_bt"] is custom_bt

    def test_default_metrics_includes_builtins(self) -> None:
        from ncaa_eval.evaluation.backtest import default_metrics

        dm = default_metrics()
        assert set(dm.keys()) >= {"brier_score", "ece", "log_loss", "roc_auc"}

    def test_explicit_metric_fns_still_works(self) -> None:
        """run_backtest with explicit metric_fns= dict uses those metrics (backward compat)."""
        from ncaa_eval.evaluation.backtest import DEFAULT_METRICS

        # DEFAULT_METRICS constant still exists and contains the 4 built-ins
        assert set(DEFAULT_METRICS.keys()) == {"log_loss", "brier_score", "roc_auc", "ece"}

    def test_run_backtest_metric_fns_none_uses_registry(self) -> None:
        """run_backtest(metric_fns=None) picks up custom registered metrics (AC #6)."""
        from ncaa_eval.evaluation.backtest import run_backtest
        from ncaa_eval.model.base import Model, ModelConfig

        class _FakeModel(Model):
            def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
                pass

            def predict_proba(self, X: pd.DataFrame) -> pd.Series:
                return pd.Series(0.5, index=X.index)

            def save(self, path: Path) -> None:
                pass

            @classmethod
            def load(cls, path: Path) -> Self:
                return cls()

            def get_config(self) -> ModelConfig:
                return ModelConfig(model_name="fake")

        @register_metric("registry_test_metric")
        def _registry_metric(
            y_true: npt.NDArray[np.float64],
            y_prob: npt.NDArray[np.float64],
        ) -> float:
            return 99.0

        seasons = [2010, 2011, 2012]
        n = 20
        rng = np.random.default_rng(0)
        base_df = pd.DataFrame(
            {
                "game_id": [f"g{i}" for i in range(n)],
                "season": 2010,
                "day_num": list(range(n)),
                "date": pd.date_range("2010-01-01", periods=n, freq="D"),
                "team_a_id": rng.integers(1000, 2000, size=n),
                "team_b_id": rng.integers(2000, 3000, size=n),
                "is_tournament": [False] * 17 + [True] * 3,
                "loc_encoding": rng.choice([1, -1, 0], size=n).astype(float),
                "team_a_won": rng.choice([True, False], size=n),
                "elo_diff": rng.normal(0, 50, size=n),
            }
        )
        server = MagicMock()
        server.serve_season_features.side_effect = lambda year, mode="batch": base_df.assign(season=year)

        result = run_backtest(
            _FakeModel(),
            server,
            seasons=seasons,
            n_jobs=1,
            metric_fns=None,
            console=Console(quiet=True),
        )

        assert "registry_test_metric" in result.summary.columns

    def test_run_backtest_explicit_metric_fns_used(self) -> None:
        """run_backtest(metric_fns={...}) uses only the provided metrics (AC #6 backward compat)."""
        from ncaa_eval.evaluation.backtest import run_backtest
        from ncaa_eval.model.base import Model, ModelConfig

        class _FakeModel(Model):
            def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
                pass

            def predict_proba(self, X: pd.DataFrame) -> pd.Series:
                return pd.Series(0.5, index=X.index)

            def save(self, path: Path) -> None:
                pass

            @classmethod
            def load(cls, path: Path) -> Self:
                return cls()

            def get_config(self) -> ModelConfig:
                return ModelConfig(model_name="fake")

        def _const(
            y_true: npt.NDArray[np.float64],
            y_prob: npt.NDArray[np.float64],
        ) -> float:
            return 0.42

        seasons = [2010, 2011, 2012]
        n = 20
        rng = np.random.default_rng(1)
        base_df = pd.DataFrame(
            {
                "game_id": [f"g{i}" for i in range(n)],
                "season": 2010,
                "day_num": list(range(n)),
                "date": pd.date_range("2010-01-01", periods=n, freq="D"),
                "team_a_id": rng.integers(1000, 2000, size=n),
                "team_b_id": rng.integers(2000, 3000, size=n),
                "is_tournament": [False] * 17 + [True] * 3,
                "loc_encoding": rng.choice([1, -1, 0], size=n).astype(float),
                "team_a_won": rng.choice([True, False], size=n),
                "elo_diff": rng.normal(0, 50, size=n),
            }
        )
        server = MagicMock()
        server.serve_season_features.side_effect = lambda year, mode="batch": base_df.assign(season=year)

        result = run_backtest(
            _FakeModel(),
            server,
            seasons=seasons,
            n_jobs=1,
            metric_fns={"const": _const},
            console=Console(quiet=True),
        )

        assert list(result.summary.columns) == ["const", "elapsed_seconds"]
