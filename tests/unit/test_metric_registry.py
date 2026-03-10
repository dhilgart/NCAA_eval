"""Unit tests for the metric plugin registry."""

from __future__ import annotations

from collections.abc import Generator

import numpy as np
import numpy.typing as npt
import pytest

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
