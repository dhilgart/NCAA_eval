"""Unit tests for the parallel cross-validation backtest orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from rich.console import Console

from ncaa_eval.evaluation.backtest import (
    DEFAULT_METRICS,
    FoldResult,
    _evaluate_fold,
    feature_cols as _feature_cols,
    run_backtest,
)
from ncaa_eval.evaluation.splitter import CVFold
from ncaa_eval.ingest.schema import Game
from ncaa_eval.model.base import Model, ModelConfig, StatefulModel

# ── Test helpers ─────────────────────────────────────────────────────────────


def _make_season_df(
    year: int,
    n_regular: int = 10,
    n_tournament: int = 3,
    *,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Create a minimal synthetic season DataFrame for testing."""
    if rng is None:
        rng = np.random.default_rng(seed=year)

    total = n_regular + n_tournament
    is_tournament = [False] * n_regular + [True] * n_tournament
    # Include synthetic feature columns (not in METADATA_COLS) so _feature_cols()
    # returns a non-empty list, exercising the stateless column-filtering code path.
    return pd.DataFrame(
        {
            "game_id": [f"{year}_{i}" for i in range(total)],
            "season": year,
            "day_num": list(range(total)),
            "date": pd.date_range(f"{year}-01-01", periods=total, freq="D"),
            "team_a_id": rng.integers(1000, 2000, size=total),
            "team_b_id": rng.integers(1000, 2000, size=total),
            "is_tournament": is_tournament,
            "loc_encoding": rng.choice([1, -1, 0], size=total),
            "team_a_won": rng.choice([True, False], size=total),
            # Synthetic features — used to verify stateless models receive only
            # non-metadata columns and _DataDependentModel has real values to use.
            "elo_diff": rng.normal(0.0, 50.0, size=total),
            "win_pct_diff": rng.uniform(-0.5, 0.5, size=total),
        }
    )


def _make_feature_server(
    season_dfs: dict[int, pd.DataFrame],
) -> MagicMock:
    """Build a mock StatefulFeatureServer returning pre-built DataFrames."""
    mock = MagicMock()
    mock.serve_season_features.side_effect = lambda year, mode="batch": season_dfs.get(year, pd.DataFrame())
    return mock


def _make_fold(
    year: int,
    n_train: int = 20,
    n_test: int = 5,
    *,
    rng: np.random.Generator | None = None,
) -> CVFold:
    """Create a CVFold with synthetic data."""
    if rng is None:
        rng = np.random.default_rng(seed=year)

    train_df = _make_season_df(year - 1, n_regular=n_train, n_tournament=0, rng=rng)
    test_df = _make_season_df(year, n_regular=0, n_tournament=n_test, rng=rng)
    return CVFold(train=train_df, test=test_df, year=year)


class _FakeStatelessModel(Model):
    """Minimal stateless model for testing — always predicts 0.5."""

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
        return ModelConfig(model_name="fake_stateless")


class _FakeStatefulModel(StatefulModel):
    """Minimal stateful model for testing — tracks state mutations."""

    def __init__(self) -> None:
        self._ratings: dict[int, float] = {}
        self._fit_count: int = 0

    def _predict_one(self, team_a_id: int, team_b_id: int) -> float:
        return 0.6

    def update(self, game: Game) -> None:
        self._ratings[game.w_team_id] = self._ratings.get(game.w_team_id, 1500.0) + 10.0
        self._ratings[game.l_team_id] = self._ratings.get(game.l_team_id, 1500.0) - 10.0

    def start_season(self, season: int) -> None:
        pass

    def get_state(self) -> dict[str, Any]:
        return {"ratings": dict(self._ratings)}

    def set_state(self, state: dict[str, Any]) -> None:
        self._ratings = dict(state["ratings"])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        super().fit(X, y)
        self._fit_count += 1

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        return super().predict_proba(X)

    def save(self, path: Path) -> None:
        pass

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls()

    def get_config(self) -> ModelConfig:
        return ModelConfig(model_name="fake_stateful")


def _constant_metric(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """A trivial metric that always returns 0.42."""
    return 0.42


# ── TestEvaluateFold ────────────────────────────────────────────────────────


class TestEvaluateFold:
    """Tests for _evaluate_fold worker function."""

    def test_stateless_model_known_predictions(self) -> None:
        """3.1: _evaluate_fold with a mock model producing known predictions."""
        fold = _make_fold(2012, n_train=15, n_test=5)
        model = _FakeStatelessModel()
        metrics_fn = {"const": _constant_metric}

        result = _evaluate_fold(fold, model, metrics_fn)

        assert result.year == 2012
        assert len(result.predictions) == 5
        assert len(result.actuals) == 5
        assert result.metrics["const"] == pytest.approx(0.42)
        assert result.elapsed_seconds >= 0.0

    def test_empty_test_fold(self) -> None:
        """3.7: Empty test fold returns NaN metrics."""
        train_df = _make_season_df(2011)
        test_df = pd.DataFrame(columns=train_df.columns).iloc[0:0]
        fold = CVFold(train=train_df, test=test_df, year=2012)
        model = _FakeStatelessModel()

        result = _evaluate_fold(fold, model, DEFAULT_METRICS)

        assert result.year == 2012
        assert len(result.predictions) == 0
        assert len(result.actuals) == 0
        for val in result.metrics.values():
            assert np.isnan(val)

    def test_stateful_model_receives_full_df(self) -> None:
        """Stateful models receive the full DataFrame (metadata + features)."""
        fold = _make_fold(2012)
        model = _FakeStatefulModel()
        metrics_fn = {"const": _constant_metric}

        result = _evaluate_fold(fold, model, metrics_fn)

        assert result.year == 2012
        assert len(result.predictions) > 0

    def test_stateless_model_receives_only_feature_cols(self) -> None:
        """3.5: Stateless models only get feature columns (no metadata)."""
        fold = _make_fold(2012)
        received_cols: list[str] = []

        class _SpyModel(_FakeStatelessModel):
            def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
                received_cols.extend(X.columns.tolist())

        model = _SpyModel()
        _evaluate_fold(fold, model, {"const": _constant_metric})

        from ncaa_eval.evaluation.backtest import METADATA_COLS

        for col in received_cols:
            assert col not in METADATA_COLS

    def test_metric_exception_produces_nan(self) -> None:
        """Per-metric exceptions produce NaN, not crash."""
        fold = _make_fold(2012)
        model = _FakeStatelessModel()

        def _raising_metric(
            y_true: np.ndarray,
            y_prob: np.ndarray,
        ) -> float:
            msg = "boom"
            raise ValueError(msg)

        result = _evaluate_fold(fold, model, {"good": _constant_metric, "bad": _raising_metric})

        assert result.metrics["good"] == pytest.approx(0.42)
        assert np.isnan(result.metrics["bad"])

    def test_fold_result_frozen(self) -> None:
        """FoldResult is a frozen dataclass."""
        fold = _make_fold(2012)
        model = _FakeStatelessModel()
        result = _evaluate_fold(fold, model, {"const": _constant_metric})
        with pytest.raises(AttributeError):
            result.year = 2099  # type: ignore[misc]


# ── TestRunBacktest ─────────────────────────────────────────────────────────


class TestRunBacktest:
    """Tests for run_backtest orchestrator."""

    def test_sequential_produces_correct_fold_count(self) -> None:
        """3.2: Sequential run produces correct fold count and metrics."""
        seasons = [2010, 2011, 2012, 2013]
        dfs = {y: _make_season_df(y) for y in seasons}
        server = _make_feature_server(dfs)
        model = _FakeStatelessModel()
        console = Console(quiet=True)

        result = run_backtest(
            model,
            server,
            seasons=seasons,
            n_jobs=1,
            metric_fns={"const": _constant_metric},
            console=console,
        )

        # 4 seasons → 3 folds (2011, 2012, 2013)
        assert len(result.fold_results) == 3
        assert isinstance(result.summary, pd.DataFrame)
        assert result.elapsed_seconds > 0.0
        assert list(result.summary.index) == [2011, 2012, 2013]

    def test_parallel_produces_correct_fold_count(self) -> None:
        """3.3 prereq: Parallel run also produces correct fold count."""
        seasons = [2010, 2011, 2012]
        dfs = {y: _make_season_df(y) for y in seasons}
        server = _make_feature_server(dfs)
        model = _FakeStatelessModel()
        console = Console(quiet=True)

        result = run_backtest(
            model,
            server,
            seasons=seasons,
            n_jobs=2,
            metric_fns={"const": _constant_metric},
            console=console,
        )

        # 3 seasons → 2 folds (2011, 2012)
        assert len(result.fold_results) == 2
        assert result.fold_results[0].year == 2011
        assert result.fold_results[1].year == 2012

    def test_stateful_model_deep_copied_per_fold(self) -> None:
        """3.4: Stateful model gets deep-copied; original unchanged."""
        seasons = [2010, 2011, 2012]
        dfs = {y: _make_season_df(y) for y in seasons}
        server = _make_feature_server(dfs)
        model = _FakeStatefulModel()
        original_ratings = dict(model._ratings)
        console = Console(quiet=True)

        run_backtest(
            model,
            server,
            seasons=seasons,
            mode="stateful",
            n_jobs=1,
            metric_fns={"const": _constant_metric},
            console=console,
        )

        # Original model state should be unchanged
        assert model._ratings == original_ratings
        assert model._fit_count == 0

    def test_progress_reporting(self) -> None:
        """3.6: Progress output is printed to console."""
        seasons = [2010, 2011, 2012]
        dfs = {y: _make_season_df(y) for y in seasons}
        server = _make_feature_server(dfs)
        model = _FakeStatelessModel()
        console = Console(file=MagicMock(), quiet=False)

        run_backtest(
            model,
            server,
            seasons=seasons,
            n_jobs=1,
            metric_fns={"const": _constant_metric},
            console=console,
        )

        # Console.print was called (fold count message + table + total time)
        assert console.file.write.call_count > 0  # type: ignore[attr-defined]

    def test_n_jobs_passed_to_joblib(self) -> None:
        """3.8: n_jobs parameter is passed through to joblib.Parallel."""
        seasons = [2010, 2011, 2012]
        dfs = {y: _make_season_df(y) for y in seasons}
        server = _make_feature_server(dfs)
        model = _FakeStatelessModel()
        console = Console(quiet=True)

        with patch("ncaa_eval.evaluation.backtest.joblib.Parallel") as mock_parallel:
            # Make the mock behave like real Parallel — return callable
            mock_instance = MagicMock()
            mock_instance.return_value = [
                FoldResult(
                    year=2011,
                    predictions=pd.Series(dtype=np.float64),
                    actuals=pd.Series(dtype=np.float64),
                    metrics={"const": 0.42},
                    elapsed_seconds=0.1,
                )
            ]
            mock_parallel.return_value = mock_instance

            run_backtest(
                model,
                server,
                seasons=seasons,
                n_jobs=4,
                metric_fns={"const": _constant_metric},
                console=console,
            )

            mock_parallel.assert_called_once_with(n_jobs=4)

    def test_summary_dataframe_structure(self) -> None:
        """3.9: Summary DataFrame has correct columns, index, and sorting."""
        seasons = [2010, 2011, 2012, 2013]
        dfs = {y: _make_season_df(y) for y in seasons}
        server = _make_feature_server(dfs)
        model = _FakeStatelessModel()
        console = Console(quiet=True)

        result = run_backtest(
            model,
            server,
            seasons=seasons,
            n_jobs=1,
            metric_fns={"const": _constant_metric},
            console=console,
        )

        assert result.summary.index.name == "year"
        assert list(result.summary.index) == sorted(result.summary.index)
        assert "const" in result.summary.columns
        assert "elapsed_seconds" in result.summary.columns

    def test_single_fold_two_seasons(self) -> None:
        """3.10: Two seasons minimum works correctly (single fold)."""
        seasons = [2010, 2011]
        dfs = {y: _make_season_df(y) for y in seasons}
        server = _make_feature_server(dfs)
        model = _FakeStatelessModel()
        console = Console(quiet=True)

        result = run_backtest(
            model,
            server,
            seasons=seasons,
            n_jobs=1,
            metric_fns={"const": _constant_metric},
            console=console,
        )

        assert len(result.fold_results) == 1
        assert result.fold_results[0].year == 2011

    def test_default_metric_fns(self) -> None:
        """3.11: Default metric_fns includes log_loss, brier_score, roc_auc, ece."""
        seasons = [2010, 2011, 2012]
        dfs = {y: _make_season_df(y) for y in seasons}
        server = _make_feature_server(dfs)
        model = _FakeStatelessModel()
        console = Console(quiet=True)

        result = run_backtest(
            model,
            server,
            seasons=seasons,
            n_jobs=1,
            console=console,
        )

        expected_metrics = {"log_loss", "brier_score", "roc_auc", "ece"}
        for fold_result in result.fold_results:
            assert set(fold_result.metrics.keys()) == expected_metrics
            # log_loss and brier_score are always finite for constant 0.5 predictions.
            assert not np.isnan(fold_result.metrics["log_loss"]), "log_loss must be finite"
            assert not np.isnan(fold_result.metrics["brier_score"]), "brier_score must be finite"
            # Constant 0.5 predictions → known values regardless of label balance.
            assert fold_result.metrics["log_loss"] == pytest.approx(0.693, abs=0.1)
            assert fold_result.metrics["brier_score"] == pytest.approx(0.25, abs=0.05)
            # roc_auc may be NaN if test fold has single-class labels (small fold, random seed).
            roc = fold_result.metrics["roc_auc"]
            if not np.isnan(roc):
                assert 0.0 <= roc <= 1.0
            # ece is always finite for constant predictions.
            assert not np.isnan(fold_result.metrics["ece"]), "ece must be finite"
            assert 0.0 <= fold_result.metrics["ece"] <= 1.0

    def test_invalid_mode_raises(self) -> None:
        """run_backtest validates mode at entry point."""
        server = _make_feature_server({})
        model = _FakeStatelessModel()
        with pytest.raises(ValueError, match="mode must be"):
            run_backtest(
                model,
                server,
                seasons=[2010, 2011],
                mode="invalid",
            )

    def test_too_few_seasons_raises(self) -> None:
        """run_backtest propagates ValueError from walk_forward_splits for < 2 seasons."""
        server = _make_feature_server({2010: _make_season_df(2010)})
        model = _FakeStatelessModel()
        with pytest.raises(ValueError, match="at least 2 seasons"):
            run_backtest(model, server, seasons=[2010])

    def test_backtest_result_frozen(self) -> None:
        """BacktestResult is a frozen dataclass."""
        seasons = [2010, 2011]
        dfs = {y: _make_season_df(y) for y in seasons}
        server = _make_feature_server(dfs)
        model = _FakeStatelessModel()
        console = Console(quiet=True)

        result = run_backtest(
            model,
            server,
            seasons=seasons,
            n_jobs=1,
            metric_fns={"const": _constant_metric},
            console=console,
        )

        with pytest.raises(AttributeError):
            result.elapsed_seconds = 999.0  # type: ignore[misc]


# ── TestDeterminism ─────────────────────────────────────────────────────────


class _DataDependentModel(_FakeStatelessModel):
    """Stateless model whose predictions vary with input data.

    Returns the mean of the first feature column as the predicted probability
    (clipped to [0.01, 0.99]).  This ensures predictions are data-dependent
    rather than constant 0.5, making the determinism test more meaningful.
    """

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        feat = _feature_cols(X)
        if not feat:
            return pd.Series(0.5, index=X.index)
        col_mean = X[feat[0]].mean()
        prob = float(np.clip(col_mean / (col_mean + 1.0), 0.01, 0.99))
        return pd.Series(prob, index=X.index)


class TestDeterminism:
    """Tests for parallel vs sequential determinism (AC #6, #7)."""

    def test_parallel_matches_sequential(self) -> None:
        """3.3: Parallel n_jobs=2 produces identical results to sequential n_jobs=1.

        NOTE: Uses _FakeStatelessModel (constant predictions).  The
        test_parallel_matches_sequential_data_dependent test below uses a
        data-dependent model to provide stronger determinism coverage.
        """
        seasons = [2010, 2011, 2012, 2013]
        dfs = {y: _make_season_df(y) for y in seasons}

        server_seq = _make_feature_server(dfs)
        server_par = _make_feature_server(dfs)
        console = Console(quiet=True)

        result_seq = run_backtest(
            _FakeStatelessModel(),
            server_seq,
            seasons=seasons,
            n_jobs=1,
            metric_fns={"const": _constant_metric},
            console=console,
        )

        result_par = run_backtest(
            _FakeStatelessModel(),
            server_par,
            seasons=seasons,
            n_jobs=2,
            metric_fns={"const": _constant_metric},
            console=console,
        )

        assert len(result_seq.fold_results) == len(result_par.fold_results)

        for fr_seq, fr_par in zip(result_seq.fold_results, result_par.fold_results):
            assert fr_seq.year == fr_par.year
            assert fr_seq.metrics == pytest.approx(fr_par.metrics)
            pd.testing.assert_series_equal(fr_seq.predictions, fr_par.predictions)
            pd.testing.assert_series_equal(fr_seq.actuals, fr_par.actuals)

        # Summary DataFrames should match (ignoring elapsed_seconds timing)
        metric_cols = [c for c in result_seq.summary.columns if c != "elapsed_seconds"]
        pd.testing.assert_frame_equal(
            result_seq.summary[metric_cols],
            result_par.summary[metric_cols],
        )

    def test_parallel_matches_sequential_data_dependent(self) -> None:
        """AC6/AC7: Determinism with data-dependent model (stronger coverage).

        Uses _DataDependentModel whose predictions vary with input feature values,
        ensuring the parallel orchestration doesn't introduce result ordering bugs
        that would be masked by constant-prediction models.
        """
        seasons = [2010, 2011, 2012, 2013]
        dfs = {y: _make_season_df(y) for y in seasons}

        server_seq = _make_feature_server(dfs)
        server_par = _make_feature_server(dfs)
        console = Console(quiet=True)

        result_seq = run_backtest(
            _DataDependentModel(),
            server_seq,
            seasons=seasons,
            n_jobs=1,
            metric_fns={"const": _constant_metric},
            console=console,
        )

        result_par = run_backtest(
            _DataDependentModel(),
            server_par,
            seasons=seasons,
            n_jobs=2,
            metric_fns={"const": _constant_metric},
            console=console,
        )

        assert len(result_seq.fold_results) == len(result_par.fold_results)

        for fr_seq, fr_par in zip(result_seq.fold_results, result_par.fold_results):
            assert fr_seq.year == fr_par.year
            assert fr_seq.metrics == pytest.approx(fr_par.metrics, rel=1e-9)
            pd.testing.assert_series_equal(fr_seq.predictions, fr_par.predictions)
            pd.testing.assert_series_equal(fr_seq.actuals, fr_par.actuals)


# ── TestEdgeCases ───────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge case tests for backtest functionality."""

    def test_empty_test_fold_in_backtest(self) -> None:
        """Fold with 0 tournament games is handled gracefully."""
        seasons = [2010, 2011]
        dfs = {
            2010: _make_season_df(2010),
            2011: _make_season_df(2011, n_tournament=0),
        }
        server = _make_feature_server(dfs)
        model = _FakeStatelessModel()
        console = Console(quiet=True)

        result = run_backtest(
            model,
            server,
            seasons=seasons,
            n_jobs=1,
            metric_fns={"const": _constant_metric},
            console=console,
        )

        # 2011 has 0 tournament games → fold exists with NaN metrics
        assert len(result.fold_results) == 1
        assert result.fold_results[0].year == 2011
        for val in result.fold_results[0].metrics.values():
            assert np.isnan(val)

    def test_feature_cols_excludes_metadata(self) -> None:
        """_feature_cols returns only non-metadata columns."""
        df = pd.DataFrame(
            {
                "game_id": [1],
                "season": [2020],
                "custom_feature": [0.5],
                "another_feature": [1.0],
            }
        )
        result = _feature_cols(df)
        assert "custom_feature" in result
        assert "another_feature" in result
        assert "game_id" not in result
        assert "season" not in result

    def test_fold_results_sorted_by_year(self) -> None:
        """Fold results are always sorted ascending by year."""
        seasons = [2010, 2011, 2012, 2013, 2014]
        dfs = {y: _make_season_df(y) for y in seasons}
        server = _make_feature_server(dfs)
        model = _FakeStatelessModel()
        console = Console(quiet=True)

        result = run_backtest(
            model,
            server,
            seasons=seasons,
            n_jobs=2,
            metric_fns={"const": _constant_metric},
            console=console,
        )

        years = [fr.year for fr in result.fold_results]
        assert years == sorted(years)


# ── TestPerformance ──────────────────────────────────────────────────────────


class TestPerformance:
    """Performance tests for AC5: 10-year Elo backtest must complete in < 60s.

    These tests require the full data pipeline (real EloModel + real game data)
    and are skipped in the standard unit test run.  Run manually or in a
    dedicated integration test suite once the data pipeline (Epic 4) is
    available end-to-end.

    To run: pytest tests/unit/test_evaluation_backtest.py::TestPerformance -v -s
    """

    @pytest.mark.skip(
        reason=(
            "AC5 integration test requires real EloModel + real game data from "
            "Epic 4 data pipeline.  Cannot be run in unit test context without "
            "full data infrastructure.  Verify manually: "
            "run_backtest(EloModel(), real_feature_server, seasons=range(2015, 2025), "
            "n_jobs=-1) must complete in < 60 seconds."
        )
    )
    def test_elo_10year_backtest_under_60_seconds(self) -> None:
        """AC5: 10-year Elo backtest completes in under 60 seconds (PRD perf target).

        Implementation notes:
        - Use real EloModel (or equivalent) with actual NCAA game data
        - Feature server must serve real feature matrices for seasons 2015–2024
        - n_jobs=-1 to use all available cores
        - Assert result.elapsed_seconds < 60.0
        """
        import time  # noqa: PLC0415

        # Placeholder body — this test is always skipped.
        # Real implementation:
        #   from ncaa_eval.model.elo import EloModel
        #   from ncaa_eval.transform.feature_serving import StatefulFeatureServer, FeatureConfig
        #   ...
        #   result = run_backtest(EloModel(), server, seasons=list(range(2015, 2025)), n_jobs=-1)
        #   assert result.elapsed_seconds < 60.0, f"Backtest took {result.elapsed_seconds:.1f}s > 60s"
        start = time.perf_counter()
        assert time.perf_counter() - start < 60.0  # pragma: no cover
