"""Unit tests for Plotly visualization adapters (evaluation/plotting.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
import pytest

from ncaa_eval.evaluation.backtest import BacktestResult, FoldResult
from ncaa_eval.evaluation.plotting import (
    COLOR_GREEN,
    COLOR_RED,
    TEMPLATE,
    plot_advancement_heatmap,
    plot_backtest_summary,
    plot_metric_comparison,
    plot_reliability_diagram,
    plot_score_distribution,
)
from ncaa_eval.evaluation.simulation import (
    BracketDistribution,
    SimulationResult,
    compute_bracket_distribution,
)

# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_backtest_result(
    years: list[int] | None = None,
    metric_names: list[str] | None = None,
) -> BacktestResult:
    """Build a synthetic BacktestResult for testing."""
    if years is None:
        years = [2020, 2021, 2022]
    if metric_names is None:
        metric_names = ["log_loss", "brier_score"]

    rng = np.random.default_rng(42)
    folds: list[FoldResult] = []
    rows: list[dict[str, object]] = []

    for y in years:
        metrics = {m: float(rng.uniform(0.1, 0.9)) for m in metric_names}
        folds.append(
            FoldResult(
                year=y,
                predictions=pd.Series(rng.uniform(0.0, 1.0, size=10)),
                actuals=pd.Series(rng.choice([0, 1], size=10).astype(np.float64)),
                metrics=metrics,
                elapsed_seconds=float(rng.uniform(0.1, 1.0)),
            )
        )
        row: dict[str, object] = {"year": y}
        row.update(metrics)
        row["elapsed_seconds"] = folds[-1].elapsed_seconds
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("year")
    return BacktestResult(
        fold_results=tuple(folds),
        summary=summary,
        elapsed_seconds=sum(f.elapsed_seconds for f in folds),
    )


def _make_simulation_result(
    n_teams: int = 8,
    n_rounds: int = 3,
    n_simulations: int = 1000,
) -> SimulationResult:
    """Build a synthetic SimulationResult for testing."""
    rng = np.random.default_rng(123)
    adv_probs = rng.dirichlet(np.ones(n_teams), size=n_rounds).T
    # Normalize each column to sum to expected number of survivors
    for r in range(n_rounds):
        col_sum = adv_probs[:, r].sum()
        if col_sum > 0:
            adv_probs[:, r] /= col_sum

    return SimulationResult(
        season=2024,
        advancement_probs=adv_probs,
        expected_points={"standard": rng.uniform(1.0, 10.0, size=n_teams)},
        method="monte_carlo",
        n_simulations=n_simulations,
        confidence_intervals=None,
        score_distribution=None,
    )


def _make_bracket_distribution() -> BracketDistribution:
    """Build a synthetic BracketDistribution for testing."""
    rng = np.random.default_rng(99)
    scores = rng.normal(100.0, 20.0, size=1000)
    return compute_bracket_distribution(scores.astype(np.float64), n_bins=30)


# ── TestPlotReliabilityDiagram ──────────────────────────────────────────────


class TestPlotReliabilityDiagram:
    """Tests for plot_reliability_diagram."""

    def test_returns_figure(self) -> None:
        """plot_reliability_diagram returns a go.Figure."""
        rng = np.random.default_rng(1)
        y_true = rng.choice([0.0, 1.0], size=200)
        y_prob = np.clip(y_true + rng.normal(0, 0.2, size=200), 0.0, 1.0)

        fig = plot_reliability_diagram(y_true, y_prob)

        assert isinstance(fig, go.Figure)

    def test_has_calibration_and_reference_traces(self) -> None:
        """Figure has calibration scatter, diagonal reference, and bar traces."""
        rng = np.random.default_rng(2)
        y_true = rng.choice([0.0, 1.0], size=200)
        y_prob = np.clip(y_true + rng.normal(0, 0.2, size=200), 0.0, 1.0)

        fig = plot_reliability_diagram(y_true, y_prob)

        # At least 3 traces: bar (bin counts), diagonal, calibration scatter
        assert len(fig.data) >= 3

        # Check diagonal reference line exists
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        diag_found = any(
            t.line is not None and getattr(t.line, "dash", None) == "dash" for t in scatter_traces
        )
        assert diag_found, "Diagonal reference line not found"

    def test_uses_color_palette(self) -> None:
        """Calibration trace uses COLOR_GREEN."""
        rng = np.random.default_rng(3)
        y_true = rng.choice([0.0, 1.0], size=200)
        y_prob = np.clip(y_true + rng.normal(0, 0.2, size=200), 0.0, 1.0)

        fig = plot_reliability_diagram(y_true, y_prob)

        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        cal_trace = [t for t in scatter_traces if t.name == "Calibration"]
        assert len(cal_trace) == 1
        assert cal_trace[0].marker.color == COLOR_GREEN

    def test_bin_counts_in_bar_trace(self) -> None:
        """Bar trace contains bin counts as y values."""
        rng = np.random.default_rng(4)
        y_true = rng.choice([0.0, 1.0], size=200)
        y_prob = np.clip(y_true + rng.normal(0, 0.2, size=200), 0.0, 1.0)

        fig = plot_reliability_diagram(y_true, y_prob, n_bins=5)

        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bar_traces) == 1
        # Bar y values should be positive integers (bin counts)
        counts = bar_traces[0].y
        assert all(c >= 0 for c in counts)
        assert sum(counts) == 200  # total samples

    def test_custom_title(self) -> None:
        """Custom title is applied."""
        rng = np.random.default_rng(5)
        y_true = rng.choice([0.0, 1.0], size=50)
        y_prob = np.clip(y_true + rng.normal(0, 0.2, size=50), 0.0, 1.0)

        fig = plot_reliability_diagram(y_true, y_prob, title="My Title")

        assert fig.layout.title.text == "My Title"

    def test_dark_template_applied(self) -> None:
        """plotly_dark template is applied."""
        import plotly.io as pio  # type: ignore[import-untyped]

        rng = np.random.default_rng(6)
        y_true = rng.choice([0.0, 1.0], size=50)
        y_prob = np.clip(y_true + rng.normal(0, 0.2, size=50), 0.0, 1.0)

        fig = plot_reliability_diagram(y_true, y_prob)

        assert fig.layout.template == pio.templates[TEMPLATE]

    def test_single_point(self) -> None:
        """Single-point input doesn't crash (edge case)."""
        y_true = np.array([1.0])
        y_prob = np.array([0.8])

        fig = plot_reliability_diagram(y_true, y_prob, n_bins=1)

        assert isinstance(fig, go.Figure)


# ── TestPlotBacktestSummary ─────────────────────────────────────────────────


class TestPlotBacktestSummary:
    """Tests for plot_backtest_summary."""

    def test_returns_figure(self) -> None:
        """plot_backtest_summary returns a go.Figure."""
        bt = _make_backtest_result()
        fig = plot_backtest_summary(bt)
        assert isinstance(fig, go.Figure)

    def test_correct_number_of_traces(self) -> None:
        """One trace per metric (excluding elapsed_seconds)."""
        bt = _make_backtest_result(metric_names=["log_loss", "brier_score", "ece"])
        fig = plot_backtest_summary(bt)
        assert len(fig.data) == 3

    def test_filtered_metrics(self) -> None:
        """When metrics parameter is specified, only those are plotted."""
        bt = _make_backtest_result(metric_names=["log_loss", "brier_score", "ece"])
        fig = plot_backtest_summary(bt, metrics=["log_loss"])
        assert len(fig.data) == 1
        assert fig.data[0].name == "log_loss"

    def test_correct_x_values(self) -> None:
        """X-axis values match summary index years."""
        years = [2018, 2019, 2020, 2021]
        bt = _make_backtest_result(years=years)
        fig = plot_backtest_summary(bt)
        assert list(fig.data[0].x) == years

    def test_correct_y_values(self) -> None:
        """Y-axis values match summary metric column."""
        bt = _make_backtest_result(years=[2020, 2021], metric_names=["log_loss"])
        fig = plot_backtest_summary(bt)
        expected = bt.summary["log_loss"].tolist()
        assert fig.data[0].y == pytest.approx(expected)

    def test_single_fold(self) -> None:
        """Works with a single fold (edge case)."""
        bt = _make_backtest_result(years=[2020])
        fig = plot_backtest_summary(bt)
        assert isinstance(fig, go.Figure)
        assert len(fig.data[0].x) == 1


# ── TestPlotMetricComparison ────────────────────────────────────────────────


class TestPlotMetricComparison:
    """Tests for plot_metric_comparison."""

    def test_returns_figure(self) -> None:
        """plot_metric_comparison returns a go.Figure."""
        bt_a = _make_backtest_result()
        bt_b = _make_backtest_result(years=[2020, 2021, 2022])
        results = {"ModelA": bt_a, "ModelB": bt_b}

        fig = plot_metric_comparison(results, "log_loss")

        assert isinstance(fig, go.Figure)

    def test_correct_number_of_traces(self) -> None:
        """One trace per model."""
        results = {
            "Elo": _make_backtest_result(),
            "XGBoost": _make_backtest_result(),
            "Ensemble": _make_backtest_result(),
        }
        fig = plot_metric_comparison(results, "log_loss")
        assert len(fig.data) == 3

    def test_trace_names_match_model_names(self) -> None:
        """Trace names match the provided model names."""
        results = {
            "Alpha": _make_backtest_result(),
            "Beta": _make_backtest_result(),
        }
        fig = plot_metric_comparison(results, "log_loss")
        names = {t.name for t in fig.data}
        assert names == {"Alpha", "Beta"}

    def test_title_includes_metric(self) -> None:
        """Title includes the metric name."""
        results = {"M": _make_backtest_result()}
        fig = plot_metric_comparison(results, "brier_score")
        assert "brier_score" in fig.layout.title.text


# ── TestPlotAdvancementHeatmap ──────────────────────────────────────────────


class TestPlotAdvancementHeatmap:
    """Tests for plot_advancement_heatmap."""

    def test_returns_figure(self) -> None:
        """plot_advancement_heatmap returns a go.Figure."""
        sim = _make_simulation_result()
        fig = plot_advancement_heatmap(sim)
        assert isinstance(fig, go.Figure)

    def test_heatmap_trace_present(self) -> None:
        """Figure contains a Heatmap trace."""
        sim = _make_simulation_result()
        fig = plot_advancement_heatmap(sim)
        heatmap_traces = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmap_traces) == 1

    def test_correct_dimensions(self) -> None:
        """Heatmap z-data has correct shape (n_teams × n_rounds)."""
        sim = _make_simulation_result(n_teams=16, n_rounds=4)
        fig = plot_advancement_heatmap(sim)
        z = fig.data[0].z
        assert np.array(z).shape == (16, 4)

    def test_team_labels_applied(self) -> None:
        """Custom team labels appear as y-axis values."""
        sim = _make_simulation_result(n_teams=4, n_rounds=2)
        labels = {0: "Duke", 1: "UNC", 2: "UK", 3: "KU"}
        fig = plot_advancement_heatmap(sim, team_labels=labels)
        y_vals = list(fig.data[0].y)
        assert y_vals == ["Duke", "UNC", "UK", "KU"]

    def test_default_labels_are_indices(self) -> None:
        """Without labels, y-axis uses team indices."""
        sim = _make_simulation_result(n_teams=4, n_rounds=2)
        fig = plot_advancement_heatmap(sim)
        y_vals = list(fig.data[0].y)
        assert y_vals == ["0", "1", "2", "3"]

    def test_color_scale_uses_red_green(self) -> None:
        """Colorscale uses COLOR_RED → COLOR_GREEN."""
        sim = _make_simulation_result()
        fig = plot_advancement_heatmap(sim)
        cscale = fig.data[0].colorscale
        assert cscale[0][1] == COLOR_RED
        assert cscale[1][1] == COLOR_GREEN


# ── TestPlotScoreDistribution ───────────────────────────────────────────────


class TestPlotScoreDistribution:
    """Tests for plot_score_distribution."""

    def test_returns_figure(self) -> None:
        """plot_score_distribution returns a go.Figure."""
        dist = _make_bracket_distribution()
        fig = plot_score_distribution(dist)
        assert isinstance(fig, go.Figure)

    def test_histogram_bar_present(self) -> None:
        """Figure contains a Bar trace for the histogram."""
        dist = _make_bracket_distribution()
        fig = plot_score_distribution(dist)
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bar_traces) == 1

    def test_percentile_lines_present(self) -> None:
        """Figure contains vertical lines for 5 percentile markers."""
        dist = _make_bracket_distribution()
        fig = plot_score_distribution(dist)
        # Scatter traces used for percentile lines (excluding the bar)
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        # 5 percentile lines: P5, P25, P50, P75, P95
        assert len(scatter_traces) == 5

    def test_percentile_values_in_names(self) -> None:
        """Percentile trace names include the percentile label."""
        dist = _make_bracket_distribution()
        fig = plot_score_distribution(dist)
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        names = {t.name for t in scatter_traces}
        for pct in [5, 25, 50, 75, 95]:
            assert any(f"P{pct}" in n for n in names), f"P{pct} line not found"

    def test_custom_title(self) -> None:
        """Custom title is applied."""
        dist = _make_bracket_distribution()
        fig = plot_score_distribution(dist, title="Pool Scores")
        assert fig.layout.title.text == "Pool Scores"

    def test_bar_color_is_green(self) -> None:
        """Histogram bars use COLOR_GREEN."""
        dist = _make_bracket_distribution()
        fig = plot_score_distribution(dist)
        bar = [t for t in fig.data if isinstance(t, go.Bar)][0]
        assert bar.marker.color == COLOR_GREEN


# ── TestProgressBarIntegration ──────────────────────────────────────────────


class TestProgressBarIntegration:
    """Tests for tqdm progress bar integration."""

    def test_run_backtest_progress_param_accepted(self) -> None:
        """run_backtest accepts progress parameter without error."""
        from unittest.mock import MagicMock

        from rich.console import Console

        from ncaa_eval.evaluation.backtest import run_backtest

        # Build minimal mock feature server
        rng = np.random.default_rng(42)
        seasons = [2010, 2011, 2012]
        dfs: dict[int, pd.DataFrame] = {}
        for y in seasons:
            total = 10
            dfs[y] = pd.DataFrame(
                {
                    "game_id": [f"{y}_{i}" for i in range(total)],
                    "season": y,
                    "day_num": list(range(total)),
                    "date": pd.date_range(f"{y}-01-01", periods=total, freq="D"),
                    "team_a_id": rng.integers(1000, 2000, size=total),
                    "team_b_id": rng.integers(1000, 2000, size=total),
                    "is_tournament": [False] * 5 + [True] * 5,
                    "loc_encoding": rng.choice([1, -1, 0], size=total),
                    "team_a_won": rng.choice([True, False], size=total),
                    "elo_diff": rng.normal(0.0, 50.0, size=total),
                }
            )

        server = MagicMock()
        server.serve_season_features.side_effect = lambda year, mode="batch": dfs.get(year, pd.DataFrame())

        # Minimal model
        from pathlib import Path
        from typing import Self

        from ncaa_eval.model.base import Model, ModelConfig

        class _Stub(Model):
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
                return ModelConfig(model_name="stub")

        console = Console(quiet=True)
        result = run_backtest(
            _Stub(),
            server,
            seasons=seasons,
            n_jobs=1,
            metric_fns={"const": lambda yt, yp: 0.42},
            console=console,
            progress=True,
        )

        assert len(result.fold_results) > 0

    def test_simulate_tournament_mc_progress_param_accepted(self) -> None:
        """simulate_tournament_mc accepts progress parameter without error."""
        from ncaa_eval.evaluation.simulation import (
            BracketNode,
            BracketStructure,
            StandardScoring,
            simulate_tournament_mc,
        )

        # Build minimal 4-team bracket
        n = 4
        leaves = [BracketNode(round_index=-1, team_index=i) for i in range(n)]
        semi_left = BracketNode(round_index=0, left=leaves[0], right=leaves[1])
        semi_right = BracketNode(round_index=0, left=leaves[2], right=leaves[3])
        root = BracketNode(round_index=1, left=semi_left, right=semi_right)
        bracket = BracketStructure(
            root=root,
            team_ids=tuple(range(n)),
            team_index_map={i: i for i in range(n)},
        )

        # Equal probability matrix
        P = np.full((n, n), 0.5, dtype=np.float64)
        np.fill_diagonal(P, 0.0)

        result = simulate_tournament_mc(
            bracket=bracket,
            P=P,
            scoring_rules=[StandardScoring()],
            season=2024,
            n_simulations=100,
            rng=np.random.default_rng(42),
            progress=True,
        )

        assert result.method == "monte_carlo"
        assert result.advancement_probs.shape == (n, 2)
