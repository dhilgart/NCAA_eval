"""Plotly visualization adapters for evaluation results.

Provides standalone functions that accept evaluation result objects
and return interactive ``plotly.graph_objects.Figure`` instances for
Jupyter notebook rendering.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go  # type: ignore[import-untyped]

from ncaa_eval.evaluation.backtest import BacktestResult
from ncaa_eval.evaluation.metrics import reliability_diagram_data
from ncaa_eval.evaluation.simulation import (
    N_ROUNDS,
    BracketDistribution,
    SimulationResult,
)

# UX spec color palette
COLOR_GREEN: str = "#28a745"
COLOR_RED: str = "#dc3545"
COLOR_NEUTRAL: str = "#6c757d"

# Extended palette for multi-trace plots (green, red, neutral, then extras)
_PALETTE: tuple[str, ...] = (
    COLOR_GREEN,
    COLOR_RED,
    COLOR_NEUTRAL,
    "#17a2b8",  # teal
    "#ffc107",  # amber
    "#6f42c1",  # purple
    "#fd7e14",  # orange
    "#20c997",  # mint
)

# Use plotly_dark template for dark-mode compatibility
TEMPLATE: str = "plotly_dark"

# Round labels for advancement heatmap
_ROUND_LABELS: tuple[str, ...] = ("R64", "R32", "S16", "E8", "F4", "Championship")


def plot_reliability_diagram(
    y_true: npt.NDArray[np.float64],
    y_prob: npt.NDArray[np.float64],
    *,
    n_bins: int = 10,
    title: str | None = None,
) -> go.Figure:
    """Reliability diagram: predicted vs. actual probability with bin counts.

    Args:
        y_true: Binary labels (0 or 1).
        y_prob: Predicted probabilities for the positive class.
        n_bins: Number of calibration bins (default 10).
        title: Optional figure title.

    Returns:
        Interactive Plotly Figure with calibration curve, diagonal
        reference, and bar overlay of per-bin sample counts.
    """
    data = reliability_diagram_data(y_true, y_prob, n_bins=n_bins)

    fig = go.Figure()

    # Bar trace: bin counts on secondary y-axis
    fig.add_trace(
        go.Bar(
            x=data.mean_predicted_value,
            y=data.bin_counts,
            name="Bin Count",
            marker_color=COLOR_NEUTRAL,
            opacity=0.3,
            yaxis="y2",
        )
    )

    # Diagonal reference line: perfect calibration
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"dash": "dash", "color": COLOR_NEUTRAL},
            name="Perfect",
            showlegend=True,
        )
    )

    # Scatter trace: calibration curve
    fig.add_trace(
        go.Scatter(
            x=data.mean_predicted_value,
            y=data.fraction_of_positives,
            mode="lines+markers",
            marker={"color": COLOR_GREEN, "size": 8},
            line={"color": COLOR_GREEN},
            name="Calibration",
            text=[str(c) for c in data.bin_counts],
            hovertemplate=(
                "Predicted: %{x:.3f}<br>" "Observed: %{y:.3f}<br>" "Count: %{text}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        template=TEMPLATE,
        title=title or "Reliability Diagram",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives",
        yaxis2={
            "title": "Bin Count",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        },
        legend={"x": 0.01, "y": 0.99},
    )

    return fig


def plot_backtest_summary(
    result: BacktestResult,
    *,
    metrics: Sequence[str] | None = None,
) -> go.Figure:
    """Per-year metric values from a backtest result.

    Args:
        result: Backtest result containing the summary DataFrame.
        metrics: Metric column names to include. Defaults to all
            metric columns (excludes ``elapsed_seconds``).

    Returns:
        Interactive Plotly Figure with one line per metric, x=year.
    """
    summary = result.summary
    if metrics is None:
        metric_cols = [c for c in summary.columns if c != "elapsed_seconds"]
    else:
        metric_cols = list(metrics)

    if not metric_cols:
        msg = "No metric columns to plot. BacktestResult.summary has no columns besides 'elapsed_seconds'."
        raise ValueError(msg)

    years = summary.index.tolist()

    fig = go.Figure()
    for i, col in enumerate(metric_cols):
        color = _PALETTE[i % len(_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=years,
                y=summary[col].tolist(),
                mode="lines+markers",
                name=col,
                line={"color": color},
                marker={"color": color},
            )
        )

    fig.update_layout(
        template=TEMPLATE,
        title="Backtest Summary",
        xaxis_title="Year",
        yaxis_title="Metric Value",
    )

    return fig


def plot_metric_comparison(
    results: Mapping[str, BacktestResult],
    metric: str,
) -> go.Figure:
    """Multi-model overlay: one line per model for a given metric across years.

    Args:
        results: Mapping of model name to BacktestResult.
        metric: Metric column name to compare.

    Returns:
        Interactive Plotly Figure with one line per model.
    """
    fig = go.Figure()

    for i, (model_name, bt) in enumerate(results.items()):
        if metric not in bt.summary.columns:
            available = [c for c in bt.summary.columns if c != "elapsed_seconds"]
            msg = f"metric {metric!r} not found in results[{model_name!r}].summary. Available: {available}"
            raise ValueError(msg)
        color = _PALETTE[i % len(_PALETTE)]
        years = bt.summary.index.tolist()
        values = bt.summary[metric].tolist()
        fig.add_trace(
            go.Scatter(
                x=years,
                y=values,
                mode="lines+markers",
                name=model_name,
                line={"color": color},
                marker={"color": color},
                hovertemplate=(f"{model_name}<br>" "Year: %{x}<br>" f"{metric}: %{{y:.4f}}<extra></extra>"),
            )
        )

    fig.update_layout(
        template=TEMPLATE,
        title=f"Model Comparison — {metric}",
        xaxis_title="Year",
        yaxis_title=metric,
    )

    return fig


def plot_advancement_heatmap(
    result: SimulationResult,
    team_labels: Mapping[int, str] | None = None,
) -> go.Figure:
    """Heatmap of per-team advancement probabilities by round.

    Args:
        result: Simulation result with ``advancement_probs`` array.
        team_labels: Optional mapping of **team index** (0..n-1, bracket
            position order) to display name.  When ``None``, team indices
            are shown as-is.  Note: keys are bracket indices, not canonical
            team IDs — use ``BracketStructure.team_index_map`` to translate
            from team IDs to indices before passing this argument.

    Returns:
        Interactive Plotly Figure showing a heatmap with teams on
        y-axis and rounds on x-axis.
    """
    adv = result.advancement_probs  # shape (n_teams, n_rounds)
    n_teams = adv.shape[0]
    n_rounds = min(adv.shape[1], N_ROUNDS)
    round_labels = list(_ROUND_LABELS[:n_rounds])

    if team_labels is not None:
        y_labels = [team_labels.get(i, str(i)) for i in range(n_teams)]
    else:
        y_labels = [str(i) for i in range(n_teams)]

    fig = go.Figure(
        data=go.Heatmap(
            z=adv[:, :n_rounds],
            x=round_labels,
            y=y_labels,
            colorscale=[[0, COLOR_RED], [1, COLOR_GREEN]],
            zmin=0.0,
            zmax=1.0,
            hovertemplate=("Team: %{y}<br>" "Round: %{x}<br>" "P(advance): %{z:.3f}<extra></extra>"),
        )
    )

    fig.update_layout(
        template=TEMPLATE,
        title="Advancement Probabilities",
        xaxis_title="Round",
        yaxis_title="Team",
        yaxis={"autorange": "reversed"},
    )

    return fig


def plot_score_distribution(
    dist: BracketDistribution,
    *,
    title: str | None = None,
) -> go.Figure:
    """Histogram of bracket score distribution with percentile markers.

    Args:
        dist: Bracket distribution with pre-computed histogram data
            and percentile values.
        title: Optional figure title.

    Returns:
        Interactive Plotly Figure with histogram bars and vertical
        percentile lines at 5th, 25th, 50th, 75th, and 95th.
    """
    # Convert bin edges to bin centers for the bar chart
    bin_centers = (dist.histogram_bins[:-1] + dist.histogram_bins[1:]) / 2.0
    bin_width = (
        float(dist.histogram_bins[1] - dist.histogram_bins[0]) if len(dist.histogram_bins) >= 2 else 1.0
    )

    fig = go.Figure()

    # Histogram bars
    fig.add_trace(
        go.Bar(
            x=bin_centers.tolist(),
            y=dist.histogram_counts.tolist(),
            width=bin_width,
            marker_color=COLOR_GREEN,
            opacity=0.7,
            name="Score Distribution",
        )
    )

    # Percentile vertical lines
    percentile_colors = {
        5: COLOR_RED,
        25: COLOR_NEUTRAL,
        50: COLOR_GREEN,
        75: COLOR_NEUTRAL,
        95: COLOR_RED,
    }
    max_count = int(np.max(dist.histogram_counts)) if len(dist.histogram_counts) > 0 else 1

    for pct, value in sorted(dist.percentiles.items()):
        color = percentile_colors.get(pct, COLOR_NEUTRAL)
        fig.add_trace(
            go.Scatter(
                x=[value, value],
                y=[0, max_count],
                mode="lines",
                line={"color": color, "dash": "dash", "width": 2},
                name=f"P{pct}: {value:.1f}",
            )
        )

    fig.update_layout(
        template=TEMPLATE,
        title=title or "Bracket Score Distribution",
        xaxis_title="Score",
        yaxis_title="Count",
        bargap=0.05,
    )

    return fig
