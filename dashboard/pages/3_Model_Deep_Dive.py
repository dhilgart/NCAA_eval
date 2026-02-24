"""Lab page — Model Deep Dive & Reliability Diagrams.

Provides detailed diagnostic views for a selected model run: reliability
diagram, per-year metric summary, feature importance, and hyperparameters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
import streamlit as st

from dashboard.lib.filters import (
    get_data_dir,
    load_available_runs,
    load_feature_importances,
    load_fold_predictions,
    load_leaderboard_data,
)
from ncaa_eval.evaluation.plotting import (
    COLOR_GREEN,
    TEMPLATE,
    plot_reliability_diagram,
)

_METRIC_COLS = ["log_loss", "brier_score", "roc_auc", "ece"]


def _render_reliability_section(data_dir: str, run_id: str, label: str) -> None:
    """Render metric explorer with year drill-down and reliability diagram."""
    st.subheader("Metric Explorer")
    st.caption("Drill-down by year. Round, seed matchup, and conference filters are post-MVP.")

    fold_preds_raw = load_fold_predictions(data_dir, run_id)
    if not fold_preds_raw:
        st.warning("No fold predictions available. Re-run training to generate diagnostic data.")
        return

    fold_df = pd.DataFrame(fold_preds_raw)
    available_years = sorted(fold_df["year"].unique().tolist())
    year_options: list[str] = ["All Years (Aggregate)"] + [str(y) for y in available_years]
    selected_year_str = st.selectbox("Fold Year", options=year_options, key="deep_dive_year")

    if selected_year_str == "All Years (Aggregate)":
        filtered = fold_df
        title_suffix = "All Years"
    else:
        filtered = fold_df[fold_df["year"] == int(selected_year_str)]
        title_suffix = selected_year_str

    st.subheader("Reliability Diagram")
    if not filtered.empty:
        y_true = filtered["team_a_won"].to_numpy().astype(np.float64)
        y_prob = filtered["pred_win_prob"].to_numpy().astype(np.float64)
        fig = plot_reliability_diagram(
            y_true, y_prob, title=f"Reliability Diagram — {label} ({title_suffix})"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No predictions available for the selected year.")


def _render_metric_summary(data_dir: str, run_id: str) -> None:
    """Render per-year metric summary table."""
    st.subheader("Per-Year Metric Summary")
    lb_raw = load_leaderboard_data(data_dir)
    if not lb_raw:
        st.info("No leaderboard data available.")
        return

    lb_df = pd.DataFrame(lb_raw)
    run_metrics = lb_df[lb_df["run_id"] == run_id].copy()
    if run_metrics.empty:
        st.info("No metric summary available for this run.")
        return

    display_cols = ["year"] + _METRIC_COLS
    display_df = run_metrics[display_cols].sort_values("year").reset_index(drop=True)
    styled = (
        display_df.style.background_gradient(cmap="RdYlGn_r", subset=["log_loss", "brier_score", "ece"])
        .background_gradient(cmap="RdYlGn", subset=["roc_auc"])
        .format({"log_loss": "{:.4f}", "brier_score": "{:.4f}", "roc_auc": "{:.4f}", "ece": "{:.4f}"})
    )
    st.dataframe(styled, use_container_width=True)


def _render_feature_importance(data_dir: str, run_id: str, model_type: str) -> None:
    """Render feature importance bar chart (XGBoost only)."""
    st.subheader("Feature Importance")
    importances = load_feature_importances(data_dir, run_id)
    if not importances:
        if model_type == "xgboost":
            st.info("Feature importance not available. Re-run training to persist model artifacts.")
        else:
            st.info("Feature importance is not available for stateful models.")
        return

    feature_names = [d["feature"] for d in importances]
    importance_values = [d["importance"] for d in importances]
    fig = go.Figure(go.Bar(x=importance_values, y=feature_names, orientation="h", marker_color=COLOR_GREEN))
    fig.update_layout(
        template=TEMPLATE,
        title="Feature Importance (Gain)",
        xaxis_title="Importance",
        yaxis_title="Feature",
        yaxis=dict(autorange="reversed"),
        height=min(max(400, len(feature_names) * 25), 2000),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_deep_dive() -> None:
    """Render the Model Deep Dive page."""
    run_id: str | None = st.session_state.get("selected_run_id")

    if run_id is None:
        st.info("Select a model run from the Leaderboard to view diagnostics.")
        st.page_link("pages/1_Lab.py", label="Go to Leaderboard")
        return

    data_dir = str(get_data_dir())
    runs = load_available_runs(data_dir)
    run = next((r for r in runs if r["run_id"] == run_id), None)
    if run is None:
        st.warning(f"Run {run_id} not found.")
        return

    model_type = str(run["model_type"])
    label = f"{model_type}-{run_id[:8]}"

    # Breadcrumbs and navigation
    col_nav, col_bc = st.columns([1, 3])
    with col_nav:
        st.page_link("pages/1_Lab.py", label="← Back to Leaderboard")
    with col_bc:
        st.caption(f"Home > Lab > {label}")

    st.header(f"Model Deep Dive: {label}")

    _render_reliability_section(data_dir, run_id, label)
    _render_metric_summary(data_dir, run_id)
    _render_feature_importance(data_dir, run_id, model_type)

    st.subheader("Hyperparameters")
    st.json(run.get("hyperparameters", {}))


_render_deep_dive()
