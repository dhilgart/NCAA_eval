"""Lab page — Backtest Leaderboard.

Displays a sortable leaderboard comparing all trained models by various
metrics, with diagnostic KPI cards and conditional formatting.
"""

from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]
import streamlit as st

from dashboard.lib.filters import get_data_dir, load_available_runs, load_leaderboard_data

_METRIC_COLS = ["log_loss", "brier_score", "roc_auc", "ece"]


def _render_leaderboard() -> None:
    """Render the backtest leaderboard page."""
    st.header("Backtest Leaderboard")

    data_dir = str(get_data_dir())
    raw = load_leaderboard_data(data_dir)

    if not raw:
        runs = load_available_runs(data_dir)
        if runs:
            st.warning("No backtest metrics available. Re-run training to generate metrics.")
        else:
            st.info(
                "No model runs available. Train a model first: `python -m ncaa_eval.cli train --model elo`"
            )
        return

    df = pd.DataFrame(raw)
    if df.empty:
        st.info("No model runs available. Train a model first: `python -m ncaa_eval.cli train --model elo`")
        return

    # -- Apply year filter -----------------------------------------------------
    selected_year = st.session_state.setdefault("selected_year", None)

    if selected_year is not None:
        year_df = df[df["year"] == selected_year]
        if year_df.empty:
            st.info(f"No backtest results for {selected_year}")
            return
        display_df = year_df.copy()
    else:
        display_df = df.groupby(["run_id", "model_type"], as_index=False)[_METRIC_COLS].mean()

    # -- Diagnostic KPI cards (st.metric) --------------------------------------
    if len(display_df) >= 1:
        best_ll = display_df["log_loss"].min()
        best_bs = display_df["brier_score"].min()
        best_auc = display_df["roc_auc"].max()
        best_ece = display_df["ece"].min()

        worst_ll = display_df["log_loss"].max()
        worst_bs = display_df["brier_score"].max()
        worst_auc = display_df["roc_auc"].min()
        worst_ece = display_df["ece"].max()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Best Log Loss",
            f"{best_ll:.4f}",
            delta=f"{best_ll - worst_ll:.4f}" if len(display_df) > 1 else None,
            delta_color="inverse",
        )
        col2.metric(
            "Best Brier",
            f"{best_bs:.4f}",
            delta=f"{best_bs - worst_bs:.4f}" if len(display_df) > 1 else None,
            delta_color="inverse",
        )
        col3.metric(
            "Best ROC-AUC",
            f"{best_auc:.4f}",
            delta=f"{best_auc - worst_auc:.4f}" if len(display_df) > 1 else None,
            delta_color="normal",
        )
        col4.metric(
            "Lowest ECE",
            f"{best_ece:.4f}",
            delta=f"{best_ece - worst_ece:.4f}" if len(display_df) > 1 else None,
            delta_color="inverse",
        )

    # -- Styled leaderboard table ----------------------------------------------
    display_df = display_df.sort_values("log_loss", ascending=True).reset_index(drop=True)

    styled = (
        display_df.style.background_gradient(
            cmap="RdYlGn_r",
            subset=["log_loss", "brier_score", "ece"],
        )
        .background_gradient(
            cmap="RdYlGn",
            subset=["roc_auc"],
        )
        .format(
            {
                "log_loss": "{:.4f}",
                "brier_score": "{:.4f}",
                "roc_auc": "{:.4f}",
                "ece": "{:.4f}",
            }
        )
    )

    event = st.dataframe(
        styled,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        key="leaderboard_selection",
    )

    # -- Click-to-navigate to Model Deep Dive ----------------------------------
    if event and event.selection and event.selection.rows:  # type: ignore[attr-defined]
        selected_idx = event.selection.rows[0]  # type: ignore[attr-defined]
        selected_run_id = str(display_df.iloc[selected_idx]["run_id"])
        st.session_state["selected_run_id"] = selected_run_id
        st.switch_page("pages/3_Model_Deep_Dive.py")


_render_leaderboard()
