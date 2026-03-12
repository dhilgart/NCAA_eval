"""Lab page — Backtest Leaderboard.

Displays a sortable leaderboard comparing all trained models by various
metrics, with diagnostic KPI cards and conditional formatting.
"""

from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]
import streamlit as st

from dashboard.lib.data_loaders import (
    get_data_dir,
    get_metric_cols as _get_metric_cols,
    load_available_runs,
    load_leaderboard_data,
)


def _style_metric_table(df: pd.DataFrame, metric_cols: list[str]) -> pd.io.formats.style.Styler:
    """Apply background gradient and formatting to a metric DataFrame."""
    # roc_auc is "higher is better"; all others are "lower is better"
    lower_better = [m for m in metric_cols if m != "roc_auc"]
    higher_better = [m for m in metric_cols if m == "roc_auc"]
    styled = df.style
    if lower_better:
        styled = styled.background_gradient(cmap="RdYlGn_r", subset=lower_better)
    if higher_better:
        styled = styled.background_gradient(cmap="RdYlGn", subset=higher_better)
    return styled.format({m: "{:.4f}" for m in metric_cols})


def _render_leaderboard() -> None:
    """Render the backtest leaderboard page."""
    # Breadcrumbs
    col_nav, col_bc = st.columns([1, 3])
    with col_nav:
        st.page_link("pages/home.py", label="← Home")
    with col_bc:
        st.caption("Home > Lab > Backtest Leaderboard")

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

    metric_cols = _get_metric_cols(df)
    display_cols = ["run_id", "model_type", "year"] + metric_cols

    if selected_year is not None:
        year_df = df[df["year"] == selected_year]
        if year_df.empty:
            st.info(f"No backtest results for {selected_year}")
            return
        display_df = year_df[display_cols].copy()
    else:
        display_df = df.groupby(["run_id", "model_type"], as_index=False)[metric_cols].mean()

    # -- Diagnostic KPI cards (st.metric) --------------------------------------
    def _fmt(v: float) -> str:
        return f"{v:.4f}" if v == v else "N/A"  # NaN check: NaN != NaN

    _BUILTIN_KPI_COLS = ("log_loss", "brier_score", "roc_auc", "ece")
    _builtins_present = all(m in display_df.columns for m in _BUILTIN_KPI_COLS)
    if len(display_df) >= 1 and _builtins_present:
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
            _fmt(best_ll),
            delta=f"{best_ll - worst_ll:.4f}" if len(display_df) > 1 and best_ll == best_ll else None,
            delta_color="inverse",
        )
        col2.metric(
            "Best Brier",
            _fmt(best_bs),
            delta=f"{best_bs - worst_bs:.4f}" if len(display_df) > 1 and best_bs == best_bs else None,
            delta_color="inverse",
        )
        col3.metric(
            "Best ROC-AUC",
            _fmt(best_auc),
            delta=f"{best_auc - worst_auc:.4f}" if len(display_df) > 1 and best_auc == best_auc else None,
            delta_color="normal",
        )
        col4.metric(
            "Lowest ECE",
            _fmt(best_ece),
            delta=f"{best_ece - worst_ece:.4f}" if len(display_df) > 1 and best_ece == best_ece else None,
            delta_color="inverse",
        )

    # -- Styled leaderboard table ----------------------------------------------
    sort_col = "log_loss" if "log_loss" in metric_cols else (metric_cols[0] if metric_cols else "")
    if sort_col and sort_col in display_df.columns:
        display_df = display_df.sort_values(sort_col, ascending=True).reset_index(drop=True)

    styled = _style_metric_table(display_df, metric_cols)

    event = st.dataframe(
        styled,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key="leaderboard_selection",
    )

    # -- Click-to-navigate to Model Deep Dive ----------------------------------
    selected_rows = event.get("selection", {}).get("rows", [])
    if selected_rows:
        selected_idx = selected_rows[0]
        selected_run_id = str(display_df.iloc[selected_idx]["run_id"])
        st.session_state["selected_run_id"] = selected_run_id
        st.switch_page("pages/3_Model_Deep_Dive.py")


_render_leaderboard()
