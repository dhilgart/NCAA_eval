"""Home page — welcome screen and summary metrics."""

from __future__ import annotations

import streamlit as st

from dashboard.lib.data_loaders import get_data_dir, load_available_runs, load_available_years

st.title("NCAA Eval Dashboard")
st.markdown("Explore model backtests, tournament simulations, and bracket predictions for March Madness.")

data_dir = str(get_data_dir())
years = load_available_years(data_dir)
runs = load_available_runs(data_dir)

if not years:
    st.error(
        "**Setup needed — no data found.**\n\n"
        "Download game data before using the dashboard:\n\n"
        "```\nncaa-eval sync --source all --dest data/\n```"
    )
elif not runs:
    st.info(
        "**Data available, but no models trained yet.**\n\n"
        "Train a model to unlock the dashboard:\n\n"
        "```\nncaa-eval train --model elo --start-year 2015 --end-year 2025\n```"
    )
else:
    col1, col2 = st.columns(2)
    col1.metric("Available Seasons", len(years))
    col2.metric("Model Runs", len(runs))

    st.markdown("---")
    st.markdown(
        "Use the **sidebar** to select a tournament year, model run, and scoring "
        "format, then navigate to a Lab or Presentation page."
    )
