"""Streamlit dashboard entry point.

Launch with::

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import streamlit as st

from dashboard.lib.filters import (
    get_data_dir,
    load_available_runs,
    load_available_scorings,
    load_available_years,
)
from dashboard.lib.styles import MONOSPACE_CSS

# --- Page config (MUST be first Streamlit command) --------------------------

st.set_page_config(
    page_title="NCAA Eval",
    page_icon=":material/sports_basketball:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS -------------------------------------------------------------

st.markdown(MONOSPACE_CSS, unsafe_allow_html=True)

# --- Page definitions -------------------------------------------------------

home = st.Page("pages/home.py", title="Home", icon=":material/home:", default=True)
leaderboard = st.Page("pages/1_Lab.py", title="Backtest Leaderboard", icon=":material/leaderboard:")
deep_dive = st.Page(
    "pages/3_Model_Deep_Dive.py",
    title="Model Deep Dive",
    icon=":material/analytics:",
)
bracket = st.Page(
    "pages/2_Presentation.py",
    title="Bracket Visualizer",
    icon=":material/account_tree:",
)
pool_scorer = st.Page("pages/4_Pool_Scorer.py", title="Pool Scorer", icon=":material/calculate:")

pg = st.navigation(
    {
        "": [home],
        "Lab": [leaderboard, deep_dive],
        "Presentation": [bracket, pool_scorer],
    }
)

# --- Session-state defaults -------------------------------------------------

if "selected_year" not in st.session_state:
    st.session_state.selected_year = None
if "selected_run_id" not in st.session_state:
    st.session_state.selected_run_id = None
if "selected_scoring" not in st.session_state:
    st.session_state.selected_scoring = "standard"

# --- Global sidebar filters -------------------------------------------------

data_dir = str(get_data_dir())

with st.sidebar:
    # Tournament Year
    years = load_available_years(data_dir)
    if years:
        st.selectbox("Tournament Year", options=years, key="selected_year")
    else:
        st.info("No data available — run `python sync.py` first")

    # Model Run
    runs = load_available_runs(data_dir)
    run_options = [r["run_id"] for r in runs] if runs else []
    if run_options:
        st.selectbox("Model Run", options=run_options, key="selected_run_id")
    else:
        st.info("No model runs available")

    # Scoring Format
    scorings = load_available_scorings()
    if scorings:
        st.selectbox("Scoring Format", options=scorings, key="selected_scoring")

# --- Run selected page ------------------------------------------------------

pg.run()
