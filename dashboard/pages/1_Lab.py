"""Lab page — Backtest Leaderboard placeholder."""

from __future__ import annotations

import streamlit as st

st.header("Backtest Leaderboard")
st.info("Coming in Story 7.3")

if st.session_state.get("selected_year"):
    st.caption(f"Selected year: {st.session_state.selected_year}")
if st.session_state.get("selected_run_id"):
    st.caption(f"Selected run: {st.session_state.selected_run_id}")
