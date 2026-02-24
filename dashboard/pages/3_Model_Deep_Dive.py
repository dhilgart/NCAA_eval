"""Lab page — Model Deep Dive placeholder."""

from __future__ import annotations

import streamlit as st

st.header("Model Deep Dive")
st.info("Coming in Story 7.4")

if st.session_state.get("selected_run_id"):
    st.caption(f"Selected run: {st.session_state.selected_run_id}")
