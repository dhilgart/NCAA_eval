"""Presentation page — Bracket Visualizer placeholder."""

from __future__ import annotations

import streamlit as st

st.header("Bracket Visualizer")
st.info("Coming in Story 7.5")

if st.session_state.get("selected_year"):
    st.caption(f"Selected year: {st.session_state.selected_year}")
if st.session_state.get("selected_scoring"):
    st.caption(f"Selected scoring: {st.session_state.selected_scoring}")
