"""Presentation page — Pool Scorer placeholder."""

from __future__ import annotations

import streamlit as st

st.header("Pool Scorer")
st.info("Coming in Story 7.6")

if st.session_state.get("selected_scoring"):
    st.caption(f"Selected scoring: {st.session_state.selected_scoring}")
