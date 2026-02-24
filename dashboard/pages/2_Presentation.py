"""Presentation page — Bracket Visualizer.

Interactive bracket visualizer showing per-game win probabilities and team
advancement odds from a trained model.  Renders a 64-team bracket tree,
advancement probability heatmap, expected-points table, and optional Monte
Carlo score distribution.
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from dashboard.lib.bracket_renderer import render_bracket_html
from dashboard.lib.filters import (
    BracketSimulationResult,
    get_data_dir,
    load_tourney_seeds,
    run_bracket_simulation,
)
from ncaa_eval.evaluation.plotting import (
    plot_advancement_heatmap,
    plot_score_distribution,
)


def _render_results(sim_data: BracketSimulationResult, scoring: str) -> None:
    """Render all bracket visualisation sections from simulation results."""
    result = sim_data.sim_result
    bracket = sim_data.bracket
    most_likely = sim_data.most_likely

    # Champion summary
    champ_label = sim_data.team_labels.get(
        bracket.team_index_map.get(most_likely.champion_team_id, -1), "Unknown"
    )
    st.success(f"Predicted Champion: **{champ_label}** (log-likelihood: {most_likely.log_likelihood:.2f})")

    # Bracket tree
    st.subheader("Most-Likely Bracket")
    bracket_html = render_bracket_html(
        bracket_team_ids=bracket.team_ids,
        most_likely_winners=most_likely.winners,
        team_labels=sim_data.team_labels,
        seed_map=bracket.seed_map,
        prob_matrix=sim_data.prob_matrix,
    )
    components.html(bracket_html, height=700, scrolling=True)

    # Advancement heatmap
    st.subheader("Advancement Probabilities")
    fig_heatmap = plot_advancement_heatmap(result, team_labels=sim_data.team_labels)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # Expected points table
    st.subheader(f"Expected Points ({scoring})")
    if scoring in result.expected_points:
        ep = result.expected_points[scoring]
        ep_data: list[dict[str, object]] = []
        for idx in range(len(bracket.team_ids)):
            team_id = bracket.team_ids[idx]
            label = sim_data.team_labels.get(idx, str(team_id))
            ep_data.append({"Team": label, "Expected Points": round(float(ep[idx]), 2)})
        ep_data.sort(key=lambda d: float(str(d["Expected Points"])), reverse=True)
        st.dataframe(ep_data, use_container_width=True, height=400)
    else:
        st.info("Expected points not available for the selected scoring rule.")

    # Score distribution (MC only)
    if result.method == "monte_carlo" and result.bracket_distributions:
        st.subheader("Score Distribution (Monte Carlo)")
        if scoring in result.bracket_distributions:
            dist = result.bracket_distributions[scoring]
            fig_dist = plot_score_distribution(dist, title=f"Bracket Score Distribution — {scoring}")
            st.plotly_chart(fig_dist, use_container_width=True)


def _render_bracket_page() -> None:
    """Render the Bracket Visualizer page."""
    # Breadcrumbs
    col_nav, col_bc = st.columns([1, 3])
    with col_nav:
        st.page_link("pages/home.py", label="← Home")
    with col_bc:
        st.caption("Home > Presentation > Bracket Visualizer")

    st.header("Bracket Visualizer")

    # Validate required session state
    selected_year: int | None = st.session_state.get("selected_year")
    selected_run_id: str | None = st.session_state.get("selected_run_id")
    selected_scoring: str | None = st.session_state.get("selected_scoring")

    if selected_run_id is None:
        st.info("Select a model run from the sidebar to visualize bracket predictions.")
        return
    if selected_year is None:
        st.info("Select a tournament year from the sidebar.")
        return

    scoring = selected_scoring or "standard"
    data_dir = str(get_data_dir())

    # Check seeds available
    seeds_raw = load_tourney_seeds(data_dir, selected_year)
    if not seeds_raw:
        st.warning(
            f"No tournament seeds available for {selected_year}. " "Run `python sync.py` to download data."
        )
        return

    # Simulation method selector
    st.subheader("Simulation Settings")
    sim_col1, sim_col2 = st.columns([1, 1])
    with sim_col1:
        method = st.selectbox(
            "Simulation Method",
            options=["analytical", "monte_carlo"],
            format_func=lambda x: "Analytical (exact)" if x == "analytical" else "Monte Carlo",
            key="bracket_sim_method",
        )
    n_sims = 10_000
    with sim_col2:
        if method == "monte_carlo":
            n_sims = st.slider(
                "Number of Simulations",
                min_value=1_000,
                max_value=100_000,
                value=10_000,
                step=1_000,
                key="bracket_n_sims",
            )

    # Run simulation
    spinner_msg = "Running tournament simulation..." if method == "monte_carlo" else "Computing bracket..."
    with st.spinner(spinner_msg):
        sim_data = run_bracket_simulation(
            data_dir=data_dir,
            run_id=selected_run_id,
            season=selected_year,
            scoring_name=scoring,
            method=method,
            n_simulations=n_sims,
        )

    if sim_data is None:
        st.warning(
            "Could not simulate bracket. Ensure the selected model has been trained "
            f"and has data for {selected_year}."
        )
        return

    _render_results(sim_data, scoring)


_render_bracket_page()
