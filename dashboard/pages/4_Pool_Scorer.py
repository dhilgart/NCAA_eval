"""Presentation page — Pool Scorer & Point Outcome Analysis.

Configure pool-specific scoring rules, score a bracket against Monte Carlo
simulations, and analyze the distribution of possible point outcomes.  Exports
a final bracket entry as CSV.
"""

from __future__ import annotations

import inspect

import streamlit as st

from dashboard.lib.filters import (
    BracketSimulationResult,
    build_custom_scoring,
    export_bracket_csv,
    get_data_dir,
    load_tourney_seeds,
    run_bracket_simulation,
    score_chosen_bracket,
)
from ncaa_eval.evaluation.plotting import plot_score_distribution
from ncaa_eval.evaluation.simulation import get_scoring

# ---------------------------------------------------------------------------
# Rendering helpers (split out for C901 compliance)
# ---------------------------------------------------------------------------


def _render_outcome_summary(dist: object) -> None:
    """Render point outcome summary metrics as ``st.metric`` cards."""
    # BracketDistribution fields: percentiles, mean, std, scores
    percentiles: dict[int, float] = getattr(dist, "percentiles", {})
    mean: float = getattr(dist, "mean", 0.0)
    std: float = getattr(dist, "std", 0.0)
    scores = getattr(dist, "scores", None)

    min_score = float(scores.min()) if scores is not None and len(scores) > 0 else 0.0
    max_score = float(scores.max()) if scores is not None and len(scores) > 0 else 0.0

    st.subheader("Outcome Summary")
    row1 = st.columns(3)
    with row1[0]:
        st.metric("Median", f"{percentiles.get(50, 0.0):.1f} pts")
    with row1[1]:
        st.metric("Mean", f"{mean:.1f} pts")
    with row1[2]:
        st.metric("Std Dev", f"{std:.1f} pts")

    row2 = st.columns(2)
    with row2[0]:
        st.metric("Min", f"{min_score:.1f} pts")
    with row2[1]:
        st.metric("Max", f"{max_score:.1f} pts")

    row3 = st.columns(4)
    with row3[0]:
        st.metric("5th %ile", f"{percentiles.get(5, 0.0):.1f} pts")
    with row3[1]:
        st.metric("25th %ile", f"{percentiles.get(25, 0.0):.1f} pts")
    with row3[2]:
        st.metric("75th %ile", f"{percentiles.get(75, 0.0):.1f} pts")
    with row3[3]:
        st.metric("95th %ile", f"{percentiles.get(95, 0.0):.1f} pts")


def _render_distribution_chart(dist: object, scoring_label: str) -> None:
    """Render score distribution histogram via ``plot_score_distribution``."""
    fig = plot_score_distribution(dist, title=f"Bracket Score Distribution — {scoring_label}")  # type: ignore[arg-type]
    st.plotly_chart(fig, use_container_width=True)


def _render_results(
    sim_data: BracketSimulationResult,
    scoring_label: str,
    use_custom: bool,
    custom_points: tuple[float, ...],
) -> None:
    """Run scoring, display outcome analysis, and offer CSV export."""
    # Determine which scoring rule to use
    if use_custom:
        scoring_rule = build_custom_scoring(custom_points)
    else:
        scoring_cls = get_scoring(scoring_label)
        sig = inspect.signature(scoring_cls)
        if "seed_map" in sig.parameters:
            scoring_rule = scoring_cls(sim_data.bracket.seed_map)
        else:
            scoring_rule = scoring_cls()

    rule_name: str = scoring_rule.name

    # Score bracket against simulations
    with st.spinner("Scoring bracket against simulations..."):
        distributions = score_chosen_bracket(sim_data, [scoring_rule])

    if rule_name not in distributions:
        st.warning(f"Scoring failed for rule '{rule_name}'.")
        return

    dist = distributions[rule_name]

    # Outcome summary metrics (AC #2)
    _render_outcome_summary(dist)

    # Score distribution histogram (AC #3)
    st.subheader("Score Distribution")
    _render_distribution_chart(dist, rule_name)

    # CSV export (AC #5)
    csv_str = export_bracket_csv(
        sim_data.bracket,
        sim_data.most_likely,
        sim_data.team_labels,
        sim_data.prob_matrix,
    )
    st.download_button(
        label="Download Bracket CSV",
        data=csv_str,
        file_name="bracket_submission.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------


def _render_scoring_config() -> tuple[bool, tuple[float, ...]]:
    """Render scoring configuration UI and return (use_custom, custom_points)."""
    st.subheader("Scoring Configuration")
    use_custom = st.checkbox("Use custom scoring", value=False, key="pool_use_custom")

    custom_points: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
    if use_custom:
        cols = st.columns(6)
        round_labels = ["R64", "R32", "S16", "E8", "F4", "NCG"]
        custom_values: list[float] = []
        defaults = [1, 2, 4, 8, 16, 32]
        for i, (col, label) in enumerate(zip(cols, round_labels)):
            with col:
                val = st.number_input(
                    label,
                    min_value=0,
                    value=defaults[i],
                    step=1,
                    key=f"pool_pts_{i}",
                )
                custom_values.append(float(val))
        custom_points = tuple(custom_values)

    return use_custom, custom_points


def _run_simulation(
    data_dir: str,
    run_id: str,
    season: int,
    scoring: str,
    n_sims: int,
) -> None:
    """Run MC simulation and store result in session state."""
    with st.spinner("Running Monte Carlo simulation..."):
        sim_data = run_bracket_simulation(
            data_dir=data_dir,
            run_id=run_id,
            season=season,
            scoring_name=scoring,
            method="monte_carlo",
            n_simulations=n_sims,
        )

    if sim_data is None:
        st.warning(
            "Could not simulate bracket. Ensure the selected model has been "
            f"trained and has data for {season}."
        )
        return

    if sim_data.sim_result.sim_winners is None:
        st.error("Monte Carlo simulation did not produce sim_winners. Cannot score bracket.")
        return

    st.session_state["pool_sim_data"] = sim_data
    st.session_state["pool_sim_key"] = (run_id, season, n_sims)


def _render_pool_scorer_page() -> None:
    """Render the Pool Scorer page."""
    # Breadcrumbs (AC matching existing pages)
    col_nav, col_bc = st.columns([1, 3])
    with col_nav:
        st.page_link("pages/home.py", label="← Home")
    with col_bc:
        st.caption("Home > Presentation > Pool Scorer")

    st.header("Pool Scorer & Point Outcome Analysis")

    # Validate required session state (empty states — AC #2.8)
    selected_year: int | None = st.session_state.get("selected_year")
    selected_run_id: str | None = st.session_state.get("selected_run_id")
    selected_scoring: str | None = st.session_state.get("selected_scoring")

    if selected_run_id is None:
        st.info("Select a model run from the sidebar to analyze pool scoring.")
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

    # Scoring configuration (AC #1)
    use_custom, custom_points = _render_scoring_config()

    # MC simulation settings
    n_sims = st.slider(
        "Number of Simulations",
        min_value=1_000,
        max_value=100_000,
        value=10_000,
        step=1_000,
        key="pool_n_sims",
    )

    # Analyze Outcomes button (AC #2, #4)
    if st.button("Analyze Outcomes", type="primary", key="pool_analyze"):
        _run_simulation(data_dir, selected_run_id, selected_year, scoring, n_sims)

    # Render results if simulation data is available
    sim_data_cached: BracketSimulationResult | None = st.session_state.get("pool_sim_data")
    cached_key = st.session_state.get("pool_sim_key")

    if sim_data_cached is not None and cached_key == (selected_run_id, selected_year, n_sims):
        _render_results(sim_data_cached, scoring, use_custom, custom_points)
    elif sim_data_cached is not None:
        st.info("Simulation parameters changed. Click **Analyze Outcomes** to re-run.")


_render_pool_scorer_page()
