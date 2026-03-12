"""Presentation page — Bracket Visualizer.

Interactive bracket visualizer showing per-game win probabilities and team
advancement odds from a trained model.  Renders a 64-team bracket tree,
advancement probability heatmap, expected-points table, and optional Monte
Carlo score distribution.  Users can override individual game picks via
an interactive "Edit Picks" section.
"""

from __future__ import annotations

import math
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from dashboard.lib.bracket_overrides import (
    _get_participants,
    apply_overrides,
    check_invalidation,
    clear_overrides,
    get_overrides,
    set_override,
)
from dashboard.lib.bracket_renderer import render_bracket_html
from dashboard.lib.data_loaders import get_data_dir, load_scoring_display_names, load_tourney_seeds
from dashboard.lib.simulation_helpers import (
    BracketSimulationResult,
    run_bracket_simulation,
    run_bracket_simulation_with_progress,
)
from ncaa_eval.evaluation.plotting import (
    plot_advancement_heatmap,
    plot_score_distribution,
)
from ncaa_eval.evaluation.simulation import MostLikelyBracket


@st.cache_data(ttl=None, show_spinner="Generating Kaggle submission...")
def _build_kaggle_csv(data_dir: str, run_id: str, season: int) -> str | None:
    """Build a full Kaggle submission CSV for all teams in the season.

    Returns ``None`` if the model type is not supported (non-Elo).
    """
    from ncaa_eval.cli.export import build_kaggle_submission

    try:
        return build_kaggle_submission(run_id=run_id, season=season, data_dir=Path(data_dir))
    except (FileNotFoundError, TypeError):
        return None


def _render_results(  # noqa: PLR0912, C901
    sim_data: BracketSimulationResult,
    scoring: str,
    edited_bracket: MostLikelyBracket,
    overrides: dict[int, int],
) -> None:
    """Render all bracket visualisation sections from simulation results."""
    display_names = load_scoring_display_names()
    scoring_label = display_names.get(scoring, scoring)
    result = sim_data.sim_result
    bracket = sim_data.bracket

    # Use the edited bracket (which equals model bracket when no overrides)
    most_likely = edited_bracket

    # Override status display
    if overrides:
        n_games = len(most_likely.winners)
        st.info(f"{len(overrides)} of {n_games} picks overridden by user.")

    # Champion summary
    champ_label = sim_data.team_labels.get(
        bracket.team_index_map.get(most_likely.champion_team_id, -1), "Unknown"
    )
    st.success(f"Predicted Champion: **{champ_label}** (log-likelihood: {most_likely.log_likelihood:.2f})")

    # Bracket tree
    st.subheader("Bracket" if not overrides else "Bracket (User-Edited)")
    bracket_html = render_bracket_html(
        bracket_team_ids=bracket.team_ids,
        most_likely_winners=most_likely.winners,
        team_labels=sim_data.team_labels,
        seed_map=bracket.seed_map,
        prob_matrix=sim_data.prob_matrix,
        overridden_games=frozenset(overrides) if overrides else None,
    )
    components.html(bracket_html, height=750, scrolling=True)

    # Advancement heatmap
    st.subheader("Advancement Probabilities")
    fig_heatmap = plot_advancement_heatmap(result, team_labels=sim_data.team_labels)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # Team Detail Expansion — pairwise win probabilities (AC #5)
    with st.expander("Pairwise Win Probabilities", expanded=False):
        st.caption("Select two teams to see the head-to-head win probability and seed matchup.")
        team_options = [sim_data.team_labels[i] for i in range(len(bracket.team_ids))]
        col_a, col_b = st.columns(2)
        with col_a:
            team_a_label = st.selectbox("Team A", options=team_options, key="pairwise_team_a")
        with col_b:
            team_b_label = st.selectbox(
                "Team B",
                options=[t for t in team_options if t != team_a_label],
                key="pairwise_team_b",
            )
        if team_a_label and team_b_label:
            idx_a = team_options.index(team_a_label)
            idx_b = team_options.index(team_b_label)
            tid_a = bracket.team_ids[idx_a]
            tid_b = bracket.team_ids[idx_b]
            seed_a = bracket.seed_map.get(tid_a, 0)
            seed_b = bracket.seed_map.get(tid_b, 0)
            prob_a_beats_b = float(sim_data.prob_matrix[idx_a, idx_b])
            st.metric(
                label=f"{team_a_label} (#{seed_a}) vs {team_b_label} (#{seed_b})",
                value=f"{prob_a_beats_b:.1%}",
                delta=f"{prob_a_beats_b - 0.5:+.1%} vs. 50%",
            )

    # Expected points table
    st.subheader(f"Expected Points ({scoring_label})")
    if scoring in result.expected_points:
        ep = result.expected_points[scoring]
        ep_data: list[dict[str, str | float]] = []
        for idx in range(len(bracket.team_ids)):
            team_id = bracket.team_ids[idx]
            label = sim_data.team_labels.get(idx, str(team_id))
            ep_data.append({"Team": label, "Expected Points": round(float(ep[idx]), 2)})
        ep_data.sort(key=lambda d: float(d["Expected Points"]), reverse=True)
        st.dataframe(ep_data, use_container_width=True, height=400)
    else:
        st.info("Expected points not available for the selected scoring rule.")

    # Score distribution (MC only)
    if result.method == "monte_carlo" and result.bracket_distributions:
        st.subheader("Score Distribution (Monte Carlo)")
        if scoring in result.bracket_distributions:
            dist = result.bracket_distributions[scoring]
            fig_dist = plot_score_distribution(dist, title=f"Bracket Score Distribution — {scoring_label}")
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info(f"Score distribution not available for scoring rule '{scoring}'.")

    # Kaggle submission export
    st.subheader("Export")
    selected_year: int | None = st.session_state.get("selected_year")
    selected_run_id: str | None = st.session_state.get("selected_run_id")
    if selected_year and selected_run_id:
        data_dir = str(get_data_dir())
        kaggle_csv = _build_kaggle_csv(data_dir, selected_run_id, selected_year)
        if kaggle_csv is not None:
            st.download_button(
                label="Export Kaggle Submission",
                data=kaggle_csv,
                file_name=f"submission_{selected_year}_{selected_run_id[:8]}.csv",
                mime="text/csv",
                key="kaggle_export_btn",
            )
        else:
            st.info("Kaggle export is available for Elo models only.")


_ROUND_LABELS: tuple[str, ...] = ("R64", "R32", "S16", "E8", "F4", "Championship")


def _render_edit_picks(  # noqa: C901
    sim_data: BracketSimulationResult,
    edited_bracket: MostLikelyBracket,
    overrides: dict[int, int],
) -> None:
    """Render the interactive Edit Picks section with selectboxes per game."""
    n_games = len(edited_bracket.winners)
    n_teams = n_games + 1
    n_rounds = int(math.log2(n_teams))

    with st.expander("Edit Picks", expanded=bool(overrides)):
        st.caption(
            "Override model predictions by selecting a different winner for any matchup. "
            "Changes cascade through later rounds automatically."
        )

        offset = 0
        games_in_round = n_teams // 2
        for round_idx in range(n_rounds):
            round_label = _ROUND_LABELS[round_idx] if round_idx < len(_ROUND_LABELS) else f"R{round_idx}"
            st.markdown(f"**{round_label}** ({games_in_round} games)")

            cols_per_row = min(4, games_in_round)
            for row_start in range(0, games_in_round, cols_per_row):
                row_end = min(row_start + cols_per_row, games_in_round)
                cols = st.columns(row_end - row_start)
                for col_idx, g in enumerate(range(row_start, row_end)):
                    game_idx = offset + g
                    team_a, team_b = _get_participants(game_idx, edited_bracket.winners, n_teams)
                    label_a = sim_data.team_labels.get(team_a, str(team_a))
                    label_b = sim_data.team_labels.get(team_b, str(team_b))

                    current_winner = edited_bracket.winners[game_idx]
                    model_winner = sim_data.most_likely.winners[game_idx]

                    options = [label_a, label_b]
                    team_indices = [team_a, team_b]
                    current_idx = team_indices.index(current_winner) if current_winner in team_indices else 0

                    # Show model's pick if different from current
                    model_label = sim_data.team_labels.get(model_winner, str(model_winner))
                    help_text = f"Model: {model_label}" if game_idx in overrides else None

                    with cols[col_idx]:
                        selected = st.selectbox(
                            f"Game {game_idx + 1}",
                            options=options,
                            index=current_idx,
                            key=f"edit_game_{game_idx}",
                            help=help_text,
                        )

                        if selected is not None:
                            sel_idx = options.index(selected)
                            selected_team = team_indices[sel_idx]
                            # Compare against the current cascaded winner (not model winner)
                            # to avoid spurious overrides when a cascade already produced
                            # a different winner from the model prediction.
                            current_winner_for_game = edited_bracket.winners[game_idx]
                            if selected_team != current_winner_for_game:
                                if selected_team != model_winner:
                                    # User explicitly chose a non-model winner
                                    set_override(game_idx, selected_team)
                                else:
                                    # User chose model winner — remove any existing override
                                    current_overrides = get_overrides()
                                    current_overrides.pop(game_idx, None)
                                    st.session_state["bracket_overrides"] = current_overrides
                                st.rerun()
                            elif game_idx in overrides and selected_team == model_winner:
                                # Current cascaded winner matches selection and equals model —
                                # stale override entry should be removed
                                current_overrides = get_overrides()
                                del current_overrides[game_idx]
                                st.session_state["bracket_overrides"] = current_overrides
                                st.rerun()

            offset += games_in_round
            games_in_round //= 2


def _render_bracket_page() -> None:  # noqa: C901
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
            f"No tournament seeds available for {selected_year}. Run `python sync.py` to download data."
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

    # Game Theory Sliders
    st.subheader("Game Theory Sliders")
    slider_col1, slider_col2, slider_col3 = st.columns([2, 2, 1])
    with slider_col1:
        upset_aggression: int = st.slider(
            "Upset Aggression",
            min_value=-5,
            max_value=5,
            value=0,
            step=1,
            help="Chalk \u2190 \u2192 Chaos",
            key="bracket_upset_aggression",
        )
    with slider_col2:
        seed_weight_pct: int = st.slider(
            "Seed-Weight",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            format="%d%%",
            help="Model \u2190 \u2192 Seeds",
            key="bracket_seed_weight",
        )
    with slider_col3:
        if st.button("Reset Sliders", key="reset_sliders"):
            st.session_state["bracket_upset_aggression"] = 0
            st.session_state["bracket_seed_weight"] = 0
            st.rerun()

    # Run simulation
    sim_data: BracketSimulationResult | None
    if method == "monte_carlo":
        progress_bar = st.progress(0, text="Running Monte Carlo simulation...")
        try:
            sim_data = run_bracket_simulation_with_progress(
                data_dir=data_dir,
                run_id=selected_run_id,
                season=selected_year,
                scoring_name=scoring,
                n_simulations=n_sims,
                progress_bar=progress_bar,
                upset_aggression=upset_aggression,
                seed_weight_pct=seed_weight_pct,
            )
        finally:
            progress_bar.empty()
    else:
        with st.spinner("Computing bracket..."):
            sim_data = run_bracket_simulation(
                data_dir=data_dir,
                run_id=selected_run_id,
                season=selected_year,
                scoring_name=scoring,
                method="analytical",
                n_simulations=n_sims,
                upset_aggression=upset_aggression,
                seed_weight_pct=seed_weight_pct,
            )

    if sim_data is None:
        st.warning(
            "Could not simulate bracket. Ensure the selected model has been trained "
            f"and has data for {selected_year}."
        )
        return

    # Override invalidation check (AC #4)
    if check_invalidation(selected_run_id, selected_year, scoring, upset_aggression, seed_weight_pct):
        st.info("Bracket parameters changed — user overrides have been reset.")

    # Apply overrides (AC #1)
    overrides = get_overrides()
    edited_bracket = apply_overrides(sim_data.most_likely, overrides, sim_data.bracket, sim_data.prob_matrix)

    # Reset button (AC #3)
    if overrides:
        if st.button("Reset to Model Predictions", key="reset_overrides"):
            clear_overrides()
            st.rerun()

    _render_results(sim_data, scoring, edited_bracket, overrides)

    # Interactive edit picks section (AC #1)
    _render_edit_picks(sim_data, edited_bracket, overrides)


_render_bracket_page()
