"""Bracket simulation orchestration for the dashboard.

Provides :func:`run_bracket_simulation` and supporting helpers that build
probability providers, team labels, and run tournament simulations.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import streamlit as st

from dashboard.lib.data_loaders import _load_team_names_uncached
from ncaa_eval.evaluation.perturbation import (
    perturb_probability_matrix,
    slider_to_temperature,
)
from ncaa_eval.evaluation.providers import EnsembleProvider as _EnsembleProvider
from ncaa_eval.evaluation.simulation import (
    BracketStructure,
    EloProvider,
    MatchupContext,
    MatrixProvider,
    MostLikelyBracket,
    SimulationResult,
    build_bracket,
    build_probability_matrix,
    compute_most_likely_bracket,
    get_scoring,
    simulate_tournament,
    simulate_tournament_mc,
)
from ncaa_eval.model.ensemble import StackedEnsemble
from ncaa_eval.model.tracking import RunStore
from ncaa_eval.transform.normalization import TourneySeedTable

logger = logging.getLogger(__name__)

# Kaggle day-number for the Round of 64 (first day of the NCAA tournament).
# Used as the MatchupContext day_num when running bracket simulations.
_ROUND_OF_64_DAY_NUM: int = 136


@dataclass(frozen=True)
class BracketSimulationResult:
    """Container for bracket simulation outputs (cache-friendly).

    Attributes:
        sim_result: Full simulation result with advancement probs and EP.
        bracket: The bracket structure with team ordering and seed map.
        most_likely: Greedy most-likely bracket picks.
        prob_matrix: Pairwise win probability matrix.
        team_labels: Mapping of bracket index → display label.
    """

    sim_result: SimulationResult
    bracket: BracketStructure
    most_likely: MostLikelyBracket
    prob_matrix: npt.NDArray[np.float64]
    team_labels: dict[int, str]


def _build_provider_from_folds(
    store: RunStore,
    run_id: str,
    season: int,
    bracket: BracketStructure,
) -> MatrixProvider | None:
    """Build a MatrixProvider from fold predictions for stateless models.

    Initialises an n×n probability matrix filled with 0.5 (neutral prior
    for any matchup not covered by backtest predictions), then does a
    vectorised fill: maps ``team_a_id`` / ``team_b_id`` columns to bracket
    indices via ``bracket.team_index_map``, drops rows where either team
    is not in the bracket, and assigns ``P[a, b] = pred_win_prob`` and
    the complement ``P[b, a] = 1 - pred_win_prob`` in one NumPy step.

    Returns ``None`` if fold predictions are missing or empty for the season.
    """
    fold_df = store.load_fold_predictions(run_id)
    if fold_df is None or fold_df.empty:
        logger.warning("No fold predictions for run %s", run_id)
        return None

    tourney_preds = fold_df[fold_df["year"] == season]
    if tourney_preds.empty:
        logger.warning("No predictions for season %d in run %s", season, run_id)
        return None

    n = len(bracket.team_ids)
    P = np.full((n, n), 0.5, dtype=np.float64)

    # Vectorized matrix fill: map team IDs to bracket indices, then assign
    a_indices = tourney_preds["team_a_id"].map(bracket.team_index_map)
    b_indices = tourney_preds["team_b_id"].map(bracket.team_index_map)
    valid = a_indices.notna() & b_indices.notna()
    a_valid = a_indices[valid].astype(int).to_numpy()
    b_valid = b_indices[valid].astype(int).to_numpy()
    probs = tourney_preds.loc[valid, "pred_win_prob"].to_numpy(dtype=np.float64)
    P[a_valid, b_valid] = probs
    P[b_valid, a_valid] = 1.0 - probs
    return MatrixProvider(P, list(bracket.team_ids))


def _build_team_labels(
    data_dir: str,
    bracket: BracketStructure,
) -> dict[int, str]:
    """Build bracket_index → ``"[seed] TeamName"`` label mapping."""
    team_names = _load_team_names_uncached(data_dir)
    labels: dict[int, str] = {}
    for team_id, idx in bracket.team_index_map.items():
        seed_num = bracket.seed_map.get(team_id, 0)
        name = team_names.get(team_id, str(team_id))
        labels[idx] = f"[{seed_num}] {name}"
    return labels


def _build_probability_provider(  # noqa: PLR0913
    store: RunStore,
    run_id: str,
    run: object,
    model: object,
    path: Path,
    season: int,
    bracket: BracketStructure,
) -> EloProvider | MatrixProvider | _EnsembleProvider | None:
    """Select and construct the appropriate probability provider for a model run.

    Returns ``None`` if the provider cannot be built (e.g. no fold predictions
    for non-Elo/non-ensemble models, or unexpected model type for ensemble runs).
    """
    model_type: str = getattr(run, "model_type", "")
    if model_type == "elo":
        return EloProvider(model)  # type: ignore[arg-type]
    if model_type == "ensemble":
        if not isinstance(model, StackedEnsemble):
            logger.warning("Expected StackedEnsemble for ensemble run %s, got %s", run_id, type(model))
            return None
        return _EnsembleProvider(model, path, season)
    mp = _build_provider_from_folds(store, run_id, season, bracket)
    return mp


@st.cache_data(ttl=None)
def run_bracket_simulation(  # noqa: PLR0913
    data_dir: str,
    run_id: str,
    season: int,
    scoring_name: str,
    method: str = "analytical",
    n_simulations: int = 10_000,
    upset_aggression: int = 0,
    seed_weight_pct: int = 0,
) -> BracketSimulationResult | None:
    """Orchestrate bracket construction, model loading, and simulation.

    Builds a 64-team bracket from seeds, loads the trained model, computes
    pairwise probabilities, and runs the tournament simulation.  When
    game-theory sliders are non-neutral, the analytical path (advancement
    probabilities, expected points, most-likely bracket) uses perturbed
    probabilities while Monte Carlo simulation uses the original model.

    Args:
        data_dir: String path to the project data directory.
        run_id: Model run identifier.
        season: Tournament season year.
        scoring_name: Scoring rule name (e.g. ``"standard"``).
        method: ``"analytical"`` or ``"monte_carlo"``.
        n_simulations: Number of MC simulations (ignored for analytical).
        upset_aggression: Upset Aggression slider value in ``[-5, +5]``.
        seed_weight_pct: Seed-Weight slider value in ``[0, 100]``.

    Returns:
        :class:`BracketSimulationResult` or ``None`` on failure.
    """
    path = Path(data_dir)
    csv_path = path / "kaggle" / "MNCAATourneySeeds.csv"
    if not csv_path.exists():
        logger.warning("Seed CSV not found: %s", csv_path)
        return None

    try:
        # Load seeds and build bracket
        seed_table = TourneySeedTable.from_csv(csv_path)
        seeds = seed_table.all_seeds(season)
        if not seeds:
            logger.warning("No seeds found for season %d", season)
            return None
        bracket = build_bracket(seeds, season)

        # Load model and create probability provider
        store = RunStore(path)
        run = store.load_run(run_id)
        model = store.load_model(run_id)
        if model is None:
            logger.warning("No model found for run %s", run_id)
            return None

        provider = _build_probability_provider(store, run_id, run, model, path, season, bracket)
        if provider is None:
            return None

        team_labels = _build_team_labels(data_dir, bracket)

        # Get scoring rule — detect constructor needs via inspection to avoid
        # hardcoded name checks when new scoring rules are added in future.
        scoring_cls = get_scoring(scoring_name)
        sig = inspect.signature(scoring_cls)
        if "seed_map" in sig.parameters:
            scoring_rule = scoring_cls(bracket.seed_map)
        else:
            scoring_rule = scoring_cls()

        # Context for tournament games (neutral site, R64 tip-off day)
        context = MatchupContext(season=season, day_num=_ROUND_OF_64_DAY_NUM, is_neutral=True)

        # Build the base probability matrix
        prob_matrix = build_probability_matrix(provider, bracket.team_ids, context)

        # Apply game-theory slider perturbation
        temperature = slider_to_temperature(upset_aggression)
        perturbed_matrix = perturb_probability_matrix(
            prob_matrix,
            bracket.seed_map,
            bracket.team_ids,
            temperature=temperature,
            seed_weight=seed_weight_pct / 100.0,
        )

        # Analytical simulation uses PERTURBED probabilities
        perturbed_provider = MatrixProvider(perturbed_matrix, list(bracket.team_ids))
        sim_result = simulate_tournament(
            bracket=bracket,
            probability_provider=perturbed_provider,
            context=context,
            scoring_rules=[scoring_rule],
            method="analytical",
        )

        # MC simulation uses ORIGINAL provider (unperturbed "reality")
        if method == "monte_carlo":
            mc_result = simulate_tournament(
                bracket=bracket,
                probability_provider=provider,
                context=context,
                scoring_rules=[scoring_rule],
                method="monte_carlo",
                n_simulations=n_simulations,
            )
            # Merge MC fields into the analytical result
            sim_result = SimulationResult(
                season=sim_result.season,
                advancement_probs=sim_result.advancement_probs,
                expected_points=sim_result.expected_points,
                method="monte_carlo",
                n_simulations=mc_result.n_simulations,
                confidence_intervals=mc_result.confidence_intervals,
                score_distribution=mc_result.score_distribution,
                bracket_distributions=mc_result.bracket_distributions,
                sim_winners=mc_result.sim_winners,
            )

        most_likely = compute_most_likely_bracket(bracket, perturbed_matrix)

        return BracketSimulationResult(
            sim_result=sim_result,
            bracket=bracket,
            most_likely=most_likely,
            prob_matrix=perturbed_matrix,
            team_labels=team_labels,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.exception("Bracket simulation failed: %s", exc)
        return None


def run_bracket_simulation_with_progress(  # noqa: PLR0913
    data_dir: str,
    run_id: str,
    season: int,
    scoring_name: str,
    n_simulations: int,
    progress_bar: st.delta_generator.DeltaGenerator,
    upset_aggression: int = 0,
    seed_weight_pct: int = 0,
) -> BracketSimulationResult | None:
    """Run MC bracket simulation with ``st.progress`` updates.

    NOT cached — progress bars require the function body to execute on every
    call.  Calls :func:`run_bracket_simulation` (cached, analytical) for
    bracket/label/most-likely setup, then re-loads the probability provider
    to build an **unperturbed** matrix for MC simulation.

    The re-load is necessary because :class:`BracketSimulationResult` only
    exposes the perturbed matrix (used for analytical EP and most-likely
    bracket picks).  MC simulation must use the raw model probabilities so
    that each simulated outcome reflects true model uncertainty rather than
    the user's game-theory slider adjustments.

    Args:
        data_dir: String path to the project data directory.
        run_id: Model run identifier.
        season: Tournament season year.
        scoring_name: Scoring rule name (e.g. ``"standard"``).
        n_simulations: Number of MC simulations.
        progress_bar: A Streamlit ``st.progress`` bar object to update.
        upset_aggression: Upset Aggression slider value in ``[-5, +5]``.
        seed_weight_pct: Seed-Weight slider value in ``[0, 100]``.

    Returns:
        :class:`BracketSimulationResult` or ``None`` on failure.
    """
    # Get the analytical result (cached) for bracket setup
    analytical = run_bracket_simulation(
        data_dir=data_dir,
        run_id=run_id,
        season=season,
        scoring_name=scoring_name,
        method="analytical",
        n_simulations=n_simulations,
        upset_aggression=upset_aggression,
        seed_weight_pct=seed_weight_pct,
    )
    if analytical is None:
        return None

    try:
        bracket = analytical.bracket
        scoring_cls = get_scoring(scoring_name)
        sig = inspect.signature(scoring_cls)
        if "seed_map" in sig.parameters:
            scoring_rule = scoring_cls(bracket.seed_map)
        else:
            scoring_rule = scoring_cls()

        context = MatchupContext(season=season, day_num=_ROUND_OF_64_DAY_NUM, is_neutral=True)

        # Build the UNPERTURBED probability matrix for MC (original model probs).
        # The cached analytical result contains only the perturbed matrix (used for
        # analytical EP and most-likely bracket).  MC simulation must use the raw model
        # probabilities, so we re-load the provider here to build an unperturbed P.
        path = Path(data_dir)
        store = RunStore(path)
        run = store.load_run(run_id)
        model = store.load_model(run_id)
        if model is None:
            return None

        provider = _build_probability_provider(store, run_id, run, model, path, season, bracket)
        if provider is None:
            return None

        P = build_probability_matrix(provider, bracket.team_ids, context)

        def _update_progress(round_completed: int, total_rounds: int) -> None:
            progress_bar.progress(
                round_completed / total_rounds,
                text=f"Simulating round {round_completed}/{total_rounds}...",
            )

        mc_result = simulate_tournament_mc(
            bracket=bracket,
            P=P,
            scoring_rules=[scoring_rule],
            season=season,
            n_simulations=n_simulations,
            progress_callback=_update_progress,
        )

        # Merge MC fields into the analytical result (same pattern as cached function)
        merged = SimulationResult(
            season=analytical.sim_result.season,
            advancement_probs=analytical.sim_result.advancement_probs,
            expected_points=analytical.sim_result.expected_points,
            method="monte_carlo",
            n_simulations=mc_result.n_simulations,
            confidence_intervals=mc_result.confidence_intervals,
            score_distribution=mc_result.score_distribution,
            bracket_distributions=mc_result.bracket_distributions,
            sim_winners=mc_result.sim_winners,
        )

        return BracketSimulationResult(
            sim_result=merged,
            bracket=analytical.bracket,
            most_likely=analytical.most_likely,
            prob_matrix=analytical.prob_matrix,
            team_labels=analytical.team_labels,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.exception("MC simulation with progress failed: %s", exc)
        return None
