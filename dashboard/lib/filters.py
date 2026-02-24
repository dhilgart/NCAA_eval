"""Shared data-loading and filter helpers for the dashboard.

All data access goes through ``ncaa_eval`` public APIs — no direct file IO.
Functions are decorated with ``@st.cache_data`` so repeated calls across page
navigations hit the in-memory cache.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]
import streamlit as st

from ncaa_eval.evaluation import list_scorings
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
)
from ncaa_eval.ingest.repository import ParquetRepository
from ncaa_eval.model.tracking import RunStore
from ncaa_eval.transform.normalization import TourneySeedTable

logger = logging.getLogger(__name__)


def get_data_dir() -> Path:
    """Resolve the project ``data/`` directory."""
    return Path(__file__).resolve().parent.parent.parent / "data"


@st.cache_data(ttl=300)
def load_available_years(data_dir: str) -> list[int]:
    """Return sorted list of available season years.

    Args:
        data_dir: String path to the project data directory.

    Returns:
        Descending-sorted list of season years, or empty list if the data
        directory does not exist or cannot be read.
    """
    path = Path(data_dir)
    if not path.exists():
        return []
    try:
        repo = ParquetRepository(path)
        seasons = repo.get_seasons()
        return sorted((s.year for s in seasons), reverse=True)
    except OSError:
        return []


@st.cache_data(ttl=300)
def load_available_runs(data_dir: str) -> list[dict[str, object]]:
    """Return serialised metadata for every saved model run.

    Args:
        data_dir: String path to the project data directory.

    Returns:
        List of dicts (one per run), serialised via ``ModelRun.model_dump()``,
        or empty list if the data directory does not exist or cannot be read.
    """
    path = Path(data_dir)
    if not path.exists():
        return []
    try:
        store = RunStore(path)
        return [run.model_dump() for run in store.list_runs()]
    except OSError:
        return []


@st.cache_data(ttl=300)
def load_leaderboard_data(data_dir: str) -> list[dict[str, object]]:
    """Load leaderboard data: run metadata joined with metric summaries.

    Args:
        data_dir: String path to the project data directory.

    Returns:
        List of dicts (serializable for st.cache_data) with keys:
        run_id, model_type, timestamp, start_year, end_year, year,
        log_loss, brier_score, roc_auc, ece.
    """
    path = Path(data_dir)
    if not path.exists():
        return []
    try:
        store = RunStore(path)
        runs = store.list_runs()
        summaries = store.load_all_summaries()
        if summaries.empty:
            return []
        runs_meta = pd.DataFrame(
            [
                {
                    "run_id": r.run_id,
                    "model_type": r.model_type,
                    "timestamp": str(r.timestamp),
                    "start_year": r.start_year,
                    "end_year": r.end_year,
                }
                for r in runs
            ]
        )
        if runs_meta.empty:
            return []
        _keep = [
            "run_id",
            "model_type",
            "timestamp",
            "start_year",
            "end_year",
            "year",
            "log_loss",
            "brier_score",
            "roc_auc",
            "ece",
        ]
        merged = summaries.merge(runs_meta, on="run_id", how="left")
        return cast(list[dict[str, object]], merged[_keep].to_dict("records"))
    except OSError:
        return []


@st.cache_data(ttl=300)
def load_fold_predictions(data_dir: str, run_id: str) -> list[dict[str, object]]:
    """Load fold-level CV predictions for a run.

    Args:
        data_dir: String path to the project data directory.
        run_id: The model run identifier.

    Returns:
        List of dicts with keys [year, game_id, team_a_id, team_b_id,
        pred_win_prob, team_a_won], or empty list if unavailable.
    """
    path = Path(data_dir)
    if not path.exists():
        return []
    try:
        store = RunStore(path)
        df = store.load_fold_predictions(run_id)
        if df is None:
            return []
        return cast(list[dict[str, object]], df.to_dict("records"))
    except OSError:
        return []


@st.cache_data(ttl=300)
def load_feature_importances(data_dir: str, run_id: str) -> list[dict[str, object]]:
    """Load feature importances for a run (XGBoost only).

    Args:
        data_dir: String path to the project data directory.
        run_id: The model run identifier.

    Returns:
        List of dicts ``{"feature": name, "importance": value}`` sorted
        descending by importance. Empty list for non-XGBoost models,
        legacy runs, or errors.
    """
    path = Path(data_dir)
    if not path.exists():
        return []
    try:
        store = RunStore(path)
        run = store.load_run(run_id)
        if run.model_type != "xgboost":
            return []
        model = store.load_model(run_id)
        if model is None:
            return []
        feature_names = store.load_feature_names(run_id) or []
        clf = getattr(model, "_clf", None)
        importances = getattr(clf, "feature_importances_", None)
        if importances is None or not feature_names or len(feature_names) != len(importances):
            return []
        pairs = sorted(
            zip(feature_names, importances.tolist()),
            key=lambda p: p[1],
            reverse=True,
        )
        return [{"feature": f, "importance": v} for f, v in pairs]
    except (OSError, KeyError):
        return []


@st.cache_data(ttl=None)
def load_available_scorings() -> list[str]:
    """Return registered scoring-rule names.

    Returns:
        Sorted list of scoring-format names (e.g. ``["fibonacci", "standard", …]``).
    """
    return list_scorings()


# ---------------------------------------------------------------------------
# Bracket Visualizer helpers (Story 7.5)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)
def load_tourney_seeds(data_dir: str, season: int) -> list[dict[str, object]]:
    """Load tournament seeds for a season from the Kaggle CSV.

    Args:
        data_dir: String path to the project data directory.
        season: Tournament season year.

    Returns:
        List of serialised seed dicts with keys: season, team_id, seed_str,
        region, seed_num, is_play_in.  Empty list if unavailable.
    """
    csv_path = Path(data_dir) / "kaggle" / "MNCAATourneySeeds.csv"
    if not csv_path.exists():
        return []
    try:
        table = TourneySeedTable.from_csv(csv_path)
        seeds = table.all_seeds(season)
        return [
            {
                "season": s.season,
                "team_id": s.team_id,
                "seed_str": s.seed_str,
                "region": s.region,
                "seed_num": s.seed_num,
                "is_play_in": s.is_play_in,
            }
            for s in seeds
        ]
    except (OSError, ValueError):
        return []


def _load_team_names_uncached(data_dir: str) -> dict[int, str]:
    """Load team ID → team name mapping (uncached internal helper).

    Args:
        data_dir: String path to the project data directory.

    Returns:
        Mapping of team_id to team_name.  Empty dict if unavailable.
    """
    path = Path(data_dir)
    if not path.exists():
        return {}
    try:
        repo = ParquetRepository(path)
        teams = repo.get_teams()
        return {t.team_id: t.team_name for t in teams}
    except OSError:
        return {}


@st.cache_data(ttl=300)
def load_team_names(data_dir: str) -> dict[int, str]:
    """Load team ID → team name mapping from the repository.

    Args:
        data_dir: String path to the project data directory.

    Returns:
        Mapping of team_id to team_name.  Empty dict if unavailable.
    """
    return _load_team_names_uncached(data_dir)


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
    for _, row in tourney_preds.iterrows():
        a_id = int(row["team_a_id"])
        b_id = int(row["team_b_id"])
        if a_id in bracket.team_index_map and b_id in bracket.team_index_map:
            i = bracket.team_index_map[a_id]
            j = bracket.team_index_map[b_id]
            prob = float(row["pred_win_prob"])
            P[i, j] = prob
            P[j, i] = 1.0 - prob
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


@st.cache_resource(ttl=None)
def run_bracket_simulation(  # noqa: PLR0913
    data_dir: str,
    run_id: str,
    season: int,
    scoring_name: str,
    method: str = "analytical",
    n_simulations: int = 10_000,
) -> BracketSimulationResult | None:
    """Orchestrate bracket construction, model loading, and simulation.

    Builds a 64-team bracket from seeds, loads the trained model, computes
    pairwise probabilities, and runs the tournament simulation.

    Args:
        data_dir: String path to the project data directory.
        run_id: Model run identifier.
        season: Tournament season year.
        scoring_name: Scoring rule name (e.g. ``"standard"``).
        method: ``"analytical"`` or ``"monte_carlo"``.
        n_simulations: Number of MC simulations (ignored for analytical).

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

        provider: EloProvider | MatrixProvider
        if run.model_type == "elo":
            provider = EloProvider(model)
        else:
            mp = _build_provider_from_folds(store, run_id, season, bracket)
            if mp is None:
                return None
            provider = mp

        team_labels = _build_team_labels(data_dir, bracket)

        # Get scoring rule
        scoring_cls = get_scoring(scoring_name)
        scoring_rule = scoring_cls(bracket.seed_map) if scoring_name == "seed_diff_bonus" else scoring_cls()

        # Context for tournament games (neutral site, day 136 = R64)
        context = MatchupContext(season=season, day_num=136, is_neutral=True)

        sim_result = simulate_tournament(
            bracket=bracket,
            probability_provider=provider,
            context=context,
            scoring_rules=[scoring_rule],
            method=method,
            n_simulations=n_simulations,
        )

        prob_matrix = build_probability_matrix(provider, bracket.team_ids, context)
        most_likely = compute_most_likely_bracket(bracket, prob_matrix)

        return BracketSimulationResult(
            sim_result=sim_result,
            bracket=bracket,
            most_likely=most_likely,
            prob_matrix=prob_matrix,
            team_labels=team_labels,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.exception("Bracket simulation failed: %s", exc)
        return None
