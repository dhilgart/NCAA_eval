"""Shared data-loading and filter helpers for the dashboard.

All data access goes through ``ncaa_eval`` public APIs — no direct file IO.
Functions are decorated with ``@st.cache_data`` so repeated calls across page
navigations hit the in-memory cache.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd  # type: ignore[import-untyped]
import streamlit as st

from ncaa_eval.evaluation import list_scorings
from ncaa_eval.ingest.repository import ParquetRepository
from ncaa_eval.model.tracking import RunStore


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
