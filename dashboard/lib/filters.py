"""Shared data-loading and filter helpers for the dashboard.

All data access goes through ``ncaa_eval`` public APIs — no direct file IO.
Functions are decorated with ``@st.cache_data`` so repeated calls across page
navigations hit the in-memory cache.
"""

from __future__ import annotations

from pathlib import Path

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
        summaries = store.load_all_summaries()
        if summaries.empty:
            return []
        runs = {r.run_id: r for r in store.list_runs()}
        rows: list[dict[str, object]] = []
        for _, row in summaries.iterrows():
            run_id = str(row["run_id"])
            meta = runs.get(run_id)
            rows.append(
                {
                    "run_id": run_id,
                    "model_type": meta.model_type if meta else "unknown",
                    "timestamp": str(meta.timestamp) if meta else "",
                    "start_year": meta.start_year if meta else 0,
                    "end_year": meta.end_year if meta else 0,
                    "year": int(row["year"]),
                    "log_loss": float(row["log_loss"]),
                    "brier_score": float(row["brier_score"]),
                    "roc_auc": float(row["roc_auc"]),
                    "ece": float(row["ece"]),
                }
            )
        return rows
    except OSError:
        return []


@st.cache_data(ttl=None)
def load_available_scorings() -> list[str]:
    """Return registered scoring-rule names.

    Returns:
        Sorted list of scoring-format names (e.g. ``["fibonacci", "standard", …]``).
    """
    return list_scorings()
