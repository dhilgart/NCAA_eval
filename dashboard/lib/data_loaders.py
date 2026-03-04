"""Cached data-loading helpers for the dashboard.

All data access goes through ``ncaa_eval`` public APIs — no direct file IO.
Functions are decorated with ``@st.cache_data`` so repeated calls across
page navigations hit the in-memory cache.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import cast

import pandas as pd  # type: ignore[import-untyped]
import streamlit as st

from ncaa_eval.evaluation import list_scoring_display_names, list_scorings
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

    Lists all runs from RunStore, loads the per-year metric summaries
    DataFrame, builds a run-metadata DataFrame from the run list, then
    merges the two on ``run_id`` (left join from summaries).  The merged
    result is restricted to a fixed column set before serialisation so
    that ``st.cache_data`` can hash it.

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
    """Load feature importances for a run.

    Uses the model's ``get_feature_importances()`` public API. Falls back
    to ``RunStore.load_feature_names`` paired with the model importances
    for legacy runs where ``get_feature_importances()`` returns ``None``.

    Args:
        data_dir: String path to the project data directory.
        run_id: The model run identifier.

    Returns:
        List of dicts ``{"feature": name, "importance": value}`` sorted
        descending by importance. Empty list for models without feature
        importances, legacy runs, or errors.
    """
    path = Path(data_dir)
    if not path.exists():
        return []
    try:
        store = RunStore(path)
        model = store.load_model(run_id)
        if model is None:
            return []
        raw = model.get_feature_importances()
        if raw is None:
            # Legacy fallback: model doesn't have feature names stored
            feature_names = store.load_feature_names(run_id) or []
            clf = getattr(model, "_clf", None)
            importances = getattr(clf, "feature_importances_", None)
            if importances is None or not feature_names or len(feature_names) != len(importances):
                return []
            raw = list(zip(feature_names, importances.tolist()))
        pairs = sorted(raw, key=lambda p: p[1], reverse=True)
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


@st.cache_data(ttl=None)
def load_scoring_display_names() -> dict[str, str]:
    """Return a mapping of scoring registry keys to display names.

    Returns:
        Dict mapping scoring name → display name (e.g. ``"fibonacci"`` → ``"Fibonacci (2-3-5-8-13-21)"``).
    """
    return list_scoring_display_names()


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


@st.cache_data(ttl=300)
def load_data_freshness(data_dir: str) -> dict[str, str | None]:
    """Return data freshness indicators for the sidebar.

    Args:
        data_dir: String path to the project data directory.

    Returns:
        Dict with ``last_sync_date`` (latest Parquet mtime as date string)
        and ``latest_game_date`` (max game date from the most recent season).
        Values are ``None`` when data is unavailable.
    """
    result: dict[str, str | None] = {"last_sync_date": None, "latest_game_date": None}
    path = Path(data_dir)
    if not path.exists():
        return result
    try:
        parquets = list(path.rglob("*.parquet"))
        if parquets:
            latest_mtime = max(p.stat().st_mtime for p in parquets)
            result["last_sync_date"] = datetime.datetime.fromtimestamp(
                latest_mtime, tz=datetime.timezone.utc
            ).strftime("%Y-%m-%d")
    except OSError:
        pass
    try:
        repo = ParquetRepository(path)
        seasons = repo.get_seasons()
        if seasons:
            max_year = max(s.year for s in seasons)
            games = repo.get_games(max_year)
            dates = [g.date for g in games if g.date is not None]
            if dates:
                result["latest_game_date"] = str(max(dates))
    except OSError:
        pass
    return result


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
