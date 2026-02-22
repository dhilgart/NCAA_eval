"""Batch opponent adjustment rating solvers: SRS, Ridge regression, Colley Matrix."""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

DEFAULT_MARGIN_CAP: int = 25
DEFAULT_RIDGE_LAMBDA: float = 20.0
DEFAULT_SRS_MAX_ITER: int = 10_000
_SRS_CONVERGENCE_TOL: float = 1e-6


class BatchRatingSolver:
    """Batch rating solver that produces full-season opponent-adjusted ratings.

    All solvers accept a pre-loaded DataFrame of compact regular-season games
    (caller must filter to ``is_tournament == False`` before passing).

    Args:
        margin_cap: Maximum point margin applied per game (default 25).
        ridge_lambda: Regularization strength for Ridge solver (default 20.0).
        srs_max_iter: Maximum iterations for SRS fixed-point convergence (default 10,000).
    """

    def __init__(
        self,
        *,
        margin_cap: int = DEFAULT_MARGIN_CAP,
        ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
        srs_max_iter: int = DEFAULT_SRS_MAX_ITER,
    ) -> None:
        self._margin_cap = margin_cap
        self._ridge_lambda = ridge_lambda
        self._srs_max_iter = srs_max_iter

    def compute_srs(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Compute SRS (Simple Rating System) ratings via fixed-point iteration.

        Args:
            games_df: DataFrame with columns ``w_team_id``, ``l_team_id``,
                ``w_score``, ``l_score`` (regular-season games only).

        Returns:
            DataFrame with columns ``["team_id", "srs_rating"]`` (zero-centered).
        """
        if games_df.empty:
            return pd.DataFrame(columns=["team_id", "srs_rating"])

        teams, idx, w_idx, l_idx = _build_team_index(games_df)
        n = len(teams)

        raw_margins = (games_df["w_score"] - games_df["l_score"]).to_numpy(dtype=float)
        margins = np.minimum(raw_margins, float(self._margin_cap))

        net_margin, n_games, avg_margin, A_norm = _build_srs_matrices(n, w_idx, l_idx, margins)

        # Fixed-point iteration: r = avg_margin + A_norm @ r
        r: npt.NDArray[np.float64] = np.zeros(n)
        for _ in range(self._srs_max_iter):
            r_new = avg_margin + A_norm @ r
            if float(np.max(np.abs(r_new - r))) < _SRS_CONVERGENCE_TOL:
                r = r_new
                break
            r = r_new

        # Zero-center ratings (enforces unique solution)
        r -= float(np.mean(r))

        return pd.DataFrame({"team_id": teams, "srs_rating": r.tolist()})

    def compute_ridge(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Compute Ridge regression ratings (regularized SRS).

        Args:
            games_df: DataFrame with columns ``w_team_id``, ``l_team_id``,
                ``w_score``, ``l_score`` (regular-season games only).

        Returns:
            DataFrame with columns ``["team_id", "ridge_rating"]``.
        """
        if games_df.empty:
            return pd.DataFrame(columns=["team_id", "ridge_rating"])

        teams, idx, w_idx, l_idx = _build_team_index(games_df)
        n = len(teams)
        n_games = len(games_df)

        X: npt.NDArray[np.float64] = np.zeros((n_games, n))
        raw_margins = (games_df["w_score"] - games_df["l_score"]).to_numpy(dtype=float)
        y = np.minimum(raw_margins, float(self._margin_cap))

        # Build design matrix (vectorized scatter)
        game_indices = np.arange(n_games)
        X[game_indices, w_idx] = 1.0
        X[game_indices, l_idx] = -1.0

        model = Ridge(alpha=self._ridge_lambda, fit_intercept=False)
        model.fit(X, y)
        ratings: list[float] = model.coef_.tolist()

        return pd.DataFrame({"team_id": teams, "ridge_rating": ratings})

    def compute_colley(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Compute Colley Matrix ratings (win/loss only, no margin).

        Args:
            games_df: DataFrame with columns ``w_team_id``, ``l_team_id``
                (regular-season games only; scores not used).

        Returns:
            DataFrame with columns ``["team_id", "colley_rating"]`` (bounded [0, 1]).
        """
        if games_df.empty:
            return pd.DataFrame(columns=["team_id", "colley_rating"])

        teams, idx, w_idx, l_idx = _build_team_index(games_df)
        n = len(teams)

        # Build C and b (vectorized)
        C: npt.NDArray[np.float64] = np.zeros((n, n))
        np.add.at(C, (w_idx, w_idx), 1.0)  # diagonal: games played
        np.add.at(C, (l_idx, l_idx), 1.0)
        np.add.at(C, (w_idx, l_idx), -1.0)  # off-diagonal: games between pair
        np.add.at(C, (l_idx, w_idx), -1.0)
        C += 2.0 * np.eye(n)  # add 2 to diagonal per Colley formulation

        wins: npt.NDArray[np.float64] = np.zeros(n)
        losses: npt.NDArray[np.float64] = np.zeros(n)
        np.add.at(wins, w_idx, 1.0)
        np.add.at(losses, l_idx, 1.0)
        b = 1.0 + (wins - losses) / 2.0

        # Solve C r = b
        try:
            r: npt.NDArray[np.float64] = np.linalg.solve(C, b)  # type: ignore[assignment]
        except np.linalg.LinAlgError:
            # Singular matrix (disconnected schedule) — fall back to lstsq
            logger.warning("Colley matrix is singular (disconnected schedule); using lstsq fallback")
            r, _, _, _ = np.linalg.lstsq(C, b, rcond=None)  # type: ignore[assignment]

        return pd.DataFrame({"team_id": teams, "colley_rating": r.tolist()})


def _build_team_index(
    games_df: pd.DataFrame,
) -> tuple[
    list[int],
    dict[int, int],
    npt.NDArray[np.intp],
    npt.NDArray[np.intp],
]:
    """Build sorted team list, index mapping, and vectorized index arrays."""
    teams: list[int] = sorted(set(games_df["w_team_id"].tolist()) | set(games_df["l_team_id"].tolist()))
    idx: dict[int, int] = {t: i for i, t in enumerate(teams)}
    w_idx: npt.NDArray[np.intp] = games_df["w_team_id"].map(idx).to_numpy(dtype=np.intp)
    l_idx: npt.NDArray[np.intp] = games_df["l_team_id"].map(idx).to_numpy(dtype=np.intp)
    return teams, idx, w_idx, l_idx


def _build_srs_matrices(
    n: int,
    w_idx: npt.NDArray[np.intp],
    l_idx: npt.NDArray[np.intp],
    margins: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Build net_margin, n_games, avg_margin, and normalized adjacency matrix for SRS."""
    net_margin: npt.NDArray[np.float64] = np.zeros(n)
    n_games: npt.NDArray[np.int64] = np.zeros(n, dtype=np.int64)
    np.add.at(net_margin, w_idx, margins)
    np.add.at(net_margin, l_idx, -margins)
    np.add.at(n_games, w_idx, 1)
    np.add.at(n_games, l_idx, 1)

    # Avoid division by zero (isolated teams with 0 games → rating stays 0)
    n_safe = np.where(n_games > 0, n_games, 1)
    avg_margin = net_margin / n_safe

    # Build opponent adjacency matrix A[i,j] = games between i and j
    A: npt.NDArray[np.float64] = np.zeros((n, n))
    np.add.at(A, (w_idx, l_idx), 1.0)
    np.add.at(A, (l_idx, w_idx), 1.0)
    # Normalize: A_norm[i,j] = fraction of team i's games against j
    A_norm = A / n_safe[:, np.newaxis]

    return net_margin, n_games, avg_margin, A_norm


# ---------------------------------------------------------------------------
# Module-level convenience functions (primary public API)
# ---------------------------------------------------------------------------


def compute_srs_ratings(
    games_df: pd.DataFrame,
    *,
    margin_cap: int = DEFAULT_MARGIN_CAP,
    max_iter: int = DEFAULT_SRS_MAX_ITER,
) -> pd.DataFrame:
    """Compute SRS ratings using default solver config.

    Args:
        games_df: DataFrame with columns ``w_team_id``, ``l_team_id``,
            ``w_score``, ``l_score`` (regular-season games only).
        margin_cap: Maximum point margin cap per game (default 25).
        max_iter: Maximum SRS iterations (default 10,000).

    Returns:
        DataFrame with columns ``["team_id", "srs_rating"]``.
    """
    return BatchRatingSolver(
        margin_cap=margin_cap,
        srs_max_iter=max_iter,
    ).compute_srs(games_df)


def compute_ridge_ratings(
    games_df: pd.DataFrame,
    *,
    lam: float = DEFAULT_RIDGE_LAMBDA,
    margin_cap: int = DEFAULT_MARGIN_CAP,
) -> pd.DataFrame:
    """Compute Ridge regression ratings.

    Args:
        games_df: DataFrame with columns ``w_team_id``, ``l_team_id``,
            ``w_score``, ``l_score`` (regular-season games only).
        lam: Ridge regularization parameter λ (default 20.0).
        margin_cap: Maximum point margin cap per game (default 25).

    Returns:
        DataFrame with columns ``["team_id", "ridge_rating"]``.
    """
    return BatchRatingSolver(
        margin_cap=margin_cap,
        ridge_lambda=lam,
    ).compute_ridge(games_df)


def compute_colley_ratings(games_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Colley Matrix win/loss-only ratings.

    Args:
        games_df: DataFrame with columns ``w_team_id``, ``l_team_id``
            (regular-season games only; scores not used).

    Returns:
        DataFrame with columns ``["team_id", "colley_rating"]``.
    """
    return BatchRatingSolver().compute_colley(games_df)
