"""Kaggle March Machine Learning Mania submission CSV export.

Generates a CSV string with ``ID,Pred`` columns covering all C(n,2)
pairwise matchups for men's D1 teams in a given season.  The ``ID``
column uses Kaggle format ``YYYY_TeamID1_TeamID2`` (lower ID first)
and ``Pred`` is the win probability for TeamID1.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from itertools import combinations

import numpy as np
import numpy.typing as npt


def format_kaggle_submission(
    season: int,
    team_ids: Sequence[int],
    prob_matrix: npt.NDArray[np.float64],
) -> str:
    """Format a probability matrix as a Kaggle submission CSV string.

    Args:
        season: Tournament season year (e.g. 2025).
        team_ids: Team IDs corresponding to matrix rows/columns.
        prob_matrix: n×n pairwise probability matrix where ``P[i,j]``
            is P(team_ids[i] beats team_ids[j]).

    Returns:
        CSV string with header ``ID,Pred`` and C(n,2) data rows.

    Raises:
        ValueError: If the matrix shape doesn't match the team count.
    """
    n = len(team_ids)
    if prob_matrix.shape != (n, n):
        msg = f"prob_matrix shape {prob_matrix.shape} != ({n}, {n})"
        raise ValueError(msg)

    # Build index: team_id → matrix row/column
    idx = {tid: i for i, tid in enumerate(team_ids)}

    buf = io.StringIO()
    buf.write("ID,Pred\n")

    # Sorted team IDs so pairs are emitted in Kaggle order
    sorted_ids = sorted(team_ids)

    for low_id, high_id in combinations(sorted_ids, 2):
        i = idx[low_id]
        j = idx[high_id]
        pred = float(prob_matrix[i, j])
        buf.write(f"{season}_{low_id}_{high_id},{pred}\n")

    return buf.getvalue()
