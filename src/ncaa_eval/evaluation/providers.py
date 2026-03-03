"""Probability provider protocols and implementations.

Provides the :class:`ProbabilityProvider` protocol and concrete
implementations for pairwise win probability computation:

* :class:`MatrixProvider` — wraps a pre-computed probability matrix.
* :class:`EloProvider` — wraps a stateful model's ``predict_matchup`` method.
* :func:`build_probability_matrix` — builds an n×n pairwise matrix.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from ncaa_eval.evaluation.bracket import MatchupContext


@runtime_checkable
class ProbabilityProvider(Protocol):
    """Protocol for pairwise win probability computation.

    All implementations must satisfy the complementarity contract:
    ``P(A beats B) + P(B beats A) = 1`` for every ``(A, B)`` pair.
    """

    def matchup_probability(
        self,
        team_a_id: int,
        team_b_id: int,
        context: MatchupContext,
    ) -> float:
        """Return P(team_a beats team_b).

        Args:
            team_a_id: First team's canonical ID.
            team_b_id: Second team's canonical ID.
            context: Matchup context (season, day_num, neutral).

        Returns:
            Probability in ``[0, 1]``.
        """
        ...

    def batch_matchup_probabilities(
        self,
        team_a_ids: Sequence[int],
        team_b_ids: Sequence[int],
        context: MatchupContext,
    ) -> npt.NDArray[np.float64]:
        """Return P(a_i beats b_i) for all pairs.

        Args:
            team_a_ids: Sequence of first-team IDs.
            team_b_ids: Sequence of second-team IDs (same length).
            context: Matchup context.

        Returns:
            1-D float64 array of shape ``(len(team_a_ids),)``.
        """
        ...


class MatrixProvider:
    """Wraps a pre-computed probability matrix as a :class:`ProbabilityProvider`.

    Args:
        prob_matrix: n×n pairwise probability matrix.
        team_ids: Sequence of team IDs matching matrix indices.
    """

    def __init__(
        self,
        prob_matrix: npt.NDArray[np.float64],
        team_ids: Sequence[int],
    ) -> None:
        self._P = prob_matrix
        self._index = {tid: i for i, tid in enumerate(team_ids)}

    def matchup_probability(
        self,
        team_a_id: int,
        team_b_id: int,
        context: MatchupContext,
    ) -> float:
        """Return P(team_a beats team_b) from the stored matrix."""
        i = self._index[team_a_id]
        j = self._index[team_b_id]
        return float(self._P[i, j])

    def batch_matchup_probabilities(
        self,
        team_a_ids: Sequence[int],
        team_b_ids: Sequence[int],
        context: MatchupContext,
    ) -> npt.NDArray[np.float64]:
        """Return batch probabilities from the stored matrix."""
        rows = np.array([self._index[a] for a in team_a_ids])
        cols = np.array([self._index[b] for b in team_b_ids])
        result: npt.NDArray[np.float64] = self._P[rows, cols].astype(np.float64)
        return result


class EloProvider:
    """Wraps a :class:`StatefulModel` as a :class:`ProbabilityProvider`.

    Uses the model's ``predict_matchup`` method for probability computation.

    Args:
        model: Any :class:`StatefulModel` instance with ``predict_matchup``.
    """

    def __init__(self, model: Any) -> None:
        if not hasattr(model, "predict_matchup"):
            msg = "model must have a predict_matchup(team_a_id, team_b_id) method"
            raise TypeError(msg)
        self._model: Any = model

    def matchup_probability(
        self,
        team_a_id: int,
        team_b_id: int,
        context: MatchupContext,
    ) -> float:
        """Return P(team_a beats team_b) via the model's ``predict_matchup``."""
        result: float = self._model.predict_matchup(team_a_id, team_b_id)
        return result

    def batch_matchup_probabilities(
        self,
        team_a_ids: Sequence[int],
        team_b_ids: Sequence[int],
        context: MatchupContext,
    ) -> npt.NDArray[np.float64]:
        """Return batch probabilities by looping ``predict_matchup``.

        Elo is O(1) per pair so looping is acceptable.
        """
        return np.array(
            [self._model.predict_matchup(a, b) for a, b in zip(team_a_ids, team_b_ids)],
            dtype=np.float64,
        )


def build_probability_matrix(
    provider: ProbabilityProvider,
    team_ids: Sequence[int],
    context: MatchupContext,
) -> npt.NDArray[np.float64]:
    """Build n×n pairwise win probability matrix.

    Uses upper-triangle batch call, then fills ``P[j,i] = 1 - P[i,j]``
    via the complementarity contract.

    Args:
        provider: Probability provider implementing the protocol.
        team_ids: Team IDs in bracket order.
        context: Matchup context.

    Returns:
        Float64 array of shape ``(n, n)``.  Diagonal is zero.
    """
    n = len(team_ids)
    rows, cols = np.triu_indices(n, k=1)
    a_ids = [team_ids[int(i)] for i in rows]
    b_ids = [team_ids[int(j)] for j in cols]

    probs = provider.batch_matchup_probabilities(a_ids, b_ids, context)

    P = np.zeros((n, n), dtype=np.float64)
    P[rows, cols] = probs
    P[cols, rows] = 1.0 - probs
    return P
