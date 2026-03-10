"""Scoring orchestration and backward-compatible re-exports for the dashboard.

New code should import from the specific submodule:

* :mod:`dashboard.lib.data_loaders` — cached data-loading functions
* :mod:`dashboard.lib.simulation_helpers` — bracket simulation orchestration
* :mod:`dashboard.lib.export` — CSV export helpers

This module retains ``score_chosen_bracket`` and ``build_custom_scoring``.
``_ROUND_LABELS`` lives in :mod:`dashboard.lib.export` and is re-exported
here for backward compatibility.  All other symbols are re-exported from
the new submodules for backward compatibility.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import streamlit as st

from ncaa_eval.evaluation.simulation import (
    DictScoring,
    compute_bracket_distribution,
    score_bracket_against_sims,
)

if TYPE_CHECKING:
    from dashboard.lib.simulation_helpers import BracketSimulationResult
    from ncaa_eval.evaluation.simulation import BracketDistribution, ScoringRule

__all__ = [
    "BracketSimulationResult",
    "_ROUND_LABELS",
    "_ROUND_OF_64_DAY_NUM",
    "_build_provider_from_folds",
    "_build_team_labels",
    "_game_win_probability",
    "_load_team_names_uncached",
    "build_custom_scoring",
    "export_bracket_csv",
    "get_data_dir",
    "load_available_runs",
    "load_available_scorings",
    "load_available_years",
    "load_data_freshness",
    "load_feature_importances",
    "load_fold_predictions",
    "load_leaderboard_data",
    "load_scoring_display_names",
    "load_team_names",
    "load_tourney_seeds",
    "run_bracket_simulation",
    "score_chosen_bracket",
]

# Re-exports for backward compatibility — consumers should migrate to
# dashboard.lib.data_loaders, dashboard.lib.simulation_helpers, dashboard.lib.export.
from dashboard.lib.data_loaders import (  # noqa: F401
    _load_team_names_uncached,
    get_data_dir,
    load_available_runs,
    load_available_scorings,
    load_available_years,
    load_data_freshness,
    load_feature_importances,
    load_fold_predictions,
    load_leaderboard_data,
    load_scoring_display_names,
    load_team_names,
    load_tourney_seeds,
)
from dashboard.lib.export import (  # noqa: F401
    _ROUND_LABELS,
    _game_win_probability,
    export_bracket_csv,
)
from dashboard.lib.simulation_helpers import (  # noqa: F401
    _ROUND_OF_64_DAY_NUM,
    BracketSimulationResult as BracketSimulationResult,
    _build_provider_from_folds,
    _build_team_labels,
    run_bracket_simulation,
)


@st.cache_data(ttl=None)
def score_chosen_bracket(
    sim_data: BracketSimulationResult,
    _scoring_rules: Sequence[ScoringRule],
    scoring_key: str,
    chosen_winners: tuple[int, ...] | None = None,
) -> dict[str, BracketDistribution]:
    """Score a bracket against MC simulations.

    Results are cached via ``@st.cache_data``.  ``_scoring_rules`` is prefixed
    with ``_`` so Streamlit skips hashing the list of rule objects (which are
    not hashable); ``scoring_key`` provides the cache discriminator instead,
    ensuring different rules produce different cache entries.

    When ``chosen_winners`` is provided, scores that bracket instead of the
    model's most-likely bracket.  The ``chosen_winners`` tuple is included in
    the cache key, so different user edits produce different cache entries.

    Args:
        sim_data: Bracket simulation result (must have MC sim_winners).
        _scoring_rules: Scoring rules to evaluate (not hashed by Streamlit).
        scoring_key: Cache-discriminating string (e.g. rule name or custom
            points repr) that uniquely identifies the scoring configuration.
        chosen_winners: Optional explicit bracket winners to score instead of
            ``sim_data.most_likely.winners``.  Pass as a tuple (hashable) for
            Streamlit cache compatibility.

    Returns:
        Mapping of ``rule_name → BracketDistribution``.

    Raises:
        ValueError: If ``sim_data.sim_result.sim_winners`` is ``None``
            (analytical mode has no MC data).
    """
    sim_winners = sim_data.sim_result.sim_winners
    if sim_winners is None:
        msg = "MC sim_winners required for pool scoring; run simulation in monte_carlo mode"
        raise ValueError(msg)

    winners = chosen_winners if chosen_winners is not None else sim_data.most_likely.winners
    chosen = np.array(winners, dtype=np.int32)
    raw_scores = score_bracket_against_sims(chosen, sim_winners, _scoring_rules)

    return {name: compute_bracket_distribution(scores) for name, scores in raw_scores.items()}


def build_custom_scoring(points_per_round: tuple[float, ...]) -> DictScoring:
    """Wrap a 6-element per-round point schedule in a :class:`DictScoring`.

    Args:
        points_per_round: Points for rounds 0–5 (R64 through Championship).
            Must contain exactly 6 elements.

    Returns:
        A ``DictScoring`` instance named ``"custom"``.

    Raises:
        ValueError: If ``points_per_round`` does not contain exactly 6 entries.
    """
    points_dict = dict(enumerate(points_per_round))
    return DictScoring(points=points_dict, scoring_name="custom")
