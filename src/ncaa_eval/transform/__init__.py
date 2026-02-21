"""Feature engineering and data transformation module."""

from __future__ import annotations

from ncaa_eval.transform.normalization import (
    ConferenceLookup,
    CoverageGateResult,
    MasseyOrdinalsStore,
    TeamNameNormalizer,
    TourneySeed,
    TourneySeedTable,
    parse_seed,
)
from ncaa_eval.transform.sequential import (
    DetailedResultsLoader,
    SequentialTransformer,
    compute_four_factors,
    compute_game_weights,
    compute_possessions,
    compute_streak,
)
from ncaa_eval.transform.serving import (
    ChronologicalDataServer,
    SeasonGames,
    rescale_overtime,
)

__all__ = [
    "ChronologicalDataServer",
    "ConferenceLookup",
    "CoverageGateResult",
    "DetailedResultsLoader",
    "MasseyOrdinalsStore",
    "SeasonGames",
    "SequentialTransformer",
    "TeamNameNormalizer",
    "TourneySeed",
    "TourneySeedTable",
    "compute_four_factors",
    "compute_game_weights",
    "compute_possessions",
    "compute_streak",
    "parse_seed",
    "rescale_overtime",
]
