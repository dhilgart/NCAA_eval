"""Feature engineering and data transformation module."""

from __future__ import annotations

from ncaa_eval.transform.calibration import (
    IsotonicCalibrator,
    SigmoidCalibrator,
)
from ncaa_eval.transform.feature_serving import (
    FeatureBlock,
    FeatureConfig,
    StatefulFeatureServer,
)
from ncaa_eval.transform.graph import (
    GraphTransformer,
    build_season_graph,
    compute_betweenness_centrality,
    compute_clustering_coefficient,
    compute_hits,
    compute_pagerank,
)
from ncaa_eval.transform.normalization import (
    ConferenceLookup,
    CoverageGateResult,
    MasseyOrdinalsStore,
    TeamNameNormalizer,
    TourneySeed,
    TourneySeedTable,
    parse_seed,
)
from ncaa_eval.transform.opponent import (
    BatchRatingSolver,
    compute_colley_ratings,
    compute_ridge_ratings,
    compute_srs_ratings,
)
from ncaa_eval.transform.sequential import (
    DetailedResultsLoader,
    SequentialTransformer,
    apply_ot_rescaling,
    compute_ewma_stats,
    compute_four_factors,
    compute_game_weights,
    compute_momentum,
    compute_per_possession_stats,
    compute_possessions,
    compute_rolling_stats,
    compute_streak,
)
from ncaa_eval.transform.serving import (
    ChronologicalDataServer,
    SeasonGames,
    rescale_overtime,
)

__all__ = [
    "BatchRatingSolver",
    "ChronologicalDataServer",
    "ConferenceLookup",
    "CoverageGateResult",
    "DetailedResultsLoader",
    "FeatureBlock",
    "FeatureConfig",
    "GraphTransformer",
    "IsotonicCalibrator",
    "MasseyOrdinalsStore",
    "SeasonGames",
    "SequentialTransformer",
    "SigmoidCalibrator",
    "StatefulFeatureServer",
    "TeamNameNormalizer",
    "TourneySeed",
    "TourneySeedTable",
    "apply_ot_rescaling",
    "build_season_graph",
    "compute_betweenness_centrality",
    "compute_clustering_coefficient",
    "compute_colley_ratings",
    "compute_ewma_stats",
    "compute_four_factors",
    "compute_game_weights",
    "compute_hits",
    "compute_momentum",
    "compute_pagerank",
    "compute_per_possession_stats",
    "compute_possessions",
    "compute_ridge_ratings",
    "compute_rolling_stats",
    "compute_srs_ratings",
    "compute_streak",
    "parse_seed",
    "rescale_overtime",
]
