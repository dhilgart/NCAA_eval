"""Graph-based centrality feature engineering for NCAA team schedules.

This module builds directed NetworkX graphs from season game results and computes
centrality-based features (PageRank, betweenness, HITS, clustering coefficient).

Graph semantics:
- Edges are directed loser → winner ("loser votes for winner quality", PageRank metaphor).
- Edge weight = min(margin, margin_cap) × optional recency multiplier.
- High PageRank ≡ transitively strong team (many high-quality wins flow toward you).

Architecture constraints:
- Pure transform-layer component: accepts pre-loaded DataFrames, does NOT load CSV files.
- No imports from ncaa_eval.ingest (Repository coupling is forbidden here).
- Caller is responsible for 2025 deduplication by (w_team_id, l_team_id, day_num)
  before calling graph functions (2025 season stores 4,545 games twice).

Walk-forward usage (Story 4.7):
- Start with an empty DiGraph at the beginning of each season.
- Add games one-by-one in chronological order via add_game_to_graph().
- Call compute_features(graph, pagerank_init=prev_pagerank) after each game.
- Store pagerank_init (result of previous compute_pagerank()) for warm-start efficiency.
"""

from __future__ import annotations

import logging

import networkx as nx  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# Module-level constants
DEFAULT_MARGIN_CAP: int = 25
DEFAULT_RECENCY_WINDOW_DAYS: int = 20
DEFAULT_RECENCY_MULTIPLIER: float = 1.5
PAGERANK_ALPHA: float = 0.85

_FEATURE_COLUMNS: list[str] = [
    "team_id",
    "pagerank",
    "betweenness_centrality",
    "hits_hub",
    "hits_authority",
    "clustering_coefficient",
]


def build_season_graph(
    games_df: pd.DataFrame,
    margin_cap: int = DEFAULT_MARGIN_CAP,
    reference_day_num: int | None = None,
    recency_window_days: int = DEFAULT_RECENCY_WINDOW_DAYS,
    recency_multiplier: float = DEFAULT_RECENCY_MULTIPLIER,
) -> nx.DiGraph:
    """Build a directed graph from season game results.

    Edges: loser → winner (loser 'votes for' winner quality, PageRank metaphor).
    Edge weight: min(margin, margin_cap) × optional_recency_multiplier.

    Parallel edges (same team pair playing multiple times) are aggregated by summing
    their weights before passing to nx.from_pandas_edgelist().

    Args:
        games_df: DataFrame with columns w_team_id, l_team_id, w_score, l_score, day_num.
        margin_cap: Maximum margin-of-victory to use as edge weight (prevents blowout distortion).
        reference_day_num: Day number used to compute recency window. Defaults to max day_num
            in games_df if None.
        recency_window_days: Games within this many days of reference_day_num get recency boost.
        recency_multiplier: Weight multiplier for recent games within recency_window_days.

    Returns:
        nx.DiGraph with edges loser→winner and "weight" attribute on each edge.
        Returns empty DiGraph if games_df is empty.
    """
    if games_df.empty:
        return nx.DiGraph()

    ref_day = int(games_df["day_num"].max()) if reference_day_num is None else reference_day_num

    # Step 1: Compute base weight (vectorized)
    df = games_df.assign(
        weight=(games_df["w_score"] - games_df["l_score"]).clip(upper=margin_cap).astype(float)
    )

    # Step 2: Apply recency multiplier (vectorized)
    within_window = df["day_num"] >= (ref_day - recency_window_days)
    df = df.assign(weight=df["weight"].where(~within_window, df["weight"] * recency_multiplier))

    # Step 3: Aggregate parallel edges (same team pair played multiple times)
    edge_df = df.groupby(["l_team_id", "w_team_id"], as_index=False)["weight"].sum()

    # Step 4: Build directed graph — loser → winner
    G: nx.DiGraph = nx.from_pandas_edgelist(
        edge_df,
        source="l_team_id",
        target="w_team_id",
        edge_attr="weight",
        create_using=nx.DiGraph,
    )
    return G


def compute_pagerank(
    G: nx.DiGraph,
    alpha: float = PAGERANK_ALPHA,
    nstart: dict[int, float] | None = None,
) -> dict[int, float]:
    """Compute PageRank for each team in the graph.

    Captures transitive win-chain strength (2 hops vs. SoS 1 hop).
    Peer-reviewed NCAA validation: 71.6% vs. 64.2% naive win-ratio (Matthews et al. 2021).

    Args:
        G: Directed graph with loser→winner edges and "weight" attribute.
        alpha: Damping factor (teleportation probability = 1 - alpha).
        nstart: Optional warm-start dictionary (team_id → initial probability).
            Initialize with previous solution for 2–5 iterations instead of 30–50.

    Returns:
        dict mapping team_id → PageRank score. Empty dict if graph has no nodes.
    """
    if G.number_of_nodes() == 0:
        return {}
    return nx.pagerank(G, alpha=alpha, weight="weight", nstart=nstart)  # type: ignore[no-any-return]


def compute_betweenness_centrality(G: nx.DiGraph) -> dict[int, float]:
    """Compute betweenness centrality for each team in the graph.

    Captures structural "bridge" position — distinct signal from PageRank (strength)
    and SoS (schedule quality).

    Args:
        G: Directed graph.

    Returns:
        dict mapping team_id → betweenness centrality score. Empty dict if graph has no nodes.
    """
    if G.number_of_nodes() == 0:
        return {}
    return nx.betweenness_centrality(G, normalized=True)  # type: ignore[no-any-return]


def compute_hits(
    G: nx.DiGraph,
    max_iter: int = 100,
) -> tuple[dict[int, float], dict[int, float]]:
    """Compute HITS hub and authority scores for each team.

    Authority ≈ PageRank (r≈0.908 correlation). Hub = "quality schedule despite losses"
    — distinct signal. Both are returned from a single nx.hits() call.

    Args:
        G: Directed graph.
        max_iter: Maximum iterations for HITS power iteration.

    Returns:
        Tuple of (hub_dict, authority_dict), each mapping team_id → score.
        Returns uniform 0.0 scores for all nodes if graph has no edges.
        Returns uniform 1/n scores on convergence failure (with warning logged).
    """
    if G.number_of_edges() == 0:
        # nx.hits raises ZeroDivisionError or returns zeros for no-edge graphs
        return {n: 0.0 for n in G.nodes()}, {n: 0.0 for n in G.nodes()}
    try:
        hub, auth = nx.hits(G, max_iter=max_iter)
    except nx.PowerIterationFailedConvergence:
        logger.warning("HITS failed to converge after %d iterations; returning uniform scores", max_iter)
        n = G.number_of_nodes()
        score = 1.0 / n if n > 0 else 0.0
        hub_uniform: dict[int, float] = dict.fromkeys(G.nodes(), score)
        auth_uniform: dict[int, float] = dict.fromkeys(G.nodes(), score)
        return hub_uniform, auth_uniform
    return hub, auth


def compute_clustering_coefficient(G: nx.DiGraph) -> dict[int, float]:
    """Compute undirected clustering coefficient for each team.

    Schedule diversity metric: low clustering = broad cross-conference scheduling.
    Uses undirected conversion so that mutual matchups count once (natural interpretation).

    Args:
        G: Directed graph (converted to undirected internally).

    Returns:
        dict mapping team_id → clustering coefficient. Empty dict if graph has no nodes.
    """
    if G.number_of_nodes() == 0:
        return {}
    return nx.clustering(G.to_undirected())  # type: ignore[no-any-return]


class GraphTransformer:
    """Transform game DataFrames into graph-based centrality features.

    Provides both batch (build + compute in one call) and incremental (add_game_to_graph)
    update strategies for walk-forward backtesting efficiency.

    Typical walk-forward usage (Story 4.7):
        transformer = GraphTransformer()
        graph = nx.DiGraph()
        prev_pagerank: dict[int, float] | None = None
        for game in chronological_games:
            transformer.add_game_to_graph(graph, ...)
            features_df = transformer.compute_features(graph, pagerank_init=prev_pagerank)
            prev_pagerank = dict(zip(features_df["team_id"], features_df["pagerank"]))
    """

    def __init__(
        self,
        margin_cap: int = DEFAULT_MARGIN_CAP,
        recency_window_days: int = DEFAULT_RECENCY_WINDOW_DAYS,
        recency_multiplier: float = DEFAULT_RECENCY_MULTIPLIER,
    ) -> None:
        self._margin_cap = margin_cap
        self._recency_window_days = recency_window_days
        self._recency_multiplier = recency_multiplier

    def build_graph(
        self,
        games_df: pd.DataFrame,
        reference_day_num: int | None = None,
    ) -> nx.DiGraph:
        """Build a season graph from a games DataFrame.

        Args:
            games_df: DataFrame with columns w_team_id, l_team_id, w_score, l_score, day_num.
            reference_day_num: Reference day for recency weighting. Defaults to max day_num.

        Returns:
            nx.DiGraph with loser→winner edges and "weight" attribute.
        """
        return build_season_graph(
            games_df,
            margin_cap=self._margin_cap,
            reference_day_num=reference_day_num,
            recency_window_days=self._recency_window_days,
            recency_multiplier=self._recency_multiplier,
        )

    def add_game_to_graph(  # noqa: PLR0913
        self,
        graph: nx.DiGraph,
        w_team_id: int,
        l_team_id: int,
        margin: int,
        day_num: int,
        reference_day_num: int,
    ) -> None:
        """Add a single game to an existing graph in-place.

        Supports incremental walk-forward updates without rebuilding the full graph.
        Edge direction: l_team_id → w_team_id (loser votes for winner).

        Args:
            graph: Existing nx.DiGraph to update in-place.
            w_team_id: Winner team ID.
            l_team_id: Loser team ID.
            margin: Margin of victory (absolute score difference).
            day_num: Day number of the game.
            reference_day_num: Reference day for recency window evaluation.
        """
        weight = float(min(margin, self._margin_cap))
        if day_num >= (reference_day_num - self._recency_window_days):
            weight *= self._recency_multiplier

        # Ensure both nodes are in the graph (isolated nodes may not appear in edge list)
        if w_team_id not in graph:
            graph.add_node(w_team_id)
        if l_team_id not in graph:
            graph.add_node(l_team_id)

        # Accumulate weight for repeated matchups
        if graph.has_edge(l_team_id, w_team_id):
            graph[l_team_id][w_team_id]["weight"] += weight
        else:
            graph.add_edge(l_team_id, w_team_id, weight=weight)

    def compute_features(
        self,
        graph: nx.DiGraph,
        pagerank_init: dict[int, float] | None = None,
    ) -> pd.DataFrame:
        """Compute all centrality features for every team node in the graph.

        Args:
            graph: nx.DiGraph with loser→winner edges.
            pagerank_init: Optional warm-start dict (team_id → probability) from a previous
                compute_pagerank() call. Reduces PageRank iterations from ~30–50 to ~2–5.

        Returns:
            pd.DataFrame with columns ["team_id", "pagerank", "betweenness_centrality",
            "hits_hub", "hits_authority", "clustering_coefficient"], one row per team node.
            Returns empty DataFrame with correct columns if graph has no nodes.
        """
        if graph.number_of_nodes() == 0:
            return pd.DataFrame(columns=_FEATURE_COLUMNS)

        pr = compute_pagerank(graph, nstart=pagerank_init)
        bc = compute_betweenness_centrality(graph)
        hub, auth = compute_hits(graph)
        cc = compute_clustering_coefficient(graph)

        team_ids = sorted(graph.nodes())
        return pd.DataFrame(
            {
                "team_id": team_ids,
                "pagerank": [pr.get(t, 0.0) for t in team_ids],
                "betweenness_centrality": [bc.get(t, 0.0) for t in team_ids],
                "hits_hub": [hub.get(t, 0.0) for t in team_ids],
                "hits_authority": [auth.get(t, 0.0) for t in team_ids],
                "clustering_coefficient": [cc.get(t, 0.0) for t in team_ids],
            }
        )

    def transform(
        self,
        games_df: pd.DataFrame,
        reference_day_num: int | None = None,
        pagerank_init: dict[int, float] | None = None,
    ) -> pd.DataFrame:
        """Convenience method: build graph then compute all centrality features.

        Args:
            games_df: DataFrame with columns w_team_id, l_team_id, w_score, l_score, day_num.
            reference_day_num: Reference day for recency weighting. Defaults to max day_num.
            pagerank_init: Optional warm-start dict for PageRank.

        Returns:
            pd.DataFrame with centrality features, one row per team.
            Returns empty DataFrame with correct columns if games_df is empty.
        """
        if games_df.empty:
            return pd.DataFrame(columns=_FEATURE_COLUMNS)
        graph = self.build_graph(games_df, reference_day_num=reference_day_num)
        return self.compute_features(graph, pagerank_init=pagerank_init)
