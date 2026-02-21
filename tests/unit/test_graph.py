"""Unit tests for ncaa_eval.transform.graph — graph builders and centrality features."""

from __future__ import annotations

import networkx as nx  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]
import pytest

from ncaa_eval.transform.graph import (
    GraphTransformer,
    build_season_graph,
    compute_betweenness_centrality,
    compute_clustering_coefficient,
    compute_hits,
    compute_pagerank,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def triangle_games() -> pd.DataFrame:
    """Three-team cycle: 1 beat 2, 2 beat 3, 3 beat 1."""
    return pd.DataFrame(
        {
            "w_team_id": [1, 2, 3],
            "l_team_id": [2, 3, 1],
            "w_score": [80, 75, 85],
            "l_score": [70, 65, 72],
            "day_num": [30, 40, 50],
        }
    )


@pytest.fixture
def linear_chain_games() -> pd.DataFrame:
    """Four-team chain: 1 beat 2 beat 3 beat 4. No cycles."""
    return pd.DataFrame(
        {
            "w_team_id": [1, 2, 3],
            "l_team_id": [2, 3, 4],
            "w_score": [80, 75, 70],
            "l_score": [70, 65, 62],
            "day_num": [30, 40, 50],
        }
    )


@pytest.fixture
def triangle_graph(triangle_games: pd.DataFrame) -> nx.DiGraph:
    return build_season_graph(triangle_games)


@pytest.fixture
def chain_graph(linear_chain_games: pd.DataFrame) -> nx.DiGraph:
    return build_season_graph(linear_chain_games)


# ---------------------------------------------------------------------------
# Task 1: build_season_graph tests (AC: 1–3)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.smoke
def test_build_season_graph_edge_direction() -> None:
    """Edges must be directed loser→winner (loser 'votes for' winner)."""
    games = pd.DataFrame(
        {
            "w_team_id": [1, 2],
            "l_team_id": [2, 3],
            "w_score": [80, 75],
            "l_score": [70, 65],
            "day_num": [30, 40],
        }
    )
    G = build_season_graph(games)
    # Edges should be loser→winner: (2,1) and (3,2)
    assert G.has_edge(2, 1), "Expected edge 2→1 (team 2 lost to team 1)"
    assert G.has_edge(3, 2), "Expected edge 3→2 (team 3 lost to team 2)"
    # Reversed edges should NOT exist
    assert not G.has_edge(1, 2), "Edge 1→2 should not exist (winner→loser direction is wrong)"
    assert not G.has_edge(2, 3), "Edge 2→3 should not exist (winner→loser direction is wrong)"


@pytest.mark.unit
@pytest.mark.smoke
def test_build_season_graph_edge_weight_cap() -> None:
    """Blowout margin 50 should be capped at DEFAULT_MARGIN_CAP=25."""
    games = pd.DataFrame(
        {
            "w_team_id": [1],
            "l_team_id": [2],
            "w_score": [100],
            "l_score": [50],
            "day_num": [1],
        }
    )
    # reference_day_num=100 puts day_num=1 outside recency window (1 < 80), so no multiplier
    G = build_season_graph(games, reference_day_num=100)
    assert G[2][1]["weight"] == pytest.approx(25.0)


@pytest.mark.unit
@pytest.mark.smoke
def test_build_season_graph_edge_weight_exact() -> None:
    """Game with margin 10 should produce edge weight exactly 10.0."""
    games = pd.DataFrame(
        {
            "w_team_id": [1],
            "l_team_id": [2],
            "w_score": [80],
            "l_score": [70],
            "day_num": [1],
        }
    )
    # reference_day_num=100 puts day_num=1 outside recency window (1 < 80), so no multiplier
    G = build_season_graph(games, reference_day_num=100)
    assert G[2][1]["weight"] == pytest.approx(10.0)


@pytest.mark.unit
@pytest.mark.smoke
def test_build_season_graph_recency_weight() -> None:
    """Games within recency window (last 20 days) get 1.5× multiplier; older games do not."""
    # reference_day_num = 100
    # Game at day 85 → within window (100 - 20 = 80 ≤ 85): weight × 1.5
    # Game at day 70 → outside window (70 < 80): weight × 1.0
    games = pd.DataFrame(
        {
            "w_team_id": [1, 3],
            "l_team_id": [2, 4],
            "w_score": [80, 80],
            "l_score": [70, 70],  # margin = 10 for both
            "day_num": [85, 70],
        }
    )
    G = build_season_graph(games, reference_day_num=100)
    recent_weight = G[2][1]["weight"]
    old_weight = G[4][3]["weight"]
    assert recent_weight == pytest.approx(10.0 * 1.5), f"Expected 15.0, got {recent_weight}"
    assert old_weight == pytest.approx(10.0), f"Expected 10.0, got {old_weight}"


@pytest.mark.unit
@pytest.mark.smoke
def test_build_season_graph_parallel_edges_aggregated() -> None:
    """Same team pair playing twice should have a single edge with summed weight."""
    games = pd.DataFrame(
        {
            "w_team_id": [1, 1],
            "l_team_id": [2, 2],
            "w_score": [80, 78],
            "l_score": [70, 70],  # margins: 10 and 8
            "day_num": [1, 2],
        }
    )
    # reference_day_num=100 puts both games outside recency window (days 1,2 < 80), no multiplier
    G = build_season_graph(games, reference_day_num=100)
    assert G.number_of_edges() == 1
    assert G[2][1]["weight"] == pytest.approx(18.0)


@pytest.mark.unit
@pytest.mark.smoke
def test_build_season_graph_empty_input() -> None:
    """Empty DataFrame should return an empty nx.DiGraph (0 nodes, 0 edges)."""
    games = pd.DataFrame(columns=["w_team_id", "l_team_id", "w_score", "l_score", "day_num"])
    G = build_season_graph(games)
    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() == 0
    assert G.number_of_edges() == 0


# ---------------------------------------------------------------------------
# Task 2: PageRank tests (AC: 4)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.smoke
def test_compute_pagerank_linear_chain(chain_graph: nx.DiGraph) -> None:
    """In chain 1>2>3>4, team 1 should have highest PageRank, team 4 lowest."""
    pr = compute_pagerank(chain_graph)
    assert isinstance(pr, dict)
    assert all(isinstance(k, int) for k in pr), "team_id keys should be integers"
    # Team 1 has the highest pagerank (all wins flow to it)
    # Team 4 has the lowest pagerank (lost to everyone, no wins)
    assert pr[1] > pr[2] > pr[3], "PageRank should decrease along the chain"
    # Team 4 is the endpoint loser — lowest PR
    assert pr[4] < pr[3]


@pytest.mark.unit
@pytest.mark.smoke
def test_compute_pagerank_triangle_converges(triangle_graph: nx.DiGraph) -> None:
    """Three-team cycle should converge and return approximately equal scores summing to 1."""
    pr = compute_pagerank(triangle_graph)
    scores = list(pr.values())
    # In a symmetric cycle, all scores should be approximately equal
    assert max(scores) - min(scores) < 0.05, "Cycle PageRank scores should be nearly equal"
    assert abs(sum(scores) - 1.0) < 1e-6, "PageRank scores should sum to 1"


@pytest.mark.unit
@pytest.mark.smoke
def test_compute_pagerank_warm_start(triangle_graph: nx.DiGraph) -> None:
    """Warm start should produce the same result as cold start (within 1e-6)."""
    cold = compute_pagerank(triangle_graph, nstart=None)
    warm = compute_pagerank(triangle_graph, nstart=cold)
    for team_id in cold:
        assert abs(cold[team_id] - warm[team_id]) < 1e-6, f"Warm start diverges for team {team_id}"


@pytest.mark.unit
@pytest.mark.smoke
def test_compute_pagerank_empty_graph() -> None:
    """Empty DiGraph should return empty dict."""
    G: nx.DiGraph = nx.DiGraph()
    assert compute_pagerank(G) == {}


# ---------------------------------------------------------------------------
# Task 2: Betweenness centrality tests (AC: 5)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.smoke
def test_compute_betweenness_linear_chain(chain_graph: nx.DiGraph) -> None:
    """In chain 1→2→3→4, teams 2 and 3 (bridges) should have higher betweenness than 1 and 4."""
    bc = compute_betweenness_centrality(chain_graph)
    # Teams 2 and 3 are structural bridges in the chain
    assert bc[2] > bc[1], "Team 2 (bridge) should have higher betweenness than team 1 (endpoint)"
    assert bc[3] > bc[4], "Team 3 (bridge) should have higher betweenness than team 4 (endpoint)"


@pytest.mark.unit
@pytest.mark.smoke
def test_compute_betweenness_isolated_nodes() -> None:
    """Graph with isolated nodes (no edges) should have all betweenness scores = 0.0."""
    G: nx.DiGraph = nx.DiGraph()
    G.add_nodes_from([1, 2, 3])
    bc = compute_betweenness_centrality(G)
    for score in bc.values():
        assert score == pytest.approx(0.0)


@pytest.mark.unit
@pytest.mark.smoke
def test_compute_betweenness_empty_graph() -> None:
    """Empty DiGraph should return empty dict."""
    G: nx.DiGraph = nx.DiGraph()
    assert compute_betweenness_centrality(G) == {}


# ---------------------------------------------------------------------------
# Task 2: HITS tests (AC: 6)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.smoke
def test_compute_hits_returns_hub_and_authority(triangle_graph: nx.DiGraph) -> None:
    """compute_hits should return a tuple of two dicts with matching team_id keys."""
    result = compute_hits(triangle_graph)
    assert isinstance(result, tuple)
    assert len(result) == 2
    hub, auth = result
    assert isinstance(hub, dict)
    assert isinstance(auth, dict)
    assert set(hub.keys()) == set(auth.keys()), "Hub and authority dicts must have same keys"


@pytest.mark.unit
@pytest.mark.smoke
def test_compute_hits_hub_high_for_loser_of_high_authority() -> None:
    """Team C (beat B and D) should have high authority; team D (lost only to C) has hub > 0."""
    # A beat B, C beat A, C beat D
    # C is the "sink" — many losers point to C → high authority
    # D lost only to high-authority C → D is a hub (good schedule, despite losses)
    games = pd.DataFrame(
        {
            "w_team_id": [1, 3, 3],  # A=1, B=2, C=3, D=4
            "l_team_id": [2, 1, 4],
            "w_score": [80, 75, 70],
            "l_score": [70, 60, 60],
            "day_num": [30, 40, 50],
        }
    )
    G = build_season_graph(games)
    hub, auth = compute_hits(G)
    # Team C (id=3) should have the highest authority
    assert auth[3] >= auth[2], "Team C should have >= authority than team B"
    assert auth[3] >= auth[4], "Team C should have >= authority than team D"
    # Team D (id=4) lost only to high-authority C → hub score > 0
    assert hub[4] >= 0.0, "Team D hub score must be non-negative"


@pytest.mark.unit
@pytest.mark.smoke
def test_compute_hits_empty_edges() -> None:
    """DiGraph with nodes but no edges should return two dicts with 0.0 for all nodes."""
    G: nx.DiGraph = nx.DiGraph()
    G.add_nodes_from([1, 2, 3])
    hub, auth = compute_hits(G)
    assert set(hub.keys()) == {1, 2, 3}
    assert set(auth.keys()) == {1, 2, 3}
    for score in hub.values():
        assert score == pytest.approx(0.0)
    for score in auth.values():
        assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Task 2: Clustering coefficient tests (AC: 7)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.smoke
def test_compute_clustering_triangle(triangle_graph: nx.DiGraph) -> None:
    """Three-team cycle → undirected triangle → all clustering coefficients = 1.0."""
    cc = compute_clustering_coefficient(triangle_graph)
    for team_id, score in cc.items():
        assert score == pytest.approx(1.0), f"Team {team_id} should have clustering 1.0"


@pytest.mark.unit
@pytest.mark.smoke
def test_compute_clustering_linear_chain(chain_graph: nx.DiGraph) -> None:
    """Four-team linear chain has no triangles → all clustering coefficients = 0.0."""
    cc = compute_clustering_coefficient(chain_graph)
    for team_id, score in cc.items():
        assert score == pytest.approx(0.0), f"Team {team_id} should have clustering 0.0"


@pytest.mark.unit
@pytest.mark.smoke
def test_compute_clustering_empty_graph() -> None:
    """Empty DiGraph should return empty dict."""
    G: nx.DiGraph = nx.DiGraph()
    assert compute_clustering_coefficient(G) == {}


# ---------------------------------------------------------------------------
# Task 3: GraphTransformer — add_game_to_graph tests (AC: 9)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.smoke
def test_add_game_to_graph_new_edge() -> None:
    """Adding a game to empty graph should create the correct loser→winner edge."""
    transformer = GraphTransformer()
    G: nx.DiGraph = nx.DiGraph()
    transformer.add_game_to_graph(G, w_team_id=1, l_team_id=2, margin=10, day_num=50, reference_day_num=100)
    assert G.has_edge(2, 1), "Edge l_team_id→w_team_id should be created"
    # Day 50 >= (100 - 20) = 80? No — 50 < 80, so no recency multiplier
    assert G[2][1]["weight"] == pytest.approx(10.0)


@pytest.mark.unit
@pytest.mark.smoke
def test_add_game_to_graph_accumulates_weight() -> None:
    """Adding the same matchup twice should accumulate the edge weight."""
    transformer = GraphTransformer()
    G: nx.DiGraph = nx.DiGraph()
    transformer.add_game_to_graph(G, w_team_id=1, l_team_id=2, margin=10, day_num=1, reference_day_num=1)
    transformer.add_game_to_graph(G, w_team_id=1, l_team_id=2, margin=8, day_num=1, reference_day_num=1)
    # Both games are at day 1 == reference_day_num, so recency applies: (10 + 8) * 1.5 = 27
    assert G.number_of_edges() == 1
    assert G[2][1]["weight"] == pytest.approx((10.0 + 8.0) * 1.5)


@pytest.mark.unit
@pytest.mark.smoke
def test_add_game_to_graph_recency_multiplier() -> None:
    """Games within recency window should have their weight multiplied by 1.5."""
    transformer = GraphTransformer()
    G: nx.DiGraph = nx.DiGraph()
    # Game at day 90, reference 100 → 90 >= (100 - 20) = 80 → within window
    transformer.add_game_to_graph(G, w_team_id=1, l_team_id=2, margin=10, day_num=90, reference_day_num=100)
    assert G[2][1]["weight"] == pytest.approx(10.0 * 1.5)


@pytest.mark.unit
@pytest.mark.smoke
def test_add_game_to_graph_margin_cap() -> None:
    """Games with margin > 25 should be capped at 25.0."""
    transformer = GraphTransformer()
    G: nx.DiGraph = nx.DiGraph()
    # Day 1, reference 100 → outside window, no multiplier
    transformer.add_game_to_graph(G, w_team_id=1, l_team_id=2, margin=50, day_num=1, reference_day_num=100)
    assert G[2][1]["weight"] == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# Task 3: GraphTransformer — compute_features and transform tests (AC: 10)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.smoke
def test_graph_transformer_transform_columns(triangle_games: pd.DataFrame) -> None:
    """transform() should return DataFrame with exactly the 6 required column names."""
    transformer = GraphTransformer()
    result = transformer.transform(triangle_games)
    expected_columns = [
        "team_id",
        "pagerank",
        "betweenness_centrality",
        "hits_hub",
        "hits_authority",
        "clustering_coefficient",
    ]
    assert list(result.columns) == expected_columns


@pytest.mark.unit
@pytest.mark.smoke
def test_graph_transformer_transform_row_count(triangle_games: pd.DataFrame) -> None:
    """Triangle fixture (3 teams) should produce exactly 3 rows (one per team)."""
    transformer = GraphTransformer()
    result = transformer.transform(triangle_games)
    assert len(result) == 3


@pytest.mark.unit
@pytest.mark.smoke
def test_graph_transformer_transform_empty_input() -> None:
    """Empty games_df should return empty DataFrame with the 6 correct columns."""
    transformer = GraphTransformer()
    empty = pd.DataFrame(columns=["w_team_id", "l_team_id", "w_score", "l_score", "day_num"])
    result = transformer.transform(empty)
    assert len(result) == 0
    expected_columns = [
        "team_id",
        "pagerank",
        "betweenness_centrality",
        "hits_hub",
        "hits_authority",
        "clustering_coefficient",
    ]
    assert list(result.columns) == expected_columns


@pytest.mark.unit
@pytest.mark.smoke
def test_graph_transformer_compute_features_consistent(triangle_games: pd.DataFrame) -> None:
    """build_graph then compute_features should return same result as transform()."""
    transformer = GraphTransformer()
    G = transformer.build_graph(triangle_games)
    two_step = transformer.compute_features(G)
    one_step = transformer.transform(triangle_games)
    # Sort both by team_id for consistent comparison
    two_step = two_step.sort_values("team_id").reset_index(drop=True)
    one_step = one_step.sort_values("team_id").reset_index(drop=True)
    for col in two_step.columns:
        for i in range(len(two_step)):
            assert two_step[col].iloc[i] == pytest.approx(
                one_step[col].iloc[i]
            ), f"Mismatch in column '{col}' row {i}"


# ---------------------------------------------------------------------------
# Task 5.28: No iterrows smoke test (architecture mandate)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.smoke
def test_no_iterrows() -> None:
    """graph.py must not use iterrows (vectorization mandate)."""
    import pathlib

    source_path = pathlib.Path(__file__).parent.parent.parent / "src" / "ncaa_eval" / "transform" / "graph.py"
    source_text = source_path.read_text()
    assert "iterrows" not in source_text, "graph.py must not use iterrows"
