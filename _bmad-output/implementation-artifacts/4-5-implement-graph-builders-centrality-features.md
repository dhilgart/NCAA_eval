# Story 4.5: Implement Graph Builders & Centrality Features

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a data scientist,
I want to convert season schedules into NetworkX directed graphs and compute PageRank, betweenness centrality, HITS (hub + authority), and clustering coefficient features,
so that I can quantify transitive team strength, structural schedule position, and schedule diversity as predictive features.

## Acceptance Criteria

1. **Given** game data for a season is available, **When** the developer builds a season graph and computes centrality features, **Then** the season schedule is converted to a NetworkX directed graph with edges directed loser→winner (loser "votes for" winner quality, following the PageRank web metaphor), using `nx.from_pandas_edgelist(source="l_team_id", target="w_team_id")` — no iterrows.
2. **And** edge weights are margin-of-victory capped at 25 points (`min(w_score − l_score, 25)`) to prevent extreme-blowout distortion.
3. **And** optional date-recency weighting multiplies edge weight by a configurable recency factor (default 1.5×) for games within the last 20 days of the `reference_day_num` (defaults to the last game's `day_num` if not provided).
4. **And** **PageRank** is computed (directed, margin-weighted, `nx.pagerank(G, alpha=0.85, weight="weight")`) — captures transitive win-chain strength (2 hops vs. SoS 1 hop); peer-reviewed NCAA validation: 71.6% vs. 64.2% naive win-ratio (Matthews et al. 2021).
5. **And** **Betweenness centrality** is computed (`nx.betweenness_centrality(G)`) — captures structural "bridge" position; distinct signal from both strength (PageRank) and schedule quality (SoS).
6. **And** **HITS** hub and authority scores are both computed via a single `nx.hits(G)` call; both hub and authority scores are returned (authority ≈ PageRank, r≈0.908; hub = "quality schedule despite losses" — distinct signal); the function handles convergence failures gracefully.
7. **And** **Clustering coefficient** is computed (`nx.clustering(G.to_undirected())`) — schedule diversity metric: low clustering = broad cross-conference scheduling.
8. **And** walk-forward incremental update strategy is implemented: PageRank supports power-iteration warm start via `nstart` parameter (initialize with previous solution; 2–5 iterations instead of 30–50); betweenness is fully recomputed per call (O(V×E); caller pre-computes and stores per game date for walk-forward efficiency over 40+ seasons).
9. **And** graph features can be computed incrementally as games are added — `GraphTransformer` provides an `add_game_to_graph()` method for in-place edge addition to an existing `nx.DiGraph`.
10. **And** `compute_features(G, pagerank_init)` returns a per-team `pd.DataFrame` with columns: `team_id`, `pagerank`, `betweenness_centrality`, `hits_hub`, `hits_authority`, `clustering_coefficient` — one row per team node in the graph.
11. **And** graph builders are covered by unit tests in `tests/unit/test_graph.py` with known small-graph fixtures including PageRank convergence and betweenness structural correctness assertions.

## Tasks / Subtasks

- [x] Task 1: Create `src/ncaa_eval/transform/graph.py` — module header, constants, and `build_season_graph()` (AC: 1–3)
  - [x] 1.1: Add module header: `from __future__ import annotations`, imports, module-level logger (`import networkx as nx  # type: ignore[import-untyped]`, `import pandas as pd  # type: ignore[import-untyped]`)
  - [x] 1.2: Define module-level constants: `DEFAULT_MARGIN_CAP: int = 25`, `DEFAULT_RECENCY_WINDOW_DAYS: int = 20`, `DEFAULT_RECENCY_MULTIPLIER: float = 1.5`, `PAGERANK_ALPHA: float = 0.85`
  - [x] 1.3: Implement `build_season_graph(games_df: pd.DataFrame, margin_cap: int = DEFAULT_MARGIN_CAP, reference_day_num: int | None = None, recency_window_days: int = DEFAULT_RECENCY_WINDOW_DAYS, recency_multiplier: float = DEFAULT_RECENCY_MULTIPLIER) -> nx.DiGraph` — fully vectorized using `nx.from_pandas_edgelist()`; returns empty DiGraph if `games_df` is empty; edges: `source="l_team_id"`, `target="w_team_id"`, `edge_attr="weight"`; weight = `(w_score − l_score).clip(upper=margin_cap)`; apply recency multiplier to games within `recency_window_days` of `reference_day_num` (which defaults to `games_df["day_num"].max()` if None); aggregate parallel edges (same team pair playing multiple times) using `groupby(["l_team_id", "w_team_id"])["weight"].sum()` before `from_pandas_edgelist()`

- [x] Task 2: Implement module-level centrality functions (AC: 4–7)
  - [x] 2.1: Implement `compute_pagerank(G: nx.DiGraph, alpha: float = PAGERANK_ALPHA, nstart: dict[int, float] | None = None) -> dict[int, float]` — `nx.pagerank(G, alpha=alpha, weight="weight", nstart=nstart)`; if `G.number_of_nodes() == 0`, return `{}`; returns `dict[int, float]` mapping team_id → score
  - [x] 2.2: Implement `compute_betweenness_centrality(G: nx.DiGraph) -> dict[int, float]` — `nx.betweenness_centrality(G, normalized=True)`; if `G.number_of_nodes() == 0`, return `{}`; no edge weights (structural betweenness — cleaner for "bridge" interpretation)
  - [x] 2.3: Implement `compute_hits(G: nx.DiGraph, max_iter: int = 100) -> tuple[dict[int, float], dict[int, float]]` — guard: if `G.number_of_edges() == 0`, return `({n: 0.0 for n in G.nodes()}, {n: 0.0 for n in G.nodes()})`; wrap `nx.hits(G, max_iter=max_iter)` in a try/except for `nx.PowerIterationFailedConvergence`; on convergence failure: log warning and return uniform scores (`1/n` each); returns `(hub_dict, authority_dict)` where both are `dict[int, float]`
  - [x] 2.4: Implement `compute_clustering_coefficient(G: nx.DiGraph) -> dict[int, float]` — `nx.clustering(G.to_undirected())`; if `G.number_of_nodes() == 0`, return `{}`; returns `dict[int, float]` mapping team_id → coefficient

- [x] Task 3: Implement `GraphTransformer` class (AC: 8–10)
  - [x] 3.1: Define `GraphTransformer` class with `__init__(self, margin_cap: int = DEFAULT_MARGIN_CAP, recency_window_days: int = DEFAULT_RECENCY_WINDOW_DAYS, recency_multiplier: float = DEFAULT_RECENCY_MULTIPLIER) -> None`
  - [x] 3.2: Implement `build_graph(self, games_df: pd.DataFrame, reference_day_num: int | None = None) -> nx.DiGraph` — delegates to `build_season_graph()` with instance config; returns `nx.DiGraph`
  - [x] 3.3: Implement `add_game_to_graph(self, graph: nx.DiGraph, w_team_id: int, l_team_id: int, margin: int, day_num: int, reference_day_num: int) -> None` — in-place update for walk-forward incremental use; weight = `min(margin, self._margin_cap)`; apply recency multiplier if `day_num >= (reference_day_num − self._recency_window_days)`; if edge `(l_team_id, w_team_id)` already exists: `graph[l_team_id][w_team_id]["weight"] += weight`; else: `graph.add_edge(l_team_id, w_team_id, weight=weight)`; also add isolated nodes if not present (`graph.add_node(team_id)`)
  - [x] 3.4: Implement `compute_features(self, graph: nx.DiGraph, pagerank_init: dict[int, float] | None = None) -> pd.DataFrame` — calls all four centrality functions; merges results into a single per-team DataFrame with columns `["team_id", "pagerank", "betweenness_centrality", "hits_hub", "hits_authority", "clustering_coefficient"]`; returns empty DataFrame with these column names if graph has no nodes; all team_ids are integers
  - [x] 3.5: Implement `transform(self, games_df: pd.DataFrame, reference_day_num: int | None = None, pagerank_init: dict[int, float] | None = None) -> pd.DataFrame` — convenience method: calls `build_graph()` then `compute_features()`; returns empty DataFrame with correct columns if `games_df` is empty

- [x] Task 4: Export public API from `src/ncaa_eval/transform/__init__.py` (AC: 1–10)
  - [x] 4.1: Import and re-export `GraphTransformer`, `build_season_graph`, `compute_pagerank`, `compute_betweenness_centrality`, `compute_hits`, `compute_clustering_coefficient` from `transform/graph.py`
  - [x] 4.2: Add all new names to `__all__`

- [x] Task 5: Write unit tests in `tests/unit/test_graph.py` (AC: 11)
  - [x] 5.1: `test_build_season_graph_edge_direction` — two-game fixture (team 1 beat team 2, team 2 beat team 3); verify graph has edges `(2, 1)` and `(3, 2)` (loser→winner direction), NOT `(1, 2)` or `(2, 3)`
  - [x] 5.2: `test_build_season_graph_edge_weight_cap` — blowout game with margin 50; verify edge weight is `min(50, 25) = 25.0`
  - [x] 5.3: `test_build_season_graph_edge_weight_exact` — game with margin 10; verify edge weight is `10.0`
  - [x] 5.4: `test_build_season_graph_recency_weight` — fixture with two games: one recent (within window), one old; verify recent game edge weight = base_weight × 1.5, old game edge weight = base_weight × 1.0
  - [x] 5.5: `test_build_season_graph_parallel_edges_aggregated` — same team pair plays twice (team 1 beats team 2 with margins 10 and 8); verify the single edge weight = 18.0 (sum of both margins)
  - [x] 5.6: `test_build_season_graph_empty_input` — empty DataFrame; verify result is an empty `nx.DiGraph` (0 nodes, 0 edges)
  - [x] 5.7: `test_compute_pagerank_linear_chain` — four-team linear chain: team 1 beat 2 beat 3 beat 4 (all margin 10); verify team 1 has highest PageRank score, team 4 has lowest; verify dict keys are integers (team_ids)
  - [x] 5.8: `test_compute_pagerank_triangle_converges` — three-team cycle (A→B→C→A); verify PageRank converges and returns approximately equal scores for all three teams (within 5% of each other); verify scores sum to approximately 1.0
  - [x] 5.9: `test_compute_pagerank_warm_start` — build a cycle graph, compute PageRank once (nstart=None), then compute again with the previous result as nstart; verify both return values are close (within 1e-6) — warm start produces same result as cold start
  - [x] 5.10: `test_compute_pagerank_empty_graph` — empty DiGraph; verify returns `{}`
  - [x] 5.11: `test_compute_betweenness_linear_chain` — four-team linear chain: 1→2→3→4; verify teams 2 and 3 (bridges) have higher betweenness than teams 1 and 4 (endpoints)
  - [x] 5.12: `test_compute_betweenness_isolated_nodes` — graph with isolated nodes (no edges); verify all betweenness scores are 0.0
  - [x] 5.13: `test_compute_betweenness_empty_graph` — empty DiGraph; verify returns `{}`
  - [x] 5.14: `test_compute_hits_returns_hub_and_authority` — three-node graph; verify `compute_hits()` returns a tuple of two dicts; verify both dicts have the same keys (team_ids)
  - [x] 5.15: `test_compute_hits_hub_high_for_loser_of_high_authority` — four-team fixture: team A beat B, C beat A, C beat D (C is high authority); verify team C has higher authority score than teams B/D; verify team D (lost only to high-authority C) has a hub score > 0
  - [x] 5.16: `test_compute_hits_empty_edges` — DiGraph with 3 nodes but no edges; verify returns two dicts with 0.0 for all nodes (no exception raised)
  - [x] 5.17: `test_compute_clustering_triangle` — three-team cycle (undirected triangle A-B-C); verify all three teams have clustering coefficient 1.0
  - [x] 5.18: `test_compute_clustering_linear_chain` — four-team linear chain; verify all teams have clustering coefficient 0.0 (no triangles in a linear chain)
  - [x] 5.19: `test_compute_clustering_empty_graph` — empty DiGraph; verify returns `{}`
  - [x] 5.20: `test_add_game_to_graph_new_edge` — start with empty DiGraph; add one game; verify graph has edge `(l_team_id, w_team_id)` with correct weight
  - [x] 5.21: `test_add_game_to_graph_accumulates_weight` — add same matchup twice; verify edge weight = sum of both weights
  - [x] 5.22: `test_add_game_to_graph_recency_multiplier` — add game within recency window; verify weight is multiplied by 1.5
  - [x] 5.23: `test_add_game_to_graph_margin_cap` — add game with margin 50; verify edge weight is capped at 25.0
  - [x] 5.24: `test_graph_transformer_transform_columns` — triangle fixture; verify `GraphTransformer().transform(games_df)` returns DataFrame with columns `["team_id", "pagerank", "betweenness_centrality", "hits_hub", "hits_authority", "clustering_coefficient"]`
  - [x] 5.25: `test_graph_transformer_transform_row_count` — triangle fixture (3 teams); verify transform returns exactly 3 rows (one per team)
  - [x] 5.26: `test_graph_transformer_transform_empty_input` — empty games_df; verify transform returns empty DataFrame with the 6 correct columns (no exception)
  - [x] 5.27: `test_graph_transformer_compute_features_consistent` — build_graph then compute_features returns same result as transform() in one call
  - [x] 5.28: `test_no_iterrows` — grep source of `graph.py` for "iterrows"; verify not found (smoke test for mandate compliance)

- [x] Task 6: Commit (AC: all)
  - [x] 6.1: Stage `src/ncaa_eval/transform/graph.py`, `src/ncaa_eval/transform/__init__.py`, `tests/unit/test_graph.py`
  - [x] 6.2: Commit: `feat(transform): implement graph builders and centrality features (Story 4.5)`
  - [x] 6.3: Update `_bmad-output/implementation-artifacts/sprint-status.yaml`: `4-5-implement-graph-builders-centrality-features` → `review`

## Dev Notes

### Story Nature: Fourth Code Story in Epic 4 — graph.py in transform/

This is a **code story** — `mypy --strict`, Ruff, `from __future__ import annotations`, and the no-iterrows mandate all apply. No notebook deliverables.

This story delivers **graph-based centrality feature infrastructure** consumed by:
- Story 4.7 (stateful feature serving) — needs `GraphTransformer` per-team feature DataFrame
- Story 4.8 (Elo feature building block) — independent, but both work on the same compact game data

### 🚨 CRITICAL: Compact Game Data — NOT the Detailed CSV Files

**The single most important architectural fact for this story:**

Graph features are built from **compact game data** (scores, margins, team IDs, day numbers). This IS in the `ParquetRepository` and the `ChronologicalDataServer`. It is NOT in `MRegularSeasonDetailedResults.csv` (which Story 4.4 used for box scores).

The graph module is a **pure transform-layer component**:
- Accept a pre-loaded `pd.DataFrame` of compact games (one row per game)
- Do NOT import from `ncaa_eval.ingest.repository` (no Repository coupling)
- Do NOT load CSV files directly (Story 4.7 will call the Repository and pass the DataFrame)

**Expected compact game DataFrame columns** (from the `Game` schema):
```
w_team_id (int), l_team_id (int), w_score (int), l_score (int),
day_num (int), season (int), loc (str), num_ot (int), is_tournament (bool)
```
The graph module only needs: `w_team_id`, `l_team_id`, `w_score`, `l_score`, `day_num`.

**Coverage:** Compact game data is available for **all seasons (1985–2025)**, unlike detailed box-score data (2003+). Graph features can be computed for the full historical range.

**2025 deduplication:** The 2025 season has 4,545 games stored twice (Kaggle + ESPN IDs). Deduplication by `(w_team_id, l_team_id, day_num)` should happen BEFORE calling graph functions — this is the responsibility of the caller (Story 4.7 or whoever fetches the data). Document this assumption in the module docstring.

### Module Placement

**New file:** `src/ncaa_eval/transform/graph.py`

Per Architecture Section 9, all feature engineering belongs in `src/ncaa_eval/transform/`. Alongside `serving.py`, `normalization.py`, and `sequential.py`.

**Modified file:** `src/ncaa_eval/transform/__init__.py` — add exports for new public API.

### Vectorized Graph Construction (CRITICAL: No iterrows)

**DO NOT** build the graph node-by-node with a loop. Use `nx.from_pandas_edgelist()`:

```python
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
    edge_df = (
        df.groupby(["l_team_id", "w_team_id"], as_index=False)["weight"]
        .sum()
    )

    # Step 4: Build directed graph — loser → winner
    G: nx.DiGraph = nx.from_pandas_edgelist(
        edge_df,
        source="l_team_id",
        target="w_team_id",
        edge_attr="weight",
        create_using=nx.DiGraph,
    )
    return G
```

**Edge direction confirmation:** `source="l_team_id"` (loser), `target="w_team_id"` (winner). A directed edge from L→W means "L votes for W's quality" — teams that receive many votes (from many losers) get high PageRank.

### PageRank Warm Start (for Walk-Forward Efficiency)

`nx.pagerank()` accepts `nstart` parameter — a dictionary mapping node → initial probability. Initialize with the previous solution to converge in 2–5 iterations instead of 30–50 (crucial for walk-forward over 40 seasons):

```python
def compute_pagerank(
    G: nx.DiGraph,
    alpha: float = PAGERANK_ALPHA,
    nstart: dict[int, float] | None = None,
) -> dict[int, float]:
    if G.number_of_nodes() == 0:
        return {}
    return nx.pagerank(G, alpha=alpha, weight="weight", nstart=nstart)
```

**Type annotation note:** Since `nx` is `# type: ignore[import-untyped]`, mypy will accept `nx.DiGraph` as the parameter/return type. The return type of `nx.pagerank()` is `dict[Any, Any]` from mypy's perspective — annotating our function return as `dict[int, float]` is correct at runtime and mypy will not complain given the `type: ignore` on import.

### HITS Convergence Guard

`nx.hits()` can fail to converge for certain graph structures. Always guard:

```python
def compute_hits(
    G: nx.DiGraph,
    max_iter: int = 100,
) -> tuple[dict[int, float], dict[int, float]]:
    if G.number_of_edges() == 0:
        # nx.hits raises ZeroDivisionError or returns zeros for no-edge graphs
        return {n: 0.0 for n in G.nodes()}, {n: 0.0 for n in G.nodes()}
    try:
        hub, auth = nx.hits(G, max_iter=max_iter)
    except nx.PowerIterationFailedConvergence:
        logger.warning(
            "HITS failed to converge after %d iterations; returning uniform scores", max_iter
        )
        n = G.number_of_nodes()
        score = 1.0 / n if n > 0 else 0.0
        uniform: dict[int, float] = dict.fromkeys(G.nodes(), score)
        return uniform, uniform
    return hub, auth  # type: ignore[return-value]
```

The `# type: ignore[return-value]` is needed because `nx.hits()` returns `dict[Any, Any]` from mypy's untyped-import perspective.

### Clustering Coefficient — Use Undirected Conversion

`nx.clustering()` on a `DiGraph` computes directed clustering coefficients (fraction of directed triangles). For the "schedule diversity" metric (do teammates play each other?), the undirected interpretation is more natural:

```python
def compute_clustering_coefficient(G: nx.DiGraph) -> dict[int, float]:
    if G.number_of_nodes() == 0:
        return {}
    return nx.clustering(G.to_undirected())  # type: ignore[return-value]
```

**Semantics:** `G.to_undirected()` merges parallel edges. If A beat B and B beat A (rare but possible), the undirected edge A-B still counts once. This gives the standard undirected clustering coefficient per node.

### `compute_features()` Return Format

The return is a **per-team DataFrame**, not per-game:

```python
def compute_features(
    self,
    graph: nx.DiGraph,
    pagerank_init: dict[int, float] | None = None,
) -> pd.DataFrame:
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=["team_id", "pagerank", "betweenness_centrality",
                                      "hits_hub", "hits_authority", "clustering_coefficient"])

    pr = compute_pagerank(graph, nstart=pagerank_init)
    bc = compute_betweenness_centrality(graph)
    hub, auth = compute_hits(graph)
    cc = compute_clustering_coefficient(graph)

    team_ids = sorted(graph.nodes())
    return pd.DataFrame({
        "team_id": team_ids,
        "pagerank": [pr.get(t, 0.0) for t in team_ids],
        "betweenness_centrality": [bc.get(t, 0.0) for t in team_ids],
        "hits_hub": [hub.get(t, 0.0) for t in team_ids],
        "hits_authority": [auth.get(t, 0.0) for t in team_ids],
        "clustering_coefficient": [cc.get(t, 0.0) for t in team_ids],
    })
```

**Note:** Using `sorted(graph.nodes())` ensures a deterministic row order (sorted by team_id integer). Story 4.7 will join this DataFrame on `team_id` to produce matchup-level delta features.

### Incremental Graph Update (Walk-Forward Support)

For Story 4.7 walk-forward backtesting, games can be added incrementally:

```python
def add_game_to_graph(
    self,
    graph: nx.DiGraph,
    w_team_id: int,
    l_team_id: int,
    margin: int,
    day_num: int,
    reference_day_num: int,
) -> None:
    """Add a single game to an existing graph in-place."""
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
```

**Caller responsibility for walk-forward:** The caller (Story 4.7) is responsible for:
1. Starting with an empty DiGraph at the beginning of each season
2. Adding games one-by-one in chronological order (via `ChronologicalDataServer`)
3. Calling `compute_features(graph, pagerank_init=prev_pagerank)` after each game to get the current state
4. Storing `pagerank_init` (result of previous `compute_pagerank()`) for warm-start efficiency

### NetworkX Import Pattern

```python
from __future__ import annotations

import logging

import networkx as nx  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)
```

NetworkX 3.x (`networkx>=3.0`) is already in `pyproject.toml` per the architecture. No new dependencies required.

**Exception classes**: `nx.PowerIterationFailedConvergence` is a real class in NetworkX 3.x. Access via `nx.PowerIterationFailedConvergence` directly.

### mypy Strict Compliance Notes

- `from __future__ import annotations` — first non-comment line
- `import networkx as nx  # type: ignore[import-untyped]` — networkx has no stubs
- `import pandas as pd  # type: ignore[import-untyped]` — same as all previous transform stories
- `nx.DiGraph` as parameter/return type — accepted by mypy because it's `Any` under `type: ignore[import-untyped]`
- `dict[int, float]` — safe annotation for centrality dicts (no subscript issue with `from __future__ import annotations`)
- `tuple[dict[int, float], dict[int, float]]` — safe return type for `compute_hits()`
- The `# type: ignore[return-value]` on `nx.hits()` and `nx.clustering()` return values is necessary because mypy sees these as returning `Any`, not the more specific types we annotate
- Use `dict.fromkeys(G.nodes(), score)` to build uniform dicts (preferred over dict comprehension for empty-graph guards)
- Type annotation for `G: nx.DiGraph` in `add_game_to_graph()` — since `nx` is `type: ignore[import-untyped]`, mypy accepts this without issue

### Test File Structure

**File:** `tests/unit/test_graph.py`

Follow the pattern from `tests/unit/test_sequential.py` and `tests/unit/test_normalization.py`:

```python
from __future__ import annotations

import networkx as nx  # type: ignore[import-untyped]
import pandas as pd
import pytest

from ncaa_eval.transform.graph import (
    GraphTransformer,
    build_season_graph,
    compute_betweenness_centrality,
    compute_clustering_coefficient,
    compute_hits,
    compute_pagerank,
)
```

**Standard fixtures:**

```python
@pytest.fixture
def triangle_games() -> pd.DataFrame:
    """Three-team cycle: 1 beat 2, 2 beat 3, 3 beat 1."""
    return pd.DataFrame({
        "w_team_id": [1, 2, 3],
        "l_team_id": [2, 3, 1],
        "w_score": [80, 75, 85],
        "l_score": [70, 65, 72],
        "day_num": [30, 40, 50],
    })


@pytest.fixture
def linear_chain_games() -> pd.DataFrame:
    """Four-team chain: 1 beat 2 beat 3 beat 4. No cycles."""
    return pd.DataFrame({
        "w_team_id": [1, 2, 3],
        "l_team_id": [2, 3, 4],
        "w_score": [80, 75, 70],
        "l_score": [70, 65, 62],
        "day_num": [30, 40, 50],
    })


@pytest.fixture
def triangle_graph(triangle_games: pd.DataFrame) -> nx.DiGraph:
    return build_season_graph(triangle_games)


@pytest.fixture
def chain_graph(linear_chain_games: pd.DataFrame) -> nx.DiGraph:
    return build_season_graph(linear_chain_games)
```

**Markers:**
- `@pytest.mark.smoke` on fast unit tests (< 1s each)
- `@pytest.mark.unit` on all tests in this file

### Previous Story Learnings (from Stories 4.2, 4.3, 4.4)

- **`from __future__ import annotations` required** — first non-comment line; enforced by Ruff
- **`logger = logging.getLogger(__name__)`** at module level (not inside functions)
- **Every public method must have at least one test** — story 4.3 code review established this mandate; violations found for `composite_pca()`. Map every public method to a test ID before starting implementation.
- **Empty-input guards** — always add `if df.empty: return empty_result` before heavy computation (prevents ZeroDivisionError, pandas warnings, networkx exceptions)
- **`frozen=True` dataclasses** — if used, remember it only prevents attribute rebinding, not mutation of mutable contents
- **Warning capture in tests** — use `unittest.mock.patch("ncaa_eval.transform.graph.logger")` not `caplog` (the `ncaa_eval` logger has `propagate=False`)
- **No direct I/O from transform module** — this module does NOT load files; it accepts pre-loaded DataFrames
- **`Iterator` from `collections.abc`** — UP035 rule; do not import from `typing`
- **L3 from Story 4.4 review (informational):** `compute_ewma_stats`, `compute_momentum`, `compute_rolling_stats`, `apply_ot_rescaling`, `compute_per_possession_stats` are not yet exported from `ncaa_eval.transform.__init__`. Story 4.7 will need them — consider adding them while you are editing `__init__.py` in this story (low-risk, 5 lines of additional export code). This is optional but helps Story 4.7.

### Architecture Guardrails (Mandatory)

1. **`from __future__ import annotations` required** — first non-comment line
2. **`mypy --strict` mandatory** — zero errors; use `# type: ignore[import-untyped]` for networkx/pandas
3. **Vectorization First** — `nx.from_pandas_edgelist()` for graph construction; NO `for` loops over DataFrame rows; NO `iterrows()`
4. **No iterrows** — use `groupby().sum()` for edge weight aggregation
5. **No direct I/O** — `graph.py` does not load CSV files or import from `ncaa_eval.ingest.repository`
6. **No imports from `ncaa_eval.ingest`** — pure transform-layer component; data is provided by caller
7. **Do NOT import from `sequential.py`** — no dependency between graph and sequential modules (they are parallel building blocks consumed by Story 4.7)

### What NOT to Do

- **Do not** use `iterrows()` or any Python `for` loop over game rows to build the graph
- **Do not** load data from `data/kaggle/*.csv` directly — this module receives pre-loaded DataFrames
- **Do not** use `nx.add_edge()` in a loop to build the initial graph — use `nx.from_pandas_edgelist()`
- **Do not** import from `ncaa_eval.ingest.repository` or `ncaa_eval.transform.serving` (the graph module has no dependency on those)
- **Do not** implement opponent adjustment rating systems (SRS, Ridge, Colley, Elo) — that belongs in Story 4.6
- **Do not** implement the feature serving layer — that belongs in Story 4.7
- **Do not** skip tests for empty-graph edge cases — networkx behaves unexpectedly on empty/isolated-node graphs
- **Do not** ignore the HITS convergence failure case — it will crash walk-forward pipelines on unusual schedule structures
- **Do not** implement Women's data — Men's only for MVP

### No New Dependencies Required

NetworkX is already in `pyproject.toml` per the architecture spec (Section 3: "NetworkX | Latest | Feature Engineering"). Verify it's importable:
```bash
conda run -n ncaa_eval python -c "import networkx; print(networkx.__version__)"
```
Expected: `3.x`

All other needed libraries (pandas, numpy) are already present.

### Running Quality Checks

```bash
POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval ruff check .
POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval mypy --strict src/ncaa_eval tests
POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval pytest tests/unit/test_graph.py -v
```

### Project Structure Notes

**New files:**
- `src/ncaa_eval/transform/graph.py` — `build_season_graph`, `compute_pagerank`, `compute_betweenness_centrality`, `compute_hits`, `compute_clustering_coefficient`, `GraphTransformer`
- `tests/unit/test_graph.py` — 28 unit tests

**Modified files:**
- `src/ncaa_eval/transform/__init__.py` — add exports for new public API (and optionally add the Story 4.4 L3 missing exports)
- `_bmad-output/implementation-artifacts/sprint-status.yaml` — update status to `review`
- `_bmad-output/implementation-artifacts/4-5-implement-graph-builders-centrality-features.md` — this story file

**No changes to:**
- `src/ncaa_eval/ingest/` (stable)
- `src/ncaa_eval/transform/serving.py` (stable)
- `src/ncaa_eval/transform/normalization.py` (stable)
- `src/ncaa_eval/transform/sequential.py` (stable — but see optional L3 note above)
- `pyproject.toml` (no new dependencies)
- Any existing test files

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 4.5 — Acceptance Criteria]
- [Source: _bmad-output/planning-artifacts/epics.md#Epic 4 — Feature Engineering Suite (FR5: Graph Representations)]
- [Source: specs/research/feature-engineering-techniques.md#Section 4 — Graph-Based Features (PageRank, betweenness, HITS, clustering)]
- [Source: specs/research/feature-engineering-techniques.md#Section 4.1 — PageRank on Win/Loss Directed Graph (Matthews et al. 2021 validation)]
- [Source: specs/research/feature-engineering-techniques.md#Section 4.4 — Incremental Graph Update Strategy (warm start)]
- [Source: specs/research/feature-engineering-techniques.md#Section 7.2 — Distinct Building Blocks (graph centrality features)]
- [Source: specs/05-architecture-fullstack.md#Section 3 — Tech Stack (NetworkX Latest, Feature Engineering)]
- [Source: specs/05-architecture-fullstack.md#Section 9 — Unified Project Structure (transform/ module)]
- [Source: specs/05-architecture-fullstack.md#Section 12 — Coding Standards (mypy --strict, vectorization)]
- [Source: _bmad-output/implementation-artifacts/4-4-implement-sequential-transformations.md#Dev Notes — mypy patterns, no-iterrows, every-public-method-tested mandate, empty-guard pattern]
- [Source: _bmad-output/implementation-artifacts/4-4-implement-sequential-transformations.md#Review Follow-ups — L3 (compute_ewma_stats et al. not exported from __init__)]
- [Source: _bmad-output/planning-artifacts/template-requirements.md — ML .fit() empty guard, empty-dict ZeroDivisionError guard, all-methods-tested mandate]
- [Matthews et al. 2021: "PageRank as a Method for Ranking NCAA Division I Men's Basketball Teams" — 71.6% vs 64.2% accuracy]

## Dev Agent Record

### Agent Model Used

Claude Sonnet 4.6 (create-story workflow)

### Debug Log References

- mypy `# type: ignore[return-value]` does not suppress `no-any-return` — must use `# type: ignore[no-any-return]` for nx function returns.
- Single-game test fixtures with low day_num default to reference_day_num = max(day_num) = same day, so recency window always applies. Tests for cap/exact weight must explicitly pass `reference_day_num=100` to put the game outside the recency window.
- `hub, auth = nx.hits(...)` return does not need a type ignore — mypy accepts `Any` for the tuple assignment without raising `no-any-return`.

### Completion Notes List

- Implemented `src/ncaa_eval/transform/graph.py` with all 4 module-level centrality functions (`compute_pagerank`, `compute_betweenness_centrality`, `compute_hits`, `compute_clustering_coefficient`) and `GraphTransformer` class (5 methods: `build_graph`, `add_game_to_graph`, `compute_features`, `transform`, `__init__`).
- All graph construction is fully vectorized via `nx.from_pandas_edgelist()` — no iterrows.
- HITS convergence failure guarded with try/except `nx.PowerIterationFailedConvergence`.
- PageRank warm-start supported via `nstart` parameter.
- Also added Story 4.4 L3 missing exports (`apply_ot_rescaling`, `compute_ewma_stats`, `compute_momentum`, `compute_per_possession_stats`, `compute_rolling_stats`) to `transform/__init__.py`.
- 28 unit tests written and passing. 264 total tests pass (no regressions).
- All quality gates pass: ruff, mypy --strict, pytest.

### File List

- src/ncaa_eval/transform/graph.py (new)
- src/ncaa_eval/transform/__init__.py (modified — graph exports + Story 4.4 L3 sequential exports)
- tests/unit/test_graph.py (new)
- _bmad-output/implementation-artifacts/4-5-implement-graph-builders-centrality-features.md (modified)
- _bmad-output/implementation-artifacts/sprint-status.yaml (modified)

## Senior Developer Review (AI)

Reviewed 2026-02-21 by Claude Sonnet 4.6 (code-review workflow).

**Outcome:** APPROVED — 4 issues fixed (1 HIGH, 3 MEDIUM)

| Severity | Finding | Fix Applied |
|----------|---------|-------------|
| HIGH | `compute_hits()` convergence failure branch (PowerIterationFailedConvergence) had zero test coverage — silent failure mode in walk-forward pipelines | Added `test_compute_hits_convergence_failure_returns_uniform` (test_graph.py) |
| MEDIUM | `compute_hits()` returned the same mutable dict object for both hub and auth on convergence failure — mutating hub would corrupt auth | Fixed `graph.py:172-173`: return two independent `dict.fromkeys()` objects |
| MEDIUM | `test_no_iterrows` used `Path(__file__)` navigation but was not marked `@pytest.mark.no_mutation` — would fail with FileNotFoundError under mutmut | Added `@pytest.mark.no_mutation` marker |
| MEDIUM | `test_compute_hits_hub_high_for_loser_of_high_authority` had a tautological assertion `hub[4] >= 0.0` (always true by construction) | Replaced with `hub[4] > 0.0`, `hub[1] > 0.0`, and `hub[4] >= hub[2]` (meaningful hub signal validation) |

**Low issues (deferred):**
- L1: `transform()` warm-start path untested through convenience method (underlying path covered via `compute_pagerank(nstart=...)`)
- L2: Module docstring missing betweenness O(V×E) recomputation caching guidance for Story 4.7

**Quality gate results after fixes:** ruff ✓ · mypy --strict ✓ · 265 tests pass (0 regressions)

## Change Log

- 2026-02-21: Implemented graph builders and centrality features (Story 4.5) — created `graph.py` with `build_season_graph`, `compute_pagerank`, `compute_betweenness_centrality`, `compute_hits`, `compute_clustering_coefficient`, and `GraphTransformer`; wrote 28 unit tests; updated `transform/__init__.py` with graph exports and Story 4.4 L3 missing sequential exports; all quality gates pass (ruff, mypy --strict, 264 tests).
- 2026-02-21: Code review — fixed 1 HIGH (HITS convergence failure untested) + 3 MEDIUM issues (shared mutable dict, missing no_mutation marker, tautological assertion); 29 tests now, 265 total.
