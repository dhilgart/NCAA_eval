# Noncompliant Docstrings — Functions with 3+ Operations Missing Detailed Description

**Date:** 2026-03-03
**Source:** Story 8.9 PO directive — docstrings for functions with 3+ operations must include
a detailed description paragraph explaining *how* (not just *what*).
**Tracking:** Story 8.4 (Fix Docstring Style Violations & Documentation Gaps)

---

## Rule

When a function performs 3 or more operations, the docstring must include a detailed
description paragraph (after the summary line) explaining *how* the function implements
its purpose. A single-line summary is insufficient.

---

## Noncompliant Functions (28 total)

| File | Line | Function | Ops | Current Summary |
|------|------|----------|-----|-----------------|
| `cli/main.py` | 44 | `train` | 5 | `Train a model on NCAA basketball data and persist run artifacts.` |
| `evaluation/metrics.py` | 51 | `_validate_inputs` | 4 | `Validate metric inputs: non-empty, matching lengths, binary y_true, probs in [0, 1].` |
| `evaluation/simulation.py` | 279 | `matchup_probability` | 3 | `Return P(team_a beats team_b) from the stored matrix.` |
| `evaluation/simulation.py` | 290 | `batch_matchup_probabilities` | 4 | `Return batch probabilities from the stored matrix.` |
| `evaluation/simulation.py` | 771 | `_traverse` | 7 | `Post-order traversal returning WPV at this node.` |
| `evaluation/simulation.py` | 865 | `_traverse_bonus` | 9 | `Post-order traversal returning WPV and accumulating bonus EP.` |
| `evaluation/simulation.py` | 926 | `_traverse` | 12 | `Return team index of the predicted winner at this node.` |
| `ingest/connectors/espn.py` | 133 | `_fetch_per_team` | 6 | `Fetch schedules for each team in the mapping and concatenate.` |
| `ingest/connectors/espn.py` | 157 | `_parse_schedule_df` | 7 | `Convert a cbbpy schedule DataFrame into Game models.` |
| `ingest/connectors/kaggle.py` | 164 | `fetch_teams` | 3 | `Parse MTeams.csv into Team models.` |
| `ingest/connectors/kaggle.py` | 198 | `fetch_seasons` | 3 | `Parse MSeasons.csv into Season models.` |
| `ingest/connectors/kaggle.py` | 206 | `_parse_games_csv` | 6 | `Parse a single games CSV, filtering to *season*.` |
| `model/base.py` | 87 | `fit` | 3 | `Reconstruct games from *X*/*y* and update sequentially.` |
| `model/elo.py` | 68 | `_predict_one` | 3 | `Return P(team_a wins) using the Elo expected-score formula.` |
| `model/elo.py` | 122 | `save` | 5 | `JSON-dump config and state to *path* directory.` |
| `model/tracking.py` | 88 | `save_run` | 5 | `Write run metadata (JSON) and predictions (Parquet).` |
| `model/tracking.py` | 282 | `list_runs` | 4 | `Scan the runs directory and return all saved ModelRun records.` |
| `transform/elo.py` | 277 | `_effective_k` | 4 | `Determine K-factor based on game count and tournament flag.` |
| `transform/feature_serving.py` | 91 | `active_blocks` | 8 | `Return the set of feature blocks that are currently enabled.` |
| `transform/feature_serving.py` | 219 | `_serve_batch` | 9 | `Compute features for all games at once (batch mode).` |
| `transform/feature_serving.py` | 243 | `_append_per_game_columns_batch` | 10 | `Collect per-game feature values and assign as DataFrame columns.` |
| `transform/feature_serving.py` | 330 | `_build_game_row` | 5 | `Build a single game row dict for stateful mode.` |
| `transform/feature_serving.py` | 377 | `_resolve_ordinal_systems` | 3 | `Determine which ordinal systems to use.` |
| `transform/feature_serving.py` | 389 | `_get_seed_nums` | 4 | `Get seed numbers for both teams. NaN if not in tournament or unseeded.` |
| `transform/feature_serving.py` | 402 | `_compute_batch_ratings` | 8 | `Compute batch ratings from regular-season games only.` |
| `transform/feature_serving.py` | 479 | `_compute_matchup_deltas` | 5 | `Compute team_A - team_B deltas for all active features.` |
| `transform/opponent.py` | 156 | `_build_team_index` | 5 | `Build sorted team list, index mapping, and vectorized index arrays.` |
| `transform/opponent.py` | 172 | `_build_srs_matrices` | 13 | `Build net_margin, n_games, avg_margin, and normalized adjacency matrix for SRS.` |

### Breakdown by Module

| Module | Count | Notes |
|--------|-------|-------|
| `transform/feature_serving.py` | 8 | Highest concentration — batch feature computation |
| `evaluation/simulation.py` | 5 | Includes nested `_traverse` closures |
| `ingest/connectors/` | 5 | Parsing pipelines (ESPN + Kaggle) |
| `model/` | 4 | Base, Elo, tracking |
| `transform/` (other) | 3 | Elo engine, opponent stats |
| `cli/` | 1 | Typer CLI entry point |
| `evaluation/metrics.py` | 1 | Input validation |
