# Story 4.6: Implement Batch Opponent Adjustment Rating Systems

Status: review

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a data scientist,
I want batch linear algebra rating solvers (SRS, Ridge, Colley) that produce opponent-adjusted team ratings for the full season,
so that I can generate features that account for schedule strength and quality of competition.

## Acceptance Criteria

1. **Given** full-season regular-season game data with scores and team matchup information is available, **When** the developer runs the SRS solver, **Then** **SRS (Simple Rating System)** is implemented as the Group A canonical representative: fixed-point iteration solve (`r_i(k+1) = avg_margin_i + avg(r_j for all opponents j)`); convergence guaranteed for connected schedules (~3,000–5,000 iterations); produces margin-adjusted batch rating.

2. **And** **Ridge regression** is implemented as the Group A λ-parameterized variant: regularized SRS via `sklearn.linear_model.Ridge`; λ (stored as `ridge_lambda` in solver config; mapped to Ridge's `alpha` parameter) is configurable in range 10–100 (default λ=20 for full-season data); exposes shrinkage as a modeler-visible tuning knob without providing a distinct signal from SRS.

3. **And** **Colley Matrix** is implemented as the Group B representative (win/loss only): build matrix `C[i,i] = 2 + t_i`, `C[i,j] = -n_ij`; RHS `b[i] = 1 + (w_i - l_i)/2`; solve via `numpy.linalg.solve(C, b)` (n≤350 makes this trivial); **or** the pre-computed "COL" system from `MMasseyOrdinals.csv` (ingested in Story 4.3) may be used as an alternative — implementation choice resolved during Story 4.6 development. **Recommended: implement the solver** (self-contained, no dependency on COL data availability).

4. **And** all three solvers produce **full-season pre-tournament snapshots** (ratings computed from all games where `is_tournament == False`); NOT in-season incremental updates (that is Story 4.8's responsibility for Elo).

5. **And** the solvers handle edge cases: teams with very few games, structurally isolated conference subgraphs (near-singular sub-blocks for SRS/Ridge), unconnected schedule components (Colley: use `numpy.linalg.lstsq` fallback when `numpy.linalg.solve` encounters a singular matrix).

6. **And** outputs are validated against the pre-computed "MAS" (Massey) system in `MMasseyOrdinals.csv` for sanity-check benchmarking (Spearman rank correlation ≥ 0.85 with MAS ordinals expected for correctly implemented SRS); this is a **manual validation step** documented in the Dev Agent Record, not an automated unit test.

7. **And** note: Elo (dynamic game-by-game rating as a feature building block) is implemented in Story 4.8, not here. Story 5.3 implements Elo as a complete predictive model. These are architecturally distinct.

8. **And** the solvers are covered by unit tests in `tests/unit/test_opponent.py` including: convergence assertions (SRS), lambda-sensitivity tests (Ridge), and win/loss isolation tests (Colley).

## Tasks / Subtasks

- [x] Task 1: Create `src/ncaa_eval/transform/opponent.py` — module header, constants, class, and module-level convenience functions (AC: 1–7)
  - [x] 1.1: Module header: `from __future__ import annotations`, imports (`logging`, `numpy as np`, `pandas as pd`, `Ridge` from sklearn), module-level logger
  - [x] 1.2: Define constants: `DEFAULT_MARGIN_CAP: int = 25`, `DEFAULT_RIDGE_LAMBDA: float = 20.0`, `DEFAULT_SRS_MAX_ITER: int = 10_000`, `_SRS_CONVERGENCE_TOL: float = 1e-6`
  - [x] 1.3: Implement `BatchRatingSolver` class with `__init__(self, margin_cap, ridge_lambda, srs_max_iter)` — all keyword-only with defaults; stores `self._margin_cap`, `self._ridge_lambda`, `self._srs_max_iter`
  - [x] 1.4: Implement `BatchRatingSolver.compute_srs(self, games_df: pd.DataFrame) -> pd.DataFrame` — full SRS solver; returns DataFrame with columns `["team_id", "srs_rating"]`; see Dev Notes for vectorized implementation; empty-input guard returns empty DataFrame with correct columns
  - [x] 1.5: Implement `BatchRatingSolver.compute_ridge(self, games_df: pd.DataFrame) -> pd.DataFrame` — Ridge regression solver; returns DataFrame with columns `["team_id", "ridge_rating"]`; empty-input guard
  - [x] 1.6: Implement `BatchRatingSolver.compute_colley(self, games_df: pd.DataFrame) -> pd.DataFrame` — Colley Matrix solver; returns DataFrame with columns `["team_id", "colley_rating"]`; empty-input guard
  - [x] 1.7: Implement module-level convenience functions `compute_srs_ratings(games_df, *, margin_cap, max_iter)`, `compute_ridge_ratings(games_df, *, lam, margin_cap)`, `compute_colley_ratings(games_df)` — create a `BatchRatingSolver` instance with the given config and call the corresponding method; these are the primary exported public API

- [x] Task 2: Export public API from `src/ncaa_eval/transform/__init__.py` (AC: 1–8)
  - [x] 2.1: Import and re-export `BatchRatingSolver`, `compute_srs_ratings`, `compute_ridge_ratings`, `compute_colley_ratings` from `transform/opponent.py`
  - [x] 2.2: Add all new names to `__all__`

- [x] Task 3: Write unit tests in `tests/unit/test_opponent.py` (AC: 8)
  - [x] 3.1: `test_srs_linear_chain_ordering` — 4-team linear chain (team 1 beat 2 beat 3 beat 4, each margin 10); verify `srs_rating[1] > srs_rating[2] > srs_rating[3] > srs_rating[4]`
  - [x] 3.2: `test_srs_convergence_assertion` — 6-team balanced round-robin; compute SRS once, then again; verify results are within 1e-4 (deterministic convergence assertion)
  - [x] 3.3: `test_srs_symmetric_wins_equal_ratings` — team A beat B (margin 10), B beat A (margin 10); verify `srs_rating[A] ≈ srs_rating[B]` within 1e-6 (symmetric schedule → identical ratings)
  - [x] 3.4: `test_srs_margin_cap_applied` — blowout fixture (margin 100); verify SRS with `margin_cap=25` produces same result as SRS on a fixture with margin capped at 25 pre-call (cap is applied inside the solver)
  - [x] 3.5: `test_srs_ratings_sum_to_zero` — any connected fixture; verify `sum(srs_ratings) ≈ 0.0` within 1e-4 (ratings are zero-centered by convention)
  - [x] 3.6: `test_srs_empty_input` — empty DataFrame; verify returns empty DataFrame with columns `["team_id", "srs_rating"]` (no exception)
  - [x] 3.7: `test_ridge_lambda_high_shrinks_ratings` — same fixture with `lam=1.0` vs `lam=100.0`; verify `mean(abs(ridge_rating[lam=100])) < mean(abs(ridge_rating[lam=1]))` (higher λ → ratings closer to zero)
  - [x] 3.8: `test_ridge_lambda_sensitivity_ordering_preserved` — same fixture at two λ values; verify team ranking order (by rating value) is the same (λ affects magnitude, not direction of ratings)
  - [x] 3.9: `test_ridge_empty_input` — empty DataFrame; verify returns empty DataFrame with columns `["team_id", "ridge_rating"]` (no exception)
  - [x] 3.10: `test_colley_ignores_margin` — two fixtures: same wins/losses but different margins (team A beat B by 5 in fixture 1, by 50 in fixture 2); verify `colley_rating` is IDENTICAL between fixtures (win/loss only)
  - [x] 3.11: `test_colley_win_loss_ratio_reflected` — team A wins 8 of 10 games vs team B wins 2 of 10 games (same opponents); verify `colley_rating[A] > colley_rating[B]`
  - [x] 3.12: `test_colley_ratings_bounded` — verify all Colley ratings in fixture are within [0, 1] (guaranteed by Colley construction)
  - [x] 3.13: `test_colley_empty_input` — empty DataFrame; verify returns empty DataFrame with columns `["team_id", "colley_rating"]` (no exception)
  - [x] 3.14: `test_all_methods_return_correct_schema` — verify each method returns a DataFrame with `team_id` column (int dtype) and method-specific rating column (float dtype); row count equals number of unique teams in fixture
  - [x] 3.15: `test_convenience_functions_match_class_methods` — verify `compute_srs_ratings(df)` produces same result as `BatchRatingSolver().compute_srs(df)` for the same fixture
  - [x] 3.16: `@pytest.mark.no_mutation` `test_no_iterrows` — verify `opponent.py` source does not contain "iterrows" (uses `Path(__file__)` navigation — must be `no_mutation` marked)

- [x] Task 4: Manual sanity-check validation against MAS ordinals (AC: 6)
  - [x] 4.1: Load a full season (e.g., 2023) compact game data from Parquet; filter to `is_tournament == False`; run `compute_srs_ratings()`
  - [x] 4.2: Load MAS ordinals for 2023 via `MasseyOrdinalsStore.pre_tournament_snapshot(season=2023, systems=["MAS"])`; extract pre-tournament snapshot
  - [x] 4.3: Compute Spearman rank correlation between SRS ratings (higher = better) and MAS ordinals (lower rank = better, so invert); verify correlation ≥ 0.85
  - [x] 4.4: Document results in Dev Agent Record completion notes

- [x] Task 5: Quality gates and commit (AC: all)
  - [x] 5.1: Run `POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval ruff check .`
  - [x] 5.2: Run `POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval mypy --strict src/ncaa_eval tests`
  - [x] 5.3: Run `POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval pytest tests/unit/test_opponent.py -v`
  - [x] 5.4: Commit: `feat(transform): implement batch opponent adjustment rating systems (Story 4.6)`
  - [x] 5.5: Update `_bmad-output/implementation-artifacts/sprint-status.yaml`: `4-6-implement-opponent-adjustments` → `review`

## Dev Notes

### Story Nature: Fifth Code Story in Epic 4 — opponent.py in transform/

This is a **code story** — `mypy --strict`, Ruff, `from __future__ import annotations`, and the no-iterrows mandate all apply. No notebook deliverables.

This story delivers **batch rating solvers** consumed by:
- Story 4.7 (stateful feature serving) — needs pre-tournament rating snapshots per team
- Story 4.8 (Elo feature building block) — independent; Elo is dynamic, not batch

### 🚨 CRITICAL: Scope Boundaries

**THIS story implements:** SRS, Ridge, Colley (full-season batch solvers)

**NOT this story — do not implement:**
- Elo (game-by-game dynamic ratings) → Story 4.8
- Elo as a complete predictive Model → Story 5.3
- LRMC (Logistic Regression Markov Chain) → **Post-MVP Backlog** (too complex, deferred)
- TrueSkill / Glicko-2 → **Post-MVP Backlog** (marginal over Elo, deferred)
- In-season incremental rating updates → Story 4.8 (Elo) / Story 4.7 (feature serving)

### Module Placement

**New file:** `src/ncaa_eval/transform/opponent.py`

Per Architecture Section 9, all feature engineering belongs in `src/ncaa_eval/transform/`. Alongside `serving.py`, `normalization.py`, `sequential.py`, and `graph.py`.

**Modified file:** `src/ncaa_eval/transform/__init__.py` — add exports for new public API.

### Data Contract: What games_df Must Look Like

The solvers accept a pre-loaded `pd.DataFrame` of compact regular-season games:

```
w_team_id (int)   — winning team ID
l_team_id (int)   — losing team ID
w_score   (int)   — winning team score (for SRS/Ridge margin)
l_score   (int)   — losing team score (for SRS/Ridge margin)
```

**Required for Colley only:** `w_team_id`, `l_team_id` (no scores needed)

**Caller responsibility:** Filter to `is_tournament == False` BEFORE calling any solver. The solvers compute full-season batch ratings from regular-season data only. Passing tournament games will silently corrupt the batch ratings (tournament outcomes don't belong in the season's batch rating).

**2025 deduplication:** Caller must deduplicate 2025 data by `(w_team_id, l_team_id, day_num)` before calling (4,545 games stored twice; dedup is ChronologicalDataServer's responsibility in Story 4.7).

**Coverage:** Compact game data (`w_score`, `l_score`) is available for all seasons 1985–2025. These solvers can be computed for the full historical range.

### SRS Implementation (Vectorized NumPy)

**Algorithm:** Fixed-point iteration. For team i:
```
r_i(k+1) = avg_margin_i + avg(r_j for all opponents j of i)
```

**Vectorized implementation using NumPy scatter:**

```python
import numpy as np  # type: ignore[import-untyped]
import numpy.typing as npt

def compute_srs(self, games_df: pd.DataFrame) -> pd.DataFrame:
    """Compute SRS ratings via fixed-point iteration."""
    if games_df.empty:
        return pd.DataFrame(columns=["team_id", "srs_rating"])

    # Build team index
    teams = sorted(set(games_df["w_team_id"]) | set(games_df["l_team_id"]))
    n = len(teams)
    idx: dict[int, int] = {t: i for i, t in enumerate(teams)}

    # Vectorized game arrays
    w_idx = games_df["w_team_id"].map(idx).to_numpy()
    l_idx = games_df["l_team_id"].map(idx).to_numpy()
    raw_margins = (games_df["w_score"] - games_df["l_score"]).to_numpy(dtype=float)
    margins = np.minimum(raw_margins, float(self._margin_cap))

    # Accumulate net margins and game counts
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
```

**Key notes:**
- `np.add.at` is the vectorized scatter operation — no Python loops over games
- `A_norm @ r` is a standard matrix-vector product — no Python loops
- Zero-centering (`r -= mean(r)`) enforces a unique solution for disconnected schedules
- Convergence is guaranteed for connected schedules; for disconnected schedules, ratings converge per-component but may drift without zero-centering

**Ruff PLR0913 check:** `compute_srs(self, games_df)` = 2 params total. Fine.

### Ridge Implementation

Frame the rating problem as regularized regression. For each game `g`:
- `X[g, idx[w_team]] = +1`, `X[g, idx[l_team]] = -1`
- `y[g] = min(w_score - l_score, margin_cap)`

Ridge solves: `min ‖Xr − y‖² + λ‖r‖²`

```python
from sklearn.linear_model import Ridge

def compute_ridge(self, games_df: pd.DataFrame) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame(columns=["team_id", "ridge_rating"])

    teams = sorted(set(games_df["w_team_id"]) | set(games_df["l_team_id"]))
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    n_games = len(games_df)
    X: npt.NDArray[np.float64] = np.zeros((n_games, n))
    w_idx = games_df["w_team_id"].map(idx).to_numpy()
    l_idx = games_df["l_team_id"].map(idx).to_numpy()
    raw_margins = (games_df["w_score"] - games_df["l_score"]).to_numpy(dtype=float)
    y = np.minimum(raw_margins, float(self._margin_cap))

    # Build design matrix (vectorized scatter)
    game_indices = np.arange(n_games)
    X[game_indices, w_idx] = 1.0
    X[game_indices, l_idx] = -1.0

    model = Ridge(alpha=self._ridge_lambda, fit_intercept=False)
    model.fit(X, y)
    ratings: list[float] = model.coef_.tolist()  # type: ignore[union-attr]

    return pd.DataFrame({"team_id": teams, "ridge_rating": ratings})
```

**Key notes:**
- `fit_intercept=False` is CRITICAL — no global offset; ratings are relative to each other
- `alpha=self._ridge_lambda` — sklearn's `alpha` IS our λ (shrinkage parameter)
- Higher λ → ratings shrink toward zero (regular-season balance with 30+ games per team; λ=20 is appropriate)
- Ridge always produces a full-rank solution even for disconnected conference subgraphs (λI regularization prevents singularity)
- For mypy: `model.coef_` is typed as `np.ndarray | None` in some sklearn versions — add `# type: ignore[union-attr]` if needed

**Design matrix memory:** For n=350 teams and 5,800 games, X is 5,800 × 350 = ~2M floats × 8 bytes = ~16MB. Acceptable.

### Colley Implementation

The Colley Matrix method produces win/loss-only ratings (no margin information). It is the Group B building block — distinct from SRS/Ridge because it explicitly ignores scoring.

**Matrix construction:**

For n teams:
- `C[i,i] = 2 + t_i` where `t_i = total games played by team i`
- `C[i,j] = -n_ij` where `n_ij = number of games between team i and team j`
- `b[i] = 1 + (w_i - l_i) / 2` where `w_i = wins`, `l_i = losses`

**Colley matrix C is always symmetric positive definite** for connected schedules → `numpy.linalg.solve(C, b)` is guaranteed to succeed.

```python
def compute_colley(self, games_df: pd.DataFrame) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame(columns=["team_id", "colley_rating"])

    teams = sorted(set(games_df["w_team_id"]) | set(games_df["l_team_id"]))
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    w_idx = games_df["w_team_id"].map(idx).to_numpy()
    l_idx = games_df["l_team_id"].map(idx).to_numpy()

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
        r: npt.NDArray[np.float64] = np.linalg.solve(C, b)
    except np.linalg.LinAlgError:
        # Singular matrix (disconnected schedule) — fall back to lstsq
        logger.warning("Colley matrix is singular (disconnected schedule); using lstsq fallback")
        r, _, _, _ = np.linalg.lstsq(C, b, rcond=None)

    return pd.DataFrame({"team_id": teams, "colley_rating": r.tolist()})
```

**Key notes:**
- Colley ratings are **bounded [0, 1]** by construction (Bayesian derivation)
- The `.5 + (w-l)/(2t)` correction distributes wins/losses through the schedule graph
- `np.linalg.solve` is O(n³) for n=350 — takes milliseconds; no need for scipy's cho_factor
- LinAlgError fallback handles the rare case of completely disconnected sub-graphs (all games within one conference with no crossover)
- The `np.linalg.lstsq` signature returns a 4-tuple: `(solution, residuals, rank, sv)` — unpack all 4

**Pre-computed COL alternative:** The MasseyOrdinalsStore (Story 4.3) already ingests the "COL" system. If the solver implementation fails for any reason, the developer can fall back to: `store.get_composite_ratings(systems=["COL"], season=year, max_day_num=128)`. However, implementing the solver is strongly recommended for self-containment.

### scipy Not Required

`numpy.linalg.solve` is sufficient for the Colley matrix (n ≤ 350 teams, symmetric positive definite). scipy would offer `scipy.linalg.cho_factor` + `cho_solve` (Cholesky decomposition, ~2× faster) but the speed difference is negligible for n=350 (~0.1ms vs ~0.05ms). **scipy is NOT in `pyproject.toml`** — do NOT add it as a dependency for this story.

Verify before committing:
```bash
grep -r "scipy" src/ncaa_eval/transform/opponent.py  # should find nothing
```

### Module-Level Convenience Functions

These wrap the class for external callers who don't need to configure a solver:

```python
def compute_srs_ratings(
    games_df: pd.DataFrame,
    *,
    margin_cap: int = DEFAULT_MARGIN_CAP,
    max_iter: int = DEFAULT_SRS_MAX_ITER,
) -> pd.DataFrame:
    """Compute SRS ratings using default solver config."""
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
    """Compute Ridge regression ratings."""
    return BatchRatingSolver(
        margin_cap=margin_cap,
        ridge_lambda=lam,
    ).compute_ridge(games_df)


def compute_colley_ratings(games_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Colley Matrix win/loss-only ratings."""
    return BatchRatingSolver().compute_colley(games_df)
```

Note: `lam` is used instead of `lambda` (Python keyword). The BatchRatingSolver constructor maps `ridge_lambda` → Ridge's `alpha` parameter internally.

### mypy Strict Compliance Notes

- `from __future__ import annotations` — first non-comment line (Ruff UP035 enforced)
- `import numpy as np  # type: ignore[import-untyped]` — check first without type:ignore; if mypy raises `error: Library stubs not installed for "numpy"` then add it
- `import numpy.typing as npt` — does NOT need type:ignore (numpy ships py.typed); use `npt.NDArray[np.float64]` for array annotations
- `from sklearn.linear_model import Ridge` — try without type:ignore first; if mypy raises untyped-import error, add `# type: ignore[import-untyped]`
- `model.coef_` — mypy may type this as `np.ndarray | None`; use `.tolist()` with `# type: ignore[union-attr]` or add a None guard
- `np.linalg.lstsq` returns `tuple[NDArray, NDArray, int, NDArray]` — unpack all 4 elements even if only the first is used
- For `dict[int, int]` (team index): no subscript issues due to `from __future__ import annotations`
- Cyclomatic complexity (C90): SRS iteration loop adds branches — if complexity > 10, extract the inner loop body to a helper function

**Ruff PLR0913 (max 5 args):**
- `BatchRatingSolver.__init__(self, margin_cap, ridge_lambda, srs_max_iter)` = 4 total ✓
- `BatchRatingSolver.compute_srs(self, games_df)` = 2 total ✓
- `compute_srs_ratings(games_df, *, margin_cap, max_iter)` = 3 total ✓
- All other methods/functions: ≤ 3 params ✓

**If Ruff C90 complexity fires on `compute_srs`:** Extract the matrix-building logic to a private helper `_build_srs_matrices(games_df, idx, margin_cap) -> tuple[...]:` to reduce per-function complexity.

### Architecture Guardrails (Mandatory)

1. **`from __future__ import annotations` required** — first non-comment line
2. **`mypy --strict` mandatory** — zero errors
3. **Vectorization First** — use `np.add.at()` for scatter, `A @ r` for matrix-vector products; NO Python `for` loops over game rows; NO `iterrows()`
4. **No direct I/O** — `opponent.py` does NOT load CSV files; it receives pre-loaded DataFrames
5. **No imports from `ncaa_eval.ingest`** — pure transform-layer component
6. **No imports from other transform modules** — opponent.py has no dependency on sequential.py, graph.py, normalization.py, or serving.py
7. **`logger = logging.getLogger(__name__)`** at module level (not inside functions or classes)
8. **Empty-input guards** — all methods/functions must return empty DataFrame with correct column names if `games_df.empty`; never raise exceptions on empty input

### What NOT to Do

- **Do not** use `iterrows()` or any Python `for` loop over game rows
- **Do not** implement Elo — Story 4.8
- **Do not** implement LRMC or TrueSkill — Post-MVP Backlog
- **Do not** import scipy — numpy.linalg.solve is sufficient and avoids a new dependency
- **Do not** add scipy to pyproject.toml
- **Do not** filter tournament games inside the solver — that is the caller's responsibility (document it, don't enforce it)
- **Do not** load CSV files directly (data/kaggle/ etc.) — accept pre-loaded DataFrames only
- **Do not** implement Women's data — Men's only for MVP
- **Do not** implement in-season incremental updates — batch only; Story 4.8 handles incremental updates
- **Do not** implement KenPom AdjEM — that is a per-possession normalization of SRS that requires separate offensive/defensive solve; out of scope

### Running Quality Checks

```bash
POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval ruff check .
POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval mypy --strict src/ncaa_eval tests
POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval pytest tests/unit/test_opponent.py -v
```

### Project Structure Notes

**New files:**
- `src/ncaa_eval/transform/opponent.py` — `BatchRatingSolver`, `compute_srs_ratings`, `compute_ridge_ratings`, `compute_colley_ratings`
- `tests/unit/test_opponent.py` — 16 unit tests

**Modified files:**
- `src/ncaa_eval/transform/__init__.py` — add opponent.py exports
- `_bmad-output/implementation-artifacts/sprint-status.yaml` — update status to `review`
- `_bmad-output/implementation-artifacts/4-6-implement-opponent-adjustments.md` — this story file

**No changes to:**
- `pyproject.toml` (no new dependencies — numpy, scikit-learn already present)
- `src/ncaa_eval/transform/graph.py` (stable)
- `src/ncaa_eval/transform/sequential.py` (stable)
- `src/ncaa_eval/transform/normalization.py` (stable)
- `src/ncaa_eval/transform/serving.py` (stable)
- Any existing test files

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 4.6 — Acceptance Criteria]
- [Source: _bmad-output/planning-artifacts/epics.md#Epic 4 — Feature Engineering Suite (FR5: Opponent Adjustments)]
- [Source: specs/research/feature-engineering-techniques.md#Section 2 — Opponent Adjustment Techniques (SRS, Ridge, Colley)]
- [Source: specs/research/feature-engineering-techniques.md#Section 2.1 — SRS: Fixed-point iteration formula, convergence analysis]
- [Source: specs/research/feature-engineering-techniques.md#Section 2.2 — Ridge: scipy.linalg/sklearn.Ridge, lambda selection]
- [Source: specs/research/feature-engineering-techniques.md#Section 2.4 — Colley: Bayesian Cholesky, COL pre-computed alternative]
- [Source: specs/research/feature-engineering-techniques.md#Section 2.5 — Equivalence and Distinctness Summary (Group A vs Group B)]
- [Source: specs/research/feature-engineering-techniques.md#Section 7.1 — Equivalence Groups: Group A (SRS/Ridge) and Group B (Colley)]
- [Source: specs/research/feature-engineering-techniques.md#Section 7.3 — Building Blocks by Story (Story 4.6 scope)]
- [Source: specs/05-architecture-fullstack.md#Section 9 — Unified Project Structure (transform/ module)]
- [Source: specs/05-architecture-fullstack.md#Section 12 — Coding Standards (mypy --strict, vectorization)]
- [Source: _bmad-output/implementation-artifacts/4-5-implement-graph-builders-centrality-features.md#Dev Notes — mypy patterns, no-iterrows, every-public-method-tested mandate, empty-guard pattern]
- [Source: _bmad-output/implementation-artifacts/4-5-implement-graph-builders-centrality-features.md#Dev Agent Record — Debug Log References: type: ignore patterns for numpy/pandas]

## Dev Agent Record

### Agent Model Used

Claude Sonnet 4.6 (create-story workflow, dev-story workflow)

### Debug Log References

**mypy type annotation fixes (numpy 2.x / sklearn 1.8):**
- `numpy` ships py.typed — do NOT use `# type: ignore[import-untyped]` on numpy import
- `pandas` does NOT ship py.typed — MUST use `# type: ignore[import-untyped]` on both src and test imports
- `sklearn.linear_model.Ridge` lacks stubs — MUST use `# type: ignore[import-untyped]` on sklearn import
- `sklearn 1.8` `Ridge.coef_` is always ndarray after fit; no `# type: ignore[union-attr]` needed
- `np.linalg.solve` / `np.linalg.lstsq` return `ndarray[..., floating[Any]]` not `ndarray[..., float64]` — use `# type: ignore[assignment]` on the assignment line
- `_build_srs_matrices()` helper extracted to reduce cyclomatic complexity of `compute_srs`
- `_build_team_index()` helper extracts common team index building logic shared by all three solvers

**ruff import ordering:**
- `from __future__ import annotations` must immediately precede stdlib imports (no blank line)
- Local imports from `ncaa_eval.*` are sorted alphabetically by module name within the transform package

### Completion Notes List

- ✅ **Task 1**: Implemented `BatchRatingSolver` with `compute_srs`, `compute_ridge`, `compute_colley` methods + 3 module-level convenience functions. All vectorized via `np.add.at` scatter and matrix operations — no `iterrows` or Python loops over game rows.
- ✅ **Task 2**: Exported `BatchRatingSolver`, `compute_srs_ratings`, `compute_ridge_ratings`, `compute_colley_ratings` from `transform/__init__.py` and added to `__all__`.
- ✅ **Task 3**: 16 unit tests written and passing in `tests/unit/test_opponent.py`. All 3 solvers covered: SRS (6 tests), Ridge (3 tests), Colley (4 tests), cross-method schema/API (3 tests).
- ✅ **Task 4 — MAS Sanity Check (2023 season)**:
  - 5,602 regular-season games (filtered `is_tournament == False`)
  - SRS computed for 363 teams
  - Spearman rank correlation with pre-tournament MAS ordinals (day_num ≤ 128): **r = 0.9859** (p = 3.37e-282)
  - Threshold ≥ 0.85: **PASS** (significantly exceeds expectation)
  - Top SRS teams (team_ids): 1417 (SRS 20.0, MAS rank 4), 1104 (SRS 19.7, MAS rank 3), 1222 (SRS 19.2, MAS rank 1)
  - Bottom SRS teams align correctly with worst MAS ranks (1254 @ rank 363, 1216 @ rank 362)
- ✅ **Task 5**: All quality gates passed — ruff, mypy --strict, pytest 16/16 + 265/265 regression tests green. Story committed.

### File List

- `src/ncaa_eval/transform/opponent.py` (new)
- `src/ncaa_eval/transform/__init__.py` (modified — added opponent exports)
- `tests/unit/test_opponent.py` (new)
- `_bmad-output/implementation-artifacts/4-6-implement-opponent-adjustments.md` (modified — this story file)
- `_bmad-output/implementation-artifacts/sprint-status.yaml` (modified — status → review)
