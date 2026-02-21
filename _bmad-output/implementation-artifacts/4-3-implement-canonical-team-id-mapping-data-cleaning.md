# Story 4.3: Implement Canonical Team ID Mapping & Data Cleaning

Status: ready-for-dev

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a data scientist,
I want a normalization layer that maps diverse team names to canonical IDs, integrates supplementary lookup tables, and ingests Massey Ordinal rankings,
so that features are computed on consistent, clean data and all pre-computed multi-system ratings are available to the feature pipeline with temporal fidelity.

## Acceptance Criteria

1. **Given** ingested data may contain varying team name formats across sources, **When** the developer runs the normalization pipeline, **Then** all team name variants are mapped to a single canonical TeamID per team using `MTeamSpellings.csv`, and the mapping handles common variations (abbreviations, mascots, "State" vs "St.", etc.).
2. **And** unmapped team names log a warning with suggested matches (if any close spellings exist) rather than raising an exception — returning `None` instead.
3. **And** the cleaning pipeline is idempotent: applying normalization twice produces the same result.
4. **And** `MNCAATourneySeeds.csv` is parsed into structured fields: `seed_num` (integer 1–16), `region` (str W/X/Y/Z), `is_play_in` (bool — True for seeds with 'a'/'b' suffix, e.g., "W11a").
5. **And** `MTeamConferences.csv` provides a `(season, team_id) → conference_abbrev` lookup for every season available.
6. **And** `MMasseyOrdinals.csv` is ingested with all 100+ ranking systems, preserving the `RankingDayNum` temporal field for each record.
7. **And** a **coverage gate** verifies whether SAG (Sagarin) and WLK (Whitlock) are present for all 23 seasons (2003–2025): if either has gaps, the fallback composite is MOR+POM+DOL (confirmed full-coverage, all margin-based); the gate returns a structured result describing which systems are recommended.
8. **And** the following composite building blocks are available (modeler selects at feature-serving time):
   - **Option A:** Simple average of selected systems' ordinal ranks (e.g., `(SAG + POM + MOR + WLK) / 4` if coverage confirmed; fallback `(MOR + POM + DOL) / 3`)
   - **Option B:** Weighted ensemble with caller-supplied system weights (dict `{system_name: weight}`)
   - **Option C:** PCA reduction of all available systems to N principal components capturing ≥90% variance
   - **Option D:** Pre-tournament snapshot — use only ordinals from the last available `RankingDayNum ≤ 128` per system per season
9. **And** ordinal feature normalization options are provided per-system-per-season: rank delta between two teams (primary matchup feature), percentile (bounded [0,1] = rank/n_teams), z-score.
10. **And** the pre-computed `COL` (Colley) and `MAS` (Massey) systems in `MMasseyOrdinals.csv` are accessible via the same API, documented as alternatives to reimplementing those solvers in Story 4.6.
11. **And** the normalization and ingestion module is covered by unit tests with known name-variant fixtures and known ordinal coverage assertions.

## Tasks / Subtasks

- [ ] Task 1: Design and implement `src/ncaa_eval/transform/normalization.py` with all public classes (AC: 1–10)
  - [ ] 1.1: Define `TourneySeed` frozen dataclass with fields: `season: int`, `team_id: int`, `seed_str: str`, `region: str`, `seed_num: int`, `is_play_in: bool`
  - [ ] 1.2: Implement `parse_seed(season: int, team_id: int, seed_str: str) -> TourneySeed` module-level function — parse region/seed_num/is_play_in from the raw seed string (e.g., "W01" → region="W", seed_num=1, is_play_in=False; "X11a" → is_play_in=True)
  - [ ] 1.3: Define `CoverageGateResult` frozen dataclass: `primary_systems: list[str]`, `fallback_used: bool`, `fallback_reason: str`, `recommended_systems: list[str]`
  - [ ] 1.4: Implement `TeamNameNormalizer` class — wraps the spelling → TeamID lookup with case-insensitive matching and warning on misses
  - [ ] 1.5: Implement `TourneySeedTable` class — wraps `(season, team_id)` → `TourneySeed` lookup
  - [ ] 1.6: Implement `ConferenceLookup` class — wraps `(season, team_id)` → `conf_abbrev` lookup
  - [ ] 1.7: Implement `MasseyOrdinalsStore` class — DataFrame-backed store with temporal filtering, coverage gate, and composite computation methods

- [ ] Task 2: Implement `TeamNameNormalizer` class (AC: 1–3)
  - [ ] 2.1: `__init__(self, spellings: dict[str, int]) -> None` — accepts pre-lowercased spelling → TeamID dict
  - [ ] 2.2: `classmethod from_csv(cls, path: Path) -> TeamNameNormalizer` — reads `MTeamSpellings.csv`, lowercases `TeamNameSpelling`, builds dict
  - [ ] 2.3: `normalize(self, name: str) -> int | None` — looks up `name.lower()` in dict; on miss, logs WARNING with any close matches found via simple prefix search; returns None on miss
  - [ ] 2.4: Idempotency: pure lookup — calling `normalize(name)` twice returns the same value
  - [ ] 2.5: No iterrows — use `dict(zip(df["TeamNameSpelling"].str.lower(), df["TeamID"].astype(int)))` for vectorized construction

- [ ] Task 3: Implement `TourneySeedTable` class (AC: 4)
  - [ ] 3.1: `__init__(self, seeds: dict[tuple[int, int], TourneySeed]) -> None` — key is `(season, team_id)`
  - [ ] 3.2: `classmethod from_csv(cls, path: Path) -> TourneySeedTable` — reads `MNCAATourneySeeds.csv`, applies `parse_seed()` to each row using vectorized pandas operations (no iterrows)
  - [ ] 3.3: `get(self, season: int, team_id: int) -> TourneySeed | None` — returns structured seed or None
  - [ ] 3.4: `all_seeds(self, season: int | None = None) -> list[TourneySeed]` — returns all seeds, optionally filtered to season

- [ ] Task 4: Implement `ConferenceLookup` class (AC: 5)
  - [ ] 4.1: `__init__(self, lookup: dict[tuple[int, int], str]) -> None` — key is `(season, team_id)`, value is `conf_abbrev`
  - [ ] 4.2: `classmethod from_csv(cls, path: Path) -> ConferenceLookup` — reads `MTeamConferences.csv` using vectorized dict construction
  - [ ] 4.3: `get(self, season: int, team_id: int) -> str | None` — returns conf abbreviation or None

- [ ] Task 5: Implement `MasseyOrdinalsStore` class (AC: 6–10)
  - [ ] 5.1: `__init__(self, df: pd.DataFrame) -> None` — stores raw DataFrame with columns `[Season, RankingDayNum, SystemName, TeamID, OrdinalRank]`
  - [ ] 5.2: `classmethod from_csv(cls, path: Path) -> MasseyOrdinalsStore` — reads `MMasseyOrdinals.csv` with proper dtypes
  - [ ] 5.3: `run_coverage_gate(self) -> CoverageGateResult` — checks SAG and WLK for all seasons 2003–2025; returns structured result with `recommended_systems` list and `fallback_used` flag
  - [ ] 5.4: `get_snapshot(self, season: int, day_num: int, systems: list[str] | None = None) -> pd.DataFrame` — returns ordinals for all teams where `RankingDayNum ≤ day_num`; pivots to wide format `{TeamID, system_1, system_2, ...}`; uses last available `RankingDayNum` per (season, system, team) before the cutoff
  - [ ] 5.5: `composite_simple_average(self, season: int, day_num: int, systems: list[str]) -> pd.Series` — Option A: mean of ordinal ranks across specified systems per team; index is TeamID
  - [ ] 5.6: `composite_weighted(self, season: int, day_num: int, weights: dict[str, float]) -> pd.Series` — Option B: weighted average using caller-supplied weights; normalizes weights to sum to 1
  - [ ] 5.7: `composite_pca(self, season: int, day_num: int, n_components: int | None = None, min_variance: float = 0.90) -> pd.DataFrame` — Option C: PCA of all available systems; `n_components=None` auto-selects to capture `min_variance` of total variance; returns `DataFrame` with columns `PC1, PC2, ...` indexed by TeamID
  - [ ] 5.8: `pre_tournament_snapshot(self, season: int, systems: list[str] | None = None) -> pd.DataFrame` — Option D: uses only ordinals from last `RankingDayNum ≤ 128` per system per season; returns same format as `get_snapshot`
  - [ ] 5.9: `normalize_rank_delta(self, snapshot: pd.DataFrame, team_a: int, team_b: int, system: str) -> float` — rank delta for matchup: `snapshot.loc[team_a, system] - snapshot.loc[team_b, system]`; positive means team_a ranked worse (higher rank number = worse)
  - [ ] 5.10: `normalize_percentile(self, season: int, day_num: int, system: str) -> pd.Series` — per-season percentile: `OrdinalRank / n_teams`; index is TeamID; uses `get_snapshot` internally
  - [ ] 5.11: `normalize_zscore(self, season: int, day_num: int, system: str) -> pd.Series` — z-score per season: `(rank - mean_rank) / std_rank`; index is TeamID

- [ ] Task 6: Export public API from `src/ncaa_eval/transform/__init__.py` (AC: 1–10)
  - [ ] 6.1: Import and re-export `TeamNameNormalizer`, `TourneySeed`, `TourneySeedTable`, `ConferenceLookup`, `MasseyOrdinalsStore`, `CoverageGateResult`, `parse_seed` from `transform/__init__.py`
  - [ ] 6.2: Keep `_MASSEY_GATE_SYSTEMS`, `_FALLBACK_SYSTEMS`, `_MASSEY_FIRST_SEASON`, `_MASSEY_LAST_SEASON` private constants in `normalization.py`

- [ ] Task 7: Write unit tests in `tests/unit/test_normalization.py` (AC: 11)
  - [ ] 7.1: Test `parse_seed`: "W01" → region="W", seed_num=1, is_play_in=False
  - [ ] 7.2: Test `parse_seed`: "X16a" → region="X", seed_num=16, is_play_in=True
  - [ ] 7.3: Test `parse_seed`: "Y11b" → is_play_in=True, seed_num=11
  - [ ] 7.4: Test `TourneySeedTable.from_csv`: loads fixture CSV, returns correct `TourneySeed` for known (season, team_id)
  - [ ] 7.5: Test `TourneySeedTable.get`: returns None for unknown (season, team_id)
  - [ ] 7.6: Test `TeamNameNormalizer.normalize`: exact match (case-insensitive) returns correct TeamID
  - [ ] 7.7: Test `TeamNameNormalizer.normalize`: known name variant (e.g., "UNLV" → TeamID for "Nevada Las Vegas") resolves correctly
  - [ ] 7.8: Test `TeamNameNormalizer.normalize`: unknown name returns None (verifies no exception raised)
  - [ ] 7.9: Test `TeamNameNormalizer` idempotency: calling `normalize(name)` twice returns same value
  - [ ] 7.10: Test `ConferenceLookup.get`: returns correct conference for known (season, team_id)
  - [ ] 7.11: Test `ConferenceLookup.get`: returns None for missing (season, team_id)
  - [ ] 7.12: Test `MasseyOrdinalsStore.run_coverage_gate` with fixture data containing SAG+WLK for all seasons → `fallback_used=False`, `recommended_systems` includes SAG/WLK
  - [ ] 7.13: Test `MasseyOrdinalsStore.run_coverage_gate` with fixture data missing WLK for some seasons → `fallback_used=True`, `recommended_systems` = MOR+POM+DOL
  - [ ] 7.14: Test `MasseyOrdinalsStore.get_snapshot` temporal filtering: only returns ordinals with `RankingDayNum ≤ day_num`
  - [ ] 7.15: Test `MasseyOrdinalsStore.get_snapshot` uses last-available RankingDayNum per (system, team) before cutoff
  - [ ] 7.16: Test `MasseyOrdinalsStore.composite_simple_average`: fixture with 2 systems, 3 teams → correct mean per team
  - [ ] 7.17: Test `MasseyOrdinalsStore.composite_weighted`: correct weighted average with known weights
  - [ ] 7.18: Test `MasseyOrdinalsStore.pre_tournament_snapshot`: uses only `RankingDayNum ≤ 128`; does NOT use day 135 ordinals
  - [ ] 7.19: Test `MasseyOrdinalsStore.normalize_percentile`: returns values in [0, 1] range
  - [ ] 7.20: Test `MasseyOrdinalsStore.normalize_zscore`: mean ≈ 0, std ≈ 1 over fixture data

- [ ] Task 8: Commit (AC: all)
  - [ ] 8.1: `git add src/ncaa_eval/transform/normalization.py src/ncaa_eval/transform/__init__.py tests/unit/test_normalization.py`
  - [ ] 8.2: Commit: `feat(transform): implement canonical team ID mapping and data cleaning (Story 4.3)`
  - [ ] 8.3: Update `_bmad-output/implementation-artifacts/sprint-status.yaml`: `4-3-implement-canonical-team-id-mapping-data-cleaning` → `review`

## Dev Notes

### Story Nature: Second Code Story in Epic 4 — normalization.py in transform/

This story extends the `src/ncaa_eval/transform/` module introduced in Story 4.2. It is a **code story** — `mypy --strict`, Ruff, `from __future__ import annotations`, and the no-iterrows mandate all apply. No notebook deliverables.

This story delivers **normalization and lookup infrastructure** consumed by:
- Story 4.4 (sequential transformations) — needs `ConferenceLookup`, `TourneySeedTable`
- Story 4.6 (batch ratings) — can use pre-computed `COL`/`MAS` from `MasseyOrdinalsStore` instead of reimplementing Colley/Massey solvers
- Story 4.7 (feature serving) — needs Massey ordinals temporal slicing via `get_snapshot` and `pre_tournament_snapshot`
- Story 4.8 (Elo features) — may use `ConferenceLookup` for conference mean-reversion in season transitions

### Module Placement

**New file:** `src/ncaa_eval/transform/normalization.py`

Per Architecture Section 9, all feature engineering and data transformation belongs in `src/ncaa_eval/transform/`. This module sits alongside `serving.py` in the transform layer. It does NOT modify any `ingest/` files.

**Modified file:** `src/ncaa_eval/transform/__init__.py` — add exports for new public API

### CSV Column Schemas (Confirmed from Real Data)

All four Kaggle CSV files have been verified with real data:

| File | Columns |
|:---|:---|
| `MTeamSpellings.csv` | `TeamNameSpelling, TeamID` |
| `MNCAATourneySeeds.csv` | `Season, Seed, TeamID` |
| `MTeamConferences.csv` | `Season, TeamID, ConfAbbrev` |
| `MMasseyOrdinals.csv` | `Season, RankingDayNum, SystemName, TeamID, OrdinalRank` |

### Seed String Parsing (`parse_seed`)

Seed strings from `MNCAATourneySeeds.csv` follow the pattern `[WXYZ][0-9]{2}[ab]?`:

```python
def parse_seed(season: int, team_id: int, seed_str: str) -> TourneySeed:
    """Parse a raw tournament seed string into a structured TourneySeed.

    Format: [WXYZ][0-9]{2}[ab]?
    Examples:
        "W01"  → region="W", seed_num=1,  is_play_in=False
        "X16a" → region="X", seed_num=16, is_play_in=True
        "Y11b" → region="Y", seed_num=11, is_play_in=True
    """
    if len(seed_str) < 3:
        msg = f"Invalid seed string: {seed_str!r}"
        raise ValueError(msg)
    region = seed_str[0]
    seed_num = int(seed_str[1:3])
    is_play_in = len(seed_str) > 3 and seed_str[3] in ("a", "b")
    return TourneySeed(
        season=season,
        team_id=team_id,
        seed_str=seed_str,
        region=region,
        seed_num=seed_num,
        is_play_in=is_play_in,
    )
```

### `TeamNameNormalizer` — Vectorized Construction

Use vectorized dict construction (no iterrows):

```python
@classmethod
def from_csv(cls, path: Path) -> TeamNameNormalizer:
    df = pd.read_csv(path)
    spellings = dict(zip(df["TeamNameSpelling"].str.lower(), df["TeamID"].astype(int)))
    return cls(spellings)
```

`normalize()` should:
1. Call `name.lower()` on input
2. Look up in dict — return TeamID on hit
3. On miss: find 3–5 prefix matches (e.g., `{k: v for k, v in self._spellings.items() if k.startswith(name.lower()[:4])}`) for the warning message
4. Log `logger.warning("TeamNameNormalizer: no match for %r; closest spellings: %s", name, closest_matches)`
5. Return `None`

Idempotency: pure dict lookup — calling twice with the same input returns the same result by definition.

### `TourneySeedTable` — Vectorized Loading

Build the `(season, team_id)` lookup without iterrows:

```python
@classmethod
def from_csv(cls, path: Path) -> TourneySeedTable:
    df = pd.read_csv(path)
    seeds: dict[tuple[int, int], TourneySeed] = {}
    # Vectorized: iterate over tuples (much faster than iterrows for dict building)
    for season, seed_str, team_id in df[["Season", "Seed", "TeamID"]].itertuples(index=False):
        ts = parse_seed(int(season), int(team_id), str(seed_str))
        seeds[(ts.season, ts.team_id)] = ts
    return cls(seeds)
```

Note: `itertuples` is acceptable for dict construction where the per-row operation is non-vectorizable (parsing a string with branching logic). This is NOT equivalent to `iterrows` — `itertuples` is 5-10× faster and does not create per-row Series objects.

### `ConferenceLookup` — Vectorized Loading

```python
@classmethod
def from_csv(cls, path: Path) -> ConferenceLookup:
    df = pd.read_csv(path)
    # Vectorized dict construction
    lookup = {
        (int(s), int(t)): str(c)
        for s, t, c in zip(df["Season"], df["TeamID"], df["ConfAbbrev"])
    }
    return cls(lookup)
```

### `MasseyOrdinalsStore` — Key Design Decisions

**Coverage gate constants:**
```python
_MASSEY_FIRST_SEASON: int = 2003    # First season in MMasseyOrdinals.csv
_MASSEY_LAST_SEASON: int = 2025     # Latest known season
_MASSEY_ALL_SEASONS: frozenset[int] = frozenset(range(_MASSEY_FIRST_SEASON, _MASSEY_LAST_SEASON + 1))
_GATE_SYSTEMS: tuple[str, ...] = ("SAG", "WLK")
_FALLBACK_SYSTEMS: tuple[str, ...] = ("MOR", "POM", "DOL")
_PRIMARY_COMPOSITE: tuple[str, ...] = ("SAG", "POM", "MOR", "WLK")
```

**`run_coverage_gate` logic:**
```python
def run_coverage_gate(self) -> CoverageGateResult:
    # For each gate system, find which seasons it covers
    covered = (
        self._df[self._df["SystemName"].isin(list(_GATE_SYSTEMS))]
        .groupby("SystemName")["Season"]
        .apply(set)
        .to_dict()
    )
    missing: list[str] = []
    for system in _GATE_SYSTEMS:
        system_seasons = covered.get(system, set())
        if not _MASSEY_ALL_SEASONS.issubset(system_seasons):
            missing.append(system)
    if missing:
        return CoverageGateResult(
            primary_systems=list(_PRIMARY_COMPOSITE),
            fallback_used=True,
            fallback_reason=f"{missing} missing for some seasons 2003–2025",
            recommended_systems=list(_FALLBACK_SYSTEMS),
        )
    return CoverageGateResult(
        primary_systems=list(_PRIMARY_COMPOSITE),
        fallback_used=False,
        fallback_reason="",
        recommended_systems=list(_PRIMARY_COMPOSITE),
    )
```

**`get_snapshot` implementation — critical temporal logic:**

For each (season, system, team), we want the ordinal from the **last `RankingDayNum ≤ day_num`** — not just any record before the cutoff. Use groupby + idxmax pattern:

```python
def get_snapshot(
    self,
    season: int,
    day_num: int,
    systems: list[str] | None = None,
) -> pd.DataFrame:
    """Return a wide-format snapshot of ordinal ranks as of day_num.

    For each (SystemName, TeamID), uses the latest RankingDayNum that is ≤ day_num.
    Returns a DataFrame with TeamID as index and one column per system.
    """
    mask = (self._df["Season"] == season) & (self._df["RankingDayNum"] <= day_num)
    filtered = self._df[mask]
    if systems is not None:
        filtered = filtered[filtered["SystemName"].isin(systems)]
    if filtered.empty:
        return pd.DataFrame()
    # Keep last available RankingDayNum per (SystemName, TeamID)
    latest = filtered.loc[
        filtered.groupby(["SystemName", "TeamID"])["RankingDayNum"].idxmax()
    ]
    # Pivot to wide format: rows=TeamID, columns=SystemName, values=OrdinalRank
    pivot = latest.pivot(index="TeamID", columns="SystemName", values="OrdinalRank")
    pivot.columns.name = None  # Clean up column index name
    return pivot
```

**`composite_simple_average` — Option A:**
```python
def composite_simple_average(
    self, season: int, day_num: int, systems: list[str]
) -> pd.Series:
    snapshot = self.get_snapshot(season, day_num, systems=systems)
    return snapshot[systems].mean(axis=1)  # index: TeamID, values: average rank
```

**`composite_weighted` — Option B:**
```python
def composite_weighted(
    self, season: int, day_num: int, weights: dict[str, float]
) -> pd.Series:
    systems = list(weights.keys())
    snapshot = self.get_snapshot(season, day_num, systems=systems)
    w = pd.Series(weights)
    w = w / w.sum()  # Normalize weights to sum=1
    return snapshot[systems].mul(w.values).sum(axis=1)  # index: TeamID
```

**`composite_pca` — Option C:**
```python
def composite_pca(
    self,
    season: int,
    day_num: int,
    n_components: int | None = None,
    min_variance: float = 0.90,
) -> pd.DataFrame:
    from sklearn.decomposition import PCA  # type: ignore[import-untyped]
    snapshot = self.get_snapshot(season, day_num)
    snapshot = snapshot.dropna()
    if n_components is None:
        # Auto-select: find minimum components for min_variance
        pca_full = PCA()
        pca_full.fit(snapshot.values)
        cumvar = pca_full.explained_variance_ratio_.cumsum()
        n_components = int((cumvar >= min_variance).argmax()) + 1
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(snapshot.values)
    cols = [f"PC{i+1}" for i in range(n_components)]
    return pd.DataFrame(components, index=snapshot.index, columns=cols)
```

**`pre_tournament_snapshot` — Option D:**
```python
def pre_tournament_snapshot(
    self, season: int, systems: list[str] | None = None
) -> pd.DataFrame:
    """Pre-tournament snapshot: last RankingDayNum ≤ 128 per system per season.

    DayNum 128 ≈ Selection Sunday in the Kaggle calendar.
    """
    return self.get_snapshot(season, day_num=128, systems=systems)
```

**Normalization methods:**
```python
def normalize_percentile(self, season: int, day_num: int, system: str) -> pd.Series:
    snap = self.get_snapshot(season, day_num, systems=[system])
    col = snap[system]
    n_teams = col.count()
    return col / n_teams  # [0, 1] bounded percentile

def normalize_zscore(self, season: int, day_num: int, system: str) -> pd.Series:
    snap = self.get_snapshot(season, day_num, systems=[system])
    col = snap[system]
    return (col - col.mean()) / col.std()

def normalize_rank_delta(
    self, snapshot: pd.DataFrame, team_a: int, team_b: int, system: str
) -> float:
    return float(snapshot.loc[team_a, system] - snapshot.loc[team_b, system])
```

### Massey Ordinals: COL and MAS as Story 4.6 Alternatives

The `MMasseyOrdinals.csv` includes pre-computed `COL` (Colley Matrix) and `MAS` (Massey's Method) systems. These are accessible via `get_snapshot(season, day_num, systems=["COL", "MAS"])`. Story 4.6 notes this as an alternative to implementing the Colley Matrix solver from scratch. The developer in Story 4.6 should:
1. First check whether the pre-computed `COL`/`MAS` from `MasseyOrdinalsStore` meets accuracy requirements
2. Only implement the full Cholesky solver if the pre-computed version is insufficient

### mypy Strict Compliance Notes

**sklearn import:** PCA is imported inside the method body to avoid top-level dependency on sklearn in mypy scanning. Add `# type: ignore[import-untyped]` on the import line:
```python
from sklearn.decomposition import PCA  # type: ignore[import-untyped]
```

**`pd.DataFrame` type annotation:** All DataFrame parameters should be annotated as `pd.DataFrame`. Avoid `Any` in return types.

**`dict[tuple[int, int], ...]` keys:** Fully typed — mypy should be satisfied with `tuple[int, int]` as the dict key type.

**`pd.Series` return type:** `pd.Series` (without subscript — pandas is not typed in mypy --strict). Use `pd.Series` (without `[int]` or `[float]`).

**`pivot` return:** `pd.DataFrame.pivot()` returns `pd.DataFrame` — annotate the return as `pd.DataFrame`.

### Architecture Guardrails (Mandatory)

From `specs/05-architecture-fullstack.md` Section 12 (Coding Standards):

1. **`from __future__ import annotations` required** — first non-comment line in all files
2. **`mypy --strict` mandatory** — all type annotations complete; no `Any` unless justified
3. **Vectorization First** — no `for` loops over pandas DataFrames for data processing; use vectorized operations. `itertuples` is acceptable for string-parsing dict construction (not equivalent to `iterrows`)
4. **No direct IO in the transform module** — the normalization classes load data from CSV paths; they do NOT read Parquet game data directly
5. **Pydantic models or TypedDicts** for data passed between layers — `TourneySeed` and `CoverageGateResult` as frozen dataclasses are appropriate for internal transform-layer types

### No New Dependencies Required

All needed libraries are already in `pyproject.toml`:
- `pandas` — CSV loading and DataFrame operations
- `numpy` — via pandas; no direct import needed
- `scikit-learn` — for PCA (Option C only); import locally inside `composite_pca` to avoid top-level `# type: ignore` propagation

### Import Pattern for Normalization Module

```python
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
```

No imports from `ncaa_eval.ingest` are needed in `normalization.py` — the normalization module is self-contained (loads CSVs from paths, not from the Repository).

### Test File Structure

**File:** `tests/unit/test_normalization.py`

Pattern from `tests/unit/test_chronological_serving.py`:

```python
from __future__ import annotations

import io
import pandas as pd
import pytest
from pathlib import Path

from ncaa_eval.transform.normalization import (
    CoverageGateResult,
    ConferenceLookup,
    MasseyOrdinalsStore,
    TeamNameNormalizer,
    TourneySeed,
    TourneySeedTable,
    parse_seed,
)
```

**Fixture approach:** Use `tmp_path` + `io.StringIO` or `tmp_path / "filename.csv"` for fixture CSVs. Do NOT reference real Kaggle data files in tests.

**Example fixture CSV for MasseyOrdinals:**
```python
@pytest.fixture
def massey_csv(tmp_path: Path) -> Path:
    content = "Season,RankingDayNum,SystemName,TeamID,OrdinalRank\n"
    # Add rows for systems SAG, POM, MOR, WLK across seasons 2003-2025...
    csv_path = tmp_path / "MMasseyOrdinals.csv"
    csv_path.write_text(content)
    return csv_path
```

**Markers to apply:**
- `@pytest.mark.smoke` on fast parsing/lookup tests (< 1s each)
- `@pytest.mark.unit` on all tests in this file

### Previous Story Intelligence (Story 4.2 Learnings)

From the 4.2 dev agent completion notes:
- `Iterator` should be imported from `collections.abc` (UP035 compliance), not `typing`
- `pd.DataFrame([...])` + `.to_dict(orient="records")` pattern works for Game reconstruction — same pattern available for any Pydantic model
- `frozen=True` dataclass only prevents attribute rebinding, not mutation of mutable fields — use `tuple[str, ...]` instead of `list[str]` in `CoverageGateResult.recommended_systems` if immutability matters (Story 4.2 LOW review item)
- Use `logger = logging.getLogger(__name__)` at module level

### 5 LOW-Severity Items from Story 4.2 (Not in Scope Here)

These carry forward from 4.2 code review as deferred improvements:
1. `_effective_date` `date=None` fallback path untested
2. `iter_games_by_date` re-sorts already-sorted output (efficiency issue)
3. `SeasonGames(frozen=True)` mutable list false-immutability
4. `_GAME_DEFAULTS` fixture date/season trap
5. `_deduplicate_2025` pandas round-trip coercion assumption

These remain in 4.2's review follow-ups and are NOT in scope for 4.3.

### What NOT to Do

- **Do not** implement any rolling averages, EWMA, or momentum features — that belongs in Story 4.4
- **Do not** implement the full SRS, Ridge, or Colley solvers — that belongs in Story 4.6 (just expose the pre-computed COL/MAS from MasseyOrdinalsStore as alternatives)
- **Do not** import from `ncaa_eval.ingest.repository` — normalization is a pure CSV-loading layer
- **Do not** modify `src/ncaa_eval/ingest/` files — they are stable
- **Do not** use `df.iterrows()` for DataFrame processing — use vectorized pandas operations or `itertuples` for dict construction
- **Do not** add Women's data files (WTeamSpellings, WNCAATourneySeeds, etc.) — Men's only for MVP; Women's support is a future enhancement
- **Do not** apply mypy exclusions to this module — it must be fully type-checked under `--strict`

### Running Quality Checks

```bash
POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval ruff check .
POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval mypy --strict src/ncaa_eval tests
POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval pytest tests/unit/test_normalization.py -v
```

### Project Structure Notes

**New files:**
- `src/ncaa_eval/transform/normalization.py` — all four lookup classes + dataclasses + `parse_seed`

**Modified files:**
- `src/ncaa_eval/transform/__init__.py` — add exports for new public API
- `tests/unit/test_normalization.py` — new test file (create in `tests/unit/`)
- `_bmad-output/implementation-artifacts/sprint-status.yaml` — update status to `review`
- `_bmad-output/implementation-artifacts/4-3-implement-canonical-team-id-mapping-data-cleaning.md` — this story file (Dev Agent Record section)

**No changes to:**
- `src/ncaa_eval/ingest/` (stable)
- `src/ncaa_eval/transform/serving.py` (stable)
- `pyproject.toml` (no new dependencies)
- Any existing test files

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 4.3 — Acceptance Criteria]
- [Source: _bmad-output/planning-artifacts/epics.md#Epic 4 — Feature overview (FR5: Advanced Transformations including normalization)]
- [Source: specs/research/feature-engineering-techniques.md#Section 5 — Massey Ordinal Systems]
- [Source: specs/research/feature-engineering-techniques.md#Section 7.3 — Building Blocks by Story (Story 4.3 scope)]
- [Source: specs/05-architecture-fullstack.md#Section 9 — Unified Project Structure (`transform/` module)]
- [Source: specs/05-architecture-fullstack.md#Section 12 — Coding Standards (mypy --strict, vectorization)]
- [Source: data/kaggle/MMasseyOrdinals.csv — columns: Season, RankingDayNum, SystemName, TeamID, OrdinalRank]
- [Source: data/kaggle/MNCAATourneySeeds.csv — columns: Season, Seed, TeamID]
- [Source: data/kaggle/MTeamConferences.csv — columns: Season, TeamID, ConfAbbrev]
- [Source: data/kaggle/MTeamSpellings.csv — columns: TeamNameSpelling, TeamID]
- [Source: src/ncaa_eval/ingest/connectors/kaggle.py — vectorized dict construction pattern: `dict(zip(df["TeamNameSpelling"].str.lower(), df["TeamID"].astype(int)))`]
- [Source: src/ncaa_eval/transform/serving.py — module structure, logging pattern, frozen dataclass usage]
- [Source: _bmad-output/implementation-artifacts/4-2-implement-chronological-data-serving-api.md#Dev Agent Record — completion notes, LOW review items]
- [Source: _bmad-output/planning-artifacts/template-requirements.md — pyarrow type:ignore pattern, no-iterrows mandate, itertuples acceptable for dict construction]

## Dev Agent Record

### Agent Model Used

Claude Sonnet 4.6 (create-story workflow)

### Debug Log References

None.

### Completion Notes List

### File List

## Change Log

| Date | Change | Author |
|:---|:---|:---|
| 2026-02-21 | Created story 4.3 — Implement Canonical Team ID Mapping & Data Cleaning | Claude Sonnet 4.6 (create-story) |
