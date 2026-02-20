# Story 3.1: Data Quality Audit

Status: ready-for-dev

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a data scientist,
I want to explore the ingested NCAA data for completeness, consistency, and anomalies,
So that I understand data quality issues before building features or models.

## Acceptance Criteria

1. **Given** the local data store is populated via the Sync CLI (Epic 2), **When** the data scientist runs the data quality audit notebook, **Then** the notebook documents the schema and structure of all ingested tables (row counts, column types, date ranges).
2. **And** missing values are quantified per column and per season for every ingested entity (Teams, Seasons, Games).
3. **And** duplicate records are identified and documented (duplicate game_ids, duplicate team names, etc.).
4. **And** anomalies and edge cases are flagged — including the 2020 COVID year with no tournament, neutral-site game distribution, OT frequency, score distribution outliers, and ESPN vs. Kaggle game ID overlap for 2025.
5. **And** raw Kaggle CSV files not yet ingested into the repository are documented (MRegularSeasonDetailedResults.csv, MNCAATourneyDetailedResults.csv, MMasseyOrdinals.csv, MNCAATourneySeeds.csv) — schema, row counts, coverage years, and relevance to future epics.
6. **And** data quality issues are summarized with specific recommended cleaning actions for Epic 4 (Feature Engineering).
7. **And** the notebook is committed to the repository with reproducible outputs (all cells executed top-to-bottom with no errors).

## Tasks / Subtasks

- [ ] Task 1: Add Jupyter to dev dependencies (AC: 7)
  - [ ] 1.1: Run `POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval poetry add --group dev "jupyterlab>=4.0,<5"` to add JupyterLab as a dev dependency
  - [ ] 1.2: Follow with `conda run -n ncaa_eval pip install "jupyterlab>=4.0,<5"` to ensure it lands in the conda env (see `poetry add` conda gap in Dev Notes)
  - [ ] 1.3: Verify `conda run -n ncaa_eval jupyter lab --version` succeeds

- [ ] Task 2: Create notebook directory structure (AC: 7)
  - [ ] 2.1: Create `notebooks/eda/` directory (use `mkdir -p notebooks/eda`)
  - [ ] 2.2: Add `notebooks/eda/` path reference to `.gitignore` exclusion check — notebooks directory should be tracked (NOT gitignored). Only `data/` is gitignored.
  - [ ] 2.3: Create `notebooks/.gitkeep` or a `notebooks/README.md` placeholder if notebook is not yet started

- [ ] Task 3: Implement Section 1 — Setup & Data Loading (AC: 1)
  - [ ] 3.1: Create `notebooks/eda/01_data_quality_audit.ipynb`
  - [ ] 3.2: Section 1 imports: `from pathlib import Path`, `import pandas as pd`, `import numpy as np`, `import plotly.express as px`, `import plotly.graph_objects as go`, `from ncaa_eval.ingest import ParquetRepository`
  - [ ] 3.3: Instantiate `repo = ParquetRepository(base_path=Path("../../data/"))` (relative to notebook location)
  - [ ] 3.4: Load all entities: `teams = repo.get_teams()`, `seasons = repo.get_seasons()`, then build games by iterating all seasons
  - [ ] 3.5: Print summary: team count, season count (min/max year), total game count

- [ ] Task 4: Implement Section 2 — Schema Audit (AC: 1)
  - [ ] 4.1: **Teams table:** Convert `teams` list to DataFrame, display `.dtypes`, `.shape`, `.describe()`, and first 5 rows. Show unique `canonical_name` count vs `team_name` count (are there teams with empty canonical_name?).
  - [ ] 4.2: **Seasons table:** Convert `seasons` to DataFrame, show year range (1985–2025 = 41 seasons), verify no gaps in year sequence (`set(range(1985, 2026)) - set(s.year for s in seasons)`).
  - [ ] 4.3: **Games table:** Build a consolidated DataFrame of ALL games by calling `repo.get_games(s.year)` for each season and concatenating. Show total row count, column types, date range of `date` field (nullable), `day_num` range, `season` min/max. This is the primary data structure for EDA.
  - [ ] 4.4: **Per-season game counts:** Group by `season` and count rows. Plot as Plotly bar chart (x=season, y=game_count). Flag any seasons with unusually low counts (COVID 2020 context).
  - [ ] 4.5: **Tournament flag audit:** Group by `season` and count `is_tournament == True` games. The 2020 season should have 0 tournament games. Show as a separate bar chart; label the 2020 bar in red.

- [ ] Task 5: Implement Section 3 — Missing Value Analysis (AC: 2)
  - [ ] 5.1: **Teams:** Show null counts per column (`team_name`, `canonical_name`). Flag any teams where `canonical_name == ""` (empty string default, not null).
  - [ ] 5.2: **Games — `date` column:** `date` is `datetime.date | None` (Kaggle compact results may have `day_num` only, ESPN has actual dates). Count null `date` values per season and show as table. Kaggle games 1985–2024 should have `date=None`; ESPN 2025 games should have actual dates.
  - [ ] 5.3: **Games — `num_ot`:** Show distribution of `num_ot` values (0, 1, 2, 3+ OT games). Count by season. Flag any games with `num_ot >= 4` as extreme outliers.
  - [ ] 5.4: **Games — `loc`:** Show distribution of `loc` values (H/A/N) per season and overall. Neutral-site games (N) should be predominantly tournament and conference tournament games.
  - [ ] 5.5: All missing-value analysis must use vectorized pandas (`.isnull().sum()`, `.value_counts()`, `.groupby().agg()` — no `iterrows()`).

- [ ] Task 6: Implement Section 4 — Duplicate Detection (AC: 3)
  - [ ] 6.1: **Duplicate game_ids:** Check `df["game_id"].duplicated().sum()` — should be 0. If any found, display duplicates with full row context.
  - [ ] 6.2: **Cross-source duplicates:** For season 2025 (both Kaggle and ESPN), check whether the same real-world game appears with different IDs. ESPN IDs start with `"espn_"`, Kaggle IDs follow `"{season}_{day_num}_{w_team_id}_{l_team_id}"`. Show 2025 game count breakdown by source prefix.
  - [ ] 6.3: **Duplicate team names:** Check `teams_df["team_name"].duplicated().sum()` and `teams_df["canonical_name"].duplicated().sum()` (excluding empty string). Report any duplicate team names.
  - [ ] 6.4: **Duplicate games by matchup:** For each season, flag any cases where the same `(w_team_id, l_team_id, day_num)` tuple appears more than once — likely data entry errors.

- [ ] Task 7: Implement Section 5 — Anomaly & Edge Case Detection (AC: 4)
  - [ ] 7.1: **Score distribution:** Plot distribution of `w_score`, `l_score`, and margin (`w_score - l_score`) using Plotly histograms. Flag games with margin > 60 or `w_score > 130` as outliers. List top-10 highest/lowest scoring games.
  - [ ] 7.2: **2020 COVID year deep-dive:** Confirm `is_tournament == False` for ALL 2020 games. Show 2020 regular season game count vs. neighboring years to confirm season was played (COVID stopped tournament, not regular season). Note: 2020 models should train but NOT be evaluated.
  - [ ] 7.3: **OT games:** Show OT game frequency by season. Plot `num_ot > 0` count per year. List all `num_ot >= 3` games (extremely rare — typically data quality flag).
  - [ ] 7.4: **Neutral-site games:** Visualize neutral-site game counts by season. These should spike during tournament months (March/April). Check if non-tournament neutral-site games (conference tournaments) are correctly flagged as `is_tournament=False`.
  - [ ] 7.5: **Team ID coverage:** Verify every `w_team_id` and `l_team_id` in games has a corresponding entry in the teams table. Missing IDs = potential data integrity issue.

- [ ] Task 8: Implement Section 6 — Raw Kaggle CSV Inventory (AC: 5)
  - [ ] 8.1: Load and document **MRegularSeasonDetailedResults.csv** (10 columns of per-game box score stats: FGM, FGA, FG3M, FG3A, FTM, FTA, OR, DR, Ast, TO, Stl, Blk, PF — both winning and losing team). Show: season range, row count, column list, null value summary. Note: NOT ingested into repository; relevant to Epic 4 (Feature Engineering).
  - [ ] 8.2: Load and document **MNCAATourneyDetailedResults.csv** — same stats as above, tournament games only. Show row count, season coverage.
  - [ ] 8.3: Load and document **MMasseyOrdinals.csv** (ordinal rankings from 100+ rating systems). Show: unique `SystemName` count, season coverage, week coverage, total row count. Identify which systems have the most complete coverage (useful for Epic 4 opponent adjustments).
  - [ ] 8.4: Load and document **MNCAATourneySeeds.csv** (seed assignments per season). Show: seasons covered, teams per year (should be 68 post-2011 First Four, 64 before). Confirm 2020 has no tournament seed data.
  - [ ] 8.5: Load and document **MTeamSpellings.csv** (alternate team name spellings). Show row count, unique team count, samples. Relevant to team name normalization in Epic 4.
  - [ ] 8.6: All CSV reads use `pd.read_csv(Path("../../data/kaggle/FileName.csv"))` (relative path from notebook location). Do NOT modify repository or save any data — read-only raw CSV access.

- [ ] Task 9: Implement Section 7 — Data Quality Summary & Recommendations (AC: 6)
  - [ ] 9.1: Write a structured markdown summary cell listing:
    - **Confirmed Issues:** Any actual data quality problems found (missing dates, duplicate IDs, impossible scores, team ID mismatches)
    - **Known Limitations:** Expected gaps (2020 no tournament, ESPN 2025 games have dates while Kaggle 1985–2024 do not, ESPN scope is most-recent season only)
    - **Recommendations for Epic 4:** Which columns need cleaning, how to handle the `date=None` problem (use `day_num` + season `day_zero` offset for Kaggle games), whether `MRegularSeasonDetailedResults.csv` should be ingested into the repository, Massey Ordinals top-5 systems to include
  - [ ] 9.2: Save the summary as a standalone markdown file `notebooks/eda/data_quality_findings.md` (separate from notebook, for easy reference by future epics)

- [ ] Task 10: Execute notebook end-to-end and commit (AC: 7)
  - [ ] 10.1: Run `conda run -n ncaa_eval jupyter nbconvert --to notebook --execute notebooks/eda/01_data_quality_audit.ipynb --output notebooks/eda/01_data_quality_audit.ipynb` to execute all cells in place
  - [ ] 10.2: Verify no cell errors (exit code 0 from nbconvert)
  - [ ] 10.3: Verify total cell execution count matches number of code cells
  - [ ] 10.4: Commit the executed notebook and findings document: `docs(eda): data quality audit notebook with executed outputs (Story 3.1)`

## Dev Notes

### Story Nature: EDA Notebook, Not Source Code

This is an **exploratory data analysis (EDA) story** — the primary deliverable is a Jupyter notebook committed with executed outputs, not Python module code. **No new code goes into `src/ncaa_eval/`** for this story. Story 3.2 and 3.3 build on the findings here.

Key constraint: The notebook is a research artifact, not production code. `mypy --strict` does NOT apply to notebooks (they are not in the `files` list in `pyproject.toml`). Ruff also does not check notebooks by default. However, all pandas operations should follow the vectorization mandate (no `iterrows()`) — this is a discipline constraint, not a lint gate.

### Jupyter Dependency Addition

JupyterLab is NOT in `pyproject.toml` yet. It must be added as a dev dependency:

```bash
POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval poetry add --group dev "jupyterlab>=4.0,<5"
conda run -n ncaa_eval pip install "jupyterlab>=4.0,<5"
```

**⚠️ `poetry add` conda env gap** — always follow `poetry add` with `conda run pip install`. See Memory note on this pattern.

**`poetry.lock` must be committed** after any `poetry add` — even a dev-group addition. Failing to update `poetry.lock` causes CI to fail with: *"pyproject.toml changed significantly since poetry.lock was last generated."*

### Repository API

All game data access goes through `ParquetRepository`:

```python
from pathlib import Path
from ncaa_eval.ingest import ParquetRepository

repo = ParquetRepository(base_path=Path("../../data/"))  # relative to notebooks/eda/

teams  = repo.get_teams()    # list[Team]
seasons = repo.get_seasons() # list[Season]

# Load ALL games (efficient: only reads existing partitions)
import pandas as pd
games_dfs = []
for s in seasons:
    games = repo.get_games(s.year)
    if games:
        games_dfs.append(pd.DataFrame([g.model_dump() for g in games]))

all_games_df = pd.concat(games_dfs, ignore_index=True)
```

**Game fields available:**
| Field | Type | Notes |
|---|---|---|
| `game_id` | `str` | Kaggle: `{season}_{day_num}_{w_id}_{l_id}`; ESPN: `espn_{id}` |
| `season` | `int` | Year, 1985–2025 |
| `day_num` | `int` | Day within season (0 = first day, 154 = tournament end approx) |
| `date` | `datetime.date \| None` | `None` for Kaggle games 1985-2024; actual date for ESPN 2025 games |
| `w_team_id` | `int` | Winner's team ID |
| `l_team_id` | `int` | Loser's team ID |
| `w_score` | `int` | Winner's final score |
| `l_score` | `int` | Loser's final score |
| `loc` | `Literal['H','A','N']` | H=home win, A=away win, N=neutral site |
| `num_ot` | `int` | Overtime periods (0 = regulation, default=0) |
| `is_tournament` | `bool` | True = NCAA Tournament game (default=False) |

**Season fields:**
- `year: int` — only field; no `day_zero` (the day_zero offset is stored in Kaggle CSVs, not persisted to Parquet)

**Team fields:**
- `team_id: int`, `team_name: str`, `canonical_name: str` (empty string = not yet mapped)

### Data Shape (Expected as of 2026-02-19)

Based on the sync run:
- **Teams:** ~384 Division I men's teams (Kaggle `MTeams.csv`)
- **Seasons:** 41 seasons (1985–2025)
- **Games:** ~41 season partitions. Approximate total: ~6,000 regular season games/year × 40 seasons + ~63 tournament games/year × 40 years = ~245,000+ games total
- **2025 Season:** Contains both Kaggle regular season games AND ESPN-enriched games (merged, no duplicates). ESPN games have actual `date` values; Kaggle games have `date=None`.

The ESPN marker file `data/.espn_synced_2025` is a hidden file (created by `SyncEngine.sync_espn()`). Its presence confirms ESPN data was merged into season 2025.

### The 2020 COVID Year

The 2020 NCAA Tournament was cancelled due to COVID-19. This has a specific impact on every downstream story:

- Regular season games were played normally through early March 2020
- The tournament (normally 63 games) was cancelled — zero `is_tournament=True` games for 2020
- `MNCAATourneySeeds.csv` has NO entries for 2020
- `MNCAATourneyCompactResults.csv` has NO entries for 2020
- Future models must: **train** on 2020 data (regular season happened) but **NOT evaluate** 2020 (no tournament outcomes)

**AC verification for 2020:**
```python
games_2020 = all_games_df[all_games_df["season"] == 2020]
assert games_2020["is_tournament"].sum() == 0, "2020 should have no tournament games"
assert len(games_2020) > 0, "2020 should have regular season games"
```

### Raw Kaggle CSV Files Not Yet in Repository

These CSVs are in `data/kaggle/` but NOT ingested into the Parquet repository. The audit should document them for Epic 4 planning:

| File | What it contains | Relevant to |
|---|---|---|
| `MRegularSeasonDetailedResults.csv` | Per-game box scores (FGM, FGA, 3P, FT, rebounds, assists, turnovers, steals, blocks, fouls) | Epic 4 feature engineering |
| `MNCAATourneyDetailedResults.csv` | Same as above, tournament only | Epic 4 + Epic 5 |
| `MMasseyOrdinals.csv` | 100+ rating system ordinal rankings per team per week | Epic 4 opponent adjustments |
| `MNCAATourneySeeds.csv` | Seed assignments (1–16 per region) per season | Epic 4 feature, Epic 7 bracket |
| `MTeamSpellings.csv` | Alternate name spellings → TeamID mapping | Epic 4 canonicalization |
| `MTeamConferences.csv` | Conference membership per season | Epic 4 features |
| `MGameCities.csv` + `Cities.csv` | Geographic game location data | Low priority (Epic 7) |

**Critical finding to document in summary:** `MRegularSeasonDetailedResults.csv` contains the core statistical features needed for Epic 4. The question of whether to ingest these into the Parquet repository (extending the `Game` schema with stat columns) vs. accessing them directly as CSV during feature engineering is a key architectural decision to raise in the findings document.

### Vectorization Mandate (Style Guide §6.2)

All pandas operations in the notebook must be vectorized — no `for` loops over rows:

```python
# ❌ Wrong — iterrows violates §6.2
for _, row in all_games_df.iterrows():
    if row["is_tournament"]:
        ...

# ✅ Correct — vectorized operations
tournament_games = all_games_df[all_games_df["is_tournament"]]
per_season_tournament = all_games_df.groupby("season")["is_tournament"].sum()
```

The ONE exception in notebooks: loading games via `repo.get_games(s.year)` inside a `for s in seasons` loop is acceptable — this is iterating over a Python list, not a DataFrame. Convert to DataFrame immediately and then use vectorized ops.

### Plotly Visualization Requirements

All charts must use Plotly (not matplotlib):

```python
import plotly.express as px
import plotly.graph_objects as go

# Bar chart example
fig = px.bar(
    per_season_df, x="season", y="game_count",
    title="Games per Season (1985–2025)",
    color_discrete_sequence=["#6c757d"],  # neutral color
)
fig.update_layout(template="plotly_dark")  # dark mode per UX spec
fig.show()
```

Use the project color palette:
- `#28a745` (green) — positive/good
- `#dc3545` (red) — anomalies/issues
- `#6c757d` (neutral gray) — structural/informational

### Notebook Location & Git Commit Convention

- **Location:** `notebooks/eda/01_data_quality_audit.ipynb`
- **Findings doc:** `notebooks/eda/data_quality_findings.md`
- **Commit type:** `docs(eda): ...` (documentation, not feat or chore)
- **Branch:** Already on `story/2-4-implement-sync-cli-smart-caching` — this should be the NEXT story branch. Create `story/3-1-data-quality-audit`.
- **Kernel:** Python 3 (ncaa_eval conda env)

**nbconvert execution command** (from repo root):
```bash
conda run -n ncaa_eval jupyter nbconvert \
  --to notebook \
  --execute \
  --ExecutePreprocessor.timeout=600 \
  notebooks/eda/01_data_quality_audit.ipynb \
  --output notebooks/eda/01_data_quality_audit.ipynb
```

`--ExecutePreprocessor.timeout=600` prevents timeout when loading all 41 seasons of game data.

### What NOT to Do

- **Do not** add any code to `src/ncaa_eval/`. This is purely exploratory.
- **Do not** call `repo.save_*()` anywhere in the notebook — read-only exploration only.
- **Do not** add `notebooks/` to the mypy `files` list in `pyproject.toml`.
- **Do not** add `notebooks/` to Ruff's check scope (it does not check `.ipynb` by default anyway).
- **Do not** commit the notebook with cleared outputs — the AC requires executed outputs to be committed.
- **Do not** try to access `cbbpy` or ESPN API directly. All data should come through the `ParquetRepository` or from the already-downloaded raw CSVs in `data/kaggle/`.
- **Do not** ingest `MRegularSeasonDetailedResults.csv` into the repository during this story — document it as a future recommendation only.
- **Do not** use `matplotlib` or `seaborn` — all visualizations use Plotly per architecture requirements.

### Previous Story Intelligence (from Story 2.4 — Sync CLI)

- `save_games()` **OVERWRITES** per-season Parquet partitions. The 2025 season partition contains BOTH Kaggle AND ESPN games merged together (no separate files). When loading `repo.get_games(2025)`, you get all games from both sources.
- ESPN games use `game_id = f"espn_{espn_game_id}"` — filtering `all_games_df["game_id"].str.startswith("espn_")` identifies ESPN-sourced games.
- Kaggle 2025 games for the regular season (before tournament cutoff) are also present — ESPN supplements, not replaces, Kaggle data for 2025.
- `Season.year` is the only field — no `day_zero`. The DayZero mapping (for converting `day_num` to calendar dates) lives in `data/kaggle/MSeasons.csv` column `DayZero`. If the notebook needs calendar dates for Kaggle games, load this CSV directly.

### Git Intelligence (Recent Commits)

Most recent work:
- `e8f5b82` Implement Sync CLI & Smart Caching (Story 2.4) — adds `sync.py`, `SyncEngine`, integration tests
- `f07ddc3` bump: version 0.1.0 → 0.2.0 — version bump after Story 2.4

**Commit pattern for this story:** `docs(eda): data quality audit notebook with executed outputs (Story 3.1)`

**Branch to create:** `story/3-1-data-quality-audit`

### Project Structure Notes

**Alignment with Architecture (Sections 5, 9):**
- Architecture Section 9 defines the unified project structure. `notebooks/` is not explicitly mentioned but is consistent with a data science project layout.
- `dashboard/` already exists at project root (empty, for Epic 7).
- `docs/` is a pure Sphinx source directory (Story 1.9) — notebooks do NOT go in `docs/`.
- `specs/research/` is for planning spikes (Story 2.1 learned this). However, Story 3.1 is not purely a planning spike — it produces an interactive notebook artifact. `notebooks/eda/` is the appropriate location.

**New files to create:**
- `notebooks/eda/01_data_quality_audit.ipynb` — the primary deliverable
- `notebooks/eda/data_quality_findings.md` — text summary of findings

**Files to modify:**
- `pyproject.toml` — Add `jupyterlab>=4.0,<5` to `[tool.poetry.group.dev.dependencies]`
- `poetry.lock` — Updated after `poetry add`

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Epic 3, Story 3.1]
- [Source: _bmad-output/planning-artifacts/epics.md#Requirements Inventory, FR2 (Persistent Local Store)]
- [Source: _bmad-output/implementation-artifacts/2-4-implement-sync-cli-smart-caching.md — SyncEngine, ESPN marker, 2025 data merge, save_games overwrite behavior]
- [Source: _bmad-output/implementation-artifacts/2-2-define-internal-data-schema-repository-layer.md — Game/Team/Season fields, ParquetRepository API]
- [Source: src/ncaa_eval/ingest/schema.py — Exact field types and constraints]
- [Source: src/ncaa_eval/ingest/repository.py — ParquetRepository implementation, hive partitioning paths]
- [Source: _bmad-output/planning-artifacts/template-requirements.md — Library-first, vectorization §6.2, Plotly, no iterrows()]
- [Source: specs/05-architecture-fullstack.md#Section 9 — Unified Project Structure]

## Dev Agent Record

### Agent Model Used

Claude Sonnet 4.6

### Debug Log References

### Completion Notes List

### File List
