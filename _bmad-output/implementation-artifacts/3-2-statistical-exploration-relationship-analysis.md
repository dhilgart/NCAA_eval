# Story 3.2: Statistical Exploration & Relationship Analysis

Status: review

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a data scientist,
I want to explore statistical distributions, correlations, and patterns in the NCAA data,
So that I can identify signals and relationships worth pursuing in feature engineering.

## Acceptance Criteria

1. **Given** the data quality audit (Story 3.1) has identified the usable dataset, **When** the data scientist runs the exploration notebook, **Then** scoring distributions are visualized (home vs. away, by seed, by conference, over time).
2. **And** home/away/neutral venue effects are quantified (e.g., win rate difference, average margin shift).
3. **And** correlations between available statistics and tournament outcomes are analyzed.
4. **And** strength-of-schedule and conference-strength signals are explored.
5. **And** seed vs. actual performance patterns are documented (upset rates by seed matchup).
6. **And** all visualizations use Plotly for interactive inline rendering.
7. **And** the notebook is committed to the repository with reproducible outputs (all cells executed top-to-bottom, no errors).

## Tasks / Subtasks

- [x] Task 1: Create notebook and load data (AC: 1–6)
  - [x] 1.1: Create `notebooks/eda/02_statistical_exploration.ipynb`
  - [x] 1.2: Load compact game data from `ParquetRepository` (all 41 seasons, deduplicated)
  - [x] 1.3: Load `MNCAATourneySeeds.csv` directly from `data/kaggle/` for seed-based analysis
  - [x] 1.4: Load `MTeamConferences.csv` for conference-based analysis
  - [x] 1.5: Load `MRegularSeasonDetailedResults.csv` and `MNCAATourneyDetailedResults.csv` for stat correlations (2003–2024)
  - [x] 1.6: Apply 2025 deduplication: drop duplicate matchup tuples `(w_team_id, l_team_id, day_num)` keeping ESPN records (game_id startswith "espn_"), then drop Kaggle duplicates for same matchup

- [x] Task 2: Section 1 — Scoring Distributions (AC: 1)
  - [x] 2.1: Plot histogram of `margin = w_score - l_score` for all seasons combined (Plotly, dark mode)
  - [x] 2.2: Plot distributions of `w_score` and `l_score` — include mean/median annotations
  - [x] 2.3: Plot per-season average margin over time (1985–2025) as a line chart — annotate COVID 2020 gap
  - [x] 2.4: Break scoring distributions by `loc` (H/A/N) — box plots per location category
  - [x] 2.5: Compute win-by-location summary table: average margin by H/A/N, count of wins per category

- [x] Task 3: Section 2 — Venue Effects / Home Court Advantage (AC: 2)
  - [x] 3.1: For regular-season games only (`is_tournament == False`), compute home win rate by season: `loc == 'H'` share of all non-neutral games; plot as line chart
  - [x] 3.2: Compute average margin for home wins vs. away wins vs. neutral games; present as bar chart
  - [x] 3.3: Test whether home advantage has changed over time — plot `home_win_rate` vs. `year`; add trend line annotation
  - [x] 3.4: Neutral site breakdown: percentage of neutral-site games that are tournament vs. non-tournament (conference tourneys), by season

- [x] Task 4: Section 3 — Seed Patterns & Upset Rates (AC: 5)
  - [x] 4.1: Load `MNCAATourneySeeds.csv`; parse seed number (e.g., "W01" → 1) and region; build lookup `(season, team_id) → seed`
  - [x] 4.2: Merge seed data onto tournament games DataFrame to produce `(w_seed, l_seed)` per game
  - [x] 4.3: Build upset rate matrix: for each seed matchup (1v16, 2v15, … 8v9), compute upset rate (lower-seeded team wins). Show as heatmap (Plotly imshow or table)
  - [x] 4.4: Plot distribution of actual vs. expected wins by seed number — bar chart of "games won past round 1" per seed 1–16
  - [x] 4.5: Document notable historical upsets by seed: highest-upset-rate matchups, most frequent "chalk" results
  - [x] 4.6: Post-2011 seasons include 68 teams (First Four); distinguish "true" 16-seeds from play-in winners

- [x] Task 5: Section 4 — Conference-Strength Analysis (AC: 4)
  - [x] 5.1: Load `MTeamConferences.csv`; build `(season, team_id) → conference` lookup
  - [x] 5.2: Compute within-conference win rates for regular-season games (using `game_id` prefix to identify games between conference-mates; approximate by checking if both teams share the same conference in that season)
  - [x] 5.3: Compute tournament representation by conference per season: number of teams seeded 1–4 vs. 5–8 vs. 9–16 by conference
  - [x] 5.4: Compute inter-conference performance: win rate when Conference A plays Conference B (top 10 conference pairs by game volume)
  - [x] 5.5: Plot top 10 conferences by cumulative tournament wins (1985–2025) as a horizontal bar chart

- [x] Task 6: Section 5 — Statistical Correlations with Tournament Outcomes (AC: 3)
  - [x] 6.1: Load `MRegularSeasonDetailedResults.csv` and `MNCAATourneyDetailedResults.csv` from `data/kaggle/` directly (2003–2025)
  - [x] 6.2: Compute per-team, per-season regular-season averages: FGM, FGA, FGPct, 3P, 3PA, 3PPct, FTM, FTA, FTPct, OR, DR, Ast, TO, Stl, Blk, PF (both team and opponent columns)
  - [x] 6.3: Merge regular-season stats with tournament outcomes (did team reach Round of 32, Sweet 16, Elite 8, Final Four, Championship?)
  - [x] 6.4: Compute Pearson correlations between each stat and tournament advance depth; display as a ranked correlation table (top 10 positive, top 10 negative)
  - [x] 6.5: Plot scatter: `FGPct` vs. `tournament_round_reached` (2003–2024); similar for `TO_rate`, `3PPct`
  - [x] 6.6: For tournament games only: compute average stat difference between winner and loser; show which stats best differentiate tournament winners from losers
  - [x] 6.7: Note 2025 limitation: `MNCAATourneyDetailedResults.csv` stops at 2024; 2025 tournament hasn't completed yet

- [x] Task 7: Section 6 — Findings Summary & Epic 4 Recommendations (AC: 7)
  - [x] 7.1: Write a Markdown cell summarizing: top predictive signals found, conference/seed/venue findings, known data limitations
  - [x] 7.2: Save `notebooks/eda/statistical_exploration_findings.md` with machine-readable findings summary (analogous to `data_quality_findings.md` from Story 3.1)
  - [x] 7.3: Include a ranked list of recommended features for Epic 4 based on correlation analysis

- [x] Task 8: Execute notebook and commit (AC: 7)
  - [x] 8.1: Run via `conda run -n ncaa_eval jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 --output-dir notebooks/eda notebooks/eda/02_statistical_exploration.ipynb`
  - [x] 8.2: Verify 0 cell errors, all outputs rendered
  - [x] 8.3: Commit: `docs(eda): statistical exploration & relationship analysis notebook (Story 3.2)`

## Dev Notes

### Story Nature: EDA Notebook, Not Source Code

This is an **exploratory data analysis (EDA) story** — the primary deliverable is a Jupyter notebook committed with executed outputs, not Python module code. **No new code goes into `src/ncaa_eval/`** for this story. Story 3.3 uses these findings to produce the consolidated feature engineering recommendations document.

Key constraint: `mypy --strict` does NOT apply to notebooks. Ruff does not check notebooks by default. The vectorization mandate (no `iterrows()`) applies as a **discipline constraint**, not a lint gate.

### Data Loading Pattern

**Compact game data (all seasons) — via ParquetRepository:**

```python
from pathlib import Path
import pandas as pd
from ncaa_eval.ingest import ParquetRepository

repo = ParquetRepository(base_path=Path("../../data/"))
seasons = repo.get_seasons()

games_dfs = []
for s in seasons:
    games = repo.get_games(s.year)
    if games:
        games_dfs.append(pd.DataFrame([g.model_dump() for g in games]))

all_games_df = pd.concat(games_dfs, ignore_index=True)
```

**CRITICAL: 2025 deduplication** — 9,090 duplicate rows (4,545 games stored twice as Kaggle + ESPN IDs) found in Story 3.1. Before any aggregation involving 2025, drop duplicates:

```python
# Prefer ESPN records; drop Kaggle duplicates where an ESPN match exists
games_2025 = all_games_df[all_games_df["season"] == 2025].copy()
# Create a canonical key for cross-source matching
games_2025["matchup_key"] = (
    games_2025["w_team_id"].astype(str) + "_" +
    games_2025["l_team_id"].astype(str) + "_" +
    games_2025["day_num"].astype(str)
)
# Keep ESPN record where duplicate exists; otherwise keep Kaggle
is_espn = games_2025["game_id"].str.startswith("espn_")
# Deduplicate: for each matchup_key, prefer ESPN
games_2025_dedup = (
    games_2025.sort_values("game_id")  # ESPN IDs start with "espn_" which sorts before numbers
    .drop_duplicates(subset=["matchup_key"], keep="last")  # espn_ comes last lexicographically
    .drop(columns=["matchup_key"])
)
# Alternative: more explicit approach
espn_keys = set(games_2025[is_espn]["matchup_key"])
kaggle_only = games_2025[~is_espn & ~games_2025["matchup_key"].isin(espn_keys)]
games_2025_dedup = pd.concat([games_2025[is_espn], kaggle_only], ignore_index=True)

# Replace 2025 in all_games_df with deduplicated version
all_games_dedup = pd.concat([
    all_games_df[all_games_df["season"] != 2025],
    games_2025_dedup
], ignore_index=True)
```

**Detailed box-score data (2003–2025) — direct CSV reads:**

```python
# Relative to notebook location (notebooks/eda/)
KAGGLE_DIR = Path("../../data/kaggle/")

reg_detail = pd.read_csv(KAGGLE_DIR / "MRegularSeasonDetailedResults.csv")
tourney_detail = pd.read_csv(KAGGLE_DIR / "MNCAATourneyDetailedResults.csv")
seeds = pd.read_csv(KAGGLE_DIR / "MNCAATourneySeeds.csv")
conferences = pd.read_csv(KAGGLE_DIR / "MTeamConferences.csv")
```

**Seed parsing:** The `Seed` column in `MNCAATourneySeeds.csv` has format like `"W01"`, `"X16"`, `"Z11a"`. Extract numeric seed:

```python
import re
seeds["seed_num"] = seeds["Seed"].str.extract(r"(\d+)").astype(int)
seeds["region"] = seeds["Seed"].str[0]  # W, X, Y, Z
seeds["is_play_in"] = seeds["Seed"].str.contains(r"[ab]$", regex=True)
```

### Data Shapes for 2025

After deduplication:
- Regular season games (compact): ~6,000 per season; 2025 total ≈ 5,600–5,900 after dedup
- Detailed box scores: 118,882 rows (2003–2025, regular season); 1,382 rows (tourney, 2003–2024)
- Tournament seeds: 2,626 rows (1985–2025); 2020 has 0 seed entries (COVID)
- Conference membership: varies by season

### 2020 COVID Year

2020 has **no tournament games** (`is_tournament == False` for all 5,328 games). When analyzing tournament patterns:
- Explicitly exclude 2020 from tournament-based analyses
- For regular-season analyses, 2020 is valid and should be included

```python
tourney_games = all_games_dedup[all_games_dedup["is_tournament"] == True]
# 2020 will naturally have 0 rows here — correct behavior
```

### Venue Effects Analysis

The `loc` field is `Literal['H', 'A', 'N']`:
- `H` = home win (favored team hosting)
- `A` = away win (favored team was visiting)
- `N` = neutral site

For regular-season home court advantage analysis, filter to non-neutral site games:

```python
non_neutral = all_games_dedup[
    (all_games_dedup["is_tournament"] == False) &
    (all_games_dedup["loc"] != "N")
]
home_win_rate = (non_neutral["loc"] == "H").mean()
```

For 2025 data, note that Kaggle and ESPN disagree on `loc` for some games (Story 3.1 finding). ESPN values are preferred for 2025 games (already handled by deduplication keeping ESPN records).

### Detailed Stats Coverage

`MRegularSeasonDetailedResults.csv` covers **2003–2025** (118,882 rows). **Pre-2003 seasons have NO box-score data** — analyses involving FGM, FGA, etc. are limited to 2003+. Document this explicitly in the notebook.

Column reference for detailed results:

| Column | Meaning |
|---|---|
| `WTeamID`, `LTeamID` | Winner/Loser team IDs (join to teams table) |
| `WFGM` / `LFGM` | Field goals made |
| `WFGA` / `LFGA` | Field goals attempted |
| `WFGM3` / `LFGM3` | 3-pointers made |
| `WFGA3` / `LFGA3` | 3-pointers attempted |
| `WFTM` / `LFTM` | Free throws made |
| `WFTA` / `LFTA` | Free throws attempted |
| `WOR` / `LOR` | Offensive rebounds |
| `WDR` / `LDR` | Defensive rebounds |
| `WAst` / `LAst` | Assists |
| `WTO` / `LTO` | Turnovers |
| `WStl` / `LStl` | Steals |
| `WBlk` / `LBlk` | Blocks |
| `WPF` / `LPF` | Personal fouls |

**Derived stats to compute:**
```python
# FG%, 3P%, FT%
df["wfg_pct"] = df["WFGM"] / df["WFGA"]
df["w3p_pct"] = df["WFGM3"] / df["WFGA3"]
df["wft_pct"] = df["WFTM"] / df["WFTA"]
# Turnover rate, assist rate, rebound rate...
df["wto_rate"] = df["WTO"] / (df["WFGA"] + 0.44 * df["WFTA"] + df["WTO"])  # per possession
```

### Correlation Analysis Approach

To correlate regular-season stats with tournament outcomes:

1. Aggregate per-team regular-season stats to **season averages** from `MRegularSeasonDetailedResults.csv`
2. Determine tournament outcome per team per season from `MNCAATourneyCompactResults.csv` (furthest round reached)
3. Merge and compute Pearson correlations

Tournament round encoding (for `MNCAATourneyCompactResults.csv`):
- `DayNum` ranges: 134–135 = First Four/Round of 64; 136–137 = Round of 32; 143–144 = Sweet 16; 150–151 = Elite 8; 157–158 = Final Four; 163–164 = Championship

```python
# Map DayNum → round name for correlation analysis
def day_to_round(day: int) -> str:
    if day <= 135: return "R64"
    elif day <= 137: return "R32"
    elif day <= 144: return "S16"
    elif day <= 151: return "E8"
    elif day <= 158: return "FF"
    else: return "CHAMP"
```

### Conference Analysis Notes

`MTeamConferences.csv` columns: `Season`, `TeamID`, `ConfAbbrev`. Conference abbreviations change over time (Big East splits, conference realignments). Treat each `(Season, ConfAbbrev)` as a distinct entity. No need to normalize conference names across years for this exploratory story.

Top conferences by historical tournament presence (expected from domain knowledge):
- ACC, Big Ten, Big 12, SEC, Big East, Pac-12 (now Pac-2 after 2023 collapse)

### Plotly Visualization Requirements

All charts must use Plotly (not matplotlib) with dark mode:

```python
import plotly.express as px
import plotly.graph_objects as go

# Standard template for all figures
TEMPLATE = "plotly_dark"
COLORS = {
    "positive": "#28a745",  # Green — improvements/positive signals
    "negative": "#dc3545",  # Red — anomalies/issues
    "neutral":  "#6c757d",  # Gray — structural/informational
    "accent":   "#17a2b8",  # Teal — secondary accent
}

# Bar chart example
fig = px.bar(df, x="season", y="win_rate", template=TEMPLATE,
             color_discrete_sequence=[COLORS["neutral"]])
fig.show()
```

Use `fig.update_layout(template="plotly_dark")` consistently across all figures.

### Vectorization Mandate

No `iterrows()` in any cell. Common patterns:

```python
# ❌ Wrong
for _, row in df.iterrows():
    compute(row)

# ✅ Correct — vectorized
result = df["col"].apply(func)  # only when vectorized ops insufficient
result = df.groupby("col").agg(...)  # preferred
```

For seed lookup joins, use `pd.merge()` rather than row-by-row assignment.

### Notebook Location & Conventions

- **Notebook:** `notebooks/eda/02_statistical_exploration.ipynb`
- **Findings doc:** `notebooks/eda/statistical_exploration_findings.md`
- **Commit type:** `docs(eda): ...`
- **Branch:** Create `story/3-2-statistical-exploration-relationship-analysis`
- **Kernel:** Python 3 (ncaa_eval conda env)

**nbconvert execution from repo root:**
```bash
conda run -n ncaa_eval jupyter nbconvert \
  --to notebook \
  --execute \
  --ExecutePreprocessor.timeout=600 \
  --output-dir notebooks/eda \
  notebooks/eda/02_statistical_exploration.ipynb
```

**DO NOT use `--output notebooks/eda/02_statistical_exploration.ipynb`** — this doubles the path and writes to wrong location. Always use `--output-dir`.

### What NOT To Do

- **Do not** add any code to `src/ncaa_eval/`. This is purely exploratory.
- **Do not** call `repo.save_*()` anywhere — read-only exploration.
- **Do not** add `notebooks/` to `mypy` scope or Ruff checks.
- **Do not** commit with cleared outputs — all cells must be executed.
- **Do not** use `matplotlib`/`seaborn` — Plotly only.
- **Do not** include 2025 data without deduplicating first.
- **Do not** analyze tournament patterns for 2020 (COVID — no tournament).
- **Do not** use `iterrows()` over DataFrames.
- **Do not** attempt to ingest `MRegularSeasonDetailedResults.csv` into the Parquet repository — read directly as CSV for this story.
- **Do not** try to access `cbbpy` or ESPN API — use already-ingested data only.

### Previous Story Intelligence (from Story 3.1 — Data Quality Audit)

**Key findings from Story 3.1 that directly affect this story:**

1. **All games have dates:** Every game in the repository has a populated `date` field. No null-date handling needed.
2. **2025 deduplication (HIGH PRIORITY):** 4,545 games stored twice (ESPN + Kaggle IDs). Use deduplication pattern above.
3. **Canonical names empty:** `canonical_name = ""` for all 380 teams — team ID numeric joins are reliable; name-based joins are not.
4. **MRegularSeasonDetailedResults.csv confirmed:** 118,882 rows, 2003–2025, 34 columns. Primary source for stat correlations.
5. **MNCAATourneySeeds.csv confirmed:** 2,626 rows, 1985–2025. Post-2011 has 68 teams.
6. **Score outliers are real:** 109 games with `w_score > 130`, 168 with `margin > 60` — historical outliers, not errors. Include in analysis.
7. **Extreme OT games:** 48 games with `num_ot >= 4` — flagged for review. Consider filtering from scoring distribution analysis.
8. **`MNCAATourneyDetailedResults.csv` coverage:** Stops at **2024**. 2025 tournament box scores not available until tournament completes.

**Repository API established in Story 3.1:**
```python
repo = ParquetRepository(base_path=Path("../../data/"))  # relative to notebooks/eda/
teams  = repo.get_teams()    # list[Team]
seasons = repo.get_seasons() # list[Season]
games  = repo.get_games(season_year)  # list[Game]
```

**Notebook structure from Story 3.1 (established pattern):**
- Section headers as Markdown cells between code sections
- Summary variables (counts, percentages) printed at end of each section
- Plotly dark mode throughout
- `data_quality_findings.md` generated by a cell in the notebook itself (Path relative to kernel CWD = `notebooks/eda/`)

### Git Intelligence (Recent Commits)

Most recent relevant work:
- `be00fc8` Data quality audit notebook and findings (Story 3.1) — established EDA notebook conventions
- `e8f5b82` Implement Sync CLI & Smart Caching (Story 2.4) — data shape, merge behavior
- `63a1ac9` Implement Kaggle and ESPN data source connectors (Story 2.3) — connector field semantics

**Commit pattern for this story:** `docs(eda): statistical exploration & relationship analysis notebook (Story 3.2)`

### Project Structure Notes

**Alignment with Architecture (Section 2.3):**
- `notebooks/` is outside `src/ncaa_eval/` — not a production module. Notebooks are research artifacts.
- Architecture mandates Plotly for all visualizations (Section 3: Tech Stack — Plotly for charts).
- `data/` is gitignored; `data/.gitkeep` must remain intact.

**No new files in `src/`:** This story does not extend the package. Epic 4 will be where feature engineering code lands in `src/ncaa_eval/transform/`.

**New files to create:**
- `notebooks/eda/02_statistical_exploration.ipynb` — primary deliverable
- `notebooks/eda/statistical_exploration_findings.md` — generated by notebook

### References

- [Source: _bmad-output/implementation-artifacts/3-1-data-quality-audit.md — key findings, data shapes, established patterns]
- [Source: notebooks/eda/data_quality_findings.md — confirmed data quality findings and Epic 4 recommendations]
- [Source: _bmad-output/planning-artifacts/epics.md#Epic 3, Story 3.2 — acceptance criteria]
- [Source: _bmad-output/planning-artifacts/epics.md#Epic 4 — signals to prioritize for feature engineering]
- [Source: src/ncaa_eval/ingest/schema.py — Game/Team/Season field types]
- [Source: src/ncaa_eval/ingest/repository.py — ParquetRepository API]
- [Source: specs/05-architecture-fullstack.md#Section 3 — Tech Stack (Plotly mandatory)]
- [Source: _bmad-output/planning-artifacts/template-requirements.md — EDA notebook conventions, vectorization mandate, Plotly requirements]

## Dev Agent Record

### Agent Model Used

Claude Sonnet 4.6

### Debug Log References

- Fixed `loc_label` KeyError: column was added to `all_games_dedup` AFTER `reg_games` copy was made; moved `loc_label` mapping to setup cell before splits.
- Fixed `favored_seed` KeyError in sort_values: moved `sort_values` before column selection.
- Fixed `include_groups=False` + group key access in seed wins cell: rewrote with cleaner champions-via-idxmax approach.
- Verified actual DayNum→round mapping from data (max DayNum=154, not 164 as story notes suggested); updated mapping to match actual data.

### Completion Notes List

- Notebook `notebooks/eda/02_statistical_exploration.ipynb` created with 38 cells (7 markdown, 31 code), executed 0 errors.
- Key findings from actual data:
  - 196,716 deduplicated games (2025: 11,454 → 6,909 after dedup)
  - Mean winning margin: 12.1 pts; home win rate: 65.8% (non-neutral regular season)
  - Home advantage: +2.2 pts over neutral site; trend decreasing (p=0.0006, statistically significant)
  - 1v16 nearly always chalk; 5v12 and 8v9 most volatile matchups
  - FGM (r=0.247), Score (r=0.221), FGPct (r=0.217) are top positive predictors of tournament advancement
  - PF (r=-0.158) and TO_rate (r=-0.136) are top negative predictors
  - FG% differential: winners average 0.476 vs 0.397 for losers (+0.078) in tournament games
  - Top conferences by tournament wins: ACC, Big Ten, Big East, SEC, Big 12
- Findings saved to `notebooks/eda/statistical_exploration_findings.md` (2,898 bytes)
- All Plotly visualizations use `plotly_dark` template; no iterrows() used anywhere

### File List

- notebooks/eda/02_statistical_exploration.ipynb
- notebooks/eda/statistical_exploration_findings.md

## Change Log

- 2026-02-20: Story 3.2 implemented — statistical exploration EDA notebook with 6 sections covering scoring distributions, venue effects, seed patterns, conference strength, statistical correlations, and findings summary. Notebook executed with 0 errors, all 31 code cells completed.
