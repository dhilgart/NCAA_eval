# NCAA_eval: EDA Findings & Feature Engineering Recommendations

Generated: 2026-02-20
Sources: Stories 3.1 (Data Quality Audit) and 3.2 (Statistical Exploration)

---

## Section 1: Confirmed Data Quality Issues & Cleaning Actions

Synthesized from `data_quality_findings.md` (Story 3.1) confirmed issues and recommendations.

| # | Issue | Severity | Cleaning Action for Epic 4 |
|---|-------|----------|---------------------------|
| 1 | `canonical_name = ""` for all 380 teams | HIGH | Build mapping from `MTeamSpellings.csv` (1,177 entries) in Story 4.3 |
| 2 | 2025 season: 4,545 games stored twice (Kaggle + ESPN IDs, same real-world game) | HIGH | Deduplicate by `(w_team_id, l_team_id, day_num)` — prefer ESPN records (API-verified `loc` and `num_ot`) |
| 3 | Box-score data not in Parquet repo (`MRegularSeasonDetailedResults.csv`, `MNCAATourneyDetailedResults.csv`) | HIGH | Epic 4 pipeline decision: ingest into `Game` schema (Option A, recommended) or access CSV directly (Option B) |
| 4 | 48 games with `num_ot >= 4` | LOW | Flag but keep — confirmed real outliers, not errors |
| 5 | 109 games with `w_score > 130`; 168 with `margin > 60` | LOW | Keep — legitimate historical data, not errors |
| 6 | 2025: `loc`/`num_ot` discrepancy between Kaggle and ESPN for same game | MEDIUM | ESPN values preferred (API-verified) — resolved by deduplication (Issue #2) |

### Deduplication Pattern for 2025

Any Epic 4 pipeline that aggregates 2025 data must apply this pattern:

```python
# Prefer ESPN records; drop Kaggle duplicates where an ESPN match exists
games_2025 = all_games_df[all_games_df["season"] == 2025].copy()
games_2025["matchup_key"] = (
    games_2025["w_team_id"].astype(str) + "_" +
    games_2025["l_team_id"].astype(str) + "_" +
    games_2025["day_num"].astype(str)
)
is_espn = games_2025["game_id"].str.startswith("espn_")
espn_keys = set(games_2025[is_espn]["matchup_key"])
kaggle_only = games_2025[~is_espn & ~games_2025["matchup_key"].isin(espn_keys)]
games_2025_dedup = pd.concat([games_2025[is_espn], kaggle_only], ignore_index=True)
```

### Box-Score Ingestion Decision (HIGH PRIORITY)

`MRegularSeasonDetailedResults.csv` (118,882 rows, 2003–2025) contains per-game features needed for most advanced features: FGM/FGA, 3P/3PA, FT/FTA, OR/DR, Ast, TO, Stl, Blk, PF for both winner and loser.

- **Option A (Recommended):** Extend the `Game` schema with box-score columns and ingest into Parquet. Cleaner API, consistent with repository-first architecture, higher upfront effort.
- **Option B:** Access `MRegularSeasonDetailedResults.csv` directly during feature engineering. Lower upfront cost, but mixes data access patterns.

**Recommendation: Option A** — consistent with the repository-first architecture. Implement in Story 4.2 or a dedicated ingestion subtask.

---

## Section 2: Ranked Feature Engineering Opportunities

Ranked by expected predictive value based on empirical evidence from Stories 3.1 and 3.2. All correlation values are Pearson r with tournament round reached (2003–2024 data, n=196,716 deduplicated games).

### Tier 1 — High Confidence (Direct Empirical Evidence from Story 3.2)

These signals have strong quantitative backing from the correlation analysis.

1. **Field Goals Made (FGM) / Scoring Volume**
   - r = **0.2628** (highest single-stat correlation with tournament advancement)
   - Winner average score: 76.9 pts vs. loser 64.8 pts
   - → Story 4.4 (rolling FGM/scoring average as sequential feature)

2. **Score / Scoring Efficiency**
   - r = **0.2349** with tournament advancement
   - Closely correlated with FGM; separating these may require possession-normalized metrics
   - → Story 4.4 (rolling scoring average)

3. **Field Goal Percentage (FGPct)**
   - r = **0.2269** with tournament advancement
   - Tournament differential: winners average **0.476** vs. losers **0.397** (+0.078 — largest single-stat gap between tournament winners and losers)
   - → Story 4.4 (sequential FGPct rolling average) + Story 4.7 (stateful feature serving)

4. **Strength of Schedule (SoS)**
   - r = **0.2970** with tournament advancement (p=3.16e-53, MEDIUM signal)
   - Mean SoS rises monotonically from R64 (0.516) → Sweet 16 → Elite 8 → Final Four → Champions (0.562)
   - Computed as mean regular-season opponent win rate up to each game date
   - → Story 4.5 (graph centrality — stronger SoS proxy via PageRank) and Story 4.6 (opponent-adjusted efficiency — most rigorous SoS)

5. **Personal Fouls (PF)**
   - r = **-0.1574** (largest single negative correlation)
   - High foul rate correlates with fewer tournament wins
   - → Story 4.4 (rolling PF average as negative feature)

6. **Turnover Rate (TO_rate)**
   - r = **-0.1424**
   - Tournament differential: winners average **0.147** vs. losers **0.155** (-0.008)
   - Computed as: `TO / (FGA + 0.44 × FTA + TO)` (per-possession normalization)
   - → Story 4.4 (rolling TO_rate)

7. **Defensive Rebounds (DR)**
   - Tournament winner differential: **+4.5 per game** (winners avg 25.9 vs. losers 21.3)
   - → Story 4.4 (rolling DR average)

### Tier 2 — Moderate Confidence (Structural / Domain Signals)

These signals have empirical backing but are more structural or domain-knowledge-driven.

8. **Home Court Advantage**
   - Home team wins **65.8%** of non-neutral regular-season games; home margin: +2.2 pts vs. neutral site
   - Linear trend: declining (slope = -0.00048/season, p=0.0006) — statistically significant decrease over 41 seasons
   - The declining trend suggests home advantage should be treated as a **time-varying feature** (not a static constant)
   - → Story 4.4 (`loc` encoding as numeric feature for stateless models; encode as H=1, A=-1, N=0)

9. **Seed Number**
   - Strong structural predictor: seed 1 averages 3.30 tournament wins; seed 16 averages 0.198
   - Most volatile matchups: **5v12**, **10v7**, **11v6** (~35–40% upset rates); **8v9** ≈ coin flip (~50%)
   - 1v16 nearly always chalk (1 historical upset: UMBC 2018)
   - Post-2011 First Four adds complexity for 11/16 seeds (`is_play_in` flag needed)
   - → Story 4.3 (`MNCAATourneySeeds.csv` integration, 2,626 rows 1985–2025; parse seed with `r"(\d+)"`)

10. **Conference Strength**
    - Top conferences by cumulative tournament wins: ACC, Big Ten, Big East, SEC, Big 12
    - Conference realignment creates year-over-year discontinuities (Big East split, Pac-12 collapse 2024)
    - Treat each `(season, conf_abbrev)` as a distinct entity — do NOT normalize conference names across years
    - → Story 4.3 (`MTeamConferences.csv` feature; 119,987 intra-conference games = 61.8% of conf-identified reg. season games)

### Tier 3 — Speculative (Requires Research Spike in Story 4.1)

These signals have theoretical support but need validation before full implementation.

11. **Graph Centrality (PageRank, betweenness)**
    - SoS (r=0.2970) is a direct proxy for network position in the win/loss graph; centrality measures may capture non-linear schedule strength
    - Graph should be directed (W→L edge) with edge weight = margin of victory
    - Incremental graph updates needed for walk-forward compatibility
    - → Story 4.5 (NetworkX — already in tech stack per Architecture Section 3); depends on 4.1 spike validation

12. **Opponent-Adjusted Efficiency**
    - Adjusts FGPct, scoring for opponent quality (KenPom-style)
    - More granular than raw SoS; target stats: FGPct (highest raw correlation) and scoring efficiency (FGM normalized by possessions)
    - Validate: opponent-adjusted efficiency should exceed SoS baseline (r=0.2970)
    - → Story 4.6 (linear algebra solver)

13. **Massey Ordinal Rankings**
    - 100+ ranking systems available via `MMasseyOrdinals.csv`
    - Top systems by season coverage (23 seasons each): AP, DOL, COL, MOR, POM
    - → Story 4.1 spike (research which systems add signal beyond raw box scores)

---

## Section 3: Known Data Limitations & Caveats

Epic 4 developers must be aware of these constraints when designing feature pipelines.

- **Box-score coverage starts 2003**: `MRegularSeasonDetailedResults.csv` covers 2003–2025 only (118,882 rows). No per-game FGM/FGA/etc. before 2003. Pre-2003 features are limited to compact results only (W/L, score, margin, location).

- **Tournament detailed results stop at 2024**: `MNCAATourneyDetailedResults.csv` has 1,382 rows through 2024. 2025 tournament box scores are unavailable until the tournament completes. Any training pipeline using tournament box scores must exclude the 2025 tournament.

- **2020 COVID year — No tournament**: 5,328 regular-season games were played, but `is_tournament == False` for all 2020 games (tournament cancelled). All feature pipelines must:
  - **Include** 2020 in training data (regular season data is valid)
  - **Exclude** 2020 from evaluation (no tournament outcomes)
  - Implement: `is_evaluation_year(season: int) -> bool: return season != 2020`

- **2025 deduplication required**: 4,545 games stored twice (Kaggle + ESPN IDs). Any aggregation touching 2025 must deduplicate first. See deduplication pattern in Section 1.

- **`canonical_name` is empty for all teams**: Team ID-based joins are reliable. Name-based joins require `MTeamSpellings.csv` mapping first (Story 4.3). `MTeamSpellings.csv` has 1,177 spelling entries for 380 teams.

- **Conference realignment discontinuities**: Conference abbreviations change across years (Big East split, Pac-12 collapse 2024). Treat each `(season, conf_abbrev)` as a distinct entity — do NOT attempt cross-year conference normalization.

- **Correlation analysis covers 2003–2024 only**: Tournament outcome correlations require detailed stats (2003+) and completed tournament data (through 2024). Correlation values in Section 2 are not applicable to pre-2003 seasons.

- **Survivorship bias in tournament stats**: Tournament games involve only 64–68 elite teams per year. Tournament-computed statistics (differentials, upset rates) are not representative of the full D1 population (~380 teams).

- **All games have dates**: All 201,261 games have a populated `date` field — Kaggle games via `DayZero + timedelta(days=day_num)`, ESPN games from API. No null-date handling needed in Epic 4 pipelines.

- **ESPN scope is current season only**: ESPN enrichment covers 2025 only. All historical tournament data (1985–2024) comes from Kaggle compact results.

---

## Section 4: Epic 4 Story-by-Story Guidance

Findings mapped to each Epic 4 story to provide specific direction for the Story 4.1 spike and subsequent implementations.

### Story 4.1 (Spike — Research Feature Engineering Techniques)

EDA provides directional evidence but does not definitively rank all feature candidates. The spike is confirmed necessary.

**Priority research questions:**
1. Do graph centrality metrics (PageRank, betweenness) add signal beyond naive SoS (r=0.2970 baseline)?
2. Which Massey Ordinal systems best complement box-score features? (Top 5 by coverage: AP, DOL, COL, MOR, POM — all 23 seasons)
3. What distribution family fits each stat? (Beta for rates like FGPct; Gamma/Poisson for counts like FGM, PF)
4. What rolling window size is optimal? (EDA has no strong evidence for 5 vs. 10 vs. 20 games — must be validated empirically)

**Reference:** `template-requirements.md` Section on "Epic 4 Normalization Design Requirements" — normalization configurability requirements (`gender_scope`, `dataset_scope`) are already specified.

### Story 4.2 (Chronological Data Serving API)

- All features show temporal trends: home advantage declining (p=0.0006), scoring patterns vary by era, conference membership changes annually
- API must support **per-game chronological streaming** for sequential features (Story 4.4) — each game should expose only information available at game time
- **2025 handling**: `get_chronological_season(2025)` must deduplicate before streaming (4,545 duplicate games)
- **2020 handling**: `get_chronological_season(2020)` must return regular-season data; the evaluation harness skips 2020

### Story 4.3 (Canonical Team ID Mapping & Data Cleaning)

- **`MTeamSpellings.csv`**: 1,177 spelling variants for 380 teams — primary canonical name mapping source
- **`MNCAATourneySeeds.csv`**: 2,626 rows, 1985–2025. Post-2011: 68 teams (First Four). Seed format: `"W01"` → parse with `r"(\d+)"` for numeric seed; `is_play_in = Seed.str.contains(r"[ab]$")`; region = `Seed.str[0]`
- **`MTeamConferences.csv`**: Conference membership per season — realignment means same team may appear under different `conf_abbrev` across years. Use `(season, team_id)` as the join key, not team name.

### Story 4.4 (Sequential Transformations)

Implement rolling features in this priority order (by expected predictive value from Section 2):

| Feature | r (tournament advancement) | Notes |
|---------|---------------------------|-------|
| Rolling FGPct | r=0.2269 (tournament diff: +0.078) | Highest tournament-game differential |
| Rolling FGM / Scoring | r=0.2628 / 0.2349 | Highest raw correlations |
| Rolling SoS (opponent win rate) | r=0.2970 | Compute as running mean of opponent regular-season win rates |
| Rolling TO_rate | r=-0.1424 | Negative predictor; include in feature set |
| Rolling PF | r=-0.1574 | Negative predictor; largest negative correlation |
| Rolling DR | winner diff: +4.5 | Defensive rebounding differential |
| `loc` encoding | 65.8% home win rate | H=1, A=-1, N=0 (or one-hot for tree models) |

**Window sizes:** EDA provides no strong evidence for optimal window. Research in Story 4.1 — candidates: last 5, 10, 20 games.

**2003 cutoff:** Rolling box-score features (FGPct, TO_rate, etc.) require `MRegularSeasonDetailedResults.csv` — available 2003–2025 only. Pre-2003 features are limited to compact stats (score, margin, loc).

### Story 4.5 (Graph Builders & Centrality)

- SoS signal (r=0.2970) is the strongest schedule-based signal; PageRank/centrality may improve on this — validate in Story 4.1 spike
- Use **`networkx`** (already in tech stack — Architecture Section 3)
- Graph should be **directed** (W→L edge) with edge weight = margin of victory
- **Incremental graph updates** needed for walk-forward compatibility (cannot recompute full graph at each time step)
- Validate: graph centrality features should exceed SoS (r=0.2970) baseline to justify the additional complexity

### Story 4.6 (Opponent Adjustments)

- Box-score coverage (2003–2025) is sufficient for efficiency adjustments
- **Target stats for adjustment:** FGPct (highest raw correlation r=0.2269) and scoring efficiency (FGM normalized by possessions)
- **Validation baseline:** Opponent-adjusted efficiency should exceed raw SoS (r=0.2970) — if not, the adjustment adds no value
- **2025 deduplication:** Apply deduplication before computing any opponent averages involving 2025 games

### Story 4.7 (Stateful Feature Serving)

Must combine outputs from all prior feature stories:

| Feature Source | Story | Notes |
|---------------|-------|-------|
| Sequential stats (FGPct, FGM, TO_rate, PF, DR, loc) | 4.4 | Rolling windows; 2003+ only for box scores |
| Graph centrality (PageRank, betweenness) | 4.5 | Directed W→L graph; incremental updates |
| Opponent-adjusted efficiency | 4.6 | Validated against SoS baseline |
| Seed info | 4.3 | (season, team_id) → seed_num, region, is_play_in |
| Conference membership | 4.3 | (season, team_id) → conf_abbrev |

**Normalization configurability required** (from `template-requirements.md` Section 9):
- `gender_scope: Literal["separate", "combined"] = "separate"`
- `dataset_scope: Literal["regular_season", "tournament", "combined"] = "regular_season"`

These must be configurable parameters, not hardcoded assumptions — normalization statistics differ substantially between regular season and tournament samples (survivorship bias in tournament data).

---

*This document serves as the Epic 4 planning reference. Story 4.1 (research spike) should read this document first before designing the feature engineering pipeline. All correlation values and data shapes are sourced from Stories 3.1 and 3.2 — do not recompute without re-running the EDA notebooks.*
