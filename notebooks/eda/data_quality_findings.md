# NCAA Basketball Data Quality Findings

Generated from: `notebooks/eda/01_data_quality_audit.ipynb` (Story 3.1)

---

## Dataset Summary

| Entity  | Count |
|---------|-------|
| Teams   | 380 |
| Seasons | 41 (1985–2025) |
| Games   | 201,261 total |

---

## Confirmed Issues

- **Canonical name coverage gap:** 380 of 380 teams (100.0%) have `canonical_name = ""` (empty-string default — not yet mapped to a canonical identifier). This will affect join operations in Epic 4.
- **Duplicate game IDs:** 0 — data is clean ✓
- **Duplicate matchup tuples (all seasons):** 9,090 rows ⚠ — all from 2025 ESPN+Kaggle overlap (see Known Limitations below)
- **Missing team IDs:** 0 — all game team IDs found in teams table ✓
- **Score outliers:** 109 games with `w_score > 130`; 168 with `margin > 60`. These appear to be legitimate historical outliers (e.g., high-scoring eras), not data errors.
- **Extreme OT games:** 48 games with `num_ot >= 4`. Review recommended — could indicate data entry errors.

---

## Known Limitations (Expected Gaps)

- **All games have dates (all seasons):** `KaggleConnector._parse_games_csv()` derives `Game.date` from `DayZero + timedelta(days=day_num)` for every Kaggle game. All 201,261 games across all 41 seasons have non-null dates. ESPN-sourced 2025 games have dates recorded directly from ESPN; Kaggle games (1985–2025) have dates derived from `MSeasons.csv` `DayZero`. The earlier assumption that "Kaggle games have `date=None`" is incorrect — the connector populates all dates.
- **2020 COVID year:** No NCAA Tournament was held — `is_tournament = False` for all 5,328 2020 games (0 tournament games as expected). Future models must **train** on 2020 data but **NOT evaluate** it.
- **2025 data is hybrid with field-level discrepancies:** Season 2025 contains 5,641 Kaggle regular-season games (dates derived from DayZero+DayNum) PLUS 5,813 ESPN-sourced games (dates from ESPN API). ESPN IDs are prefixed `espn_`. ⚠ **9,090 rows (4,545 games) appear in BOTH sources** under different game IDs — the same real-world game is stored twice. Additionally, Kaggle and ESPN disagree on `loc` and `num_ot` for some of these games (e.g., same game recorded as `loc='A', num_ot=1` in Kaggle vs `loc='N', num_ot=0` in ESPN). Epic 4 pipelines must deduplicate 2025 data by (w_team_id, l_team_id, day_num) and prefer ESPN values for `loc` and `num_ot` when sources conflict.
- **ESPN scope is current season only:** ESPN enrichment is limited to 2025. Kaggle compact results are the only source for historical tournament data.
- **No box-score data in repository:** `MRegularSeasonDetailedResults.csv` (118,882 rows, 2003–2025) and `MNCAATourneyDetailedResults.csv` (1,382 rows, 2003–**2024**) are NOT ingested into the Parquet store. Note: tournament detailed results stop at 2024 — 2025 tournament box scores will not be available until the tournament completes. These files contain field-goal, rebound, assist, turnover, and foul stats critical for Epic 4 features.

---

## Recommendations for Epic 4 (Feature Engineering)

1. **Box-score ingestion decision (HIGH PRIORITY):** `MRegularSeasonDetailedResults.csv` contains the per-game statistical features (FGM/FGA, 3P, FT, OR/DR, Ast, TO, Stl, Blk, PF) needed for most advanced features. Two options:
   - *Option A (Extend schema):* Add box-score columns to the `Game` model and ingest them into the Parquet store. Cleaner API, higher upfront effort.
   - *Option B (Separate CSV access):* Access `MRegularSeasonDetailedResults.csv` directly during feature engineering. Lower upfront cost, but mixes data access patterns.
   **Recommendation: Option A** — consistent with the repository-first architecture.

2. **2025 deduplication (HIGH PRIORITY):** The 2025 season stores 4,545 games twice (once as Kaggle IDs, once as ESPN IDs). Any feature pipeline that aggregates 2025 data must filter to a single source per game — prefer ESPN records (which have API-verified `loc` and `num_ot`) and drop Kaggle records where a matching ESPN record exists for the same (w_team_id, l_team_id, day_num).

3. **Date field is already populated:** All games already have `Game.date` populated (Kaggle via DayZero+DayNum, ESPN via API). No additional date reconstruction is needed for Epic 4 feature pipelines. The DayZero formula is available via `KaggleConnector.load_day_zeros()` if needed for external data alignment.

4. **canonical_name mapping:** 380 unmapped teams need canonical names to enable team-level joins with external data sources (BartTorvik, KenPom, etc.). Use `MTeamSpellings.csv` (1,177 spelling entries for 380 teams) to build the mapping.

5. **Massey Ordinals top systems:** For opponent-adjustment features in Epic 4, prioritize systems with broadest season coverage. Top candidates by seasons covered:
SystemName  seasons_covered
        AP               23
       DOL               23
       COL               23
       MOR               23
       POM               23

6. **2020 exclusion pattern:** All evaluation pipelines must skip 2020 as an evaluation year. A utility constant or predicate function (e.g., `is_evaluation_year(season: int) -> bool: return season != 2020`) should be defined in the Feature Engineering layer.

7. **Tournament seed data:** `MNCAATourneySeeds.csv` covers 1985–2025 with 2,626 rows. Post-2011 seasons have 68 teams (First Four). This is a valuable feature for Epic 4 seed-based features and for Epic 7 bracket visualization.

8. **Conference membership:** `MTeamConferences.csv` (not yet audited in detail) provides conference membership per season — useful for within-conference game features.
