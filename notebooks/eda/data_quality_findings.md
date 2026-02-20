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
- **Duplicate game IDs:** 0 None found — data is clean ✓
- **Duplicate matchup tuples (all seasons):** 9090 ⚠ 9090 rows — all from 2025 ESPN+Kaggle overlap (see below)
- **Missing team IDs:** 0 All game team IDs found in teams table ✓
- **Score outliers:** 109 games with `w_score > 130`; 168 with `margin > 60`. These appear to be legitimate historical outliers (e.g., high-scoring eras), not data errors.
- **Extreme OT games:** 48 games with `num_ot >= 4`. Review recommended — could indicate data entry errors.

---

## Known Limitations (Expected Gaps)

- **`date = None` for Kaggle games (1985–2024):** The `Game.date` field is `None` for all Kaggle compact results. Only ESPN-sourced 2025 games have actual dates. To convert `day_num` to calendar dates for Kaggle games, use `data/kaggle/MSeasons.csv` column `DayZero` as the season start offset.
- **2020 COVID year:** No NCAA Tournament was held — `is_tournament = False` for all 5,328 2020 games (0 tournament games as expected). Future models must **train** on 2020 data but **NOT evaluate** it.
- **2025 data is hybrid:** Season 2025 contains 5,641 Kaggle regular-season games (no dates) PLUS 5,813 ESPN-sourced games (with actual dates). ESPN IDs are prefixed `espn_`. ⚠ **9,090 rows (4,545 games) appear in BOTH sources** under different game IDs — the same real-world game is stored twice. Epic 4 pipelines must deduplicate 2025 data by (w_team_id, l_team_id, day_num) before aggregating.
- **ESPN scope is current season only:** ESPN enrichment is limited to 2025. Historical 2025 data from Kaggle is the only source for pre-current-season tournament results.
- **No box-score data in repository:** `MRegularSeasonDetailedResults.csv` (118,882 rows, 2003–2025) and `MNCAATourneyDetailedResults.csv` (1,382 rows) are NOT ingested into the Parquet store. These contain field-goal, rebound, assist, turnover, and foul stats critical for Epic 4 features.

---

## Recommendations for Epic 4 (Feature Engineering)

1. **Box-score ingestion decision (HIGH PRIORITY):** `MRegularSeasonDetailedResults.csv` contains the per-game statistical features (FGM/FGA, 3P, FT, OR/DR, Ast, TO, Stl, Blk, PF) needed for most advanced features. Two options:
   - *Option A (Extend schema):* Add box-score columns to the `Game` model and ingest them into the Parquet store. Cleaner API, higher upfront effort.
   - *Option B (Separate CSV access):* Access `MRegularSeasonDetailedResults.csv` directly during feature engineering. Lower upfront cost, but mixes data access patterns.
   **Recommendation: Option A** — consistent with the repository-first architecture.

2. **2025 deduplication (HIGH PRIORITY):** The 2025 season stores 4,545 games twice (once as Kaggle IDs, once as ESPN IDs). Any feature pipeline that aggregates 2025 data must filter to a single source per game — e.g., prefer ESPN records (which have actual dates) and drop Kaggle records where a matching ESPN record exists for the same (w_team_id, l_team_id, day_num).

3. **Date reconstruction for Kaggle games:** Use `data/kaggle/MSeasons.csv` column `DayZero` (format: YYYY-MM-DD) as the season start date. Apply `day_zero + timedelta(days=day_num)` to recover calendar dates for Kaggle-sourced games.

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
