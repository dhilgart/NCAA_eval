# NCAA Basketball Statistical Exploration Findings

Generated from: `notebooks/eda/02_statistical_exploration.ipynb` (Story 3.2)

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total deduplicated games analyzed | 196,716 |
| Seasons covered | 1985–2025 (41 seasons) |
| Tournament games (through 2024) | 2,518 |
| Mean winning margin (all games) | 12.1 pts |
| Median winning margin | 10.0 pts |
| Overall home win rate (reg. season, non-neutral) | 0.658 (65.8%) |

---

## Section 1: Scoring Distributions

- Mean winning margin: **12.1 pts** (median: 10.0)
- Score distributions are right-skewed; winner scores average 76.9 pts, loser 64.8 pts
- 48 extreme OT games (≥4 OT) exist — confirmed real (Story 3.1 audit)
- Score outliers (w_score > 130, 109 games) are legitimate historical data

---

## Section 2: Venue Effects / Home Court Advantage

- Home team wins **65.8%** of non-neutral regular-season games
- Home win average margin: **13.6 pts** vs. neutral site: **11.4 pts**
- Home court advantage over neutral: **+2.2 pts**
- Linear trend slope: -0.00048/season (p=0.0006) — decreasing, statistically significant

---

## Section 3: Seed Patterns & Upset Rates

- 1 vs. 16 matchups: nearly always chalk (1 historical upset in data: UMBC 2018)
- Most upset-prone classic matchups: 5v12, 10v7, 11v6 (historically ~35–40% upset rates)
- 8v9 matchups approach 50% upset rate — essentially a coin flip
- Average tournament wins by seed 1: 3.30; seed 16: 0.198
- Post-2011 First Four adds play-in complexity for 11/16 seeds

---

## Section 4: Conference Strength

- Top conferences by tournament wins: acc, big_ten, big_east, sec, big_twelve
- Conference realignment (Big East split, Pac-12 collapse) creates analysis discontinuities
- Intra-conference games: 119,987 (61.8% of conf-identified reg. season games)

---

## Section 5: Statistical Correlations with Tournament Advancement

**Data coverage:** Regular-season stats 2003–2025; tournament outcomes 2003–2024.

### Top Positive Correlations (r with tournament round reached):
- **FGM**: r = 0.2470
- **Score**: r = 0.2205
- **FGPct**: r = 0.2171

### Top Negative Correlations:
- **FTA**: r = -0.0103
- **TO_rate**: r = -0.1358
- **PF**: r = -0.1576

### Tournament Game Differentials (Winner − Loser):
- FG%: Winners average 0.476 vs. 0.397 losers (diff: +0.078)
- TO Rate: Winners average 0.147 vs. 0.155 (diff: -0.008)
- Def Rebounds: Winners average 25.9 vs. 21.3 (diff: +4.5)

---

## Known Data Limitations

- **Box-score coverage:** 2003–2025 only (no pre-2003 detailed stats)
- **2020 COVID:** No tournament; excluded from all tournament analyses
- **2025 tournament:** Incomplete at data export time; excluded from outcome correlations
- **2025 deduplication:** Applied — ESPN records preferred over Kaggle duplicates
- **Conference names:** Not normalized across realignment years
