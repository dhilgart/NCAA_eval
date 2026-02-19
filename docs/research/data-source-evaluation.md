# Data Source Evaluation — NCAA Basketball Prediction Platform

**Date:** 2026-02-18
**Story:** 2.1 (Spike) — Evaluate Data Sources
**Status:** Complete

## Executive Summary

This document evaluates candidate NCAA basketball data sources for the NCAA_eval platform. Seven source categories were assessed for data coverage, API accessibility, rate limits, cost, and data quality. The recommended priority order is:

1. **Kaggle MMLM** (Primary) — Free, comprehensive historical data 1985+
2. **BartTorvik / cbbpy** (Secondary) — Free advanced metrics 2008+
3. **KenPom / kenpompy** (Optional) — $20/yr subscription, fragile scraping
4. **ESPN API** (Deferred) — Free but undocumented, current-season focus
5. **Sports Reference / sportsipy** (Skip) — Broken, anti-scraping blocks
6. **FiveThirtyEight** (Skip) — Data no longer updated
7. **Other sources** (Deferred) — Massey Ratings, SportsDataIO, NCAA portal

## Source Comparison Table

| Source | Cost | Coverage | Access Method | Python Package | Status | Priority |
|:---|:---|:---|:---|:---|:---|:---|
| Kaggle MMLM | Free | 1985–present | `kaggle` CLI/API | `kaggle` 2.0.0 | Active (annual) | 1 — Primary |
| BartTorvik | Free | 2008–present | Web scraping | `cbbpy` 2.1.2 | Partially working | 2 — Secondary |
| KenPom | $20/yr | 2002–present | Web scraping | `kenpompy` 0.5.0 | Fragile | 3 — Optional |
| ESPN API | Free | Current season | Undocumented REST | via `cbbpy` | Working | 4 — Deferred |
| Sports Reference | Free | 1947–present | Blocked scrapers | `sportsipy` 0.6.0 | Broken | Skip |
| FiveThirtyEight | Free | 2014–2018 | GitHub static files | None | Stale (last updated 2018) | Skip |
| Massey Ratings | Free | Varies | Web only | None (community scraper) | Available | Deferred |
| SportsDataIO | Paid | Current | REST API | None | Active | Deferred |
| NCAA Official | Free | Limited | Web only | None | Limited value | Deferred |

---

## Detailed Source Evaluations

### 1. Kaggle March Machine Learning Mania (MMLM)

**Recommendation: PRIMARY SOURCE**

#### Data Coverage

The Kaggle MMLM competition provides the most comprehensive freely available NCAA basketball dataset. CSV files include both Men's ("M" prefix) and Women's ("W" prefix) data:

| File | Year Range | Description |
|:---|:---|:---|
| `MRegularSeasonCompactResults.csv` | 1985–present | Season, DayNum, WTeamID, WScore, LTeamID, LScore, WLoc, NumOT |
| `MRegularSeasonDetailedResults.csv` | 2003–present | Compact + box score stats (FGM/A, FGM3/A, FTM/A, OR, DR, Ast, TO, Stl, Blk, PF) |
| `MNCAATourneyCompactResults.csv` | 1985–present | Tournament games, compact format |
| `MNCAATourneyDetailedResults.csv` | 2003–present | Tournament games, detailed format |
| `MNCAATourneySeeds.csv` | 1985–present | Season, Seed (e.g., "W01"), TeamID |
| `MNCAATourneySlots.csv` | 1985–present | Bracket structure (Slot, StrongSeed, WeakSeed) |
| `MTeams.csv` | N/A | TeamID, TeamName, FirstD1Season, LastD1Season |
| `MSeasons.csv` | 1985–present | Season metadata, DayZero offset, region assignments |
| `MTeamConferences.csv` | 1985–present | Team-to-conference mapping per season |
| `MTeamCoaches.csv` | 1985–present | Coach assignments by season and day range |
| `MConferences.csv` | N/A | Conference ID to name mapping |
| `MGameCities.csv` | 2010–present | Game locations (city, state) |
| `MMasseyOrdinals.csv` | 2003–present | 100+ third-party ranking systems (see below) |
| `MConferenceTourneyGames.csv` | 2001–present | Conference tournament results |
| `MSecondaryTourneyCompactResults.csv` | Varies | NIT and other secondary tournaments |

#### MasseyOrdinals — Key Asset

The `MMasseyOrdinals.csv` file contains ordinal rankings from 100+ ranking systems compiled by Kenneth Massey (masseyratings.com). Structure:

```
Season, RankingDayNum, SystemName, TeamID, OrdinalRank
```

Notable systems include:
- **POM** — KenPom (Ken Pomeroy) rankings
- **SAG** — Jeff Sagarin ratings
- **RPI** — Rating Percentage Index
- **MOR** — Massey ratings
- **COL** — Colley Matrix
- **AP** — Associated Press poll
- **USA** — USA Today Coaches poll

**This is the single most valuable file for modeling.** It provides KenPom-derived ordinal rankings without requiring a KenPom subscription or fragile scraping. Rankings are published at multiple points during each season, enabling temporal feature engineering.

**Limitation:** Only ordinal ranks (1st, 2nd, etc.) are included, not underlying efficiency values (AdjO, AdjD, AdjEM). For granular efficiency metrics, BartTorvik or KenPom is needed.

#### API Access

- **Package:** `kaggle` v2.0.0 (installed, Apache 2.0 license)
- **Authentication:** Requires `~/.kaggle/kaggle.json` API token (not currently configured)
- **v2.0.0 changes:** Major version bump. New `kagglesdk` dependency replaces internal API client. New `protobuf` dependency. CLI interface likely stable; Python import paths may have changed.
- **CLI usage:**
  ```bash
  kaggle competitions download -c march-machine-learning-mania-2025
  kaggle competitions files -c march-machine-learning-mania-2025
  ```

#### 2026 Competition Status

- The 2025 competition launched ~Feb 12, 2025 with a $50K prize pool and March 16 deadline.
- Web search results reference a March 21, 2026 submission deadline, suggesting the 2026 competition exists or is imminent.
- **Action needed:** Configure Kaggle API credentials and verify `march-machine-learning-mania-2026` competition existence.

#### Rate Limits & Terms of Service

- Generous rate limits; datasets download as a single ZIP file
- Must accept competition rules on kaggle.com before API download works (403 error otherwise)
- Data for personal/educational/research use; **no redistribution** (data must be downloaded from Kaggle at runtime, not committed to repo)
- The `data/` directory is already `.gitignore`d in this project

#### Data Quality

- High quality, maintained by Kaggle staff from NCAA records
- Consistent TeamIDs across all files and seasons
- No missing values in core results files
- 2020 season has regular-season data but no tournament games (COVID cancellation)
- Play-in ("First Four") games need special handling for 64-team bracket construction

#### Live Verification Needed

1. Configure `~/.kaggle/kaggle.json` credentials
2. Test `kaggle competitions download` CLI in v2.0.0
3. Test Python API imports (check if `kaggle.api.kaggle_api_extended.KaggleApi` still works in v2.0.0)
4. Confirm 2026 competition existence
5. Count exact ranking systems in current MasseyOrdinals file

---

### 2. BartTorvik / cbbpy

**Recommendation: SECONDARY SOURCE**

#### Data Coverage

BartTorvik (barttorvik.com) provides tempo-free advanced metrics for NCAA D1 men's basketball from 2008 to present:

- **T-Rank** — BartTorvik's overall team rating
- **Adjusted Efficiency** — AdjOE (offense), AdjDE (defense), adjusted for opponent strength
- **Four Factors** — eFG%, TO%, OR%, FTRate (both offensive and defensive)
- **Tempo** — Possessions per 40 minutes, adjusted
- **Strength of Schedule**
- **Game-by-game results and box scores**
- **Player statistics**

#### Python Package: `cbbpy` v2.1.2

- **Installed:** v2.1.2 (Apache 2.0 license)
- **Author:** Daniel Cowan (dcstats)
- **GitHub:** 25 stars, 5 open issues, last pushed 2025-11-26
- **Dependencies:** beautifulsoup4, joblib, lxml, numpy, pandas, platformdirs, python-dateutil, pytz, `rapidfuzz`, requests, tqdm
- **Data sources:** Scrapes both BartTorvik and ESPN

**Available functions (verified via import):**
- `get_game()`, `get_game_info()`, `get_game_boxscore()`, `get_game_pbp()`
- `get_games_range()`, `get_games_season()`, `get_games_team()`, `get_games_conference()`
- `get_team_schedule()`, `get_game_ids()`
- `get_conference_schedule()`, `get_teams_from_conference()`
- `get_player_info()`

#### Live Testing Results

**`get_team_schedule()` — WORKS:**
```python
import cbbpy.mens_scraper as ms
result = ms.get_team_schedule('Duke', 2025)
# Returns DataFrame with 39 rows, 12 columns:
# team, team_id, season, game_id, game_day, game_time,
# opponent, opponent_id, season_type, game_status, tv_network, game_result
```

**`get_games_range()` — PARTIALLY BROKEN:**
- Scraped 22 games on 2025-03-15 but took ~237 seconds
- Crashed with `KeyError: 'game_day'` during post-processing
- Log shows `KeyError: 'isConferenceGame'` and `KeyError: 'displayName'` on ESPN game info/PBP endpoints
- These errors suggest ESPN has changed their JSON response structure since cbbpy v2.1.2

#### Known Issues (from GitHub)

| # | Issue | Status |
|:---|:---|:---|
| 64 | `get_games_season` hangs or runs extremely slow for 2025 | Open |
| 63 | TypeError with numpy.ndarray | Open |
| 54 | ESPN shot chart inconsistencies | Open |
| 51 | Scraping pauses and doesn't finish | Open |

#### Access Method & Rate Limits

- **Scraping-based** — No official API. Scrapes ESPN and BartTorvik HTML/JSON endpoints
- **Rate limits:** Self-imposed by cbbpy (polite scraping with delays), but no explicit server-side enforcement documented
- **Speed concern:** ~237 seconds for 22 games is very slow for bulk historical data collection. A full season (~5,000+ games) would take hours.
- **Cost:** Free, no subscription required

#### Risk Assessment

- **Medium risk:** Active but infrequent maintenance (last commit Jan 2025). Known performance issues with 2025 season data.
- **Mitigation:** Use `get_team_schedule()` (works reliably) rather than `get_games_range()`. Consider scraping BartTorvik directly for bulk data if cbbpy's ESPN-based functions remain broken.

---

### 3. KenPom / kenpompy

**Recommendation: OPTIONAL (Priority 3)**

#### Data Coverage

KenPom (kenpom.com) provides the gold-standard tempo-free efficiency ratings from 2002 to present:

- **AdjEM** — Adjusted Efficiency Margin (primary ranking metric)
- **AdjO / AdjD** — Adjusted Offensive/Defensive Efficiency (per 100 possessions)
- **AdjT** — Adjusted Tempo
- **Four Factors** (offensive and defensive)
- **Strength of Schedule** (multiple formulations)
- **Luck** — Record deviation from efficiency margin prediction
- **Conference-level statistics**
- **Game-by-game efficiency breakdowns**

#### Cost

- **$19.99/year subscription** required for full data access
- Basic summary page partially visible without subscription
- No official API — subscription grants web access only

#### Python Package: `kenpompy` v0.5.0

- **Installed:** v0.5.0 (GPL-3.0 license)
- **Author:** Jared Andrews (j-andrews7)
- **GitHub:** 83 stars, 3 open issues, last pushed 2025-11-02
- **Dependencies:** bs4, `cloudscraper` 1.2.71, `mechanicalsoup`, pandas
- **PyPI versions:** 0.1.0 through 0.5.0 (11 releases total)

**Recent activity (from GitHub commits):**
- 2025-11-02: Version bump, changelog
- 2025-10-31: "New season fixes" (logging, FanMatch, conference changes)
- More active than expected — community contributions addressing seasonal breakage

**Available functions (verified via import):**
- `get_efficiency()`, `get_fourfactors()`, `get_teamstats()`
- `get_pointdist()`, `get_height()`, `get_kpoy()`, `get_playerstats()`

#### Scraping Viability — HIGH RISK

The scraping stack is inherently fragile:

1. **`cloudscraper`** bypasses Cloudflare JS challenges — an arms race with Cloudflare's evolving protections
2. **`mechanicalsoup`** simulates browser login — breaks if KenPom changes auth flow
3. **HTML parsing** — breaks if KenPom changes page layout

`cloudscraper` v1.2.71 is installed (latest). Whether it can currently bypass kenpom.com's Cloudflare protection **requires live testing with a subscription.**

#### Terms of Service

- KenPom ToS generally **prohibits automated scraping and data redistribution**
- Using `kenpompy` for personal research is in a gray area — technically against ToS but common in the Kaggle community
- **Raw KenPom data must NOT be committed to the repository**
- Any connector must require the user's own subscription credentials

#### Why Optional?

1. **Kaggle MasseyOrdinals already provides KenPom rankings** (as "POM" system) — sufficient for most modeling needs
2. **BartTorvik provides similar metrics for free** — adjusted efficiency, Four Factors, SoS
3. **Scraping is fragile** — investment may break at any time
4. **ToS compliance risk** — even for personal use

#### Live Verification Needed

1. Attempt `kenpompy` login with active subscription
2. Verify `cloudscraper` can bypass current Cloudflare protection
3. Test `get_efficiency()` data return for current season

---

### 4. ESPN Undocumented API

**Recommendation: DEFERRED (Priority 4)**

#### API Endpoints (verified working)

The ESPN public API is undocumented but functional. Live test confirmed:

**Scoreboard endpoint:**
```
GET https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?dates=YYYYMMDD
```

Returns JSON with structure:
```json
{
  "leagues": [...],
  "events": [
    {
      "id": "401706881",
      "name": "Louisville Cardinals at Duke Blue Devils",
      "date": "2025-03-16T00:30Z",
      "competitions": [
        {
          "competitors": [
            {"team": {"id": "150", "displayName": "Duke Blue Devils"}, "score": "73"},
            {"team": {"id": "97", "displayName": "Louisville Cardinals"}, "score": "62"}
          ],
          "venue": {...},
          "attendance": ...,
          "neutralSite": false,
          "playByPlayAvailable": true
        }
      ]
    }
  ]
}
```

Other likely endpoints (based on ESPN API patterns):
- `/rankings` — AP/Coaches polls
- `/standings` — Conference standings
- `/teams/{teamId}` — Team info
- `/summary?event={gameId}` — Game details

#### Assessment

| Factor | Assessment |
|:---|:---|
| Data quality | Good — official ESPN data |
| Stability | **Low** — undocumented, can change without notice |
| Historical data | Limited — primarily current season scoreboard |
| Rate limits | Unknown — no documentation |
| Coverage via cbbpy | `cbbpy` wraps these endpoints but has parsing bugs |
| Unique value | Real-time scores for current season |

#### Why Deferred?

- Kaggle provides historical game data more reliably
- Undocumented APIs can break at any time
- `cbbpy` already wraps ESPN but has active bugs (`KeyError` on response fields)
- Limited historical depth compared to Kaggle

---

### 5. Sports Reference / sportsipy

**Recommendation: SKIP**

#### Package Status

- **Version:** `sportsipy` 0.6.0 (installed, only version on PyPI)
- **GitHub:** 548 stars, **130 open issues**, last pushed 2025-01-31
- **License:** MIT

#### Why Skip?

Sports Reference (sports-reference.com) has implemented aggressive anti-scraping measures:
- CAPTCHAs, rate limiting, and IP blocking for automated access
- `sportsipy` has been largely broken since Sports Reference tightened scraping defenses
- 130 open issues with no recent meaningful maintenance suggest the project is effectively abandoned
- The data available via Sports Reference overlaps heavily with Kaggle MMLM data

---

### 6. FiveThirtyEight / Nate Silver

**Recommendation: SKIP**

#### Current Status

- **GitHub repo:** `fivethirtyeight/data` (17,284 stars, still public)
- **NCAA-related directories found:**
  - `march-madness-predictions/` — **last updated February 2018**
  - `march-madness-predictions-2015/` and `march-madness-predictions-2018/`
  - `historical-ncaa-forecasts/`
  - `ncaa-womens-basketball-tournament/`

The March Madness predictions data was **last updated in 2018** — 8 years ago. Nate Silver departed FiveThirtyEight, and the NCAA prediction model has not been maintained since.

#### Why Skip?

- Data is 8 years stale — no value for current modeling
- No replacement source from the ABC/538 brand
- Historical forecasts could theoretically serve as benchmark comparison data, but this is low priority

---

### 7. Other Sources

#### Massey Ratings (masseyratings.com)

- Aggregates 100+ college basketball ranking systems
- **Already available via Kaggle MasseyOrdinals** — the `MMasseyOrdinals.csv` file is sourced from Massey Ratings
- No official API; web-only access
- Community scraper exists (`github.com/Carrigan/mncaa`) but adds no value over Kaggle data
- **Assessment: No independent connector needed**

#### SportsDataIO

- Professional sports data API with NCAA basketball coverage
- Free trial exists but limited to UEFA Champions League data only
- **Paid plans required for NCAA data** — pricing not publicly listed (sales contact required)
- High-quality REST API with official documentation
- **Assessment: Deferred** — cost unknown, and Kaggle + BartTorvik cover the same data for free

#### NCAA Official Data

- NCAA.org provides limited public data (team rosters, standings, brackets)
- No public API or bulk download capability
- The NCAA partners with colleges for analytics challenges but doesn't provide open data feeds
- **Assessment: Not viable** as a programmatic data source

---

## Data Entity Coverage vs. Architecture Requirements

The architecture (Section 4.1) requires three core entities:

| Entity | Required Fields | Kaggle | BartTorvik | KenPom | ESPN |
|:---|:---|:---|:---|:---|:---|
| **Team** | TeamID (int), Name (str), CanonicalName | `MTeams.csv` ✅ | Via schedule data | N/A | Via team endpoint |
| **Game** | GameID, Season, Date, WTeamID, LTeamID, WScore, LScore, Loc | Compact + Detailed results ✅ | Via `get_team_schedule()` ✅ | N/A | Via scoreboard ✅ |
| **Season** | Year (int) | `MSeasons.csv` ✅ | Implicit ✅ | Implicit ✅ | Implicit ✅ |

### Additional Data Mapping

| Data Need | Kaggle | BartTorvik | KenPom |
|:---|:---|:---|:---|
| Tournament seeds | `MNCAATourneySeeds.csv` ✅ | ❌ | ❌ |
| Bracket structure | `MNCAATourneySlots.csv` ✅ | ❌ | ❌ |
| Massey ordinals (100+ systems) | `MMasseyOrdinals.csv` ✅ | ❌ | Via "POM" in ordinals |
| Box score stats | Detailed results (2003+) ✅ | Via game boxscore | ❌ |
| Adjusted efficiency | ❌ (ordinal only) | AdjOE, AdjDE ✅ | AdjO, AdjD, AdjEM ✅ |
| Four Factors | ❌ | eFG%, TO%, OR%, FTRate ✅ | Same ✅ |
| Conference assignments | `MTeamConferences.csv` ✅ | ❌ | ✅ |
| Coach data | `MTeamCoaches.csv` ✅ | ❌ | ❌ |

### Team Name/ID Mapping Challenges

Cross-source team identification is a known challenge:
- **Kaggle** uses integer TeamIDs (e.g., 1104 = Alabama)
- **ESPN** uses different integer IDs (e.g., 150 = Duke)
- **BartTorvik/KenPom** use team name strings
- `cbbpy` includes `rapidfuzz` dependency — useful for fuzzy name matching in Story 4.3
- The Kaggle `MTeams.csv` file provides a canonical TeamID-to-Name mapping that should serve as the primary reference

**Recommendation:** Build a mapping table from Kaggle TeamIDs to ESPN IDs and BartTorvik/KenPom names as part of Story 4.3 (Canonical Team ID Mapping).

---

## Recommended Priority Order

### Priority 1: Kaggle MMLM (Required)

**Rationale:**
- Only source providing tournament seeds, bracket structure, and MasseyOrdinals
- Complete historical game data from 1985 (compact) and 2003 (detailed)
- Clean, consistent data maintained by Kaggle staff
- Free, reliable API access via `kaggle` CLI
- MasseyOrdinals provides KenPom rankings without subscription/scraping risk

**Implementation:** Story 2.3 connector #1. Download full competition dataset as ZIP, extract to `data/kaggle/`.

### Priority 2: BartTorvik via cbbpy (Required)

**Rationale:**
- Best free source for adjusted efficiency metrics and Four Factors
- Data from 2008+ fills the advanced metrics gap that Kaggle's ordinal-only MasseyOrdinals can't cover
- `cbbpy` is installed and partially working (team schedules work, bulk scraping has bugs)
- `rapidfuzz` dependency useful for later team name matching

**Implementation:** Story 2.3 connector #2. Use `get_team_schedule()` for reliable data. Consider direct BartTorvik scraping for bulk historical data if cbbpy's `get_games_range()` remains broken.

### Priority 3: KenPom via kenpompy (Optional)

**Rationale:**
- Gold-standard efficiency ratings but **high redundancy** with BartTorvik data
- Requires $20/yr subscription + fragile scraping chain
- KenPom ordinal rankings already available in Kaggle MasseyOrdinals
- Only justified if granular KenPom efficiency values (AdjEM, AdjO, AdjD) are needed beyond what BartTorvik provides

**Implementation:** Deferred. Only build connector if BartTorvik data proves insufficient for modeling needs.

### Priority 4: ESPN API (Deferred)

**Rationale:**
- Primary value is real-time current-season scores — not needed until late-stage features
- Undocumented and unstable
- Historical data available more reliably from Kaggle
- `cbbpy` already wraps ESPN endpoints (when working)

**Implementation:** Deferred to Story 2.3 as optional connector, or addressed ad-hoc when real-time data is needed.

### Skip: Sports Reference, FiveThirtyEight

**Rationale:**
- Sports Reference: Broken scrapers, aggressive anti-bot measures, 130 open issues
- FiveThirtyEight: Data last updated 2018, no longer maintained

---

## Licensing & Cost Implications

| Source | Cost | License Concern | Action Required |
|:---|:---|:---|:---|
| Kaggle MMLM | Free | No redistribution — download at runtime | Users must accept competition rules on kaggle.com |
| BartTorvik | Free | No explicit API ToS | Respectful scraping (rate limiting) |
| KenPom | $20/yr | ToS prohibits scraping | Users must provide own subscription; data not redistributable |
| ESPN API | Free | Undocumented — no explicit ToS | Use at own risk; may break without notice |
| `kaggle` package | Free | Apache 2.0 | No concern |
| `cbbpy` package | Free | Apache 2.0 | No concern |
| `kenpompy` package | Free | GPL-3.0 | **GPL license** — if used, must comply with GPL for derived works |

**Note on kenpompy GPL-3.0:** The `kenpompy` package uses GPL-3.0. If integrated as a required dependency, the project's license must be GPL-compatible. Since KenPom is optional (Priority 3) and the project may be open-sourced, consider keeping `kenpompy` as a soft/optional dependency only.

---

## Items Requiring Live Verification

The following items could not be fully verified during this spike and should be confirmed before implementing data connectors in Story 2.3:

| # | Item | Blocked By | Impact |
|:---|:---|:---|:---|
| 1 | Kaggle 2026 competition existence | No API credentials configured | Determines latest available data year |
| 2 | `kaggle` v2.0.0 Python API imports | No API credentials | May need import path updates |
| 3 | `kenpompy` Cloudflare bypass | No KenPom subscription | Determines if KenPom connector is viable |
| 4 | `cbbpy` `get_games_range()` fix | Upstream bug | Determines if bulk BartTorvik scraping is practical |
| 5 | ESPN API rate limits | Manual testing needed | Affects connector design |
| 6 | BartTorvik exact year range | cbbpy testing needed | Determines historical data boundary |

**Recommendation:** Configure Kaggle API credentials (`~/.kaggle/kaggle.json`) as a prerequisite for Story 2.3. KenPom subscription is optional and can be deferred.

---

## Appendix: Package Version Summary

| Package | Version | Last PyPI Release | GitHub Stars | Open Issues | Last Push |
|:---|:---|:---|:---|:---|:---|
| `kaggle` | 2.0.0 | 2.0.0 (latest) | N/A | N/A | N/A |
| `cbbpy` | 2.1.2 | 2.1.2 (latest) | 25 | 5 | 2025-11-26 |
| `kenpompy` | 0.5.0 | 0.5.0 (latest) | 83 | 3 | 2025-11-02 |
| `sportsipy` | 0.6.0 | 0.6.0 (only version) | 548 | 130 | 2025-01-31 |
| `cloudscraper` | 1.2.71 | 1.2.71 (latest) | N/A | N/A | N/A |
