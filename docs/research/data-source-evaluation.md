# Data Source Evaluation — NCAA Basketball Prediction Platform

**Date:** 2026-02-19 (updated)
**Story:** 2.1 (Spike) — Evaluate Data Sources
**Status:** Complete

## Executive Summary

This document evaluates 18 candidate NCAA basketball data sources for the NCAA_eval platform, covering API-accessible sources, Python packages, scrape-only analytics sites, and paid services. The recommended priority order is:

1. **Kaggle MMLM** (Primary) — Free, comprehensive historical data 1985+, MasseyOrdinals with 100+ ranking systems
2. **BartTorvik / cbbdata API** (Secondary) — Free REST API with ~30 endpoints, adjusted efficiency metrics 2008+
3. **sportsdataverse-py** (Secondary alt.) — Free Python package wrapping ESPN APIs, play-by-play 2002+
4. **KenPom / kenpompy** (Optional) — $20/yr subscription, gold-standard efficiency but fragile scraping
5. **ESPN API** (Deferred) — Free but undocumented, current-season focus
6. **ncaa-api** (Deferred) — Free REST API for live NCAA.com data
7. **Nate Silver / Silver Bulletin** (Monitor) — SBCB/COOPER ratings, no API or structured data
8. **EvanMiya** (Monitor) — Paid, unique lineup-level analytics, no API
9. **Warren Nolan** (Deferred) — Free, NET/RPI/Nitty Gritty reports, scrapable
10. **Haslametrics** (Deferred) — Free, scrapable, independent predictions
11. **Sports Reference / sportsipy** (Skip) — Broken, anti-scraping blocks
12. **FiveThirtyEight** (Skip) — Data last updated 2018
13. **Other sources** (Deferred/Skip) — Sagarin (in MasseyOrdinals), TeamRankings, ShotQuality ($3K/yr), SportsDataIO, NCAA portal, Massey Ratings

## Source Comparison Table

| # | Source | Cost | Coverage | Access Method | Python Pkg | Status | Priority |
|:---|:---|:---|:---|:---|:---|:---|:---|
| 1 | Kaggle MMLM | Free | 1985–present | `kaggle` CLI/API | `kaggle` 2.0.0 | Active (annual) | Primary |
| 2 | BartTorvik / cbbdata | Free | 2008–present | REST API (~30 endpoints) | R only; Python via `cbbpy` 2.1.2 | API has 2025-26 issues | Secondary |
| 3 | sportsdataverse-py | Free | 2002–present | ESPN API wrapper | `sportsdataverse` 0.0.40 | Active (98 stars) | Secondary alt. |
| 4 | KenPom | $20/yr | 2002–present | Web scraping | `kenpompy` 0.5.0 | Fragile | Optional |
| 5 | ESPN API | Free | Current season | Undocumented REST | via `cbbpy` / `sportsdataverse` | Working | Deferred |
| 6 | ncaa-api | Free | Current | REST API (ncaa.com proxy) | None (TypeScript) | Active (210 stars) | Deferred |
| 7 | Nate Silver / SBCB | Free/Paid | 1950–present | No API; Substack | None | COOPER coming 2026 | Monitor |
| 8 | EvanMiya | Paid | Current + recent | No API; Shiny app | None | Active (100+ D1 teams) | Monitor |
| 9 | Warren Nolan | Free | Current + ~4yr | No API; HTML scraping | None | Active | Deferred |
| 10 | Haslametrics | Free | Current season | No API; HTML scraping | None | Active | Deferred |
| 11 | Sagarin | Free | Current season | No API; HTML page | None | Active | Skip (in MasseyOrdinals) |
| 12 | TeamRankings | Free/Paid | Current + hist. | No API; scraping | None (community scraper) | Active | Deferred |
| 13 | ShotQuality | $50-250/mo | Current | REST API (paid) | None | Active | Skip (too expensive) |
| 14 | Sports Reference | Free | 1947–present | Blocked scrapers | `sportsipy` 0.6.0 | Broken | Skip |
| 15 | FiveThirtyEight | Free | 2014–2018 | GitHub static | None | Stale | Skip |
| 16 | Massey Ratings | Free | Varies | Web only | None | Available | Skip (in Kaggle) |
| 17 | SportsDataIO | Paid | Current | REST API | None | Active | Deferred |
| 18 | NCAA Official | Free | Limited | Web / `ncaa-api` | None | robots.txt blocks | Deferred |

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

### 6. cbbdata API (BartTorvik REST Alternative)

**Recommendation: SECONDARY (access via REST API, complement to cbbpy)**

#### Overview

The cbbdata API is a REST API built by Andrew Weatherman that provides structured access to BartTorvik data. It was created as a successor to the `toRvik` R package and offers ~30 endpoints covering team ratings, game logs, player stats, and NET rankings.

- **Base URL:** `https://cbbdata.aweatherman.com/`
- **GitHub:** `andreweatherman/cbbdata` — 21 stars, 13 open issues, last pushed 2024-10-31
- **Backend:** Flask + Python, SQL queries and direct file transfers
- **Client:** R package only (`cbbdata`); no official Python client

#### Key Endpoints

| Function | Description |
|:---|:---|
| `cbd_torvik_player_game()` | Detailed player box scores and advanced metrics (2008+) |
| `cbd_torvik_game_box()` | Game-level box scores |
| `cbd_torvik_factors()` | Four Factors data |
| `cbd_torvik_current_resume()` | NET rankings (updated daily) |
| `cbd_torvik_game_prediction()` | Game predictions powered by BartTorvik |
| `cbd_kenpom_ratings()` | KenPom ratings (if KenPom email matches cbbdata account) |

#### Authentication

- Free registration via the R client (`cbd_create_account()`)
- API key issued at registration
- No Python client exists — would need to call REST endpoints directly via `httpx`/`requests`

#### Known Issues

- **GitHub issue #20:** "No Bart Torvik Data for 2025-26 Season" (open)
- **GitHub issue #21:** Related 2025-26 data availability problems
- Last pushed Oct 2024 — may be slow to address current season data gaps
- R-only client limits Python ecosystem integration

#### Assessment

The REST API approach is more reliable than cbbpy's HTML scraping, but the 2025-26 data availability issues and R-only client are concerns. For Story 2.3, calling the REST endpoints directly from Python (via `httpx`) is feasible and would bypass cbbpy's scraping fragility.

---

### 7. sportsdataverse-py (ESPN API Wrapper)

**Recommendation: SECONDARY ALTERNATIVE**

#### Overview

`sportsdataverse-py` is a Python package that wraps ESPN's public APIs for multiple sports, including men's college basketball. It provides structured access to play-by-play data, box scores, and team/player information.

- **Package:** `sportsdataverse` v0.0.40 on PyPI
- **GitHub:** `sportsdataverse/sportsdataverse-py` — 98 stars, 32 open issues, last pushed 2026-01-25
- **License:** MIT
- **Supports:** pandas and polars DataFrames

#### Data Coverage

- Play-by-play data back to 2002
- Box scores and game summaries
- Team rosters and information
- Schedule data
- Based on ESPN's public (undocumented) APIs

#### Assessment

| Factor | Assessment |
|:---|:---|
| Data coverage | Good — play-by-play from 2002, broader than cbbpy's range |
| Maintenance | Active — last pushed Jan 2026, more active than cbbpy |
| Stability | Medium — still wraps undocumented ESPN endpoints |
| Python support | Native Python package (unlike cbbdata's R-only client) |
| Unique value | Play-by-play data, polars support |

#### Why Secondary Alternative?

- Overlaps significantly with cbbpy's ESPN coverage
- More actively maintained than cbbpy (pushed Jan 2026 vs Nov 2025)
- 32 open issues suggest some rough edges
- Worth evaluating in Story 2.3 as a potential replacement for cbbpy's ESPN scraping if cbbpy bugs persist

---

### 8. ncaa-api (henrygd)

**Recommendation: DEFERRED**

#### Overview

A free, open-source REST API that proxies live data from ncaa.com. Built in TypeScript, self-hostable, and actively maintained.

- **GitHub:** `henrygd/ncaa-api` — 210 stars, 3 open issues, last pushed 2026-02-15
- **License:** MIT
- **Rate limit:** 5 requests/second
- **Self-hostable:** Yes (Cloudflare Workers compatible)

#### Data Available

- Live scores and game data
- Team standings and stats
- Scoreboard data across sports
- Current-season focus (not historical)

#### Assessment

Active and well-maintained but limited to current NCAA.com data. No historical depth makes it unsuitable as a primary or secondary source. Potentially useful for real-time score updates in later stories.

---

### 9. Nate Silver / Silver Bulletin (SBCB)

**Recommendation: MONITOR**

#### Overview

After departing FiveThirtyEight, Nate Silver launched the **Silver Bulletin College Basketball (SBCB)** ratings on his Substack. SBCB is an enhanced Elo rating system applied to NCAA basketball games dating back to the 1949-50 season (250,000+ games).

#### Methodology

- **Base:** Elo rating framework (average team = 1500)
- **Margin of victory:** `(3 + s)^0.85` diminishing returns multiplier
- **Home court advantage:** Calculated per-team, does not transfer to neutral sites
- **Travel distance:** `~8 × m^(1/3)` Elo points
- **K-factor:** 38 (regular season), 56 (early season), 47.5 (tournament)

**Two versions:**
1. **Pure Elo** (free) — reverts ratings toward conference averages; interconference play matters most
2. **Bayesian Elo** (paid Substack) — incorporates AP/Coaches Poll priors, blended 50/50 with Pure Elo

#### COOPER Model

Nate Silver has announced **COOPER** as his next-generation college basketball model, planned for the 2026 tournament season. Details are sparse as of Feb 2026 — may supersede or extend SBCB.

#### Data Availability

- **No API** — ratings published as Substack posts with embedded tables
- **No downloadable CSV or structured data**
- Ratings visible on the free tier (Pure Elo); Bayesian version requires paid Substack subscription (~$8/month)
- Would require scraping Substack posts to extract ratings programmatically

#### Assessment

Intellectually valuable methodology but impractical as a data source: no API, no bulk data, and scraping Substack posts is fragile and low-yield. The COOPER model may change this if Silver publishes structured data. Worth monitoring but not building a connector for.

---

### 10. FiveThirtyEight (Legacy)

**Recommendation: SKIP**

#### Current Status

- **GitHub repo:** `fivethirtyeight/data` (17,284 stars, still public)
- **NCAA-related directories found:**
  - `march-madness-predictions/` — **last updated February 2018**
  - `march-madness-predictions-2015/` and `march-madness-predictions-2018/`
  - `historical-ncaa-forecasts/`
  - `ncaa-womens-basketball-tournament/`

The March Madness predictions data was **last updated in 2018** — 8 years ago. Nate Silver departed FiveThirtyEight, and the NCAA prediction model has not been maintained since. See Section 9 for Silver's current work.

#### Why Skip?

- Data is 8 years stale — no value for current modeling
- No replacement source from the ABC/538 brand
- Historical forecasts could theoretically serve as benchmark comparison data, but this is low priority

---

### 11. Scrape-Only Sources (No API)

The following sources provide valuable analytics data but have no API and would require web scraping or manual data extraction.

#### 11a. EvanMiya (evanmiya.com)

**Recommendation: OPTIONAL — Paid, unique player-level data**

- **Creator:** Evan Miyakawa
- **Cost:** Paid subscription (tiered; exact pricing not publicly listed). Trusted by 100+ D1 programs; NCAA-approved for coaching staff use.
- **Data:** Bayesian Performance Rating (BPR), team O-Rate/D-Rate, lineup ratings (2–5 man combos), player impact metrics, transfer portal rankings, game predictions
- **Coverage:** Current season + recent history; play-by-play-derived metrics
- **Tech stack:** Built with R/Shiny
- **Access:** No API; web-only dashboard. Scraping feasibility unknown — Shiny apps use WebSocket/reactive rendering, making traditional scraping difficult.
- **Unique value:** Lineup-level analytics and player-specific BPR are unique among evaluated sources. No other source provides adjusted 2-5 man lineup data.
- **Assessment:** High-quality proprietary data, but paid access + Shiny rendering + no export/API make programmatic integration impractical. Worth considering as a manual enrichment source for model validation rather than automated ingestion.

#### 11b. Sagarin Ratings (sagarin.com)

**Recommendation: SKIP — Already in MasseyOrdinals**

- **Creator:** Jeff Sagarin
- **URL:** `sagarin.com/sports/cbsend.htm`
- **Cost:** Free (web access)
- **Data:** Composite ratings combining Elo-chess and Bayesian approaches, SoS rankings
- **Coverage:** Current season, updated throughout the year
- **Access:** No API; static HTML page with ratings tables. Previously hosted on USA Today; now only on sagarin.com.
- **Anti-scraping:** No known measures, but HTML format varies and is not consistently structured
- **Unique value:** None — **Sagarin ordinal rankings already available in Kaggle MasseyOrdinals as "SAG"**
- **Assessment:** No independent connector needed. Kaggle MasseyOrdinals provides historical Sagarin rankings in structured CSV format.

#### 11c. Haslametrics (haslametrics.com)

**Recommendation: DEFERRED — Free, unique predictive approach**

- **Creator:** Erik Haslam
- **Cost:** Free
- **Data:** Predictive team ratings, game score predictions, efficiency metrics
- **Coverage:** Current season
- **Access:** No API; standard HTML tables. Scraping feasible (no Cloudflare/JS challenges reported). Community scraper exists (`github.com/fattmarley/cbbscraper` — scrapes KenPom, Haslametrics, and BartTorvik using Selenium).
- **Anti-scraping:** Minimal — appears to be a simple static site
- **Unique value:** Independent predictive model that differs from KenPom/BartTorvik approaches. Rankings are included in Massey Ratings composite.
- **Assessment:** Free and scrapable, but the data largely overlaps with BartTorvik's efficiency metrics. Haslametrics ordinal rankings are available via Kaggle MasseyOrdinals. Direct scraping only justified if granular Haslametrics predictions (game-level point spreads) are desired.

#### 11d. Warren Nolan (warrennolan.com)

**Recommendation: DEFERRED — Useful for NET/RPI verification**

- **Cost:** Free
- **Data:** Live RPI, NET rankings, Elo ratings, SoS, Nitty Gritty reports (used by NCAA selection committee), bracket projections, conference rankings, schedule data
- **Coverage:** Current season; historical data back several years (2022+ confirmed via URLs)
- **Access:** No API; standard HTML pages with data tables. URL structure is predictable (e.g., `/basketball/2026/rpi-live`)
- **Anti-scraping:** No known aggressive measures
- **Unique value:** NET rankings and Nitty Gritty reports mirror NCAA selection committee data. Only public source for comprehensive NET team sheets.
- **Assessment:** Valuable as a verification source for NET/RPI data. Scraping is feasible but adds maintenance burden. NET rankings are also available via cbbdata API (`cbd_torvik_current_resume()`). Defer unless NET-specific data becomes a modeling requirement.

#### 11e. TeamRankings (teamrankings.com)

**Recommendation: DEFERRED — Paid premium features**

- **Cost:** Free (basic stats) / Paid (premium picks and projections)
- **Data:** Team rankings, historical stats, win probabilities, pace/efficiency metrics, conference comparisons, opponent-adjusted stats
- **Coverage:** Current season + historical data
- **Access:** No official API. Community Python scraper exists (`github.com/MichaelE919/ncaa-stats-webscraper` — uses requests + BeautifulSoup to scrape team stats).
- **Anti-scraping:** Unknown — scraper projects exist, suggesting scraping is possible
- **Unique value:** Broad statistical coverage, but nothing not available from Kaggle + BartTorvik
- **Assessment:** Low priority. Overlaps significantly with existing sources. Community scraper could serve as a starting point if ever needed.

#### 11f. ShotQuality (shotqualitybets.com)

**Recommendation: SKIP — Expensive, betting-focused**

- **Creator:** Simon Gerszberg (Colgate University startup)
- **Cost:** Standard $49.99/month, Premium $249.99/month (CSV export requires Premium)
- **Data:** Proprietary shot-level data from computer vision analysis (50M+ data points). Expected scores, regression-based standings, matchup simulators.
- **Coverage:** NCAA Men's Basketball, NBA, WNBA, international
- **Access:** Has an API (`shotqualitybets.com/build`) — Premium plan required. CSV export only on Premium.
- **Unique value:** Truly unique computer-vision-derived shot quality data not available anywhere else
- **Assessment:** Too expensive for this project ($3,000/year for Premium). The shot-level data is unique and valuable but not essential for the current modeling scope. Would only be justified for professional-grade modeling with budget allocation.

---

### 12. Other Sources

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

#### NCAA Official Data (ncaa.org / stats.ncaa.org)

- NCAA.org provides limited public data (team rosters, standings, brackets)
- `stats.ncaa.org` has comprehensive stats but **robots.txt blocks all crawlers** (`Disallow: /`)
- No public API or bulk download capability
- The `ncaa-api` (Section 8) proxies some of this data via ncaa.com endpoints
- **Assessment: Not viable** as a direct programmatic data source

---

## Data Entity Coverage vs. Architecture Requirements

The architecture (Section 4.1) requires three core entities:

| Entity | Required Fields | Kaggle | BartTorvik | KenPom | ESPN | sportsdataverse |
|:---|:---|:---|:---|:---|:---|:---|
| **Team** | TeamID (int), Name (str), CanonicalName | `MTeams.csv` ✅ | Via schedule data | N/A | Via team endpoint | Via team endpoint ✅ |
| **Game** | GameID, Season, Date, WTeamID, LTeamID, WScore, LScore, Loc | Compact + Detailed results ✅ | Via `get_team_schedule()` ✅ | N/A | Via scoreboard ✅ | Via scoreboard ✅ |
| **Season** | Year (int) | `MSeasons.csv` ✅ | Implicit ✅ | Implicit ✅ | Implicit ✅ | Implicit ✅ |

### Additional Data Mapping

| Data Need | Kaggle | BartTorvik | KenPom | sportsdataverse | Scrape-only |
|:---|:---|:---|:---|:---|:---|
| Tournament seeds | `MNCAATourneySeeds.csv` ✅ | ❌ | ❌ | ❌ | Warren Nolan ⚠️ |
| Bracket structure | `MNCAATourneySlots.csv` ✅ | ❌ | ❌ | ❌ | ❌ |
| Massey ordinals (100+ systems) | `MMasseyOrdinals.csv` ✅ | ❌ | Via "POM" in ordinals | ❌ | Massey web ⚠️ |
| Box score stats | Detailed results (2003+) ✅ | Via game boxscore | ❌ | Play-by-play (2002+) ✅ | ❌ |
| Adjusted efficiency | ❌ (ordinal only) | AdjOE, AdjDE ✅ | AdjO, AdjD, AdjEM ✅ | ❌ | EvanMiya (paid) ⚠️ |
| Four Factors | ❌ | eFG%, TO%, OR%, FTRate ✅ | Same ✅ | ❌ | ❌ |
| Conference assignments | `MTeamConferences.csv` ✅ | ❌ | ✅ | ❌ | ❌ |
| Coach data | `MTeamCoaches.csv` ✅ | ❌ | ❌ | ❌ | ❌ |
| NET rankings | ❌ | Via cbbdata API ✅ | ❌ | ❌ | Warren Nolan ✅ |
| Play-by-play | ❌ | Via cbbpy ⚠️ | ❌ | ✅ (2002+) | ❌ |
| Lineup analytics | ❌ | ❌ | ❌ | ❌ | EvanMiya (paid) ⚠️ |
| Game predictions | ❌ | Via cbbdata API ✅ | ❌ | ❌ | Haslametrics ⚠️ |

*✅ = Available, ❌ = Not available, ⚠️ = Available but requires scraping or paid access*

### Team Name/ID Mapping Challenges

Cross-source team identification is a known challenge:
- **Kaggle** uses integer TeamIDs (e.g., 1104 = Alabama)
- **ESPN** uses different integer IDs (e.g., 150 = Duke)
- **BartTorvik/KenPom** use team name strings
- **Warren Nolan/Haslametrics/EvanMiya** use team name strings (varying formats)
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
- MasseyOrdinals provides KenPom, Sagarin, RPI, and 100+ other rankings without subscription/scraping risk

**Implementation:** Story 2.3 connector #1. Download full competition dataset as ZIP, extract to `data/kaggle/`.

### Priority 2: BartTorvik via cbbdata API / cbbpy (Required)

**Rationale:**
- Best free source for adjusted efficiency metrics and Four Factors (2008+)
- Two access paths: cbbdata REST API (~30 endpoints) or cbbpy HTML scraping
- cbbdata REST API is more stable than cbbpy's scraping, but R-only client means calling REST directly from Python
- `cbbpy` is installed and partially working (team schedules work, bulk scraping has bugs)
- `rapidfuzz` dependency useful for later team name matching

**Implementation:** Story 2.3 connector #2. Evaluate cbbdata REST API (via `httpx`) vs. cbbpy. Use `get_team_schedule()` for reliable cbbpy data. Consider cbbdata REST endpoints for bulk historical data if cbbpy's `get_games_range()` remains broken.

### Priority 3: sportsdataverse-py (Secondary Alternative)

**Rationale:**
- Native Python package wrapping ESPN APIs — more Pythonic than cbbdata's R client
- Play-by-play data back to 2002 — unique historical depth
- More actively maintained than cbbpy (pushed Jan 2026 vs Nov 2025)
- Supports both pandas and polars DataFrames

**Implementation:** Evaluate in Story 2.3 alongside cbbpy. May replace cbbpy for ESPN-based data if cbbpy bugs persist.

### Priority 4: KenPom via kenpompy (Optional)

**Rationale:**
- Gold-standard efficiency ratings but **high redundancy** with BartTorvik data
- Requires $20/yr subscription + fragile scraping chain
- KenPom ordinal rankings already available in Kaggle MasseyOrdinals
- Only justified if granular KenPom efficiency values (AdjEM, AdjO, AdjD) are needed beyond what BartTorvik provides

**Implementation:** Deferred. Only build connector if BartTorvik data proves insufficient for modeling needs.

### Priority 5: ESPN API (Deferred)

**Rationale:**
- Primary value is real-time current-season scores — not needed until late-stage features
- Undocumented and unstable
- Historical data available more reliably from Kaggle
- `cbbpy` and `sportsdataverse` already wrap ESPN endpoints

**Implementation:** Deferred to Story 2.3 as optional connector, or addressed ad-hoc when real-time data is needed.

### Monitor: Nate Silver (SBCB / COOPER), EvanMiya

**Rationale:**
- Silver's COOPER model for 2026 tournament may provide structured data — worth checking in March 2026
- EvanMiya's lineup-level analytics are unique but require paid subscription + Shiny scraping
- Neither is practical for automated ingestion today

**Implementation:** Check for COOPER data availability before tournament season. Consider EvanMiya only if lineup data becomes a modeling requirement.

### Deferred Scrape-Only: Warren Nolan, Haslametrics, TeamRankings

**Rationale:**
- All provide free data that could supplement primary sources
- Warren Nolan is the most useful (NET/RPI/Nitty Gritty) but NET is also available via cbbdata
- Haslametrics offers independent predictions but overlaps with BartTorvik
- TeamRankings overlaps heavily with existing sources

**Implementation:** Only build scrapers if specific data gaps are identified during modeling (Epic 5).

### Skip: Sports Reference, FiveThirtyEight, Sagarin, ShotQuality

**Rationale:**
- Sports Reference: Broken scrapers, aggressive anti-bot measures, 130 open issues
- FiveThirtyEight: Data last updated 2018, no longer maintained
- Sagarin: Already in Kaggle MasseyOrdinals as "SAG" — no independent value
- ShotQuality: $3,000/year for Premium CSV export — too expensive for this project

---

## Licensing & Cost Implications

| Source | Cost | License Concern | Action Required |
|:---|:---|:---|:---|
| Kaggle MMLM | Free | No redistribution — download at runtime | Users must accept competition rules on kaggle.com |
| BartTorvik / cbbdata | Free | No explicit API ToS | Respectful API usage / rate limiting |
| sportsdataverse-py | Free | MIT license | No concern |
| KenPom | $20/yr | ToS prohibits scraping | Users must provide own subscription; data not redistributable |
| ESPN API | Free | Undocumented — no explicit ToS | Use at own risk; may break without notice |
| EvanMiya | Paid (tiered) | Subscription required; no redistribution | Users must provide own subscription |
| ShotQuality | $50-250/mo | Paid API; CSV export Premium only | Too expensive; skip |
| Warren Nolan | Free | No explicit scraping policy | Respectful scraping with delays |
| Haslametrics | Free | No explicit scraping policy | Respectful scraping with delays |
| `kaggle` package | Free | Apache 2.0 | No concern |
| `cbbpy` package | Free | Apache 2.0 | No concern |
| `kenpompy` package | Free | GPL-3.0 | **GPL license** — if used, must comply with GPL for derived works |
| `sportsdataverse` pkg | Free | MIT | No concern |

**Note on kenpompy GPL-3.0:** The `kenpompy` package uses GPL-3.0. If integrated as a required dependency, the project's license must be GPL-compatible. Since KenPom is optional (Priority 4) and the project may be open-sourced, consider keeping `kenpompy` as a soft/optional dependency only.

---

## Data Processing Approaches Worth Replicating

Several evaluated sources employ sophisticated data processing techniques. Even where we skip the source itself, the underlying methodologies inform our own feature engineering (Epic 4) and modeling (Epic 5) approaches.

### Tempo-Free Efficiency (KenPom / BartTorvik)

Both KenPom and BartTorvik normalize basketball statistics to a per-possession basis, eliminating tempo effects:

- **Possessions formula:** `FGA - OR + TO + 0.475 × FTA` (KenPom uses 0.475 FT multiplier)
- **Adjusted Offensive Efficiency (AdjOE):** Points scored per 100 possessions, adjusted for opponent defensive strength. Calculated as `Raw_OE × (National_Avg_OE / Opponent_AdjDE)`.
- **Adjusted Defensive Efficiency (AdjDE):** Points allowed per 100 possessions, adjusted similarly.
- **Adjusted Tempo (AdjT):** Estimated possessions per 40 minutes if playing against a team at average D1 tempo.
- **Four Factors (Dean Oliver):** eFG%, Turnover Rate, Offensive Rebound Rate, Free Throw Rate — the four factors that "truly swing a basketball game," in descending order of importance.

**Key difference — BartTorvik's recency weighting:** BartTorvik adds time decay that KenPom does not. Games older than 40 days lose 1% emphasis per day until reaching 60% weight at 80+ days old. This captures team improvement/regression over a season.

**Replication value:** *HIGH* — Per-possession efficiency with opponent adjustment is the foundation of modern CBB analytics. We should implement this as a core feature transformation in Story 4.4 (Sequential Transformations). The recency weighting is worth implementing as a configurable parameter.

### Enhanced Elo Rating System (Nate Silver / SBCB)

Silver's SBCB extends basic Elo with several basketball-specific enhancements:

- **Margin of victory with diminishing returns:** `(3 + score_diff)^0.85` — prevents blowouts from distorting ratings
- **Per-team home court advantage:** Calculated individually (not a single global constant), excluded at neutral sites
- **Travel distance adjustment:** `~8 × miles^(1/3)` Elo points — diminishing impact
- **Variable K-factor:** 56 (early season, high uncertainty) → 38 (regular season) → 47.5 (tournament, high leverage)
- **Conference regression:** Between seasons, ratings regress toward conference average, making interconference play especially informative
- **Bayesian prior integration:** AP/Coaches Poll rankings as preseason priors, blended 50/50 with pure Elo

**Replication value:** *HIGH* — We plan to implement an Elo model as the reference stateful model (Story 5.3). Silver's enhancements (margin of victory diminishing returns, per-team home court, variable K-factor) represent the state of the art and should be implemented as configurable parameters. The FiveThirtyEight NBA Elo methodology (publicly documented) provides additional implementation guidance.

### Bayesian Performance Rating (EvanMiya)

EvanMiya's BPR is a multi-stage player evaluation framework:

1. **Multi-Year Regularized Adjusted Plus-Minus (RAPM):** Play-by-play data across 4 consecutive seasons. Bayesian regression assigns each player offensive/defensive coefficients based on scoring outcomes when they're on court. Regularization prevents erratic ratings for low-usage players.
2. **College-Specific Box Plus-Minus (Box BPR):** Box score stats regressed against RAPM coefficients to find college-specific skill weights (not NBA-trained). Determines which college skills predict winning.
3. **Prior-Informed RAPM:** Box BPR serves as Bayesian prior mean; play-by-play data adjusts from there. >50% weight on box score, remainder on plus-minus impact.
4. **Preseason Projections:** Recruiting ratings + historical performance + positional development → preseason prior distributions. Retains ~15% weight by season end.

**Core prediction:** `Expected_points = intercept + Σ(offensive_coefficients) - Σ(defensive_coefficients)`

**Replication value:** *MEDIUM* — The Bayesian framework for combining box score + play-by-play + priors is architecturally interesting for our modeling (Story 5.2+). However, we lack play-by-play data from Kaggle (only aggregate game stats), and BPR is player-level rather than team-level. The concept of using Bayesian priors from recruiting/preseason data is worth considering for team-level modeling.

### Shot Quality Model (ShotQuality)

ShotQuality uses computer vision to derive expected points from 5 components (up to 100 variables total):

1. **Defensive Distance:** Closest defender distance, multiple defenders within 5 feet, team defense quality
2. **Shooter Ability:** Elite/average/poor shooter classification based on historical data
3. **Play Type:** Transition, pick-and-roll, cut, isolation — what preceded the shot
4. **Shot Type:** Dunk, catch-and-shoot, off-the-dribble, shot distance
5. **Key Inferences:** Late shot clock → rushed, tall players → poor FT shooters, etc.

**Expected points = shot_probability × points_if_made**

**Replication value:** *LOW* — Requires proprietary computer vision data we don't have and can't access at reasonable cost ($3K/yr). However, the concept of modeling shot quality beyond simple FG% is valuable. We can approximate some of this using publicly available data: eFG% + FT rate + 3PT rate from Kaggle/BartTorvik serve as coarse proxies for shot quality.

### Massey Composite Methodology

Kenneth Massey aggregates 100+ independent ranking systems into a composite. His own ratings use:

- **Diminishing-returns margin of victory** (similar to Silver's approach)
- **Full-season re-analysis:** All games are re-weighted each week to find ratings that best explain all observed results
- **Home court factored in**
- **Early-season de-weighting**
- **Implicit schedule strength** from the rating model itself

**Replication value:** *HIGH* — The Kaggle MasseyOrdinals file is our most valuable single feature source. For our modeling, we should extract the historically most accurate ranking systems from MasseyOrdinals (top 10 by prior-year tournament accuracy) and use their ordinal rankings as features. A Kaggle top-1% gold solution (2023) used exactly this approach: filtered MasseyOrdinals to the top 10 historically accurate systems, then ensembled XGBoost + logistic regression + random forest on those rankings plus seed differences.

### NET Ranking (NCAA Official)

The NCAA Evaluation Tool (NET) combines:

- **Team Value Index (TVI):** Result-based, rewards beating quality opponents especially on the road
- **Adjusted Net Efficiency:** Points per 100 possessions margin, adjusted for opponent strength and game location

**Replication value:** *MEDIUM* — NET is the official selection committee metric. Available via cbbdata API. As a feature, NET rankings are useful but are already captured within MasseyOrdinals. The TVI concept (weighting wins by quality and location) is worth implementing as a custom feature.

### Community Approaches (Kaggle Competition Insights)

Based on publicly shared Kaggle MMLM solutions:

- **Top performers consistently use MasseyOrdinals** (not raw game stats) as primary features
- **Seed difference** remains a powerful standalone feature
- **Monte Carlo simulation** (2024 winner) — simulate tournament outcomes probabilistically rather than predicting individual games deterministically
- **Ensemble of simple models** (XGBoost + logistic regression + random forest) outperforms complex deep learning in this domain
- **Deep learning attempts** (CNN, RNN, ResNet, feedforward) have been explored but generally don't outperform simpler models for tournament prediction
- **Custom Elo implementations** within Kaggle competitions have been attempted but rarely beat using pre-computed external ratings

**Replication value:** *HIGH* — Our modeling framework (Epic 5) should support both stateful (Elo) and stateless (XGBoost) approaches, with MasseyOrdinals as a primary feature source. The Monte Carlo tournament simulator is already planned for Story 6.5.

### Multi-Source Scraping Pattern (cbbscraper)

The community `cbbscraper` project (`github.com/fattmarley/cbbscraper`) uses Selenium to scrape KenPom, Haslametrics, and BartTorvik simultaneously, cross-referencing predictions from all three sources. This multi-source consensus approach — checking agreement between independent models — is a valuable pattern for our platform.

**Replication value:** *MEDIUM* — Rather than scraping, we can achieve the same consensus effect using MasseyOrdinals (which already contains 100+ independent ranking systems). However, for game-level predictions (not just rankings), a multi-source scraping approach could add value in later stories.

---

## Items Requiring Live Verification

The following items could not be fully verified during this spike and should be confirmed before implementing data connectors in Story 2.3:

| # | Item | Blocked By | Impact |
|:---|:---|:---|:---|
| 1 | Kaggle 2026 competition existence | No API credentials configured | Determines latest available data year |
| 2 | `kaggle` v2.0.0 Python API imports | No API credentials | May need import path updates |
| 3 | `kenpompy` Cloudflare bypass | No KenPom subscription | Determines if KenPom connector is viable |
| 4 | `cbbpy` `get_games_range()` fix | Upstream bug | Determines if bulk BartTorvik scraping is practical |
| 5 | cbbdata REST API 2025-26 data availability | GitHub issues #20/#21 | Determines if REST API is viable for current season |
| 6 | `sportsdataverse` v0.0.40 functionality | Not yet tested | Determines if it replaces cbbpy for ESPN data |
| 7 | ESPN API rate limits | Manual testing needed | Affects connector design |
| 8 | BartTorvik exact year range | cbbpy/cbbdata testing needed | Determines historical data boundary |
| 9 | COOPER model data availability | Not yet announced | Check in March 2026 for structured data |

**Recommendation:** Configure Kaggle API credentials (`~/.kaggle/kaggle.json`) as a prerequisite for Story 2.3. Test `sportsdataverse` as an alternative to `cbbpy`. KenPom subscription is optional and can be deferred.

---

## Appendix: Package Version Summary

| Package | Version | Last PyPI Release | GitHub Stars | Open Issues | Last Push |
|:---|:---|:---|:---|:---|:---|
| `kaggle` | 2.0.0 | 2.0.0 (latest) | N/A | N/A | N/A |
| `cbbpy` | 2.1.2 | 2.1.2 (latest) | 25 | 5 | 2025-11-26 |
| `kenpompy` | 0.5.0 | 0.5.0 (latest) | 83 | 3 | 2025-11-02 |
| `sportsdataverse` | 0.0.40 | 0.0.40 (latest) | 98 | 32 | 2026-01-25 |
| `sportsipy` | 0.6.0 | 0.6.0 (only version) | 548 | 130 | 2025-01-31 |
| `cloudscraper` | 1.2.71 | 1.2.71 (latest) | N/A | N/A | N/A |
