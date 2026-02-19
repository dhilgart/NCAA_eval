# Story 2.3: Implement Data Source Connectors

Status: ready-for-dev

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a data scientist,
I want connectors for each prioritized external data source that fetch raw data and map it to the internal schema,
So that I can ingest NCAA data from multiple sources into a unified format.

## Acceptance Criteria

1. **Given** the spike findings (Story 2.1) identify prioritized sources and the internal schema (Story 2.2) is defined, **When** the developer calls a connector for a specific source, **Then** the connector fetches raw data from the external source.
2. **And** raw data is cleaned and mapped to the internal Team/Game/Season schema.
3. **And** team name normalization maps diverse source-specific names to canonical IDs.
4. **And** each connector handles its source's quirks (authentication, pagination, data format).
5. **And** connectors raise clear errors on network failures, auth issues, or unexpected data formats.
6. **And** each connector is covered by tests (using mocked API responses).

## Tasks / Subtasks

- [ ] Task 1: Define Connector ABC and error types (AC: 4, 5)
  - [ ] 1.1: Create `src/ncaa_eval/ingest/connectors/` package with `__init__.py`
  - [ ] 1.2: Create `src/ncaa_eval/ingest/connectors/base.py` with `Connector` ABC and exception hierarchy
  - [ ] 1.3: Define abstract methods: `fetch_teams() -> list[Team]`, `fetch_games(season: int) -> list[Game]`, `fetch_seasons() -> list[Season]`
  - [ ] 1.4: Define exceptions: `ConnectorError` (base), `AuthenticationError`, `DataFormatError`, `NetworkError`
  - [ ] 1.5: Verify ABC and error types pass `mypy --strict`
- [ ] Task 2: Implement KaggleConnector (AC: 1, 2, 3, 4, 5)
  - [ ] 2.1: Create `src/ncaa_eval/ingest/connectors/kaggle.py`
  - [ ] 2.2: Implement `download(force=False)` to download+extract competition CSVs via `kaggle` API
  - [ ] 2.3: Parse `MTeams.csv` -> `list[Team]` (columns: TeamID, TeamName)
  - [ ] 2.4: Parse `MRegularSeasonCompactResults.csv` -> `list[Game]` (set `is_tournament=False`)
  - [ ] 2.5: Parse `MNCAATourneyCompactResults.csv` -> `list[Game]` (set `is_tournament=True`)
  - [ ] 2.6: Parse `MSeasons.csv` -> `list[Season]` + extract DayZero dates for date conversion
  - [ ] 2.7: Compute calendar `date` from `DayNum` and season's `DayZero`
  - [ ] 2.8: Construct `game_id` as `"{season}_{day_num}_{w_team_id}_{l_team_id}"` (matches test convention)
  - [ ] 2.9: Handle auth via `~/.kaggle/kaggle.json` or env vars `KAGGLE_USERNAME`/`KAGGLE_KEY`
  - [ ] 2.10: Raise `AuthenticationError` with helpful message on missing/invalid credentials
  - [ ] 2.11: Raise `DataFormatError` when CSV columns don't match expected schema
- [ ] Task 3: Implement ESPN connector via cbbpy (AC: 1, 2, 3, 4, 5)
  - [ ] 3.1: Create `src/ncaa_eval/ingest/connectors/espn.py`
  - [ ] 3.2: Implement `fetch_games(season)` using `cbbpy.mens_scraper.get_games_season(season)` with per-team schedule fallback
  - [ ] 3.3: Map ESPN game data to Game schema: determine winner/loser by score, ensure `w_score > l_score`
  - [ ] 3.4: Extract `date` from ESPN `game_day`, compute `day_num` from season DayZero mapping
  - [ ] 3.5: Determine `loc` from ESPN home/away/neutral context (map to `H`/`A`/`N`)
  - [ ] 3.6: Implement team name -> team_id lookup via provided team mapping dict
  - [ ] 3.7: Handle ESPN quirks: rate limiting, missing fields, broken endpoints
  - [ ] 3.8: Raise `NetworkError` on connection failures, `DataFormatError` on unexpected ESPN response shape
- [ ] Task 4: Wire up module exports (AC: 1)
  - [ ] 4.1: Update `src/ncaa_eval/ingest/connectors/__init__.py` to export all connector classes and exceptions
  - [ ] 4.2: Update `src/ncaa_eval/ingest/__init__.py` to re-export from connectors subpackage
- [ ] Task 5: Add dependencies to pyproject.toml (AC: all)
  - [ ] 5.1: Add `kaggle` dependency (requires Python 3.11+, compatible with project's 3.12)
  - [ ] 5.2: Add `cbbpy` dependency
  - [ ] 5.3: Run `POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval poetry lock` and install
  - [ ] 5.4: Add `# type: ignore[import-untyped]` comments for both packages
- [ ] Task 6: Unit tests for KaggleConnector (AC: 6)
  - [ ] 6.1: Create `tests/unit/test_kaggle_connector.py`
  - [ ] 6.2: Create CSV fixture files in `tests/fixtures/kaggle/` matching Kaggle MMLM format
  - [ ] 6.3: Test `fetch_teams()` -- MTeams.csv -> Team models (verify team_id, team_name)
  - [ ] 6.4: Test `fetch_games(season)` -- regular season CSV -> Game models with correct field mapping
  - [ ] 6.5: Test tournament games have `is_tournament=True`
  - [ ] 6.6: Test `fetch_seasons()` -- MSeasons.csv -> Season models
  - [ ] 6.7: Test date computation from DayNum + DayZero
  - [ ] 6.8: Test game_id format: `"{season}_{day_num}_{w_team_id}_{l_team_id}"`
  - [ ] 6.9: Test error handling: missing CSV, malformed CSV, auth failure (mock KaggleApi)
- [ ] Task 7: Unit tests for ESPN connector (AC: 6)
  - [ ] 7.1: Create `tests/unit/test_espn_connector.py`
  - [ ] 7.2: Create mock DataFrame fixtures matching cbbpy output columns
  - [ ] 7.3: Test `fetch_games(season)` -- ESPN data -> Game models with correct winner/loser ordering
  - [ ] 7.4: Test team name -> team_id mapping (hit and miss scenarios)
  - [ ] 7.5: Test `loc` mapping from ESPN home/away/neutral context
  - [ ] 7.6: Test error handling: network failure, unexpected response shape
  - [ ] 7.7: Mock `cbbpy.mens_scraper` functions
- [ ] Task 8: Run full quality pipeline (AC: all)
  - [ ] 8.1: `ruff check .` passes
  - [ ] 8.2: `mypy --strict src/ncaa_eval tests` passes
  - [ ] 8.3: `pytest` passes with all new tests green
  - [ ] 8.4: `mutmut run` on new connector modules (update `paths_to_mutate` in pyproject.toml)

## Dev Notes

### Resolved Story 2.1 Review Follow-ups

The Story 2.1 code review identified several items requiring resolution before Story 2.3. Web research during story creation has resolved them:

| Follow-up | Resolution |
|:---|:---|
| **[HIGH] Verify BartTorvik metrics via cbbpy** | **Confirmed: cbbpy does NOT provide BartTorvik data.** cbbpy is exclusively an ESPN scraper. AdjOE/AdjDE are not available through cbbpy. |
| **[HIGH] Validate sportsdataverse before relying on it** | **Deferred.** Package has 28 open GitHub issues, multiple data loader bugs, and ESPN API instability. Not reliable enough for MVP. |
| **[MEDIUM] Validate cbbdata REST API** | **Confirmed: cbbdata is R-only.** No Python client exists. The Flask API has no documented HTTP endpoints for direct Python access. |
| **[MEDIUM] Resolve cbbpy-vs-cbbdata for Priority 2** | **Resolved: Use cbbpy for ESPN game data only.** BartTorvik efficiency metrics are deferred -- Kaggle's MasseyOrdinals already includes KenPom-derived rankings ("POM" system). Direct BartTorvik access can be added in a future story via web scraping or the andrewsundberg Kaggle dataset. |
| **[LOW] Confirm sportsdataverse v0.0.40 installed** | **Moot.** sportsdataverse deferred from Story 2.3 scope. |

### Connector Scope Decision

**Implement two connectors:**
1. **KaggleConnector** (Primary) -- Historical game data 1985+, teams, seeds, tournaments
2. **EspnConnector** (Secondary) -- Current-season game data, calendar dates, schedule enrichment via cbbpy

**Deferred:**
- BartTorvik efficiency metrics (no Python client; Kaggle MasseyOrdinals provides partial coverage)
- sportsdataverse (unreliable; ESPN data already available via cbbpy)
- KenPom (paid subscription + GPL-3.0 license + fragile scraping)
- Warren Nolan (contradicted its own "Deferred" status in research doc)

### Connector ABC Design

```
Connector (ABC)
  fetch_teams() -> list[Team]
  fetch_games(season: int) -> list[Game]
  fetch_seasons() -> list[Season]
```

Not all connectors implement all methods. `EspnConnector.fetch_teams()` and `fetch_seasons()` should raise `NotImplementedError` -- ESPN provides game data only; Teams and Seasons come from Kaggle exclusively.

The ABC lives in `src/ncaa_eval/ingest/connectors/base.py`. Exceptions live in the same module.

### KaggleConnector Implementation Details

**Competition:** `march-machine-learning-mania-2025` (or latest available year). The connector should accept the competition name as a constructor parameter to support future years.

**kaggle API (v2.0.0):**
```python
from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore[import-untyped]
api = KaggleApi()
api.authenticate()  # reads ~/.kaggle/kaggle.json or env vars
api.competition_download_files(competition, path=extract_dir, force=force)
```

**Breaking change note:** kaggle v1.7.3 removed Swagger; imports from `kaggle.api.kaggle_api` (without `_extended`) are broken. Always use `kaggle.api.kaggle_api_extended.KaggleApi`.

**CSV file mapping (Men's tournament only -- M prefix):**

| CSV File | Entity | Key Fields |
|:---|:---|:---|
| `MTeams.csv` | Team | TeamID (int), TeamName (str) |
| `MRegularSeasonCompactResults.csv` | Game | Season, DayNum, WTeamID, LTeamID, WScore, LScore, WLoc, NumOT |
| `MNCAATourneyCompactResults.csv` | Game | Same as above (set `is_tournament=True`) |
| `MSeasons.csv` | Season | Season (int), DayZero (date str YYYY-MM-DD) |

**Field mapping for Game:**
- `game_id` = `f"{row.Season}_{row.DayNum}_{row.WTeamID}_{row.LTeamID}"` (matches existing test_repository.py convention)
- `season` = `row.Season`
- `day_num` = `row.DayNum`
- `date` = `DayZero + timedelta(days=row.DayNum)` (DayZero comes from MSeasons.csv)
- `w_team_id` = `row.WTeamID`
- `l_team_id` = `row.LTeamID`
- `w_score` = `row.WScore`
- `l_score` = `row.LScore`
- `loc` = `row.WLoc` (already H/A/N -- maps directly)
- `num_ot` = `row.NumOT`
- `is_tournament` = based on source CSV file

**DayZero date conversion:** MSeasons.csv provides a `DayZero` column (YYYY-MM-DD string) for each season. The calendar date for a game is `datetime.date.fromisoformat(day_zero) + datetime.timedelta(days=day_num)`.

**2020 COVID season:** Regular season data exists but no tournament CSV rows. The connector should handle this gracefully (no error on empty tournament file for 2020).

**Constructor parameters (keep under PLR0913 limit of 5):**
- `extract_dir: Path` -- where to download/find CSV files
- `competition: str = "march-machine-learning-mania-2025"` -- competition name

The `download()` method is separate from `fetch_*()` methods. This separates the network-dependent download step from the pure parsing step, making testing easier.

### EspnConnector Implementation Details

**cbbpy v2.1.2 API:**
```python
import cbbpy.mens_scraper as ms  # type: ignore[import-untyped]
# Option A (preferred if working): entire season
df = ms.get_games_season(2025)
# Option B (fallback): per-team schedule
df = ms.get_team_schedule("Duke", 2025)
```

**Known issues with cbbpy:**
- `get_games_range()` is broken (KeyError on `game_day` and `isConferenceGame`)
- `get_games_season()` may also fail for certain seasons; use `get_team_schedule()` as fallback
- Scraping speed: ~10s per team schedule, ~237s for 22 games via range endpoint -- very slow for bulk historical ingestion. ESPN connector is best suited for current/recent seasons only.

**Critical field mapping challenges:**

| Challenge | Solution |
|:---|:---|
| Winner/loser ordering | ESPN returns both teams' scores without guaranteed order. Parse `game_result` string (e.g., "W 75-60") or compare competitor scores. Ensure `w_score > l_score` before constructing Game. |
| Team IDs | ESPN uses different integer IDs than Kaggle. Accept a `team_name_to_id: dict[str, int]` mapping (built from Kaggle MTeams.csv). Use `rapidfuzz.fuzz.ratio()` for fuzzy matching when exact match fails. |
| `day_num` | ESPN provides calendar dates, not DayNum. Compute `day_num = (game_date - day_zero).days` using a `season_day_zeros: dict[int, datetime.date]` mapping (from KaggleConnector's MSeasons.csv). |
| `loc` | cbbpy's `get_team_schedule()` doesn't directly expose H/A/N. Infer from: neutral-site flag if available, or whether the team in question is listed as home team. Default to `"N"` if ambiguous. |
| `game_id` | Use ESPN game ID: `f"espn_{espn_game_id}"` to avoid collision with Kaggle game_ids. |

**Constructor parameters:**
- `team_name_to_id: dict[str, int]` -- mapping from team name strings to Kaggle TeamIDs
- `season_day_zeros: dict[int, datetime.date]` -- mapping from season year to DayZero date (for day_num computation)

Both mappings are produced by KaggleConnector -- Kaggle data must be loaded first to create the cross-source mapping context.

### Exception Hierarchy

```
ConnectorError (base, extends Exception)
  AuthenticationError -- credentials missing/invalid/expired
  DataFormatError -- CSV/API response doesn't match expected schema
  NetworkError -- connection failure, timeout, HTTP errors
```

All exceptions should include the source name and a human-readable message. Example: `AuthenticationError("kaggle: credentials not found. Create ~/.kaggle/kaggle.json or set KAGGLE_USERNAME/KAGGLE_KEY environment variables.")`.

### Dependency Notes

**New production dependencies:**
- `kaggle` -- v2.0.0 requires Python >= 3.11 (project uses 3.12, compatible). Pin `kaggle = "^2.0.0"`.
- `cbbpy` -- v2.1.2 requires Python >= 3.9. Pin `cbbpy = "^2.1.2"`. Brings `rapidfuzz` as transitive dependency (useful for team name matching in this story and Story 4.3).

**Install command:**
```bash
POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval poetry add "kaggle>=2.0.0" "cbbpy>=2.1.2"
```

**mypy notes:** Neither package ships `py.typed`. All imports need `# type: ignore[import-untyped]`:
```python
from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore[import-untyped]
import cbbpy.mens_scraper as ms  # type: ignore[import-untyped]
```

### Architecture Compliance

- **Section 5.1 (Ingestion Engine):** Connectors implement the `sync(source)` interface concept. Each connector fetches from one external source and produces domain models.
- **Section 8.2 (Data Access Layer):** Connectors produce `list[Team]`, `list[Game]`, `list[Season]` objects. The Repository layer (Story 2.2) handles persistence. Connectors do NOT write to disk directly.
- **Section 2.5 (Repository Pattern):** Connector -> domain models -> Repository.save_*() is the data flow.
- **Section 12 (Coding Standards):** `mypy --strict`, `from __future__ import annotations`, Google-style docstrings.

### File Structure

**New files to create:**
- `src/ncaa_eval/ingest/connectors/__init__.py` -- Package init with exports
- `src/ncaa_eval/ingest/connectors/base.py` -- Connector ABC + exception hierarchy
- `src/ncaa_eval/ingest/connectors/kaggle.py` -- KaggleConnector
- `src/ncaa_eval/ingest/connectors/espn.py` -- EspnConnector (cbbpy-backed)
- `tests/unit/test_kaggle_connector.py` -- KaggleConnector unit tests
- `tests/unit/test_espn_connector.py` -- EspnConnector unit tests
- `tests/fixtures/kaggle/` -- CSV fixture files for Kaggle tests

**Files to modify:**
- `src/ncaa_eval/ingest/__init__.py` -- Add connector exports
- `pyproject.toml` -- Add kaggle/cbbpy deps, update mutmut paths_to_mutate

### Testing Standards

- **All tests use mocked external dependencies.** No real network calls in unit tests.
- Mock `kaggle.api.kaggle_api_extended.KaggleApi` for download tests. For CSV parsing tests, use real CSV fixture files in `tests/fixtures/kaggle/`.
- Mock `cbbpy.mens_scraper` functions for ESPN tests. Return pre-built DataFrames matching cbbpy's actual output columns.
- Class-based test organization: `TestKaggleConnectorTeams`, `TestKaggleConnectorGames`, `TestEspnConnectorGames`, etc.
- `@pytest.mark.smoke` on the first happy-path test per class.
- `tmp_path` fixture for any file I/O. No `yield` on fixtures without teardown (PT022).
- `dict[str, Any]` for test factory kwargs (Pydantic mypy compatibility).
- Fixture factory helpers (e.g., `_make_kaggle_csv()`) following `test_repository.py` pattern.
- Pandera `DataFrameSchema` for validating raw CSV DataFrames before Pydantic conversion (catches upstream format changes early).

### Team Name Normalization (AC: 3)

For Story 2.3, "team name normalization" means:
- Kaggle provides canonical `TeamID` -> `TeamName` mapping via MTeams.csv
- Each Team's `canonical_name` starts as `""` (default from schema) -- full normalization happens in Story 4.3
- ESPN connector uses a `team_name_to_id` dict (built from Kaggle teams) to translate ESPN team names to Kaggle TeamIDs
- When fuzzy matching fails (no match above threshold), log a warning with the unmatched name and skip/flag the game rather than crashing

### Previous Story Intelligence

**From Story 2.2 (Schema & Repository):**
- Pydantic v2 models in `src/ncaa_eval/ingest/schema.py` with `populate_by_name=True`
- `Game` has `@model_validator(mode="after")` enforcing `w_score > l_score` AND `w_team_id != l_team_id` -- connectors MUST ensure correct winner/loser ordering before constructing Game objects
- `day_num: int` with `ge=0` is REQUIRED (no default) -- all connectors must provide a value
- `date: datetime.date | None` defaults to None -- Kaggle compact results don't need dates, but computing them from DayZero is straightforward
- `game_id: str` with `min_length=1` -- no canonical format enforced by schema; connector defines the format
- ParquetRepository in `src/ncaa_eval/ingest/repository.py` with `save_games()`, `save_teams()`, `save_seasons()`
- pyarrow imports need `# type: ignore[import-untyped]`
- `_apply_model_defaults()` handles schema evolution for missing columns in Parquet reads

**From Story 2.1 (Spike):**
- `cbbpy.get_team_schedule('Duke', 2025)` returns 39-row DataFrame with columns: `team, team_id, season, game_id, game_day, game_time, opponent, opponent_id, season_type, game_status, tv_network, game_result`
- `get_games_range()` is broken (KeyError on `game_day`)
- Kaggle CLI requires `~/.kaggle/kaggle.json` credentials (not configured during spike)
- ESPN scoreboard API confirmed working via curl

**From Epic 1 (Toolchain):**
- `from __future__ import annotations` required in ALL Python files
- `mypy --strict` mandatory
- Google-style docstrings
- `import pandas as pd  # type: ignore[import-untyped]` for pandas imports
- PLR0913: max 5 function arguments (use config objects if needed)
- Ruff PT022: use `return` (not `yield`) in fixtures without teardown

### Git Intelligence

Recent commits follow conventional commit format:
- `7a15683` Define internal data schema & Parquet repository layer (Story 2.2)
- `840feaa` Evaluate NCAA data sources for ingestion pipeline (Story 2.1)
- Branch pattern: `story/2-3-implement-data-source-connectors`
- Commit pattern: `feat(ingest): ...` for connector implementation

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Epic 2, Story 2.3]
- [Source: specs/05-architecture-fullstack.md#Section 5.1 -- Ingestion Engine]
- [Source: specs/05-architecture-fullstack.md#Section 8.2 -- Data Access Layer]
- [Source: specs/05-architecture-fullstack.md#Section 2.5 -- Repository Pattern]
- [Source: specs/05-architecture-fullstack.md#Section 12 -- Coding Standards]
- [Source: specs/03-prd.md#FR1, FR2, FR3]
- [Source: specs/research/data-source-evaluation.md -- Source evaluation findings]
- [Source: _bmad-output/implementation-artifacts/2-1-evaluate-data-sources.md -- Story 2.1 review follow-ups]
- [Source: _bmad-output/implementation-artifacts/2-2-define-internal-data-schema-repository-layer.md -- Schema + Repository implementation details]
- [Source: _bmad-output/planning-artifacts/template-requirements.md -- Testing/Pydantic/mypy patterns]

## Dev Agent Record

### Agent Model Used

{{agent_model_name_version}}

### Debug Log References

### Completion Notes List

### Change Log

### File List
