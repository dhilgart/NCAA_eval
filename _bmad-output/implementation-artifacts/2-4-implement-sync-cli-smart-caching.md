# Story 2.4: Implement Sync CLI & Smart Caching

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a data scientist,
I want a CLI command `python sync.py --source [kaggle|espn|all] --dest <path>` that populates my local store with smart caching,
So that I can fetch historical data once and prefer local data on subsequent runs.

## Acceptance Criteria

1. **Given** data source connectors (Story 2.3) and the Repository layer (Story 2.2) are implemented, **When** the developer runs `python sync.py --source kaggle --dest data/`, **Then** the sync command fetches data from the specified source and persists it via the Repository.
2. **And** `--source all` fetches from all configured sources (kaggle then espn, in that order).
3. **And** on subsequent runs, the caching layer checks for valid local data before making remote API calls (Parquet files checked before calling connector fetch methods).
4. **And** the cache can be bypassed with a `--force-refresh` flag.
5. **And** sync progress is displayed to the user (source being fetched, records written, cache hits reported).
6. **And** the sync command is covered by integration tests validating the full fetch-store-cache cycle.

## Tasks / Subtasks

- [x] Task 1: Add CLI and progress dependencies (AC: 1–6)
  - [x] 1.1: Add `typer[all]` as a production dependency (provides CLI + rich for progress)
  - [x] 1.2: Run `POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval poetry add "typer[all]>=0.15"`
  - [x] 1.3: Add `sync.py` (root) to mypy `files` list in `pyproject.toml` to enforce strict type-checking

- [x] Task 2: Implement `SyncEngine` class (AC: 1–5)
  - [x] 2.1: Create `src/ncaa_eval/ingest/sync.py` with `SyncResult` dataclass and `SyncEngine` class
  - [x] 2.2: `SyncResult` fields: `source: str`, `teams_written: int`, `seasons_written: int`, `games_written: int`, `seasons_cached: int`
  - [x] 2.3: `SyncEngine.__init__(self, repository: Repository, data_dir: Path)` — stores repo reference and data root
  - [x] 2.4: Implement `sync_kaggle(force_refresh: bool = False) -> SyncResult`:
    - Create `KaggleConnector(extract_dir=self._data_dir / "kaggle")`
    - Call `connector.download(force=force_refresh)` (CSV-level caching already built-in)
    - Cache check teams: if `{data_dir}/teams.parquet` exists and not `force_refresh` → skip `fetch_teams()` + save
    - Cache check seasons: if `{data_dir}/seasons.parquet` exists and not `force_refresh` → load seasons from repo instead of fetch
    - Cache check games per-season: if `{data_dir}/games/season={year}/data.parquet` exists and not `force_refresh` → record as cache hit, skip `fetch_games(year)`
    - Print progress: `"[kaggle] teams: {n}"`, `"[kaggle] season {year}: {n} games written"`, `"[kaggle] season {year}: cache hit, skipped"`
  - [x] 2.5: Implement `sync_espn(force_refresh: bool = False) -> SyncResult`:
    - Load `team_name_to_id` from `repo.get_teams()` and `season_day_zeros` from KaggleConnector's CSV cache
    - Raise `RuntimeError` with helpful message if teams or seasons are empty (Kaggle not synced)
    - ESPN scope: sync only the most recent available season (max season year from `repo.get_seasons()`)
    - Cache check via `.espn_synced_{year}` marker file → skip if exists and not `force_refresh`
    - Create `EspnConnector(team_name_to_id=..., season_day_zeros=...)`, merge with existing Kaggle games before saving
    - Print progress: `"[espn] season {year}: {n} games written"` or `"[espn] season {year}: cache hit, skipped"`
  - [x] 2.6: Implement `sync_all(force_refresh: bool = False) -> list[SyncResult]`:
    - Call `sync_kaggle(force_refresh)` first, then `sync_espn(force_refresh)`
    - Return both results

- [x] Task 3: Create `sync.py` CLI at project root (AC: 1–5)
  - [x] 3.1: Create `sync.py` at project root as a thin Typer wrapper
  - [x] 3.2: Define Typer `app` with one command taking:
    - `source: str` — `typer.Option("all", help="Source to sync: kaggle | espn | all")`
    - `dest: Path` — `typer.Option(Path("data/"), help="Local data directory")`
    - `force_refresh: bool` — `typer.Option(False, "--force-refresh", help="Bypass cache and re-fetch all data")`
  - [x] 3.3: Instantiate `ParquetRepository(base_path=dest)` and `SyncEngine(repo, dest)`
  - [x] 3.4: Route to `engine.sync_kaggle()`, `engine.sync_espn()`, or `engine.sync_all()` based on `source`
  - [x] 3.5: Catch `ConnectorError` and `RuntimeError`; print user-friendly messages and `raise typer.Exit(code=1)`
  - [x] 3.6: Print summary on completion (total records written, cache hits, elapsed time)

- [x] Task 4: Update module exports (AC: 1)
  - [x] 4.1: Add `SyncEngine` and `SyncResult` to `src/ncaa_eval/ingest/__init__.py` exports and `__all__`

- [x] Task 5: Integration tests (AC: 6)
  - [x] 5.1: Create `tests/integration/test_sync.py`
  - [x] 5.2: Test Kaggle full cycle — patch `KaggleConnector`, verify `repo.get_teams()` and `repo.get_games(season)` return expected records after sync
  - [x] 5.3: Test Kaggle cache hit — call `sync_kaggle()` once (writes Parquet), call again with `force_refresh=False`, assert `KaggleConnector.fetch_games` NOT called the second time
  - [x] 5.4: Test `--force-refresh` — write Parquet manually, call `sync_kaggle(force_refresh=True)`, assert `KaggleConnector.download` called with `force=True` and fetch methods called
  - [x] 5.5: Test ESPN dependency guard — call `sync_espn()` on empty repository, assert `RuntimeError` raised with message mentioning "kaggle"
  - [x] 5.6: Test `sync_all` order — verify Kaggle is invoked before ESPN via method-level mocking
  - [x] 5.7: Test CLI via `typer.testing.CliRunner` — invoke `sync.py app` with `--source kaggle --dest {tmp_path}`, verify exit code 0 and Parquet files created

- [x] Task 6: Update mutation testing config and run quality pipeline (AC: all)
  - [x] 6.1: Add `src/ncaa_eval/ingest/sync.py` to `paths_to_mutate` in `[tool.mutmut]` (already covered by `src/ncaa_eval/ingest/`)
  - [x] 6.2: `ruff check .` passes
  - [x] 6.3: `mypy --strict src/ncaa_eval tests sync.py` passes (see Task 1.3)
  - [x] 6.4: `pytest` passes with all new tests green (integration tests marked `@pytest.mark.integration`)
  - [x] 6.5: `mutmut run` on ingest module — 538 killed / 239 survived / 1 suspicious (69% score)

## Dev Notes

### Architecture: `sync.py` vs `ncaa_eval.cli`

Story 5.5 plans `python -m ncaa_eval.cli train --model elo ...`. Story 2.4 uses `python sync.py ...` (project-root script). These are **intentionally separate**:
- `sync.py` = data pipeline command (ingestion/warehouse)
- `ncaa_eval.cli` = model training command

Both use Typer. Do NOT collapse them into one CLI in this story — Story 5.5 will create `ncaa_eval.cli` later.

`sync.py` at the project root is a thin Typer wrapper. All business logic lives in `src/ncaa_eval/ingest/sync.py` (the `SyncEngine` class) so it can be imported, tested, and called from notebooks.

### Typer + `from __future__ import annotations` Compatibility

Typer uses `typing.get_type_hints()` at runtime for CLI parameter generation. With PEP 563 lazy annotations (`from __future__ import annotations`), all annotations become strings. Typer 0.15+ handles this correctly via `get_type_hints(fn, include_extras=True)` which evaluates lazy strings.

**However**, `sync.py` at the project root is NOT inside the `src/ncaa_eval/` package. Include `from __future__ import annotations` at the top of `sync.py` (required by Ruff), but be aware that the module's global namespace must contain all referenced types. Use `from pathlib import Path` directly in `sync.py`.

### `SyncEngine` Design

```python
# src/ncaa_eval/ingest/sync.py

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

from ncaa_eval.ingest.connectors.espn import EspnConnector
from ncaa_eval.ingest.connectors.kaggle import KaggleConnector
from ncaa_eval.ingest.repository import Repository

@dataclasses.dataclass
class SyncResult:
    source: str
    teams_written: int = 0
    seasons_written: int = 0
    games_written: int = 0
    seasons_cached: int = 0

class SyncEngine:
    def __init__(self, repository: Repository, data_dir: Path) -> None:
        self._repo = repository
        self._data_dir = data_dir
```

**Keep `SyncEngine.__init__` args to ≤5 (PLR0913 limit).** Current design is fine (2 args).

### Caching Logic Detail

**Kaggle CSV-level caching** (already built into Story 2.3):
- `KaggleConnector.download(force=False)` → if `{data_dir}/kaggle/MTeams.csv` exists, skips API call. Maps directly to `--force-refresh`.

**Parquet-level caching** (Story 2.4 adds this):
- Teams: check `(data_dir / "teams.parquet").exists()`
- Seasons: check `(data_dir / "seasons.parquet").exists()`
- Games per season: check `(data_dir / "games" / f"season={year}" / "data.parquet").exists()`

```python
def sync_kaggle(self, force_refresh: bool = False) -> SyncResult:
    result = SyncResult(source="kaggle")
    connector = KaggleConnector(extract_dir=self._data_dir / "kaggle")
    connector.download(force=force_refresh)  # CSV-level cache

    # Teams: Parquet-level cache
    teams_path = self._data_dir / "teams.parquet"
    if force_refresh or not teams_path.exists():
        teams = connector.fetch_teams()
        self._repo.save_teams(teams)
        result.teams_written = len(teams)
        typer.echo(f"[kaggle] teams: {len(teams)} written")
    else:
        typer.echo("[kaggle] teams: cache hit, skipped")

    # Seasons: Parquet-level cache
    seasons_path = self._data_dir / "seasons.parquet"
    if force_refresh or not seasons_path.exists():
        seasons = connector.fetch_seasons()
        self._repo.save_seasons(seasons)
        result.seasons_written = len(seasons)
    else:
        seasons = self._repo.get_seasons()  # load for game-loop

    # Games: per-season Parquet-level cache
    for season in seasons:
        game_path = self._data_dir / "games" / f"season={season.year}" / "data.parquet"
        if not force_refresh and game_path.exists():
            result.seasons_cached += 1
            typer.echo(f"[kaggle] season {season.year}: cache hit, skipped")
            continue
        games = connector.fetch_games(season.year)
        self._repo.save_games(games)
        result.games_written += len(games)
        typer.echo(f"[kaggle] season {season.year}: {len(games)} games written")

    return result
```

### ESPN Caching Strategy (Game ID Prefix)

ESPN games use `game_id = f"espn_{espn_game_id}"` (from Story 2.3). To check if ESPN has been synced for a season WITHOUT reading all games:
- Check if any `.espn_synced_{year}` marker file exists in `data_dir` (simpler than inspecting Parquet contents)
- Create marker file after successful ESPN sync for that season
- On `force_refresh=True`, delete markers before syncing

Marker files avoid parsing Parquet just to check presence of ESPN records:
```
{data_dir}/.espn_synced_2025  ← empty file, created after ESPN sync for 2025
```

```python
def _espn_marker(self, year: int) -> Path:
    return self._data_dir / f".espn_synced_{year}"
```

### ESPN Season Scope

ESPN connector is designed for CURRENT/RECENT season data only. `cbbpy.get_games_season()` takes ~10s per season and may fail on older seasons. **Default: sync only the most recent season** (max year from `repo.get_seasons()`).

Do NOT attempt to sync all historical seasons from ESPN — it would take >6 hours. The Architecture justifies this: "ESPN connector is best suited for current/recent seasons only" [Source: `2-3-implement-data-source-connectors.md#EspnConnector Implementation Details`].

### ESPN Dependency on Kaggle Data

`EspnConnector` requires two mappings from Kaggle data:
1. `team_name_to_id: dict[str, int]` — built from `repo.get_teams()`
2. `season_day_zeros: dict[int, datetime.date]` — built from `repo.get_seasons()`

These MUST be populated (Kaggle synced) before calling `sync_espn()`. If `repo.get_teams()` returns empty list → raise `RuntimeError`:

```python
def sync_espn(self, force_refresh: bool = False) -> SyncResult:
    teams = self._repo.get_teams()
    seasons = self._repo.get_seasons()
    if not teams or not seasons:
        raise RuntimeError(
            "ESPN sync requires Kaggle data to be synced first. "
            "Run: python sync.py --source kaggle --dest <path>"
        )
    team_name_to_id = {t.team_name: t.team_id for t in teams}
    season_day_zeros = {s.year: s.day_zero for s in seasons}  # see Season model note below
    ...
```

**⚠️ Season model check:** `Season` currently has only `year: int`. ESPN needs `day_zero: datetime.date` from `MSeasons.csv`. Check `src/ncaa_eval/ingest/schema.py` — if `Season` has no `day_zero` field, `KaggleConnector` builds the day_zero mapping internally and it's not persisted to Parquet. In that case, load day_zeros by re-running `KaggleConnector(extract_dir=...).fetch_seasons()` from the already-downloaded CSVs (no network call), OR refactor to persist `day_zero` in `Season`. Check the actual schema before implementing and use the simpler approach.

**Investigation needed:** Open `src/ncaa_eval/ingest/schema.py` at the start of implementation to confirm `Season`'s fields. If `day_zero` is missing, the simplest fix is to load it from the existing CSV files (already downloaded) rather than modifying the schema mid-story.

### CLI `sync.py` Skeleton

```python
# sync.py (project root)
from __future__ import annotations

from pathlib import Path

import typer

from ncaa_eval.ingest import ParquetRepository, SyncEngine
from ncaa_eval.ingest.connectors import ConnectorError

app = typer.Typer(help="NCAA_eval data sync command")

VALID_SOURCES = ("kaggle", "espn", "all")

@app.command()
def main(
    source: str = typer.Option("all", help="Source to sync: kaggle | espn | all"),
    dest: Path = typer.Option(Path("data/"), help="Local data directory path"),
    force_refresh: bool = typer.Option(
        False, "--force-refresh", help="Bypass cache and re-fetch all data"
    ),
) -> None:
    """Fetch NCAA data from external sources and persist to local store."""
    if source not in VALID_SOURCES:
        typer.echo(f"Error: --source must be one of: {', '.join(VALID_SOURCES)}", err=True)
        raise typer.Exit(code=1)

    dest.mkdir(parents=True, exist_ok=True)
    repo = ParquetRepository(base_path=dest)
    engine = SyncEngine(repository=repo, data_dir=dest)

    try:
        if source == "kaggle":
            engine.sync_kaggle(force_refresh=force_refresh)
        elif source == "espn":
            engine.sync_espn(force_refresh=force_refresh)
        else:
            engine.sync_all(force_refresh=force_refresh)
    except ConnectorError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
```

### mypy Configuration for Root `sync.py`

Add `sync.py` to mypy's `files` list in `pyproject.toml`:

```toml
[tool.mypy]
strict = true
files = ["src/ncaa_eval", "tests", "sync.py"]
plugins = ["pydantic.mypy"]
```

Also update the Nox `typecheck` session if it hard-codes paths:
```python
session.run("mypy", "--strict", "src/ncaa_eval", "tests", "noxfile.py", "sync.py")
```

### Integration Test Patterns

Use `typer.testing.CliRunner` for CLI tests (clean, no subprocess):
```python
from typer.testing import CliRunner
from sync import app  # import the Typer app from root sync.py

runner = CliRunner()

def test_cli_sync_kaggle(tmp_path: Path, ...) -> None:
    result = runner.invoke(app, ["--source", "kaggle", "--dest", str(tmp_path)])
    assert result.exit_code == 0
```

**Mocking `KaggleConnector`** in integration tests:
```python
from unittest.mock import MagicMock, patch

@patch("ncaa_eval.ingest.sync.KaggleConnector")
def test_sync_kaggle_full(MockConnector: MagicMock, tmp_path: Path) -> None:
    instance = MockConnector.return_value
    instance.fetch_teams.return_value = [Team(team_id=1, team_name="Duke")]
    instance.fetch_seasons.return_value = [Season(year=2025)]
    instance.fetch_games.return_value = [...]  # fixture games
    ...
```

Integration tests go in `tests/integration/` and should be marked `@pytest.mark.integration` to exclude from pre-commit smoke runs.

### `pyarrow` Parquet Path Discovery

`ParquetRepository` uses `{base_path}/games/season={year}/data.parquet` for games. The cache check path in `SyncEngine` must match this exactly. Verify against `src/ncaa_eval/ingest/repository.py` before implementing (path is documented as `games/season={year}/data.parquet` in Story 2.2).

### What NOT to Do

- **Do not** add ESPN-derived games to Kaggle's Parquet partition — ESPN games have their own `game_id` prefix (`espn_`). The `ParquetRepository.save_games()` likely overwrites the partition file. You MUST verify whether `save_games()` overwrites or appends. If it overwrites, storing Kaggle and ESPN games together requires loading existing Kaggle games first, merging, then saving. Check `repository.py` implementation. If it appends (unlikely), simpler. **Resolve this before writing any ESPN save logic.**

- **Do not** run ESPN for all historical seasons. ESPN is current-season only. Sync max 1-2 seasons.

- **Do not** add a `--seasons` flag in this story (not in AC). Keep it simple.

- **Do not** use `iterrows()` for DataFrame loops (Style Guide Section 6.2 mandate). The `SyncEngine` doesn't iterate DataFrames, so this is N/A — but if you add any DataFrame processing in sync.py, use vectorized ops.

### Previous Story Intelligence

**From Story 2.3 (Connectors) — Critical context:**
- `KaggleConnector(extract_dir: Path, competition: str = "march-machine-learning-mania-2025")` — constructor takes `extract_dir`, not `data_dir`. Pass `data_dir / "kaggle"` as `extract_dir`.
- `KaggleConnector.download(force: bool = False)` — handles CSV-level caching. `force=True` re-downloads even if CSVs exist.
- `EspnConnector(team_name_to_id: dict[str, int], season_day_zeros: dict[int, datetime.date])` — requires BOTH mappings. Neither has a default.
- `game_id` format for Kaggle: `f"{season}_{day_num}_{w_team_id}_{l_team_id}"`
- `game_id` format for ESPN: `f"espn_{espn_game_id}"`
- `ConnectorError` hierarchy: `AuthenticationError`, `DataFormatError`, `NetworkError` — all importable from `ncaa_eval.ingest.connectors`
- `fetch_teams()` and `fetch_seasons()` are NON-abstract (raise `NotImplementedError` by default). ESPN raises `NotImplementedError` for both. Only call `fetch_teams()` and `fetch_seasons()` on `KaggleConnector`.

**From Story 2.2 (Repository) — Critical context:**
- `ParquetRepository(base_path: Path)` — constructor takes one arg
- `repo.save_games(games: list[Game])` — check if this OVERWRITES or APPENDS to existing partition. If it overwrites, merging Kaggle + ESPN games in same partition requires loading existing, merging, and saving the combined list.
- `repo.get_seasons()` returns `list[Season]` where `Season` has only `year: int` — no `day_zero` field.
- `repo.get_teams()` returns `list[Team]` where `Team` has `team_id`, `team_name`, `canonical_name`.
- Parquet path: `{base_path}/games/season={year}/data.parquet` (hive partitioning)

**From Epic 1 (Toolchain):**
- `from __future__ import annotations` required in ALL Python files
- `mypy --strict` mandatory — Typer + mypy strict is known to work in Python 3.12
- Google-style docstrings (single backticks for inline code, NOT RST double-backticks)
- PLR0913: max 5 function arguments (use config objects if needed)
- PT022: use `return` (not `yield`) in fixtures without teardown
- `dict[str, Any]` for Pydantic test fixtures (not `dict[str, object]`)

### Git Intelligence

Recent relevant commits:
- `63a1ac9` feat(ingest): implement Kaggle and ESPN data source connectors (Story 2.3) — files created in `src/ncaa_eval/ingest/connectors/`
- `7a15683` Define internal data schema & Parquet repository layer (Story 2.2) — `repository.py`, `schema.py`
- Commit pattern for this story: `feat(ingest): implement sync CLI and smart caching (Story 2.4)`
- Branch: `story/2-4-implement-sync-cli-smart-caching`

### Project Structure Notes

**Alignment with Architecture (Section 9):**
- `sync.py` at project root: not in Architecture spec but consistent with `python sync.py` AC from epics.md
- `src/ncaa_eval/ingest/sync.py`: fits within `ingest/` module per Architecture Section 5.1 ("Interfaces: `sync(source)`, `update_cache()`")
- `tests/integration/test_sync.py`: correct location per testing strategy (`tests/integration/` for I/O or external deps)

**New files to create:**
- `sync.py` (project root) — Typer CLI entry point
- `src/ncaa_eval/ingest/sync.py` — `SyncEngine` class + `SyncResult` dataclass

**Files to modify:**
- `src/ncaa_eval/ingest/__init__.py` — Add `SyncEngine`, `SyncResult` exports
- `pyproject.toml` — Add `typer[all]>=0.15` dep, update mypy `files`, update mutmut `paths_to_mutate`
- `noxfile.py` — Add `sync.py` to mypy session if paths are hard-coded
- `poetry.lock` — Updated after `poetry add`

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Epic 2, Story 2.4]
- [Source: specs/05-architecture-fullstack.md#Section 5.1 — Ingestion Engine (`sync(source)`, `update_cache()`)]
- [Source: specs/05-architecture-fullstack.md#Section 8.2 — Data Access Layer (Parquet + Repository)]
- [Source: specs/05-architecture-fullstack.md#Section 9 — Unified Project Structure]
- [Source: specs/05-architecture-fullstack.md#Section 12 — Coding Standards]
- [Source: specs/03-prd.md#FR2 (Persistent Local Store), FR3 (Smart Caching)]
- [Source: _bmad-output/implementation-artifacts/2-3-implement-data-source-connectors.md — Connector constructors, game_id formats, ESPN dependency on Kaggle]
- [Source: _bmad-output/implementation-artifacts/2-2-define-internal-data-schema-repository-layer.md — ParquetRepository API, Parquet path structure]
- [Source: _bmad-output/planning-artifacts/template-requirements.md — Library-First (Typer for CLI), PLR0913, PT022, dict[str, Any] for fixtures, caret-pin new deps]

## Dev Agent Record

### Agent Model Used

Claude Sonnet 4.6

### Debug Log References

- Season model has no `day_zero` field → loaded via `KaggleConnector._load_day_zeros()` from cached CSVs (no network call).
- `save_games()` OVERWRITES per-season partition → `sync_espn` loads existing Kaggle games, merges with ESPN games, then saves the combined list.
- ESPN caching uses `.espn_synced_{year}` marker files to avoid parsing Parquet to detect ESPN records.
- CLI tests (`test_cli_*`) marked `@pytest.mark.no_mutation` because they `import sync` (project-root module) by module name, which fails in mutmut's `mutants/` CWD environment.
- Installed typer 0.24.0 (`>=0.15` constraint satisfied); installed via `conda run pip install` in addition to `poetry add` due to conda env isolation.
- Added `sync.py` to noxfile.py typecheck session (hard-coded path list).

### Completion Notes List

- Implemented `SyncResult` dataclass and `SyncEngine` class in `src/ncaa_eval/ingest/sync.py`.
- `sync_kaggle`: three-tier Parquet cache (teams, seasons, per-season games) with CSV-level pre-cache via `KaggleConnector.download(force=...)`.
- `sync_espn`: marker-file caching (`.espn_synced_{year}`); merges ESPN games with existing Kaggle games before saving to avoid overwrite data loss; syncs most-recent season only.
- `sync_all`: calls kaggle then espn in order, returns `list[SyncResult]`.
- `sync.py` (project root): thin Typer CLI wrapper; catches `ConnectorError` and `RuntimeError`; prints elapsed-time summary.
- 8 integration tests: full cycle, cache hit, force-refresh, ESPN guard, order, CLI happy-path, CLI invalid source, CLI ESPN guard.
- Quality pipeline: ruff ✅, mypy --strict ✅ (30 source files, 0 issues), pytest 148/148 ✅, mutmut 538 killed / 239 survived (69%).

### File List

**New files:**
- `src/ncaa_eval/ingest/sync.py` — SyncResult dataclass and SyncEngine class
- `sync.py` — Typer CLI entry point at project root
- `tests/integration/test_sync.py` — 11 integration tests

**Modified files:**
- `src/ncaa_eval/ingest/__init__.py` — Added SyncEngine, SyncResult to exports and `__all__`
- `src/ncaa_eval/ingest/connectors/kaggle.py` — Renamed `_load_day_zeros` → `load_day_zeros` (public API); added `fetch_team_spellings()`; added zipfile auto-extraction for kaggle 2.0; fixed date format to `%m/%d/%Y`; updated auth error message
- `src/ncaa_eval/ingest/connectors/espn.py` — Switched `fuzz.ratio` → `fuzz.token_set_ratio`; removed `get_games_season` fallback; fixed docstring RST markup
- `tests/fixtures/kaggle/MSeasons.csv` — Updated DayZero format to `%m/%d/%Y` (matches real Kaggle data)
- `README.md` — Updated Kaggle API setup instructions for new access_token flow
- `pyproject.toml` — Added `typer[all]>=0.15,<2` dependency; added `sync.py` to mypy `files`; added `sync.py` to check-manifest ignore list
- `poetry.lock` — Updated with typer 0.24.0 and transitive deps
- `noxfile.py` — Added `sync.py` to typecheck session
- `_bmad-output/implementation-artifacts/sprint-status.yaml` — Status updated to done

### Senior Developer Review (AI)

**Reviewer:** Claude Sonnet 4.6 (BMAD code-review workflow) — 2026-02-19

**Outcome:** Approved with fixes applied (12 issues auto-fixed across 2 review passes, 151 tests pass)

**Round 1 Issues Fixed:**
- [H1][HIGH] Renamed `KaggleConnector._load_day_zeros()` → `load_day_zeros()` — private method was accessed cross-class with `# noqa: SLF001` suppression; promoted to public API [`src/ncaa_eval/ingest/connectors/kaggle.py`]
- [M1][MEDIUM] Added 3 ESPN integration tests (full cycle, cache hit, force-refresh) — caching marker logic and Kaggle+ESPN merge were completely untested [`tests/integration/test_sync.py`]
- [M2][MEDIUM] Added missing assertions to `test_sync_kaggle_cache_hit` — `fetch_teams`, `fetch_seasons` call counts and `teams_written`/`seasons_written == 0` not previously verified [`tests/integration/test_sync.py`]
- [M3][MEDIUM] Added `sync.py` to `[tool.check-manifest]` ignore list — root script was git-tracked but absent from distribution and ignore list [`pyproject.toml`]
- [M4][MEDIUM] Pinned typer to `>=0.15,<2` — unbounded `>=0.15` violates project convention of bounding major-version breaks [`pyproject.toml`]
- [L1][LOW] Fixed RST `:class:` markup in module docstring — replaced with Google-style single backticks [`src/ncaa_eval/ingest/sync.py`]

**Round 2 Issues Fixed (post-review changes: zip extraction, spellings, fuzzy matching, README):**
- [C1][CRITICAL] Fixed 11 failing unit tests — `load_day_zeros()` date format changed to `%m/%d/%Y` (real Kaggle 2.0 format) but test fixture still used ISO format; updated `tests/fixtures/kaggle/MSeasons.csv` to match real data format [`tests/fixtures/kaggle/MSeasons.csv`]
- [M5][MEDIUM] Vectorized `fetch_team_spellings()` — replaced `iterrows()` loop with `dict(zip(...))` per Style Guide §6.2 mandate [`src/ncaa_eval/ingest/connectors/kaggle.py`]
- [M6][MEDIUM] Vectorized `_build_espn_team_map()` — replaced `iterrows()` with `.tolist()` extraction per Style Guide §6.2 [`src/ncaa_eval/ingest/sync.py`]
- [M7][MEDIUM] Fixed auth error message — `~/.kaggle/kaggle.json` contradicted README's new `~/.kaggle/access_token` instructions; updated error to reference README [`src/ncaa_eval/ingest/connectors/kaggle.py`]
- [M8][MEDIUM] Mocked `_build_espn_team_map` and `fetch_team_spellings` in ESPN tests — tests were implicitly calling real cbbpy internal CSV; now properly isolated [`tests/integration/test_sync.py`]
- [L2][LOW] Fixed stale `fetch_games()` docstring — still said "Attempts get_games_season() first" after that code path was removed [`src/ncaa_eval/ingest/connectors/espn.py`]
- [L3][LOW] Fixed RST double backticks (``) → Google-style single backticks in `_fetch_schedule_df`, `_build_espn_team_map`, and `EspnConnector` module docstrings [`src/ncaa_eval/ingest/connectors/espn.py`, `src/ncaa_eval/ingest/sync.py`]

### Change Log

- 2026-02-19: Second code review — 7 issues fixed (1 CRITICAL, 4 MEDIUM, 2 LOW); 11 unit tests restored; ESPN tests properly isolated from cbbpy internals; test count remains 151
- 2026-02-19: Code review complete — 6 issues fixed (1 HIGH, 4 MEDIUM, 1 LOW); 3 ESPN integration tests added; test count 148 → 151
- 2026-02-19: Implemented Sync CLI & Smart Caching (Story 2.4) — SyncEngine with Parquet-level caching, ESPN marker-file caching, Typer CLI wrapper, 8 integration tests. All quality checks pass.
