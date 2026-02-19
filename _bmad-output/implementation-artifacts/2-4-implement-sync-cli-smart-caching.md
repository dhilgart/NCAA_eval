# Story 2.4: Implement Sync CLI & Smart Caching

Status: ready-for-dev

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

- [ ] Task 1: Add CLI and progress dependencies (AC: 1–6)
  - [ ] 1.1: Add `typer[all]` as a production dependency (provides CLI + rich for progress)
  - [ ] 1.2: Run `POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval poetry add "typer[all]>=0.15"`
  - [ ] 1.3: Add `sync.py` (root) to mypy `files` list in `pyproject.toml` to enforce strict type-checking

- [ ] Task 2: Implement `SyncEngine` class (AC: 1–5)
  - [ ] 2.1: Create `src/ncaa_eval/ingest/sync.py` with `SyncResult` dataclass and `SyncEngine` class
  - [ ] 2.2: `SyncResult` fields: `source: str`, `teams_written: int`, `seasons_written: int`, `games_written: int`, `seasons_cached: int`
  - [ ] 2.3: `SyncEngine.__init__(self, repository: Repository, data_dir: Path)` — stores repo reference and data root
  - [ ] 2.4: Implement `sync_kaggle(force_refresh: bool = False) -> SyncResult`:
    - Create `KaggleConnector(extract_dir=self._data_dir / "kaggle")`
    - Call `connector.download(force=force_refresh)` (CSV-level caching already built-in)
    - Cache check teams: if `{data_dir}/teams.parquet` exists and not `force_refresh` → skip `fetch_teams()` + save
    - Cache check seasons: if `{data_dir}/seasons.parquet` exists and not `force_refresh` → load seasons from repo instead of fetch
    - Cache check games per-season: if `{data_dir}/games/season={year}/data.parquet` exists and not `force_refresh` → record as cache hit, skip `fetch_games(year)`
    - Print progress: `"[kaggle] teams: {n}"`, `"[kaggle] season {year}: {n} games written"`, `"[kaggle] season {year}: cache hit, skipped"`
  - [ ] 2.5: Implement `sync_espn(force_refresh: bool = False) -> SyncResult`:
    - Load `team_name_to_id` and `season_day_zeros` from repository (Kaggle data MUST be synced first)
    - Raise `RuntimeError` with helpful message if teams or seasons are empty (Kaggle not synced)
    - ESPN scope: sync only the most recent available season (max season year from `repo.get_seasons()`)
    - Cache check per season: if `{data_dir}/games/season={year}/data.parquet` already contains ESPN-prefixed game IDs → skip (see Dev Notes for game_id prefix strategy)
    - Create `EspnConnector(team_name_to_id=..., season_day_zeros=...)`
    - Call `connector.fetch_games(year)`, save via repository
    - Print progress: `"[espn] season {year}: {n} games written"` or `"[espn] season {year}: cache hit, skipped"`
  - [ ] 2.6: Implement `sync_all(force_refresh: bool = False) -> list[SyncResult]`:
    - Call `sync_kaggle(force_refresh)` first, then `sync_espn(force_refresh)`
    - Return both results

- [ ] Task 3: Create `sync.py` CLI at project root (AC: 1–5)
  - [ ] 3.1: Create `sync.py` at project root as a thin Typer wrapper
  - [ ] 3.2: Define Typer `app` with one command taking:
    - `source: str` — `typer.Option("all", help="Source to sync: kaggle | espn | all")`
    - `dest: Path` — `typer.Option(Path("data/"), help="Local data directory")`
    - `force_refresh: bool` — `typer.Option(False, "--force-refresh", help="Bypass cache and re-fetch all data")`
  - [ ] 3.3: Instantiate `ParquetRepository(base_path=dest)` and `SyncEngine(repo, dest)`
  - [ ] 3.4: Route to `engine.sync_kaggle()`, `engine.sync_espn()`, or `engine.sync_all()` based on `source`
  - [ ] 3.5: Catch `ConnectorError` subclasses; print user-friendly messages and `raise typer.Exit(code=1)`
  - [ ] 3.6: Print summary on completion (total records written, cache hits, elapsed time)

- [ ] Task 4: Update module exports (AC: 1)
  - [ ] 4.1: Add `SyncEngine` and `SyncResult` to `src/ncaa_eval/ingest/__init__.py` exports and `__all__`

- [ ] Task 5: Integration tests (AC: 6)
  - [ ] 5.1: Create `tests/integration/test_sync.py`
  - [ ] 5.2: Test Kaggle full cycle — patch `KaggleConnector` with `unittest.mock.patch`, inject fixture data from `tests/fixtures/kaggle/`, verify `repo.get_teams()` and `repo.get_games(season)` return expected records after sync
  - [ ] 5.3: Test Kaggle cache hit — call `sync_kaggle()` once (writes Parquet), call again with `force_refresh=False`, assert `KaggleConnector.fetch_games` NOT called the second time (use `patch` + call count)
  - [ ] 5.4: Test `--force-refresh` — write Parquet manually, call `sync_kaggle(force_refresh=True)`, assert `KaggleConnector.download` called with `force=True` and fetch methods called
  - [ ] 5.5: Test ESPN dependency guard — call `sync_espn()` on empty repository, assert `RuntimeError` raised with message mentioning "kaggle"
  - [ ] 5.6: Test `sync_all` order — mock both connectors, verify Kaggle is invoked before ESPN
  - [ ] 5.7: Test CLI via `typer.testing.CliRunner` — invoke `sync.py app` with `--source kaggle --dest {tmp_path}`, verify exit code 0 and Parquet files created

- [ ] Task 6: Update mutation testing config and run quality pipeline (AC: all)
  - [ ] 6.1: Add `src/ncaa_eval/ingest/sync.py` to `paths_to_mutate` in `[tool.mutmut]` (already covered by `src/ncaa_eval/ingest/`)
  - [ ] 6.2: `ruff check .` passes
  - [ ] 6.3: `mypy --strict src/ncaa_eval tests sync.py` passes (see Task 1.3)
  - [ ] 6.4: `pytest` passes with all new tests green (integration tests may need `@pytest.mark.integration`)
  - [ ] 6.5: `mutmut run` on ingest module (new sync.py logic)

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

### Completion Notes List

### File List
