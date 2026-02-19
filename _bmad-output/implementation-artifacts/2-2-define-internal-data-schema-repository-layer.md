# Story 2.2: Define Internal Data Schema & Repository Layer

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a data scientist,
I want a unified internal data schema (Team, Game, Season entities) with a Repository pattern abstracting storage,
So that all downstream code works against a consistent API regardless of storage backend.

## Acceptance Criteria

1. **Given** the Architecture specifies Team, Game, and Season entities, **When** the developer imports the data layer, **Then** Team, Game, and Season are defined as typed data structures (Pydantic models or dataclasses).
2. **And** Team includes: `TeamID` (int), `Name` (str), `CanonicalName` (str).
3. **And** Game includes: `GameID`, `Season`, `Date`, `WTeamID`, `LTeamID`, `WScore`, `LScore`, `Loc`.
4. **And** Season includes: `Year` (int).
5. **And** a Repository interface abstracts read/write operations (`get_games(season)`, `get_teams()`, `save_games(games)`).
6. **And** at least one concrete Repository implementation exists (SQLite or Parquet -- decision finalized here).
7. **And** the repository is covered by unit tests validating round-trip read/write.

## Tasks / Subtasks

- [x] Task 1: Define Pydantic v2 schema models (AC: 1, 2, 3, 4)
  - [x] 1.1: Create `src/ncaa_eval/ingest/schema.py` with `Team`, `Game`, `Season` Pydantic models
  - [x] 1.2: Add field validation constraints (e.g., `season >= 1985`, score >= 0, `loc` in {"H", "A", "N"})
  - [x] 1.3: Add `model_config` for JSON serialization aliases matching architecture naming
  - [x] 1.4: Verify all models pass `mypy --strict`
- [x] Task 2: Define Repository abstract interface (AC: 5)
  - [x] 2.1: Create `src/ncaa_eval/ingest/repository.py` with `Repository` ABC
  - [x] 2.2: Define methods: `get_teams()`, `get_games(season)`, `get_seasons()`, `save_teams(teams)`, `save_games(games)`, `save_seasons(seasons)`
  - [x] 2.3: Type all method signatures for `mypy --strict`
- [x] Task 3: Implement Parquet Repository (AC: 6)
  - [x] 3.1: Implement `ParquetRepository` extending `Repository` ABC
  - [x] 3.2: Store Team data as `{base_path}/teams.parquet`
  - [x] 3.3: Store Game data partitioned by season as `{base_path}/games/season={year}/data.parquet`
  - [x] 3.4: Store Season data as `{base_path}/seasons.parquet`
  - [x] 3.5: Use `pyarrow` engine explicitly for all Parquet read/write
  - [x] 3.6: Handle schema evolution: new nullable columns must not break reads of older files
- [x] Task 4: Wire up module exports (AC: 1)
  - [x] 4.1: Update `src/ncaa_eval/ingest/__init__.py` to export schema models and repository classes
- [x] Task 5: Unit tests for schema models (AC: 1, 2, 3, 4)
  - [x] 5.1: Test valid construction of Team, Game, Season
  - [x] 5.2: Test field validation rejects invalid data (negative scores, invalid loc values, etc.)
  - [x] 5.3: Test serialization round-trip (model -> dict -> model)
- [x] Task 6: Unit tests for Parquet Repository (AC: 7)
  - [x] 6.1: Test round-trip: save_teams -> get_teams
  - [x] 6.2: Test round-trip: save_games -> get_games (single season)
  - [x] 6.3: Test round-trip: save_games -> get_games (multiple seasons, verify partition isolation)
  - [x] 6.4: Test get_games for nonexistent season returns empty list
  - [x] 6.5: Test save_games appends/overwrites correctly
  - [x] 6.6: Test repository creates directories automatically
- [x] Task 7: Run full quality pipeline (AC: all)
  - [x] 7.1: `ruff check .` passes
  - [x] 7.2: `mypy --strict src/ncaa_eval tests` passes
  - [x] 7.3: `pytest` passes with all new tests green
  - [x] 7.4: `mutmut run` on new modules (update `[tool.mutmut]` paths_to_mutate)

## Dev Notes

### Storage Decision: Parquet (Finalized)

Architecture Section 8.2 specifies:
- **Primary Store: Parquet** for immutable game data and large prediction sets (performance)
- **Metadata Store: SQLite** for tracking Model Runs and Configurations (Story 5.5 scope)

**Decision for Story 2.2:** Implement `ParquetRepository` as the concrete Repository. SQLite is deferred to Story 5.5 (ModelRun/Prediction metadata tracking). The Repository ABC enables adding a SQLite implementation later without changing any business logic.

### Data Modeling: Use Pydantic v2 Models

Architecture Section 12 mandates: "All data structures passed between Logic and UI must use Pydantic models or TypedDicts."

**Use Pydantic v2 `BaseModel`**, not dataclasses or TypedDicts. Rationale:
- Pydantic is already an indirect dependency (via pandera)
- Built-in validation, serialization, JSON Schema generation
- `model_dump()` converts to dict for pandas DataFrame construction
- Fully typed, passes `mypy --strict` (with pydantic.mypy plugin)
- Preferred over TypedDicts because validation catches bad data at ingestion time

**mypy configuration required:** Add the `pydantic.mypy` plugin to `pyproject.toml`:
```toml
[tool.mypy]
plugins = ["pydantic.mypy"]

[tool.pydantic-mypy]
init_typed = true
init_forbid_extra = true
warn_required_dynamic_aliases = true
```
[Source: specs/05-architecture-fullstack.md#Section 12]

### Schema Design Guidance

**Core entity fields** (from AC):
- **Team:** `team_id` (int), `team_name` (str), `canonical_name` (str — default empty until Story 4.3)
- **Game:** `game_id` (str), `season` (int), `day_num` (int), `date` (date | None), `w_team_id` (int), `l_team_id` (int), `w_score` (int), `l_score` (int), `loc` (str: H/A/N), `num_ot` (int, default 0), `is_tournament` (bool, default False)
- **Season:** `year` (int)

**Field naming:** Use `snake_case` for all Pydantic fields (project convention). The AC uses PascalCase names (`TeamID`, `WTeamID`) as abstract identifiers — map these to `team_id`, `w_team_id`, etc.

**`game_id` format:** Use a composite key that ensures uniqueness. Kaggle data uses `{season}_{day_num}_{w_team_id}_{l_team_id}` as a natural key. Alternatively, use a simple integer if Kaggle provides one. The dev agent should decide based on what works best across data sources.

**`date` field:** Store as `datetime.date | None`. Kaggle provides `DayNum` (int offset from season DayZero) not calendar dates. The connector (Story 2.3) will convert DayNum to calendar date using `MSeasons.csv` DayZero. Other sources (BartTorvik, ESPN) provide actual dates. The `day_num` field preserves Kaggle's native format for lossless round-trip.

**Schema evolution:** Include only AC-required fields plus `day_num`, `num_ot`, and `is_tournament` as essential extensions. Detailed box score fields (FGM, FGA, etc.) can be added as optional fields in a later story or stored in a separate `DetailedGame` model. Do NOT add BartTorvik/KenPom efficiency fields here — those belong in the transform layer (Epic 4).

### Parquet Implementation Guidance

**Engine:** Use `engine="pyarrow"` explicitly in all `pd.read_parquet()` and `df.to_parquet()` calls. pyarrow is the pandas default and provides the best performance, schema enforcement, and predicate pushdown support.

**Partitioning:** Game data should be partitioned by `season`. This matches the dominant access pattern (`get_games(season)`) and enables Parquet predicate pushdown to skip irrelevant partitions.

**Directory structure:**
```
data/
  teams.parquet
  seasons.parquet
  games/
    season=2023/
      data.parquet
    season=2024/
      data.parquet
```

**Schema evolution safety:** When reading Parquet files, handle the case where older files lack columns that were added later. The `pyarrow.dataset` API fills missing columns with nulls automatically. Alternatively, validate column presence and add missing columns as null after read.

**Repository constructor:** Accept a `base_path: Path` parameter pointing to the data directory. Default to `data/` relative to project root (already `.gitignore`d).

### Pydantic <-> DataFrame Conversion Pattern

The dev agent will need to convert between Pydantic model lists and DataFrames frequently:

```python
# Models -> DataFrame
df = pd.DataFrame([g.model_dump() for g in games])

# DataFrame -> Models
games = [Game(**row) for row in df.to_dict(orient="records")]
```

Consider adding convenience class methods on the models (e.g., `Game.from_dataframe(df) -> list[Game]`) or a utility function. Keep it simple — do not over-engineer.

### Dependency Notes

**pydantic** is already available as a transitive dependency of pandera but should be declared as a direct dependency in `pyproject.toml` since this story uses it directly. Add `pydantic = "*"` to `[tool.poetry.dependencies]`.

**pyarrow** is required for Parquet operations. pandas can use pyarrow if installed, but it's not currently listed in `pyproject.toml`. Add `pyarrow = "*"` to `[tool.poetry.dependencies]`.

Run `POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval poetry add pydantic pyarrow` to add both.

### Review Follow-up from Story 2.1 (Informational)

Story 2.1 flagged that the **product owner has not yet approved the MVP source selection** (Kaggle, BartTorvik, sportsdataverse, Warren Nolan). This does NOT block Story 2.2 because:
- The core schema fields (Team, Game, Season) are universal across ALL sources
- Source-specific fields (efficiency metrics, play-by-play) belong in the transform layer (Epic 4), not the base schema
- The schema is extensible — new optional fields can be added to the Pydantic models without breaking existing code

The dev agent should design the schema around Kaggle MMLM fields (confirmed primary source) and ensure extensibility for additional sources.
[Source: _bmad-output/implementation-artifacts/2-1-evaluate-data-sources.md#Review Follow-ups]

### Project Structure Notes

**New files to create:**
- `src/ncaa_eval/ingest/schema.py` — Pydantic models for Team, Game, Season
- `src/ncaa_eval/ingest/repository.py` — Repository ABC + ParquetRepository
- `tests/unit/test_schema.py` — Schema model tests
- `tests/unit/test_repository.py` — Repository round-trip tests

**Files to modify:**
- `src/ncaa_eval/ingest/__init__.py` — Export schema models and repository classes
- `pyproject.toml` — Add pydantic, pyarrow dependencies; add pydantic.mypy plugin config; update mutmut paths_to_mutate

**Alignment with project structure:** All new files go under `src/ncaa_eval/ingest/` per Architecture Section 9. Tests go under `tests/unit/` per project convention.

### Testing Standards

- **Unit tests only** for this story — no external I/O beyond tmp_path
- Use `tmp_path` pytest fixture for Parquet file operations (auto-cleanup)
- Use `@pytest.mark.smoke` on core round-trip tests for pre-commit inclusion
- Consider `@pytest.mark.property` Hypothesis tests for schema validation (generate random valid/invalid Game data)
- Pandera integration: Consider adding a `GameSchema = pa.DataFrameSchema(...)` for validating Game DataFrames in the repository layer. This follows the pattern from Story 1.8. [Source: _bmad-output/planning-artifacts/template-requirements.md#Pandera-specific notes]

### Previous Story Intelligence

**From Story 2.1 (Spike — Evaluate Data Sources):**
- Kaggle MMLM is the confirmed primary source with fields: Season, DayNum, WTeamID, LTeamID, WScore, LScore, WLoc, NumOT (compact) + box score stats (detailed, 2003+)
- MasseyOrdinals provides 100+ ranking systems (including KenPom as "POM")
- Team name/ID mapping across sources is a known challenge — `canonical_name` field lays groundwork for Story 4.3
- `data/` directory is already `.gitignore`d
- cbbpy includes `rapidfuzz` (useful for future team name matching)

**From Epic 1 (Toolchain):**
- `from __future__ import annotations` required in ALL Python files (Ruff FA100)
- `mypy --strict` mandatory on `src/ncaa_eval` and `tests`
- Google-style docstrings (Ruff pydocstyle convention)
- Pandera for DataFrame validation — `import pandera.pandas as pa` (NOT top-level `import pandera as pa`)
- `import pandas as pd  # type: ignore[import-untyped]` required for pandas imports
- Structured logging available via `ncaa_eval.utils.logger`
- Ruff PT022: use `return` (not `yield`) in fixtures with no teardown
- Marker-based mutmut exclusion via `@pytest.mark.no_mutation`
[Source: _bmad-output/planning-artifacts/template-requirements.md]

### Git Intelligence

Recent commits follow conventional commit format. Pattern for this story:
- Branch: `story/2-2-define-internal-data-schema-repository-layer`
- Commits: `feat(ingest): ...` for schema and repository implementation
- Story branch created from `main` at `840feaa`

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Epic 2, Story 2.2]
- [Source: specs/05-architecture-fullstack.md#Section 4.1 — Core Entities]
- [Source: specs/05-architecture-fullstack.md#Section 8.2 — Data Access Layer]
- [Source: specs/05-architecture-fullstack.md#Section 12 — Coding Standards]
- [Source: specs/05-architecture-fullstack.md#Section 2.5 — Repository Pattern]
- [Source: specs/03-prd.md#FR1, FR2]
- [Source: specs/research/data-source-evaluation.md — Kaggle MMLM Data Coverage]
- [Source: _bmad-output/implementation-artifacts/2-1-evaluate-data-sources.md — Previous Story Intelligence]
- [Source: _bmad-output/planning-artifacts/template-requirements.md — Pydantic/Pandera/Testing patterns]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

- pyarrow imports require `# type: ignore[import-untyped]` (no py.typed marker)
- Pydantic mypy plugin enforces strict init types — test fixtures use `dict[str, Any]` not `dict[str, object]`
- mutmut `mutants/` dir doesn't have `ncaa_eval.utils` — added `--ignore=tests/unit/test_logger.py` to mutmut config

### Completion Notes List

- Implemented Team, Game, Season as Pydantic v2 BaseModel classes with field validation and JSON serialization aliases
- Game uses `Literal["H", "A", "N"]` for location, composite string game_id, and optional date/num_ot/is_tournament fields
- Game model includes cross-field `@model_validator` enforcing `w_score > l_score` and `w_team_id != l_team_id`
- Repository ABC defines 6 abstract methods: get_teams, get_games, get_seasons, save_teams, save_games, save_seasons
- ParquetRepository uses pyarrow for reads/writes with explicit schemas; games partitioned by season (hive-style)
- Uses `pyarrow.dataset` API for game reads with predicate pushdown on season filter
- Schema evolution handled in `get_games`: null-fills `num_ot` and `is_tournament` after pyarrow schema unification across mixed-version partitions
- `save_teams([])` and `save_seasons([])` are no-ops (consistent with `save_games([])`)
- 30 schema tests + 19 repository tests = 49 new tests, all passing (73 total)
- Mutation testing: 134/151 killed (88.7%); 17 surviving mutants are all equivalent (library default behaviors)
- Added pydantic, pyarrow as direct dependencies; configured pydantic.mypy plugin

### Change Log

- 2026-02-19: Implemented Story 2.2 — Pydantic v2 schema models (Team, Game, Season), Repository ABC, ParquetRepository, 44 unit tests, full quality pipeline passing
- 2026-02-19: Code review (AI) — 1 HIGH + 4 MEDIUM issues found and fixed: (H1) added schema evolution test + `get_games` null-fill for pyarrow schema unification; (M2) `w_score > l_score` model validator; (M3) `w_team_id != l_team_id` model validator; (M4) empty-list guard in `save_teams`/`save_seasons`; (M1) added template-requirements.md to File List. LOW: smoke marker on Season round-trip, `dict[str, Any]` in test helpers. 73 total tests passing.

### File List

- `src/ncaa_eval/ingest/schema.py` (new) — Pydantic v2 models for Team, Game, Season
- `src/ncaa_eval/ingest/repository.py` (new) — Repository ABC + ParquetRepository implementation
- `src/ncaa_eval/ingest/__init__.py` (modified) — Added exports for schema and repository classes
- `tests/unit/test_schema.py` (new) — 27 unit tests for schema models
- `tests/unit/test_repository.py` (new) — 17 unit tests for ParquetRepository round-trips
- `pyproject.toml` (modified) — Added pydantic/pyarrow deps, pydantic.mypy plugin, updated mutmut paths
- `_bmad-output/implementation-artifacts/sprint-status.yaml` (modified) — Story status updated
- `_bmad-output/planning-artifacts/template-requirements.md` (modified) — Added Pydantic/pyarrow/mutmut learnings from Story 2.2
- `poetry.lock` (modified) — Lock file updated for new dependencies
