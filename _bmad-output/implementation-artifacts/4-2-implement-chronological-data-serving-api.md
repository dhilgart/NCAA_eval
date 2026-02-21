# Story 4.2: Implement Chronological Data Serving API

Status: review

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a data scientist,
I want a `get_chronological_season(year)` API that streams game data in strict date order with temporal boundary enforcement,
so that I can train models with walk-forward validation without risk of data leakage.

## Acceptance Criteria

1. **Given** the Repository layer (Epic 2) contains populated game data, **When** the developer calls `get_chronological_season(2023)`, **Then** games are returned strictly ordered by date within the season.
2. **And** the API makes it impossible to access games beyond a specified cutoff date.
3. **And** requesting data for a future date raises a clear error.
4. **And** the 2020 COVID year returns regular season data but flags the absence of tournament games.
5. **And** the API supports iteration (streaming) for memory-efficient processing of large seasons.
6. **And** temporal boundary enforcement is covered by unit tests including edge cases (season boundaries, same-day games, 2025 deduplication, OT rescaling).

## Tasks / Subtasks

- [x] Task 1: Design and implement `src/ncaa_eval/transform/serving.py` (AC: 1–5)
  - [x] 1.1: Define `SeasonGames` frozen dataclass with fields: `year: int`, `games: list[Game]`, `has_tournament: bool`
  - [x] 1.2: Define `_NO_TOURNAMENT_SEASONS: frozenset[int] = frozenset({2020})` module-level constant
  - [x] 1.3: Define `ChronologicalDataServer` class with `__init__(self, repository: Repository) -> None`
  - [x] 1.4: Implement `get_chronological_season(year, cutoff_date=None) -> SeasonGames` — see Dev Notes for full contract
  - [x] 1.5: Implement `iter_games_by_date(year, cutoff_date=None) -> Iterator[list[Game]]` — yields batches of games grouped by calendar date
  - [x] 1.6: Add `rescale_overtime(score: int, num_ot: int) -> float` module-level function using formula `score × 40.0 / (40 + 5 × num_ot)`

- [x] Task 2: Implement 2025 deduplication helper (AC: 1)
  - [x] 2.1: Add `_deduplicate_2025(games: list[Game]) -> list[Game]` private function (called only when `year == 2025`)
  - [x] 2.2: Deduplicate by canonical triplet `(w_team_id, l_team_id, day_num)` — see Dev Notes for ESPN preference logic
  - [x] 2.3: Prefer ESPN records for `loc` and `num_ot` when both Kaggle and ESPN records exist for the same game
  - [x] 2.4: Do NOT use iterrows or for-loops over DataFrames — vectorize the deduplication logic

- [x] Task 3: Implement temporal boundary enforcement (AC: 2, 3)
  - [x] 3.1: If `cutoff_date` is provided and `cutoff_date > datetime.date.today()`, raise `ValueError("Cannot request future game data: cutoff_date {cutoff_date} is beyond today")`
  - [x] 3.2: Filter games to only those with `game.date <= cutoff_date` when cutoff is provided
  - [x] 3.3: `cutoff_date = None` means "return all games for the season" — no restriction

- [x] Task 4: Implement 2020 COVID year handling (AC: 4)
  - [x] 4.1: `has_tournament = year not in _NO_TOURNAMENT_SEASONS` — naturally follows from the data but must be explicitly signalled in the return type
  - [x] 4.2: For 2020, regular-season games are returned normally; `has_tournament=False` warns downstream consumers not to attempt tournament evaluation
  - [x] 4.3: No special filtering needed — 2020 simply has no `is_tournament=True` games in the repository

- [x] Task 5: Export public API from `src/ncaa_eval/transform/__init__.py` (AC: 1–5)
  - [x] 5.1: Import and re-export `ChronologicalDataServer`, `SeasonGames`, and `rescale_overtime` from `transform/__init__.py`
  - [x] 5.2: Keep `_deduplicate_2025` and `_NO_TOURNAMENT_SEASONS` private (not exported)

- [x] Task 6: Write unit tests in `tests/unit/test_chronological_serving.py` (AC: 6)
  - [x] 6.1: Test: games returned in ascending date order for a multi-game fixture
  - [x] 6.2: Test: `cutoff_date` filters correctly — games after cutoff excluded; games on cutoff date included
  - [x] 6.3: Test: future `cutoff_date` (tomorrow or later) raises `ValueError` with descriptive message
  - [x] 6.4: Test: 2020 year returns `has_tournament=False`; other years return `has_tournament=True`
  - [x] 6.5: Test: `iter_games_by_date` yields correct batches per calendar date, in order
  - [x] 6.6: Test: same-day games (multiple games on the same date) appear in the same batch
  - [x] 6.7: Test: 2025 deduplication reduces duplicate games — fixture with two records for same `(w_team_id, l_team_id, day_num)` → only one survives
  - [x] 6.8: Test: 2025 deduplication prefers ESPN `loc` and `num_ot` over Kaggle values when records differ
  - [x] 6.9: Test: `rescale_overtime(score=75, num_ot=0)` → `75.0` (no change for regulation game)
  - [x] 6.10: Test: `rescale_overtime(score=80, num_ot=1)` → `80 × 40 / 45 ≈ 71.11` (5-OT-minute penalty)
  - [x] 6.11: Test: empty season (no games in repository) → returns `SeasonGames(year=year, games=[], has_tournament=False)`

- [x] Task 7: Commit (AC: all)
  - [x] 7.1: `git add src/ncaa_eval/transform/serving.py src/ncaa_eval/transform/__init__.py tests/unit/test_chronological_serving.py`
  - [x] 7.2: Commit: `feat(transform): implement chronological data serving API (Story 4.2)`
  - [x] 7.3: Update `_bmad-output/implementation-artifacts/sprint-status.yaml`: `4-2-implement-chronological-data-serving-api` → `review`

## Dev Notes

### Story Nature: First Code Story in Epic 4 — transform/ Module

This is the **first code-bearing story in Epic 4**. It introduces the `src/ncaa_eval/transform/` module. No pipeline logic from Stories 4.3–4.7 belongs here — this story delivers the data serving infrastructure only: chronological ordering, temporal boundary enforcement, 2025 deduplication, and OT rescaling.

This story is **a code story** — `mypy --strict`, Ruff, `from __future__ import annotations`, and the no-iterrows mandate all apply. All code in `src/ncaa_eval/transform/` must be type-annotated.

### Module Placement

**New file:** `src/ncaa_eval/transform/serving.py`

Per Architecture Section 9, all feature engineering and data transformation belongs in `src/ncaa_eval/transform/`. The chronological serving layer sits between the ingest layer (which stores `Game` objects via `ParquetRepository`) and the transform/model layers (which consume games sequentially for feature computation).

The existing `src/ncaa_eval/transform/__init__.py` is currently empty except for the module docstring — this story adds the first exports.

### API Design Reference

The `ChronologicalDataServer` class wraps a `Repository` instance:

```python
from __future__ import annotations

import datetime
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator

from ncaa_eval.ingest.repository import Repository
from ncaa_eval.ingest.schema import Game

_NO_TOURNAMENT_SEASONS: frozenset[int] = frozenset({2020})


@dataclass(frozen=True)
class SeasonGames:
    """Result of a chronological season query."""
    year: int
    games: list[Game]
    has_tournament: bool


class ChronologicalDataServer:
    """Serves game data in strict chronological order for walk-forward modeling."""

    def __init__(self, repository: Repository) -> None:
        self._repo = repository

    def get_chronological_season(
        self,
        year: int,
        cutoff_date: datetime.date | None = None,
    ) -> SeasonGames:
        """Return all games for *year* sorted ascending by date.

        Args:
            year: The season year (e.g., 2023 for the 2022-23 season).
            cutoff_date: If provided, only games on or before this date are
                returned. Raises ValueError if cutoff_date is in the future.

        Returns:
            SeasonGames with games sorted by date and has_tournament flag.

        Raises:
            ValueError: If cutoff_date is after today's date.
        """
        ...

    def iter_games_by_date(
        self,
        year: int,
        cutoff_date: datetime.date | None = None,
    ) -> Iterator[list[Game]]:
        """Yield batches of games grouped by calendar date, in chronological order.

        Each yielded list contains all games played on a single calendar date.
        Dates with no games are skipped. Respects the same cutoff_date semantics
        as get_chronological_season().

        Args:
            year: The season year.
            cutoff_date: Optional temporal cutoff.

        Yields:
            Non-empty list[Game] for each calendar date, in ascending date order.
        """
        ...


def rescale_overtime(score: int, num_ot: int) -> float:
    """Rescale a game score to a 40-minute equivalent for OT normalization.

    Args:
        score: Raw final score (not adjusted).
        num_ot: Number of overtime periods played (0 for regulation).

    Returns:
        Score normalized to 40-minute equivalent.

    Example:
        >>> rescale_overtime(80, 1)   # 1 OT: 80 × 40 / 45 ≈ 71.11
        71.11111111111111
        >>> rescale_overtime(75, 0)   # Regulation: no change
        75.0
    """
    return score * 40.0 / (40 + 5 * num_ot)
```

### CRITICAL: 2025 Deduplication

**The 2025 season stores 4,545 games twice** — once with a Kaggle-sourced `game_id` and once with an ESPN-sourced `game_id` for the same physical game. This is a known data quality issue documented in `notebooks/eda/eda_findings_synthesis.md` Section 1.

**Deduplication key:** `(w_team_id, l_team_id, day_num)` — these three fields uniquely identify a game in a season.

**When duplicates exist, prefer the ESPN record** for `loc` and `num_ot`. The Kaggle record tends to have `loc="N"` for all neutral-site games, while ESPN has the more accurate H/A/N distinction. Similarly, `num_ot` from ESPN is preferred.

**Vectorized deduplication approach** (no iterrows — project mandate):

```python
import pandas as pd  # type: ignore[import-untyped]

def _deduplicate_2025(games: list[Game]) -> list[Game]:
    if not games:
        return games
    # Sort so ESPN records (prefixed "espn_") come after Kaggle records.
    # Then use drop_duplicates(keep="last") to keep ESPN over Kaggle.
    df = pd.DataFrame([g.model_dump() for g in games])
    # ESPN game_ids are prefixed "espn_"; Kaggle game_ids are numeric strings.
    df["_is_espn"] = df["game_id"].str.startswith("espn_")
    df = df.sort_values("_is_espn")  # Kaggle first, ESPN last
    df = df.drop_duplicates(subset=["w_team_id", "l_team_id", "day_num"], keep="last")
    df = df.drop(columns=["_is_espn"]).reset_index(drop=True)
    return [Game(**row) for row in df.to_dict(orient="records")]
```

> ⚠️ **Implementation note:** `model_dump()` is the Pydantic v2 method (replaces `dict()`). Verify this works with the existing `Game` model before finalizing.

**Alternative if `model_dump()` causes issues:** Build the dict manually from `Game` fields — see `repository.py` for the pattern used in `save_games()`.

### Temporal Boundary Enforcement Details

The `cutoff_date` parameter enforces a strict "as-of" date:

- `cutoff_date = None` → return all games for the season (no restriction)
- `cutoff_date = date(2023, 3, 1)` → return only games where `game.date <= date(2023, 3, 1)`
- `cutoff_date > datetime.date.today()` → raise `ValueError` with message like: `f"Cannot request future game data: cutoff_date {cutoff_date} is after today ({datetime.date.today()})"`

**All games have dates.** Per MEMORY.md: `KaggleConnector._parse_games_csv()` derives `date = DayZero + timedelta(days=day_num)` for every Kaggle game. ESPN-sourced games have actual API-recorded dates. The `date` field is NEVER `None` for any game in the store. However, the `Game.date` field is typed as `datetime.date | None` (Pydantic schema evolution support) — the serving layer should handle the theoretical `None` case gracefully (log a warning and fall back to `day_num`-based ordering if needed, but this should never occur in practice).

**Sorting:** Sort by `(game.date, game.game_id)` — the secondary sort by `game_id` ensures deterministic ordering for same-day games. Do NOT sort by `day_num` alone (it's not a date in the calendar-day sense for ESPN data).

### 2020 COVID Year — `has_tournament=False`

The 2020 tournament was cancelled due to COVID-19. The Parquet store contains regular-season games for 2020 but zero `is_tournament=True` records. The `has_tournament` flag on `SeasonGames` explicitly signals this to downstream consumers (Story 6.2 walk-forward splitter needs this to skip evaluation for 2020).

The implementation is simple:
```python
has_tournament = year not in _NO_TOURNAMENT_SEASONS
```

No special filtering of returned games is needed — 2020 simply has no tournament games in the repository.

### OT Rescaling — Why and How

When computing rolling averages or season totals, overtime games inflate scoring statistics. A game that ends 80-75 after 1 OT involves 45 minutes of play, not 40. The standard correction (from Edwards 2021, top Kaggle MMLM documented solution):

```
adjusted_score = raw_score × 40 / (40 + 5 × num_ot)
```

Examples:
- Regulation (num_ot=0): `score × 40/40 = score` (no change)
- 1 OT (num_ot=1): `score × 40/45 ≈ 0.889 × score`
- 2 OT (num_ot=2): `score × 40/50 = 0.8 × score`
- 3 OT (num_ot=3): `score × 40/55 ≈ 0.727 × score`

`rescale_overtime()` returns a `float` (not `int`) to preserve precision. Downstream code (Story 4.4 rolling stats) should use these float scores rather than the raw integer scores when computing per-game efficiency. The function applies to individual scores (both `w_score` and `l_score` independently).

### Architecture Guardrails (Mandatory)

From `specs/05-architecture-fullstack.md` Section 12 (Coding Standards):

1. **`from __future__ import annotations` required** — enforced by Ruff; every Python file in `src/` must include this as the first non-comment line
2. **`mypy --strict` mandatory** — all type annotations must be complete; no `Any` unless explicitly justified
3. **Vectorization First** — no `for` loops over Pandas DataFrames for data processing; use vectorized pandas/numpy operations
4. **No direct IO in the transform module** — the serving layer must call `Repository.get_games()`, not read Parquet files directly
5. **Pydantic models or TypedDicts** for all data structures passed between layers — `SeasonGames` dataclass is appropriate for the return type; it does NOT need to be a Pydantic model since it's an internal transform-layer type

**Running quality checks:**
```bash
POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval ruff check .
POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval mypy --strict src/ncaa_eval tests
POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval pytest tests/unit/test_chronological_serving.py -v
```

### Import Pattern for Transform Module

The `Game` and `Repository` types live in `src/ncaa_eval/ingest/`. The transform module imports from ingest:

```python
from ncaa_eval.ingest.repository import Repository
from ncaa_eval.ingest.schema import Game
```

This is the correct import direction (transform → ingest). The dashboard imports from both, but the dashboard is NOT in scope for this story.

### Test File Structure

**File:** `tests/unit/test_chronological_serving.py`

Pattern from `tests/unit/test_repository.py` and `tests/unit/test_schema.py`:

```python
from __future__ import annotations

import datetime
import pytest
from ncaa_eval.transform.serving import (
    ChronologicalDataServer,
    SeasonGames,
    rescale_overtime,
)
# ... fixture definitions using tmp_path + ParquetRepository
```

Use the existing `ParquetRepository` (backed by `tmp_path`) as the repository in tests — do NOT mock the repository. This tests the real integration path. See `tests/unit/test_repository.py` for the fixture pattern.

**Markers to apply:**
- `@pytest.mark.smoke` on fast sorting/filtering tests (< 1s each)
- `@pytest.mark.unit` on all tests in this file
- No `@pytest.mark.slow` needed (no network I/O; pure in-memory with tmp_path)

### pyarrow / pandas Import Annotations

Following the established pattern in `repository.py`:
```python
import pandas as pd  # type: ignore[import-untyped]
```

pyarrow imports are needed if reading directly, but this module should use `Repository.get_games()` exclusively — no raw pyarrow imports needed in `serving.py`.

### What NOT to Do

- **Do not** add feature transformation logic (rolling averages, graph construction, etc.) to this story — that belongs in Stories 4.4–4.6
- **Do not** modify `src/ncaa_eval/ingest/repository.py` or `schema.py` — they are complete and stable
- **Do not** add `serving.py` to the mypy exclude list — it must be type-checked
- **Do not** use `datetime.datetime` — use `datetime.date` consistently (the `Game.date` field is `datetime.date | None`)
- **Do not** apply deduplication to seasons other than 2025 — the 4,545 duplicates are specific to the 2025 Kaggle + ESPN dual-ingestion
- **Do not** place this module in `src/ncaa_eval/ingest/` — the architectural boundary is clear: `ingest/` fetches and stores; `transform/` serves and transforms
- **Do not** raise an error for `year=2020` — 2020 is valid; it just has no tournament games and `has_tournament=False`

### Dependency on 4.1 Research Findings

From `specs/research/feature-engineering-techniques.md` Section 7.3 (Building Blocks Catalog — Story 4.2 scope):

| Building Block | Implementation |
|:---|:---|
| Walk-forward game iterator with date guards | `get_chronological_season()` + `iter_games_by_date()` |
| 2025 deduplication by `(w_team_id, l_team_id, day_num)` | `_deduplicate_2025()` private helper |
| 2020 COVID flag | `SeasonGames.has_tournament` |
| OT rescaling preprocessing | `rescale_overtime()` module-level function |

All four building blocks are in scope for this story.

### git Commit Convention

```bash
feat(transform): implement chronological data serving API (Story 4.2)
```

Branch: currently on `main` per git log (Story 4.1 committed to `main` via PR #17). Create a feature branch per story branching convention before starting:

```bash
git checkout -b story/4-2-implement-chronological-data-serving-api
```

Then commit normally. The code review workflow will merge via PR.

### Project Structure Notes

**New files:**
- `src/ncaa_eval/transform/serving.py` — chronological data server + OT rescaling

**Modified files:**
- `src/ncaa_eval/transform/__init__.py` — add exports
- `tests/unit/test_chronological_serving.py` — new test file (create in `tests/unit/`)
- `_bmad-output/implementation-artifacts/sprint-status.yaml` — update status to `review`
- `_bmad-output/implementation-artifacts/4-2-implement-chronological-data-serving-api.md` — this story file (Dev Agent Record section)

**No changes to:**
- `src/ncaa_eval/ingest/` (stable)
- `pyproject.toml` (no new dependencies — all needed packages are already in the stack: `pandas`, `pydantic`, `pyarrow`)
- Any test in `tests/integration/` (integration tests are for sync/ingest pipeline)

### References

- [Source: specs/05-architecture-fullstack.md#Section 9 — Unified Project Structure (`transform/` module location)]
- [Source: specs/05-architecture-fullstack.md#Section 12 — Coding Standards (mypy --strict, vectorization, no direct IO in UI)]
- [Source: _bmad-output/planning-artifacts/epics.md#Story 4.2 — Acceptance Criteria]
- [Source: _bmad-output/planning-artifacts/epics.md#Epic 4 — Feature overview (FR4: Chronological Serving, NFR4: Leakage Prevention)]
- [Source: specs/research/feature-engineering-techniques.md#Section 7.3 — Building Blocks by Story (Story 4.2 scope)]
- [Source: specs/research/feature-engineering-techniques.md#Section 6.4 — Edwards 2021 OT rescaling formula]
- [Source: src/ncaa_eval/ingest/repository.py — Repository ABC and ParquetRepository (data access patterns)]
- [Source: src/ncaa_eval/ingest/schema.py — Game, Team, Season Pydantic models]
- [Source: notebooks/eda/eda_findings_synthesis.md#Section 1 — 2025 deduplication pattern and ESPN preference logic]
- [Source: _bmad-output/planning-artifacts/template-requirements.md — pyarrow type: ignore pattern, `from __future__ import annotations` requirement, no-iterrows mandate, pytest fixture patterns]

## Dev Agent Record

### Agent Model Used

Claude Sonnet 4.6 (create-story workflow)

### Debug Log References

None.

### Completion Notes List

- Implemented `src/ncaa_eval/transform/serving.py` with `SeasonGames` dataclass, `ChronologicalDataServer`, `rescale_overtime`, `_deduplicate_2025`, and `_effective_date` helper.
- `_deduplicate_2025` uses fully vectorized pandas (sort + drop_duplicates) with ESPN records preferred via `_is_espn` sort column; no iterrows.
- Temporal boundary enforcement raises `ValueError` for future `cutoff_date`; `None` cutoff returns all season games.
- `has_tournament` flag derived from `_NO_TOURNAMENT_SEASONS` constant (not data inference) so downstream consumers get a reliable signal even before all games are ingested.
- `_effective_date` private helper unifies `date` / `day_num` fallback logic used by both sorting and `iter_games_by_date` grouping; logs a warning if `game.date is None` (should never occur in practice).
- `Iterator` imported from `collections.abc` (UP035 compliance).
- 25 unit tests covering all 11 story subtasks; 165 total tests pass with no regressions.
- `mypy --strict` clean on 31 source files; Ruff clean on new files.

### File List

- `src/ncaa_eval/transform/serving.py` (new)
- `src/ncaa_eval/transform/__init__.py` (modified — added public exports)
- `tests/unit/test_chronological_serving.py` (new)
- `_bmad-output/implementation-artifacts/4-2-implement-chronological-data-serving-api.md` (this file)
- `_bmad-output/implementation-artifacts/sprint-status.yaml` (updated status)

## Change Log

| Date | Change | Author |
|:---|:---|:---|
| 2026-02-21 | Created story 4.2 — Implement Chronological Data Serving API | Claude Sonnet 4.6 (create-story) |
| 2026-02-21 | Implemented chronological data serving API — Tasks 1–7, 25 tests, all ACs satisfied | Claude Sonnet 4.6 (dev-story) |
