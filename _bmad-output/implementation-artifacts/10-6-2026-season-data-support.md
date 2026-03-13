# Story 10.6: 2026 Season Data Support

Status: ready-for-dev

## Story

As a **data scientist**,
I want **the pipeline to correctly sync, deduplicate, and serve 2026 NCAA season data end-to-end**,
so that **I can train models and generate bracket predictions for the 2026 tournament**.

## Acceptance Criteria

1. **Kaggle competition slug updated** — `KaggleConnector` default `competition` parameter is updated to `"march-machine-learning-mania-2026"` and `ncaa-eval sync` successfully downloads 2026 CSV files.

2. **Massey ordinals constant updated** — `_MASSEY_LAST_SEASON` is updated to `2026` in `src/ncaa_eval/transform/normalization.py`; all docstrings referencing "2003–2025" are updated to "2003–2026".

3. **End-to-end pipeline validated** — `ncaa-eval sync`, feature serving (`DataServer`), and model training all execute without error for a year range that includes 2026 (e.g., `--start-year 2015 --end-year 2026`).

4. **ESPN deduplication verified for 2026** — If 2026 exhibits the same Kaggle+ESPN game-duplication pattern as 2025 (same game stored under both a Kaggle ID and an ESPN ID), confirm `_deduplicate_espn_overlap()` correctly collapses duplicates and ESPN records are preferred. Spot-check: game count for 2026 after dedup should match expected regular-season game counts.

5. **graph.py deduplication comment generalized** — The comment at `src/ncaa_eval/transform/graph.py` lines 14–15 that explicitly calls out "2025 season stores 4,545 games twice" is updated to be season-agnostic (e.g., mention that callers are responsible for deduplicating any season with ESPN+Kaggle overlap).

6. **Dashboard and CLI example text updated** — The `--end-year 2025` example in `dashboard/pages/home.py` line 29 is updated to `--end-year 2026`; similarly update any CLI help-text examples in `src/ncaa_eval/cli/` that hardcode 2025 as the end year.

## Tasks / Subtasks

- [ ] Task 1: Update Kaggle competition slug (AC: #1)
  - [ ] 1.1: In `src/ncaa_eval/ingest/connectors/kaggle.py` line 104, change the default `competition` parameter from `"march-machine-learning-mania-2025"` to `"march-machine-learning-mania-2026"`
  - [ ] 1.2: Verify the Kaggle competition is live before running sync (`kaggle competitions list` or check https://www.kaggle.com/competitions/march-machine-learning-mania-2026)
  - [ ] 1.3: Run `ncaa-eval sync --kaggle` and confirm 2026 CSV files download successfully

- [ ] Task 2: Update Massey ordinals season bound (AC: #2)
  - [ ] 2.1: In `src/ncaa_eval/transform/normalization.py` line 36, change `_MASSEY_LAST_SEASON: int = 2025` to `_MASSEY_LAST_SEASON: int = 2026`
  - [ ] 2.2: Update all docstrings in `normalization.py` that reference "2003–2025" to "2003–2026" (lines ~79, 357, 381)

- [ ] Task 3: Validate end-to-end pipeline for 2026 (AC: #3)
  - [ ] 3.1: Run `ncaa-eval sync` (both Kaggle and ESPN) and confirm no errors for 2026
  - [ ] 3.2: Confirm cbbpy's `mens_team_map.csv` includes season 2026 — if not, the ESPN connector will log a warning and fall back to latest available year; document this if it occurs
  - [ ] 3.3: Run feature serving through `DataServer` for a range including 2026 (e.g., `--start-year 2015 --end-year 2026`) — confirm no index errors, missing data exceptions, or assertion failures
  - [ ] 3.4: Run `ncaa-eval train --model elo --start-year 2015 --end-year 2026` to confirm model training succeeds with 2026 data

- [ ] Task 4: Verify ESPN deduplication for 2026 (AC: #4)
  - [ ] 4.1: After sync, query the local data store for 2026 games and count records before and after deduplication
  - [ ] 4.2: If duplication exists (same `(w_team_id, l_team_id, day_num)` found with both Kaggle and ESPN game IDs), confirm `_deduplicate_espn_overlap()` in `src/ncaa_eval/transform/serving.py` handles it correctly and ESPN records are preferred
  - [ ] 4.3: Add a brief comment in `_deduplicate_espn_overlap()` noting it has been validated for 2026 (or update existing comments if they reference only 2025)

- [ ] Task 5: Generalize graph.py deduplication comment (AC: #5)
  - [ ] 5.1: Update `src/ncaa_eval/transform/graph.py` lines 14–15: replace the 2025-specific comment ("2025 season stores 4,545 games twice") with a season-agnostic statement such as: "Caller is responsible for deduplicating games for any season with ESPN+Kaggle overlap before calling graph functions (e.g., 2025 stores ~4,545 games twice; check for similar patterns in subsequent seasons)"

- [ ] Task 6: Update dashboard and CLI example text (AC: #6)
  - [ ] 6.1: `dashboard/pages/home.py` line 29: change `--end-year 2025` to `--end-year 2026`
  - [ ] 6.2: Audit `src/ncaa_eval/cli/` files (`main.py`, `export.py`, `predict.py`, `train.py`) for hardcoded `2025` in help text or docstring examples — update to `2026`
  - [ ] 6.3: Note: the `end_year: int = typer.Option(2025, ...)` default in `cli/main.py` line 47 is intentional (users explicitly opt-in to including the current season) — do NOT change the default value, only update example text

## Dev Notes

### Key Files to Touch

| File | Line(s) | Change Type |
|------|---------|-------------|
| `src/ncaa_eval/ingest/connectors/kaggle.py` | 104 | Code — competition slug |
| `src/ncaa_eval/transform/normalization.py` | 36, ~79, 357, 381 | Code + docs — Massey constant |
| `src/ncaa_eval/transform/graph.py` | 14–15 | Comment only |
| `src/ncaa_eval/transform/serving.py` | ~102–128 | Validation + optional comment |
| `dashboard/pages/home.py` | 29 | Example text only |
| `src/ncaa_eval/cli/main.py` | ~47 | Docstring/help example only (NOT the default value) |
| `src/ncaa_eval/cli/export.py` | ~31, 87 | Docstring examples only |
| `src/ncaa_eval/cli/predict.py` | ~166, 220 | Docstring examples only |

### What Does NOT Need to Change

The codebase was designed well for forward-compatibility. These are confirmed NOT to need changes:

- **Season discovery** — `KaggleConnector.load_day_zeros()` reads `MSeasons.csv` dynamically; 2026 is auto-discovered once Kaggle includes it
- **ESPN scope** — `sync.py` line 213 uses `max(s.year for s in seasons)` — automatically targets 2026
- **Deduplication logic** — `_deduplicate_espn_overlap()` in `serving.py` is generic; no year-specific code
- **Bracket structure** — 64-team, 6-round structure hardcoded in `bracket.py` matches 2026 NCAA tournament format (no expansion announced)
- **No-tournament seasons** — `NO_TOURNAMENT_SEASONS = frozenset({2020})` in `serving.py` line 28; do NOT add 2026 unless cancelled
- **Walk-forward splitter** — season-agnostic; no changes needed
- **Test fixtures** — tests using `end_year=2025` remain valid; do NOT mass-update to 2026 (these are test data, not constraints)
- **`cli/main.py` default `end_year=2025`** — intentional; users must opt-in to 2026

### 2025 Data Deduplication Precedent

Per memory and prior EDA findings: In 2025, both Kaggle and ESPN sources stored the same 4,545 games, resulting in duplicate rows. The fix was implemented in `_deduplicate_espn_overlap()` using `drop_duplicates(subset=["w_team_id", "l_team_id", "day_num"], keep="last")` with ESPN records sorted last (preferred). Validate 2026 follows the same pattern.

If 2026 does NOT have the duplicate pattern (e.g., Kaggle releases 2026 data before ESPN has it), deduplication still runs but has no effect — this is safe.

### cbbpy Team Map Watch

If cbbpy hasn't been updated for 2026 yet, `_build_espn_team_map()` in `sync.py` will log:
```
espn: season 2026 not in cbbpy team map; using 2025
```
This is a graceful fallback — ESPN sync will use the 2025 team map. Document any divergence between 2025 and 2026 team IDs if it occurs. Monitor cbbpy releases: `conda run -n ncaa_eval pip install --upgrade cbbpy`.

### External Dependency Check (Before Starting Task 1)

The 2026 Kaggle competition may not be live yet. Check before updating the slug:
```bash
conda run -n ncaa_eval kaggle competitions list | grep march-machine-learning
```
If the 2026 competition is not yet available, the slug update and sync tasks should be deferred — but all other tasks (Massey constant, comment updates, dashboard text) can proceed independently.

### mypy / ruff Impact

- `_MASSEY_LAST_SEASON` change: type is `int`, no mypy concern
- All other changes are comments, strings, and example text — no type checking impact
- Run `ruff check .` and `mypy --strict src/ncaa_eval tests` before committing

### Project Structure Notes

- Ingest connectors: `src/ncaa_eval/ingest/connectors/`
- Normalization: `src/ncaa_eval/transform/normalization.py`
- Serving/deduplication: `src/ncaa_eval/transform/serving.py`
- Graph features: `src/ncaa_eval/transform/graph.py`
- CLI: `src/ncaa_eval/cli/`
- Dashboard pages: `dashboard/pages/`

### References

- [Source: src/ncaa_eval/ingest/connectors/kaggle.py#L104]
- [Source: src/ncaa_eval/transform/normalization.py#L36]
- [Source: src/ncaa_eval/transform/normalization.py#L79,357,381]
- [Source: src/ncaa_eval/transform/graph.py#L14-15]
- [Source: src/ncaa_eval/transform/serving.py#L28,102-128]
- [Source: src/ncaa_eval/ingest/sync.py#L36-65,213]
- [Source: dashboard/pages/home.py#L29]
- [Source: src/ncaa_eval/cli/main.py#L47]

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
