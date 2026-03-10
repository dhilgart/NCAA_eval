# Story 9.9: CLI `predict` Command

Status: done

## Story

As a **data scientist**,
I want to **run `ncaa-eval predict <run-id>` from the command line to generate win-probability predictions for current-season matchups**,
so that **I can get predictions without launching the dashboard or running a notebook**.

## Acceptance Criteria

1. **Given** a saved model run (identified by `<run-id>`), **when** the user runs `ncaa-eval predict --run-id <run-id>`, **then** the model is loaded from the run artifact directory via `RunStore.load_model(run_id)`.

2. **Given** a loaded stateful (Elo) model, **when** `predict` is invoked with `--season <year>`, **then** win probabilities are computed for all pairwise team matchups in that season using `EloProvider` + `build_probability_matrix()` (same pattern as `export.py`).

3. **Given** a loaded stateless model (XGBoost, LogisticRegression), **when** `predict` is invoked with `--season <year>`, **then** a feature server is constructed from the model's `feature_config`, features are served for the season, and `model.predict_proba()` generates game-level predictions for all regular+tournament games in the dataset.

4. **Given** `--output <path>` is provided, **when** predictions are generated, **then** output is written to the specified file path. **Given** `--output` is omitted, **then** output is written to stdout (pipe-safe, no Rich formatting on stdout).

5. **Given** any invocation, **then** output is CSV with columns: `season`, `team_a_id`, `team_b_id`, `pred_win_prob`. For stateful models producing a pairwise matrix, rows cover all C(n,2) team pairs. For stateless models, rows cover actual games in the dataset.

6. **Given** `--run-id` references a nonexistent run or a run with no saved model, **then** the CLI prints a clear error message and exits with code 1.

7. **Given** the command is implemented, **then** it is documented in `docs/user-guide.md` under the CLI reference section.

## Tasks / Subtasks

- [x] Task 1: Create `src/ncaa_eval/cli/predict.py` orchestration module (AC: #1, #2, #3, #5)
  - [x] 1.1: Implement `run_predict()` function following the same thin-orchestration pattern as `export.py:run_export()`
  - [x] 1.2: Implement stateful model path — load model, collect team IDs from season games via `ParquetRepository`, build pairwise probability matrix via `EloProvider` + `build_probability_matrix()`, format as CSV
  - [x] 1.3: Implement stateless model path — load model, load `feature_names` from `RunStore.load_feature_names()`, set up `StatefulFeatureServer` from `model.feature_config`, serve features for the season, call `model.predict_proba()`, format as CSV
  - [x] 1.4: Implement `format_predictions_csv()` helper that produces the `season,team_a_id,team_b_id,pred_win_prob` CSV string

- [x] Task 2: Register `predict` command in `src/ncaa_eval/cli/main.py` (AC: #1, #4, #6)
  - [x] 2.1: Add `predict` command to the Typer `app` with options: `--run-id` (required), `--season` (required), `--data-dir` (default `data/`), `--output` (optional Path)
  - [x] 2.2: Wire error handling for `FileNotFoundError` / `TypeError` → exit code 1 (same pattern as `export` command)

- [x] Task 3: Write unit tests in `tests/unit/test_cli_predict.py` (AC: #1–#6)
  - [x] 3.1: Test stateful model predict writes valid CSV to file
  - [x] 3.2: Test stateful model predict writes CSV to stdout when no `--output`
  - [x] 3.3: Test stateless model predict writes valid CSV with game-level rows
  - [x] 3.4: Test nonexistent run-id exits with error code 1
  - [x] 3.5: Test missing season data exits with error

- [x] Task 4: Update documentation (AC: #7)
  - [x] 4.1: Add `predict` command to `docs/user-guide.md` CLI reference section

## Dev Notes

### Architecture Pattern — Follow `export.py` Exactly

The `predict` command follows the **identical thin-CLI-wrapper pattern** established by Story 9.1:

```
main.py (Typer command definition + error handling)
  └── predict.py (pure orchestration — no CLI concerns)
        ├── run_predict() — top-level entry called by main.py
        └── build_predictions() — pure function returning CSV string
```

Key: `build_predictions()` should be a pure function with no I/O side-effects — callers decide where to write. `run_predict()` handles Rich console output and file/stdout routing.

### Stateful vs. Stateless Model Paths

**Stateful (Elo):**
```python
store = RunStore(base_path=data_dir)
model = store.load_model(run_id)  # Returns Model | None
# isinstance(model, StatefulModel) → True
repo = ParquetRepository(base_path=data_dir)
games = repo.get_games(season)
team_ids = sorted({g.w_team_id for g in games} | {g.l_team_id for g in games})
provider = EloProvider(model)
context = MatchupContext(season=season, day_num=154, is_neutral=True)
prob_matrix = build_probability_matrix(provider, team_ids, context)
# Format matrix into CSV rows: all C(n,2) pairs
```

**Stateless (XGBoost, LogisticRegression):**
```python
store = RunStore(base_path=data_dir)
model = store.load_model(run_id)
feat_names = store.load_feature_names(run_id)  # list[str] | None
# isinstance(model, StatefulModel) → False
# Need feature server to build feature matrix
from ncaa_eval.cli.train import _setup_feature_server
server = _setup_feature_server(data_dir, model.feature_config)
# Serve features for the season, get DataFrame
# model.predict_proba(X[feat_names]) → pd.Series of probabilities
```

**Critical:** The stateless path requires importing `_setup_feature_server` from `train.py` (or extracting it to a shared module). Since this is a private function, consider whether to:
- (a) Import it directly (acceptable — both `train.py` and `predict.py` are in the same `cli` package), OR
- (b) Move it to a shared location if it would simplify things

Option (a) is preferred to minimize scope.

### Existing Imports to Reuse

```python
from ncaa_eval.evaluation.bracket import MatchupContext
from ncaa_eval.evaluation.providers import EloProvider, build_probability_matrix
from ncaa_eval.ingest import ParquetRepository
from ncaa_eval.model.base import StatefulModel
from ncaa_eval.model.tracking import RunStore
```

### CSV Output Format

```csv
season,team_a_id,team_b_id,pred_win_prob
2025,1101,1102,0.6234
2025,1101,1103,0.4512
2025,1102,1103,0.5891
```

For the pairwise matrix (stateful models), `team_a_id < team_b_id` always (same convention as Kaggle export). For game-level predictions (stateless models), `team_a_id` and `team_b_id` follow the game record ordering.

### Feature Serving for Stateless Models

`_setup_feature_server` in `train.py` (line ~73) creates the feature pipeline:
```python
def _setup_feature_server(data_dir: Path, feature_config: FeatureConfig) -> StatefulFeatureServer:
    repo = ParquetRepository(base_path=data_dir)
    data_server = ChronologicalDataServer(repo)
    return StatefulFeatureServer(config=feature_config, data_server=data_server)
```

After building the server, the training pipeline calls `server.serve(start_year, end_year)` to get a combined DataFrame with feature columns and metadata. The predict command needs the same flow but for a single season.

### Test Pattern — Follow `test_cli_export.py`

Use `typer.testing.CliRunner` with `@patch` on `ncaa_eval.cli.predict.RunStore` and `ncaa_eval.cli.predict.ParquetRepository`. Mock `store.load_model()` to return either a `MagicMock(spec=StatefulModel)` or a plain `MagicMock()` for stateless models.

### Key File Locations

| File | Action |
|------|--------|
| `src/ncaa_eval/cli/predict.py` | **NEW** — prediction orchestration |
| `src/ncaa_eval/cli/main.py` | **MODIFY** — add `predict` command |
| `tests/unit/test_cli_predict.py` | **NEW** — unit tests |
| `docs/user-guide.md` | **MODIFY** — CLI reference section |

### Project Structure Notes

- New files go in existing `src/ncaa_eval/cli/` and `tests/unit/` directories — no new directories needed
- Follow the `export.py` / `test_cli_export.py` naming convention exactly
- `from __future__ import annotations` required in all new Python files
- All code must pass `mypy --strict` and `ruff check`

### Previous Story Intelligence (9.8)

- Quality gates: 1092 tests passing, ruff clean, mypy --strict clean (99 files)
- Code review identified 5 issues in 9.8 — watch for: missing state invalidation, over-broad logic in helper functions, duplicate helper functions
- Pattern: always separate pure logic (in `lib/` or orchestration module) from UI/CLI concerns

### Git Intelligence

Recent commits all follow `feat(scope): description (Story X.Y)` pattern. Stories 9.1–9.8 are done, all merged to main. The `export` command (Story 9.1) is the most directly relevant precedent — follow its patterns exactly.

### References

- [Source: `src/ncaa_eval/cli/main.py` — Typer app and command registration pattern]
- [Source: `src/ncaa_eval/cli/export.py` — thin orchestration pattern, `build_kaggle_submission()` + `run_export()`]
- [Source: `src/ncaa_eval/model/tracking.py` — `RunStore`, `ModelRun`, `Prediction` classes]
- [Source: `src/ncaa_eval/model/base.py` — `Model` ABC, `StatefulModel` template, `predict_proba()`, `predict_matchup()`]
- [Source: `src/ncaa_eval/evaluation/providers.py` — `EloProvider`, `build_probability_matrix()`]
- [Source: `src/ncaa_eval/cli/train.py` — `_setup_feature_server()`, `_generate_tournament_predictions()`]
- [Source: `tests/unit/test_cli_export.py` — test pattern with `CliRunner` + mocked `RunStore`/`ParquetRepository`]
- [Source: `_bmad-output/planning-artifacts/epics.md#Story 9.9` — acceptance criteria]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

- Fixed ruff import sorting (I001) after moving `_setup_feature_server` import to module level
- Fixed mypy `[attr-defined]` errors by changing `model: object` → `model: Model` in `_build_stateless_predictions()`
- Fixed ruff format spacing in tuple append expression
- Added `type: ignore[import-untyped]` for pandas import in test file

### Completion Notes List

- Implemented `predict.py` with `build_predictions()` (pure) and `run_predict()` (CLI wrapper), supporting both stateful (Elo pairwise matrix) and stateless (game-level) model paths
- Registered `predict` command in `main.py` with `--run-id`, `--season`, `--data-dir`, `--output` options; error handling mirrors `export` command
- 5 unit tests covering: stateful CSV to file, stateful CSV to stdout, stateless CSV, nonexistent run-id error, missing season data error
- Added `predict` command to `docs/user-guide.md` Getting Started section with option table and output format example
- All 1101 tests pass (1097 unit + 4 integration), ruff clean, mypy --strict clean (101 files)

### Change Log

- 2026-03-10: Implemented CLI `predict` command (Story 9.9)
- 2026-03-10: Code review fixes applied (5 issues — 3 High, 2 Medium) — Round 1
- 2026-03-10: Code review fixes applied (5 issues — 1 High, 2 Medium, 1 Low + 1 test added) — Round 2

### Code Review Fixes Applied

**Round 1 (Dev agent):**

**H1 — Pipe-safety bug:** `run_predict()` status message contaminated stdout in pipe mode. Fixed by `Console(stderr=output is None)` so progress messages route to stderr when writing CSV to stdout.

**H2 — Dead code:** `_build_stateless_predictions()` contained `is_stateful = isinstance(model, StatefulModel)` which is always `False` (caller guarantees stateless context). Removed the check; `mode="batch"` used directly. Also removed now-unused `from typing import Literal` import.

**H3 — False docstring claim:** `build_predictions()` docstring claimed "pure orchestration, no I/O side-effects" but the function instantiates `RunStore` and `ParquetRepository` (filesystem I/O). Updated docstring to accurately describe the function as an "orchestration layer."

**M1 — Unused mock:** `test_stateless_predict_writes_csv` patched `ParquetRepository` unnecessarily (stateless path doesn't use it directly). Removed the `@patch` decorator and `mock_repo_cls` parameter.

**M2 — Weak test assertions:** `test_stateful_predict_writes_csv_to_stdout` used substring checks. Upgraded to proper CSV parsing with row count and column ordering assertions (compatible with CliRunner's stream-mixing behavior).

**Round 2 (Code review agent — 2026-03-10):**

**H1 (Second reviewer) — Pipe-safety regression in CLI path:** The Round 1 fix in `predict.py` was negated by `main.py` passing the module-level `console = Console()` (stdout) to `run_predict()`. Since the `console or Console(stderr=output is None)` short-circuits on non-None console, the stderr redirect never activated. Fixed by constructing `Console(stderr=True)` in `main.py`'s `predict` command when `output is None`.

**M1 (Second reviewer) — AttributeError uncaught:** `model.feature_config` (stateless path) raises `AttributeError` if a plugin subclass forgets to set it. Added `AttributeError` to the `except` tuple in `main.py`'s predict command.

**M2 (Second reviewer) — Fragile mock pattern:** `test_stateless_predict_writes_csv` used `mock_model.__class__ = type("FakeStatelessModel", (), {})` to make `isinstance(mock, StatefulModel)` return False. Replaced with `MagicMock(spec=LogisticRegressionModel)`.

**M3 (Second reviewer) — Missing pipe-safety test:** Added `test_stdout_contains_only_csv_when_no_output_arg` that calls `run_predict()` directly (bypassing CliRunner stream merging) to verify `sys.stdout` contains only CSV lines.

**L2 (Second reviewer) — Docstring gap:** Added `TypeError` and `AttributeError` to `run_predict()` docstring `Raises` section.

### File List

- `src/ncaa_eval/cli/predict.py` — **NEW** — prediction orchestration module
- `src/ncaa_eval/cli/main.py` — **MODIFIED** — added `predict` Typer command + code review fixes (H1, M1)
- `tests/unit/test_cli_predict.py` — **NEW** — 6 unit tests for predict CLI (+ 1 added by code review)
- `docs/user-guide.md` — **MODIFIED** — added predict command to CLI reference
- `_bmad-output/planning-artifacts/template-requirements.md` — **MODIFIED** — added pipe-safety and dead-code-from-branching learnings
- `_bmad-output/implementation-artifacts/9-9-cli-predict-command.md` — **MODIFIED** — story status/tasks
- `_bmad-output/implementation-artifacts/sprint-status.yaml` — **MODIFIED** — story status update
