# Story 5.5: Implement Model Run Tracking & Training CLI

Status: ready-for-dev

## Story

As a data scientist,
I want model run metadata tracked and a CLI for launching training jobs,
So that I can reproduce results, compare runs, and train models from the terminal.

## Acceptance Criteria

1. **CLI entry point** — `python -m ncaa_eval.cli train --model elo --start-year 2015 --end-year 2025` launches a training job from the terminal.
2. **ModelRun record** — A `ModelRun` Pydantic record is created for each training run with: `run_id` (UUID), `model_type` (str), `hyperparameters` (JSON-serializable dict), `timestamp` (datetime), `git_hash` (str), `start_year` (int), `end_year` (int).
3. **Prediction records** — `Prediction` Pydantic records are created for each game prediction with: `run_id`, `game_id`, `season`, `team_a_id`, `team_b_id`, `pred_win_prob` (float).
4. **Persistence** — ModelRun and Prediction records are persisted to the local store (JSON for ModelRun metadata, Parquet for predictions).
5. **Progress display** — Training progress is displayed via Rich progress bars in the terminal (season-by-season progress).
6. **Results summary** — A results summary (model name, seasons trained, number of predictions, run metadata) is printed on completion.
7. **Plugin model support** — The `--model` flag accepts any registered plugin model name (built-in: `"elo"`, `"xgboost"`, `"logistic_regression"`; external user-registered names also work via `get_model(name)`).
8. **Integration tests** — The CLI and tracking are covered by integration tests validating the full train-track-persist cycle.

## Tasks / Subtasks

- [x] Task 1: Define `ModelRun` and `Prediction` data entities (AC: #2, #3)
  - [x] 1.1 Create `src/ncaa_eval/model/tracking.py` with `ModelRun(BaseModel)` and `Prediction(BaseModel)` Pydantic models
  - [x] 1.2 `ModelRun` fields: `run_id: str` (UUID4 string), `model_type: str`, `hyperparameters: dict[str, Any]`, `timestamp: datetime`, `git_hash: str`, `start_year: int`, `end_year: int`, `prediction_count: int`
  - [x] 1.3 `Prediction` fields: `run_id: str`, `game_id: str`, `season: int`, `team_a_id: int`, `team_b_id: int`, `pred_win_prob: float` (constrained to [0.0, 1.0])

- [x] Task 2: Implement `RunStore` persistence layer (AC: #4)
  - [x] 2.1 Create `RunStore` class in `src/ncaa_eval/model/tracking.py` (or separate file if tracking.py grows large)
  - [x] 2.2 `RunStore.__init__(base_path: Path)` — stores runs under `base_path / "runs/"`
  - [x] 2.3 `save_run(run: ModelRun, predictions: list[Prediction])` — writes `run.json` (ModelRun metadata) + `predictions.parquet` (Prediction records via PyArrow) under `base_path / "runs" / run_id /`
  - [x] 2.4 `load_run(run_id: str) -> ModelRun` — reads `run.json`
  - [x] 2.5 `load_predictions(run_id: str) -> pd.DataFrame` — reads `predictions.parquet`
  - [x] 2.6 `list_runs() -> list[ModelRun]` — scans `runs/` directory for all `run.json` files

- [ ] Task 3: Implement training pipeline function (AC: #1, #5, #6, #7)
  - [ ] 3.1 Create `src/ncaa_eval/cli/__init__.py` (make it a package)
  - [ ] 3.2 Create `src/ncaa_eval/cli/train.py` with the `train()` function that orchestrates: model instantiation via `get_model()`, feature serving via `StatefulFeatureServer`, and model training via `model.fit(X, y)`
  - [ ] 3.3 For stateful models (Elo): iterate seasons chronologically, call `model.fit(X, y)` once with all seasons concatenated (StatefulModel.fit() handles per-game iteration internally)
  - [ ] 3.4 For stateless models (XGBoost): build combined feature matrix across all training seasons, call `model.fit(X, y)` once
  - [ ] 3.5 Generate predictions on tournament games for each season within the training range
  - [ ] 3.6 Create ModelRun record with git hash via `subprocess.run(["git", "rev-parse", "--short", "HEAD"])`
  - [ ] 3.7 Display progress via `rich.progress.Progress` (track season-by-season progress)
  - [ ] 3.8 Print results summary via `rich.console.Console` and `rich.table.Table`

- [ ] Task 4: Implement Typer CLI entry point (AC: #1, #7)
  - [ ] 4.1 Create `src/ncaa_eval/cli/main.py` with the Typer app and `train` command
  - [ ] 4.2 Add `src/ncaa_eval/cli/__main__.py` to support `python -m ncaa_eval.cli`
  - [ ] 4.3 CLI options: `--model` (required, str), `--start-year` (int, default 2015), `--end-year` (int, default 2025), `--data-dir` (Path, default "data/"), `--output-dir` (Path, default "data/"), `--config` (optional Path to JSON config override)
  - [ ] 4.4 Validate `--model` against `list_models()` and print available models on error
  - [ ] 4.5 Instantiate model via `get_model(name)` with optional config override from `--config`

- [x] Task 5: Write unit tests for tracking entities (AC: #2, #3, #4)
  - [x] 5.1 Test `ModelRun` creation with all fields
  - [x] 5.2 Test `ModelRun` JSON round-trip serialization
  - [x] 5.3 Test `Prediction` creation and `pred_win_prob` constraint [0.0, 1.0]
  - [x] 5.4 Test `RunStore.save_run()` / `load_run()` round-trip (`tmp_path`)
  - [x] 5.5 Test `RunStore.load_predictions()` returns correct DataFrame
  - [x] 5.6 Test `RunStore.list_runs()` discovers saved runs
  - [x] 5.7 Test `RunStore.load_run()` raises `FileNotFoundError` on missing run

- [ ] Task 6: Write integration tests for CLI (AC: #1, #8)
  - [ ] 6.1 Test CLI `train` command with `"logistic_regression"` model on synthetic data (fastest model for testing)
  - [ ] 6.2 Test CLI produces `run.json` and `predictions.parquet` output files
  - [ ] 6.3 Test CLI with invalid `--model` name prints error with available models
  - [ ] 6.4 Test CLI with `--config` override applies custom hyperparameters

- [ ] Task 7: Run quality gates (AC: all)
  - [ ] 7.1 `ruff check src/ tests/` passes
  - [ ] 7.2 `mypy --strict src/ncaa_eval tests` passes
  - [ ] 7.3 `pytest` passes with all new tests green and zero regressions

## Dev Notes

### Critical: CLI Architecture Decision — `python -m ncaa_eval.cli`

The project uses Typer (0.24.0 installed) with Rich integration. The existing `sync.py` at the repo root is the reference for Typer patterns. Story 5.5 creates a proper CLI package at `src/ncaa_eval/cli/` following the architecture spec's `python -m ncaa_eval.cli train` invocation pattern.

**Do NOT modify `sync.py`** — it remains as the standalone sync CLI. The new training CLI lives in the `ncaa_eval` package.

### Critical: Training Pipeline — Stateful vs Stateless Model Handling

**Stateful models (Elo, StatefulModel subclasses):**
- `StatefulModel.fit(X, y)` already handles per-game sequential iteration internally (template method in `base.py` lines 71–199)
- X must contain metadata columns: `team_a_id`, `team_b_id`, `season`, `day_num`, `date`, `loc_encoding`, `game_id`, `is_tournament`
- `y` is binary: 1 = team_a won, 0 = team_b won
- Seasons must be provided in chronological order — `fit()` calls `start_season()` when season changes
- **All columns (metadata + features) must be present** — `StatefulModel._to_games()` reconstructs Game objects from the DataFrame

**Stateless models (XGBoost, LogisticRegression):**
- `Model.fit(X, y)` expects a standard feature matrix — only numeric feature columns, no metadata
- The CLI must split metadata columns from feature columns before calling `fit()`
- XGBoost handles its own validation split internally (`validation_fraction` config)

**Column separation pattern:**
```python
METADATA_COLS = [
    "game_id", "season", "day_num", "date", "team_a_id", "team_b_id",
    "is_tournament", "loc_encoding", "team_a_won",
]
feature_cols = [c for c in df.columns if c not in METADATA_COLS]

# Stateful: pass full DataFrame (metadata + features)
# Stateless: pass df[feature_cols] only
```

### Critical: Label Convention

`StatefulFeatureServer` produces DataFrames where `team_a = w_team_id` (winner), meaning `team_a_won` is always 1 for regular-season games. For XGBoost training, either:
1. Randomize team assignment in the training data to balance labels
2. Set `scale_pos_weight` appropriately

For this story, document the convention. Label randomization is a training strategy concern — the CLI should at minimum warn if all labels are 1.

### Critical: Predictions — Tournament Games Only

The AC states predictions are for game predictions. For the training CLI's MVP scope:
- Train on all games (regular season + tournament) within the year range
- Generate predictions on **tournament games** within the training range for tracking purposes
- For stateful models: predictions are naturally produced during `fit()` as the model processes each game. Capture tournament predictions during the walk-forward pass.
- For stateless models: after `fit()`, call `predict_proba()` on tournament game rows from the feature matrix.

### Critical: Git Hash Extraction

Use `subprocess.run()` to capture the current git hash:
```python
import subprocess

def _get_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
```

This is safe — `subprocess.run` with a fixed argument list, no shell=True, no user input.

### Critical: Persistence Format

**ModelRun metadata → JSON** (human-readable, small):
```
data/runs/<run_id>/run.json
```

**Predictions → Parquet** (efficient columnar storage):
```
data/runs/<run_id>/predictions.parquet
```

Use PyArrow (^23.0.1 installed) for Parquet writes — consistent with `ParquetRepository` in the ingest layer. Define an explicit PyArrow schema for Predictions:

```python
_PREDICTION_SCHEMA = pa.schema([
    ("run_id", pa.string()),
    ("game_id", pa.string()),
    ("season", pa.int64()),
    ("team_a_id", pa.int64()),
    ("team_b_id", pa.int64()),
    ("pred_win_prob", pa.float64()),
])
```

### Critical: Rich Progress Bars (Not tqdm)

Typer 0.24.0 bundles Rich (14.3.2 installed). Use `rich.progress.Progress` for terminal output — consistent with the Typer ecosystem. Do NOT use tqdm for CLI output (tqdm is a transitive dependency from Kaggle/CBBpy, not a project choice).

```python
from rich.progress import Progress

with Progress() as progress:
    task = progress.add_task("Training...", total=num_seasons)
    for year in range(start_year, end_year + 1):
        # ... process season ...
        progress.advance(task)
```

### Critical: Integration Tests — Use Synthetic Data

Full integration tests requiring real data (Parquet store with 40+ seasons) are too heavy for unit test runs. Instead:
- Create a small fixture with 2–3 synthetic seasons using `_make_train_data()` patterns from `test_model_xgboost.py`
- For CLI tests, use `typer.testing.CliRunner` to invoke the command
- Mark full-data integration tests as `@pytest.mark.integration` if added

**However**, the `StatefulFeatureServer` requires a `ChronologicalDataServer` which requires a `Repository` with real Parquet data. For unit testing the CLI logic:
1. Mock the feature serving layer to return a pre-built DataFrame
2. Test the tracking entities (ModelRun, Prediction, RunStore) independently with `tmp_path`
3. Test the CLI command structure with `CliRunner` and mocked internals

### File Structure

```
src/ncaa_eval/cli/
├── __init__.py              # NEW — package marker, may export app
├── __main__.py              # NEW — enables `python -m ncaa_eval.cli`
├── main.py                  # NEW — Typer app with train command
└── train.py                 # NEW — training pipeline orchestration

src/ncaa_eval/model/
├── __init__.py              # EXISTING — may add tracking exports
├── base.py                  # EXISTING — DO NOT MODIFY
├── registry.py              # EXISTING — DO NOT MODIFY
├── logistic_regression.py   # EXISTING — DO NOT MODIFY
├── elo.py                   # EXISTING — DO NOT MODIFY
├── xgboost_model.py         # EXISTING — DO NOT MODIFY
└── tracking.py              # NEW — ModelRun, Prediction, RunStore
```

```
tests/unit/
├── test_model_tracking.py   # NEW — ModelRun, Prediction, RunStore tests
└── test_cli_train.py        # NEW — CLI integration tests
```

### Existing Test Patterns to Follow

**CLI testing with Typer** — use `typer.testing.CliRunner`:
```python
from typer.testing import CliRunner
from ncaa_eval.cli.main import app

runner = CliRunner()

def test_train_invalid_model():
    result = runner.invoke(app, ["train", "--model", "nonexistent"])
    assert result.exit_code != 0
    assert "Available" in result.output
```

**Pydantic model testing** — see `test_model_xgboost.py` for config JSON round-trip patterns.

**tmp_path for persistence** — see `test_model_elo.py` and `test_model_xgboost.py` for save/load round-trip with `tmp_path`.

### Dependencies Already Available

| Package | Usage | In pyproject.toml? |
|:---|:---|:---|
| `typer` (0.24.0 installed) | CLI framework | Yes (`typer = {version = ">=0.15,<2", extras = ["all"]}`) |
| `rich` (14.3.2 installed) | Progress bars, tables | Yes (transitive via Typer[all]) |
| `pydantic` (^2.12.5) | ModelRun, Prediction data models | Yes |
| `pyarrow` (^23.0.1) | Parquet persistence for predictions | Yes |
| `pandas` | DataFrame operations | Yes |

**No new dependencies needed.**

### Existing Codebase Context (DO NOT Reimplement)

| Building Block | Module | Relevant API | Story |
|:---|:---|:---|:---|
| Model ABC | `model.base` | `Model`, `StatefulModel`, `ModelConfig` | 5.2 |
| Plugin registry | `model.registry` | `get_model(name)`, `list_models()`, `ModelNotFoundError` | 5.2 |
| EloModel | `model.elo` | `EloModel`, `EloModelConfig` | 5.3 |
| XGBoostModel | `model.xgboost_model` | `XGBoostModel`, `XGBoostModelConfig` | 5.4 |
| LogisticRegression | `model.logistic_regression` | `LogisticRegressionModel` (test fixture) | 5.2 |
| Feature serving | `transform.feature_serving` | `StatefulFeatureServer`, `FeatureConfig` | 4.7 |
| Chrono data server | `transform.serving` | `ChronologicalDataServer` | 4.2 |
| ParquetRepository | `ingest` | `ParquetRepository(base_path=...)` | 2.2 |
| Game schema | `ingest.schema` | `Game` (Pydantic model) | 2.2 |
| Sync CLI (pattern) | `sync.py` | Typer app pattern to follow | 2.4 |

### Project Conventions (Must Follow)

- `from __future__ import annotations` required in all Python files
- Conventional commits: `feat(model): implement model run tracking and training CLI — story 5.5`
- `mypy --strict` mandatory
- No `for` loops over DataFrames for metric calculations (NFR1)
- Test file naming: `tests/unit/test_model_tracking.py`, `tests/unit/test_cli_train.py`
- `Literal` types for constrained string fields where applicable

### Previous Story Intelligence (Story 5.4)

Key learnings from the XGBoost story (previous story in epic):
- **File-checking pattern for `load()`**: Check ALL expected files exist BEFORE reading any. Apply same pattern to `RunStore.load_run()`.
- **`_is_fitted` guard pattern**: XGBoostModel added `_is_fitted` flag after code review. Consider if ModelRun needs similar state tracking.
- **Pydantic `Field` validators**: Use `Annotated[float, Field(ge=0.0, le=1.0)]` for `pred_win_prob` constraint — follows the `validation_fraction` pattern from Story 5.4 code review.
- **Session-scoped test fixtures**: Use `@pytest.fixture(scope="session")` for expensive fixture creation (e.g., trained model instances for CLI tests).
- **XGBoost 3.x**: `use_label_encoder` parameter removed — do NOT pass it.
- **PyArrow schema**: Follow `ParquetRepository` pattern with explicit schema (see `ingest/repository.py`).

### Git Intelligence (Recent Commits)

| Commit | Files | Pattern |
|:---|:---|:---|
| `ff02116` (Story 5.4) | `model/xgboost_model.py`, `test_model_xgboost.py` | Stateless model + config + save/load + tests |
| `625744d` (Story 5.3) | `model/elo.py`, `test_model_elo.py` | Stateful model wrapping feature engine |
| `812a844` (Story 5.2) | `model/base.py`, `model/registry.py` | ABC + plugin registry foundation |

### Project Structure Notes

- New package: `src/ncaa_eval/cli/` (4 files: `__init__.py`, `__main__.py`, `main.py`, `train.py`)
- New file: `src/ncaa_eval/model/tracking.py`
- New test files: `tests/unit/test_model_tracking.py`, `tests/unit/test_cli_train.py`
- No modifications to existing files (except possibly `model/__init__.py` to export tracking types)

### References

- [Source: _bmad-output/planning-artifacts/epics.md — Story 5.5 AC]
- [Source: specs/05-architecture-fullstack.md — Section 4.1 (ModelRun, Prediction entities), Section 6.1 (Training workflow), Section 10.3 (CLI design)]
- [Source: specs/research/modeling-approaches.md — Section 8 (MVP scope), Section 5 (interface design)]
- [Source: src/ncaa_eval/model/base.py — Model, StatefulModel (fit/predict_proba contract)]
- [Source: src/ncaa_eval/model/registry.py — get_model, list_models, ModelNotFoundError]
- [Source: src/ncaa_eval/model/xgboost_model.py — Stateless model pattern, save/load file-checking]
- [Source: src/ncaa_eval/model/elo.py — Stateful model pattern, state persistence]
- [Source: src/ncaa_eval/transform/feature_serving.py — StatefulFeatureServer, FeatureConfig]
- [Source: src/ncaa_eval/transform/serving.py — ChronologicalDataServer]
- [Source: src/ncaa_eval/ingest/repository.py — ParquetRepository, PyArrow schema pattern]
- [Source: sync.py — Typer CLI pattern reference]
- [Source: _bmad-output/implementation-artifacts/5-4-implement-reference-stateless-model-xgboost.md — Previous story learnings]
- [Source: _bmad-output/planning-artifacts/template-requirements.md — Accumulated project conventions]

## Dev Agent Record

### Agent Model Used

{{agent_model_name_version}}

### Debug Log References

### Completion Notes List

### File List
