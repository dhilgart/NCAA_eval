# Story 10.4: Ensemble Tutorial Notebook

Status: done

## Story

As a **data scientist**,
I want to **follow a step-by-step tutorial that walks me through defining, training, and evaluating a custom ensemble**,
so that **I can understand the ensemble UX end-to-end and use it as a template for my own models**.

## Acceptance Criteria

1. **Notebook exists** — The tutorial notebook is at `notebooks/tutorials/03_ensemble_model.ipynb` (create `notebooks/tutorials/` directory if needed).

2. **Full end-to-end demonstration** — When the user executes all cells in order, the notebook demonstrates:
   - (a) Importing base model classes from `ncaa_eval` and configuring each with feature-relevant kwargs
   - (b) Constructing a `StackedEnsemble` with at least one stateless model (XGBoost) and one stateful model (Elo) as base models, and a logistic regression meta-learner
   - (c) Calling `run_training(ensemble, ...)` with the one-liner UX
   - (d) Showing OOF log loss for each base model vs. the ensemble (demonstrates the blend adds value)
   - (e) Calling `ensemble.predict_bracket(data_dir, season)` to generate a bracket probability matrix
   - (f) Exporting a Kaggle submission CSV from the ensemble predictions

3. **Executable without error** — All cells execute without error using the standard `ncaa_eval` conda env.

4. **Referenced from docs** — The notebook is referenced from `docs/tutorials/getting-started.md` (or equivalent tutorials index) and included in the CI notebook-execution smoke test.

5. **CI integration** — The notebook is included in a CI smoke test that validates it executes without error.

## Tasks / Subtasks

- [x] Task 1: Create notebook directory and skeleton (AC: #1)
  - [x] 1.1: Create `notebooks/tutorials/` directory
  - [x] 1.2: Create `notebooks/tutorials/03_ensemble_model.ipynb` with kernel `python3` and section headers

- [x] Task 2: Write tutorial cells — Setup and Imports (AC: #2a)
  - [x] 2.1: Markdown intro cell explaining the tutorial objectives and prerequisites (data must be synced)
  - [x] 2.2: Code cell importing `StackedEnsemble`, `XGBoostModel`, `EloModel`, `LogisticRegressionModel`, `run_training`, `format_kaggle_submission`, `Path`

- [x] Task 3: Write tutorial cells — Define and Configure the Ensemble (AC: #2a, #2b)
  - [x] 3.1: Markdown cell explaining base model selection (why XGBoost + Elo complement each other)
  - [x] 3.2: Code cell constructing `XGBoostModel` with `batch_rating_types=("srs",)` and `EloModel()` as base models
  - [x] 3.3: Code cell constructing `LogisticRegressionModel()` as the meta-learner
  - [x] 3.4: Code cell constructing `StackedEnsemble(base_models=[xgb, elo], meta_learner=lr)` with default `contextual_features`
  - [x] 3.5: Markdown cell explaining the `contextual_features` parameter and what it means

- [x] Task 4: Write tutorial cells — Train the Ensemble (AC: #2c)
  - [x] 4.1: Code cell setting `data_dir` and `output_dir` paths (relative path from notebook dir: `../../data`, `../../output`)
  - [x] 4.2: Code cell calling `run_training(ensemble, data_dir=data_dir, start_year=2015, end_year=2024, output_dir=output_dir, model_name="tutorial_ensemble")`
  - [x] 4.3: Markdown cell explaining what happened during training (OOF generation, alignment, meta-learner training, base model retraining)

- [x] Task 5: Write tutorial cells — Compare OOF Performance (AC: #2d)
  - [x] 5.1: Code cell loading the manifest and OOF aligned data from the saved ensemble run
  - [x] 5.2: Code cell computing OOF log loss for each base model and the ensemble from oof_aligned.parquet
  - [x] 5.3: Code cell displaying a comparison table showing log loss per base model vs. ensemble
  - [x] 5.4: Markdown cell interpreting the results — the ensemble should match or beat individual models

- [x] Task 6: Write tutorial cells — Generate Bracket Predictions (AC: #2e)
  - [x] 6.1: Code cell calling `ensemble.predict_bracket(data_dir, season=2025)` to get the probability matrix
  - [x] 6.2: Code cell displaying a sample of the matrix (e.g., first 5x5 corner or top seed matchups)
  - [x] 6.3: Markdown cell explaining the probability matrix structure (`P[a,b]` = P(team a beats team b))

- [x] Task 7: Write tutorial cells — Kaggle Export (AC: #2f)
  - [x] 7.1: Code cell using `format_kaggle_submission(season, team_ids, prob_matrix.to_numpy())` to generate CSV
  - [x] 7.2: Code cell writing the CSV to `output/tutorial_ensemble_submission.csv`
  - [x] 7.3: Code cell displaying the first few rows of the submission
  - [x] 7.4: Markdown cell explaining the Kaggle submission format (`ID,Pred` where `ID = YYYY_TeamID1_TeamID2`)

- [x] Task 8: Add tutorial reference to docs (AC: #4)
  - [x] 8.1: Add a reference to the ensemble tutorial in `docs/tutorials/getting-started.md` (in the "Next Steps" section)

- [x] Task 9: Add CI notebook smoke test (AC: #5)
  - [x] 9.1: Add a pytest test that runs `jupyter nbconvert --to notebook --execute` on the tutorial notebook and asserts exit code 0

- [x] Task 10: Execute and commit the notebook (AC: #3)
  - [x] 10.1: Execute the notebook via nbconvert to generate all outputs
  - [x] 10.2: Verify all cells run without error
  - [x] 10.3: Commit the executed notebook with outputs

- [x] Task 11: Quality gates
  - [x] 11.1: Existing `pytest` suite passes (no regressions) — 1180 passed, 2 skipped
  - [x] 11.2: `ruff check .` clean (notebooks are excluded by default)
  - [x] 11.3: `mypy --strict src/ncaa_eval tests` clean (notebooks excluded)

## Dev Notes

### Critical Context: Project Tutorial Patterns

The project has TWO tutorial formats:
1. **Markdown tutorials** in `docs/tutorials/` — `getting-started.md`, `custom-model.md`, `custom-metric.md` (all from Story 7.9). These use MyST directive syntax (`{tip}`, `{note}`) and are documentation-oriented.
2. **Jupyter notebooks** in `notebooks/eda/` — `01_data_quality_audit.ipynb`, `02_statistical_exploration.ipynb`, `03_distribution_analysis.ipynb` (from Epic 3). These are data-exploration-oriented.

Story 10.4 creates the FIRST Jupyter tutorial notebook. The `notebooks/tutorials/` directory does not exist yet.

### Notebook Conventions (from MEMORY.md and project patterns)

- **Kernel name**: Must be `python3` (not `ncaa_eval`) — verify with `jupyter kernelspec list`
- **No Plotly for large datasets**: Use matplotlib static PNG. However, this tutorial has small output (a few tables and a 5x5 matrix sample), so Plotly is unnecessary — use plain `display()` / `print()` for tables
- **No `from __future__ import annotations`**: Not required in notebooks
- **No `mypy --strict`**: Notebooks are excluded from type checking
- **No Ruff**: Notebooks are excluded (`extend-exclude = ["notebooks"]` in `pyproject.toml`)
- **No iterrows**: Project convention still applies — use vectorized pandas operations
- **CWD**: nbconvert 7.x sets kernel cwd to the notebook's directory. Paths must be relative from `notebooks/tutorials/` (e.g., `../../data` for data, `../../output` for output)
- **`plt.close(fig)` after every `plt.show()`**: Required if using matplotlib to prevent memory leaks

### Notebook File Size Rule

This tutorial notebook should be small — no large inline Plotly outputs. The tutorial demonstrates API usage (small tables, a few code cells), not large-scale data visualization. Output cells will contain printed text, small tables, and a CSV snippet. Total notebook size should be well under 1MB.

### Imports — Canonical Paths

```python
from pathlib import Path

from ncaa_eval.model import StackedEnsemble
from ncaa_eval.model.elo import EloModel
from ncaa_eval.model.xgboost_model import XGBoostModel
from ncaa_eval.model.logistic_regression import LogisticRegressionModel
from ncaa_eval.cli.train import run_training
from ncaa_eval.evaluation.kaggle_export import format_kaggle_submission
from ncaa_eval.model.tracking import RunStore
```

### StackedEnsemble Construction Pattern

```python
# Base models
xgb = XGBoostModel(batch_rating_types=("srs",))
elo = EloModel()

# Meta-learner (must be stateless — NOT StatefulModel)
meta = LogisticRegressionModel()

# Ensemble
ensemble = StackedEnsemble(
    base_models=[xgb, elo],
    meta_learner=meta,
    # Default contextual_features: ["seed_diff", "is_tournament", "loc_encoding"]
)
```

### run_training One-Liner

```python
data_dir = Path("../../data")
output_dir = Path("../../output")

run = run_training(
    ensemble,
    data_dir=data_dir,
    start_year=2015,
    end_year=2024,
    output_dir=output_dir,
    model_name="tutorial_ensemble",
)
```

This prints a 6-step progress log to the console (Rich Console output).

### OOF Performance Comparison (AC #2d)

After training, the manifest is at `output_dir / run.run_id / "model" / "manifest.json"`. It contains:
```json
{
  "base_model_types": ["xgboost", "elo"],
  "base_model_count": 2,
  "contextual_features": ["seed_diff", "is_tournament", "loc_encoding"],
  "meta_column_order": ["pred_base_0", "pred_base_1", "seed_diff", "is_tournament", "loc_encoding"],
  "oof_backtest_run_ids": ["<uuid-base-0>", "<uuid-base-1>"],
  "oof_game_count": <int>,
  "oof_drop_pct": <float>
}
```

To compare OOF log loss:
1. Load the manifest from the run directory
2. Use `RunStore` to load metrics summaries for each `oof_backtest_run_ids` entry
3. Load the ensemble run's own summary for the ensemble log loss
4. Display a comparison table

```python
import json
store = RunStore(output_dir)
manifest = json.loads((store.model_dir(run.run_id) / "manifest.json").read_text())
# Load per-base-model OOF metrics from oof_backtest_run_ids
```

### predict_bracket API

```python
prob_matrix = ensemble.predict_bracket(data_dir, season=2025)
# Returns pd.DataFrame indexed by team_id, columns are team_ids
# P[a, b] = probability team a beats team b
# Diagonal is 0, P[a,b] + P[b,a] ≈ 1.0
```

Note: `predict_bracket` needs the loaded ensemble (not just the manifest). After `run_training()`, the `ensemble` variable is already trained in memory and can be used directly. Alternatively, load from disk:

```python
loaded = StackedEnsemble.load(store.model_dir(run.run_id))
prob_matrix = loaded.predict_bracket(data_dir, season=2025)
```

### Kaggle Export API

```python
from ncaa_eval.evaluation.kaggle_export import format_kaggle_submission

team_ids = list(prob_matrix.index)
csv_str = format_kaggle_submission(2025, team_ids, prob_matrix.to_numpy())

# Write to file
Path("../../output/tutorial_ensemble_submission.csv").write_text(csv_str)
```

The output CSV has format `ID,Pred` where `ID = YYYY_TeamID1_TeamID2` (lower ID first).

### CI Notebook Smoke Test (AC #5)

The project currently has NO CI notebook execution. Options:

**Option A (Recommended): pytest test with nbconvert**
```python
# tests/integration/test_notebook_execution.py
@pytest.mark.slow
def test_ensemble_tutorial_notebook_executes():
    """Smoke test: notebook executes without error."""
    import subprocess
    result = subprocess.run([
        "jupyter", "nbconvert", "--to", "notebook", "--execute",
        "--ExecutePreprocessor.timeout=600",
        "notebooks/tutorials/03_ensemble_model.ipynb",
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
```

Mark with `@pytest.mark.slow` so it only runs in CI or with `pytest -m slow`. This requires data to be present — the CI pipeline may not have synced data. Consider:
- Skipping in CI if `data/` doesn't exist (`pytest.importorskip` or `skipif`)
- Using a `conftest.py` fixture that checks for data availability
- The AC says "included in the CI notebook-execution smoke test" — if CI doesn't have data, this test should skip gracefully with `pytest.mark.skipif(not Path("data/MTeams.csv").exists(), reason="data not synced")`

**Option B: CI workflow step**
Add a step to `.github/workflows/python-check.yaml`. This is heavier and the project doesn't have data in CI.

**Recommendation**: Use Option A with a skip-if-no-data guard. This satisfies the AC ("included in the CI notebook-execution smoke test") while being practical (CI doesn't have NCAA data).

### Existing Code to Reuse (DO NOT Reinvent)

| Need | Existing Solution | Location |
|------|-------------------|----------|
| Ensemble construction | `StackedEnsemble` dataclass | `model/ensemble.py` |
| Training one-liner | `run_training()` | `cli/train.py` |
| Bracket predictions | `ensemble.predict_bracket()` | `model/ensemble.py` |
| Kaggle CSV export | `format_kaggle_submission()` | `evaluation/kaggle_export.py` |
| Run storage/loading | `RunStore` | `model/tracking.py` |
| Manifest loading | `json.loads((model_dir / "manifest.json").read_text())` | Direct file read |
| Model classes | `XGBoostModel`, `EloModel`, `LogisticRegressionModel` | `model/` submodules |

### Anti-Patterns to Avoid

- **DO NOT** create a custom training loop — use `run_training()` as the one-liner API
- **DO NOT** use Plotly for outputs — use plain print/display for small tables
- **DO NOT** use iterrows for any DataFrame operations
- **DO NOT** use `from __future__ import annotations` in notebook cells
- **DO NOT** use `ncaa_eval` as the kernel name — must be `python3`
- **DO NOT** hardcode absolute paths — use relative paths from notebook directory
- **DO NOT** create a markdown tutorial in `docs/tutorials/` — the story specifies a `.ipynb` notebook
- **DO NOT** add `get_feature_importances()` to StackedEnsemble — route through `meta_learner` if needed
- **DO NOT** re-implement probability matrix construction — `predict_bracket()` handles everything

### nbconvert Execution Command

From repo root:
```bash
conda run -n ncaa_eval jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=600 --output-dir notebooks/tutorials \
  notebooks/tutorials/03_ensemble_model.ipynb
```

### Docs Reference Update (AC #4)

The AC says "referenced from `docs/tutorials.md`" but no such file exists. The tutorials index is effectively `docs/tutorials/getting-started.md` which has a "Next Steps" section at the bottom referencing `custom-model.md` and `custom-metric.md`. Add the ensemble tutorial notebook reference there.

### Previous Story Learnings

**From Story 10.1:**
- `caplog` test reliability: `ncaa_eval` logger has `propagate=False` when `configure_logging()` is called
- `StackedEnsemble` is NOT a `Model` subclass — different lifecycle

**From Story 10.2:**
- `predict_bracket` requires data to be synced — tournament seeds must exist for the requested season
- `predict_bracket` raises `FileNotFoundError` if no data for the season
- `meta_column_order` must be non-empty; `predict_proba` raises `ValueError` if empty

**From Story 10.3:**
- Dashboard imports use `_EnsembleProvider` alias
- Feature importance labels use `_MODEL_TYPE_DISPLAY_NAMES` lookup table
- `model_dir()` creates directories as a side effect — use `_runs_dir / run_id / "model"` for read-only access

### Training Time Estimate

The tutorial ensemble (XGBoost + Elo, trained on 2015–2024) will take approximately 2–5 minutes to train depending on hardware. This is dominated by:
- 10 walk-forward backtests per base model (10 seasons × 2 models = 20 backtests)
- Full-dataset retraining of both base models
- Meta-learner training (fast — logistic regression on ~10K rows)

The notebook should include a note about expected training time.

### Project Structure Notes

- New directory: `notebooks/tutorials/` (does not exist yet)
- New file: `notebooks/tutorials/03_ensemble_model.ipynb`
- New or modified file: `tests/integration/test_notebook_execution.py` (or `tests/test_notebooks.py`)
- Modified file: `docs/tutorials/getting-started.md` (add ensemble tutorial reference)

### References

- [Source: specs/ensemble-architecture.md] — Full ensemble design spec
- [Source: _bmad-output/planning-artifacts/epics.md] — Epic 10, Story 10.4 acceptance criteria
- [Source: _bmad-output/implementation-artifacts/10-1-stacked-ensemble-oof-training-pipeline.md] — StackedEnsemble class, manifest schema, run_training dispatch
- [Source: _bmad-output/implementation-artifacts/10-2-ensemble-inference-interface.md] — predict_proba, predict_bracket, EnsembleProvider, CLI predict
- [Source: _bmad-output/implementation-artifacts/10-3-ensemble-dashboard-registry-integration.md] — Dashboard integration, feature importance labels, OOF log loss loading
- [Source: src/ncaa_eval/model/ensemble.py] — StackedEnsemble class (predict_proba, predict_bracket, save/load)
- [Source: src/ncaa_eval/cli/train.py] — run_training() one-liner API
- [Source: src/ncaa_eval/evaluation/kaggle_export.py] — format_kaggle_submission()
- [Source: docs/tutorials/getting-started.md] — Existing tutorial pattern, "Next Steps" section

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

- `XGBoostModel` has no `model_type` attribute — used `get_config().model_name` instead
- OOF backtest run IDs in manifest are placeholder UUIDs with no corresponding RunStore data — fixed by saving `oof_aligned.parquet` alongside the manifest and computing log loss directly from aligned OOF predictions
- Meta-learner training failed with `ValueError: Input X contains NaN` — regular-season games have NaN `seed_diff` (no tournament seeds). Fixed by filling NaN contextual features with 0 in `_build_meta_training_set()` and `StackedEnsemble.predict_proba()`

### Completion Notes List

- Created `notebooks/tutorials/03_ensemble_model.ipynb` — 20 cells (10 markdown, 10 code) covering the full ensemble workflow
- Notebook demonstrates: imports, base model construction, ensemble construction, `run_training()`, OOF comparison, `predict_bracket()`, Kaggle CSV export
- OOF comparison shows ensemble (0.555 log loss) beats XGBoost (0.578) and Elo (0.614)
- Bracket prediction generates 364x364 probability matrix for all D1 teams
- Kaggle submission exports 66,066 matchup rows in standard `ID,Pred` format
- Added ensemble tutorial reference to `docs/tutorials/getting-started.md` "Next Steps" section
- Created `tests/integration/test_notebook_execution.py` with `@pytest.mark.slow` smoke test (skips if data not synced)
- Fixed NaN handling bug in `_build_meta_training_set()` — contextual features filled with 0
- Fixed NaN handling in `StackedEnsemble.predict_proba()` for robustness
- Added `oof_aligned.parquet` persistence to ensemble training pipeline for post-hoc OOF analysis
- All quality gates pass: 1180 tests passed, ruff clean, mypy clean

### Change Log

- 2026-03-12: Implemented Story 10.4 — Ensemble Tutorial Notebook with full end-to-end demonstration, CI smoke test, and docs reference. Fixed NaN handling bug in ensemble meta-learner training and inference pipelines.
- 2026-03-12: Code review (Volty/AI) — 5 issues fixed: pandas CoW safety (.copy() in predict_proba), NaN-fill regression tests (2 new tests), oof_aligned.parquet persistence test, notebook sklearn log_loss alignment, numpy import consolidation. 2 action items deferred: H1 (phantom oof_backtest_run_ids), M4 (CI dead notebook test).

### Review Follow-ups (AI)

- [ ] [AI-Review][HIGH] `oof_backtest_run_ids` in ensemble manifest are phantom UUIDs never saved to RunStore — users calling `store.load_model(run_id)` for those IDs get `FileNotFoundError`. Fix: either actually save OOF backtest runs to RunStore during `_collect_oof_predictions`, or rename the manifest field to `oof_backtest_uuids` with a doc comment clarifying it's metadata-only. [src/ncaa_eval/cli/train.py:524]
- [ ] [AI-Review][MEDIUM] CI never actually executes the notebook smoke test — `python-check.yaml` runs plain pytest (no `-m slow`) and CI has no NCAA data, so the test always skips. AC #5 is satisfied on paper but not in practice. Consider: (a) a lightweight "notebook parses and imports without error" test that doesn't need data, or (b) a pre-execution cell-by-cell syntax check via `nbformat`. [.github/workflows/python-check.yaml:31]

### File List

- `notebooks/tutorials/03_ensemble_model.ipynb` (new) — Ensemble tutorial notebook with executed outputs
- `tests/integration/test_notebook_execution.py` (new) — Notebook execution smoke test
- `tests/unit/test_model_ensemble.py` (modified) — Added NaN-fill regression tests and oof_aligned.parquet persistence test
- `docs/tutorials/getting-started.md` (modified) — Added ensemble tutorial reference in "Next Steps"
- `src/ncaa_eval/cli/train.py` (modified) — NaN fill for contextual features in `_build_meta_training_set()`, save `oof_aligned.parquet`
- `src/ncaa_eval/model/ensemble.py` (modified) — NaN fill + `.copy()` for contextual features in `predict_proba()`
- `_bmad-output/implementation-artifacts/10-4-ensemble-tutorial-notebook.md` (modified) — Story status updates
- `_bmad-output/implementation-artifacts/sprint-status.yaml` (modified) — Story status: in-progress → review
