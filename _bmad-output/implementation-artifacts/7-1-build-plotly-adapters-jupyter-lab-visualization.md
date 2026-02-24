# Story 7.1: Build Plotly Adapters for Jupyter Lab Visualization

Status: ready-for-dev

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a data scientist,
I want API methods on evaluation and simulation result objects that return interactive Plotly figures,
So that I can visualize calibration, metrics, and results directly in Jupyter notebooks without leaving the ncaa_eval API surface.

## Acceptance Criteria

1. **Given** a `BacktestResult` with fold evaluation data, **When** the developer calls `plot_backtest_summary(result)`, **Then** it returns a `plotly.graph_objects.Figure` that renders inline in Jupyter showing per-year metrics as a grouped bar or line chart.

2. **Given** predictions and actuals from a single fold or aggregated backtest, **When** the developer calls `plot_reliability_diagram(y_true, y_prob)`, **Then** it returns a Figure showing predicted vs. actual probability with per-bin sample counts as bar annotations.

3. **Given** multiple `BacktestResult` objects keyed by model name, **When** the developer calls `plot_metric_comparison(results, metric)`, **Then** it returns a Figure with multi-model overlay lines for the specified metric across years.

4. **Given** a `SimulationResult` with advancement probabilities, **When** the developer calls `plot_advancement_heatmap(result, team_labels)`, **Then** it returns a heatmap Figure showing per-team probability of advancing to each round.

5. **Given** a `BracketDistribution` from Monte Carlo simulation, **When** the developer calls `plot_score_distribution(dist)`, **Then** it returns a histogram Figure with percentile markers (5th, 25th, 50th, 75th, 95th).

6. All figures use the project's functional color palette: Green (`#28a745`), Red (`#dc3545`), Neutral (`#6c757d`).

7. All figures are interactive (hover tooltips, zoom, pan) — standard Plotly behavior.

8. Evaluation metrics and fold data remain available as Pandas DataFrames via existing `BacktestResult.summary` and `FoldResult.metrics` (no changes needed — already satisfied).

9. Real-time progress bars are provided for long-running backtest and simulation loops when executed in Jupyter cells via `tqdm.auto`.

10. Adapters are covered by unit tests validating returned `plotly.graph_objects.Figure` structure and data content.

## Tasks / Subtasks

- [ ] Task 1: Add `tqdm` dependency (AC: #9)
  - [ ] 1.1 Add `tqdm` to `[tool.poetry.dependencies]` in `pyproject.toml`
  - [ ] 1.2 Ensure `tqdm` is installed in the conda env (`pip install tqdm` if needed)

- [ ] Task 2: Create `src/ncaa_eval/evaluation/plotting.py` module (AC: #1–7)
  - [ ] 2.1 Implement `plot_reliability_diagram(y_true, y_prob, *, n_bins=10, title=None) -> go.Figure` — scatter + bar overlay showing calibration bins, diagonal reference line, bin counts as text annotations
  - [ ] 2.2 Implement `plot_backtest_summary(result: BacktestResult, *, metrics=None) -> go.Figure` — grouped bar chart (x=year, grouped by metric) or line chart showing per-year metric values from `result.summary` DataFrame
  - [ ] 2.3 Implement `plot_metric_comparison(results: dict[str, BacktestResult], metric: str) -> go.Figure` — multi-model overlay line chart: one line per model, x=year, y=metric value
  - [ ] 2.4 Implement `plot_advancement_heatmap(result: SimulationResult, team_labels: dict[int, str] | None = None) -> go.Figure` — heatmap with teams on y-axis, rounds on x-axis, cell values = P(advance)
  - [ ] 2.5 Implement `plot_score_distribution(dist: BracketDistribution, *, title=None) -> go.Figure` — histogram from `dist.histogram_bins`/`dist.histogram_counts` with vertical lines at percentile markers
  - [ ] 2.6 Define module-level color constants: `COLOR_GREEN`, `COLOR_RED`, `COLOR_NEUTRAL` from UX spec
  - [ ] 2.7 Apply consistent dark-mode-compatible Plotly template (`plotly_dark`) and functional color palette to all figures

- [ ] Task 3: Integrate progress bars into backtest and simulation (AC: #9)
  - [ ] 3.1 Add optional `progress: bool = False` parameter to `run_backtest()` — when True, wrap fold iteration with `tqdm.auto.tqdm` for Jupyter-aware progress bar display
  - [ ] 3.2 Add optional `progress: bool = False` parameter to `simulate_tournament_mc()` — when True, wrap simulation batch iteration with `tqdm.auto.tqdm`
  - [ ] 3.3 Ensure progress bars are no-ops in non-interactive environments (tqdm.auto handles this)

- [ ] Task 4: Export public API (AC: all)
  - [ ] 4.1 Add all plotting functions to `evaluation/__init__.py` and `__all__`
  - [ ] 4.2 Re-export from top-level `ncaa_eval.__init__` if appropriate (follow existing pattern)

- [ ] Task 5: Write unit tests (AC: #10)
  - [ ] 5.1 Test each plotting function returns `go.Figure`
  - [ ] 5.2 Test figure data content: correct number of traces, correct axis labels, correct data values
  - [ ] 5.3 Test reliability diagram: diagonal reference line present, bin counts in annotations
  - [ ] 5.4 Test color palette application: traces use `COLOR_GREEN`, `COLOR_RED`, `COLOR_NEUTRAL`
  - [ ] 5.5 Test edge cases: empty BacktestResult (1 fold), single-point reliability diagram, SimulationResult with no bracket distributions
  - [ ] 5.6 Test progress bar integration: verify `run_backtest(progress=True)` and `simulate_tournament_mc(progress=True)` don't error (mock tqdm)

- [ ] Task 6: Verify quality gates (AC: all)
  - [ ] 6.1 `mypy --strict src/ncaa_eval tests`
  - [ ] 6.2 `ruff check .`
  - [ ] 6.3 Full test suite passes

## Dev Notes

### Architectural Decision: Standalone Functions vs. Methods on Frozen Dataclasses

`BacktestResult`, `FoldResult`, `SimulationResult`, and `BracketDistribution` are **frozen dataclasses**. Adding methods to them would introduce a Plotly import into the core evaluation module, coupling the data layer to the visualization layer.

**Decision:** Use **standalone functions** in a new `evaluation/plotting.py` module. Functions accept result objects as arguments and return `go.Figure`. This keeps the data structures clean and the visualization dependency optional.

```python
# Usage pattern:
from ncaa_eval.evaluation.plotting import plot_reliability_diagram, plot_backtest_summary

fig = plot_backtest_summary(backtest_result)
fig.show()  # renders inline in Jupyter

fig2 = plot_reliability_diagram(fold.actuals.to_numpy(), fold.predictions.to_numpy())
fig2.show()
```

### Existing Code — DO NOT Reimplement

All of the following exist and must NOT be modified (except for adding `progress` parameter):

- `evaluation/metrics.py`: `reliability_diagram_data()` returns `ReliabilityData` — **USE THIS** for reliability diagram plotting (it provides `fraction_of_positives`, `mean_predicted_value`, `bin_counts`, `bin_edges`)
- `evaluation/backtest.py`: `BacktestResult`, `FoldResult`, `run_backtest()` — use `.summary` DataFrame directly for backtest summary plots
- `evaluation/simulation.py`: `SimulationResult`, `BracketDistribution` — use `.advancement_probs`, `.bracket_distributions` for simulation plots
- `evaluation/splitter.py`: `CVFold`, `walk_forward_splits` — no changes
- `evaluation/__init__.py`: existing exports — extend, don't replace

### Plotting Module Design

**File:** `src/ncaa_eval/evaluation/plotting.py`

```python
"""Plotly visualization adapters for evaluation results.

Provides standalone functions that accept evaluation result objects
and return interactive ``plotly.graph_objects.Figure`` instances for
Jupyter notebook rendering.
"""

from __future__ import annotations

import plotly.graph_objects as go

# UX spec color palette
COLOR_GREEN = "#28a745"
COLOR_RED = "#dc3545"
COLOR_NEUTRAL = "#6c757d"

# Use plotly_dark template for dark-mode compatibility
TEMPLATE = "plotly_dark"
```

**Function signatures:**

```python
def plot_reliability_diagram(
    y_true: npt.NDArray[np.float64],
    y_prob: npt.NDArray[np.float64],
    *,
    n_bins: int = 10,
    title: str | None = None,
) -> go.Figure:
    """Reliability diagram: predicted vs. actual probability with bin counts."""

def plot_backtest_summary(
    result: BacktestResult,
    *,
    metrics: Sequence[str] | None = None,
) -> go.Figure:
    """Per-year metric values from a backtest result."""

def plot_metric_comparison(
    results: Mapping[str, BacktestResult],
    metric: str,
) -> go.Figure:
    """Multi-model overlay: one line per model for a given metric across years."""

def plot_advancement_heatmap(
    result: SimulationResult,
    team_labels: Mapping[int, str] | None = None,
) -> go.Figure:
    """Heatmap of per-team advancement probabilities by round."""

def plot_score_distribution(
    dist: BracketDistribution,
    *,
    title: str | None = None,
) -> go.Figure:
    """Histogram of bracket score distribution with percentile markers."""
```

### Reliability Diagram Implementation Notes

Call `reliability_diagram_data(y_true, y_prob, n_bins=n_bins)` from `metrics.py` to get `ReliabilityData`. Then:

1. **Scatter trace**: `x=mean_predicted_value`, `y=fraction_of_positives` — main calibration curve (color: `COLOR_GREEN`)
2. **Diagonal line**: `go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash', color=COLOR_NEUTRAL))` — perfect calibration reference
3. **Bar trace** (secondary y-axis): `x=mean_predicted_value`, `y=bin_counts` — shows sample counts per bin (color: `COLOR_NEUTRAL`, opacity=0.3)
4. **Layout**: dual y-axis (left = fraction of positives, right = bin count), dark template, title

### Backtest Summary Implementation Notes

Use `result.summary` DataFrame directly. Columns are metric names + `elapsed_seconds`. Index is year.

- Default: show all metric columns (exclude `elapsed_seconds`)
- Line chart with one trace per metric: `x=years`, `y=metric_values`
- Use color cycling from the palette (extend with additional colors if >3 metrics)

### Multi-Model Comparison Implementation Notes

For each model name → BacktestResult, extract `result.summary[metric]` series. Plot all as overlaid lines on the same axes.

- Color per model (cycle through a palette)
- Hover: show model name, year, metric value

### Advancement Heatmap Implementation Notes

`SimulationResult.advancement_probs` is shape `(n_teams, n_rounds)`. Team IDs are in `SimulationResult.bracket.team_ids` (accessible via the bracket's `team_ids` attribute on the BracketStructure, if stored). If `team_labels` is provided, map team IDs to labels; otherwise display team index.

Round labels: ["R64", "R32", "S16", "E8", "F4", "Championship"]

Use `go.Heatmap` with `colorscale` mapping to green (high probability) and red (low probability).

### Score Distribution Implementation Notes

`BracketDistribution` provides `histogram_bins` (bin edges) and `histogram_counts` (bin counts). Convert to a bar chart. Add vertical lines at percentile values from `dist.percentiles` dict (keys: 5, 25, 50, 75, 95).

### Progress Bar Integration

**Backtest progress (Task 3.1):** In `run_backtest()`, the fold evaluation is dispatched via `joblib.Parallel`. For sequential execution (`n_jobs=1`), wrap the loop with `tqdm`. For parallel execution, use `tqdm` as a callback or wrap the result collection. Since joblib doesn't natively support tqdm, the simplest approach:
- When `progress=True` and `n_jobs=1`, use `tqdm` around the sequential fold loop
- When `progress=True` and `n_jobs != 1`, use a Rich progress bar (already using Rich console) or a joblib callback

**Important:** The `progress` parameter must be added in a backwards-compatible way (default `False`). Avoid breaking existing callers.

**Simulation progress (Task 3.2):** `simulate_tournament_mc` uses a vectorized batch approach. If the function already processes all simulations in one vectorized call, progress bars add little value. Consider whether to wrap at the batch level or skip for the vectorized path. If the function has an internal loop (e.g., per-simulation), wrap that loop. Otherwise, wrap the overall call with a simple start/done indicator.

### mypy Considerations

- `plotly` does NOT ship `py.typed` — use `type: ignore[import-untyped]` on the plotly import
- All function signatures must be fully annotated
- `go.Figure` return type: use `plotly.graph_objects.Figure` after import

### Project Structure Notes

```
src/ncaa_eval/evaluation/
  __init__.py      # existing — add new exports
  backtest.py      # existing — MODIFY (add progress parameter)
  metrics.py       # existing — NO CHANGES
  plotting.py      # NEW — all Plotly visualization functions
  simulation.py    # existing — MODIFY (add progress parameter to simulate_tournament_mc)
  splitter.py      # existing — NO CHANGES
```

```
tests/unit/
  test_evaluation_plotting.py  # NEW — all plotting tests
  test_evaluation_backtest.py  # existing — add progress parameter test
```

### Architecture Constraints

- **Type Sharing (§12 Architecture):** Plotting functions accept and return typed objects (`BacktestResult`, `go.Figure`). No untyped dicts.
- **No Direct IO in UI (§12):** Plotting functions don't read files — they accept pre-computed result objects.
- **mypy --strict:** All new code must pass. plotly import gets `type: ignore[import-untyped]`.
- **`from __future__ import annotations`:** Required in all new files.
- **Google docstring style:** `Args:`, `Returns:`, `Raises:` format per project convention.
- **Vectorization (NFR1):** Plotting functions use pre-computed data — no metric recalculation.
- **Frozen dataclasses:** Do NOT add methods to frozen dataclasses. Use standalone functions.

### Dependencies

**Add:**
- `tqdm` — progress bars for Jupyter/terminal (add to `[tool.poetry.dependencies]`)

**Already available:**
- `plotly` — already in `pyproject.toml` (version `*`, latest is 6.5.x)
- `numpy`, `pandas` — data handling
- `rich` — existing progress/console output in `run_backtest`

### Previous Story Learnings (Stories 6.1–6.6)

- **Google docstring style**: NOT NumPy style. Use `Args:`, `Returns:`, `Raises:`.
- **Frozen dataclasses**: All result containers are frozen. Do NOT try to add methods. Use standalone functions.
- **Protocol type contracts**: Return types precisely annotated — `npt.NDArray[np.float64]` shape comments.
- **Registry pattern**: Follow `model/registry.py` for any new registries (not needed for this story).
- **`from __future__ import annotations`**: Required in ALL Python files.
- **`ReliabilityData`** from `metrics.py` already provides everything needed for reliability diagrams — don't recompute.
- **`BacktestResult.summary`** is a DataFrame with year index and metric columns + `elapsed_seconds` — use directly.
- **`BracketDistribution`** provides pre-computed `histogram_bins`, `histogram_counts`, `percentiles` — use directly.
- **Module-level constants**: Use `N_ROUNDS = 6` from `simulation.py` if needed. Don't redefine.

### Performance Targets

| Operation | Target |
|:---|:---|
| Any single plot function | < 100 ms |
| Reliability diagram (1000 predictions, 10 bins) | < 50 ms |
| Advancement heatmap (64 teams × 6 rounds) | < 50 ms |

### Plotly Version Notes

Plotly 6.5.x is the current latest (as of Feb 2026). Key points:
- `plotly.graph_objects` is the recommended API for typed, validated figure construction
- `plotly_dark` template is built-in and provides dark-mode styling
- Figures render inline in JupyterLab 4.x via the `plotly` renderer (auto-detected)
- No special renderer configuration needed for JupyterLab

### MEMORY.md Notebook File Size Rule

**CRITICAL:** The MEMORY.md rule about large Plotly outputs applies to this story. Plotly embeds data as JSON. For the plotting functions in this story, the data sizes are small (max 64 teams × 6 rounds for heatmap, ~20 years of metric data for backtest) so inline Plotly is safe. However, if any downstream notebook uses these adapters on datasets with 800K+ rows, switch to matplotlib. The functions in this story operate on **pre-aggregated** result objects, not raw data — this is inherently safe.

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 7.1 — acceptance criteria]
- [Source: specs/05-architecture-fullstack.md §3 — Plotly in tech stack, §7.2 — st.plotly_chart, §12 — coding standards]
- [Source: specs/04-front-end-spec.md §4.2 — color palette, §5.2 — performance targets]
- [Source: src/ncaa_eval/evaluation/metrics.py — ReliabilityData, reliability_diagram_data()]
- [Source: src/ncaa_eval/evaluation/backtest.py — BacktestResult, FoldResult, run_backtest()]
- [Source: src/ncaa_eval/evaluation/simulation.py — SimulationResult, BracketDistribution]
- [Source: _bmad-output/implementation-artifacts/6-6-implement-tournament-scoring-user-defined-point-schedules.md — previous story learnings]

## Dev Agent Record

### Agent Model Used

{{agent_model_name_version}}

### Debug Log References

### Completion Notes List

### File List
