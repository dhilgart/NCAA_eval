# Story 7.6: Build Presentation Page — Pool Scorer & Point Outcome Analysis

Status: ready-for-dev

## Story

As a quantitative strategist,
I want to configure pool-specific scoring rules, score my chosen bracket against Monte Carlo simulations, and analyze the distribution of possible point outcomes,
so that I can understand my bracket's scoring potential under different pool formats and export a final entry.

## Acceptance Criteria

1. **Scoring Rule Selection & Custom Input**: The user can select from registered scoring rules (Standard, Fibonacci, Seed-Difference Bonus) via the sidebar's global "Scoring Format" filter or switch on-page. The user can also input a custom per-round point schedule via 6 number inputs (one per round: R64, R32, S16, E8, F4, Championship).

2. **Monte Carlo Outcome Analysis**: Clicking "Analyze Outcomes" runs `simulate_tournament()` in MC mode (reusing `run_bracket_simulation()` from `filters.py`), then calls `score_bracket_against_sims()` to score the most-likely bracket against all simulated outcomes. Results display the point outcome distribution: min, max, median, mean, 5th/25th/75th/95th percentiles, and standard deviation.

3. **Score Distribution Visualization**: A histogram of simulated bracket scores is rendered via `plot_score_distribution()` from `ncaa_eval.evaluation.plotting`, showing percentile markers and the mean score line.

4. **Simulation Progress**: MC simulation (10K+ iterations) displays progress via `st.spinner()` during computation to prevent UI freezing.

5. **Generate Submission Export**: The user can click "Generate Submission" to export the most-likely bracket picks as a CSV file with columns: `game_number`, `round`, `team_id`, `team_name`, `seed`, `win_probability`. Uses `st.download_button()` — no server-side file writes.

6. **Caching & Performance**: Simulation results are cached via `@st.cache_data` to avoid re-running on page navigation. Switching scoring rules should NOT trigger a full re-simulation if `sim_winners` are already cached — only the scoring step re-runs.

## Tasks / Subtasks

- [x] Task 1: Add scoring helper functions to `dashboard/lib/filters.py` (AC: #1, #2, #6)
  - [x] 1.1: `score_chosen_bracket(sim_data, scoring_rules)` — calls `score_bracket_against_sims()` with the most-likely bracket's `winners` array against `sim_result.sim_winners`; returns `dict[str, BracketDistribution]`
  - [x] 1.2: `build_custom_scoring(points_per_round)` — wraps a 6-element tuple in `DictScoring` from `ncaa_eval.evaluation.simulation`
  - [x] 1.3: `export_bracket_csv(bracket, most_likely, team_labels, prob_matrix)` — builds CSV string from most-likely bracket picks for download
- [x] Task 2: Build the Pool Scorer page in `dashboard/pages/4_Pool_Scorer.py` (AC: #1-6)
  - [x] 2.1: Replace placeholder with full page implementation
  - [x] 2.2: Add breadcrumb navigation matching existing pages (`Home > Presentation > Pool Scorer`)
  - [x] 2.3: Add custom scoring rule input (6 `st.number_input` fields for per-round points)
  - [x] 2.4: Add "Analyze Outcomes" button that triggers MC simulation + scoring
  - [x] 2.5: Render point outcome summary metrics (min, max, median, mean, percentiles, std) via `st.metric` cards
  - [x] 2.6: Render score distribution histogram via `plot_score_distribution()`
  - [x] 2.7: Add "Generate Submission" download button via `st.download_button()` with CSV export
  - [x] 2.8: Handle empty states (no model, no seeds, no MC data)
- [x] Task 3: Write tests (AC: all)
  - [x] 3.1: Unit tests for `score_chosen_bracket()` (mock sim_data, validate distribution output)
  - [x] 3.2: Unit tests for `build_custom_scoring()` (validate DictScoring wrapping)
  - [x] 3.3: Unit tests for `export_bracket_csv()` (validate CSV structure, column names, row count)
  - [x] 3.4: Page rendering tests for `4_Pool_Scorer.py` (mock data, empty states, successful render)
- [ ] Task 4: Verify quality gates (AC: all)
  - [ ] 4.1: `mypy --strict src/ncaa_eval tests dashboard` — 0 errors
  - [ ] 4.2: `ruff check .` — all modified files pass
  - [ ] 4.3: `pytest` — all tests pass

## Dev Notes

### Architecture & Data Flow

**Pool Scorer Pipeline:**
```
[Sidebar: year, run_id, scoring_name]
  → run_bracket_simulation(method="monte_carlo", n_simulations=N)  # cached
    → BracketSimulationResult with sim_result.sim_winners
  → score_bracket_against_sims(most_likely.winners, sim_winners, [scoring_rule])
    → {rule_name: per_sim_scores}  # shape (n_simulations,)
  → compute_bracket_distribution(per_sim_scores)
    → BracketDistribution (percentiles, mean, std, histogram)
  → plot_score_distribution(dist)
    → Plotly histogram with percentile markers
```

**Key insight — scoring vs. simulation separation:**
The Bracket Visualizer (Story 7.5) already runs `simulate_tournament()` which returns `sim_winners` (shape `(n_sims, 63)`). The Pool Scorer should reuse these MC results and only re-run the scoring layer. The `sim_winners` array is stored on `SimulationResult.sim_winners` and is `None` for analytical mode — the Pool Scorer MUST use MC mode.

**Important: `score_bracket_against_sims` scores YOUR picks against simulated outcomes.** This is different from `SimulationResult.bracket_distributions` which scores the chalk bracket (pre-game favorites) against simulations. The Pool Scorer should use `score_bracket_against_sims` with `most_likely.winners` (the model's best bracket) to answer "how would my bracket perform across all possible tournament outcomes?"

### Existing Code to Reuse (DO NOT REIMPLEMENT)

| Function | Module | Purpose |
|---|---|---|
| `run_bracket_simulation()` | `dashboard.lib.filters` | Orchestrates bracket sim; returns `BracketSimulationResult` (cached) |
| `score_bracket_against_sims()` | `ncaa_eval.evaluation.simulation` | Scores chosen bracket vs. MC sim_winners |
| `compute_bracket_distribution()` | `ncaa_eval.evaluation.simulation` | Computes percentiles, histogram from raw scores |
| `plot_score_distribution()` | `ncaa_eval.evaluation.plotting` | Plotly histogram with percentile markers |
| `DictScoring` | `ncaa_eval.evaluation.simulation` | Custom scoring from `{round_idx: points}` dict |
| `get_scoring()` / `list_scorings()` | `ncaa_eval.evaluation.simulation` | Scoring rule registry |
| `BracketSimulationResult` | `dashboard.lib.filters` | Dataclass with `sim_result`, `bracket`, `most_likely`, `prob_matrix`, `team_labels` |

### Data Structures Reference

**`score_bracket_against_sims()` signature:**
```python
def score_bracket_against_sims(
    chosen_bracket: npt.NDArray[np.int32],   # shape (63,) — round-major winners
    sim_winners: npt.NDArray[np.int32],      # shape (n_sims, 63) — from sim_result
    scoring_rules: Sequence[ScoringRule],     # list of rules to score
) -> dict[str, npt.NDArray[np.float64]]:     # rule_name → per-sim scores (n_sims,)
```

**`MostLikelyBracket.winners`:** `tuple[int, ...]` — 63 team *indices* (bracket-position indices, not team IDs) in round-major order. This is directly compatible with `score_bracket_against_sims` as `chosen_bracket` (convert to `np.array(winners, dtype=np.int32)`).

**`BracketDistribution` fields:**
- `scores`: `ndarray(n_simulations,)` — raw per-sim scores
- `percentiles`: `dict[int, float]` — keys 5, 25, 50, 75, 95
- `mean`: `float`
- `std`: `float`
- `histogram_bins`: `ndarray(n_bins + 1,)`
- `histogram_counts`: `ndarray(n_bins,)`

**`DictScoring` constructor:** `DictScoring(points: dict[int, float], name: str)` — round_idx (0-based) → points.

**Round indexing:** 0=R64, 1=R32, 2=S16, 3=E8, 4=F4, 5=Championship.

### CSV Export Format

The "Generate Submission" CSV should contain one row per game (63 rows), sorted in round-major order:
```csv
game_number,round,team_id,team_name,seed,win_probability
1,R64,1234,Connecticut,1,0.952
2,R64,1345,Duke,2,0.891
...
63,Championship,1234,Connecticut,1,0.347
```

Use `io.StringIO` to build the CSV in memory, then pass to `st.download_button(data=csv_string, file_name="bracket_submission.csv", mime="text/csv")`. No server-side file writes.

### Caching Strategy

- **Simulation results**: Reuse `run_bracket_simulation()` with `@st.cache_data(ttl=None)` — already implemented in filters.py
- **Scoring results**: `score_chosen_bracket()` should also use `@st.cache_data(ttl=None)` — key includes `(run_id, season, scoring_name, n_simulations)`
- **Custom scoring**: NOT cacheable (user can change inputs freely) — compute inline
- **Critical**: When the user switches scoring rules, only re-run scoring (not simulation). The `sim_winners` array is scoring-rule-independent.

### UI Layout

```
┌─ Breadcrumb: Home > Presentation > Pool Scorer ─────────────────┐
│                                                                   │
│  ┌─ Scoring Configuration ─────────────────────────────────────┐ │
│  │ [Sidebar scoring] or Custom:                                 │ │
│  │ R64: [1]  R32: [2]  S16: [4]  E8: [8]  F4: [16]  NCG: [32]│ │
│  │ [x] Use custom scoring  [ Analyze Outcomes ]                 │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─ Outcome Summary (st.metric cards) ─────────────────────────┐ │
│  │ Median: 85pts  Mean: 84.3pts  Std: 18.2pts                  │ │
│  │ Min: 32pts     Max: 148pts                                   │ │
│  │ 5th%: 54pts    25th%: 72pts   75th%: 96pts   95th%: 115pts │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─ Score Distribution ────────────────────────────────────────┐ │
│  │ [Plotly histogram with percentile lines]                     │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  [ Download Bracket CSV ]                                         │
└───────────────────────────────────────────────────────────────────┘
```

### Project Structure Notes

**Modified files:**
- `dashboard/pages/4_Pool_Scorer.py` — replace placeholder with full Pool Scorer implementation
- `dashboard/lib/filters.py` — add `score_chosen_bracket()`, `build_custom_scoring()`, `export_bracket_csv()`

**New test files:**
- `tests/unit/test_pool_scorer_page.py` — page rendering tests (follow pattern from `tests/unit/test_bracket_page.py`)

**Extended test files:**
- `tests/unit/test_dashboard_filters.py` — add tests for `score_chosen_bracket`, `build_custom_scoring`, `export_bracket_csv`

### Imports Required

```python
# In 4_Pool_Scorer.py:
from dashboard.lib.filters import (
    BracketSimulationResult,
    build_custom_scoring,
    export_bracket_csv,
    get_data_dir,
    load_tourney_seeds,
    run_bracket_simulation,
    score_chosen_bracket,
)
from ncaa_eval.evaluation.plotting import plot_score_distribution

# In filters.py (new imports):
from ncaa_eval.evaluation.simulation import (
    DictScoring,
    ScoringRule,
    compute_bracket_distribution,
    score_bracket_against_sims,
)
```

### Testing Standards

- Follow test patterns from `tests/unit/test_bracket_page.py` and `tests/unit/test_bracket_renderer.py`
- Mock `run_bracket_simulation` return value using `BracketSimulationResult` with fake data
- Mock `st.plotly_chart`, `st.download_button`, `st.metric`, `st.spinner` for page assertions
- Test empty states: no model selected, no seeds, analytical-only sim (no `sim_winners`)
- Test CSV export: validate 63 rows, correct columns, proper round labels
- Test custom scoring: verify `DictScoring` wrapping with 6-element points dict
- No `iterrows()` anywhere — vectorized operations only

### Previous Story Intelligence (7.5)

- `run_bracket_simulation()` already exists and is cached — reuse it directly, but force `method="monte_carlo"` for the Pool Scorer (analytical mode has `sim_winners=None`)
- `BracketSimulationResult.most_likely.winners` is a `tuple[int, ...]` of 63 bracket indices — convert to `np.array(winners, dtype=np.int32)` for `score_bracket_against_sims()`
- Breadcrumb pattern: `col_nav, col_bc = st.columns([1, 3])` with `st.page_link` and `st.caption`
- Session state keys: `selected_year`, `selected_run_id`, `selected_scoring` (from sidebar)
- The `n_simulations` and `scoring_name` are both cache keys for `run_bracket_simulation()` — switching scoring rule currently triggers full re-simulation (known Story 7.5 deferred issue). Accept this for now.
- `inspect.signature(scoring_cls)` pattern is used to detect `seed_map` parameter — reuse this pattern if `score_chosen_bracket` needs to instantiate scoring rules
- Extracted `_render_results()` pattern for C901 compliance — split the Pool Scorer render into similar helper functions if complexity warrants it

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 7.6] — AC definitions
- [Source: src/ncaa_eval/evaluation/simulation.py:995-1034] — `score_bracket_against_sims()` implementation
- [Source: src/ncaa_eval/evaluation/simulation.py:966-992] — `compute_bracket_distribution()` implementation
- [Source: src/ncaa_eval/evaluation/simulation.py:688-698] — `BracketDistribution` dataclass
- [Source: src/ncaa_eval/evaluation/plotting.py:275-279] — `plot_score_distribution()` signature
- [Source: dashboard/lib/filters.py:362-453] — `run_bracket_simulation()` orchestrator
- [Source: dashboard/pages/2_Presentation.py] — Bracket Visualizer page (pattern reference)
- [Source: dashboard/app.py:47] — Pool Scorer page registration and navigation
- [Source: specs/04-front-end-spec.md#5.1] — < 500ms interaction response target
- [Source: specs/04-front-end-spec.md#4.2] — Dark mode, green/red/neutral palette

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

### Completion Notes List

- Task 1: Added `score_chosen_bracket()`, `build_custom_scoring()`, `export_bracket_csv()`, and `_game_win_probability()` helper to `dashboard/lib/filters.py`. All 7 new unit tests pass. Added imports for `BracketDistribution`, `DictScoring`, `compute_bracket_distribution`, `score_bracket_against_sims` from simulation module.
- Task 2: Replaced Pool Scorer placeholder with full page implementation. Breadcrumbs, custom scoring config (6 number inputs), MC simulation with spinner, outcome summary metrics (9 st.metric cards), score distribution histogram via plot_score_distribution(), CSV download button, and empty state handling. Refactored into `_render_scoring_config()`, `_run_simulation()`, `_render_results()`, `_render_outcome_summary()`, `_render_distribution_chart()` for C901 compliance.
- Task 3: 14 tests total — 7 filter helper tests (score_chosen_bracket, build_custom_scoring, export_bracket_csv) and 7 page rendering tests (empty states: no run, no year, no seeds, sim failure, no sim_winners; success: metrics + chart + download; button triggers MC sim).

### File List

- `dashboard/lib/filters.py` — added 3 Pool Scorer helper functions + internal `_game_win_probability` helper
- `dashboard/pages/4_Pool_Scorer.py` — replaced placeholder with full Pool Scorer page implementation
- `tests/unit/test_dashboard_filters.py` — added 7 tests for new helpers (TestScoreChosenBracket, TestBuildCustomScoring, TestExportBracketCsv)
- `tests/unit/test_pool_scorer_page.py` — new file with 7 page rendering tests
