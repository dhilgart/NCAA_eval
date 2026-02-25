# Story 7.5: Build Presentation Page — Bracket Visualizer

Status: done

## Story

As a quantitative strategist,
I want an interactive bracket visualizer showing per-game win probabilities and team advancement odds from my trained model,
so that I can visually inspect the full 64-team tournament bracket and identify where my model diverges from chalk before committing to a pool entry.

## Acceptance Criteria

1. **Bracket Simulation On-the-Fly**: When the user selects a model run and tournament year from the sidebar, the page loads the trained model, constructs the bracket from `TourneySeedTable`, runs `simulate_tournament()` (analytical by default, MC optional), and displays results — all within the `< 500ms` cached-interaction target after first computation.

2. **Advancement Probability Heatmap**: The page renders the existing `plot_advancement_heatmap()` Plotly figure showing per-team advancement probabilities across all 6 rounds (R64 → Championship), using the functional color palette (green = high, red = low).

3. **Most-Likely Bracket Display**: The page renders the most-likely bracket (from `compute_most_likely_bracket()`) as an HTML/CSS bracket tree via `st.components.html()` — showing all 63 games across 6 rounds with team names, seeds, and win probabilities. All four regions must be visible simultaneously without horizontal scrolling (wide mode).

4. **Score Distribution (MC mode)**: When the user opts into Monte Carlo simulation (toggle or selectbox), `plot_score_distribution()` renders the bracket-score histogram with percentile markers for the selected scoring rule.

5. **Team Detail Expansion**: Clicking or hovering over a matchup in the heatmap shows the pairwise win probability and seed matchup in a tooltip or expander section.

6. **Breadcrumb Navigation**: Breadcrumb shows context (e.g., `Home > Presentation > Bracket Visualizer`).

## Tasks / Subtasks

- [x] Task 1: Add seed and simulation data-loading functions to `dashboard/lib/filters.py` (AC: #1)
  - [x] 1.1: `load_tourney_seeds(data_dir, season)` — loads `TourneySeedTable` from Kaggle CSV, returns serialized seed dicts for the selected season
  - [x] 1.2: `load_team_names(data_dir)` — loads team ID → team name mapping from `ParquetRepository.get_teams()`
  - [x] 1.3: `run_bracket_simulation(data_dir, run_id, season, scoring_name, method, n_simulations)` — orchestrates bracket construction, model loading, probability matrix computation, and `simulate_tournament()` call; returns `BracketSimulationResult` dataclass; cached with `@st.cache_resource(ttl=None)`
- [x] Task 2: Build the HTML/CSS bracket tree renderer (AC: #3)
  - [x] 2.1: Created `dashboard/lib/bracket_renderer.py` — pure function `render_bracket_html()` returning self-contained HTML/CSS string using CSS flexbox layout
  - [x] 2.2: Used project functional color palette (green→neutral→red) interpolation for win-probability coloring
  - [x] 2.3: Displays seed number, team name, and win probability at each node
  - [x] 2.4: Layout fits wide-mode desktop with all 4 regions visible simultaneously (left/center/right flex)
- [x] Task 3: Implement the bracket visualizer page in `dashboard/pages/2_Presentation.py` (AC: #1-6)
  - [x] 3.1: Replaced placeholder with full page implementation
  - [x] 3.2: Added breadcrumb navigation matching existing pages
  - [x] 3.3: Added simulation method selector (Analytical / Monte Carlo with N slider)
  - [x] 3.4: Renders advancement heatmap via `plot_advancement_heatmap()`
  - [x] 3.5: Renders bracket tree via `st.components.html()` calling `render_bracket_html()`
  - [x] 3.6: Renders score distribution (MC only) via `plot_score_distribution()`
  - [x] 3.7: Added expected-points table showing per-team EP for selected scoring rule
  - [x] 3.8: Handles empty states (no model, no seeds, no data, simulation failure)
- [x] Task 4: Write tests (AC: all)
  - [x] 4.1: Unit tests for new `filters.py` loader functions (mock TourneySeedTable / Team)
  - [x] 4.2: Unit tests for `bracket_renderer.py` (64-team bracket, XSS escaping, all 4 regions, probability colors)
  - [x] 4.3: Page rendering tests for `2_Presentation.py` (mock data, empty states, successful render)
- [x] Task 5: Verify quality gates (AC: all)
  - [x] 5.1: `mypy --strict src/ncaa_eval tests dashboard` — 0 errors (93 files)
  - [x] 5.2: `ruff check .` — all modified files pass
  - [x] 5.3: `pytest` — 850 passed, 1 skipped (14 new tests from code review round 2)

#### Review Follow-ups (AI) — Applied by Code Review (Round 1)
- [x] [AI-Review][HIGH] `_build_provider_from_folds` used `iterrows()` — replaced with vectorized `.map()` + numpy indexing [dashboard/lib/filters.py:328]
- [x] [AI-Review][HIGH] AC #5 (Team Detail Expansion) not implemented — added pairwise win probability expander to `_render_results()` [dashboard/pages/2_Presentation.py]
- [x] [AI-Review][MEDIUM] `run_bracket_simulation`, `_build_provider_from_folds`, `_build_team_labels` had zero unit tests — added 12 new tests [tests/unit/test_dashboard_filters.py]
- [x] [AI-Review][MEDIUM] Magic number `136` (day_num) — extracted to `_ROUND_OF_64_DAY_NUM` constant [dashboard/lib/filters.py]
- [x] [AI-Review][MEDIUM] Hardcoded `"seed_diff_bonus"` string check for scoring constructor — replaced with `inspect.signature(cls)` [dashboard/lib/filters.py]
- [x] [AI-Review][MEDIUM] `@st.cache_resource` used for serializable `BracketSimulationResult` — changed to `@st.cache_data` [dashboard/lib/filters.py]
- [x] [AI-Review][LOW] Double-cast `float(str(d["Expected Points"]))` in EP table sort — simplified to `float(d["Expected Points"])` [dashboard/pages/2_Presentation.py]
- [x] [AI-Review][LOW] Module docstring claimed all functions use `@st.cache_data` — updated to reflect `@st.cache_data` usage [dashboard/lib/filters.py]
- [x] [AI-Review][LOW] No test for mirrored right-region layout — added `test_right_regions_are_mirrored` [tests/unit/test_bracket_renderer.py]

#### Review Follow-ups (AI) — Applied by Code Review (Round 2)
- [x] [AI-Review][HIGH] AC #4 (MC score distribution) had zero test coverage — added `TestMCModeRender` with 2 tests [tests/unit/test_bracket_page.py]
- [x] [AI-Review][MEDIUM] `ep_data` typed as `dict[str, object]` required `type: ignore[arg-type]` on sort — changed annotation to `list[dict[str, str | float]]` [dashboard/pages/2_Presentation.py:86]
- [x] [AI-Review][MEDIUM] AC #5 pairwise expander had no assertion in `TestSuccessfulRender` — added `mock_st.expander.assert_called()` check [tests/unit/test_bracket_page.py]
- [x] [AI-Review][LOW] No user feedback when MC mode's scoring key absent from `bracket_distributions` — added `st.info()` message [dashboard/pages/2_Presentation.py:103]
- [ ] [AI-Review][MEDIUM] `n_simulations` is a cache key for analytical mode causing spurious cache misses; `scoring_name` as cache key causes full re-simulation when switching scoring rules — architectural refactor (split cache into prob-matrix/bracket layer and scoring layer) deferred to future story

## Dev Notes

### Architecture & Data Flow

**Simulation Pipeline (run on first page load, then cached):**
```
TourneySeedTable.from_csv(kaggle_csv_path)
    → seeds = table.all_seeds(season)
    → bracket = build_bracket(seeds, season)
    → model = RunStore(data_dir).load_model(run_id)
    → provider = EloProvider(model) or MatrixProvider(P, team_ids)
    → P = build_probability_matrix(provider, bracket.team_ids, context)
    → sim_result = simulate_tournament(bracket, provider, context, scoring_rules, method)
    → most_likely = compute_most_likely_bracket(bracket, P)
```

**Key challenge: Loading the Kaggle CSV for seeds.**
The `TourneySeedTable.from_csv()` expects the raw `MNCAATourneySeeds.csv` file. This CSV is downloaded by the sync CLI to the data directory. The dev agent must locate where `sync.py` stores the Kaggle CSVs. Check `src/ncaa_eval/ingest/connectors/` for the `KaggleConnector` to find the raw CSV path. Likely stored at `{data_dir}/kaggle/` or `{data_dir}/raw/`. If the CSV isn't directly accessible, the dev may need to add a `get_tourney_seeds()` method to `ParquetRepository` or read seeds from the games parquet (which has `w_team_id` / `l_team_id` but not seed info).

**Key challenge: Building a ProbabilityProvider from a saved model.**
The `RunStore.load_model(run_id)` returns a `Model` ABC instance. For Elo models, wrap with `EloProvider(model)`. For XGBoost/stateless models, the provider needs a feature matrix — this may require the full feature-serving pipeline. **Simplest MVP approach**: for Elo, use `EloProvider`; for XGBoost, load fold predictions from `fold_predictions.parquet` and construct a `MatrixProvider` from the game-level predicted probabilities for tournament games only. This avoids re-running the full feature pipeline in the dashboard.

**Caching strategy:**
- `@st.cache_data(ttl=300)` for seed/team loading (changes rarely)
- `@st.cache_data(ttl=None)` for simulation results (deterministic given same inputs; invalidated by key change)
- Analytical simulation is fast (< 100ms for 64 teams) — no progress bar needed
- MC simulation (10K+ sims) takes 1-5s — use `st.spinner()` during computation

### Bracket Tree Rendering Approach

**Use `st.components.html()` with embedded HTML/CSS** — NOT a full custom React/D3 component. Rationale:
- No npm build step, no `package.json`, no frontend toolchain
- The bracket is a static render from simulation results (no bidirectional communication needed)
- CSS flexbox/grid layouts can render 64-team brackets cleanly
- Proven approach: see [Responsive NCAA Tournament Bracket (CodePen)](https://codepen.io/COLTstreet/pen/JxrPba) and [CSS Grid Bracket (CodePen)](https://codepen.io/webcraftsman/pen/XWPYEZv)

**Layout structure:**
```
┌───────────────────────────────────────────────────────────────┐
│  Left side: Region W (top) + Region X (bottom)               │
│  Center: Final Four + Championship                            │
│  Right side: Region Y (top) + Region Z (bottom)              │
└───────────────────────────────────────────────────────────────┘
```

Each game node shows: `[seed] Team Name (XX.X%)` with background color scaled by probability (green for favorites, red for upsets, neutral for ~50%).

**CSS must use dark-mode colors** matching `.streamlit/config.toml` theme:
- Background: dark (`#0E1117` or similar Streamlit dark)
- Text: white/light gray
- Green: `#28a745` / Red: `#dc3545` / Neutral: `#6c757d`

### Existing Code to Reuse (DO NOT REIMPLEMENT)

| Function | Module | Purpose |
|---|---|---|
| `build_bracket()` | `ncaa_eval.evaluation.simulation` | Construct 64-team bracket tree from seeds |
| `build_probability_matrix()` | `ncaa_eval.evaluation.simulation` | Build n×n pairwise win-prob matrix |
| `compute_advancement_probs()` | `ncaa_eval.evaluation.simulation` | Phylourny exact advancement probabilities |
| `compute_expected_points()` | `ncaa_eval.evaluation.simulation` | `adv_probs @ points_vector` |
| `compute_most_likely_bracket()` | `ncaa_eval.evaluation.simulation` | Greedy most-likely bracket traversal |
| `simulate_tournament()` | `ncaa_eval.evaluation.simulation` | High-level orchestrator (analytical + MC) |
| `simulate_tournament_mc()` | `ncaa_eval.evaluation.simulation` | Vectorized MC engine |
| `plot_advancement_heatmap()` | `ncaa_eval.evaluation.plotting` | Plotly heatmap of advancement probs |
| `plot_score_distribution()` | `ncaa_eval.evaluation.plotting` | Plotly histogram of bracket scores |
| `MatrixProvider` / `EloProvider` | `ncaa_eval.evaluation.simulation` | Probability provider wrappers |
| `get_scoring()` / `list_scorings()` | `ncaa_eval.evaluation.simulation` | Scoring rule registry |
| `RunStore.load_model()` | `ncaa_eval.model.tracking` | Load trained model artifact |
| `RunStore.list_runs()` | `ncaa_eval.model.tracking` | List available model runs |
| `TourneySeedTable.from_csv()` | `ncaa_eval.transform.normalization` | Parse seeds CSV |
| `TourneySeedTable.all_seeds()` | `ncaa_eval.transform.normalization` | Get seeds for a season |

### Data Structures Reference

**`SimulationResult` fields used by this page:**
- `advancement_probs`: `ndarray(n_teams, n_rounds)` — feeds heatmap
- `expected_points`: `dict[str, ndarray(n_teams,)]` — feeds EP table
- `bracket_distributions`: `dict[str, BracketDistribution]` — feeds score distribution (MC only)
- `method`: `"analytical"` or `"monte_carlo"` — controls which sections render

**`MostLikelyBracket` fields:**
- `winners`: `tuple[int, ...]` — team indices in round-major order (games 0-31 = R64, 32-47 = R32, etc.)
- `champion_team_id`: Canonical team ID of predicted champion
- `log_likelihood`: Sum of log-probs

**`BracketStructure` fields:**
- `team_ids`: `tuple[int, ...]` — 64 team IDs in bracket-position order (leaf order)
- `team_index_map`: `dict[int, int]` — team_id → index
- `seed_map`: `dict[int, int]` — team_id → seed number

**Round indexing:** 0=R64, 1=R32, 2=S16, 3=E8, 4=F4, 5=Championship.
**Round labels:** `("R64", "R32", "S16", "E8", "F4", "Championship")` — defined in `plotting.py` as `_ROUND_LABELS`.

### Project Structure Notes

**New files:**
- `dashboard/lib/bracket_renderer.py` — HTML/CSS bracket tree renderer (pure function, no Streamlit imports)

**Modified files:**
- `dashboard/lib/filters.py` — add seed loading, team name loading, simulation orchestration
- `dashboard/pages/2_Presentation.py` — replace placeholder with full implementation

**Test files:**
- `tests/unit/test_bracket_renderer.py` — new
- `tests/unit/test_dashboard_filters.py` — extend with new loader tests
- `tests/unit/test_bracket_page.py` — new page rendering tests

### Game Theory Sliders — OUT OF SCOPE

Story 7.7 (spike) will research the mathematical mechanism for probability perturbation. **Do NOT implement sliders in this story.** The bracket visualizer should display the model's raw probabilities. A future story will add the slider mechanism after the spike defines the transformation.

### Testing Standards

- Mock `RunStore`, `TourneySeedTable`, and simulation functions in page tests
- Use `unittest.mock.patch` for `st.plotly_chart`, `st.components.html` assertions
- Test empty states (no model selected, no seeds for year, simulation failure)
- Bracket renderer: test with a small 4-team or 8-team bracket for simplicity, verify HTML contains expected team names and probabilities
- Follow existing test patterns from `tests/unit/test_leaderboard_page.py` and `tests/unit/test_deep_dive_page.py`

### References

- [Source: specs/04-front-end-spec.md#4.1] — "Interactive Bracket Tree: Custom Streamlit Component (D3.js or Mermaid.js wrapper)" — we're using `st.components.html()` with CSS instead (simpler, no build step, spec intent preserved)
- [Source: specs/04-front-end-spec.md#3.1] — "Visualize Bracket" flow: model probability matrix → bracket visualizer
- [Source: specs/04-front-end-spec.md#5.1] — Desktop only, wide mode, < 500ms interaction response
- [Source: specs/04-front-end-spec.md#4.2] — Dark mode, green/red/neutral palette, monospace fonts
- [Source: src/ncaa_eval/evaluation/simulation.py] — Full simulation engine API
- [Source: src/ncaa_eval/evaluation/plotting.py] — `plot_advancement_heatmap()`, `plot_score_distribution()`
- [Source: src/ncaa_eval/model/tracking.py] — `RunStore.load_model()` for trained model persistence
- [Source: dashboard/pages/3_Model_Deep_Dive.py] — Pattern for page layout, breadcrumbs, data loading
- [Source: dashboard/pages/1_Lab.py] — Pattern for session state, empty states, KPI metrics

### Previous Story Intelligence (7.4)

- `RunStore.load_model()` was added in Story 7.4 — it returns a `Model` ABC instance (`XGBoost` or `Elo`)
- Fold predictions are stored per-run in `fold_predictions.parquet` — contains per-game `pred_win_prob` which could be reused for building a `MatrixProvider` if needed
- Breadcrumb pattern: `st.caption(f"Home > Lab > {label}")` — adapt for Presentation section
- Page rendering function pattern: define `_render_*()` functions, call at module level
- Session state pattern: `st.session_state.get("selected_year")` / `st.session_state.get("selected_run_id")`

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

### Completion Notes List

- Used `@st.cache_data(ttl=None)` for simulation results — correct decorator for serializable dataclass outputs (code review corrected the original `@st.cache_resource` choice)
- For XGBoost models, builds `MatrixProvider` from `fold_predictions.parquet` filtered to tournament season — avoids re-running full feature pipeline in dashboard; matrix fill uses vectorized `.map()` + numpy indexing (no `iterrows()`)
- Bracket renderer uses 64-team hardcoded layout (6 rounds) matching NCAA tournament structure — not generalized for arbitrary bracket sizes
- HTML escaping applied to all team labels to prevent XSS via crafted team names
- Extracted `_render_results()` from `_render_bracket_page()` to keep complexity under C901 threshold
- Extracted `_build_provider_from_folds()` and `_build_team_labels()` helpers from `run_bracket_simulation()` for same reason
- Scoring instantiation uses `inspect.signature(cls)` to detect `seed_map` parameter — avoids hardcoded `if scoring_name == "seed_diff_bonus"` check
- AC #5 pairwise win probability expander added to `_render_results()` — selectbox pair lets user choose any two teams and see head-to-head probability

### File List

**New files:**
- `dashboard/lib/bracket_renderer.py` — HTML/CSS bracket tree renderer (315 lines)
- `tests/unit/test_bracket_renderer.py` — 14 tests for bracket renderer
- `tests/unit/test_bracket_page.py` — 5 tests for page rendering

**Modified files:**
- `dashboard/lib/filters.py` — added `load_tourney_seeds`, `load_team_names`, `BracketSimulationResult`, `run_bracket_simulation` + helpers
- `dashboard/pages/2_Presentation.py` — replaced placeholder with full bracket visualizer implementation; CR2 fixes: `ep_data` typing, MC missing-scoring feedback
- `tests/unit/test_dashboard_filters.py` — added `TestLoadTourneySeeds` (3 tests) and `TestLoadTeamNames` (3 tests)
- `tests/unit/test_bracket_page.py` — CR2: added `TestMCModeRender` (2 tests), expander assertion in `TestSuccessfulRender`
- `_bmad-output/planning-artifacts/template-requirements.md` — added Story 7.5 template learnings
