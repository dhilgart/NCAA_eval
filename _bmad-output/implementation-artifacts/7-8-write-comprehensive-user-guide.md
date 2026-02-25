# Story 7.8: Write Comprehensive User Guide

Status: ready-for-dev

## Story

As a data scientist,
I want a comprehensive guide explaining the evaluation metrics, model types, and how to interpret the results,
so that I can understand what the platform measures and make informed decisions based on its outputs.

## Acceptance Criteria

1. **Metrics Explained**: All evaluation metrics are explained (Log Loss, Brier Score, ROC-AUC, ECE) with intuitive descriptions and examples showing what "good" vs "bad" values look like.

2. **Model Types Documented**: Model types are documented (Stateful vs. Stateless) with guidance on when to use each, including the reference implementations (Elo, XGBoost).

3. **Result Interpretation Covered**: How to read reliability diagrams, what calibration means in plain language, and how to use bracket simulations for pool strategy.

4. **Tournament Scoring Explained**: The tournament scoring systems are explained (Standard 1-2-4-8-16-32, Fibonacci 1-1-2-3-5-8, Seed-Difference Bonus) with worked examples showing point calculations.

5. **Sphinx-Compatible & Integrated**: The guide is written in Markdown (MyST-compatible) and integrated into the Sphinx documentation site via `docs/index.rst`.

6. **Accessible from Documentation Site**: The guide is accessible from the project's documentation site when built with `nox -s docs`.

## Tasks / Subtasks

- [ ] Task 1: Create the user guide document (AC: #1, #2, #3, #4, #5)
  - [ ] 1.1: Write the "Getting Started" overview section (quick-start: sync, train, evaluate, dashboard)
  - [ ] 1.2: Write the "Evaluation Metrics" section (Log Loss, Brier Score, ROC-AUC, ECE — with intuitive explanations, formulas, and "good vs bad" value ranges)
  - [ ] 1.3: Write the "Model Types" section (Stateful vs Stateless, Elo reference, XGBoost reference, Model ABC plugin system)
  - [ ] 1.4: Write the "Interpreting Results" section (reliability diagrams, calibration, over-confidence vs under-confidence)
  - [ ] 1.5: Write the "Tournament Simulation" section (Monte Carlo methodology, bracket distribution, Expected Points, confidence intervals)
  - [ ] 1.6: Write the "Tournament Scoring" section (Standard, Fibonacci, Seed-Difference Bonus — with worked examples and point tables)
  - [ ] 1.7: Write the "Dashboard Guide" section (Lab pages: Leaderboard, Model Deep Dive; Presentation pages: Bracket Visualizer, Pool Scorer — what each page shows and how to use it)
  - [ ] 1.8: Write the "Game Theory Sliders" subsection (Upset Aggression, Seed-Weight — what they do mathematically in plain language, practical usage tips)
- [ ] Task 2: Integrate into Sphinx documentation (AC: #5, #6)
  - [ ] 2.1: Add `user-guide` to the `docs/index.rst` toctree (new "User Guide" section)
  - [ ] 2.2: Verify the guide builds cleanly with `nox -s docs` (no warnings/errors)
- [ ] Task 3: Quality gates (AC: all)
  - [ ] 3.1: Run `ruff check .` — pass
  - [ ] 3.2: Run `mypy --strict src/ncaa_eval tests dashboard` — pass (no changes to Python code expected, but verify no regressions)
  - [ ] 3.3: Run `pytest` — pass (no test changes expected, but verify no regressions)
  - [ ] 3.4: Verify `nox -s docs` builds without errors

## Dev Notes

### Content Scope — What to Write vs What Exists

**This story creates ONE new file:** `docs/user-guide.md` — a comprehensive end-user guide for the NCAA_eval platform.

**What already exists (DO NOT duplicate):**
- `README.md` — installation, Kaggle API setup, basic data sync commands
- `CONTRIBUTING.md` — developer workflow (fork, setup, branch, PR)
- `docs/STYLE_GUIDE.md` — coding standards (internal dev reference)
- `docs/TESTING_STRATEGY.md` — testing approach (internal dev reference)
- `docs/testing/*.md` — 7 detailed testing guides (internal dev reference)
- `docs/api/` — auto-generated API reference from docstrings

**The user guide bridges the gap between README (installation) and API Reference (code docs).** It answers: "I've installed the project and synced data — now what do I do and how do I interpret what I see?"

### Target Audience

The user guide is for a **data scientist** who:
- Has already installed the project (per README)
- Has synced data via CLI (`python sync.py --source kaggle`)
- Wants to train models, evaluate them, and make bracket picks
- Needs to understand what the metrics and visualizations mean
- May or may not be familiar with March Madness pool strategy

### Document Format — MyST Markdown

The project uses `myst_parser` in Sphinx (see `docs/conf.py`). Write the guide in **Markdown** (`.md`), NOT reStructuredText (`.rst`). MyST handles Markdown natively.

**File path:** `docs/user-guide.md`

MyST-specific features available:
- Standard Markdown headings, lists, tables, code blocks
- Cross-references: `` {ref}`label` `` or standard `[text](relative-path.md)` links
- Admonitions: `````{note}````` / `````{warning}````` / `````{tip}````` blocks
- Math: `$inline$` and `$$display$$` for metric formulas

### Sphinx Integration — index.rst Changes

Add a new toctree section to `docs/index.rst`:

```rst
.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   user-guide
```

Place this **before** the "Developer Guides" section so end-user content appears first.

### Evaluation Metrics — What to Explain

All metrics are implemented in `src/ncaa_eval/evaluation/metrics.py`:

1. **Log Loss** (`sklearn.metrics.log_loss`) — measures how well predicted probabilities match actual outcomes. Perfect = 0.0, random baseline = 0.693 (ln(2)). Lower is better. Penalizes confident wrong predictions severely (log scale).

2. **Brier Score** (`sklearn.metrics.brier_score_loss`) — mean squared error of probability predictions. Perfect = 0.0, random baseline = 0.25. Lower is better. More forgiving of confident wrong predictions than log loss.

3. **ROC-AUC** (`sklearn.metrics.roc_auc_score`) — discrimination ability: can the model distinguish winners from losers? Perfect = 1.0, random = 0.5. Higher is better. Does NOT measure calibration.

4. **ECE** (Expected Calibration Error, custom numpy implementation) — measures how well predicted probabilities correspond to actual win rates. Perfect = 0.0. Example: if model says "70% win probability" for 100 games, about 70 should actually be wins. Lower is better.

5. **Reliability Diagrams** — visual representation of calibration. X-axis = predicted probability (binned), Y-axis = actual win rate in that bin. Perfect calibration = diagonal line. Points above diagonal = under-confident, below = over-confident.

### Model Types — What to Explain

Defined in `src/ncaa_eval/model/base.py`:

1. **Stateful Models** (`StatefulModel` subclass) — maintain internal state that updates game-by-game through a season. Example: `EloModel` (`src/ncaa_eval/model/elo.py`) — team ratings evolve with each game result. Best for: capturing in-season trajectory and momentum.

2. **Stateless Models** (direct `Model` subclass) — standard batch training on feature matrices. Example: `XGBoostModel` (`src/ncaa_eval/model/xgboost_model.py`) — gradient-boosted trees trained on feature snapshots. Best for: combining many feature dimensions, high accuracy when features are strong.

3. **Plugin Registry** (`src/ncaa_eval/model/registry.py`) — models register via `@register_model("name")`. Users can create custom models by subclassing `Model` or `StatefulModel`.

### Tournament Scoring — What to Explain

Defined in `src/ncaa_eval/evaluation/simulation.py` (scoring classes):

| Round | Standard | Fibonacci | Seed-Diff Bonus |
|-------|----------|-----------|-----------------|
| R64   | 1        | 1         | 1 + seed_diff   |
| R32   | 2        | 1         | 2 + seed_diff   |
| S16   | 4        | 2         | 4 + seed_diff   |
| E8    | 8        | 3         | 8 + seed_diff   |
| F4    | 16       | 5         | 16 + seed_diff  |
| Championship | 32 | 8     | 32 + seed_diff  |

**Worked example** (Standard scoring): If your bracket correctly picks 20 R64 games, 10 R32 games, 4 S16 games, 2 E8 games, 1 F4, and the champion: `20*1 + 10*2 + 4*4 + 2*8 + 1*16 + 1*32 = 20 + 20 + 16 + 16 + 16 + 32 = 120 points`.

### Dashboard Pages — What to Explain

**Lab Pages:**
- **Backtest Leaderboard** (`dashboard/pages/1_Lab.py`) — sortable table of model runs with Log Loss, Brier, ROC-AUC, ECE. Diagnostic cards with deltas. Click to drill down.
- **Model Deep Dive** (`dashboard/pages/3_Model_Deep_Dive.py`) — reliability diagram, metric drill-down by year/round/seed, feature importance (XGBoost).

**Presentation Pages:**
- **Bracket Visualizer** (`dashboard/pages/2_Presentation.py`) — 64-team bracket tree, clickable matchups, Game Theory sliders (Upset Aggression, Seed-Weight), advancement heatmap.
- **Pool Scorer** (`dashboard/pages/4_Pool_Scorer.py`) — configure scoring rules, run MC simulation, view point distribution, export CSV submission.

### Game Theory Sliders — Plain Language

From Story 7.7 spike (`specs/research/game-theory-slider-mechanism.md`):

- **Upset Aggression** (range -5 to +5, default 0): Controls how much probabilities are pushed toward 50/50. Positive values = more upsets in your bracket picks. Negative values = more chalk (favor the favorites). Mathematically: power transform `p' = p^alpha / (p^alpha + (1-p)^alpha)` where `alpha = 2^(-v/3)`.

- **Seed-Weight** (range 0% to 100%, default 0%): Blends model predictions with historical seed-vs-seed win rates. At 0% = pure model. At 100% = pure historical seed performance. Useful when you trust seed-based priors more than the model for certain matchups.

**Note:** Story 7.5 deferred slider implementation (sliders are NOT yet in the codebase). The user guide should describe the concept and note the feature as "coming soon" or describe it based on the spike research document. Check the current state of `dashboard/pages/2_Presentation.py` before writing — if sliders have been implemented by a later story, document them as functional.

### Project Structure Notes

- **New file:** `docs/user-guide.md`
- **Modified file:** `docs/index.rst` (add toctree entry)
- No Python code changes — this is a documentation-only story
- No test changes expected — run quality gates to verify no regressions

### Writing Style Guidelines

- **Plain language first, formulas second.** Lead with intuition ("Log Loss punishes confident wrong predictions more than anything else"), then show the formula for reference.
- **Use concrete examples.** "A Log Loss of 0.55 means your model is better than random (0.693) but has room to improve toward a well-calibrated model (~0.45-0.50)."
- **Use MyST admonition blocks** for tips and warnings: `````{tip}````` for practical advice, `````{warning}````` for common pitfalls.
- **Include a table of contents** at the top via a MyST `{contents}` directive.
- **Keep sections scannable.** Use bullet points and tables over long paragraphs.
- **Cross-reference the API docs** where appropriate: "For implementation details, see the [API Reference](api/modules.rst)."

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 7.8] — AC definitions
- [Source: src/ncaa_eval/evaluation/metrics.py] — Metric implementations (Log Loss, Brier, ROC-AUC, ECE)
- [Source: src/ncaa_eval/evaluation/simulation.py] — Tournament simulation and scoring classes
- [Source: src/ncaa_eval/model/base.py] — Model ABC, StatefulModel, StatelessModel definitions
- [Source: src/ncaa_eval/model/elo.py] — Elo reference model
- [Source: src/ncaa_eval/model/xgboost_model.py] — XGBoost reference model
- [Source: src/ncaa_eval/model/registry.py] — Plugin registry
- [Source: dashboard/pages/1_Lab.py] — Backtest Leaderboard page
- [Source: dashboard/pages/2_Presentation.py] — Bracket Visualizer page
- [Source: dashboard/pages/3_Model_Deep_Dive.py] — Model Deep Dive page
- [Source: dashboard/pages/4_Pool_Scorer.py] — Pool Scorer page
- [Source: dashboard/lib/filters.py] — Dashboard filter utilities and simulation orchestration
- [Source: specs/research/game-theory-slider-mechanism.md] — Game Theory slider mechanism research
- [Source: docs/conf.py] — Sphinx configuration (Furo theme, MyST parser, autodoc + napoleon)
- [Source: docs/index.rst] — Current toctree structure (Developer Guides, Testing Guides, API Reference)
- [Source: README.md] — Existing installation and data sync documentation
- [Source: CONTRIBUTING.md] — Existing developer workflow documentation

### Previous Story Intelligence (Story 7.7)

- Story 7.7 was a spike (research-only, no code). It produced `specs/research/game-theory-slider-mechanism.md`.
- The spike recommends a two-slider approach (Upset Aggression + Seed-Weight) with an optional three-slider variant (adding Chalk Bias).
- Slider implementation was deferred from Story 7.5. The sliders may or may not be implemented when this story is developed — check the codebase.
- All dashboard pages (7.1-7.6) are done. The dashboard is fully functional.
- Quality gates at Story 7.7: 865 tests passed, 1 skipped.
- Commit style: `feat(docs): ...` or `docs(guide): ...` — use conventional commits.

### Git Intelligence

Recent Epic 7 commits:
- `cf522ee` — Research game theory slider mechanism spike (Story 7.7)
- `672cc83` — feat(dashboard): Pool Scorer page with MC outcome analysis and CSV export
- `21b52d8` — feat(dashboard): Build Presentation Page — Bracket Visualizer (Story 7.5)
- `2cd6d92` — feat(dashboard): Model Deep Dive page with reliability diagrams and feature importance
- `f84327d` — feat(epic-7): Build Lab Page — Backtest Leaderboard (Story 7.3)
- `04eb940` — Build Streamlit app shell with navigation and global filters (Story 7.2)

Pattern: all dashboard stories are complete and merged. This is a documentation-only story.

## Dev Agent Record

### Agent Model Used

{{agent_model_name_version}}

### Debug Log References

### Completion Notes List

### File List
