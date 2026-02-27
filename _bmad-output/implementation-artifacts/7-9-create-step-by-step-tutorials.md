# Story 7.9: Create Step-by-Step Tutorials

Status: ready-for-dev

## Story

As a data scientist,
I want step-by-step tutorials for common tasks,
so that I can quickly learn how to use the platform's key workflows.

## Acceptance Criteria

1. **Getting Started Tutorial**: A "Getting Started" tutorial covers the full pipeline: sync data, train a model, evaluate, and view results in the dashboard. Includes runnable code examples and expected outputs.

2. **Custom Model Tutorial**: A "How to Create a Custom Model" tutorial walks through subclassing the Model ABC, registering via the plugin registry, and running evaluation. Includes runnable code examples and expected outputs.

3. **Custom Metric Tutorial**: A "How to Add a Custom Metric" tutorial demonstrates extending the evaluation engine via the plugin registry. Includes runnable code examples and expected outputs.

4. **Sphinx Integration**: Tutorials are written in Sphinx-compatible Markdown (MyST) and integrated into the auto-generated documentation via `docs/index.rst`.

5. **TOC Directive Fix**: The `{contents}` directive is removed from `docs/user-guide.md` (conflicts with Furo's built-in right-sidebar TOC). All other documentation files are searched for `{contents}` directives that may also need removal.

6. **README Enhancement**: The project `README.md` is reviewed and enhanced:
   - Must include a link to the GitHub Pages documentation site (`https://dhilgart.github.io/NCAA_eval/`)
   - Add appropriate status badges (docs, version, Python version, license)
   - Review what else should be added (quick feature overview, dashboard screenshot placeholder, model training quickstart) and what should be removed
   - Ensure all existing links work correctly

## Tasks / Subtasks

- [ ] Task 1: Create the Getting Started tutorial (AC: #1, #4)
  - [ ] 1.1: Write `docs/tutorials/getting-started.md` — full pipeline walkthrough: install → sync → train Elo → train XGBoost → evaluate → launch dashboard → interpret results
  - [ ] 1.2: Include CLI command examples with expected output snippets (trimmed for readability)
  - [ ] 1.3: Include code examples showing how to use the Python API directly (import model, fit, predict_proba)
  - [ ] 1.4: Document the dashboard launch command and how to navigate each page

- [ ] Task 2: Create the Custom Model tutorial (AC: #2, #4)
  - [ ] 2.1: Write `docs/tutorials/custom-model.md` — step-by-step guide to creating a custom model
  - [ ] 2.2: Show a complete working example: a simple custom stateless model (e.g., weighted average of features → probability) subclassing `Model` directly
  - [ ] 2.3: Show a complete working example: a simple custom stateful model subclassing `StatefulModel` — implement `update()`, `_predict_one()`, `start_season()`, `get_state()`, `set_state()`
  - [ ] 2.4: Show plugin registry usage: `@register_model("my_model")` decorator, then `python -m ncaa_eval.cli train --model my_model`
  - [ ] 2.5: Document how to save/load custom models (`save(path)`, `load(path)`)
  - [ ] 2.6: Document how to run evaluation/backtest with the custom model

- [ ] Task 3: Create the Custom Metric tutorial (AC: #3, #4)
  - [ ] 3.1: Write `docs/tutorials/custom-metric.md` — step-by-step guide to extending evaluation
  - [ ] 3.2: Show how to write a custom metric function that accepts `(y_true, y_prob)` numpy arrays
  - [ ] 3.3: Show how to integrate the custom metric into the backtest pipeline
  - [ ] 3.4: Show how to create a custom tournament scoring rule by subclassing `ScoringRule`

- [ ] Task 4: Integrate tutorials into Sphinx docs (AC: #4)
  - [ ] 4.1: Add a new "Tutorials" toctree section to `docs/index.rst` — place between "User Guide" and "Developer Guides"
  - [ ] 4.2: List all three tutorials in the toctree: `tutorials/getting-started`, `tutorials/custom-model`, `tutorials/custom-metric`
  - [ ] 4.3: Verify `nox -s docs` builds cleanly (no warnings/errors beyond suppressed ones)

- [ ] Task 5: Fix `{contents}` TOC directive (AC: #5)
  - [ ] 5.1: Remove the `{contents}` directive block (lines 3-6) from `docs/user-guide.md` — Furo theme provides built-in right-sidebar TOC
  - [ ] 5.2: Search ALL `.md` and `.rst` files under `docs/` for other `{contents}` directives; remove any found
  - [ ] 5.3: Verify user guide still renders correctly with `nox -s docs`

- [ ] Task 6: Enhance README.md (AC: #6)
  - [ ] 6.1: Add documentation badge linking to `https://dhilgart.github.io/NCAA_eval/`
  - [ ] 6.2: Add Python version badge (3.12+)
  - [ ] 6.3: Add a brief "Features" section summarizing the platform's capabilities (data ingestion, feature engineering, model training, evaluation, tournament simulation, interactive dashboard)
  - [ ] 6.4: Add a "Documentation" section linking to the full docs site, user guide, and tutorials
  - [ ] 6.5: Add a "Quick Start" model training section showing `python -m ncaa_eval.cli train --model elo` and dashboard launch (`streamlit run dashboard/app.py`)
  - [ ] 6.6: Review and fix any broken links (e.g., `contributing.md` relative path)
  - [ ] 6.7: Remove "Created from cookiecutter" line if it's not useful for end users

- [ ] Task 7: Quality gates (AC: all)
  - [ ] 7.1: Run `ruff check .` — pass
  - [ ] 7.2: Run `mypy --strict src/ncaa_eval tests dashboard` — pass (no Python code changes expected)
  - [ ] 7.3: Run `pytest` — pass (no test changes expected)
  - [ ] 7.4: Verify `nox -s docs` builds without errors
  - [ ] 7.5: Manually verify tutorial code examples are accurate against the actual codebase API

## Dev Notes

### Content Scope — What to Write vs What Exists

**This story creates 3 new tutorial files and modifies 3 existing files:**
- NEW: `docs/tutorials/getting-started.md`
- NEW: `docs/tutorials/custom-model.md`
- NEW: `docs/tutorials/custom-metric.md`
- MODIFIED: `docs/index.rst` (add Tutorials toctree section)
- MODIFIED: `docs/user-guide.md` (remove `{contents}` directive)
- MODIFIED: `README.md` (enhance with badges, features, docs links)

**What already exists (DO NOT duplicate):**
- `docs/user-guide.md` — comprehensive end-user guide (metrics, models, dashboard, Game Theory sliders). Tutorials should REFERENCE the user guide, not repeat its content.
- `README.md` — installation, Kaggle API setup, sync commands. Tutorials should reference the README for setup steps.
- `CONTRIBUTING.md` — developer workflow for contributing code.
- `docs/STYLE_GUIDE.md` — coding standards.
- `docs/TESTING_STRATEGY.md` — testing approach.

**The tutorials fill the "hands-on practice" gap.** The user guide explains concepts; tutorials provide step-by-step workflows with runnable code.

### Target Audience

Same as user guide: a **data scientist** who has installed the project and wants to learn by doing. Tutorials should be "follow along" documents — each step should be executable.

### Document Format — MyST Markdown

Same as user guide: use MyST Markdown (`.md`). MyST features available:
- Standard Markdown headings, lists, tables, code blocks
- Admonitions: `````{note}````` / `````{tip}````` / `````{warning}`````
- Cross-references via `[text](relative-path.md)` links

**Do NOT use `{contents}` directive** — Furo theme provides built-in right-sidebar TOC.

**File paths:**
- `docs/tutorials/getting-started.md`
- `docs/tutorials/custom-model.md`
- `docs/tutorials/custom-metric.md`

### Model ABC — Key Interfaces for Tutorial

The custom model tutorial must demonstrate these interfaces accurately:

**Stateless model** (subclass `Model` directly):
```python
from ncaa_eval.model.base import Model, ModelConfig
from ncaa_eval.model.registry import register_model

class MyConfig(ModelConfig):
    my_param: float = 1.0

@register_model("my_model")
class MyModel(Model):
    def __init__(self, config: MyConfig | None = None) -> None: ...
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...
    def predict_proba(self, X: pd.DataFrame) -> pd.Series: ...
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> Self: ...
    def get_config(self) -> ModelConfig: ...
```

**Stateful model** (subclass `StatefulModel`):
```python
from ncaa_eval.model.base import StatefulModel, ModelConfig
from ncaa_eval.ingest.schema import Game

@register_model("my_stateful")
class MyStatefulModel(StatefulModel):
    def update(self, game: Game) -> None: ...
    def _predict_one(self, team_a_id: int, team_b_id: int) -> float: ...
    def start_season(self, season: int) -> None: ...
    def get_state(self) -> dict[str, Any]: ...
    def set_state(self, state: dict[str, Any]) -> None: ...
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> Self: ...
    def get_config(self) -> ModelConfig: ...
```

**CRITICAL**: Before writing the tutorial, READ the actual source files to verify signatures:
- `src/ncaa_eval/model/base.py` — Model ABC, StatefulModel
- `src/ncaa_eval/model/registry.py` — @register_model decorator
- `src/ncaa_eval/model/elo.py` — reference StatefulModel implementation
- `src/ncaa_eval/model/xgboost_model.py` — reference stateless Model implementation
- `src/ncaa_eval/model/logistic_regression.py` — minimal stateless example (~30 lines)

### Evaluation Extension — Key Interfaces for Tutorial

The custom metric tutorial must demonstrate:

**Custom metric function:**
```python
import numpy as np

def my_metric(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Custom metric matching the (y_true, y_prob) -> float contract."""
    ...
```

**Custom scoring rule** (subclass `ScoringRule`):
```python
from ncaa_eval.evaluation.simulation import ScoringRule

class MyScoring(ScoringRule):
    def score_correct_pick(self, round_num: int, seed_a: int, seed_b: int) -> float:
        ...
```

**CRITICAL**: Before writing the tutorial, READ the actual source files:
- `src/ncaa_eval/evaluation/metrics.py` — existing metric implementations
- `src/ncaa_eval/evaluation/simulation.py` — ScoringRule ABC and implementations
- `src/ncaa_eval/evaluation/backtest.py` — backtest pipeline entry point

### `{contents}` Directive — Why Remove It

The `{contents}` MyST directive generates an inline table of contents at the top of the page. However, the Furo Sphinx theme already provides a **built-in right-sidebar TOC** that auto-generates from headings. Having both is redundant and the inline `{contents}` block conflicts visually with Furo's sidebar navigation.

**Current state in `docs/user-guide.md`** (lines 3-6):
```markdown
```{contents}
:depth: 2
:local:
```
```

Remove these 4 lines entirely. No replacement needed — Furo's sidebar handles it.

Search all other docs files for similar directives. As of the last analysis, `user-guide.md` is the only file using `{contents}`.

### README Enhancement — What to Add

**Current README state:**
- 3 badges: PRs Welcome, Conventional Commits, GitHub Actions CI
- Basic: description, prerequisites, Kaggle auth, `poetry install`, sync commands, contributing link

**What to ADD:**
- Badge: Documentation site link (`https://dhilgart.github.io/NCAA_eval/`)
- Badge: Python 3.12+ version
- Brief "Features" section (1-2 paragraphs or bullet list)
- "Documentation" section linking to docs site, user guide, tutorials
- "Quick Start" section for model training + dashboard (beyond just data sync)
- Dashboard launch command: `poetry run streamlit run dashboard/app.py`

**What to FIX:**
- `[Contributing](contributing.md)` link — verify this resolves correctly (file is `CONTRIBUTING.md`, case-sensitive on Linux). Previous story 7.8 completion notes mention "fixed broken README relative link" — verify the fix is in place.

**What to CONSIDER removing:**
- "Created from cookiecutter-python-template" line at bottom — not useful for end users of this specific project

### Project Structure Notes

- All new files go under `docs/tutorials/` (create directory if needed)
- No Python code changes — this is a documentation-only story (plus README)
- No test changes expected — run quality gates to verify no regressions

### Writing Style Guidelines

- **Follow-along format.** Each tutorial should be structured as numbered steps the user can execute sequentially.
- **Show expected output.** After each command or code block, show trimmed expected output so the user can verify success.
- **Use MyST admonition blocks** for tips and warnings.
- **Cross-reference the user guide** for conceptual explanations: "For a detailed explanation of Log Loss, see the [User Guide](../user-guide.md#evaluation-metrics)."
- **Keep code examples minimal but complete.** The custom model example should be copy-pasteable and working.
- **Test your code examples.** Verify imports, function signatures, and expected behavior against the actual codebase before writing them into the tutorial.

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 7.9] — AC definitions
- [Source: _bmad-output/implementation-artifacts/7-8-write-comprehensive-user-guide.md] — Previous story context
- [Source: docs/user-guide.md] — Existing user guide (DO NOT duplicate content)
- [Source: docs/index.rst] — Current toctree structure (add Tutorials section)
- [Source: docs/conf.py] — Sphinx configuration (Furo theme, MyST parser)
- [Source: README.md] — Current README to enhance
- [Source: src/ncaa_eval/model/base.py] — Model ABC, StatefulModel definitions
- [Source: src/ncaa_eval/model/registry.py] — Plugin registry (@register_model)
- [Source: src/ncaa_eval/model/elo.py] — Elo reference implementation (StatefulModel)
- [Source: src/ncaa_eval/model/xgboost_model.py] — XGBoost reference implementation (Model)
- [Source: src/ncaa_eval/model/logistic_regression.py] — Minimal stateless example
- [Source: src/ncaa_eval/evaluation/metrics.py] — Metric implementations
- [Source: src/ncaa_eval/evaluation/simulation.py] — ScoringRule ABC and scoring implementations
- [Source: src/ncaa_eval/evaluation/backtest.py] — Backtest pipeline
- [Source: src/ncaa_eval/cli/train.py] — Training CLI commands
- [Source: dashboard/app.py] — Dashboard entry point
- [Source: .github/workflows/main-updated.yaml] — CI/CD pipeline (docs build + GH Pages publish)

### Previous Story Intelligence (Story 7.8)

- Story 7.8 created `docs/user-guide.md` — comprehensive 21KB guide covering metrics, models, dashboard, Game Theory sliders
- User guide uses `{contents}` directive (line 3-6) that needs removal per THIS story's AC
- Story 7.8 also committed sphinx-apidoc generated `docs/api/*.rst` files and fixed CI docs build
- Story 7.8 code review fixes included: adding missing `_predict_one` to StatefulModel plugin docs, fixing broken README relative link, adding workflow file to File List
- Fibonacci scoring values in the codebase are `2-3-5-8-13-21` (not `1-1-2-3-5-8` from original epic — story 7.8 corrected this)
- Quality gates at Story 7.8: 865 tests passed, 1 skipped
- Commit style used: `docs(guide): ...` — use conventional commits for this documentation story

### Git Intelligence

Recent Epic 7 commits:
- `a81b307` — Write comprehensive user guide (Story 7.8) (#44)
- `cf522ee` — Research game theory slider mechanism spike (Story 7.7)
- `672cc83` — feat(dashboard): Pool Scorer page with MC outcome analysis and CSV export
- `21b52d8` — feat(dashboard): Build Presentation Page — Bracket Visualizer (Story 7.5)

Pattern: documentation-only stories use `docs(...)` scope. Dashboard stories use `feat(dashboard)`.

### README Badge Reference

Current badges in README:
```markdown
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg?style=flat-square)](https://conventionalcommits.org)
[![Github Actions](https://github.com/dhilgart/NCAA_eval/actions/workflows/python-check.yaml/badge.svg)](https://github.com/dhilgart/NCAA_eval/actions/workflows/python-check.yaml)
```

Suggested additions:
```markdown
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue?style=flat-square)](https://dhilgart.github.io/NCAA_eval/)
[![Python](https://img.shields.io/badge/python-3.12+-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
```

## Dev Agent Record

### Agent Model Used

{{agent_model_name_version}}

### Debug Log References

### Completion Notes List

### File List
