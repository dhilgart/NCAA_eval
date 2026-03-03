# Codebase Audit — Pass 2 Addendum

**Date:** 2026-03-02
**Methodology:** Cross-agent review of Pass 1 findings, plus targeted source-file inspection of `simulation.py`, `dashboard/app.py`, `noxfile.py`, `.pre-commit-config.yaml`, and `.github/workflows/`

---

## Agent Commentary on Pass 1 Report

### Winston (Architect)

The PM's findings on missing product features (1.2 user-editable bracket, 1.3 Kaggle export, 1.4 ensemble blending) are architecturally significant beyond their product-level impact. The simulation engine's `ProbabilityProvider` protocol is well-designed and extensible, but nothing downstream of it supports user overrides or multi-model composition. Adding user-editable brackets would require a new `UserOverrideProvider` that wraps an existing provider and substitutes user picks at specific bracket nodes -- this is a non-trivial architectural extension, not a simple UI feature. The QA finding about `scoring_from_config` being untested (3.21) is particularly concerning because this factory function is the primary public interface for config-driven scoring instantiation; its 5-branch dispatch is exactly the kind of code that accumulates regressions when the registry grows. I also want to escalate 2.11/3.28 (ESPN exception swallowing): these are listed as separate findings but they are fundamentally the same issue -- the architectural problem is that the ingest layer has no concept of a "partial success" result type that carries both data and error metadata back to callers.

### Quinn (QA)

Winston's finding about private attribute access on `EloFeatureEngine` (3.4) has a direct testability implication that the Pass 1 report does not capture: tests that validate Elo provider behavior are forced to use `_predict_one` (or construct an `EloProvider` wrapper that uses it internally), meaning the tests are coupled to an implementation detail. If the method name changes, both production code in `simulation.py` AND test code break simultaneously, making the tests less useful as a regression safety net. The SM's finding about Fibonacci values (1.8) should have accompanying test coverage regardless of which values the PO chooses -- there should be a parameterized test asserting the actual point values match a documented constant, and currently there is none. I also note that the `noxfile.py` typecheck session includes `sync.py` and `noxfile.py` itself, but the CI workflow (`python-check.yaml`) runs `pre-commit run --all-files` instead of `nox` -- meaning CI and local `nox` have different quality gate paths, which can lead to "works on my machine" divergence.

### Bob (SM / Scrum Master)

The consolidated report reveals a systemic process gap: multiple features were explicitly deferred as "post-MVP" during story implementation (Game Theory sliders via 7.7 spike, Metric Explorer drill-downs via 3.x, Candidate Entry flagging, JSON export option) but NO backlog items were ever created for any of them. This is not an isolated oversight -- it is a repeating pattern across at least 4 epic ACs. The correct remediation is not to fix each one individually but to do a single backlog grooming sweep that creates placeholder stories for every deferred AC item, tagged with a "post-mvp" label. I count at minimum 6 untracked post-MVP items from the Pass 1 report alone: (1) game theory sliders implementation, (2) Metric Explorer round/seed/conference drill-downs, (3) Candidate Entry flagging, (4) JSON export for Pool Scorer, (5) CLI `predict` command, (6) per-game explainability. Each needs a story in the backlog or an explicit PO decision to drop it. The 2.17 finding (Story 2.3 open follow-ups) is also a tracking failure -- the SM should have ensured the AI-review follow-up items were captured as separate stories or added to the sprint backlog before closing the story.

### John (PM / Product Manager)

Winston's architectural findings about simulation.py (3.1, 3.10) and the scoring registry have a product impact that the architect framed purely as code quality. The real risk is extensibility: when users want to add custom scoring rules (a common pool bracket scenario -- "our office pool gives 3 points per correct Final Four pick plus seed bonus"), they currently have to understand a 1,291-line module, find the registry decorator pattern, and add their class in the right location. The "Fibonacci values mismatch" (1.8) is more than a PO decision -- it represents a data integrity risk: if a user selects "Fibonacci" scoring expecting the classic (1-1-2-3-5-8) and gets (2-3-5-8-13-21), their pool standings calculations will be silently wrong. I would escalate this from "requires PO direction" to a Category 3 bug -- at minimum the UI label should match the actual values. Additionally, the `dashboard/app.py` sidebar tells users to `run python sync.py first` but the official CLI entry point is `ncaa-eval sync` (per the CLI package) -- this inconsistency will confuse new users following the getting-started tutorial.

### Paige (Tech Writer)

The PM's findings about tutorial inaccuracy (3.17) and missing troubleshooting (3.18) align with a broader documentation-code divergence pattern I see across the report. The `dashboard/app.py` sidebar info message says `python sync.py` while the tutorials reference potentially different commands -- there is no single source of truth for "how do I get data." The user guide's 50-line specification for non-existent game theory sliders (2.19) should be reclassified from "might require human judgment" to Category 3: documenting a feature that does not exist is objectively wrong regardless of product direction. Either the documentation should be removed or clearly marked as "planned." Additionally, the noxfile has a `docs` session that runs `sphinx-apidoc` and `sphinx-build`, but the CI workflow in `main-updated.yaml` only runs `sphinx-build` (no `sphinx-apidoc` step), meaning the auto-generated API docs from the nox session may differ from what gets published to GitHub Pages. This is a documentation build inconsistency that can produce stale API references on the published site.

### Murat (TEA / Test Engineering Architect)

Quinn's finding about `scoring_from_config` being untested (3.21) and the CLI only testing logistic regression (3.22) point to a structural gap in the test architecture: the scoring subsystem and the model training pipeline both rely on registry/factory patterns, but neither registry's dispatch logic has test coverage proportional to its branch complexity. For `scoring_from_config` specifically, a parameterized test covering all 5 branches plus the unknown-type error path would be 6 test cases -- mechanical to write and high-value. The architect's identification of two `noqa: PLR0913` suppressions in `simulation.py` (lines 1042 and 1218) signals functions that are too complex for their current test coverage level. The `simulate_tournament` orchestrator in particular has a `method` dispatch branch that selects between analytical and Monte Carlo paths, and the interaction between `scoring_rules=None` defaulting and `SeedDiffBonusScoring` isinstance-checking (line 1267) is a subtle logic path that deserves a dedicated test. I also note that the `.pre-commit-config.yaml` has no `exclude` patterns for the `template/` directory despite the MEMORY.md documenting that template files contain Jinja2 syntax that breaks hooks -- this either means the hooks are currently failing silently on template files, or the template files happen to not trigger any violations, but it is a latent risk.

---

## New Issues (Pass 2 Discoveries)

### P2-1: `rapidfuzz` Is an Undeclared Dependency (Category 3 -- Bug)
- **Severity:** Major
- **Files:** `src/ncaa_eval/ingest/sync.py:16`, `src/ncaa_eval/ingest/connectors/espn.py:18`
- **Description:** Both modules do `from rapidfuzz import fuzz`, but `rapidfuzz` is not listed in `pyproject.toml` under `[tool.poetry.dependencies]`. It presumably gets installed as a transitive dependency (likely via `cbbpy`), but this is fragile -- if the upstream dependency drops `rapidfuzz`, the ingest layer breaks at import time with no clear error message.
- **Fix:** Add `rapidfuzz` explicitly to `[tool.poetry.dependencies]`
- **Discovered by:** Winston (Architect) + Murat (TEA), cross-referencing import statements against pyproject.toml

### P2-2: CI Workflow Does Not Run `nox` -- Quality Gate Divergence (Category 3 -- Process)
- **Severity:** Minor
- **Files:** `.github/workflows/python-check.yaml`, `noxfile.py`
- **Description:** The noxfile defines a 3-session quality pipeline (lint -> typecheck -> tests), but the CI workflow runs `pre-commit run --all-files` + standalone `pytest --cov` instead of `nox`. This means: (a) CI does not run `mypy` as a standalone pass (it relies on pre-commit's mypy hook, which has `pass_filenames: false` and different config than noxfile's explicit file list), (b) noxfile includes `sync.py` and `noxfile.py` in mypy scope but pre-commit's mypy hook only checks `^(src/|tests/)` files, and (c) local developers running `nox` get a different quality gate than CI. The noxfile was presumably created to be the canonical quality gate but is not used in CI.
- **Fix:** Either run `nox` in CI or remove `noxfile.py` and document pre-commit as the canonical gate
- **Discovered by:** Quinn (QA) + Murat (TEA)

### P2-3: `peaceiris/actions-gh-pages@v3.8.0` Is Deprecated (Category 3 -- Maintenance)
- **Severity:** Minor
- **Files:** `.github/workflows/main-updated.yaml:52`
- **Description:** The `peaceiris/actions-gh-pages` action at v3.8.0 is outdated. GitHub has been migrating away from older action runners, and this version may stop working. The recommended replacement is `actions/deploy-pages` with `actions/upload-pages-artifact`, or at minimum upgrading to v4 of the peaceiris action.
- **Fix:** Upgrade to `peaceiris/actions-gh-pages@v4` or migrate to `actions/deploy-pages`
- **Discovered by:** Winston (Architect)

### P2-4: CI Missing `sphinx-apidoc` Step -- Published Docs May Have Stale API References (Category 3 -- Documentation)
- **Severity:** Minor
- **Files:** `.github/workflows/main-updated.yaml:49`, `noxfile.py:46-47`
- **Description:** The `noxfile.py` docs session runs `sphinx-apidoc -f -e -o docs/api src/ncaa_eval` before `sphinx-build`, but the CI workflow's `publish-github-page` job only runs `sphinx-build`. If new modules are added to `src/ncaa_eval`, their API stubs will not be generated during CI, and the published documentation will be missing those modules until someone runs `nox -s docs` locally and commits the generated `.rst` files.
- **Fix:** Add `poetry run sphinx-apidoc -f -e -o docs/api src/ncaa_eval` before `sphinx-build` in the CI workflow
- **Discovered by:** Paige (Tech Writer)

### P2-5: No Coverage Threshold Enforced (Category 2 -- Might Require Human Judgment)
- **Severity:** Minor
- **Files:** `pyproject.toml` (tool.coverage section), `.github/workflows/python-check.yaml:31`
- **Description:** CI runs `pytest --cov=src/ncaa_eval --cov-report=term-missing` but there is no `--cov-fail-under=XX` flag and no `fail_under` in `[tool.coverage.report]`. Coverage can silently regress without failing the build.
- **Decision needed:** What minimum coverage threshold is appropriate? (Current coverage level should be measured first)
- **Discovered by:** Murat (TEA)

### P2-6: Dashboard Package Excluded from All Quality Gates (Category 2 -- Might Require Human Judgment)
- **Severity:** Major
- **Files:** `noxfile.py:30-33`, `.pre-commit-config.yaml:67`, `pyproject.toml:50`
- **Description:** The `dashboard/` directory is not included in mypy's scope (pyproject.toml `files = ["src/ncaa_eval", "tests", "sync.py"]`), not in the noxfile typecheck session, and has 6 `# type: ignore` suppressions across its files. The pre-commit mypy hook only checks `^(src/|tests/)`. The dashboard is where users interact with the product, yet it receives no static type checking. Similarly, `dashboard/` is excluded from `check-manifest` and not part of the distributed package.
- **Tradeoff:** Dashboard code uses `streamlit` which has poor type stubs, so strict mypy may be impractical. But at minimum `--follow-imports=normal` or a relaxed mypy config for `dashboard/` would catch import errors and basic type mismatches.
- **Discovered by:** Winston (Architect) + Murat (TEA)

### P2-7: `template/` Directory Has Committed `.ruff_cache` Files (Category 3 -- Hygiene)
- **Severity:** Cosmetic
- **Files:** `template/{{cookiecutter.project_slug}}/.ruff_cache/` (10+ cache files)
- **Description:** The cookiecutter template directory contains committed Ruff cache files (`.ruff_cache/0.8.4/...`, `.ruff_cache/0.15.1/...`). These are build artifacts that should never be committed. The `.gitignore` likely has `.ruff_cache` but the template's nested copy was committed before the rule existed or the exclude pattern does not cover the template path.
- **Fix:** `git rm -r --cached template/\{\{cookiecutter.project_slug\}\}/.ruff_cache` and add to `.gitignore` if not already covered
- **Discovered by:** Quinn (QA)

### P2-8: `dashboard/app.py` Sidebar References `sync.py` Instead of CLI Command (Category 3 -- UX)
- **Severity:** Minor
- **File:** `dashboard/app.py:69`
- **Description:** The sidebar info message says `No data available -- run 'python sync.py' first` but the project has a proper CLI with `ncaa-eval sync` (via typer). The root `sync.py` is documented as a convenience wrapper (Finding 2.1), but a new user following the getting-started tutorial may encounter a different command. The sidebar message should reference the canonical CLI command, or at minimum both options.
- **Fix:** Change to `run 'ncaa-eval sync' first` or `run 'python sync.py' first`
- **Discovered by:** John (PM) + Paige (Tech Writer)

---

## Reclassifications

### 2.11 -> Duplicate of 3.28 (ESPN Exception Swallowing)
- **Rationale:** Finding 2.11 (Architect: `EspnConnector._fetch_per_team` swallows exceptions at DEBUG level) and 3.28 (PM: ESPN connector silently swallows all per-team exceptions) describe the identical issue in the identical file/lines. The Category 3 classification is correct -- silent data loss is unambiguously a bug. Remove 2.11 as a duplicate.

### 2.19 -> Category 3 (User Guide Documents Non-Existent Feature)
- **Rationale:** The user guide containing 50 lines of specification for game theory sliders that do not exist is not a judgment call. Documentation that describes non-existent functionality is objectively misleading. The small `{note}` admonition is insufficient -- a user skimming the guide will believe the feature exists. This should be Category 3 with a fix of either removing the section or adding a prominent "NOT YET IMPLEMENTED" banner.

### 1.8 -> Split: Category 1 (which values?) + Category 3 (UI label must match code)
- **Rationale:** The PO needs to decide which Fibonacci sequence is canonical (Cat 1), but regardless of that decision, the current UI label says "Fibonacci" while the code uses non-standard values (2-3-5-8-13-21 is not a standard Fibonacci sequence starting from 1). The label should at minimum display the actual values, e.g., "Fibonacci (2-3-5-8-13-21)" so users are not misled. The label accuracy fix is Cat 3.

---

## Cross-Cutting Patterns

### Pattern A: "Deferred but Never Tracked"
Findings 1.1, 1.9, 1.10, 1.14, 2.17 all share the same root cause: a feature was explicitly deferred during story implementation, but no follow-up story or backlog item was created. This happened at least 5 times across different epics and different stories, suggesting a systemic gap in the Definition of Done. **Recommended process fix:** The DoD should include "all deferred items have backlog stories created" as a checklist item.

### Pattern B: "Private API Leakage Across Module Boundaries"
Findings 3.4, 3.11, 3.13, 3.25 all involve one module accessing another module's private (`_`-prefixed) attributes, constants, or methods. This is not just a naming convention violation -- it creates implicit coupling contracts that mypy and the public API surface do not enforce. The pattern appears in 4 different parts of the codebase (transform -> elo internals, model -> elo internals, evaluation -> elo internals, dashboard -> model internals, tests -> dashboard internals). **Recommended architectural fix:** Conduct a single focused pass to expose public APIs for every cross-module private access.

### Pattern C: "Dual Quality Gates, Neither Complete"
The project has TWO quality gate mechanisms (noxfile.py AND pre-commit hooks) that overlap but do not align. The noxfile checks `sync.py` and `noxfile.py` with mypy but pre-commit does not. Pre-commit runs `ruff-format` as a fixer but noxfile only checks formatting. CI uses pre-commit, not nox. Local development can use either. This creates confusion about what the canonical gate is and risks "works locally, fails in CI" or vice versa. **Recommended fix:** Pick one system as canonical, align the other, and document the decision.

### Pattern D: "Silent Failure in Data Pipelines"
Findings 3.12 (backtest swallows exceptions), 3.28 (ESPN swallows exceptions), 2.11 (duplicate of 3.28), and 3.3 (hardcoded 2025 dedup) all represent the same anti-pattern: error conditions that are silently absorbed rather than surfaced. In a data pipeline for making bracket predictions, silent failures can lead to incorrect predictions based on incomplete or corrupted data. The ingest layer, the evaluation layer, and the deduplication logic all independently chose to suppress errors. **Recommended fix:** Establish a project convention for error handling in data pipeline code -- at minimum `logger.warning()` with structured context, ideally a result type that carries warnings alongside data.

### Pattern E: "No Dashboard in the Quality Perimeter"
Findings P2-6, 3.5, 3.13, 2.12, 2.13, 2.14, 3.31, 3.32, 3.33, 3.34 collectively reveal that the dashboard is the least-governed part of the codebase despite being the primary user-facing surface. It is excluded from mypy, has its own ad-hoc `type: ignore` suppressions, has no coverage measurement, contains a 621-line kitchen-sink module, uses undocumented Streamlit APIs, and has multiple UX issues. The dashboard accumulated more findings than any other component but receives the least automated quality enforcement.

---

## Updated Metrics

### Pass 2 New/Reclassified Items

| ID | Category | Severity | Type |
|----|:--------:|:--------:|------|
| P2-1 | 3 | Major | Undeclared dependency |
| P2-2 | 3 | Minor | CI/quality gate divergence |
| P2-3 | 3 | Minor | Deprecated CI action |
| P2-4 | 3 | Minor | Docs build gap |
| P2-5 | 2 | Minor | No coverage threshold |
| P2-6 | 2 | Major | Dashboard excluded from type checking |
| P2-7 | 3 | Cosmetic | Committed cache files in template |
| P2-8 | 3 | Minor | Sidebar references wrong CLI command |
| Reclass: 2.11 | -- | -- | Duplicate of 3.28 (removed) |
| Reclass: 2.19 | 3 | Major | Was Cat 2, now Cat 3 |
| Reclass: 1.8 | Split | -- | Partially Cat 1, partially Cat 3 |

### Revised Totals (Post-Pass 2)

| Category | Pass 1 | Pass 2 Delta | Revised |
|----------|:------:|:------------:|:-------:|
| 1. Requires PO Direction | 15 | -1 (partial 1.8 split) | 14 |
| 2. Might Require Human Judgment | 21 | -1 (2.11 dup) -1 (2.19 reclass) +2 (P2-5, P2-6) | 21 |
| 3. Obviously Needs Fixing | 34 | +1 (2.19 reclass) +1 (1.8 label) +6 (P2-1..4,7,8) | 42 |
| **Total distinct issues** | **70** | **+7 net** | **77** |

---

## Pass 2 Agent Sign-offs

- **Winston (Architect):** "The dual quality gate problem (Pattern C) is the most actionable cross-cutting finding. The undeclared `rapidfuzz` dependency (P2-1) is a latent breakage risk. The private API leakage pattern (Pattern B) should be addressed as a single architectural initiative, not piecemeal."
- **Quinn (QA):** "The CI/nox divergence (P2-2) means local and CI quality gates are not equivalent. The missing coverage threshold (P2-5) and dashboard type-checking gap (P2-6) are the biggest testing blind spots. Template `.ruff_cache` files should be cleaned up immediately."
- **Bob (SM):** "Pattern A ('deferred but never tracked') is a process failure that produced at least 6 orphaned post-MVP items. A single backlog grooming session to create stories for all deferred AC items is the highest-leverage process fix."
- **John (PM):** "The Fibonacci label mismatch (1.8 partial reclass) is a data integrity issue for any user relying on the scoring display. The sidebar `sync.py` reference (P2-8) will confuse the first-run experience. The user guide documenting non-existent sliders (2.19 reclass) actively misleads users."
- **Paige (Tech Writer):** "The CI docs build gap (P2-4) means published API docs may diverge from actual code. The `sync.py` vs `ncaa-eval sync` inconsistency (P2-8) needs a single canonical reference. The non-existent slider documentation (2.19) should be removed, not just flagged."
- **Murat (TEA):** "The scoring subsystem is the biggest test architecture gap -- `scoring_from_config` (3.21) and scoring value correctness (1.8) both lack any test coverage. The dashboard being excluded from type checking (P2-6) means an entire layer of the application has no static analysis safety net."
