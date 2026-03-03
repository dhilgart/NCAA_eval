# Codebase Audit — Pass 3 Addendum: Documentation & Spec Compliance Review

**Date:** 2026-03-03
**Methodology:** Line-by-line review of documentation, specs, and planning artifacts against actual codebase state. Triggered by PO feedback identifying three systemic issues: (1) documentation commands not validated with E2E tests, (2) PEP 20 missing from PR checklist, (3) codebase-wide PEP 20 non-compliance.

**Documents Reviewed:**
- `docs/STYLE_GUIDE.md` (869 lines)
- `docs/TESTING_STRATEGY.md` (297 lines)
- `docs/testing/*` (7 guides, ~1,800 lines total)
- `specs/01-brainstorming-session-results.md`
- `specs/02-project-brief.md`
- `specs/03-prd.md`
- `specs/04-front-end-spec.md`
- `specs/05-architecture-fullstack.md`
- `.github/pull_request_template.md`
- `_bmad-output/planning-artifacts/epics.md`

---

## New Issues (Pass 3 Discoveries)

### STYLE GUIDE FINDINGS

#### P3-1: PR Template Missing Three Style Guide Gates (Category 3 — Bug)
- **Severity:** Critical
- **Files:** `.github/pull_request_template.md`, `docs/STYLE_GUIDE.md:578-609`
- **Description:** The Style Guide Section 7 ("PR Checklist Summary") explicitly lists three gates as PR-time manual review items: (1) **PEP 20 compliance**, (2) **SOLID principles**, (3) **Pure functions / functional design**. None of these appear in the actual PR template's checklist. Every PR merged to date was never reviewed against these criteria. The Style Guide even says "The actual PR template is at `.github/pull_request_template.md`" — implying they should be in sync.
- **Fix:** Add three new checkboxes to the PR template "Code Quality" section:
  - `[ ] **PEP 20 compliance** — Simple (complexity ≤ 10), explicit (no magic numbers), readable (full domain words), flat (nesting ≤ 3), consistent with project patterns`
  - `[ ] **SOLID principles** — Single responsibility, open for extension, Liskov substitution, interface segregation, dependency inversion`
  - `[ ] **Pure function design** — Business logic is pure, side effects at edges, no I/O mixed with calculations`
- **Discovered by:** PO feedback + document review

#### P3-2: PEP 20 Section Covers Only 5 of 19 Aphorisms (Category 3 — Incomplete)
- **Severity:** Major
- **File:** `docs/STYLE_GUIDE.md:236-352`
- **Description:** The Zen of Python has 19 aphorisms. Section 6 covers only 5: Simple > Complex (#3), Explicit > Implicit (#2), Readability Counts (#7), Flat > Nested (#5), One Obvious Way (#13). Missing aphorisms with high project relevance:
  - **#10 "Errors should never pass silently"** — Directly violated by Pattern D (ESPN swallowing, backtest swallowing, hardcoded 2025 dedup). This is the most impactful missing principle.
  - **#11 "Unless explicitly silenced"** — Relevant to `# type: ignore` and `# noqa:` usage patterns that have no guidance.
  - **#4 "Complex is better than complicated"** — The counterbalance to #3; the project has legitimately complex code (simulation, Elo engines) that needs this guidance.
  - **#12 "Refuse the temptation to guess"** — Reinforces mypy --strict rationale.
  - **#6 "Sparse is better than dense"** — Relevant to the 110-char line length and explicit naming.
  - **#8/#9 "Special cases / practicality"** — Frames the vectorization exceptions properly.
  - **#17/#18 "Hard to explain = bad idea"** — Reinforces complexity limits.
- **Fix:** Expand Section 6 to cover at minimum aphorisms #4, #6, #8/#9, #10/#11, #12, #17/#18 with project-specific examples and review checklist items.
- **Discovered by:** PEP 20 gap analysis

#### P3-3: Active Ruff Rules Table Incomplete (Category 3 — Inaccurate)
- **Severity:** Minor
- **File:** `docs/STYLE_GUIDE.md:132-141`
- **Description:** The "Active Ruff Rules" table lists only `I`, `UP`, `PT`, `TID25`. Missing from the table: `C90` (McCabe complexity), `PLR0911` (too many returns, max 6), `PLR0912` (too many branches, max 12), `PLR0913` (too many args, max 5). Also, the Suppressed Rules table omits `PLR2004` (magic value comparison — suppressed with comment "too aggressive for data science" in pyproject.toml).
- **Fix:** Add C90, PLR0911, PLR0912, PLR0913 to Active Rules table; add PLR2004 to Suppressed Rules table.
- **Discovered by:** pyproject.toml cross-reference

#### P3-4: Pydantic mypy Plugin Not Documented (Category 3 — Gap)
- **Severity:** Minor
- **File:** `docs/STYLE_GUIDE.md:153-180`
- **Description:** pyproject.toml configures `plugins = ["pydantic.mypy"]` with `init_typed = true`, `init_forbid_extra = true`, `warn_required_dynamic_aliases = true`. None of this is documented in the Style Guide's mypy section despite Pydantic being central to the project.
- **Fix:** Add a "Pydantic Integration" subsection to Section 4.
- **Discovered by:** pyproject.toml cross-reference

#### P3-5: Project Layout Diagram Inaccurate (Category 3 — Stale)
- **Severity:** Minor
- **File:** `docs/STYLE_GUIDE.md:616-632`
- **Description:** Multiple inaccuracies: (1) Missing `cli/` directory entirely — a major module. (2) `utils/` described as "logging, assertions" but contains only `logger.py`. (3) `models/` (plural) used in example path instead of `model/` (singular). (4) Test structure shown as mirroring src/ but actual tests use `unit/`/`integration/` organization. (5) Root `__init__.py` claim "re-exports public API" is false — `from ncaa_eval import EloModel` fails.
- **Fix:** Update layout diagram to reflect actual directory structure.
- **Discovered by:** codebase cross-reference

#### P3-6: ISP/Protocol Claim Contradicts Codebase Reality (Category 3 — Inaccurate)
- **Severity:** Minor
- **File:** `docs/STYLE_GUIDE.md:774-800`
- **Description:** Section 10 ISP states "MyPy strict mode: Protocols preferred over abstract classes." The codebase uses ABCs overwhelmingly (Model ABC, Repository ABC, Connector ABC). Only `simulation.py` uses 2 Protocols. The claim does not reflect reality.
- **Fix:** Rewrite ISP guidance to acknowledge ABCs as the primary pattern, with Protocols as a complement for structural typing.
- **Discovered by:** codebase cross-reference

#### P3-7: `noqa`/`type: ignore` Usage Not Documented (Category 3 — Gap)
- **Severity:** Minor
- **File:** `docs/STYLE_GUIDE.md` (entire document)
- **Description:** The codebase has 7+ `# noqa: PLR0913` suppressions in `src/` and 6+ `# type: ignore` in dashboard/. No guidance exists on when lint/type suppressions are acceptable, how they should be documented (e.g., always include the specific code), or what the escalation path is (refactor vs. suppress).
- **Fix:** Add a "Lint Suppression Policy" subsection.

---

### TESTING DOCUMENTATION FINDINGS

#### P3-8: All Documented Test Directory Structures Are Stale (Category 3 — Stale)
- **Severity:** Major
- **Files:** `docs/TESTING_STRATEGY.md:206-218`, `docs/testing/conventions.md:9-27`
- **Description:** Both documents show a test directory tree with 8 file references. **All 8 are wrong**: `test_metrics.py` (actual: `test_evaluation_metrics.py`), `test_features.py` (doesn't exist), `test_sync_pipeline.py` (actual: `test_sync.py`), `test_training_pipeline.py` (doesn't exist), `sample_games.csv` (doesn't exist), `sample_predictions.json` (doesn't exist). The documentation predates the actual test files and was never updated.
- **Fix:** Update directory trees to reflect actual file names.

#### P3-9: Five Documented Test Marker Categories Have Zero Tests (Category 3 — Doc/Code Gap)
- **Severity:** Major
- **Files:** All testing docs
- **Description:** The testing strategy extensively documents five marker-based test categories with examples and best practices, but zero actual tests exist for any of them:
  - `@pytest.mark.performance` — 0 tests (documented: vectorization compliance, 60s backtest target)
  - `@pytest.mark.regression` — 0 tests (documented: bug recurrence prevention)
  - `@pytest.mark.fuzz` — 0 tests (documented: crash resilience via Hypothesis)
  - `@pytest.mark.mutation` — 0 tests (documented: mutation testing candidates)
  - `@pytest.mark.slow` — 0 tests (documented: tests over 5 seconds)
- **Fix:** Either implement exemplar tests for each category or remove the categories from documentation until they're needed.

#### P3-10: `@pytest.mark.unit` Used 114 Times But Not Registered (Category 3 — Bug)
- **Severity:** Major
- **Files:** `tests/unit/test_chronological_serving.py`, `tests/unit/test_sequential.py`, `tests/unit/test_normalization.py`, `tests/unit/test_graph.py`
- **Description:** 114 test functions/classes are decorated with `@pytest.mark.unit`, but this marker is NOT registered in `pyproject.toml`'s `[tool.pytest.ini_options] markers` list. With `--strict-markers` enabled (line 108 of pyproject.toml), using unregistered markers should cause a test collection error. This is either silently broken or masked by some other mechanism.
- **Fix:** Either register `unit` in pyproject.toml markers or remove the decorator from all 114 usages.

#### P3-11: Zero E2E Integration Tests for Documented Commands (Category 3 — Testing Gap)
- **Severity:** Critical
- **Files:** All testing docs, `tests/` directory
- **Description:** The documentation describes 18+ commands that users should be able to run. **None** have E2E integration tests validating they actually work:
  - `pytest -m smoke` — no test validates the suite completes in <5s
  - `pytest --cov=src/ncaa_eval --cov-report=term-missing` — no test
  - `nox` (all sessions) — no test
  - `nox -s lint`, `nox -s typecheck`, `nox -s tests` — no tests
  - `mutmut run` — no test
  - `edgetest` — no test (and edgetest may not even be configured)
  - `check-manifest` — no test (not even in pre-commit hooks)
  - All `pytest -m <marker>` filter commands — no tests
  - `ncaa-eval sync`, `ncaa-eval train` — no E2E smoke tests
  - `streamlit run dashboard/app.py` — no test

  The PO specifically flagged this: "the commands in the documentation all need to be covered with E2E integration tests."
- **Fix:** Create a dedicated test suite that validates every documented command exits successfully.

#### P3-12: `check-manifest` Not in Pre-Commit Hooks (Category 3 — Process Gap)
- **Severity:** Minor
- **Files:** `.pre-commit-config.yaml`, `docs/TESTING_STRATEGY.md:86`, `docs/testing/execution.md:35`
- **Description:** Both TESTING_STRATEGY.md and execution.md list `check-manifest` as a Tier 1 pre-commit check. It is NOT configured in `.pre-commit-config.yaml`. The tool exists as a dev dependency but is not wired into any automated gate.
- **Fix:** Either add `check-manifest` to pre-commit hooks or remove it from Tier 1 documentation.

#### P3-13: Smoke Test Time Budget Inconsistency (Category 3 — Inconsistency)
- **Severity:** Minor
- **Files:** `docs/TESTING_STRATEGY.md:85,76`, `docs/testing/conventions.md:118`, `docs/testing/execution.md:22,43`, `pyproject.toml:122`
- **Description:** The smoke test time budget varies: "< 5 seconds total" in conventions.md and execution.md table, "< 10 seconds total" in TESTING_STRATEGY.md and execution.md header, and pyproject.toml marker says "< 10 seconds total". The distinction (10s for all Tier 1 checks vs. 5s for smoke tests alone) is never clarified.
- **Fix:** Standardize on one set of numbers and make the distinction explicit.

#### P3-14: Noxfile Session Config Doesn't Match Documentation (Category 3 — Stale)
- **Severity:** Minor
- **Files:** `docs/testing/conventions.md:266-278`, `noxfile.py`
- **Description:** Documentation shows nox tests session as: `session.install("pytest", "pytest-cov", "hypothesis"); session.run("pytest", "-m", "smoke", "--cov=src/ncaa_eval")`. Actual noxfile uses `python=False` (no virtualenv), runs full suite (`pytest --tb=short`), and does not install packages.
- **Fix:** Update documentation to match actual noxfile behavior.

#### P3-15: Testing Docs Reference 10+ Non-Existent APIs (Category 3 — Stale)
- **Severity:** Minor
- **Files:** All `docs/testing/` guides
- **Description:** Code examples reference classes/functions that don't exist or have different names: `ChronologicalDataAPI` (actual: `ChronologicalDataServer`), `calculate_brier_score()` (actual: `brier_score()`), `update_elo_rating()` (actual: `EloFeatureEngine.update_game()`), `WalkForwardSplitter` (actual: `walk_forward_splits()`), `Game` as TypedDict (actual: Pydantic model), and many more.
- **Fix:** Update all code examples to use actual API names.

#### P3-16: Fixture Naming Convention Not Followed (Category 3 — Inconsistency)
- **Severity:** Cosmetic
- **Files:** `docs/testing/conventions.md:35`, `tests/conftest.py`
- **Description:** Documentation says fixture naming convention is `<resource>_fixture()`. Actual fixtures are `temp_data_dir()` and `sample_game_records()` — neither follows the `_fixture()` suffix.
- **Fix:** Either update the convention to match reality or rename fixtures.

---

### SPEC & EPICS FINDINGS

#### P3-17: NFR3 Plugin Registry Only 2/4 Covered (Category 1 — Requires PO Direction)
- **Severity:** Major
- **File:** `_bmad-output/planning-artifacts/epics.md:85`
- **Description:** PRD says NFR3 requires plugin registries for: (1) models, (2) scoring functions, (3) **metrics**, (4) **feature generators**. The coverage map claims Epic 5 covers NFR3 fully, but only model and scoring registries exist. No metric registry or feature-generator registry has been implemented or has any story. Story 7.9 includes a "How to Add a Custom Metric" tutorial — documenting a feature that doesn't exist.
- **Decision needed:** Are metric and feature-generator plugin registries required for MVP?

#### P3-18: Post-MVP Backlog Missing 15+ Deferred Items (Category 3 — Process Gap)
- **Severity:** Major
- **File:** `_bmad-output/planning-artifacts/epics.md:972-1066`
- **Description:** The Post-MVP Backlog captures spike-derived items (model plugins, data sources, rating systems) but is completely missing all implementation-deferred items. At least 15 orphaned items identified:
  1. Game Theory Slider implementation (from 7.5/7.7)
  2. User-Editable Bracket (from UX Spec Flow 1)
  3. Kaggle Submission Export (from PRD mission)
  4. Metric Explorer: round/seed/conference drill-downs (from 7.4)
  5. Candidate Entry Flagging (from 7.5)
  6. CLI `predict` command (from PRD)
  7. Model Ensemble/Blending (competitive necessity)
  8. JSON Export for Pool Scorer (from 7.6)
  9. st.progress for Simulation (from 7.6)
  10. Per-Game Prediction Explainability (from PRD)
  11. Demo/Sample Data for zero-setup onboarding
  12. Custom Metric Plugin Registry (from NFR3)
  13. Custom Feature Generator Plugin Registry (from NFR3)
  14. Confusion Matrix in Model Deep Dive (from PRD 3.2)
  15. Public Bracket Competitive ROI Simulation (from UX Spec Flow 2)
- **Fix:** Expand Story 8.12 scope to include backlog grooming for all 15 items, or create entries now.

#### P3-19: Story 1.9 Missing from epics.md (Category 3 — Gap)
- **Severity:** Minor
- **File:** `_bmad-output/planning-artifacts/epics.md`, `_bmad-output/implementation-artifacts/sprint-status.yaml`
- **Description:** Sprint-status.yaml has `1-9-restructure-docs-sphinx-source: done` but Story 1.9 does not exist in epics.md at all.
- **Fix:** Add Story 1.9 to epics.md (retroactively document what was done).

#### P3-20: Architecture Spec Stale — Multiple Discrepancies (Category 2 — Low Priority)
- **Severity:** Minor
- **File:** `specs/05-architecture-fullstack.md`
- **Description:** Several spec claims don't match implementation: (1) `docs/specs/` path (actual: `/specs/`), (2) `dashboard/components/` (actual: `dashboard/lib/`), (3) Ingestion Engine uses `requests` (actual: `kaggle`/`cbbpy`), (4) Only 2 dashboard pages described (actual: 5). These are expected divergences from initial planning but the spec was never updated.
- **Decision needed:** Should architecture spec be marked as "initial design — see implementation for current state" or updated?

#### P3-21: edgetest Referenced in Story 1.7 AC but Never Implemented (Category 3 — Stale)
- **Severity:** Minor
- **File:** `_bmad-output/planning-artifacts/epics.md:238`
- **Description:** Story 1.7 AC says "edgetest is configured for dependency compatibility testing." No edgetest configuration exists in pyproject.toml, nox, or CI. The PR template includes an "Edge compatibility" checkbox that references edgetest. This AC was never fulfilled and no one flagged it.
- **Fix:** Either implement edgetest or remove from AC and PR template.

#### P3-22: Story 3.2 AC References Plotly — Deliberately Abandoned (Category 3 — Stale AC)
- **Severity:** Cosmetic
- **File:** `_bmad-output/planning-artifacts/epics.md:381`
- **Description:** Story 3.2 AC says "all visualizations use Plotly for interactive inline rendering." MEMORY.md documents this was deliberately abandoned for matplotlib (Plotly caused ~800MB notebooks). The AC text was never updated.
- **Fix:** Update Story 3.2 AC to reflect the matplotlib decision.

---

### CROSS-CUTTING PATTERNS (Pass 3)

#### Pattern F: "Documentation Written Aspirationally, Never Maintained"

Findings P3-2, P3-3, P3-5, P3-8, P3-9, P3-13, P3-14, P3-15 all share the same root cause: documentation was written during planning phases (Stories 1.2, 1.3) and never updated as implementation progressed through Epics 2-7. The testing documentation is the worst offender — every file name, API name, and directory structure reference is stale. The Style Guide has drifted from pyproject.toml configuration. This is not individual oversight; it's a systemic process gap where documentation maintenance was never part of the Definition of Done for implementation stories.

#### Pattern G: "PR Template Diverged from Style Guide Quality Gates"

Finding P3-1 is the most impactful: the Style Guide defines 9 PR gates, the PR template implements only 6. The three missing gates (PEP 20, SOLID, pure functions) are the "design quality" gates — the ones that require human judgment and cannot be automated. Because they were never in the PR template, they were never checked during any PR review. This explains the PO's observation that "many functions do not abide by the STYLE_GUIDE, especially PEP 20."

#### Pattern H: "Test Documentation Describes a Framework That Doesn't Exist"

Findings P3-9, P3-10, P3-11, P3-15 collectively reveal that the testing documentation describes a sophisticated 4-dimensional, 8-marker testing framework with specialized guides for each dimension. In reality: 2 of 8 markers are actively used (smoke, integration), 1 is partially used (property, 2 tests), 1 is used but unregistered (unit, 114 tests), and 4 have zero implementations. The documentation creates a false impression of testing maturity.

---

## Updated Metrics

### Pass 3 New Items

| ID | Category | Severity | Type |
|----|:--------:|:--------:|------|
| P3-1 | 3 | Critical | PR template missing 3 quality gates |
| P3-2 | 3 | Major | PEP 20 covers only 5/19 aphorisms |
| P3-3 | 3 | Minor | Ruff rules table incomplete |
| P3-4 | 3 | Minor | Pydantic mypy plugin undocumented |
| P3-5 | 3 | Minor | Project layout diagram inaccurate |
| P3-6 | 3 | Minor | ISP/Protocol claim inaccurate |
| P3-7 | 3 | Minor | No lint suppression policy |
| P3-8 | 3 | Major | Test directory structures all stale |
| P3-9 | 3 | Major | 5 marker categories with 0 tests |
| P3-10 | 3 | Major | @pytest.mark.unit unregistered (114 uses) |
| P3-11 | 3 | Critical | Zero E2E tests for documented commands |
| P3-12 | 3 | Minor | check-manifest not in pre-commit |
| P3-13 | 3 | Minor | Smoke test time budget inconsistent |
| P3-14 | 3 | Minor | Nox session docs don't match actual |
| P3-15 | 3 | Minor | Testing docs reference wrong APIs |
| P3-16 | 3 | Cosmetic | Fixture naming convention not followed |
| P3-17 | 1 | Major | NFR3 only 2/4 covered |
| P3-18 | 3 | Major | Post-MVP backlog missing 15+ items |
| P3-19 | 3 | Minor | Story 1.9 missing from epics.md |
| P3-20 | 2 | Minor | Architecture spec stale |
| P3-21 | 3 | Minor | edgetest never implemented |
| P3-22 | 3 | Cosmetic | Story 3.2 AC references abandoned Plotly |

### Revised Totals (Post-Pass 3)

| Category | Pass 2 Revised | Pass 3 Delta | Revised |
|----------|:------:|:------------:|:-------:|
| 1. Requires PO Direction | 14 | +1 (P3-17) | 15 |
| 2. Might Require Human Judgment | 21 | +1 (P3-20) | 22 |
| 3. Obviously Needs Fixing | 42 | +20 (P3-1..16, P3-18..19, P3-21..22) | 62 |
| **Total distinct issues** | **77** | **+22** | **99** |

---

## Pass 3 Agent Sign-offs

- 🏃 **Bob (SM):** "Pattern F ('documentation written aspirationally, never maintained') is the root cause of the PO's first complaint. Pattern G ('PR template diverged from Style Guide') is the root cause of the second. Both are process failures that compounded across 45+ merged PRs."
- 📚 **Paige (Tech Writer):** "The testing documentation is the single biggest documentation debt. Every file name, API name, and directory structure reference is wrong. Recommendation: rewrite the testing docs from scratch using actual codebase state, rather than trying to patch 40+ individual references."
- 🧪 **Murat (TEA):** "P3-10 (unregistered @pytest.mark.unit) and P3-11 (zero E2E tests for documented commands) are the two highest-priority testing fixes. The unregistered marker is a latent failure; the missing E2E tests mean we have zero confidence that the documented toolchain actually works."
- 🏗️ **Winston (Architect):** "P3-1 (PR template missing design quality gates) explains why PEP 20 violations accumulated over 45+ PRs. The fix is mechanical (add 3 checkboxes), but the real remediation requires a codebase-wide PEP 20 review — which is Story 8.9 in the updated epic."
- 📋 **John (PM):** "P3-17 (NFR3 only 2/4 covered) is a product requirement gap, not just tech debt. The tutorial documents 'How to Add a Custom Metric' for a feature that doesn't exist. Either implement the metric registry or remove the tutorial claim."
- 🧪 **Quinn (QA):** "Pattern H ('test framework that doesn't exist') creates false confidence. A reviewer seeing 8 marker categories in the docs assumes comprehensive testing exists. In reality, only `smoke` and `integration` are meaningfully used."
