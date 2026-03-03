# Epic 8: Codebase Improvements & Technical Debt Resolution

## Overview

This epic addresses findings from the comprehensive multi-agent codebase audit conducted on 2026-03-02. Stories are ordered by category: Category 3 items (obviously need fixing) first, followed by a story to gather human input on Category 1 and 2 items.

**Source:** `_bmad-output/planning-artifacts/codebase-audit-report.md`, `codebase-audit-pass2-addendum.md`, and `codebase-audit-pass3-addendum.md`

---

## Story 8.1: Code Architecture Cleanup — Simulation Module Split & Kitchen Sink Refactors

**Priority:** High (Category 3 — Obviously Needs Fixing)
**Estimate:** Large

### Description

The `simulation.py` module at 1,291 lines contains 7+ distinct responsibilities and `dashboard/lib/filters.py` at 621 lines is a kitchen-sink module. Both violate SRP and make maintenance difficult.

### Acceptance Criteria

- [ ] `simulation.py` is split into: `bracket.py` (data structures, construction), `scoring.py` (ScoringRule protocol, registry, implementations), `providers.py` (ProbabilityProvider protocol, MatrixProvider, EloProvider), `simulation.py` (Monte Carlo + analytical engines, orchestrator)
- [ ] `dashboard/lib/filters.py` is split into: `data_loaders.py`, `simulation_helpers.py`, `export.py`; `filters.py` retains only sidebar filter logic
- [ ] `run_training()` God Function in `cli/train.py` decomposed into smaller functions (remove `noqa: PLR0913, C901, PLR0912` suppressions)
- [ ] All existing tests pass without modification (or with import path updates only)
- [ ] `ruff check .` and `mypy --strict src/ncaa_eval tests` pass

### Audit References
- 3.1 (simulation.py mega-module), 3.5 (filters.py kitchen sink), 3.6 (run_training God function)

---

## Story 8.2: Expose Public APIs & Eliminate Private Attribute Access

**Priority:** High (Category 3 — Obviously Needs Fixing)
**Estimate:** Medium

### Description

Multiple modules access private (`_`-prefixed) attributes of `EloFeatureEngine` and other classes across module boundaries. This creates implicit coupling that bypasses type safety.

### Acceptance Criteria

- [ ] `EloFeatureEngine` exposes public methods: `has_ratings() -> bool`, `set_ratings(ratings)`, `set_game_counts(counts)`, `predict_matchup(team_a, team_b) -> float`
- [ ] `feature_serving.py:301` uses `has_ratings()` instead of accessing `_ratings`
- [ ] `model/elo.py` uses `set_ratings()` and `set_game_counts()` instead of direct attribute assignment
- [ ] `simulation.py` calls public `predict_matchup()` instead of `_predict_one`
- [ ] `dashboard/lib/filters.py` uses a public `get_feature_importances()` method on Model ABC instead of `getattr(model, "_clf")`
- [ ] `splitter.py` imports a public `NO_TOURNAMENT_SEASONS` constant instead of `_NO_TOURNAMENT_SEASONS`
- [ ] `Calibrator` Protocol or ABC created for `IsotonicCalibrator` / `SigmoidCalibrator`
- [ ] `Scoring registry uses `dict[str, type[ScoringRule]]` instead of bare `dict[str, type]`
- [ ] All test assertions updated to use public APIs (e.g., `matchup_probability()` instead of `._P`)
- [ ] All existing tests pass; `mypy --strict` passes

### Audit References
- 3.4 (EloFeatureEngine private access x3), 3.10 (scoring registry), 3.11 (private constant import), 3.13 (dashboard _clf access), 3.15 (no calibrator ABC), 3.25 (tests access private ._P), Pattern B

---

## Story 8.3: Fix Data Pipeline Resilience — ESPN Error Handling, Retry Logic, Typer Decoupling

**Priority:** High (Category 3 — Obviously Needs Fixing)
**Estimate:** Medium

### Description

The data pipeline has multiple silent failure modes: ESPN exceptions swallowed at DEBUG level, no retry logic for network operations, SyncEngine coupled to `typer.echo`, hardcoded 2025 deduplication, and duplicate fuzzy match logic.

### Acceptance Criteria

- [ ] ESPN connector logs per-team failures at WARNING level with team name and exception message
- [ ] ESPN connector reports total success/failure count after batch fetch (e.g., "Fetched 340/362 teams, 22 failed")
- [ ] `tenacity` retry decorator added to ESPN per-team fetch (3 retries, exponential backoff)
- [ ] `SyncEngine` uses `logging` instead of `typer.echo` for all progress output; `typer` import removed from `ingest/sync.py`
- [ ] 2025 deduplication logic generalized: deduplicate any year where ESPN-prefix duplicates exist, not just `if year == 2025`
- [ ] Fuzzy match logic centralized in a single utility function used by both `sync.py` and `espn.py`
- [ ] `backtest.py:183` exception handler logs the caught exception at WARNING level before substituting NaN
- [ ] `repository.py:102` uses `pydantic.fields.PydanticUndefined` instead of `Ellipsis` as sentinel
- [ ] `rapidfuzz` added explicitly to `[tool.poetry.dependencies]` in `pyproject.toml`
- [ ] All tests pass

### Audit References
- 3.2 (SyncEngine/typer coupling), 3.3 (hardcoded 2025), 3.7 (duplicate margin cap), 3.8 (duplicate fuzzy match), 3.12 (backtest swallows exceptions), 3.14 (Ellipsis sentinel), 3.28 (ESPN silent exceptions), 3.29 (no retry logic), P2-1 (rapidfuzz undeclared)

---

## Story 8.4: Fix Docstring Style Violations & Documentation Gaps

**Priority:** Medium (Category 3 — Obviously Needs Fixing)
**Estimate:** Small

### Description

5 modules use NumPy-style docstrings instead of the Google-style mandate. Tutorial output is inaccurate. README is missing license section. No troubleshooting documentation exists.

### Acceptance Criteria

- [ ] All NumPy-style docstrings converted to Google-style in: `metrics.py`, `transform/elo.py`, `model/elo.py`, `model/base.py`, `model/tracking.py`
- [ ] `_resolve_team_id` in `espn.py` has a `Returns:` section in its docstring
- [ ] 28 functions with 3+ operations updated to include detailed docstring descriptions (see PEP 20 compliance report, Story 8.9 addendum — `_bmad-output/planning-artifacts/noncompliant-docstrings.md`)
- [ ] `docs/tutorials/getting-started.md` expected output updated to match actual CLI output
- [ ] Troubleshooting section added to user guide covering: Kaggle auth, ESPN rate limits, Parquet version mismatches
- [ ] README.md has a "## License" section referencing GPL-3.0
- [ ] User guide section on game theory sliders (lines 527-575) either removed or prominently marked as "NOT YET IMPLEMENTED" with a clear banner (not just a small note)
- [ ] Dashboard sidebar message updated to reference the canonical CLI command
- [ ] `ruff check .` passes

### Audit References
- 3.16 (NumPy-style docstrings x5), 3.17 (tutorial inaccuracy), 3.18 (no troubleshooting), 3.19 (no license in README), 3.20 (missing Returns), 2.19→3 (reclass: non-existent feature docs), P2-8 (sidebar CLI reference)

---

## Story 8.5: Testing Gaps — Missing Tests & Dead Code Cleanup

**Priority:** Medium (Category 3 — Obviously Needs Fixing)
**Estimate:** Medium

### Description

Several exported functions have zero test coverage, CLI tests only cover one model type, and there is dead test code.

### Acceptance Criteria

- [ ] `scoring_from_config` has parameterized tests covering all 5 dispatch branches + unknown-type error
- [ ] CLI training pipeline tested with XGBoost model type (in addition to existing logistic_regression tests)
- [ ] CLI training pipeline tested with Elo model type (stateful path)
- [ ] Dead `sample_game_records` fixture removed from `tests/conftest.py`
- [ ] Empty `tests/test_ncaa_eval.py` file removed
- [ ] Fibonacci scoring test added asserting actual point values match documented values
- [ ] All tests pass; no regressions

### Audit References
- 3.21 (scoring_from_config untested), 3.22 (CLI only tests LR), 3.23 (dead fixture), 3.24 (empty test file), 1.8 (Fibonacci values need test)

---

## Story 8.6: Type Safety & Configuration Improvements

**Priority:** Medium (Category 3 — Obviously Needs Fixing)
**Estimate:** Small

### Description

Several configuration values use stringly-typed fields and magic constants are duplicated.

### Acceptance Criteria

- [ ] `FeatureConfig` fields use `Literal` types or enums: `batch_rating_types`, `ordinal_composite`, `gender_scope`, `calibration_method`
- [ ] `DEFAULT_MARGIN_CAP` centralized in a single shared constants location (not duplicated in `graph.py` and `opponent.py`)
- [ ] Fibonacci scoring UI label displays actual point values (e.g., "Fibonacci (2-3-5-8-13-21)") so users are not misled regardless of which values PO chooses
- [ ] `ruff check .` and `mypy --strict src/ncaa_eval tests` pass

### Audit References
- 3.7 (duplicate constant), 3.9 (stringly-typed config), 1.8 partial (label must match code)

---

## Story 8.7: Sprint Housekeeping & CI/CD Improvements

**Priority:** Low (Category 3 — Obviously Needs Fixing)
**Estimate:** Small

### Description

Sprint status is stale, CI has quality gate divergence, and minor infrastructure issues need cleanup.

### Acceptance Criteria

- [ ] Sprint-status.yaml: all epic statuses (1-7) updated from `in-progress` to `done`
- [ ] Story 6.6 Dev Agent Record: `{{agent_model_name_version}}` template variable filled in
- [ ] `.github/workflows/main-updated.yaml`: `sphinx-apidoc` step added before `sphinx-build`
- [ ] `.github/workflows/main-updated.yaml`: `peaceiris/actions-gh-pages` upgraded to v4
- [ ] Committed `.ruff_cache` files in `template/` directory removed via `git rm --cached`
- [ ] Quality gate documented: either nox or pre-commit is declared canonical (with the other aligned or removed)

### Audit References
- 3.26 (epic statuses stale), 3.27 (story 6.6 incomplete), P2-2 (CI/nox divergence), P2-3 (deprecated action), P2-4 (docs build gap), P2-7 (committed cache), Pattern C

---

## Story 8.8: Dashboard UX Quick Fixes

**Priority:** Low (Category 3 — Obviously Needs Fixing)
**Estimate:** Small

### Description

Several minor but clearly-wrong dashboard UX issues.

### Acceptance Criteria

- [ ] Bracket renderer font sizes increased to be readable (minimum 12px for team names, 10px for probabilities)
- [ ] Dashboard home page shows prominent "Setup needed" message when no data exists (not just sidebar info)
- [ ] "Refresh Data" button added to sidebar to clear `st.cache_data` manually
- [ ] Breadcrumb navigation consistent across all pages
- [ ] Data freshness indicator added: show last sync date and latest game date in sidebar

### Audit References
- 3.30 (no freshness indicators), 3.31 (tiny bracket font), 3.32 (no first-run validation), 3.33 (no cache refresh), 3.34 (inconsistent breadcrumbs)

---

## Story 8.9: Add PEP 20, SOLID & Pure Function Gates to PR Template + Codebase PEP 20 Review

**Priority:** High (Category 3 — Obviously Needs Fixing)
**Estimate:** Large

### Description

The PR template is missing three quality gates that the Style Guide explicitly requires: PEP 20 compliance, SOLID principles, and Pure function design. Because these gates were never in the PR template, no PR was ever reviewed against them, which explains why many functions violate PEP 20. Additionally, the Style Guide's PEP 20 section covers only 5 of 19 aphorisms — critically missing #10 ("Errors should never pass silently"), which the codebase violates extensively (Pattern D).

### Acceptance Criteria

**PR Template Fixes:**
- [ ] `.github/pull_request_template.md` Code Quality section includes new checkbox: **PEP 20 compliance** — Simple (complexity ≤ 10), explicit (no magic numbers), readable (full domain words), flat (nesting ≤ 3), consistent with project patterns
- [ ] PR template includes new checkbox: **SOLID principles** — Single responsibility, open for extension, Liskov substitution, interface segregation, dependency inversion
- [ ] PR template includes new checkbox: **Pure function design** — Business logic is pure, side effects at edges, no I/O mixed with calculations

**Style Guide Expansion:**
- [ ] `docs/STYLE_GUIDE.md` Section 6 expanded to cover at minimum these additional PEP 20 aphorisms with project-specific examples and review checklist items:
  - #4 "Complex is better than complicated" — when complexity is the right choice
  - #6 "Sparse is better than dense"
  - #8/#9 "Special cases aren't special enough / practicality beats purity" — frames vectorization exceptions
  - #10/#11 "Errors should never pass silently / Unless explicitly silenced" — establishes error handling convention, references Pattern D
  - #12 "Refuse the temptation to guess" — reinforces mypy --strict
  - #17/#18 "Hard to explain = bad idea / Easy to explain = may be good" — reinforces complexity limits
- [ ] Style Guide Section 6 PEP 20 Code Review Checklist updated to include items for all newly-added aphorisms
- [ ] Style Guide Section 7 PR Checklist Summary table matches the updated PR template (all 9+ gates present in both)

**Style Guide Accuracy Fixes (P3-3 through P3-7):**
- [ ] Active Ruff Rules table updated: add `C90`, `PLR0911`, `PLR0912`, `PLR0913`
- [ ] Suppressed Rules table updated: add `PLR2004` with rationale
- [ ] Pydantic mypy plugin documented in Section 4
- [ ] Project layout diagram updated: add `cli/`, fix `model/` (not `models/`), fix `utils/` description, note test directory uses `unit/`/`integration/` not src-mirror
- [ ] ISP guidance updated: acknowledge ABCs as primary pattern, Protocols as complement
- [ ] Lint Suppression Policy subsection added: when `# noqa`/`# type: ignore` is acceptable, require specific error codes, escalation path

**Codebase PEP 20 Review:**
- [ ] All source files in `src/ncaa_eval/` reviewed for PEP 20 compliance; findings documented in a PEP 20 compliance report
- [ ] Specific attention to PEP 20 #10: all silent exception swallowing identified and logged (extends Pattern D beyond Pass 1/2 findings)
- [ ] For each violation found: either fix in this story (if small/mechanical) or document as a follow-up item with file:line reference
- [ ] `ruff check .` and `mypy --strict src/ncaa_eval tests` pass after all changes

### Audit References
- P3-1 (PR template missing 3 gates), P3-2 (PEP 20 5/19 aphorisms), P3-3 (Ruff table incomplete), P3-4 (Pydantic mypy undocumented), P3-5 (layout diagram), P3-6 (ISP claim), P3-7 (no suppression policy), Pattern G (PR template diverged from Style Guide)

---

## Story 8.10: Documentation Command E2E Integration Tests

**Priority:** High (Category 3 — Obviously Needs Fixing)
**Estimate:** Large

### Description

Zero E2E integration tests exist validating that documented toolchain commands actually work. The PO specifically flagged this: "the commands in the documentation all need to be covered with E2E integration tests to assure that they work properly. I got a bunch of errors when following the commands in the user guide." This story creates a dedicated test suite that validates every documented command exits successfully.

### Acceptance Criteria

**E2E Test Suite Creation:**
- [ ] New test file `tests/integration/test_documented_commands.py` created
- [ ] E2E test validates `pytest -m smoke` completes successfully and finishes in under 10 seconds
- [ ] E2E test validates `pytest` (full suite) completes with exit code 0
- [ ] E2E test validates `pytest --cov=src/ncaa_eval --cov-report=term-missing` produces coverage output
- [ ] E2E test validates `ruff check .` exits with code 0
- [ ] E2E test validates `ruff format --check .` exits with code 0
- [ ] E2E test validates `mypy --strict src/ncaa_eval tests` exits with code 0
- [ ] E2E test validates each `nox` session individually: `nox -s lint`, `nox -s typecheck`, `nox -s tests`
- [ ] E2E test validates `ncaa-eval --help` prints help text (CLI is importable and functional)
- [ ] E2E test validates `ncaa-eval sync --help` prints sync help
- [ ] E2E test validates `ncaa-eval train --help` prints train help
- [ ] E2E test validates `check-manifest` runs (or the check is removed from documentation if not configured)
- [ ] All E2E tests marked with `@pytest.mark.integration` and `@pytest.mark.slow`

**Documentation Fixes to Match Reality:**
- [ ] Any documented command that does NOT work is either: (a) fixed so it works, or (b) removed from documentation
- [ ] `docs/tutorials/getting-started.md` commands validated: each command runs and produces output matching the documented expected output (update expected output if different)

### Audit References
- P3-11 (zero E2E tests for documented commands), P3-12 (check-manifest not in pre-commit), PO feedback

---

## Story 8.11: Fix Testing Documentation Staleness & Marker Gaps

**Priority:** Medium (Category 3 — Obviously Needs Fixing)
**Estimate:** Medium

### Description

The testing documentation was written aspirationally during Story 1.3 and never updated as the codebase evolved. Every file name, API name, and directory structure reference is stale. Five documented marker categories have zero implementations. The `@pytest.mark.unit` marker is used 114 times but not registered. This story brings testing docs in sync with reality.

### Acceptance Criteria

**Testing Documentation Updates:**
- [ ] `docs/TESTING_STRATEGY.md` directory tree updated to match actual test file names
- [ ] `docs/testing/conventions.md` directory tree updated to match actual test file names
- [ ] All `docs/testing/` code examples updated to use actual API names (e.g., `ChronologicalDataServer` not `ChronologicalDataAPI`, `brier_score()` not `calculate_brier_score()`, `EloFeatureEngine.update_game()` not `update_elo_rating()`)
- [ ] `docs/testing/conventions.md` nox session example updated to match actual `noxfile.py` behavior
- [ ] `docs/testing/conventions.md` pre-commit hook example updated to match actual `.pre-commit-config.yaml`
- [ ] Smoke test time budget standardized across all documents (one consistent number)
- [ ] Fixture naming convention either updated to match reality or fixtures renamed
- [ ] All `docs/testing/` "Story X.Y" references updated to past tense (stories are complete)

**Marker Registration:**
- [ ] `@pytest.mark.unit` either: (a) registered in `pyproject.toml` markers list, or (b) removed from all 114 usages
- [ ] Documented markers with zero tests (`performance`, `regression`, `fuzz`, `mutation`, `slow`) — for each: either remove from documentation OR add at least one exemplar test demonstrating the category

**edgetest Cleanup:**
- [ ] `edgetest` either configured and functional, or removed from: (a) PR template, (b) all testing documentation, (c) pyproject.toml

### Audit References
- P3-8 (stale directory trees), P3-9 (5 markers with 0 tests), P3-10 (unregistered @pytest.mark.unit), P3-13 (time budget inconsistency), P3-14 (nox docs mismatch), P3-15 (wrong API names), P3-16 (fixture naming), P3-21 (edgetest), Pattern H (test framework on paper only)

---

## Story 8.12: Epics & Backlog Grooming — Track All Deferred Items

**Priority:** Medium (Category 3 — Process Gap)
**Estimate:** Small

### Description

The Post-MVP Backlog in epics.md is missing 15+ items that were deferred during implementation but never given backlog entries (Pattern A). Story 1.9 exists in sprint-status but not in epics.md. Several story ACs reference abandoned approaches.

### Acceptance Criteria

- [ ] All 15 deferred items from P3-18 added to the Post-MVP Backlog section of epics.md with: description, origin story, and priority estimate
- [ ] Story 1.9 added to epics.md (retroactive documentation of what was done)
- [ ] Story 3.2 AC updated to reflect matplotlib decision (not Plotly)
- [ ] Story 1.7 AC for edgetest marked as deferred or removed (cross-ref with Story 8.11)
- [ ] FR Coverage Map updated: NFR3 marked as "Partial — model and scoring registries only" with note that metric and feature-generator registries are in Post-MVP
- [ ] Architecture spec `specs/05-architecture-fullstack.md` annotated at top: "This document reflects initial design decisions. See the implementation and epics.md for current state."

### Audit References
- P3-17 (NFR3 partial), P3-18 (15+ missing backlog items), P3-19 (missing Story 1.9), P3-20 (stale architecture spec), P3-22 (stale Plotly AC), Pattern A (deferred but never tracked)

---

## Story 8.13: Gather PO Direction on Category 1 & 2 Items

**Priority:** Required — After Stories 8.1-8.12

### Description

The codebase audit (Pass 1, 2, and 3) identified 15 items requiring PO direction and 22 items requiring human judgment. This story gathers decisions on all of them in a single pass.

### Acceptance Criteria

- [ ] PO reviews the full audit report (`codebase-audit-report.md`, `codebase-audit-pass2-addendum.md`, and `codebase-audit-pass3-addendum.md`)
- [ ] For each Category 1 item (1.1-1.15, P3-17), PO provides one of: "Implement" (create follow-up story), "Accept as-is" (close with rationale), or "Defer" (add to post-MVP backlog with label)
- [ ] For each Category 2 item (2.1-2.21, P3-20), PO provides direction or delegates to technical lead
- [ ] Decisions documented in a PO decision log artifact
- [ ] Follow-up stories created for any items the PO approves for implementation

### Key Decisions Needed

1. **Game Theory Sliders** — Create implementation story using 7.7 spike findings?
2. **User-Editable Bracket** — Scope and priority for this core UX feature?
3. **Kaggle Submission Export** — Required for stated product mission?
4. **Model Ensemble/Blending** — In scope for this product?
5. **Demo/Sample Data** — Ship a bundled sample dataset for zero-setup onboarding?
6. **Feature Config CLI** — Add `--feature-config` to training CLI?
7. **team_a_won Label Bias** — Implement randomization or accept current mitigation?
8. **Fibonacci Scoring Values** — (1-1-2-3-5-8) or (2-3-5-8-13-21)?
9. **Metric Explorer Drill-Down** — Accept year-only or add round/seed/conference?
10. **Candidate Entry Flagging** — Still desired?
11. **CLI `predict` Command** — Build standalone prediction capability?
12. **Dashboard Type Checking** — Add mypy coverage for `dashboard/`?
13. **Coverage Threshold** — Set `--cov-fail-under=XX` in CI?
14. **NFR3 Metric & Feature-Generator Registries** — Implement plugin registries for metrics and feature generators, or accept model + scoring registries as sufficient? (P3-17)
15. **Architecture Spec Maintenance** — Update `specs/05-architecture-fullstack.md` to match implementation, or freeze as historical? (P3-20)

### Audit References
- All Category 1 items (1.1-1.15, P3-17), all Category 2 items (2.1-2.21, P3-20), Pattern A (deferred but never tracked)
