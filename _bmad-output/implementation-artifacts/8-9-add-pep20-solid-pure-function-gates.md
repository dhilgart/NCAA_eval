# Story 8.9: Add PEP 20, SOLID & Pure Function Gates to PR Template + Codebase PEP 20 Review

Status: in-progress

## Story

As a developer,
I want the PR template to enforce PEP 20, SOLID, and Pure Function design quality gates that the Style Guide already requires, and the Style Guide expanded to cover critical missing PEP 20 aphorisms with a full codebase PEP 20 compliance review,
so that every future PR is reviewed against these design principles and existing violations are documented for remediation.

## Acceptance Criteria

### AC1: PR Template — Add Three Missing Quality Gates

1. `.github/pull_request_template.md` Code Quality section includes new checkbox: **PEP 20 compliance** — Simple (complexity <= 10), explicit (no magic numbers), readable (full domain words), flat (nesting <= 3), consistent with project patterns
2. PR template includes new checkbox: **SOLID principles** — Single responsibility, open for extension, Liskov substitution, interface segregation, dependency inversion
3. PR template includes new checkbox: **Pure function design** — Business logic is pure, side effects at edges, no I/O mixed with calculations
4. The three new checkboxes align exactly with the Style Guide Section 7 PR Checklist Summary table (gates must be identical in both locations)

### AC2: Style Guide Section 6 — Expand PEP 20 Aphorisms

5. `docs/STYLE_GUIDE.md` Section 6 expanded to cover at minimum these additional PEP 20 aphorisms with project-specific examples and review checklist items:
   - #4 "Complex is better than complicated" — when complexity is the right choice (e.g., simulation engine is legitimately complex)
   - #6 "Sparse is better than dense"
   - #8/#9 "Special cases aren't special enough / practicality beats purity" — frames vectorization exceptions (e.g., `# noqa: PLR2004` for data science magic values)
   - #10/#11 "Errors should never pass silently / Unless explicitly silenced" — establishes error handling convention, references Pattern D (ESPN DEBUG-level exceptions, backtest NaN substitution, hardcoded 2025 dedup)
   - #12 "Refuse the temptation to guess" — reinforces `mypy --strict` rationale
   - #17/#18 "Hard to explain = bad idea / Easy to explain = may be good" — reinforces complexity limits
6. Section 6 PEP 20 Code Review Checklist updated to include items for all newly-added aphorisms

### AC3: Style Guide Accuracy Fixes (P3-3 through P3-7)

7. Active Ruff Rules table updated: add `C90` (McCabe complexity), `PLR0911` (too many returns, max 6), `PLR0912` (too many branches, max 12), `PLR0913` (too many args, max 5)
8. Suppressed Rules table updated: add `PLR2004` (magic value comparison) with rationale "too aggressive for data science code with domain-specific constants"
9. Pydantic mypy plugin documented in Section 4 — cover `plugins = ["pydantic.mypy"]`, `init_typed = true`, `init_forbid_extra = true`, `warn_required_dynamic_aliases = true`
10. Project layout diagram updated: add `cli/` directory, fix `model/` (not `models/`), fix `utils/` description (contains `logger.py`, not "logging, assertions"), note test directory uses `unit/`/`integration/` not src-mirror
11. ISP guidance updated: acknowledge ABCs as the primary pattern (Model, Repository, Connector), Protocols as complement for structural typing (ProbabilityProvider, ScoringRule)
12. Lint Suppression Policy subsection added covering: when `# noqa`/`# type: ignore` is acceptable, require specific error codes (never bare `# noqa`), escalation path (refactor preferred over suppress), examples of good vs bad suppressions

### AC4: Style Guide Section 7 — PR Checklist Sync

13. Style Guide Section 7 PR Checklist Summary table matches the updated PR template — all gates present in both locations, wording identical

### AC5: Codebase PEP 20 Compliance Review

14. All source files in `src/ncaa_eval/` reviewed for PEP 20 compliance; findings documented in a PEP 20 compliance report at `_bmad-output/planning-artifacts/pep20-compliance-report.md`
15. Specific attention to PEP 20 #10 ("Errors should never pass silently"): all silent exception swallowing identified and documented — extends Pattern D beyond Pass 1/2 findings
16. For each violation found: either fix in this story (if small/mechanical — e.g., adding a missing warning log) or document as a follow-up item with `file:line` reference
17. `ruff check .` and `mypy --strict src/ncaa_eval tests` pass after all changes

## Tasks / Subtasks

- [x] Task 1: Update PR Template (AC: #1-3)
  - [x] 1.1 Read current `.github/pull_request_template.md`
  - [x] 1.2 Add three new checkboxes to Code Quality section
  - [x] 1.3 Verify checkbox wording matches Style Guide Section 7 exactly

- [x] Task 2: Expand Style Guide Section 6 — PEP 20 Aphorisms (AC: #5-6)
  - [x] 2.1 Read current `docs/STYLE_GUIDE.md` Section 6
  - [x] 2.2 Add aphorism #4 with project example (simulation engine complexity)
  - [x] 2.3 Add aphorism #6 with project example
  - [x] 2.4 Add aphorisms #8/#9 with vectorization exception framing
  - [x] 2.5 Add aphorisms #10/#11 with Pattern D error handling convention
  - [x] 2.6 Add aphorism #12 with mypy --strict rationale
  - [x] 2.7 Add aphorisms #17/#18 with complexity limit reinforcement
  - [x] 2.8 Update the PEP 20 Code Review Checklist with items for all new aphorisms

- [x] Task 3: Fix Style Guide Accuracy (AC: #7-12)
  - [x] 3.1 Update Active Ruff Rules table — add C90, PLR0911, PLR0912, PLR0913
  - [x] 3.2 Update Suppressed Rules table — add PLR2004 with rationale
  - [x] 3.3 Add Pydantic mypy plugin documentation to Section 4
  - [x] 3.4 Fix project layout diagram (add cli/, fix model/ singular, fix utils/ description, fix test structure)
  - [x] 3.5 Update ISP guidance — ABCs primary, Protocols complement
  - [x] 3.6 Add Lint Suppression Policy subsection

- [x] Task 4: Sync Section 7 PR Checklist (AC: #13)
  - [x] 4.1 Verify Style Guide Section 7 table matches updated PR template
  - [x] 4.2 Fix any mismatches between the two

- [ ] Task 5: Codebase PEP 20 Compliance Review (AC: #14-16)
  - [ ] 5.1 Audit all `src/ncaa_eval/` files for PEP 20 #10 violations (silent exception swallowing)
  - [ ] 5.2 Audit for PEP 20 #2 violations (magic numbers — reference `ruff check` PLR2004 results)
  - [ ] 5.3 Audit for PEP 20 #3/#5 violations (complexity/nesting — reference `ruff check` C901/PLR results)
  - [ ] 5.4 Fix small/mechanical violations directly (e.g., changing `logger.debug` to `logger.warning` for exception handlers)
  - [ ] 5.5 Write compliance report to `_bmad-output/planning-artifacts/pep20-compliance-report.md`

- [ ] Task 6: Final Verification (AC: #17)
  - [ ] 6.1 Run `ruff check .`
  - [ ] 6.2 Run `mypy --strict src/ncaa_eval tests`
  - [ ] 6.3 Run `pytest -m smoke` to verify no regressions

## Dev Notes

### Key Files to Modify

| File | Changes |
|------|---------|
| `.github/pull_request_template.md` | Add 3 checkboxes to Code Quality section |
| `docs/STYLE_GUIDE.md` | Section 4 (mypy/Pydantic), Section 6 (PEP 20 expansion), Section 7 (PR checklist sync), Section 10 (ISP fix), layout diagram, new Lint Suppression subsection |

### Key Files to Create

| File | Purpose |
|------|---------|
| `_bmad-output/planning-artifacts/pep20-compliance-report.md` | Codebase PEP 20 audit findings |

### Architecture Patterns and Constraints

- **No code logic changes in this story** — this is a documentation, template, and audit story. The only code-level changes are mechanical fixes to exception log levels (e.g., `logger.debug` -> `logger.warning`).
- **Pattern D** (from audit): Silent failure in data pipelines — ESPN connector, backtest metrics, 2025 dedup. Stories 8.3 fixes the actual code; this story documents the convention and gates to prevent recurrence.
- **Pattern G** (from audit): PR template diverged from Style Guide — this story closes that gap.

### Existing Content to Preserve

The Style Guide already has comprehensive sections for:
- **PEP 20 Section 6**: 5 aphorisms with examples — EXTEND, don't rewrite
- **Pure Functions Section 6.2**: Complete with good/bad examples, testing strategy table — NO CHANGES needed
- **SOLID Section 10**: 5 principles with examples and review checklist — ISP wording fix only
- **PR Checklist Section 7**: Has the 9-gate table — just needs checkbox wording sync

### Source Document References

- [Source: `_bmad-output/planning-artifacts/codebase-audit-pass3-addendum.md` — P3-1 (PR template), P3-2 (PEP 20), P3-3..P3-7 (accuracy fixes), Pattern G]
- [Source: `_bmad-output/planning-artifacts/codebase-audit-pass2-addendum.md` — Pattern D (silent failures)]
- [Source: `_bmad-output/planning-artifacts/codebase-audit-report.md` — Findings 3.12, 3.28 (specific exception swallowing)]
- [Source: `docs/STYLE_GUIDE.md` — Sections 4, 6, 7, 10, layout diagram]
- [Source: `.github/pull_request_template.md` — Current template with 7 code quality checkboxes]

### Ruff Configuration Context

Current `pyproject.toml` rules (active but undocumented in Style Guide):
```toml
[tool.ruff.lint]
select = ["I", "UP", "PT", "TID25", "C90", "PLR0911", "PLR0912", "PLR0913"]
ignore = ["E501", "D1", "D415", "PLR2004"]

[tool.ruff.lint.mccabe]
max-complexity = 10

[tool.ruff.lint.pylint]
max-returns = 6
max-branches = 12
max-args = 5
```

### mypy Pydantic Configuration (to document)

Current `pyproject.toml` settings:
```toml
[tool.mypy]
plugins = ["pydantic.mypy"]

[tool.pydantic-mypy]
init_typed = true
init_forbid_extra = true
warn_required_dynamic_aliases = true
```

### Known PEP 20 Violations (Pre-Identified)

**Pattern D — Silent Exception Swallowing (#10 "Errors should never pass silently"):**
- `src/ncaa_eval/evaluation/backtest.py:183` — bare `except Exception` substitutes NaN without logging
- `src/ncaa_eval/ingest/connectors/espn.py:141` — exceptions logged at DEBUG (should be WARNING)
- `src/ncaa_eval/ingest/connectors/espn.py:240` — exception returns None silently

**Magic Numbers (#2 "Explicit is better than implicit"):**
- `cli/train.py:150` — `0.95`, `0.05` (thresholds without names)
- `cli/train.py:208` — `2` (hardcoded multiplier)
- `evaluation/simulation.py` — multiple: `64`, `0.5`, `100`, `10_000`
- `transform/serving.py:185` — `2025` (hardcoded year, addressed by Story 8.3)
- Note: PLR2004 is intentionally suppressed; the compliance report should document each instance and assess whether a named constant is warranted

**Complexity (#3 "Simple is better than complex"):**
- `cli/train.py:73` `run_training()` — 70+ statements, 3 `noqa` suppressions (addressed by Story 8.1)

### Dependencies on Other Stories

- **Story 8.1** refactors `run_training()` (complexity) and splits `simulation.py` — this story documents the violations but does NOT fix the refactoring; it only documents them in the compliance report
- **Story 8.3** fixes ESPN exception handling, retry logic, and hardcoded 2025 dedup — this story establishes the convention that Story 8.3 implements
- **Story 8.4** fixes docstring style violations — orthogonal to this story
- **Story 8.7** addresses sprint housekeeping and CI — orthogonal

### What NOT to Do

- Do NOT rewrite existing PEP 20 or SOLID sections — only extend
- Do NOT fix `run_training()` complexity or `simulation.py` split — that's Story 8.1
- Do NOT add retry logic or fix ESPN connectors — that's Story 8.3
- Do NOT change any test assertions or test files
- Do NOT add new ruff rules or change `pyproject.toml` lint configuration
- Do NOT create mock data or example files — all examples in docs should use existing codebase patterns

### Project Structure Notes

- All documentation files: `docs/` directory
- PR template: `.github/pull_request_template.md`
- Planning artifacts: `_bmad-output/planning-artifacts/`
- Style Guide uses Markdown, integrated into Sphinx docs site via `myst_parser`

### Testing Strategy

This story has no testable code changes beyond verifying:
1. `ruff check .` passes (no new lint violations introduced)
2. `mypy --strict src/ncaa_eval tests` passes (no type regressions)
3. `pytest -m smoke` passes (no test regressions from any mechanical fixes)

If any exception log level changes are made (Task 5.4), verify the affected tests still pass.

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
