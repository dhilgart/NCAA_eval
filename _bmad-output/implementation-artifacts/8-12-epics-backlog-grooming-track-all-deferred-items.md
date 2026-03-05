# Story 8.12: Epics & Backlog Grooming — Track All Deferred Items

Status: done

## Story

As a project maintainer,
I want the Post-MVP Backlog in epics.md to be comprehensive and accurate,
so that all deferred work is tracked in one place with origin, description, and priority — and stale story ACs are corrected.

## Acceptance Criteria

1. All 15 deferred items from P3-18 added to the Post-MVP Backlog section of `epics.md` with: description, origin story, and priority estimate.
2. Story 1.9 retroactively added to Epic 1 in `epics.md` (document what was done — see `_bmad-output/implementation-artifacts/1-9-restructure-docs-sphinx-source.md` for the implemented story).
3. Story 3.2 AC updated to reflect the matplotlib decision (replace "Plotly for interactive inline rendering" with "matplotlib for static PNG rendering" — see Story 3.1 EDA notebook file-size lesson).
4. Story 1.7 AC for edgetest marked as deferred or removed (edgetest was removed from docs, PR template, and pyproject.toml in Story 8.11).
5. FR Coverage Map updated: NFR3 marked as "Partial — model and scoring registries only" with note that metric and feature-generator registries are in Post-MVP.
6. Architecture spec `specs/05-architecture-fullstack.md` annotated at top with banner: "This document reflects initial design decisions. See the implementation and epics.md for current state."

## Tasks / Subtasks

- [x] Task 1: Add 15 deferred items to Post-MVP Backlog in epics.md (AC: #1)
  - [x] 1.1 Add "Game Theory Slider Implementation" — origin: Stories 7.5/7.7, spike research in `specs/research/game-theory-slider-mechanism.md`
  - [x] 1.2 Add "User-Editable Bracket" — origin: UX Spec Flow 1
  - [x] 1.3 Add "Kaggle Submission Export" — origin: PRD mission statement
  - [x] 1.4 Add "Metric Explorer: Round/Seed/Conference Drill-Downs" — origin: Story 7.4
  - [x] 1.5 Add "Candidate Entry Flagging" — origin: Story 7.5
  - [x] 1.6 Add "CLI `predict` Command" — origin: PRD
  - [x] 1.7 Add "Model Ensemble/Blending" — origin: competitive necessity
  - [x] 1.8 Add "JSON Export for Pool Scorer" — origin: Story 7.6
  - [x] 1.9 Add "st.progress for Simulation" — origin: Story 7.6, UX Spec
  - [x] 1.10 Add "Per-Game Prediction Explainability" — origin: PRD
  - [x] 1.11 Add "Demo/Sample Data for Zero-Setup Onboarding" — origin: UX need
  - [x] 1.12 Add "Custom Metric Plugin Registry" — origin: NFR3 (PRD)
  - [x] 1.13 Add "Custom Feature Generator Plugin Registry" — origin: NFR3 (PRD)
  - [x] 1.14 Add "Confusion Matrix in Model Deep Dive" — origin: PRD 3.2
  - [x] 1.15 Add "Public Bracket Competitive ROI Simulation" — origin: UX Spec Flow 2

- [x] Task 2: Add Story 1.9 retroactively to epics.md (AC: #2)
  - [x] 2.1 Add "### Story 1.9: Restructure docs/ as Pure Sphinx Source Directory" after Story 1.8 in Epic 1 section
  - [x] 2.2 Write user story and acceptance criteria matching the implemented story (reference `1-9-restructure-docs-sphinx-source.md`)

- [x] Task 3: Fix stale Story ACs (AC: #3, #4)
  - [x] 3.1 In Story 3.2 AC, replace "all visualizations use Plotly for interactive inline rendering" with "all visualizations use matplotlib for static PNG rendering (Plotly inline outputs caused ~800MB notebook files — see Story 3.1 findings)"
  - [x] 3.2 In Story 1.7, either remove "And edgetest is configured for dependency compatibility testing" or append "(Deferred: removed in Story 8.11 — never automated in CI)"

- [x] Task 4: Update FR Coverage Map (AC: #5)
  - [x] 4.1 Change NFR3 row from "Epic 5" / "Plugin-registry extensibility" to: "Epic 5 (Partial)" / "Plugin-registry extensibility — model and scoring registries implemented; metric and feature-generator registries deferred to Post-MVP"

- [x] Task 5: Annotate architecture spec (AC: #6)
  - [x] 5.1 Add banner to top of `specs/05-architecture-fullstack.md`: "> **Note:** This document reflects initial design decisions from project planning (2026-02). The implementation has diverged in several areas. See `_bmad-output/planning-artifacts/epics.md` and the actual codebase for current state."

- [x] Task 6: Add any additional deferred items discovered during Epic 8 implementation (AC: #1)
  - [x] 6.1 Review AI code review follow-ups from Stories 8-1, 8-3 for orphaned items not covered by completed stories
  - [x] 6.2 Add any remaining uncovered items to Post-MVP Backlog with appropriate origin references

## Dev Notes

### Scope & Nature

This is a **documentation-only story** — no production source code changes. All edits are to planning artifacts (`epics.md`, `specs/05-architecture-fullstack.md`). No tests need to pass beyond existing `ruff check .` on docs.

### Key Source Documents

- **P3-18 item list**: `_bmad-output/planning-artifacts/codebase-audit-pass3-addendum.md` lines 167-186 — the definitive list of 15 missing backlog items
- **Existing Post-MVP Backlog**: `_bmad-output/planning-artifacts/epics.md` lines 972-1066 — current backlog items (model plugins, rating systems, data source connectors)
- **Epic 8 planning doc**: `_bmad-output/planning-artifacts/epic-8-codebase-improvements.md` — Story 8.12 spec with full AC list
- **Story 1.9 implementation**: `_bmad-output/implementation-artifacts/1-9-restructure-docs-sphinx-source.md` — the implemented story to retroactively add
- **Codebase audit reports**: `_bmad-output/planning-artifacts/codebase-audit-report.md`, `codebase-audit-pass2-addendum.md`, `codebase-audit-pass3-addendum.md`

### Post-MVP Backlog Entry Format

Follow the existing format in epics.md (lines 976-1065). Each entry should include:

```markdown
### <Item Title> (Origin: Story X.Y / PRD / UX Spec, <date>)

<1-2 sentence description of the deferred feature/capability.>

- **Effort:** <Low/Medium/High> — <brief justification>
- **Distinctness:** <how this differs from existing functionality>
- **Source:** <story or document reference>
- **Deferred because:** <why not included in MVP>
```

For simpler items (like CLI commands or UI enhancements), a shorter format is acceptable:
```markdown
### <Item Title> (Origin: Story X.Y)

<Description.>

- **Effort:** <estimate>
- **Source:** <reference>
- **Deferred because:** <reason>
```

### Files to Modify

1. `_bmad-output/planning-artifacts/epics.md` — 4 changes:
   - Add 15 entries to Post-MVP Backlog section (after line 1065)
   - Add Story 1.9 to Epic 1 section (after Story 1.8, ~line 257)
   - Fix Story 3.2 AC (line 381)
   - Fix Story 1.7 AC (line 238)
   - Update FR Coverage Map NFR3 row (line 85-86)
2. `specs/05-architecture-fullstack.md` — Add historical-document banner at top

### Existing Post-MVP Backlog Items (Do Not Duplicate)

The following items already exist in the backlog and should NOT be re-added:
- LightGBM Model Plugin
- CatBoost Model Plugin
- Glicko-2 & TrueSkill Model Plugins
- LSTM & Transformer Model Plugins
- Bayesian Logistic Regression Model Plugin
- LRMC Rating System
- TrueSkill / Glicko-2 Rating Systems
- Nate Silver / SBCB Elo Rating Scraping
- BartTorvik Direct Scraping
- Warren Nolan Scraping

### Additional Deferred Items from Epic 8 Code Reviews

These were identified during Epic 8 implementation and may warrant Post-MVP entries:

- **PLR0913 on `run_training()` public API** (Story 8.1 review): 7 keyword args could be bundled into a `DateRange` dataclass. Acknowledged tech debt via `# noqa: PLR0913`.
- **ESPN marker-file caching design flaw** (Story 8.3): `marker.touch()` runs even after partial failures, permanently caching incomplete data. Future improvement: `.espn_synced_{year}.json` metadata file.
- **Log format stripping level prefix** (Story 8.3 review): `sync.py` root logger uses `format="%(message)s"` — WARNING-level messages appear without "WARNING:" label.
- **Dashboard mypy coverage** (audit finding P2-6 / Category 2): `dashboard/` excluded from all type-checking. Awaiting PO direction in Story 8.13.

The dev should assess which of these merit a formal Post-MVP backlog entry vs. being tracked only in Story 8.13's Category 1/2 items.

### Risks & Gotchas

1. **Do NOT modify any `src/`, `tests/`, or `dashboard/` files** — this story is purely documentation/planning artifacts.
2. **Preserve existing backlog entry formatting** — new entries must match the style of existing entries (lines 976-1065).
3. **Story 1.7 edgetest AC** — Story 8.11 already removed edgetest from docs, PR template, and pyproject.toml. The AC in epics.md just needs to reflect that decision.
4. **NFR3 is a Category 1 item for Story 8.13** — do NOT resolve the question of whether metric/feature-generator registries should be implemented. Just update the FR Coverage Map to accurately reflect current state ("Partial").
5. **Line numbers are approximate** — they reflect the file state at story creation time and may shift if other stories edit `epics.md` concurrently. Use content-based search to locate the correct sections.

### References

- [Source: `_bmad-output/planning-artifacts/codebase-audit-pass3-addendum.md` — P3-17, P3-18, P3-19, P3-20, P3-22]
- [Source: `_bmad-output/planning-artifacts/epic-8-codebase-improvements.md` — Story 8.12 spec]
- [Source: `_bmad-output/planning-artifacts/codebase-audit-report.md` — Category 1 items 1.1-1.15, Pattern A]
- [Source: `_bmad-output/implementation-artifacts/1-9-restructure-docs-sphinx-source.md` — Story 1.9 implementation]
- [Source: `_bmad-output/implementation-artifacts/8-11-fix-testing-documentation-staleness-marker-gaps.md` — edgetest removal]

### Previous Story Intelligence (Story 8.11)

- **Scope**: Documentation-only changes. No production source files. Quality gates: ruff clean, mypy clean.
- **Key learning**: When updating documentation to match reality, be thorough — check ALL references, not just the obvious ones. Story 8.11 had to fix 40+ stale references across 8 docs files.
- **Marker cleanup pattern**: Story 8.11 removed `fuzz`/`mutation` markers from `pyproject.toml` and kept `performance`/`regression` as aspirational. This is the precedent for how to handle unused-but-planned items.
- **edgetest outcome**: Fully removed — dev dependency from pyproject.toml, `[tool.edgetest]` config, references in PR template, and all testing docs. This is the authoritative state.

### Git Intelligence

Recent commits are all Epic 8 squash merges (Stories 8.6-8.11). Key patterns:
- Conventional commit format: `type(scope): description`
- For documentation stories: `docs(scope): Story X.Y — <title>`
- All commits include `template-requirements.md` updates
- Story branch pattern: `story/8-12-epics-backlog-grooming-track-all-deferred-items`

## Dev Agent Record

### Agent Model Used

claude-opus-4-6

### Debug Log References

None — documentation-only story, no debugging required.

### Completion Notes List

- Added all 15 P3-18 deferred items to Post-MVP Backlog in epics.md, each with description, effort estimate, distinctness, source reference, and deferral reason
- Added 2 additional items from Epic 8 code reviews: PLR0913 run_training() API refactor (Story 8.1) and ESPN marker-file caching metadata (Story 8.3)
- Log format (Story 8.3) and dashboard mypy (P2-6) are already tracked under Story 8.13 Category 1/2 items — not added to Post-MVP Backlog to avoid duplication
- Story 1.9 retroactively added to Epic 1 section of epics.md with user story and acceptance criteria matching the implemented story
- Story 3.2 AC updated: Plotly → matplotlib with explanation of ~800 MB notebook file issue
- Story 1.7 edgetest AC marked with strikethrough and deferred annotation (removed in Story 8.11)
- FR Coverage Map NFR3 row updated to "Epic 5 (Partial)" with note about metric and feature-generator registries
- Architecture spec `specs/05-architecture-fullstack.md` annotated with historical-document banner
- Total Post-MVP Backlog now contains 27 items (10 existing + 15 from P3-18 + 2 from Epic 8 reviews)

### File List

- `_bmad-output/planning-artifacts/epics.md` (modified — 5 edits: Post-MVP Backlog additions, Story 1.9, Story 3.2 AC, Story 1.7 AC, FR Coverage Map NFR3)
- `specs/05-architecture-fullstack.md` (modified — historical-document banner added)
- `_bmad-output/implementation-artifacts/sprint-status.yaml` (modified — status update)
- `_bmad-output/implementation-artifacts/8-12-epics-backlog-grooming-track-all-deferred-items.md` (modified — task completion, Dev Agent Record)

### Change Log

- 2026-03-04: Implemented all 6 tasks — 17 Post-MVP Backlog entries added, Story 1.9 retroactively documented, stale ACs fixed, FR Coverage Map updated, architecture spec annotated
