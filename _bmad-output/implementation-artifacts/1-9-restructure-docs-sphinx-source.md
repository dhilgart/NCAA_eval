# Story 1.9: Restructure docs/ as Pure Sphinx Source Directory

Status: done

## Story

As a developer,
I want `docs/` to be a pure Sphinx source directory with planning specs moved to a top-level `specs/` directory,
so that all documentation in `docs/` is processed by Sphinx and the directory has a single, clear purpose.

## Acceptance Criteria

1. **Given** Sphinx is configured in `docs/` (Story 1.7), **When** `nox -s docs` is run, **Then** STYLE_GUIDE.md, TESTING_STRATEGY.md, and all testing/ guides are rendered as HTML pages in the Sphinx output alongside the API reference.
2. **And** `docs/` contains only Sphinx source files (no excluded planning artifacts).
3. **And** planning specs live at `specs/` (project root) with `specs/archive/` for legacy documents.
4. **And** the Sphinx HTML navigation has three sections: Developer Guides, Testing Guides, and API Reference.
5. **And** `check-manifest` passes cleanly with updated ignore patterns.
6. **And** `nox` (full pipeline: lint, typecheck, tests) passes with no regressions.

## Tasks / Subtasks

- [x] Task 1: Add myst-parser dependency (AC: 1)
  - [x] 1.1: Add `myst-parser = "*"` to pyproject.toml dev dependencies
  - [x] 1.2: Run `poetry lock && pip install myst-parser`

- [x] Task 2: Move planning artifacts out of docs/ (AC: 2, 3)
  - [x] 2.1: `git mv docs/specs specs`
  - [x] 2.2: `git mv docs/archive specs/archive`

- [x] Task 3: Update Sphinx configuration (AC: 1, 2, 4)
  - [x] 3.1: Add `myst_parser` to extensions in docs/conf.py
  - [x] 3.2: Add `source_suffix` mapping for .rst and .md
  - [x] 3.3: Simplify `exclude_patterns` to `["_build"]`
  - [x] 3.4: Restructure docs/index.rst toctree with 3 sections

- [x] Task 4: Fix broken cross-references (AC: 1)
  - [x] 4.1: Update relative links in docs/TESTING_STRATEGY.md (specs moved)
  - [x] 4.2: Update display-text references in docs/STYLE_GUIDE.md (specs moved)

- [x] Task 5: Update configuration and BMAD artifacts (AC: 5)
  - [x] 5.1: Add `specs/**` to check-manifest ignore, remove redundant entries
  - [x] 5.2: Update epics.md inputDocuments paths
  - [x] 5.3: Update implementation-readiness-report paths
  - [x] 5.4: Update template-requirements.md docs pattern guidance

- [x] Task 6: Verification (AC: 1, 5, 6)
  - [x] 6.1: `nox -s docs` builds successfully with Markdown content in HTML output
  - [x] 6.2: `check-manifest` passes cleanly
  - [x] 6.3: `nox` (lint, typecheck, tests) passes with no regressions

## Dev Notes

### Origin

This story was identified during Story 1.7 code review when the "split personality" of `docs/` was flagged: half Sphinx source, half standalone Markdown. The `exclude_patterns` workaround added in 1.7 was a band-aid.

### Architecture Impact

Architecture spec Section 9 originally showed `docs/specs/`. After this story, specs are at project root. The architecture spec itself (now at `specs/05-architecture-fullstack.md`) is a living planning document and should be updated if needed in future stories.

### myst-parser

`myst-parser` enables Sphinx to process `.md` files natively. It converts Markdown to Sphinx doctree nodes, so Markdown files participate in toctree navigation, cross-referencing, and search indexing. No manual RST conversion needed.

### Files NOT to update

- Story files 1-1 through 1-7 (historical records, `[Source: docs/specs/...]` are provenance markers)
- noxfile.py (docs/ path unchanged)
- .gitignore (`docs/_build/` still correct)
- .github/pull_request_template.md (links to `../docs/STYLE_GUIDE.md` still valid)
- CONTRIBUTING.md (`nox -s docs` instructions still valid)
- _bmad/bmm/config.yaml (`project_knowledge: docs` still correct)

## Dev Agent Record

### Agent Model Used

claude-opus-4-6

### Completion Notes List

- `poetry lock --no-update` not supported in this Poetry version; used `poetry lock` instead
- `poetry install --with dev` failed with nvidia-nccl-cu12 conflict in conda env; used `pip install myst-parser` directly
- myst-parser generates `myst.xref_missing` warnings for Markdown anchor links and external file refs; suppressed via `suppress_warnings` in conf.py
- 3 remaining Sphinx warnings are mermaid code blocks in testing guides (harmless — no mermaid extension installed)

### File List

- pyproject.toml
- poetry.lock
- docs/conf.py
- docs/index.rst
- docs/TESTING_STRATEGY.md
- docs/STYLE_GUIDE.md
- specs/ (moved from docs/specs/ — 5 files)
- specs/archive/ (moved from docs/archive/ — 3 files)
- _bmad-output/planning-artifacts/epics.md
- _bmad-output/planning-artifacts/implementation-readiness-report-2026-02-15.md
- _bmad-output/planning-artifacts/template-requirements.md

### Change Log

- 2026-02-18: Implemented all 6 tasks, all verification passing
- 2026-02-18: Code review fixes — converted dead Sphinx xref links in TESTING_STRATEGY.md to text references; added explanatory comment to suppress_warnings in conf.py
- 2026-02-18: Code review #2 fixes — converted dead xref link for .github/pull_request_template.md in STYLE_GUIDE.md to text reference; suppressed mermaid Pygments warnings via misc.highlighting_failure in conf.py; Sphinx build now 0 warnings
