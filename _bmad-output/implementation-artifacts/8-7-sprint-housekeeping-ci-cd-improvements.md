# Story 8.7: Sprint Housekeeping & CI/CD Improvements

Status: review

## Story

As a **project maintainer**,
I want sprint-status metadata corrected, CI workflows aligned with the local quality pipeline, and minor infrastructure hygiene completed,
so that the project's tracking, CI, and documentation reflect the true state of the codebase.

## Acceptance Criteria

1. **Sprint-status.yaml epic statuses updated** — All completed epics (1-7) changed from `in-progress` to `done`
2. **Story 6.6 Dev Agent Record filled** — `{{agent_model_name_version}}` template variable in `_bmad-output/implementation-artifacts/6-6-implement-tournament-scoring-user-defined-point-schedules.md:239` replaced with actual model info: "Claude Opus 4.6 / Claude Sonnet 4.6 (code review)" (both models contributed per git history)
3. **CI `sphinx-apidoc` step added** — `.github/workflows/main-updated.yaml` runs `poetry run sphinx-apidoc -f -e -o docs/api src/ncaa_eval` before `sphinx-build` in the `publish-github-page` job (aligning CI with `noxfile.py` docs session)
4. **`peaceiris/actions-gh-pages` upgraded** — `.github/workflows/main-updated.yaml` updated from `peaceiris/actions-gh-pages@v3.8.0` to `peaceiris/actions-gh-pages@v4`
5. **`.ruff_cache` cleanup verified** — Confirm no `.ruff_cache` files tracked in `template/` directory; if any exist, remove with `git rm -r --cached`; ensure `.gitignore` covers `template/**/.ruff_cache` pattern
6. **Quality gate canonical declaration** — Document in `docs/STYLE_GUIDE.md` (or equivalent) that pre-commit is the canonical quality gate (it runs in CI and on every commit), while nox serves as the local convenience runner; align any gaps between the two (see Dev Notes for details)

## Tasks / Subtasks

- [x] Task 1: Update sprint-status.yaml epic statuses (AC: #1)
  - [x] 1.1 Change `epic-1` through `epic-7` from `in-progress` to `done`
  - [x] 1.2 Verify all stories within those epics are `done` (sanity check)
  - [x] 1.3 Leave `epic-8` as `in-progress` (still active)
  - [x] 1.4 Leave `epic-9` as `backlog` (not yet started)

- [x] Task 2: Fill Story 6.6 Dev Agent Record (AC: #2)
  - [x] 2.1 Open `_bmad-output/implementation-artifacts/6-6-implement-tournament-scoring-user-defined-point-schedules.md`
  - [x] 2.2 Replace `{{agent_model_name_version}}` at line 239 with `Claude Opus 4.6 (implementation) / Claude Sonnet 4.6 (code review)`

- [x] Task 3: Add `sphinx-apidoc` step to CI workflow (AC: #3)
  - [x] 3.1 In `.github/workflows/main-updated.yaml`, add a new step before "Build docs":
    ```yaml
    - name: Generate API docs
      run: poetry run sphinx-apidoc -f -e -o docs/api src/ncaa_eval
    ```
  - [x] 3.2 Verify the step order: Install deps -> Generate API docs -> Build docs -> Deploy

- [x] Task 4: Upgrade `peaceiris/actions-gh-pages` to v4 (AC: #4)
  - [x] 4.1 Change `peaceiris/actions-gh-pages@v3.8.0` to `peaceiris/actions-gh-pages@v4` on line 52

- [x] Task 5: Verify `.ruff_cache` cleanup (AC: #5)
  - [x] 5.1 Run `git ls-files template/ | grep ruff_cache` to confirm no tracked cache files
  - [x] 5.2 If any found, run `git rm -r --cached template/{{cookiecutter.project_slug}}/.ruff_cache`
  - [x] 5.3 Verify `.gitignore` contains `.ruff_cache` entry (should already be present)
  - [x] 5.4 **Pre-investigation result**: As of 2026-03-04, `git ls-files template/` returns empty — no tracked files exist in `template/` at all. This AC is already satisfied. Still verify `.gitignore` coverage.

- [x] Task 6: Document canonical quality gate (AC: #6)
  - [x] 6.1 Add a "Quality Gate Architecture" subsection to `docs/STYLE_GUIDE.md` Section 10 (or appropriate section)
  - [x] 6.2 Declare **pre-commit as canonical** (runs in CI via `.github/workflows/`, enforced on every commit)
  - [x] 6.3 Declare **nox as local convenience runner** (runs same checks via `nox` command, useful for manual full-suite runs)
  - [x] 6.4 Document the known divergences and their rationale:
    - nox mypy scope includes `noxfile.py` and `sync.py`; pre-commit mypy only checks `^(src/|tests/)` — this is intentional (pre-commit only checks changed files' scope)
    - nox runs `ruff format --check` (check-only); pre-commit runs `ruff-format` (auto-fix) — this is intentional (pre-commit should fix, nox should only check)
    - CI does NOT run nox (it runs pre-commit + standalone pytest) — this is the status quo and acceptable since pre-commit covers lint + typecheck + smoke tests
  - [x] 6.5 Optionally add `sync.py` and `noxfile.py` to the pre-commit mypy `files` regex to close the scope gap

- [x] Task 7: Run quality gates (all ACs)
  - [x] 7.1 `ruff check .` passes
  - [x] 7.2 `ruff format --check .` passes
  - [x] 7.3 `mypy --strict src/ncaa_eval tests` passes
  - [x] 7.4 `pytest` passes (full suite, 922 passed, 1 skipped)
  - [x] 7.5 `actionlint` passes on modified workflow file (if available locally) — not installed standalone; will be validated by pre-commit hook at commit time

## Dev Notes

### Current State Analysis

**CI Workflow** (`.github/workflows/main-updated.yaml`):
- Two jobs: `bump-version` (commitizen) and `publish-github-page` (Sphinx docs → GH Pages)
- Docs build is missing `sphinx-apidoc` step that `noxfile.py` includes → published API docs may be stale
- Uses `peaceiris/actions-gh-pages@v3.8.0` — outdated, v4 is current

**Quality Gate Landscape** (Pattern C from audit):
- **Pre-commit** (`.pre-commit-config.yaml`): ruff-lint, ruff-format, mypy-strict (src/tests only), pytest-smoke, commitizen, actionlint
- **Nox** (`noxfile.py`): lint (ruff check+format), typecheck (mypy on src/tests/noxfile/sync), tests (full pytest), docs (sphinx-apidoc+build)
- **CI**: Uses pre-commit + standalone pytest, does NOT run nox
- The divergence is documented but the "canonical" system was never officially declared

**`.ruff_cache` in template/**:
- Audit P2-7 flagged committed cache files, but as of current HEAD (3198aad), `git ls-files template/` returns empty
- The files were likely cleaned up during Story 8.1 (code architecture cleanup)
- Still verify `.gitignore` has `.ruff_cache` coverage

**Story 6.6 agent record**:
- File: `_bmad-output/implementation-artifacts/6-6-implement-tournament-scoring-user-defined-point-schedules.md`
- Line 239 has unfilled `{{agent_model_name_version}}`
- Git history shows: implementation commits by Claude Opus 4.6, code review commits by Claude Sonnet 4.6

### Key File Locations

| Purpose | File Path | Lines |
|---|---|---|
| Sprint status | `_bmad-output/implementation-artifacts/sprint-status.yaml` | 42-136 |
| Story 6.6 agent record | `_bmad-output/implementation-artifacts/6-6-implement-tournament-scoring-user-defined-point-schedules.md` | 239 |
| CI workflow | `.github/workflows/main-updated.yaml` | 1-57 |
| Noxfile | `noxfile.py` | 1-47 |
| Pre-commit config | `.pre-commit-config.yaml` | 1-79 |
| Style guide | `docs/STYLE_GUIDE.md` | TBD |
| Gitignore | `.gitignore` | TBD |

### Architecture & Convention Compliance

- **Conventional commits**: `chore(ci): ...` or `fix(ci): ...` for CI changes; `docs(sprint): ...` for status updates
- **No code changes to src/ or tests/** — this story is purely infrastructure/documentation
- **YAML edits must preserve comments** — sprint-status.yaml has STATUS DEFINITIONS block that must be kept intact
- **Workflow YAML**: use consistent indentation (2 spaces), follow existing step naming patterns

### Previous Story Intelligence (8.6)

Story 8.6 completed successfully with all quality gates passing (922/922 tests). Key learnings:
- The project has 922 tests as of story 8.6 completion
- `ruff check .`, `mypy --strict`, and full pytest suite all pass on current main
- Literal type aliases pattern established for FeatureConfig
- No regressions from 8.6 that affect this story

### Git Intelligence

Recent commits show Epic 8 stories being merged via PRs (squash merge pattern). The project follows:
- PR-based workflow with conventional commit titles
- Co-authored-by tags for Claude models
- Each story gets its own feature branch

### Risks & Gotchas

1. **YAML comment preservation**: When editing `sprint-status.yaml`, ensure the STATUS DEFINITIONS comment block (lines 7-34) is preserved exactly
2. **actionlint validation**: The modified workflow file should pass actionlint; the pre-commit config includes actionlint hook
3. **Template path escaping**: If Task 5 requires `git rm` on `template/{{cookiecutter.project_slug}}`, the curly braces must be escaped in shell commands
4. **Scope discipline**: This is a small housekeeping story — do NOT expand scope to fix other CI issues or add new quality gates beyond what the ACs specify

### References

- [Source: _bmad-output/planning-artifacts/codebase-audit-report.md — Finding 3.26 (epic statuses stale)]
- [Source: _bmad-output/planning-artifacts/codebase-audit-report.md — Finding 3.27 (story 6.6 incomplete)]
- [Source: _bmad-output/planning-artifacts/codebase-audit-pass2-addendum.md — P2-2 (CI/nox divergence, Pattern C)]
- [Source: _bmad-output/planning-artifacts/codebase-audit-pass2-addendum.md — P2-3 (deprecated peaceiris action)]
- [Source: _bmad-output/planning-artifacts/codebase-audit-pass2-addendum.md — P2-4 (missing sphinx-apidoc in CI)]
- [Source: _bmad-output/planning-artifacts/codebase-audit-pass2-addendum.md — P2-7 (committed .ruff_cache)]
- [Source: _bmad-output/planning-artifacts/epic-8-codebase-improvements.md — Story 8.7 definition]
- [Source: .github/workflows/main-updated.yaml — Current CI workflow]
- [Source: noxfile.py — Local quality gate runner]
- [Source: .pre-commit-config.yaml — Canonical quality gate]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

No debug issues encountered.

### Completion Notes List

- Updated epic-1 through epic-7 statuses from `in-progress` to `done` in sprint-status.yaml after verifying all constituent stories are `done`
- Replaced `{{agent_model_name_version}}` template variable in Story 6.6 with actual model info
- Added `sphinx-apidoc` step to CI workflow before docs build step, aligning CI with noxfile docs session
- Upgraded `peaceiris/actions-gh-pages` from v3.8.0 to v4
- Verified no `.ruff_cache` files tracked in `template/` directory; `.gitignore` already covers `.ruff_cache/`
- Added Section 11 "Quality Gate Architecture" to `docs/STYLE_GUIDE.md` declaring pre-commit as canonical and nox as local convenience runner
- Closed mypy scope gap by adding `noxfile.py` and `sync.py` to pre-commit mypy `files` regex
- All quality gates pass: ruff check, ruff format, mypy --strict, pytest (922 passed)

### Change Log

- 2026-03-04: Implemented all 7 tasks for Sprint Housekeeping & CI/CD Improvements

### File List

- `_bmad-output/implementation-artifacts/sprint-status.yaml` — Updated epic-1 through epic-7 to `done`, story 8-7 to `in-progress` then `review`
- `_bmad-output/implementation-artifacts/6-6-implement-tournament-scoring-user-defined-point-schedules.md` — Replaced agent model template variable
- `.github/workflows/main-updated.yaml` — Added sphinx-apidoc step, upgraded peaceiris/actions-gh-pages to v4
- `.pre-commit-config.yaml` — Extended mypy files regex to include `noxfile.py` and `sync.py`
- `docs/STYLE_GUIDE.md` — Added Section 11: Quality Gate Architecture
- `_bmad-output/implementation-artifacts/8-7-sprint-housekeeping-ci-cd-improvements.md` — Updated tasks, status, dev agent record
