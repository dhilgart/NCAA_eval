# Story 10.5: CI Bumpversion Fix & Output Directory Gitignore

Status: ready-for-dev

## Story

As a **developer**,
I want **the bumpversion CI job to succeed on every main push and the `output/` directory to be excluded from git**,
so that **version tags are automatically created and generated outputs don't pollute git status**.

## Acceptance Criteria

1. `output/` is added to `.gitignore` — `git status` no longer shows it as untracked after this change.
2. Root cause of the `bump-version` job failure in `.github/workflows/main-updated.yaml` is identified and fixed — the job succeeds on the next push to main.
3. No regression to the `publish-github-page` job in the same workflow.

## Tasks / Subtasks

- [ ] Task 1: Fix `.gitignore` — exclude `output/` directory (AC: #1)
  - [ ] 1.1: Add `output/` under the `# Project-specific` section of `.gitignore`

- [ ] Task 2: Investigate and fix the `bump-version` CI failure (AC: #2, #3)
  - [ ] 2.1: Inspect the actual error in the failing job: https://github.com/dhilgart/NCAA_eval/actions/runs/23031452020/job/66890478889
  - [ ] 2.2: Determine root cause — likely candidates: outdated `commitizen-tools/commitizen-action@0.27.1`, `tag_format` mismatch, or GitHub token permissions change
  - [ ] 2.3: Apply the fix to `.github/workflows/main-updated.yaml` (and/or `pyproject.toml` if commitizen config is the cause)
  - [ ] 2.4: Verify `publish-github-page` job is unaffected

## Dev Notes

### CI Workflow Context

- Workflow file: `.github/workflows/main-updated.yaml`
- The `bump-version` job uses `commitizen-tools/commitizen-action@0.27.1` — this is a pinned older version; check https://github.com/commitizen-tools/commitizen-action/releases for the latest stable release
- The `publish-github-page` job is independent (no `needs:` dependency on bump-version); fixing bump-version should not affect it
- Both jobs share the same `if: "!startsWith(github.event.head_commit.message, 'bump:')"` guard

### Commitizen Config (pyproject.toml lines 173–177)

```toml
[tool.commitizen]
name = "cz_conventional_commits"
version = "0.9.0"
tag_format = "$version"
version_files = ["pyproject.toml:version", "docs/conf.py:release"]
```

Known gotcha: the `commitizen-action` injects its own commitizen version — if there's a mismatch between the action's bundled `cz` and the `tag_format` or `version_scheme`, it can fail silently or with a cryptic error. Check the job logs carefully.

### `.gitignore` Fix

- Current state: `?? output/` appears in `git status` — the directory is generated output (e.g., from `ncaa-eval` CLI commands) and should never be committed
- Target: add `output/` under the `# Project-specific` section (around line 149–163 of `.gitignore`)
- The fix is one line; no other files need to change

### Project Structure Notes

- `.gitignore` is at repo root: `.gitignore`
- CI workflow files: `.github/workflows/main-updated.yaml`, `.github/workflows/python-check.yaml`
- `pyproject.toml` is at repo root

### References

- Failing CI job: https://github.com/dhilgart/NCAA_eval/actions/runs/23031452020/job/66890478889
- commitizen-action releases: https://github.com/commitizen-tools/commitizen-action/releases
- [Source: .github/workflows/main-updated.yaml]
- [Source: pyproject.toml#tool.commitizen]
- [Source: .gitignore#Project-specific]

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
