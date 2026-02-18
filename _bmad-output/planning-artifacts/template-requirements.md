# Cookie-Cutter Template Requirements

**Purpose:** Capture all learnings and decisions from NCAA_eval to create a reusable project template with Cruft support and BMAD integration.

**Status:** Living Document - Update throughout implementation
**Target:** Post-project completion - see related backlog story

---

## 1. Dev Stack & Architecture

### Core Technologies
- **Language/Runtime:** Python (confirmed)
- **Package Manager:** Poetry (confirmed - see pyproject.toml)
- **Python Version:** 3.12 — pin to 3.12.x; do NOT use 3.14 or latest (Discovered: Story 1.5 — see below)
- **Project Layout:** src layout (confirmed - see project structure)
- **Local Dev OS:** Linux/WSL required for full toolchain (mutmut 3.x Unix-only — see Lessons Learned)

### Key Dependencies
```yaml
# To be populated from pyproject.toml as project matures
production:
  - [dependency]: [version] # [reason for inclusion]

development:
  - pytest: "*" # Testing framework
  - pytest-cov: "*" # Coverage reporting (Discovered: Story 1.3)
  - hypothesis: "*" # Property-based and fuzz testing
  - mutmut: "*" # Mutation testing for test quality
  - [other dependencies]: [version] # [reason for inclusion]
```

**Discovered in Story 1.3 (2026-02-16):**
- pytest-cov is essential for coverage commands referenced in testing strategy
- Test marker taxonomy in pyproject.toml enables multi-dimensional test organization

**Discovered in Story 1.5 (2026-02-17):**
- `mutmut = "*"` — mutmut 3.x is **Windows-incompatible**: unconditionally imports `resource` (Unix-only stdlib). Requires Linux or WSL for local use. CI (ubuntu-latest) is unaffected.
- `poetry.lock` can go stale even when `pyproject.toml` is correct — always run `poetry lock --no-update && poetry install --with dev` after adding deps to ensure lock file is current (pytest-cov 7.0.0 was missing despite being in pyproject.toml)
- Always include `[tool.coverage.run]` alongside `[tool.coverage.report]` in pyproject.toml from day one — branch coverage is disabled by default and omitting this section means the coverage report never shows branch gaps

### Session Management (Nox) ⭐ (Discovered Story 1.6)

**Nox orchestrates the full quality pipeline** — running `nox` executes Ruff -> Mypy -> Pytest in order.

#### Nox + Conda/Poetry: Use `python=False` ⭐

When the project uses conda + Poetry (`POETRY_VIRTUALENVS_CREATE=false`), Nox sessions must **reuse the active environment** — not create per-session virtualenvs (which would duplicate the entire dependency tree and break conda-managed packages).

```python
# ✅ Correct: reuse active conda/Poetry environment
@nox.session(python=False)
def lint(session: nox.Session) -> None:
    """Run Ruff linting and formatting checks."""
    session.run("ruff", "check", ".", "--fix")
    session.run("ruff", "format", "--check", ".")

# ❌ Wrong: creates isolated virtualenv, must reinstall ALL deps per session
@nox.session
def lint(session: nox.Session) -> None:
    session.install("ruff")  # Slow, duplicative
    session.run("ruff", "check", ".", "--fix")
```

**Key rules for `python=False` mode:**
- Do NOT call `session.install()` — deprecated without a virtualenv
- Do NOT pass `external=True` to `session.run()` — not needed when `python=False`
- Set default sessions via `nox.options.sessions = ["lint", "typecheck", "tests"]`
- **Include `noxfile.py` in the typecheck session's mypy invocation** — otherwise the noxfile is only type-checked manually once at implementation time, not continuously. Add it as an explicit path: `session.run("mypy", ..., "src/pkg", "tests", "noxfile.py")` (Discovered: Story 1.6 code review)

#### Nox vs Pre-commit: Complementary, Not Redundant ⭐

| Aspect | Pre-commit | Nox |
|---|---|---|
| **When** | Automatic on `git commit` | Manual developer command |
| **Scope** | Only changed files (except mypy/pytest) | Entire project |
| **Tests** | Smoke tests only (`-m smoke`) | Full test suite |
| **Purpose** | Fast gate before commit | Comprehensive validation before PR |

**Template Pattern:** Include both. Pre-commit for fast automatic checks, Nox for thorough manual validation.

### Configuration Management
- Type checking: mypy with strict settings (see pyproject.toml)
- Linting: Ruff (confirmed - Story 1.4)
- Formatting: Ruff (confirmed - Story 1.4)

### Pre-Commit Hooks ⭐ (Discovered Story 1.4)

#### Critical Pattern: Local mypy hook required for src-layout projects

**DO NOT use `mirrors-mypy`** for projects with a local package. Use a local hook instead:

```yaml
# ❌ Wrong: mirrors-mypy cannot type-check tests that import local package
- repo: https://github.com/pre-commit/mirrors-mypy
  rev: v1.14.1
  hooks:
    - id: mypy
      files: ^(src/|tests/)
      additional_dependencies: [...]  # Can't add local packages!

# ✅ Correct: local hook uses Poetry virtualenv where package is installed
- repo: local
  hooks:
    - id: mypy
      name: mypy-strict
      entry: poetry run mypy
      args: [--strict, --show-error-codes, --namespace-packages]
      language: system
      types: [python]
      files: ^(src/|tests/)
      pass_filenames: false
```

**Rationale:** `mirrors-mypy` creates an isolated virtualenv. Local packages (not on PyPI) cannot be added as `additional_dependencies`. Test files that import the local package will fail with `[import-not-found]`. The `language: system` local hook uses the Poetry virtualenv where the local package is installed.

#### Hook Configuration Template

```yaml
default_install_hook_types: [pre-commit, commit-msg, pre-push]
default_stages: [commit, push]

repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    hooks:
      - id: end-of-file-fixer
      - id: trailing-whitespace
      - id: debug-statements
      - id: no-commit-to-branch
      - id: check-merge-conflict
      - id: check-toml
      - id: check-yaml
      - id: detect-private-key

  - repo: https://github.com/commitizen-tools/commitizen
    hooks:
      - id: commitizen
      - id: commitizen-branch
        stages: [post-commit, push]

  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff
        args: [--fix]
        types: [python]
      - id: ruff-format
        types: [python]

  - repo: local
    hooks:
      - id: mypy  # Use local hook (see pattern above)
        ...
      - id: pytest-smoke
        entry: poetry run pytest -m smoke --tb=short -q
        language: system
        types: [python]
        pass_filenames: false
        stages: [commit]
```

#### GitHub Actions CI Must Match the Toolchain ⭐ (Discovered Story 1.4 Code Review Round 2)

When migrating pre-commit hooks from one toolchain to another (e.g., Pipenv/invoke → Poetry), **update CI workflows at the same time**. The CI runs the same hooks; if the CI setup script doesn't match the new toolchain, it breaks silently or noisily on every PR.

**Pattern for GitHub Actions with Poetry:**
```yaml
- name: Set up Python 3.12          # Must match pyproject.toml python version!
  uses: actions/setup-python@v5
  with:
    python-version: "3.12"

- name: Install Poetry
  run: pip install poetry

- name: Install dependencies
  run: poetry install --with dev

- name: Run pre-commit hooks
  run: SKIP=no-commit-to-branch,commitizen-branch poetry run pre-commit run --all-files
```

**CI Python version MUST match pyproject.toml** — using Python 3.10 in CI when the project requires `>=3.12` causes subtle breakage (mypy strict mode, syntax errors, f-string features).

#### Ruff Rule Selection: Use Explicit Codes, Not Prefixes ⭐ (Discovered Story 1.4 Code Review Round 2)

`extend-select = ["PLR09"]` in Ruff selects the ENTIRE PLR09xx family, including rules with Ruff defaults that were never configured or documented:
- PLR0914 (too-many-locals, default: 15) — data science functions routinely hit this
- PLR0916 (too-many-boolean-expressions, default: 6) — pandas filter chains hit this

**Always use explicit rule codes** for rules with configurable thresholds:
```toml
# ❌ Selects 7+ rules, some with undocumented defaults
extend-select = ["PLR09"]

# ✅ Selects exactly the 3 rules with configured thresholds
extend-select = [
    "PLR0911",  # Too many return statements (configured: max-returns = 6)
    "PLR0912",  # Too many branches (configured: max-branches = 12)
    "PLR0913",  # Too many arguments (configured: max-args = 5)
]
```

#### Hooks to EXCLUDE from template ⚠️ (Discovered Story 1.4 Human Review)

**DO NOT include `codespell`** — false positive rate is too high with `--write-changes`:
- Corrupts BMAD bracket-notation syntax: `[M]ake` → `[M]ache`, `[M]ore` → `[M]or`
- Flags valid English/domain words: "wit" → "with", "ser" → "set"
- The `--write-changes` flag applies every "fix" silently without human review
- **Genuine typos are better caught by editor spell-check or human review**

**DO NOT include `blacken-docs`** — removes intentional formatting in documentation:
- Black provides no configuration option to preserve aligned inline comments
- Documentation code examples serve a pedagogical purpose; they should be formatted for human comprehension, not compiler compliance
- **Code examples in `.md` files are illustrative, not compiled — Black formatting is inappropriate**

#### "Style Sweep" Commit Pattern

When first activating pre-commit on an established repo, expect a large auto-fix commit:
- Run `pre-commit run --all-files` to apply all formatting fixes at once
- **Always `git diff` the result and review before staging** — treat auto-fix output like AI-generated code
- Commit separately as `style: auto-fix trailing whitespace, EOF, and newlines via pre-commit`
- Document all affected files in the story File List (may be 80+ files)
- ⚠️ Only trailing-whitespace, end-of-file-fixer, and similar non-semantic hooks are safe to apply in bulk without review

**Template Action Items:**
- [ ] Export final pyproject.toml as template base
- [ ] Document rationale for each major dependency
- [ ] Create default configurations for all tooling
- [ ] Include .pre-commit-config.yaml with local mypy hook pattern

---

## 2. Testing Strategy

**Reference:** [TESTING_STRATEGY.md](./TESTING_STRATEGY.md)

### Testing Framework
- Framework: Pytest (confirmed - Story 1.3)
- Coverage Target: 80% overall (module-specific: 75-95%, Story 1.3)
- Coverage Tool: pytest-cov (confirmed - Story 1.3)
- Coverage Philosophy: Signal, not gate - identify gaps, don't block PRs (Story 1.3)

### Test Organization
```
tests/
  ├── conftest.py     # Shared fixtures
  ├── unit/           # Unit tests (fast, isolated)
  ├── integration/    # Integration tests (I/O, external deps)
  └── fixtures/       # Test data files
```

**Discovered in Story 1.3 (2026-02-16):**

**Discovered in Story 1.5 (2026-02-17):**

### Coverage Configuration ⭐ (Discovered Story 1.5)

Always include **both** coverage sections in `pyproject.toml`. Omitting `[tool.coverage.run]` silently disables branch coverage:

```toml
[tool.coverage.run]
branch = true
source = ["src/your_package"]

[tool.coverage.report]
show_missing = true
exclude_lines = [
    'pragma: no cover',
    'def __repr__',
    'raise AssertionError',
    'raise NotImplementedError',
    'if __name__ == .__main__.:',
]
```

### Fixture Pattern: Ruff PT022 — `return` not `yield` ⭐ (Discovered Story 1.5)

Ruff PT022 flags `yield` in fixtures when there is no teardown code. Use `return` and update the return type annotation:

```python
# ❌ Triggers Ruff PT022 when no teardown exists
@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Iterator[Path]:
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    yield data_dir  # no teardown → PT022 violation

# ✅ Correct: return + concrete type (tmp_path handles cleanup automatically)
@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir
```

**Rationale:** pytest's `tmp_path` fixture handles cleanup automatically. No teardown = no need for `yield`. Using `return` also drops the `Iterator` import.

### Mutmut 3.x: Marker-Based Test Exclusion ⭐ (Discovered Story 1.5)

Mutmut 3.x runs pytest from a `mutants/` subdirectory. Tests that use `Path(__file__)` to navigate to the project root will fail because `Path(__file__).parent.parent.parent` resolves to `mutants/`, not the project root.

**Pattern:** Exclude these tests via a pytest marker, not `-k` name matching. Name-based exclusion silently breaks if the test is renamed; marker-based exclusion is self-maintaining.

```toml
# ✅ Correct: marker-based exclusion in [tool.mutmut]
pytest_add_cli_args_test_selection = ["tests/", "-m", "not no_mutation"]

# ❌ Fragile: breaks silently if test_src_directory_structure is renamed
pytest_add_cli_args_test_selection = ["tests/", "-k", "not test_src_directory_structure"]
```

```python
# Apply to any Path(__file__)-dependent structural test:
@pytest.mark.smoke
@pytest.mark.no_mutation  # Path(__file__)-dependent; incompatible with mutmut runner dir
def test_src_directory_structure() -> None: ...
```

```toml
# Register the marker in [tool.pytest.ini_options]:
markers = [
    # ... other markers ...
    "no_mutation: Tests incompatible with mutmut runner directory (Path(__file__)-dependent structural tests)",
]
```

### Mutmut 3.x: Expected Behavior ⭐ (Discovered Story 1.5)

- Creates `mutants/` directory AND `.mutmut-cache` file in project root — **add both to `.gitignore`**
- "Stopping early, could not find any test case for any mutant" = **expected** when target module has no business logic yet (e.g., only `__init__.py`)
- Exit code 1 from mutmut in this case is expected/normal
- Runs tests WITH `--strict-markers` (inherited from `pyproject.toml` `addopts`) — unknown markers will cause errors

### 4-Dimensional Testing Model ⭐
Tests are organized across four **orthogonal dimensions**:
1. **Scope** (What): Unit vs Integration
2. **Approach** (How): Example-based vs Property-based vs Fuzz-based
3. **Purpose** (Why): Functional vs Performance vs Regression
4. **Execution** (When): Tier 1 (pre-commit < 10s) vs Tier 2 (PR/CI full suite)

**Rationale:** Traditional "unit/integration only" taxonomy is insufficient. Orthogonal dimensions clarify test type selection and enable precise test filtering via markers.

### Test Marker Taxonomy ⭐
8 pytest markers defined in pyproject.toml:
- `smoke` - Pre-commit tests (< 10s total)
- `slow` - Excluded from pre-commit (> 5s each)
- `integration` - I/O or external dependencies
- `property` - Hypothesis property-based tests
- `fuzz` - Hypothesis fuzz-based tests for crash resilience
- `performance` - Speed/efficiency compliance
- `regression` - Prevent bug recurrence
- `mutation` - Mutation testing coverage

**Template Pattern:** Define markers in [tool.pytest.ini_options] with clear descriptions

### Hub-and-Spoke Documentation Architecture ⭐
Testing strategy uses 1 main document (TESTING_STRATEGY.md) + 7 focused guides:
- test-scope-guide.md (Unit vs Integration)
- test-approach-guide.md (Example/Property/Fuzz)
- test-purpose-guide.md (Functional/Performance/Regression)
- execution.md (4-tier execution model)
- quality.md (Mutation testing, coverage)
- conventions.md (Fixtures, markers, organization)
- domain-testing.md (Performance, data leakage)

**Rationale:** Improves navigability, reduces cognitive load, better GitHub UX vs single comprehensive document (60KB+)

### CI/CD Integration
- 4-tier quality gates: Pre-commit (< 10s) → PR/CI (full) → AI review → Owner review
- Pre-commit: lint, format, type-check, smoke tests
- PR/CI: full test suite, coverage report, selective mutation testing

**Template Action Items:**
- [ ] Create test structure skeleton
- [ ] Document testing conventions and patterns
- [ ] Provide example tests for each layer
- [ ] Export quality gate configurations

---

## 3. BMAD Integration

### BMAD Version
- Version used: [Document BMAD version at project start]
- Version at completion: [Document BMAD version at project end]
- Breaking changes handled: [Document any migration notes]

### Configuration
**Files to preserve:**
- `_bmad/bmm/config.yaml` - Core configuration structure
- [Other config files to template]

### Agent Modifications

#### Modified Agents
```yaml
# Document each agent modification
agent_name:
  file: _bmad/path/to/agent.md
  modifications:
    - [What changed]
    - [Why it changed]
    - [Date of change]
  preserve_in_template: yes/no
```

#### Custom Agents
```yaml
# Document any custom agents created
agent_name:
  file: path/to/custom/agent.md
  purpose: [What does this agent do]
  preserve_in_template: yes/no
```

### Workflow Modifications
```yaml
# Document workflow customizations
workflow_name:
  file: _bmad/path/to/workflow.yaml
  modifications: [List changes]
  rationale: [Why changed]
```

**Template Action Items:**
- [ ] Create update strategy for BMAD version management
- [ ] Document which BMAD modifications to preserve vs. reset
- [ ] Create hooks for BMAD updates

---

## 4. Project Structure

### Directory Layout
```
project-root/
├── _bmad/                    # [Document if customized]
├── _bmad-output/            # [Generated artifacts]
├── docs/                    # [Document required docs]
├── src/                     # [Document src structure]
├── tests/                   # [Document test structure]
└── [other directories]
```

### Required Documentation
- [ ] README.md template
- [ ] TESTING_STRATEGY.md template
- [ ] STYLE_GUIDE.md template
- [ ] [Other docs discovered as essential]

**Template Action Items:**
- [ ] Define minimal required structure
- [ ] Define optional structure with clear use cases
- [ ] Document structure evolution decisions

---

## 5. Cruft Configuration

### Cruft Setup
```yaml
# cookiecutter.json structure to support
default_context:
  project_name: "[project_name]"
  [other variables to parameterize]
```

### Update Strategy
- How often to sync from template: [Define cadence]
- Conflicts resolution: [Define approach]
- Version pinning: [Define strategy]

**Template Action Items:**
- [ ] Identify which files should update automatically
- [ ] Identify which files should never update (local only)
- [ ] Create .cruft.json configuration
- [ ] Document update procedure

---

## 6. Quality Standards

### Code Quality Gates
**Reference:** [TESTING_STRATEGY.md](./TESTING_STRATEGY.md) (QUALITY_GATES.md merged into testing strategy - Story 1.3)

- Type checking: mypy --strict (mandatory)
- Test coverage: 80% overall (75-95% module-specific, informational not blocking)
- Linting: ruff (configured in pyproject.toml)
- Documentation: Google-style docstrings, visual-first principle

**Discovered in Story 1.3 (2026-02-16):**

### 4-Tier Quality Gates ⭐
1. **Tier 1: Pre-commit** (< 10s total) - Lint, format, type-check, smoke tests
2. **Tier 2: PR/CI** (minutes) - Full test suite, coverage, mutation testing
3. **Tier 3: AI Review** - Docstring quality, architecture alignment, test quality
4. **Tier 4: Owner Review** - Functional correctness, strategic alignment

**Rationale:** Clear separation of concerns, fast feedback loop, comprehensive validation

### Visual-First Documentation Principle ⭐
"Pictures > Words" for human-facing documentation:
- Decision trees → Mermaid flowcharts
- Architecture docs → Diagrams
- Workflows → Sequence diagrams
- State machines → State diagrams

**Rationale:** Visuals improve comprehension vs text. Text acceptable for agent-facing docs and super-user technical documentation.

**Template Pattern:** PR checklist includes "Visual-first" requirement for user-facing docs

### PR Template Enforcement in Code Review ⭐
Code review workflow generates PRs following .github/pull_request_template.md structure:
- All template sections included (PR Type, Description, Checklists)
- Appropriate items marked [x] or N/A based on story type
- Code review notes appended at bottom
- Ensures consistency between manual and automated PRs

**Rationale:** Standardized PR format improves review process, ensures all checklist items are considered, maintains project quality standards.

**Template Pattern:** Code review workflow instructions.xml generates template-formatted PR bodies

**Discovered in Story 1.3 (2026-02-16):** PR #3 initially bypassed template, required manual reformatting and workflow update.

### Style Guide
**Reference:** [STYLE_GUIDE.md](./STYLE_GUIDE.md)

**Template Action Items:**
- [ ] Export final style guide
- [ ] Export quality gate configurations
- [ ] Create pre-commit hooks template

---

## 7. Lessons Learned

### What Worked Well

**Story 1.3 - Testing Strategy (2026-02-16):**
- ✅ **4-dimensional testing model** - Orthogonal dimensions (Scope/Approach/Purpose/Execution) clarify test type selection better than traditional taxonomy
- ✅ **Hub-and-spoke documentation** - 1 main doc + 7 focused guides improves navigability and reduces cognitive load vs single comprehensive document
- ✅ **Visual-first principle** - Mermaid flowcharts for decision trees significantly improve comprehension vs text-based trees
- ✅ **Coverage as signal, not gate** - Informational coverage targets avoid counterproductive test-padding while still identifying gaps
- ✅ **Documentation-first approach** - Defining testing strategy (Story 1.3) before implementation (Story 1.5) ensures alignment and prevents rework
- ✅ **Adversarial code review workflow** - Finding 3-10 specific issues per review ensures thorough validation and catches gaps

### What Didn't Work

**Story 1.4 - Code Quality Toolchain (2026-02-17 Human Review):**
- ❌ **`codespell` with `--write-changes`** — auto-applied incorrect "corrections" that corrupted BMAD template syntax and valid English/domain words. 11 corruptions across 7 files required manual revert. Removed from pre-commit config entirely.
- ❌ **`blacken-docs`** — reformatted Python code examples in all markdown docs, removing intentionally aligned inline comments (e.g., `st.integers(...)        # Integers in range`). Black provides no option to preserve alignment. Removed from pre-commit config entirely.
- ❌ **"Style Sweep" without diff review** — running `pre-commit run --all-files` and committing without reviewing the diff. Auto-fix hooks with write flags require human review before staging — treat like AI-generated code.

**Story 1.4 - Code Quality Toolchain (2026-02-17 Code Review Round 2):**
- ❌ **CI workflows not updated with pre-commit hook migration** — When `.pre-commit-config.yaml` was migrated from Pipenv/invoke to Poetry, the two GitHub Actions CI workflows that run those hooks were not updated. CI was left broken (Pipenv/Python-3.10 setup against a Poetry/Python-3.12 project). Always update CI when changing the toolchain.
- ❌ **PLR09 prefix over-selected Ruff rules** — `"PLR09"` in extend-select selects the entire PLR09xx family including PLR0914 (too-many-locals) and PLR0916 (too-many-boolean-expressions) which are not documented in STYLE_GUIDE.md and use Ruff defaults that will block legitimate data science code. Use explicit codes.
- ❌ **Library Requirements table not updated to match implementation** — The Dev Notes table still referenced `mirrors-mypy` after the implementation correctly switched to a local hook. Story documentation must be updated atomically with implementation changes.

**Story 1.6 - Configure Session Management (2026-02-18 Code Review):**
- ❌ **RST double backticks in Google-style docstrings** — Module-level docstrings used RST `` ``code`` `` syntax instead of Google-style single `` `code` ``. All Python files in this project use Google docstring style; single backticks are correct for inline code. Ruff does not catch this automatically.
- ❌ **noxfile.py excluded from automated mypy enforcement** — The typecheck session's mypy invocation scoped to `src/ncaa_eval tests`, leaving `noxfile.py` type-checked only once at implementation time. Fixed by adding `noxfile.py` as an explicit path to the mypy invocation in the typecheck session.

**Story 1.5 - Configure Testing Framework (2026-02-17):**
- ❌ **Mutmut 3.x is Windows-incompatible** — imports `resource` (Unix-only stdlib) unconditionally in `__main__.py`. Any `mutmut` invocation fails on Windows with `ModuleNotFoundError: No module named 'resource'`. Required switching to WSL for local mutation testing. **Template action: Document WSL as required for full local toolchain; note CI (ubuntu-latest) is unaffected.**
- ❌ **Python 3.14 is too new for this toolchain** — several dev dependencies (mutmut, certain hypothesis strategies) have not yet certified compatibility with Python 3.14 alpha/beta. **Template action: Pin CI and dev environments to Python 3.12 (latest stable LTS-equivalent); use `python = ">=3.12,<3.14"` as the pyproject.toml constraint until toolchain catches up.**
- ❌ **`[tool.coverage.run]` omitted** — Initial implementation only had `[tool.coverage.report]`. Branch coverage was silently disabled; the coverage report showed line coverage only. Found and fixed in code review. **Template action: Always scaffold both sections in the base pyproject.toml template.**
- ❌ **Mutmut exclusion via `-k` name match is fragile** — Initially used `-k "not test_src_directory_structure"`. Renamed tests silently break this exclusion, making all mutants appear "killed" due to the structural test failure. Replaced with `@pytest.mark.no_mutation` + `-m "not no_mutation"`. **Template action: Show marker-based exclusion pattern in template mutmut config.**

### Would Do Differently

**Story 1.3 - Testing Strategy (2026-02-16):**
- ⚠️ **Missing dependency caught in review** - pytest-cov was referenced extensively but not added to pyproject.toml until code review. Template should include essential test dependencies upfront.
- ⚠️ **Timing inconsistency across docs** - pyproject.toml said "< 5 seconds" while main doc said "< 10 seconds". Template should establish single source of truth for constraints.

**Story 1.5 - Configure Testing Framework (2026-02-17):**
- ⚠️ **WSL setup mid-story** — Mutmut Windows blocker required setting up WSL dev environment during story execution. Template should document WSL as a prerequisite upfront, not as a reactive fix.
- ⚠️ **`poetry.lock` staleness** — pytest-cov was in `pyproject.toml` but not installed because `poetry.lock` was out of date. Run `poetry lock --no-update` whenever deps change; don't assume lock file is current.

### Process Improvements

**Story 1.4 - Create Story Workflow (2026-02-16):**
- ✅ **Commit SM updates before dev-story** - Always commit story files and sprint-status.yaml updates immediately after create-story completes. This ensures dev-story starts with clean git status, enabling accurate tracking of implementation changes and preventing commit conflicts.
  - **Pattern:** `feat(sm): create Story X.Y and update sprint tracking`
  - **Files:** Story markdown file + sprint-status.yaml
  - **Timing:** After story creation, before invoking dev-story
  - **Rationale:** Clean separation between SM work (story planning) and Dev work (implementation). Git status becomes accurate indicator of actual code changes.

---

## 8. Template Implementation Checklist

### Phase 1: Extract
- [ ] Finalize this document with all decisions
- [ ] Extract all configuration files
- [ ] Document all custom modifications
- [ ] Identify parameterizable values

### Phase 2: Template Creation
- [ ] Create cookiecutter structure
- [ ] Parameterize project-specific values
- [ ] Create cruft configuration
- [ ] Write template README

### Phase 3: BMAD Integration
- [ ] Document BMAD version compatibility
- [ ] Create BMAD update hooks
- [ ] Test BMAD initialization in template

### Phase 4: Validation
- [ ] Generate test project from template
- [ ] Verify all features work
- [ ] Test cruft update flow
- [ ] Test BMAD integration

### Phase 5: Documentation
- [ ] Write template usage guide
- [ ] Document customization points
- [ ] Document update procedures
- [ ] Create troubleshooting guide

---

## 9. Template Metadata

```yaml
template_name: "bmad-python-project"
source_project: "NCAA_eval"
created_by: "Volty"
creation_date: "[Date when template is created]"
bmad_version_min: "[Minimum compatible BMAD version]"
bmad_version_max: "[Maximum tested BMAD version]"
python_version_min: "[Minimum Python version]"
```

---

## Next Steps

1. **During Implementation:** Update this document as you make decisions
2. **Weekly Review:** Scan for new patterns worth capturing
3. **Before Retrospective:** Do final comprehensive review
4. **Post-Project:** Execute the implementation checklist

**Related Artifacts:**
- Backlog Story: [Link when created]
- Project Docs: [docs/](./index.md)
- BMAD Config: [_bmad/bmm/config.yaml](../_bmad/bmm/config.yaml)

---

### Documentation: Sphinx + Furo for src-layout Projects ⭐ (Discovered Story 1.7 SM)

**`sphinx.ext.napoleon` is MANDATORY for Google-style docstrings** — without it, docstrings render as plain text. Since the project enforces Google style via `[tool.ruff.lint.pydocstyle] convention = "google"`, napoleon is always required alongside autodoc.

**`sys.path.insert` must point to `src/` for src-layout** in `docs/conf.py`:
```python
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
```

**`docs/conf.py` must NOT be added to mypy scope** — it uses Sphinx implicit globals and cannot be type-checked with `mypy --strict`.

**`docs` Nox session must NOT be in `nox.options.sessions`** — documentation is on-demand, not part of the Ruff → Mypy → Pytest quality pipeline.

### Cookiecutter Template Artifacts (Legacy Cleanup) ⭐ (Discovered Story 1.7 SM)

Projects based on `Lee-W/cookiecutter-python-template` contain stubs referencing the old `pipenv/invoke` toolchain:

| File | Problem | Action |
|---|---|---|
| `mkdocs.yml` | MkDocs stub; conflicts with Sphinx | Delete if using Sphinx |
| `CONTRIBUTING.md` | References `inv env.init-dev`, `inv git.commit` etc. | Rewrite to reflect Poetry/Nox/commitizen |
| `CHANGELOG.md` | Empty stub | Leave as-is; `cz bump --changelog` will populate |

**Template Action:** Include updated CONTRIBUTING.md and choose one documentation system (Sphinx or MkDocs) at project start — not both.

### check-manifest for Poetry Projects ⭐ (Discovered Story 1.7 SM)

check-manifest validates sdist distributions include all VCS-tracked files. Poetry projects need `[tool.check-manifest]` ignore patterns for non-distribution files (BMAD dirs, docs, config files). Include default ignore patterns in the base `pyproject.toml` template tuned for BMAD + Poetry projects.

### edgetest Scaffolding ⭐ (Discovered Story 1.7 SM)

For projects using `"*"` (unconstrained) versions, edgetest has limited value until dependency bounds are added. Configure scaffolding early but note the tool becomes meaningful only when bounds exist. Use `pytest tests/ -m smoke` as the edgetest command (fast, validates basic compatibility).

*Last Updated: 2026-02-18 (Story 1.7 SM - Sphinx/napoleon, cookiecutter cleanup, check-manifest, edgetest)*
*Next Review: [Set cadence - weekly? sprint boundaries?]*
