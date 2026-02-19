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
- `poetry.lock` can go stale even when `pyproject.toml` is correct — always run `poetry lock && poetry install --with dev` after adding deps to ensure lock file is current (pytest-cov 7.0.0 was missing despite being in pyproject.toml). **Note: Poetry 2.x removed `--no-update`; use `poetry lock` alone.**
- `poetry.lock` must be regenerated after ANY constraint change, not just additions — changing `>=0.15` to `>=0.15,<2` is sufficient to break CI with: *"pyproject.toml changed significantly since poetry.lock was last generated"*. This applies to both dev agents AND code review agents that modify pyproject.toml as part of fixes. (Discovered: Story 2.4 Code Review × 2)
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

#### `pre-commit install` Is Required After Clone ⭐ (Discovered Story 2.2 Code Review #2)

Having `.pre-commit-config.yaml` in the repo does NOT mean hooks are active. The hooks must be explicitly installed into `.git/hooks/` via `pre-commit install`. Without this, commits bypass all hooks silently. CI catches violations via `pre-commit run --all-files` (which doesn't require installation), creating a gap where CI fails but local commits succeed.

**Template Action:** Add `pre-commit install` to the post-clone setup instructions in README and CONTRIBUTING.md. Consider adding a check in the Nox lint session:

```python
# In noxfile.py lint session — warn if hooks not installed
import os
if not os.path.exists(".git/hooks/pre-commit"):
    session.warn("Pre-commit hooks not installed! Run: pre-commit install")
```

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

**Story 2.1 - Data Source Evaluation Spike (2026-02-19 Code Review):**
- ⚠️ **Spike package validation blocked by credentials/subscriptions** — Task 2 mandated live testing of `kaggle` and `kenpompy`, both blocked by infrastructure (missing API key, no subscription). Template for data source evaluation spikes should include a "credentials checklist" upfront so blockers are resolved before dev-story begins, not discovered mid-spike.
  - **Pattern:** Add a "Pre-Spike Prerequisites" section listing required credentials, accounts, and subscriptions before dev-story is invoked.

- ⚠️ **Rate limits are a required AC dimension but easy to omit** — AC 1 for data source evaluations explicitly requires "rate limits and terms of service" per source, but several sources had no rate limit documentation initially. When building data source evaluation spikes, rate limits should be treated as a required field — even "not documented" or "not applicable" are valid answers that must be explicit.

- ⚠️ **"Items requiring live verification" need specific test procedures, not just blockers** — Research spike subtasks that produce a live verification list should require runnable commands, not just descriptions of what's blocked. Template should explicitly require: `- Blocker description AND specific shell command to run when unblocked`.

- ✅ **Adversarial code review catches undocumented test gaps** — The review caught that `sportsdataverse` (Priority 3 recommendation) was never live-tested, and BartTorvik's core metrics were never retrieved via cbbpy. For documentation/research spikes, code review still finds valuable gaps — it just shifts from code quality to claim validation and documentation completeness. Confirmed: adversarial review works for spikes, not just code stories.

**Story 2.1 - Data Source Evaluation Spike (2026-02-19 Code Review Round 3):**
- ❌ **Spike stories must include a decision-gate AC for scoping choices** — The dev agent selected 4 specific data sources for MVP (committing them to epics.md) without human stakeholder approval. The spike ACs only required "recommended priority order" — not a final selection. Scoping decisions that directly affect downstream stories (2.2-2.4 in this case) must be gated by human review. **Template pattern for future spike stories:**
  ```
  **And** the product owner reviews the spike findings and approves which
  [sources/technologies/approaches] to include in the MVP scope before
  downstream stories begin implementation.
  ```
  - **Applies to:** Stories 4.1 (feature techniques), 5.1 (modeling approaches), 6.4 (simulation confidence), 7.7 (slider mechanism), and any future spike story whose output constrains downstream implementation scope.
  - **Rationale:** Spikes produce recommendations. Decisions belong to the product owner. The dev agent must present options with trade-offs, not commit selections unilaterally. (Discovered: Story 2.1 Code Review Round 3)

- ❌ **Do not select untested components for MVP scope** — sportsdataverse-py was marked "⚠️ Not performed — package not tested during this spike" in the research document yet was selected as MVP Source #3. Similarly, Warren Nolan was categorized as "Deferred Scrape-Only" in the research recommendations but promoted to MVP Source #4. Selections should be consistent with the evidence gathered during the spike. (Discovered: Story 2.1 Code Review Round 3)

---

## 8. Cookie-Cutter Improvements Feedback Loop

### `cookie-cutter-improvements.md` — Required Template Artifact

The cookie-cutter template MUST ship with an empty `cookie-cutter-improvements.md` file at the project root (or `_bmad-output/planning-artifacts/`). This file serves the same role for **future projects built from the template** that `template-requirements.md` serves for NCAA_eval: a living document where the team captures learnings, conventions, and gotchas that should flow back upstream into the cookie-cutter template.

**Template content (shipped empty with scaffold):**
```markdown
# Cookie-Cutter Improvements

Learnings discovered in this project that should be contributed back to the
cookie-cutter template. Review periodically and submit upstream PRs.

---

<!-- Add entries below as: ### Category / #### Finding Title / description -->
```

**Hooks:** Every workflow/agent that currently writes to `template-requirements.md` must also be configured to write to `cookie-cutter-improvements.md` in projects generated from the template. Specifically:
- **Code review workflow** (`_bmad/bmm/workflows/4-implementation/code-review/instructions.xml`, Step 4.5) — the template learning capture step
- **Any agent** following the MEMORY.md rule: *"All agents must automatically add learnings/discoveries to template-requirements.md when they identify patterns, gotchas, or conventions"*
- **Retrospective workflows** — if they produce template-worthy findings

**Important distinction:**
- `template-requirements.md` = learnings for building the ORIGINAL template (NCAA_eval → cookie-cutter)
- `cookie-cutter-improvements.md` = learnings from USING the template in a new project (new-project → upstream template PR)

(Discovered: Story 2.1 Code Review Round 3, user request)

---

## 9. Template Implementation Checklist

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
- [ ] Include empty `cookie-cutter-improvements.md` with scaffold content
- [ ] Configure BMAD hooks to write to `cookie-cutter-improvements.md` instead of `template-requirements.md`

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

## 10. Template Metadata

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

### Commitizen `version_files` Must Track All Version Strings ⭐ (Discovered Story 1.7 Code Review)

When Sphinx `docs/conf.py` hardcodes `release = "0.1.0"`, it **must** be listed in commitizen's `version_files` or it drifts silently on every `cz bump`:

```toml
[tool.commitizen]
version_files = ["pyproject.toml:version", "docs/conf.py:release"]
```

**Template Action:** Always include `docs/conf.py:release` in `version_files` when Sphinx is configured.

### Keep `docs/` as Pure Sphinx Source ⭐ (Updated Story 1.9)

Keep `docs/` as a pure Sphinx source directory. Move planning specs and archives to a top-level `specs/` directory. Use `myst-parser` to integrate Markdown developer guides (style guide, testing strategy) directly into the Sphinx toctree. This eliminates the need for defensive `exclude_patterns` beyond `_build`.

```python
# docs/conf.py — minimal exclude_patterns when docs/ is pure source
exclude_patterns = ["_build"]
```

**Template Action:** Add `myst-parser` to dev dependencies alongside Sphinx and Furo. Keep `docs/` as pure Sphinx source from project start. Planning specs go in `specs/` at project root.

### myst-parser Requires `suppress_warnings` for Markdown TOC Links (Discovered Story 1.9 Code Review)

When myst-parser processes Markdown files that contain TOC anchor links (e.g., `[section](#section)`) or references to files outside the Sphinx source tree, it generates `myst.xref_missing` warnings. These are valid Markdown constructs (they work on GitHub) but have no Sphinx equivalent. Suppression is mandatory — not a band-aid:

```python
# docs/conf.py — required when using myst-parser with existing Markdown docs
suppress_warnings = ["myst.xref_missing", "misc.highlighting_failure"]
```

**Template Action:** Always include both suppressions in `docs/conf.py` when `myst_parser` is in extensions: `myst.xref_missing` (Markdown anchor links and out-of-tree refs) and `misc.highlighting_failure` (fenced code blocks with unsupported lexers like `mermaid`). Add explanatory comments so future developers don't remove them.

### Use Text References for Out-of-Tree Files in Sphinx Markdown (Discovered Story 1.9 Code Review)

Markdown links to files outside the Sphinx source tree (e.g., `[specs/file.md](../specs/file.md)` from `docs/`) render as dead `<span class="xref myst">` with no `href` in Sphinx HTML. Use plain backtick-quoted text references instead:

```markdown
# BAD — dead link in Sphinx HTML (works on GitHub only)
- [`specs/architecture.md`](../specs/architecture.md) - Architecture docs

# GOOD — renders as styled code text in both GitHub and Sphinx
- `specs/architecture.md` - Architecture docs
```

**Template Action:** In Markdown files processed by Sphinx, use backtick-quoted text for any reference to files outside `docs/`. Reserve Markdown links for files within the Sphinx source tree.

### Spike Research Outputs Belong in `specs/research/`, Not `docs/` (Discovered Story 2.1)

Spike stories that produce research documents (e.g., data source evaluations, technology comparisons, feasibility analyses) are planning artifacts, not developer documentation. Their output belongs in `specs/research/`, not `docs/research/`. Placing them in `docs/` violates the "pure Sphinx source" principle from Story 1.9 — Sphinx would either need `exclude_patterns` to hide them or they'd appear in the built documentation as unfinished planning notes.

```
# BAD — planning artifact pollutes Sphinx source tree
docs/research/data-source-evaluation.md

# GOOD — planning artifact in specs/ alongside other planning docs
specs/research/data-source-evaluation.md
```

**Template Action:** When creating spike stories, set the output path to `specs/research/<document-name>.md`. Reserve `docs/` exclusively for content that should appear in the built Sphinx documentation.

### Don't Import Private Symbols from Implementation in Tests ⭐ (Discovered Story 1.8 Code Review)

Tests importing `_PRIVATE_NAMES` from implementation modules create fragile coupling — if the internal is renamed, tests break for non-behavioral reasons. Prefer:

1. **Hardcode observable values** — if the behavior is documented (e.g., logger name `"ncaa_eval"` is documented in the module docstring), hardcode it in tests. It IS the observable API.
2. **Test behavior, not internals** — instead of asserting `handler.formatter._fmt == _LOG_FORMAT`, emit a log message and assert the captured output contains the expected pipe-delimited components.
3. **If you must access private stdlib attributes** (like `logging.Formatter._fmt`), that's a red flag the test is over-specified. Rewrite to test observable output.

```python
# ❌ Fragile: imports private impl details + accesses private stdlib attr
from ncaa_eval.utils.logger import _LOG_FORMAT, _ROOT_LOGGER_NAME
assert handler.formatter._fmt == _LOG_FORMAT

# ✅ Tests observable behavior
log = get_logger("fmtcheck")
log.info("format-test")
captured = capsys.readouterr()
assert " | ncaa_eval.fmtcheck | " in captured.err  # pipe-delimited format
assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", captured.err)  # timestamp
```

### Shared Autouse Fixture vs Repeated `teardown_method` ⭐ (Discovered Story 1.8 Code Review)

When multiple test classes in the same file need the same cleanup (e.g., resetting a singleton logger), use a **module-level autouse fixture** instead of repeating `teardown_method` in each class:

```python
# ❌ Repeated teardown in three classes (DRY violation)
class TestA:
    def teardown_method(self) -> None:
        root = logging.getLogger("ncaa_eval")
        root.handlers.clear(); root.setLevel(logging.WARNING)

class TestB:
    def teardown_method(self) -> None:  # identical copy
        ...

# ✅ Single autouse fixture applies to entire module (including classes without teardown)
@pytest.fixture(autouse=True)
def _reset_ncaa_eval_logger() -> Iterator[None]:
    """Reset the ncaa_eval root logger between every test in this module."""
    yield
    root = logging.getLogger("ncaa_eval")
    root.handlers.clear()
    root.setLevel(logging.WARNING)
```

Note: Use `yield` + teardown code. Ruff PT022 only flags `yield` when there is NO teardown code — using `yield` with cleanup after it is correct and expected.

### Assertion Functions Must Raise `ValueError` (not `KeyError`) for Missing Columns ⭐ (Discovered Story 1.8 Code Review)

Any assertion function that accepts column name(s) and accesses `df[col]` must validate column existence first. Letting a `KeyError` propagate violates AC-level requirements for "clear error messages":

```python
# ❌ df[col] raises KeyError for missing column — confusing to users
def assert_dtypes(df: pd.DataFrame, expected: Mapping[str, str | type]) -> None:
    for col, dtype in expected.items():
        actual = str(df[col].dtype)  # KeyError if col not in df!

# ✅ Validate first, raise ValueError with descriptive message
def assert_dtypes(df: pd.DataFrame, expected: Mapping[str, str | type]) -> None:
    missing_cols = set(expected.keys()) - set(df.columns)
    if missing_cols:
        msg = f"assert_dtypes failed: columns not found in DataFrame: {missing_cols}"
        raise ValueError(msg)
    for col, dtype in expected.items():
        ...
```

This pattern applies to `assert_dtypes`, `assert_no_nulls` (specific-columns mode), and `assert_value_range`.

### `follow_imports = "silent"` Does NOT Suppress `[import-untyped]` Under Mypy `--strict` ⭐ (Discovered Story 1.8)

Despite the mypy docs, `follow_imports = "silent"` in `[tool.mypy]` does not suppress `[import-untyped]` errors when `--strict` mode is active. The `--strict` flag re-enables this check. Fix: use targeted `# type: ignore[import-untyped]` on the import line itself:

```python
import pandas as pd  # type: ignore[import-untyped]
```

Add this to every file that imports pandas (or other untyped third-party libs). Do NOT install `pandas-stubs` — the project explicitly chose not to.

### Library-First: Surface Existing Libraries Before Reimplementing ⭐ (Discovered Story 1.8 Post-Review)

Before writing a custom implementation for any common data engineering concern, explicitly surface whether a battle-tested library already exists — even if it adds a dependency. Custom code is only justified when no library exists, the library is too heavy, or the story explicitly prohibits new deps.

**Common patterns and their preferred libraries:**

| Custom code you might write | Library to consider first |
|---|---|
| DataFrame validation (nulls, dtypes, ranges) | **Pandera** (`pandera.pandas`) |
| Data modeling / schema validation | **Pydantic** (`pydantic`) |
| Retry logic / backoff | **tenacity** |
| CLI argument parsing | **Click** or **Typer** |
| Progress bars | **tqdm** or **rich** |

**Decision rule:** If a function's docstring could be the README of a popular PyPI package, that package probably already exists.

**Pandera-specific notes (Discovered Story 1.8):**
- Import as `import pandera.pandas as pa` for pandas validation (the top-level `import pandera as pa` is deprecated in pandera ≥ 0.20)
- Import the exception separately: `from pandera.errors import SchemaError`
- Pandera IS fully typed — no `# type: ignore[import-untyped]` needed
- `pa.DataFrameSchema({col: pa.Column(...)}, strict=False)` — `strict=False` allows extra columns beyond the schema; required for subset-column validation functions
- When a validation function should always verify column existence (even with no constraints), always build the schema: use `pa.Column(checks=checks or None)` instead of early-returning

### Keep Story File in Sync with Mid-Story Pivots (Discovered Story 1.8 Code Review Round 2)

When a story changes direction mid-implementation (e.g., replacing custom code with a library), the story file must be updated **atomically** with the code change:
- **Acceptance Criteria**: Revise to reflect the new approach
- **Tasks/Subtasks**: Update completed/unchecked status to match reality
- **File List**: Add/remove files to match actual git changes
- **Completion Notes**: Update test counts, coverage numbers, and deliverables
- **Change Log**: Document the pivot and rationale

Stale story files create false audit trails — the code review found 3 Critical issues where deleted files were still claimed as deliverables with fabricated test counts.

### Pydantic mypy Plugin: Use `dict[str, Any]` in Test Fixtures (Discovered Story 2.2)

The `pydantic.mypy` plugin (with `init_typed = true`) enforces strict constructor types. Test fixtures returning `dict[str, object]` for Pydantic model kwargs fail mypy because `object` is not assignable to specific field types. Use `dict[str, Any]` instead:

```python
# ❌ Fails mypy with pydantic.mypy plugin (init_typed = true)
@pytest.fixture
def valid_kwargs(self) -> dict[str, object]:
    return {"field_a": 1, "field_b": "value"}

# ✅ Any bypasses the strict init check (acceptable in tests)
@pytest.fixture
def valid_kwargs(self) -> dict[str, Any]:
    return {"field_a": 1, "field_b": "value"}
```

### pyarrow Requires `# type: ignore[import-untyped]` (Discovered Story 2.2)

pyarrow (like pandas) has no `py.typed` marker. All pyarrow imports need the same treatment:

```python
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.dataset as ds  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
```

### Mutmut 3.x: `--ignore` for Import-Level Failures (Discovered Story 2.2)

Beyond the `no_mutation` marker for `Path(__file__)` tests, mutmut's `mutants/` directory may lack modules not in `paths_to_mutate`. Tests that import those modules fail at collection time. Use `--ignore` in mutmut config:

```toml
pytest_add_cli_args_test_selection = [
    "tests/", "-m", "not no_mutation",
    "--ignore=tests/unit/test_logger.py"  # imports ncaa_eval.utils not in mutant
]
```

### pyarrow Schema Evolution: Generic `_apply_model_defaults` Before Pydantic Construction ⭐ (Discovered Story 2.2, Updated Code Review #2)

When using `pyarrow.dataset` with hive-partitioned Parquet files, reading a dataset that mixes **old-schema partitions** (fewer columns) with **new-schema partitions** causes pyarrow to unify schemas and fill missing cells with `null`. For Pydantic fields typed as non-nullable (e.g., `bool`, `int`) with Python defaults, passing `null`/`None` raises `ValidationError` even if the field has a `default=`.

**Pattern (generic):** Instead of hardcoding `fillna` per column, iterate `model.model_fields` to apply defaults for any column with a non-None Pydantic default. This is self-maintaining as the schema evolves:

```python
from typing import Any

def _apply_model_defaults(df: pd.DataFrame, model: type[Game]) -> None:
    """Fill null values in *df* with non-None Pydantic field defaults."""
    sentinel: Any = ...  # PydanticUndefined is represented as Ellipsis
    for name, field_info in model.model_fields.items():
        default = field_info.default
        if name in df.columns and default is not sentinel and default is not None:
            df[name] = df[name].fillna(default)

# Usage in get_games (or equivalent reader), after to_pandas():
df = table.to_pandas()
_apply_model_defaults(df, Game)
return [Game(**row) for row in df.to_dict(orient="records")]
```

**Why generic > hardcoded:** The initial implementation hardcoded `fillna` for `num_ot` and `is_tournament`. Adding a future field with a non-null default (e.g., `is_conference: bool = Field(default=False)`) required remembering to update the reader. The generic approach handles all current and future fields automatically.

**Test it:** Write a schema evolution test that creates both an old-schema partition and a new-schema partition, then reads the old one via the dataset API. Without `fillna`, the test exposes the `ValidationError`. With it, the test confirms model defaults are applied.

```python
# Snippet: create old partition manually, then call repo.get_games(old_season)
partition_dir.mkdir(parents=True, exist_ok=True)
pq.write_table(pa.Table.from_pydict(old_data, schema=old_schema), partition_dir / "data.parquet")
games = repo.get_games(old_season)
assert games[0].num_ot == 0            # default, not null
assert games[0].is_tournament is False  # default, not null
```

**Root cause:** `Pydantic field default` ≠ `None → default`. Pydantic v2 uses the default only when the key is **absent** from the input dict. When pyarrow schema unification adds the column as null, the key IS present (value = `None`), so Pydantic attempts to validate `None` against a non-nullable type.

### Pydantic Cross-Field Invariants: Always Add `@model_validator` ⭐ (Discovered Story 2.2 Code Review)

Data models for domain entities (games, transactions, events) almost always have **cross-field business invariants** beyond per-field constraints. Forgetting these creates schema that accepts semantically impossible data.

**Common basketball/sports game invariants:**
- Winner's score > loser's score (`w_score > l_score`)
- A team can't play itself (`w_team_id != l_team_id`)

```python
from pydantic import model_validator

class Game(BaseModel):
    ...

    @model_validator(mode="after")
    def _check_game_integrity(self) -> Game:
        if self.w_score <= self.l_score:
            msg = f"w_score ({self.w_score}) must be greater than l_score ({self.l_score})"
            raise ValueError(msg)
        if self.w_team_id == self.l_team_id:
            msg = f"w_team_id and l_team_id must differ (both are {self.w_team_id})"
            raise ValueError(msg)
        return self
```

**Template pattern:** When designing a new data entity, explicitly list "what combinations of field values are semantically impossible?" and add a `@model_validator(mode="after")` for each one. Review these during the code review phase — per-field validators don't catch cross-field logic.

### Code Review: Exclude `_bmad-output/` from Git-vs-File-List Discrepancy Checks ⭐ (Discovered Story 2.2 Code Review)

BMAD workflow artifacts (`sprint-status.yaml`, `template-requirements.md`, story `.md` files) are updated by the toolchain itself, not the developer. They are not story deliverables. Flagging them as "files changed but not in story File List" creates recurring false-positive MEDIUM findings.

**Fix:** In `code-review/instructions.xml`, the `_bmad/` and `_bmad-output/` exclusion that already applies to the code review scope must also explicitly apply to the git-vs-File-List discrepancy check. Both the `<critical>` header and the discrepancy action should state: *"exclude `_bmad/` and `_bmad-output/` paths from all checks."*

### ABC Optional Capabilities: Prefer Non-Abstract Defaults Over NotImplementedError Overrides ⭐ (Discovered Story 2.3 Code Review)

When a base class defines methods that not all subclasses support, making them `@abc.abstractmethod` forces subclasses to "implement" them by raising `NotImplementedError` — a Liskov Substitution Principle violation.

**Pattern:** Only the universally-required method is abstract; optional capabilities have concrete default bodies:

```python
class Connector(abc.ABC):
    @abc.abstractmethod
    def fetch_games(self, season: int) -> list[Game]: ...  # REQUIRED — all sources

    def fetch_teams(self) -> list[Team]:  # OPTIONAL — not all sources provide this
        raise NotImplementedError(f"{type(self).__name__} does not provide team data")

    def fetch_seasons(self) -> list[Season]:  # OPTIONAL
        raise NotImplementedError(f"{type(self).__name__} does not provide season data")
```

**Result:** Subclasses that don't support `fetch_teams`/`fetch_seasons` simply inherit the default — no redundant override needed. Subclasses that DO support them override to return data. Callers use `try/except NotImplementedError` or `isinstance` to probe capabilities.

**Anti-pattern:**
```python
# ❌ LSP violation: ABC contract claims fetch_teams returns list[Team]
#    but subclass "implements" it by raising instead
class EspnConnector(Connector):
    def fetch_teams(self) -> list[Team]:
        raise NotImplementedError("ESPN doesn't have team data")  # ← lying to callers
```

### Validate Domain-Constrained Values Before Pydantic Construction (Discovered Story 2.3 Code Review)

When parsing raw data that maps to a `Literal[...]` typed field, never use `cast()` alone — `cast()` is zero-cost at runtime and won't catch invalid values. Validate explicitly and raise the appropriate connector exception before handing to Pydantic:

```python
# ❌ cast() doesn't validate — invalid WLoc causes ValidationError, not DataFormatError
loc=cast("Literal['H', 'A', 'N']", str(row["WLoc"])),

# ✅ Validate first, raise appropriate error, then cast
wloc = str(row["WLoc"])
if wloc not in ("H", "A", "N"):
    raise DataFormatError(f"kaggle: {filename} has unexpected WLoc value: {wloc!r}")
loc = cast("Literal['H', 'A', 'N']", wloc)
```

This ensures callers catch `DataFormatError` (the connector's error contract), not raw Pydantic `ValidationError`.

### Dependency Version Pinning: `^` Not `>=` for Libraries With Known Breaking Changes (Discovered Story 2.3 Code Review)

Use caret pinning (`^x.y.z` = semver-compatible) rather than lower-bound-only (`>=x.y.z`) for any library with documented breaking changes between major versions:

```toml
# ❌ Allows future majors (kaggle v3, v4) that may break existing API calls
kaggle = ">=2.0.0"

# ✅ Semver-compatible: allows patch/minor updates, blocks major-version breaks
kaggle = "^2.0.0"
```

**Rule:** If the Dev Notes mention a breaking change between versions of a library, use `^` not `>=`.

**Important:** Changing constraint syntax (e.g., `>=` → `^`) constitutes a significant change to `pyproject.toml`. Always run `poetry lock` after, or CI will fail with: *"pyproject.toml changed significantly since poetry.lock was last generated."*

### Precompute Derived Collections in `__init__` (Discovered Story 2.3 Code Review)

Avoid rebuilding derived collections inside hot-path methods. When `__init__` receives a mapping that's later used in case-insensitive lookups, precompute the lowercased version:

```python
# ❌ Rebuilds lower_map on every call — O(N×M) for N games × M teams
def _resolve_team_id(name: str, mapping: dict[str, int]) -> int | None:
    lower_map = {k.lower(): v for k, v in mapping.items()}  # rebuilt every call!

# ✅ Precompute once in __init__, pass as argument
class EspnConnector:
    def __init__(self, team_name_to_id: dict[str, int], ...):
        self._lower_team_map = {k.lower(): v for k, v in team_name_to_id.items()}

def _resolve_team_id(name: str, lower_map: dict[str, int], original: dict[str, int]) -> int | None:
    exact = lower_map.get(name.lower())  # O(1) lookup
    ...
```

**Rule:** Any dict comprehension inside a method that's called in a loop is a candidate for precomputation in `__init__`.

### `poetry add` Does Not Install Into Active Conda Env (Discovered Story 2.4)

When running `POETRY_VIRTUALENVS_CREATE=false conda run -n ncaa_eval poetry add <pkg>`, Poetry updates `pyproject.toml` and `poetry.lock` but the package may not actually land in the conda env. Always follow `poetry add` with `conda run -n ncaa_eval pip install <pkg>` to guarantee the package is importable from the conda env.

**Workaround until resolved:** Use `conda run -n ncaa_eval pip install <pkg>` as the authoritative installation step; let `poetry add` update `pyproject.toml`/`poetry.lock` for dependency tracking.

### Overwrite-First Repository Writes Require Merge Before Save (Discovered Story 2.4)

When `save_X()` **overwrites** the entire partition (e.g., `save_games()` per-season), adding a second source's data for the same partition requires: **load existing → merge → save combined**. Otherwise the first source's data is silently lost.

```python
# ❌ Overwrites Kaggle games with ESPN games — Kaggle data LOST
self._repo.save_games(espn_games)

# ✅ Merge first, then save combined list
existing = self._repo.get_games(year)
self._repo.save_games(existing + espn_games)
```

**Detection:** Check `save_*()` docstrings for "overwrite" vs "append" behavior before writing any logic that calls `save_*()` for the same key from multiple sources.

### ESPN Marker-File Caching (Discovered Story 2.4)

When the cache-check criterion would require reading and inspecting Parquet contents (e.g., "does this partition already contain ESPN-prefixed IDs?"), use lightweight marker files instead:

```
{data_dir}/.espn_synced_2025   ← empty file; touch() after sync, unlink() on force-refresh
```

This avoids loading Parquet just to detect presence, while remaining robust to partial failures (only create marker after a successful sync).

### Tests Importing Project-Root Modules by Name Fail in Mutmut (Discovered Story 2.4)

Tests that do `import sync` (where `sync.py` is at the project root, not in a package) fail with `ModuleNotFoundError` when mutmut runs them from `mutants/`. Even with `sys.path.insert` inside the test function, the import can fail.

**Fix:** Mark such tests `@pytest.mark.no_mutation` to exclude them from mutmut. The engine-level tests still provide mutation coverage.

```python
@pytest.mark.integration
@pytest.mark.no_mutation  # imports project-root sync.py by name — fails in mutmut context
def test_cli_sync_kaggle(...) -> None:
    import sync as sync_module
    ...
```

### Private Helper Methods Accessed Cross-Class Require a Public Interface (Discovered Story 2.4 Code Review)

When `Class A` needs to call a helper on `Class B` that is logically useful to external callers, rename it from `_private` to `public`. Suppressing `SLF001` with `# noqa` hides the design smell instead of fixing it.

```python
# ❌ Suppresses linting, couples SyncEngine to KaggleConnector internals
season_day_zeros = kaggle_connector._load_day_zeros()  # noqa: SLF001

# ✅ Rename to public; docstring documents the public contract
def load_day_zeros(self) -> dict[int, datetime.date]:
    """Load and cache the season → DayZero mapping."""
    ...
```

**Rule:** If a `# noqa: SLF001` exists in a caller, question whether the method belongs on the public API. If yes, rename it. If not, consider moving the logic to the callee or a shared utility.

### ESPN-style Multi-Source Caching Requires Full Cycle Tests (Discovered Story 2.4 Code Review)

When a caching strategy involves marker files (or any lightweight sentinel), the following paths must each be tested:
1. Happy-path: sentinel created after successful sync
2. Cache hit: sentinel exists → fetch methods not called
3. Force-refresh: sentinel deleted → re-fetch occurs, sentinel re-created
4. Merge logic: existing data + new source data combined before overwrite-save

Omitting these tests lets bugs in the entire cache lifecycle go undetected even when the dependency guard and ordering tests pass.

*Last Updated: 2026-02-19 (Story 2.4 Code Review — private method public API promotion, multi-source cache test coverage)*
*Next Review: [Set cadence - weekly? sprint boundaries?]*
