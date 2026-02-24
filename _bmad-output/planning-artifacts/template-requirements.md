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

### Session-Scoped Fixtures for Expensive Operations ⭐ (Discovered Story 5.4 Code Review)

When tests share an expensive resource (e.g., a trained ML model for property-based tests), use `@pytest.fixture(scope="session")` rather than a module-level mutable global with `noqa: PLW0603`.

```python
# ❌ Anti-pattern: module-level singleton with global statement
_MODEL: MyModel | None = None

def _get_model() -> MyModel:
    global _MODEL  # noqa: PLW0603
    if _MODEL is None:
        _MODEL = MyModel()
        _MODEL.fit(X, y)
    return _MODEL

# ✅ Correct: session-scoped pytest fixture
@pytest.fixture(scope="session")
def trained_model() -> MyModel:
    """Return a trained model shared across all tests in the session."""
    model = MyModel()
    model.fit(X, y)
    return model

# Test method accepts fixture as argument (pytest injects it)
def test_predict_bounded(self, trained_model: MyModel, ...) -> None:
    preds = trained_model.predict_proba(X_test)
    ...
```

**Rationale:** Session fixtures are properly scoped, discoverable, type-checked, and pytest-managed. Globals with `noqa` suppress legitimate linting, create test ordering dependencies, and are invisible in test signatures.

### Hypothesis + Session Fixture Pattern ⭐ (Discovered Story 5.4 Code Review)

Property-based tests (`@given`) combined with expensive setup (model training) require two settings to avoid flakiness:
1. `@settings(deadline=None)` — disables the 200ms per-example deadline
2. A **session-scoped fixture** for the trained model — prevents re-training on every Hypothesis example

```python
@given(x=st.lists(st.floats(...), min_size=50, max_size=50))
@settings(max_examples=50, deadline=None)
def test_predictions_bounded(self, trained_model: MyModel, x: list[float]) -> None:
    preds = trained_model.predict_proba(pd.DataFrame({"x": x}))
    assert (preds >= 0.0).all()
```

### Defensive Private Attribute Access in Tests ⭐ (Discovered Story 5.4 Code Review)

When tests must access private attributes for behavioral verification (e.g., `_clf.best_iteration` on a wrapped estimator), use `getattr` with a `None` default and assert before comparison:

```python
# ❌ Fragile: raises TypeError if best_iteration is None
assert model._clf.best_iteration < 500

# ✅ Defensive: guards against None; explicit assertion message
best_iteration: int | None = getattr(model._clf, "best_iteration", None)
assert best_iteration is not None, "Expected best_iteration to be set (early stopping should have fired)"
assert best_iteration < 500, f"Early stopping should fire before 500; got {best_iteration}"
```

### Testable CLI Orchestration: Pass Console as Parameter ⭐ (Discovered Story 5.5 Code Review)

When a CLI orchestration function (`run_training`, `run_evaluation`, etc.) uses `rich.console.Console` for terminal output, accept it as an **optional parameter** rather than using a module-level singleton. This keeps the function pure (no hidden I/O state) and lets tests suppress output.

```python
# ❌ Anti-pattern: module-level Console — sprays to stdout in unit tests
console = Console()

def run_training(...) -> ModelRun:
    console.print("Training...")  # Uncapturable in unit tests

# ✅ Correct: Console as optional parameter with sensible default
def run_training(..., console: Console | None = None) -> ModelRun:
    _console = console or Console()
    _console.print("Training...")  # Tests pass Console(quiet=True) to suppress
```

**Rationale:** Module-level `Console()` instances are side-effectful singletons. Functional design rule requires side effects pushed to edges — the CLI entry point creates and passes the Console; the pipeline function remains testable.

### `TYPE_CHECKING` Empty Block Is a Ruff Blind Spot ⭐ (Discovered Story 5.5 Code Review)

Ruff does **not** flag an `if TYPE_CHECKING: pass` block with nothing meaningful inside it. This scaffold artifact can survive into production, polluting the import list with an unused `TYPE_CHECKING` import and misleading future readers.

```python
# ❌ Invisible to ruff — must be caught in code review
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # Ruff sees this as valid; no F401, no warning
```

**Rule:** During code review, always scan for `if TYPE_CHECKING:` blocks containing only `pass` — they are dead code and should be removed along with the `TYPE_CHECKING` import.

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

### Pydantic Field Constraints for Bounded Hyperparameters ⭐ (Discovered Story 5.4 Code Review)

Use `Annotated[type, Field(gt=..., lt=...)]` for hyperparameters that have meaningful mathematical bounds. Without these, invalid values (e.g., `validation_fraction=0.0`) produce cryptic errors deep in the call stack.

```python
from typing import Annotated
from pydantic import BaseModel, Field

class MyModelConfig(BaseModel):
    # ❌ No validation — 0.0 causes train_test_split to fail cryptically
    validation_fraction: float = 0.1

    # ✅ Validated at construction — clear Pydantic error at the boundary
    validation_fraction: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.1
    subsample: Annotated[float, Field(gt=0.0, le=1.0)] = 0.8
    n_estimators: Annotated[int, Field(gt=0)] = 500
```

**Rule of thumb:** Any float parameter that feeds into a random split (`test_size`, `validation_fraction`), a sampling ratio (`subsample`, `colsample_bytree`), or a probability should be constrained at the Pydantic level.

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

**Story 2.4 - Sync CLI & Smart Caching (2026-02-19 Code Review Round 2):**
- ❌ **Test fixture date format not updated with production code change** — When `load_day_zeros()` was updated to parse `%m/%d/%Y` (the actual Kaggle CSV format), the test fixture `MSeasons.csv` was not updated from ISO `YYYY-MM-DD`. This broke 11 unit tests silently — they passed the first time but failed after the format change. **Template pattern: Test fixture CSVs must use the same format as the real upstream data. When changing date format parsing in production code, always update the corresponding fixture files in the same commit.**
- ❌ **`iterrows()` used in new dict-building methods** — `fetch_team_spellings()` (new) and `_build_espn_team_map()` (new function) both used `for _, row in df.iterrows()` which violates Style Guide §6.2. For simple dict construction, use `dict(zip(df["col1"].str.lower(), df["col2"].astype(int)))`. For complex conditional logic per-row, extract `.tolist()` first and iterate over the Python list (not the DataFrame).
- ❌ **Integration tests coupled to external library internals** — ESPN tests mocked `KaggleConnector` and `EspnConnector` but did NOT mock the module-level `_build_espn_team_map()` function, which reads a CSV from `cbbpy.__file__`'s package directory. Tests were silently calling real cbbpy code, coupling them to cbbpy's internal file structure. **Template pattern: When using `@patch` to mock classes, also patch any module-level helper functions that call into external library internals. If a function reads from `pkg.__file__`, it MUST be mocked in unit/integration tests.**
- ❌ **User-facing error messages not updated with API changes** — When the README was updated to reflect new Kaggle 2.0 auth (`access_token` file) the error message in `download()` still referenced `~/.kaggle/kaggle.json` and `KAGGLE_USERNAME/KAGGLE_KEY`. Error messages are user documentation — they must be updated atomically with README changes. (Discovered: Story 2.4 Code Review Round 2)

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

**Story 4.3 - Canonical Team ID Mapping & Data Cleaning (2026-02-21 Code Review):**
- ❌ **ML library `.fit()` calls crash on empty DataFrames** — `composite_pca()` called `PCA().fit(snapshot.values)` with no guard after `snapshot.dropna()`. When no data matched season/day_num filters, this raised `ValueError`. **Template pattern: always guard `if df.empty: return pd.DataFrame()` before any sklearn/ML method that expects non-empty input.**
- ❌ **Empty-dict inputs cause silent ZeroDivisionError in weighted aggregations** — `composite_weighted({})` caused `w.sum() = 0` → division by zero. **Template pattern: validate dict inputs are non-empty at method entry with a `ValueError` before any arithmetic.**
- ❌ **Every public method in a store/manager class must have at least one test** — `composite_pca()` (AC 8c, Option C) was the only composite method without a test and happened to have the worst edge-case bugs. When a class exposes 11+ methods, it's easy to miss one. **Template pattern: map every public method to a test ID during story planning; leave blank IDs as explicit TODOs.**
- ℹ️ **pandas `.isin()` accepts any iterable** — passing a `tuple` or `frozenset` directly is idiomatic; wrapping in `list()` is unnecessary clutter (PEP 20). Use `.isin(_GATE_SYSTEMS)` not `.isin(list(_GATE_SYSTEMS))`.

**Story 4.4 - Sequential Transformations (2026-02-21 Code Review):**
- ❌ **"Every public method → test" mandate violated AGAIN on the very next story** — `DetailedResultsLoader.get_season_long_format()` had no test despite `get_team_season()` (sorted/empty) having 2 tests. The pattern: devs test the methods they use during manual verification, skipping ones not needed for their own feature computation. **Code review must explicitly enumerate all public methods and cross-check against the test list. Story planning must list every method → test ID.**
- ❌ **`int(pd.Series.max())` crashes on empty Series** — `int(float('nan'))` raises `ValueError`. Any public utility function that calls `.max()`, `.min()`, `.mean()`, or `.sum()` on a user-provided numeric Series must guard against empty input: `if series.empty: return <appropriate_empty_result>`. Fixed in `compute_game_weights` to return `pd.Series([], dtype=float)`.
- ℹ️ **`rolling_full_*` (expanding mean) stays unweighted even when time-decay weights are provided** — this is a semantic inconsistency: windowed rolling stats use BartTorvik time-decay weights but the full-season aggregate ignores them. Intentional design decision (the full aggregate is a separate "baseline" feature), but worth documenting so future developers don't assume consistency.
- ℹ️ **`per100` stats for opponent columns are semantically non-standard** — `opp_score_per100 = opp_score × 100 / team_possessions` (opponent score relative to OUR possessions) is not a standard basketball metric. Including `opp_score` in `_COUNTING_STATS` propagates it through all normalization. Review whether opponent stats should be excluded from per-possession normalization in downstream stories.

**Story 4.8 - Elo Feature Building Block (2026-02-22 Code Review):**
- ❌ **`iterrows()` in test code is a project convention violation** — `test_delta_elo_equals_elo_a_minus_elo_b` used `for _, row in result.iterrows()` to assert `delta_elo == elo_a - elo_b`. The no-iterrows mandate applies to ALL code including tests. **Template pattern: Use vectorized assertions in tests: `(result["elo_a"] - result["elo_b"]).tolist() == pytest.approx(result["delta_elo"].tolist())` or `np.testing.assert_allclose(result["col_a"], result["col_b"])`.** (Discovered: Story 4.8 Code Review, 2026-02-22)
- ❌ **`_empty_frame()` must return columns matching the active feature blocks** — When a season has zero games, `_empty_frame()` was hardcoded to return only `_META_COLUMNS + ["delta_elo"]`. Any caller that expects all active feature block columns (ordinal, batch_rating, Elo, etc.) to be present on empty results will get a `KeyError`. **Template pattern: Compute the empty frame's column set from `self.config.active_blocks()` at runtime — the same logic used by the non-empty path. Never hardcode a fixed column list in `_empty_frame()` for multi-block architectures.** (Discovered: Story 4.8 Code Review, 2026-02-22)
- ❌ **Zero-margin edge case in `_margin_multiplier` silently zeroes the rating update** — `0 ** 0.85 == 0`, so a game that rounds to zero margin after OT-rescaling produces `k_eff = 0`, meaning neither team's rating is updated at all. This silent no-op corrupts Elo state without any logging or error. **Template pattern: Apply a minimum floor of 1 before exponentiation: `floored = max(1, capped)`. Add a test for zero-margin behavior to document the design intent and prevent future regressions.** (Discovered: Story 4.8 Code Review, 2026-02-22)
- ❌ **K-factor threshold boundary tests verified the getter in isolation, not the actual game update** — `test_regular_season_k_after_threshold` called `_effective_k(101, False)` to verify the return value but never confirmed that `update_game()` applied the correct K-factor in the critical game at the threshold boundary (game #20 vs game #21). An off-by-one in the threshold comparison would pass this test while silently mismapping K factors. **Template pattern: For any stateful method with a phase-transition threshold, write a test that verifies the rating change *magnitude* differs on either side of the boundary — not just the intermediate getter method.** (Discovered: Story 4.8 Code Review, 2026-02-22)
- ❌ **OT-adjusted margin computation had no explicit test** — The story notes explicitly require `rescale_overtime()` before computing margin for the K_eff formula, yet no test verified that an OT game's margin is smaller than the same raw-score game played in regulation. A refactor could silently remove the OT rescaling and all existing tests would pass. **Template pattern: For any multi-step numeric pipeline (rescale → cap → exponentiate), write at least one test that verifies the output differs when the first step is skipped (e.g., OT vs non-OT for same raw scores).** (Discovered: Story 4.8 Code Review, 2026-02-22)

**Story 4.7 - Stateful Feature Serving (2026-02-22 Code Review):**
- ❌ **`set_index()` inside a game loop = O(N×G) — pre-index once outside the loop** — `_lookup_rating()` called `rating_df.set_index("team_id")` inside every game iteration, once per rating type per team. For a 5,000-game season × 3 types × 2 teams = 30,000 redundant O(T) index builds. **Template pattern: Any lookup table used in a loop MUST be indexed before the loop. Call `series = df.set_index("key")["col"]` once, then `series.get(team_id, np.nan)` inside the loop.** (Discovered: Story 4.7 Code Review, 2026-02-22)
- ❌ **Coverage-gate / expensive initializer called inside a game loop** — `_resolve_ordinal_systems()` called `run_coverage_gate()` (a full CSV scan) once per game when `ordinal_systems is None`. For 5,000 games this is 5,000 full-scan invocations. **Template pattern: Any method that calls an expensive initializer (CSV scan, DB query, ML fit) must be cached above the hot loop. Pass the resolved value as a parameter rather than re-computing it inside each iteration.** (Discovered: Story 4.7 Code Review, 2026-02-22)
- ❌ **`df.at[idx, col]` in a game loop is equivalent to iterrows — avoid** — The original `_serve_batch` built a DataFrame from metadata rows, then used `df.at[idx, col]` inside an enumerated game loop to write ordinal, seed, and batch rating columns cell by cell. This is O(N) cell writes with pandas overhead. **Template pattern: Accumulate per-game values into Python lists inside the loop, then assign as columns in bulk (`df["col"] = list_of_values`) after the loop exits.** Single column assignment is O(1) vs O(N) cell writes. (Discovered: Story 4.7 Code Review, 2026-02-22)
- ❌ **Complexity ceiling exceeded when refactoring** — Extracting the batch-indexed pre-computation into `_serve_batch` caused its cyclomatic complexity to exceed the C901 threshold (15 > 10) and PLR0912 branch count (14 > 12). **Template pattern: When refactoring a complex orchestration method, budget for helper extraction before starting. If a method must coordinate >4 conditional feature blocks, plan to extract at least one `_collect_*` or `_append_*` helper upfront.** (Discovered: Story 4.7 Code Review, 2026-02-22)
- ❌ **Calibrator test asymmetry — isotonic tested for monotonicity, sigmoid was not** — `IsotonicCalibrator` had a `test_monotonicity` test but `SigmoidCalibrator` did not, despite the sigmoid mapping being monotone by construction. Boundary inputs (0.0, 1.0) were unverified for the sigmoid's log-odds clip path. **Template pattern: When two classes share a mathematical property (monotonicity, probability bounds, calibration leakage), both must be tested for that property — parity in test coverage between analogous classes.** (Discovered: Story 4.7 Code Review, 2026-02-22)

**Story 4.6 - Opponent Adjustment Rating Systems (2026-02-21 Code Review):**
- ❌ **Iterative solvers must log a warning on non-convergence** — `compute_srs` ran the fixed-point loop up to `srs_max_iter` without any `for...else` branch. Disconnected schedules or degenerate inputs silently return unconverged ratings. **Template pattern: All iterative numeric algorithms with a max-iteration guard MUST use `for...else` to log a warning when the loop exhausts without meeting the convergence criterion.** Example: `for _ in range(max_iter): ... break / else: logger.warning("did not converge after %d iterations", max_iter)`.
- ❌ **Unused tuple unpacking values slip past Ruff and mypy** — `_build_team_index()` returned 4 values but all 3 callers only used 3, silently assigning the unused `idx` dict to a named variable. Likewise `_build_srs_matrices()` returned `net_margin` and `n_games` that were never referenced. Ruff F841 does NOT flag unused names from tuple unpacking (only simple assignments). **Template pattern: Use `_` explicitly for discarded return values in tuple unpacking: `teams, _, w_idx, l_idx = helper()`. Code reviewers must specifically check all multi-value unpacking calls.**
- ❌ **Integer dtype assertions that allow `object` are semantically vacuous** — `assert df["col"].dtype in (np.dtype("int64"), np.dtype("int32"), object)` passes even if the column holds Python strings or mixed types. **Template pattern: Use `pd.api.types.is_integer_dtype(df["col"])` for any integer-dtype assertion in tests. The `object` allowance negates the test's intent entirely.**

**Story 4.1 - Feature Engineering Techniques Spike (2026-02-21 SM retrospective):**
- ❌ **Spike stories must include a post-PO-approval checklist AC for updating downstream epic story descriptions** — After Story 4.1 was approved, the SM had to manually read through the research document and update ACs in epics.md for Stories 4.3–4.7 and add a new Story 4.8 placeholder. This work was not in the story's AC list and was only discovered as necessary after PO approval. **Template pattern for all future spike stories whose output defines implementation scope:**
  ```
  **And** after PO approval of the spike findings, the SM updates the downstream
  story descriptions in epics.md to incorporate all building blocks and scope
  decisions from the research document — adding new story placeholders as needed
  and moving deferred items to the Post-MVP Backlog.
  ```
  - **Applies to:** All research spikes that produce a scope-defining document: 4.1 (feature techniques → Epic 4 stories), 5.1 (modeling approaches → Epic 5 stories), 6.4 (simulation confidence → Story 6.5), 7.7 (slider mechanism → Story 7.5), and any future spike.
  - **Why this matters:** Research spikes are not "done" when the document is approved. The SM still needs to propagate the findings back into the epic AC structure before the next story can be created with correct context. Without this AC, the create-story workflow operates on stale story descriptions that don't reflect research findings.
  - **Who does the work:** SM agent — not the dev agent. This is a sprint planning/story maintenance task, not implementation. (Discovered: Story 4.1, 2026-02-21)

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

### EDA Notebooks: Separate Conventions from Production Code ⭐ (Discovered Story 3.1)

EDA notebooks in `notebooks/eda/` are exploration artifacts, not production code. Different rules apply:

| Rule | `src/` production code | `notebooks/eda/` notebooks |
|---|---|---|
| `mypy --strict` | ✅ Required | ❌ Not applicable |
| Ruff linting | ✅ Required | ❌ Not applied by default |
| `from __future__ import annotations` | ✅ Required | ❌ Not required |
| No-iterrows mandate | ✅ Enforced by linting | ✅ Apply as discipline only |
| Commit with executed outputs | N/A | ✅ Required |

**Template pattern:** When a story is EDA/exploration type, add a "Story Nature" section to Dev Notes stating: *"This is an EDA story — the primary deliverable is a notebook artifact, not module code. `mypy --strict` and Ruff do NOT apply. No-iterrows is a style discipline but not a lint gate."*

### `gitignore` Data Directory Pattern ⭐ (Discovered Story 3.1)

To track a `data/` directory in git while ignoring its contents (downloaded/generated data files), use the `.gitkeep` negation pattern:

```gitignore
# Ignore all data directory contents
data/*
# But track the directory itself via a gitkeep
!data/.gitkeep
```

**Critical:** Never delete `data/.gitkeep` during development (e.g., after populating `data/` with real files). On a fresh clone, `data/` will not exist if `.gitkeep` is absent. Any code that uses `Path("data/")` will break. **Code review must check for unstaged deletions of `.gitkeep`** — this is a common accident when working with data directories.

### nbconvert Execution: Use `--output-dir`, Not `--output` ⭐ (Discovered Story 3.1)

When executing Jupyter notebooks via `nbconvert` from the **repo root** for in-place output (same directory as source):

```bash
# ❌ WRONG — doubles the path: writes to notebooks/eda/notebooks/eda/file.ipynb
conda run -n ncaa_eval jupyter nbconvert --to notebook --execute \
  --output notebooks/eda/01_data_quality_audit.ipynb \
  notebooks/eda/01_data_quality_audit.ipynb

# ✅ CORRECT — writes output next to source notebook
conda run -n ncaa_eval jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=600 \
  --output-dir notebooks/eda \
  notebooks/eda/01_data_quality_audit.ipynb
```

**Rule:** Always use `--output-dir <directory>` and omit `--output` when the target location is the same directory as the source notebook. The `--output` flag appends to the input directory path, causing doubling.

**Note:** nbconvert 7.x sets the kernel working directory to the **notebook's directory** by default. Relative file paths in notebook cells resolve relative to `notebooks/eda/`, not the repo root. Use `Path("../../data/")` or `Path(__file__).parent.parent.parent / "data"` style paths, not `Path("data/")`.

### Multi-Source Data: Verify Connector Behavior Against Architecture Assumptions ⭐ (Discovered Story 3.1 Code Review)

When writing EDA notebooks that analyze data ingested by connectors, **read the connector source code** before writing narrative about expected field behavior. Connector implementation often differs from the architecture document's description.

**Example (NCAA_eval Story 3.1):** The architecture and Dev Notes described `Game.date = None` for Kaggle games (1985–2024). However, `KaggleConnector._parse_games_csv()` derives `game_date = day_zeros.get(s) + timedelta(days=day_num)` for ALL seasons, populating `date` for every game. The notebook's output correctly showed 0 null dates — but the narrative described it as wrong behavior.

**Template pattern:** Before writing "expected behavior" narrative in a notebook cell, verify it by reading the actual data loading code. If the data surprises you, investigate why — don't assume the data is wrong.

### Epic 4 Normalization Design Requirements ⭐ (User Preference, 2026-02-20)

These are explicit user requirements for how the Epic 4 feature engineering / normalization system must be designed. Capture in Epic 4 story acceptance criteria.

#### 1. Gender normalization scope — default separate, user-selectable

Distribution analysis (Story 3.2) shows Men and Women have the same *shape* labels but different *location/scale*:
- Score: Men ~70, Women ~64
- FTPct: Women systematically higher than Men
- FGM3/FGA3: Men have historically higher 3-point volumes

**Default:** Normalize Men and Women **separately** (calibrate to each sport's own distribution).

**User must be able to override** to "combined" (shared normalization across genders) — for example when building a single M+W model or when the user intentionally wants cross-gender comparability.

**Implementation pattern:**
```python
# Normalization config should expose a gender_scope parameter:
gender_scope: Literal["separate", "combined"] = "separate"  # default: separate
```

#### 2. Dataset-type normalization scope — default regular season only, user-selectable

Tournament data has **survivorship bias** (only 64–68 elite teams qualify) and **30× smaller sample size** than regular season (~2,276 vs ~118,882 games for Men's). Normalizing against tournament distributions biases toward elite-team performance ranges.

**Default:** Normalize using **regular season statistics only**.

**User must be able to override** to "tournament", "combined", or "all_games" — for example when building tournament-only predictors where the user wants normalization anchored to tournament-level play.

**Implementation pattern:**
```python
dataset_scope: Literal["regular_season", "tournament", "combined"] = "regular_season"  # default
```

#### 3. Advanced distribution fitting — Epic 4 investigation required

Story 3.2 distribution analysis only fit Normal and Log-Normal. The following distributions should be investigated in Epic 4 to find the best fit per stat:

| Stat group | Candidate distributions | Why |
|---|---|---|
| Bounded [0,1] rates: FGPct, 3PPct, FTPct, TO_rate | **Beta**, Logit-Normal | Natural support on [0,1]; Beta parameterizes shape explicitly |
| Right-skewed counting: Blk, Stl, OR | **Gamma**, **Weibull**, Log-Normal | Gamma/Weibull often beat Log-Normal for non-negative count data |
| Moderate-skew counting: FGM3, FGA3, Ast, TO, PF, FTM, FTA | **Negative Binomial**, Gamma, Normal | Negative Binomial handles overdispersed count data |
| High-volume (approx. normal): Score, FGM, FGA, DR | Normal, **Skew-Normal** | Already approximately normal; Skew-Normal may capture mild tails |

**Goodness-of-fit metrics** to use for distribution selection:
- Kolmogorov-Smirnov test (distribution-free, large-N valid)
- Anderson-Darling test (more sensitive to tails)
- AIC / BIC (when fitting parametric distributions with different parameter counts)
- Visual: Q-Q plots against each candidate distribution (not just Normal)

#### 4. Normalization pipeline configurability

The normalization system must expose all of the above as **user-configurable parameters** at the preprocessing pipeline level. Defaults encode the recommendations above. Per-stat overrides should be possible (e.g., use "combined" gender normalization for FGPct but "separate" for Score).

### Parallel DayNum-to-Round Mapping Functions Must Share Identical Cutoffs ⭐ (Discovered Story 3.2 Code Review)

When a notebook defines two parallel mapping functions from the same DayNum input — one returning a numeric round index, one returning a human-readable name — they MUST use **identical boundary values**. Mismatched cutoffs corrupt any analysis that uses the numeric encoding:

```python
# ❌ Inconsistent: day 147 → round_num=3 ("E8") but round_name="Final Four"
def day_to_round_num(day: int) -> int:
    if day <= 146: return 2   # S16/E8 combined
    elif day <= 148: return 3  # ← actually FF days!

def day_to_round_name(day: int) -> str:
    if day <= 146: return "Elite 8"
    elif day <= 152: return "Final Four"  # ← correct

# ✅ Consistent: derive name from num, or use a single lookup table
ROUND_MAP = [(135, 0, "First Four"), (139, 1, "R64/R32"), (144, 2, "Sweet 16"),
             (146, 3, "Elite 8"), (152, 4, "Final Four"), (999, 5, "Champion")]

def _day_to_round(day: int) -> tuple[int, str]:
    for cutoff, num, name in ROUND_MAP:
        if day <= cutoff:
            return num, name
    return 5, "Champion"
```

**Template pattern:** Never write two separate if/elif ladders for the same data. Use a single lookup table and derive both the numeric index and the string label from it.

### Multi-Notebook Findings Files: Overwrite, Don't Append Duplicate Sections ⭐ (Discovered Story 3.2 Code Review)

When multiple notebooks contribute to the same markdown findings file, **later notebooks must overwrite or replace** placeholder sections from earlier notebooks — never append a second copy of the same section heading. Duplicate headings cause:

1. Contradictory recommendations (the placeholder may have different values than the computed output)
2. Malformed documents that confuse Epic downstream consumers

**Pattern:** Notebook 02 should write a placeholder section that notebook 03 *replaces* (using a known sentinel marker) or notebook 03 should use a regex replace to update the section in-place:

```python
# ✅ In notebook 03's summary cell — replace section, don't append
import re
findings_path = Path("statistical_exploration_findings.md")
content = findings_path.read_text()
new_section = f"## Section 7: Box-Score Statistical Distribution Analysis\n\n{new_content}"
# Replace from the previous Section 7 header to the next --- separator
updated = re.sub(r"## Section 7:.*?(?=\n---|\Z)", new_section, content, flags=re.DOTALL)
findings_path.write_text(updated)
```

**Alternative:** Omit the Section 7 placeholder from notebook 02 entirely; only notebook 03 writes Section 7.

### EDA Story File List: All Committed Files Must Be Listed, Including "Extension" Notebooks ⭐ (Discovered Story 3.2 Code Review)

Any file committed to the story branch — including notebooks added in later commits on the same branch labeled "extension" — must appear in the story's Dev Agent Record File List. The git branch is the audit boundary, not the commit message. If it's on the branch, it's a story deliverable.

**Anti-pattern:** Adding `03_distribution_analysis.ipynb` with commit `"Story 3.2 extension"` while the story File List only lists `02_statistical_exploration.ipynb`.

### `apply(lambda r: ..., axis=1)` Is Iterrows-Equivalent — Vectorize It ⭐ (Discovered Story 3.2 Code Review)

`.apply(func, axis=1)` applies a Python function row-by-row through the DataFrame. For operations that construct strings from two columns (like sorted conference pair keys), this is always vectorizable:

```python
# ❌ Row-wise apply — iterrows-equivalent
df["conf_pair"] = df.apply(
    lambda r: "_vs_".join(sorted([r["w_conf"], r["l_conf"]])), axis=1
)

# ✅ Vectorized — uses numpy sort on 2D array
sorted_cols = pd.DataFrame(
    np.sort(df[["w_conf", "l_conf"]].values, axis=1),
    columns=["conf_a", "conf_b"],
    index=df.index,
)
df["conf_pair"] = sorted_cols["conf_a"] + "_vs_" + sorted_cols["conf_b"]
```

**Rule:** If `apply(axis=1)` accesses only 2 columns and performs a string/arithmetic operation, it can always be vectorized. The no-iterrows mandate in EDA notebooks includes `apply(axis=1)` equivalents.

### Documentation Synthesis Story Reviews: Content Accuracy over Code Quality ⭐ (Discovered Story 3.3 Code Review)

For stories with no code deliverables (pure markdown synthesis), the adversarial code review shifts from code quality to **content accuracy and internal consistency**. Key checks:

1. **Cross-section consistency**: The same item appearing in multiple sections (e.g., a metric in a ranked list AND in a story guidance table AND in a priority table) must be consistent in value, ordering, and label
2. **Ranked list completeness**: If a feature appears in Section A as a ranked item, it must appear in all downstream tables that reference that feature set — omissions create gaps in implementation guidance
3. **Stated criteria vs actual ordering**: When a document says "ranked by X" but uses a multi-factor ordering that deviates from strict X-ordering, the rationale must be stated explicitly. Silent deviations mislead readers

**Template pattern:** Doc-only story reviews should include explicit checks:
- [ ] Every item in the ranking appears in all relevant downstream tables
- [ ] Ranking order is explained if it deviates from the stated primary criterion
- [ ] Signal labels (e.g., "MEDIUM signal") from upstream notebooks are explained in context if they appear counterintuitive in the synthesized document

### Ranked Lists Require Explicit Ordering Rationale ⭐ (Discovered Story 3.3 Code Review)

When a technical document presents a ranked list with a stated criterion (e.g., "ranked by expected predictive value") but the actual ordering uses a composite of multiple factors, the deviation from the primary metric ordering **must be explained**. Silent deviations:
- Lead readers to wrong implementation priorities (implement FGPct before SoS despite SoS having 31% higher correlation)
- Create internal contradictions between sections that reference the same ranked items
- Confuse downstream story developers who use the list as implementation order guidance

**Pattern:** Add a "Ranking rationale:" paragraph immediately after any ranked list header that explains the composite criteria and why the top-metric item may not appear first.

### Extension Notebook Outputs Must Be Committed With the Notebook ⭐ (Discovered Story 3.2/3.3 Code Review)

When a story adds an "extension" notebook (e.g., `03_distribution_analysis.ipynb` added in a separate commit labeled "Story 3.2 extension"), **any files that notebook writes to disk must be staged and committed in the same commit as the notebook itself.** Failure to do this:
- Leaves generated artifacts in an untracked state that future git checkouts won't contain
- Makes synthesis stories (Story 3.3) miss the unpersisted output when reading only the committed findings files
- Sends downstream implementers on research spikes for questions already empirically answered

**Root cause (Story 3.2):** `03_distribution_analysis.ipynb` used `with open(findings_path, "a")` to append Section 7 to `statistical_exploration_findings.md`. The notebook was committed with executed outputs showing "Section 7 appended..." but the updated `.md` file was never staged. Story 3.3's dev agent only read `statistical_exploration_findings.md` (as directed) and never saw Section 7 — so distribution fitting was treated as an open research question despite being empirically solved.

**Code review checklist for extension notebooks:**
- [ ] For each `open(path, "w"/"a")` call in the notebook, verify the output file is staged and committed
- [ ] Check the notebook's execution output — if it says "appended to X.md", verify X.md shows that content in `git diff`
- [ ] Extension notebook commits should always include: the `.ipynb` file AND all output files it writes

**Append-mode risk:** `open(path, "a")` in notebooks means re-executing the notebook appends a duplicate section. Prefer the regex-replace pattern from the "Multi-Notebook Findings Files" note above. If append is used, note in the notebook's first cell that re-execution will duplicate sections — and add a guard:
```python
# Guard against duplicate sections
findings_text = findings_path.read_text()
if "## Section 7:" not in findings_text:
    with open(findings_path, "a") as f:
        f.write(section)
```

### Section Numbering in Multi-Notebook Findings Files ⭐ (Discovered Story 3.2/3.3)

When multiple notebooks contribute numbered sections to the same findings file, coordinate section numbers explicitly. If a planned section is dropped, downstream notebook section numbers become orphaned:
- `02_statistical_exploration.ipynb` → Sections 1–5 + "Known Data Limitations"
- A planned Section 6 was never created
- `03_distribution_analysis.ipynb` hardcoded "Section 7" → gap in numbering

**Pattern:** Define section numbers in a comment at the top of each notebook:
```python
# Section assignments for statistical_exploration_findings.md:
# Section 1: Scoring Distributions (02_statistical_exploration.ipynb)
# Section 2: Venue Effects (02_statistical_exploration.ipynb)
# ...
# Section 6: [RESERVED for Women's Analysis or drop and renumber]
# Section 7: Box-Score Distributions (03_distribution_analysis.ipynb)
```
Or renumber sequentially when sections are dropped rather than leaving gaps.

*Last Updated: 2026-02-20 (Story 3.1 Code Review — EDA notebook conventions, gitkeep pattern, nbconvert --output-dir, connector behavior verification)*

### `datetime.date.today()` — Capture Once Before Branching ⭐ (Discovered Story 4.2 Code Review)

Any function that validates a date against "today" and then uses "today" in an error message must capture `today` **once**, before the conditional check. Calling `datetime.date.today()` twice — once for comparison and once for the error message — creates a midnight race condition where both calls may return different dates:

```python
# ❌ Race condition: two calls may return different dates at midnight
if cutoff_date is not None and cutoff_date > datetime.date.today():
    today = datetime.date.today()  # ← may return different day!
    msg = f"cutoff_date {cutoff_date} is after today ({today})"  # contradictory!
    raise ValueError(msg)

# ✅ Capture once, use consistently
if cutoff_date is not None:
    today = datetime.date.today()
    if cutoff_date > today:
        msg = f"cutoff_date {cutoff_date} is after today ({today})"
        raise ValueError(msg)
```

**General rule:** Any value from `datetime.date.today()`, `datetime.datetime.now()`, `time.time()`, or `uuid.uuid4()` that is used multiple times in the same logical operation must be captured in a variable before the first use.

### `frozen=True` Dataclasses Do NOT Freeze Mutable Fields ⭐ (Discovered Story 4.2 Code Review)

`@dataclass(frozen=True)` prevents attribute *rebinding* but does NOT prevent mutation of mutable field contents:

```python
@dataclass(frozen=True)
class SeasonGames:
    games: list[Game]

season = SeasonGames(year=2024, games=[...], has_tournament=True)
season.games = []            # ← FrozenInstanceError ✓
season.games.append(game)    # ← silently succeeds! ✗ (false immutability)
```

**When to use `list` vs `tuple` in frozen dataclasses:**
- Use `list[T]` when: callers are expected to consume the data read-only, mutability is acceptable, or changing to `tuple` would break story spec
- Use `tuple[T, ...]` when: true immutability is required and downstream callers must not modify the collection

**Template pattern:** Document in the class docstring if `frozen=True` only prevents rebinding (e.g., "Note: `games` is a list and can be mutated by callers — do not modify after construction").

*Updated: 2026-02-21 (Story 4.2 Code Review — datetime.today() double-call race condition, frozen dataclass mutable field false-immutability)*
*Updated: 2026-02-20 (Story 3.2 extension — Epic 4 normalization design requirements from user)*
*Updated: 2026-02-20 (Story 3.2 Code Review — day-to-round mapping consistency, multi-notebook findings files, story file list completeness, apply-vs-vectorize)*
*Updated: 2026-02-20 (Story 3.3 Code Review — documentation synthesis story review pattern, ranked list rationale requirement)*
*Updated: 2026-02-20 (Story 3.3 Code Review Round 2 — extension notebook output commit gap, section numbering in multi-notebook findings files)*
*Updated: 2026-02-20 (Story 4.1 Code Review — research spike patterns: code examples, composite coverage gates, arXiv verification, Kaggle board limitation, spike sprint-status task wording)*

---

## Research Spike Patterns (Discovered: Story 4.1 Code Review, 2026-02-20)

### Code Examples in Research Docs Must Be Anti-Pattern-Free
Research spike documents (like `specs/research/`) will be used as implementation references by downstream developers. Any code shown — even pseudocode — must follow project mandates:
- **No `iterrows()`** — use `nx.from_pandas_edgelist()`, `.itertuples()`, or fully vectorized patterns
- **No magic numbers without comments**
- Always add an explicit warning note if a code snippet is pseudocode-only and must be rewritten for production

### Composite Ranking Recommendation Gate: Verify Coverage Before Recommending
When a research spike recommends a composite of external data systems (e.g., Massey ordinals SAG+POM+MOR+WLK), always:
1. Cross-check the recommended members against the confirmed full-coverage set
2. If any member is unconfirmed, make the recommendation conditional with an explicit fallback using only confirmed-coverage members
3. Add a verification step to the implementation story (4.3 in this case)

Pattern: *"Use SAG+POM+MOR+WLK if coverage confirmed; fallback: MOR+POM+DOL (confirmed full-coverage margin systems)"*

### arXiv Citation Verification in Research Spikes
ArXiv IDs use `YYMM.NNNNN` format. An ID of `2508.xxxxx` = August 2025. Always verify:
- The paper actually exists at arxiv.org
- The year stated in the document matches the arXiv submission year
- Papers near the knowledge cutoff (2025+) require explicit verification — do not rely on training knowledge alone

### Kaggle Discussion Boards Require Authentication
Kaggle competition discussion boards return JavaScript boilerplate when fetched without authentication. Research spikes cannot access board content directly. When writing story ACs for Kaggle research spikes:
- ✅ Use: *"Review documented community solutions and published writeups for Kaggle MMLM"*
- ❌ Avoid: *"Review Kaggle discussion boards"* (inaccessible without login)
- Secondary sources (Medium, mlcontests.com, public GitHub repos) are the practical approach

### Spike Sprint-Status Task Wording
Spike story task descriptions for the final sprint-status update should say `→ review` not `→ done`. Spikes go through code review like any other story before reaching `done`. Using `→ done` in the task description creates a false-completion scenario when the agent correctly sets `→ review`.

Correct pattern:
```
- [ ] X.X: Update sprint-status.yaml: `story-key` → `review` (code review advances to `done`)
```
*Next Review: [Set cadence - weekly? sprint boundaries?]*

### Never Use `git -C <path>` — Use Plain `git` Commands ⭐ (Discovered Story 4.3 create-story, 2026-02-21)

The Claude permission system matches Bash commands against pre-approved patterns in `.claude/settings.json`. The project pre-approves `git <cmd>` forms (e.g., `git log`, `git status`, `git commit`). Using `git -C /abs/path <cmd>` is a structurally different command and is NOT matched by the pre-approval — it requires an extra user approval prompt.

Since the primary working directory is always set to the project root (e.g., `/home/dhilg/git/NCAA_eval`), the `-C <path>` flag is never necessary.

**Rule for all agents:** Use `git log`, `git status`, `git diff`, `git add`, `git commit`, etc. directly — never `git -C /path/to/repo log ...`.

**Template action:** In any BMAD workflow instructions that reference git commands, write them without `-C`. If the workflow must specify an absolute path for some reason, use `cd` in a prior step rather than `-C`.

### Source-Scanning Tests Must Be Marked `@pytest.mark.no_mutation` ⭐ (Discovered Story 4.5 Code Review, 2026-02-21)

Any test that reads a source file via `Path(__file__).parent.../src/module.py` to check for banned patterns (e.g., `"iterrows" not in source_text`) will fail under mutmut because `__file__` resolves to the `mutants/` directory. `mutants/src/...` only contains files in `paths_to_mutate`; other source files aren't there.

**Always mark such tests `@pytest.mark.no_mutation`:**
```python
@pytest.mark.unit
@pytest.mark.smoke
@pytest.mark.no_mutation  # Uses Path(__file__) to find source; incompatible with mutmut
def test_no_iterrows() -> None:
    source_path = pathlib.Path(__file__).parent.parent.parent / "src" / "pkg" / "module.py"
    assert "iterrows" not in source_path.read_text()
```

**Template action:** Include this reminder in story templates for any `test_no_<banned_pattern>` test.

### Return Independent Dicts in Exception Fallback Paths (Discovered Story 4.5 Code Review, 2026-02-21)

When a function returns multiple dicts (e.g., hub+auth from `compute_hits()`), the exception/fallback path must return **two independent dict objects** — not the same object twice.

```python
# ❌ Bug: hub and auth are the same object — mutating hub corrupts auth
uniform = dict.fromkeys(nodes, score)
return uniform, uniform

# ✅ Correct: two independent dicts
return dict.fromkeys(nodes, score), dict.fromkeys(nodes, score)
```

This pattern applies anywhere a function returns multiple collections from a single-object fallback (dicts, lists, etc.).

### Test Convergence Failure Paths with `patch.object` (Discovered Story 4.5 Code Review, 2026-02-21)

Any numeric algorithm with a convergence exception guard (e.g., `except nx.PowerIterationFailedConvergence`) needs an explicit test. Use `unittest.mock.patch.object` to force the exception without constructing a pathological graph:

```python
from unittest.mock import patch

def test_hits_convergence_failure() -> None:
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])
    with patch.object(nx, "hits", side_effect=nx.PowerIterationFailedConvergence(1)):
        hub, auth = compute_hits(G)
    # verify fallback behavior + that hub/auth are independent objects
    hub[1] = 999.0
    assert auth[1] != 999.0
```

**Template action:** For any function that catches a convergence/iteration exception and returns a fallback, add a `test_<function>_convergence_failure` test to the story's AC list.

### Interface Pseudocode in Research Spikes Must Be Import-Complete (Discovered Story 5.1 Code Review, 2026-02-22)

When a research spike document defines an ABC interface with pseudocode, the import block must be **complete and correct**. Downstream story dev agents will copy-paste this pseudocode as the starting point for their implementation. Missing imports are copy-paste traps that cause runtime errors or `mypy --strict` failures.

**Checklist for interface pseudocode in research docs:**
- [ ] All types used in signatures have corresponding imports (`Path`, `pd.DataFrame`, `pd.Series`, `Any`, etc.)
- [ ] `from __future__ import annotations` is the first import (required by project convention)
- [ ] `from pathlib import Path` if any method signatures accept or return paths
- [ ] Domain types (e.g., `Game`, `Team`, `Season` from `ncaa_eval.ingest.schema`) imported if used in abstract method signatures
- [ ] **Check EVERY code block independently** — imports in one block do not carry to a separate code block later in the document
- [ ] Decorator factories have return type annotations (e.g., `register_model() -> Callable[...]`) — `mypy --strict` fails on unannotated functions
- [ ] API references match the actual library's method style (e.g., `XGBClassifier.load_model` is instance method, not class method)
- [ ] All abstract methods on a parent ABC are either overridden or explicitly given a concrete implementation with a clear `NotImplementedError` in subclasses
- [ ] `type: ignore[X]` comments are only included if `mypy --strict` actually raises that error code — spurious ignores mislead implementors

**Discovered in Story 5.1 Round 1:** `Model.save(path: Path)` was used without `from pathlib import Path`.
**Discovered in Story 5.1 Round 2:** `StatefulModel.update(game: Game)` used without `from ncaa_eval.ingest.schema import Game`; `StatefulFeatureServer` type annotation in separate dispatch code block had no import; `register_model` decorator missing `Callable` return type; `XGBClassifier.load_model()` incorrectly documented as class method; spurious `# type: ignore[override]` on `StatelessModel.predict()`.

### Dual-ABC Patterns: Document Evaluation Pipeline Dispatch Before Implementation (Discovered Story 5.1 Code Review, 2026-02-22)

When a research spike proposes a dual-ABC architecture (e.g., `StatefulModel` with per-game `predict(id_a, id_b)` vs. `StatelessModel` with batch `predict_proba(X)`), the document **must** show how a caller polymorphically dispatches across both types. Without this, the evaluation pipeline (which must call predictions uniformly) has no spec to follow and will either duplicate the dispatch logic inconsistently or stall in Story design.

**Template pattern:** Add a `## Evaluation Pipeline Dispatch` subsection to the ABC interface section of any dual-contract design, showing an `isinstance`-dispatch snippet:

```python
if isinstance(model, StatefulModel):
    return model.predict(team_a_id, team_b_id)
elif isinstance(model, StatelessModel):
    return model.predict_proba(X_test)
```

**Discovered in Story 5.1:** The dual-contract ABC was specified but no dispatch guidance was provided; added in code review (Section 5.3 of `specs/research/modeling-approaches.md`).

### Spike Story Post-PO SM Task Must Be an Explicit Task (Not Just an AC) (Discovered Story 5.1 Code Review, 2026-02-22)

AC 8 of Story 5.1 (post-PO SM downstream update) was present as an acceptance criterion but had no corresponding task in the Tasks section. This violates the Story 4.1 SM retrospective pattern in template-requirements.md (see "Spike story post-PO-approval checklist" in BMAD Workflow Preferences). The task must explicitly exist so the SM has a tracked work item to complete.

**Template pattern:** All spike stories must include as the **last task**:
```
- [ ] Task N: Post-PO SM downstream update (AC: #N) — SM WORK, NOT DEV WORK
  - [ ] N.1 After PO approves spike findings, SM updates downstream story descriptions in epics.md
  - [ ] N.2 SM adds new story placeholders as needed
  - [ ] N.3 SM moves deferred items to Post-MVP Backlog
  - [ ] N.4 SM updates sprint-status.yaml: `story-key` → `done` (only after all post-PO work complete)
```

**Discovered in Story 5.1 Code Review, 2026-02-22.** Prior retrospective about AC existence discovered in Story 4.1 SM work (2026-02-21).

### Code Review Must Not Advance Sprint-Status Past `review` for Open PO Gates (Discovered Story 5.1 Code Review Round 2, 2026-02-22)

When a spike story has a **PO decision gate AC** (e.g., "PO reviews and approves scope"), the code review agent must set sprint-status to `review`, NOT `done`. The story should not advance to `done` until:
1. The PO has reviewed and approved the spike findings (PO gate AC)
2. The SM has completed the downstream epic update (post-PO SM AC)

**Failure mode:** Round 1 review of Story 5.1 found "All ACs implemented" and set sprint-status → `done`, but AC 7 (PO gate) was explicitly unfulfilled. The sprint-status had to be reverted to `review` in Round 2.

**Rule:** If any AC contains the phrase "product owner reviews", "PO approves", "decision gate", or similar, the code review agent MUST leave the story in `review` state regardless of how many other ACs are satisfied. Advancing to `done` requires human PO action, not agent action.

### Multi-Block Pseudocode: Each Code Block Has Independent Imports (Discovered Story 5.1 Code Review Round 3, 2026-02-23)

When fixing an import gap in one code block (e.g., Section 5.2 ABC definition), **always check every other code block in the same document independently**. Imports added to Block A do not carry into Block B. Over three rounds of Story 5.1 review, each round found a new import gap in a different code block:
- Round 1: `Path` missing from Block A (§5.2 ABC)
- Round 2: `Game` missing from Block A; `StatefulFeatureServer` missing from Block B (§5.3 dispatch)
- Round 3: `Model` and `StatefulModel` missing from Block B (§5.3 dispatch — same block Round 2 partially fixed)

**Rule:** After any import fix, scan ALL other code blocks in the document for the same class of gap, especially blocks that reference types defined in the same ABC.

**Updated checklist item:** "Check EVERY code block independently — and after fixing imports in one block, re-scan ALL blocks for the same type."

### Concrete Methods in ABC Pseudocode: Use `raise NotImplementedError`, Not `...` (Discovered Story 5.1 Round 3, 2026-02-23)

In Python, `...` (Ellipsis) is idiomatic for abstract method stubs. Using `...` as the body of a **concrete** helper method in pseudocode creates ambiguity: Story 5.2 dev agents may either treat it as abstract (wrong — defeats template inheritance) or copy `...` literally (causes `None` return → `TypeError` at runtime).

**Rule:** In research doc pseudocode:
- **Abstract methods** → use `...` body or `pass` (Python convention for "subclass must implement")
- **Concrete methods with placeholder bodies** → use `raise NotImplementedError("message")` + a docstring comment explicitly stating "CONCRETE — implement in `StatefulModel` body, not subclasses"

### Classmethod Factory Return Types: Use `Self`, Not Parent Class (Discovered Story 5.1 Round 3, 2026-02-23)

`load(cls, path: Path) -> "Model"` (or any similar factory classmethod returning the parent class) prevents type narrowing: `EloModel.load(path)` returns `Model`, not `EloModel`. Use `typing.Self` (Python 3.11+, PEP 673) for all factory classmethods:

```python
from typing import Self

@classmethod
@abstractmethod
def load(cls, path: Path) -> Self: ...
```

**Rule:** Any `@classmethod` that creates and returns an instance of `cls` should return `Self`, not the ABC name.

### Config Spec → Hyperparameter Table Sync (Discovered Story 5.1 Round 3, 2026-02-23)

When adding a parameter to a Pydantic config class in one section (e.g., §5.5), always update ALL hyperparameter tables in other sections that describe that model (e.g., §6.4). Over rounds of Story 5.1 review:
- Round 2 added `min_child_weight` to `XGBoostModelConfig` (§5.5) and §6.4 table correctly
- Round 2 added `early_game_threshold` to `EloModelConfig` (§5.5) but missed the §6.4 Elo table

**Rule:** Whenever a parameter is added to or removed from a Pydantic model config in pseudocode, search the document for all hyperparameter tables referencing that model and update them to match.

### `pd.isna()` Is the Universal Null Guard for pandas Values (Discovered Story 5.2 Code Review, 2026-02-23)

When extracting scalar values from a DataFrame row (via `X.loc[idx]` or `itertuples()`), null checks must handle three distinct representations: Python `None`, `float('nan')` (numeric columns), and `pd.NaT` (datetime columns). The pattern `isinstance(raw, float) and pd.isna(raw)` only catches float NaN and silently fails for `pd.NaT`.

**Rule:** Always use `pd.isna(value)` as the sole null guard — it handles all three types uniformly:
```python
# ❌ Misses pd.NaT — crashes when date column has NaT values
if raw_date is not None and not (isinstance(raw_date, float) and pd.isna(raw_date)):
    date_val = pd.Timestamp(raw_date).date()

# ✅ Handles None, float NaN, and pd.NaT correctly
if not pd.isna(raw_date):
    date_val = pd.Timestamp(raw_date).date()
```

### `itertuples()` Must Be Used Consistently in ABC Template Methods (Discovered Story 5.2 Code Review, 2026-02-23)

When an ABC template method is documented to use `itertuples()` for row iteration (e.g., `predict_proba()`), all sibling template methods that iterate the same DataFrame must also use `itertuples()`. Mixing `itertuples()` in one method with `X.loc[idx]` in another creates a ~5–10× performance inconsistency that is invisible from the outside.

**Rule:** Pick one iteration pattern per class and apply it consistently. When object construction (not pure calculation) requires per-row iteration, `itertuples()` is preferred over `X.loc[idx]` for both speed and style consistency.

**Companion rule:** Hoist all `"col_name" in X.columns` checks outside the loop — they evaluate identically on every iteration and pay O(c) overhead n times unnecessarily.

### Pydantic Config `model_name`: Use `Literal` to Lock the Name at the Type Level (Discovered Story 5.2 Code Review, 2026-02-23)

When a concrete `ModelConfig` subclass has a fixed `model_name`, type it as `Literal["name"]` not `str`. This prevents `LogisticRegressionConfig(model_name="wrong")` from passing mypy silently and locks the config identity at the type level.

```python
# ❌ Allows any string — no enforcement
model_name: str = "logistic_regression"

# ✅ Enforced at type-check time
model_name: Literal["logistic_regression"] = "logistic_regression"
```

### Pytest Generator Fixtures: Annotate as `Generator[None, None, None]` (Discovered Story 5.2 Code Review, 2026-02-23)

A pytest fixture that uses `yield` for setup/teardown is a generator function. Typing it `-> None` triggers `mypy --strict`'s `[misc]` error and requires a suppression comment. The correct annotation uses `collections.abc.Generator`:

```python
# ❌ Triggers mypy [misc] — requires type: ignore[misc]
@pytest.fixture(autouse=True)
def _clean_registry() -> None:  # type: ignore[misc]
    ...
    yield
    ...

# ✅ mypy strict clean — no suppression needed
from collections.abc import Generator

@pytest.fixture(autouse=True)
def _clean_registry() -> Generator[None, None, None]:
    ...
    yield
    ...
```

### Stateful Model `set_state()`: Always Validate Input Dict Structure (Discovered Story 5.3 Code Review, 2026-02-23)

When a `set_state(state)` method restores model internals from a snapshot dict, it must validate the dict structure before applying it. Without validation, missing keys cause opaque `KeyError` on access, non-dict values cause `TypeError` deep inside the engine, and malformed JSON loaded from disk corrupts the model silently — with no error until the next `predict()` call.

**Pattern:**
```python
def set_state(self, state: dict[str, Any]) -> None:
    if "ratings" not in state or "game_counts" not in state:
        missing = {"ratings", "game_counts"} - state.keys()
        msg = f"set_state() state dict missing required keys: {missing}"
        raise KeyError(msg)
    if not isinstance(state["ratings"], dict) or not isinstance(state["game_counts"], dict):
        msg = "set_state() 'ratings' and 'game_counts' must be dicts"
        raise TypeError(msg)
    # direct assignment to private attrs — see encapsulation note in elo.py set_state()
```

**Rule:** Every `set_state()` implementation must have a test for missing keys (`pytest.raises(KeyError)`) and wrong types (`pytest.raises(TypeError)`).

### Multi-File `save()` / `load()`: Check All Files Before Loading (Discovered Story 5.3 Code Review, 2026-02-23)

When `save()` writes multiple files (e.g., `config.json` + `state.json`), `load()` must verify ALL expected files exist before attempting to read any of them. A crashed or interrupted `save()` leaves the directory in a half-written state; loading partial state produces confusing errors instead of a clear "save was incomplete" message.

**Pattern:**
```python
@classmethod
def load(cls, path: Path) -> Self:
    missing = [p for p in (path / "config.json", path / "state.json") if not p.exists()]
    if missing:
        missing_names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(
            f"Incomplete save at {path!r}: missing {missing_names}. "
            "The save may have been interrupted."
        )
```

**Rule:** Include `test_load_missing_X_raises` tests for each expected save file.

### Config Bridge Functions: Use `dataclasses.fields()` Not Manual Field Copy (Discovered Story 5.3 Code Review, 2026-02-23)

When a Pydantic config class (`EloModelConfig`) must be converted to a frozen dataclass (`EloConfig`), avoid manually listing every field. Manual mapping silently omits any new field added to the dataclass later, causing silent fallback to defaults.

**Pattern:**
```python
@staticmethod
def _to_elo_config(config: EloModelConfig) -> EloConfig:
    elo_field_names = {f.name for f in dataclasses.fields(EloConfig)}
    kwargs = {k: v for k, v in config.model_dump().items() if k in elo_field_names}
    return EloConfig(**kwargs)
```

The `if k in elo_field_names` filter drops Pydantic-only fields (like `model_name`) that don't exist in the dataclass. Any new field added to `EloConfig` is picked up automatically as long as `EloModelConfig` also adds the matching field.

### `fit()` Accumulation vs Reset: Document and Test (Discovered Story 5.3 Code Review, 2026-02-23)

`StatefulModel.fit()` is designed for sequential accumulation — calling it twice on the same instance accumulates ratings from both datasets. This differs from scikit-learn convention where `fit()` resets state. Callers who expect idempotency will get silent bugs in train/validation split workflows.

**Rule:** Any stateful model that inherits an accumulating `fit()` must:
1. Document this behavior in the class or method docstring: "Calling `fit()` accumulates state. To start fresh, instantiate a new model."
2. Include a `test_fit_twice_accumulates_state` test that asserts ratings change after the second call.

### `set_state()` Must Coerce String Keys to `int` (Discovered Story 5.3 Code Review Round 2, 2026-02-23)

`set_state()` validates structure (missing keys, wrong types) but must also coerce string dict keys to `int`. JSON serialization always produces string keys. A caller who does `json.loads(text)` and passes the result directly to `set_state()` will end up with string-keyed internal dicts; then `get_rating(team_id_int)` silently returns `initial_rating` for every team — wrong predictions with no error signal.

The previous round's validation fix was incomplete: it checked types but not key types.

**Pattern:**
```python
def set_state(self, state: dict[str, Any]) -> None:
    # ... existing structure validation ...
    # Coerce str → int for JSON-decoded dict compatibility
    self._engine._ratings = {int(k): float(v) for k, v in ratings.items()}
    self._engine._game_counts = {int(k): int(v) for k, v in game_counts.items()}
```

**Rule:** The docstring for `set_state()` must mention that string keys are accepted and coerced to `int`. Add a `test_set_state_coerces_string_keys` test that passes `{"1": 1600.0}` and verifies `get_rating(1)` returns `1600.0`.

### Property-Based Tests for Pure Prediction Functions (Discovered Story 5.3 Code Review Round 2, 2026-02-23)

Style guide Section 6.2 mandates property-based tests (Hypothesis) for pure functions. `_predict_one(team_a_id, team_b_id) -> float` is pure: same ratings → same probability, no I/O, no side effects. Despite Hypothesis being installed (`pyproject.toml: hypothesis = "*"`), the initial implementation had no property tests.

**Key invariants to property-test for any prediction function:**
1. **Bounded output:** `0.0 ≤ P(A wins) ≤ 1.0` for all valid rating inputs.
2. **Symmetry:** `P(A beats B) + P(B beats A) == 1.0`.

**Pattern:**
```python
from hypothesis import given, strategies as st

class TestPredictOneProperties:
    @given(
        rating_a=st.floats(min_value=0.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
        rating_b=st.floats(min_value=0.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
    )
    def test_predict_one_always_in_unit_interval(self, rating_a: float, rating_b: float) -> None:
        model = EloModel()
        model._engine._ratings[1] = rating_a
        model._engine._ratings[2] = rating_b
        prob = model._predict_one(1, 2)
        assert 0.0 <= prob <= 1.0

    @given(
        rating_a=st.floats(min_value=0.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
        rating_b=st.floats(min_value=0.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
    )
    def test_predict_one_symmetric(self, rating_a: float, rating_b: float) -> None:
        model = EloModel()
        model._engine._ratings[1] = rating_a
        model._engine._ratings[2] = rating_b
        prob_ab = model._predict_one(1, 2)
        prob_ba = model._predict_one(2, 1)
        assert prob_ab + prob_ba == pytest.approx(1.0, abs=1e-12)
```

**Rule:** Every `_predict_one()` implementation must have Hypothesis tests for the bounded-output and symmetry invariants.

### Configurable Hyperparameters: Expose ALL Tunable Knobs in the Config Class (Discovered Story 5.4 Code Review, 2026-02-23)

When a framework parameter (like XGBoost's `scale_pos_weight`) is mentioned in the story's AC with phrasing "set accordingly if…", it must be an **actual field** in the `ModelConfig` subclass — not just documented in a docstring. Documentation without a settable field means the AC is only partially met (the caller cannot configure the parameter without subclassing or monkey-patching).

**Pattern:**
```python
class XGBoostModelConfig(ModelConfig):
    scale_pos_weight: float | None = None  # None → framework default (1.0)
    # ... other fields

def __init__(self, config: XGBoostModelConfig | None = None) -> None:
    kwargs: dict[str, object] = { ... base params ... }
    if self._config.scale_pos_weight is not None:  # Only pass if set
        kwargs["scale_pos_weight"] = self._config.scale_pos_weight
    self._clf = XGBClassifier(**kwargs)
```

**Rule:** Every AC that says "set X accordingly" or "configure X" must result in a field on the config. Docstring documentation alone ≠ AC satisfaction.

### Unfitted Model Guards: All I/O Methods Must Check `_is_fitted` (Discovered Story 5.4 Code Review, 2026-02-23)

Both `predict_proba()` and `save()` must guard against being called before `fit()`. Without guards:
- `predict_proba()` raises an opaque internal library error (XGBoost core error)
- `save()` silently persists an empty/invalid model that loads successfully but predicts garbage

**Pattern:**
```python
class SomeModel(Model):
    def __init__(self, config=None) -> None:
        self._is_fitted = False  # Track fit state

    def fit(self, X, y) -> None:
        # ... training ...
        self._is_fitted = True

    def predict_proba(self, X) -> pd.Series:
        if not self._is_fitted:
            msg = "Model must be fitted before calling predict_proba"
            raise RuntimeError(msg)
        # ... prediction ...

    def save(self, path) -> None:
        if not self._is_fitted:
            msg = "Model must be fitted before saving"
            raise RuntimeError(msg)
        # ... save files ...

    @classmethod
    def load(cls, path) -> Self:
        # ... load files ...
        instance._is_fitted = True  # Mark loaded model as fitted
        return instance
```

**Rule:** Every stateless `Model` implementation must initialize `_is_fitted = False` in `__init__`, set `_is_fitted = True` after `fit()` and after `load()`, and guard `predict_proba()` and `save()` with clear `RuntimeError` messages.

### Hypothesis Deadline Violations: Training-Heavy Tests Need `deadline=None` (Discovered Story 5.4 Code Review, 2026-02-23)

When a Hypothesis `@given` test body includes model training (even with a small `n_estimators`), the first invocation can trigger a `DeadlineExceeded` flaky failure. Hypothesis's default deadline is 200ms; ML model training often exceeds this on the first example (no JIT warm-up).

Two patterns to avoid this:

**Pattern A — Module-level fixture (preferred for shared trained models):**
```python
_TRAINED_MODEL: XGBoostModel | None = None

def _get_trained_model() -> XGBoostModel:
    global _TRAINED_MODEL  # noqa: PLW0603
    if _TRAINED_MODEL is None:
        X, y = _make_train_data()
        _TRAINED_MODEL = XGBoostModel(XGBoostModelConfig(n_estimators=20, early_stopping_rounds=5))
        _TRAINED_MODEL.fit(X, y)
    return _TRAINED_MODEL

@given(feat_a=st.lists(...), feat_b=st.lists(...))
@settings(max_examples=50, deadline=None)  # deadline=None because predict is fast but JIT varies
def test_predict_proba_bounded(self, feat_a, feat_b) -> None:
    model = _get_trained_model()  # Never trains inside @given body
    X_test = pd.DataFrame({"feat_a": feat_a, "feat_b": feat_b})
    preds = model.predict_proba(X_test)
    assert (preds >= 0.0).all() and (preds <= 1.0).all()
```

**Pattern B — Suppress deadline explicitly:**
```python
@settings(max_examples=50, deadline=None)
```

**Rule:** Any `@given` test that trains a model OR calls predict on an untriggered JIT path must use `deadline=None`. Training must NEVER happen inside the `@given` body — train once, test many examples.

### Duplicate Tests Inflate Test Count; Use Distinct Assertions (Discovered Story 5.4 Code Review, 2026-02-23)

When story tasks map to distinct requirements (e.g., "trains successfully" vs "output length matches input"), the tests must be genuinely distinct. A test that calls `model.fit(X, y)` and asserts `len(preds) == len(X)` covers BOTH "trains successfully" AND "length matches." A second test with identical assertions inflates the count and provides no additional coverage.

**Distinguishing techniques:**
- "Trains successfully" → assert no exception raised (no `assert` on predictions at all, or use `model.fit(X, y)  # Must not raise`)
- "Output length matches input" → use a *subset* of training data for prediction to prove length = `len(X_subset)`, not `len(X_train)`
- "Output is bounded [0,1]" → check `(preds >= 0).all() and (preds <= 1).all()`

**Rule:** Before adding a test, verify it exercises a code path or assertion not already covered by existing tests. If two tests have identical `assert` statements on the same data, merge or differentiate them.

### AC Text vs. Dev Notes Dataclass Spec Conflict: Dev Notes Win (Discovered Story 6.1 Code Review, 2026-02-23)

When a high-level AC mentions fields in a structured return type (e.g., "bin counts, bin edges") but the Dev Notes dataclass spec lists different fields, the Dev Notes take precedence as the authoritative specification. However, the code review should flag the discrepancy so the PO can confirm whether the AC was intentionally simplified or the spec omitted something valuable downstream.

**Pattern:** When the AC says field X is required but the spec omits it — add it anyway if:
1. Downstream stories will need it (avoid a breaking dataclass change later), OR
2. The caller would have to recompute it from stored fields (convenience ergonomics)

In Story 6.1: AC5 mentioned `bin_edges` in `ReliabilityData`; the Dev Notes only specified `n_bins`. Since `bin_edges = np.linspace(0.0, 1.0, n_bins+1)` — easily derivable — and Story 7.4 (reliability diagrams) will need them for visualization, `bin_edges` was added during code review rather than forcing Story 7.4 to recompute and risk inconsistency.

### Validate y_true Is Binary in All Custom Metric Functions (Discovered Story 6.1 Code Review, 2026-02-23)

When writing custom metric functions that wrap sklearn (which has its own validation) AND implement custom numpy logic (ECE), the shared `_validate_inputs()` helper must validate `y_true ∈ {0, 1}`. Without this, custom functions silently accept non-binary labels while sklearn-backed wrappers raise, creating inconsistent behavior.

**Pattern:** `_validate_inputs()` must check:
1. Non-empty arrays
2. Matching lengths
3. **y_true is binary** (`np.all((y_true == 0) | (y_true == 1))`)
4. y_prob ∈ [0, 1]

### Vectorized Binning Comment Must Match 0-Indexed vs. 1-Indexed Reality (Discovered Story 6.1 Code Review, 2026-02-23)

When using `np.digitize(y_prob, bin_edges[1:-1])` (passing only interior edges), the returned indices are **0-indexed** (0 for values below the first interior edge). Clipping to `[0, n_bins-1]` is correct. A comment saying "assigns bin index starting at 1; clip to [1, n_bins]" is **wrong** and misleads maintainers.

**Rule:** Use interior edges (`bin_edges[1:-1]`) when you want 0-indexed bins `[0..n_bins-1]`. Use full edges (`bin_edges`) when you want 1-indexed bins `[1..n_bins]`. Never mix the indexing convention in comments vs. code.

### n_bins Validation Must Be Explicit for Custom Binning Functions (Discovered Story 6.1 Code Review, 2026-02-23)

A function that implements custom numpy binning (like ECE with `np.digitize` + `np.bincount`) must explicitly validate `n_bins >= 1`. Without it, `n_bins=0` causes a cryptic numpy internal error (`ValueError: 'list' argument must have no negative elements`). Functions that delegate to sklearn get this validation for free via sklearn's `InvalidParameterError`.

**Pattern:** Add at the start of any function with `n_bins` parameter:
```python
if n_bins < 1:
    msg = f"n_bins must be >= 1, got {n_bins}."
    raise ValueError(msg)
```

### `frozen=True` Dataclasses with Numpy Array Fields: Use `.copy()` at Construction (Discovered Story 6.1 Code Review Round 2, 2026-02-23)

When a `frozen=True` dataclass contains `npt.NDArray` fields (numpy arrays), the frozen semantics only prevent field *rebinding* — callers can still mutate array contents in-place. This silently breaks the immutability contract:

```python
@dataclass(frozen=True)
class ReliabilityData:
    fraction_of_positives: npt.NDArray[np.float64]

result = compute_reliability(...)
result.fraction_of_positives = np.array([0.0])  # ← FrozenInstanceError ✓
result.fraction_of_positives[0] = 0.999          # ← silently succeeds! ✗
```

**Fix:** Return `.copy()` of all numpy array fields in the constructor call so callers receive an independent copy:

```python
return ReliabilityData(
    fraction_of_positives=fraction_of_positives.copy(),
    mean_predicted_value=mean_predicted_value.copy(),
    bin_counts=bin_counts.copy(),
    bin_edges=bin_edges.copy(),
    n_bins=n_bins,
)
```

This also protects against the case where the array returned from a library call (e.g., sklearn) is an internal view — copying ensures the dataclass owns its data.

**Related:** The general `frozen=True` / mutable field pattern is documented above (Story 4.2, `list[Game]` case). This entry specifically addresses numpy array fields where `.copy()` is the idiomatic fix (vs. `tuple()` for lists).

### Test Coverage Symmetry: All Input Validation Tests Must Cover ALL Functions (Discovered Story 6.1 Code Review Round 2, 2026-02-23)

When a shared validation helper (`_validate_inputs`) covers N metrics, edge-case tests for each validation condition should exist for **all N metrics** — not just the first few. Asymmetric coverage leads to silently untested code paths.

**Pattern:** For each validation condition (empty arrays, mismatched lengths, non-binary y_true, probs outside [0,1]), add a dedicated `test_<condition>_<metric>` test for every public function. Use a checklist during code review:
- `test_empty_arrays_raise_<metric>` → 5 tests for 5 metrics ✓
- `test_mismatched_lengths_<metric>` → 5 tests ✓
- `test_non_binary_y_true_<metric>` → must cover ALL metrics (log_loss, brier, roc_auc, ece, reliability) — Story 6.1 originally missed `roc_auc`.

### NumPy Docstring Style Drift: Enforce Google Style for All src/ Modules (Discovered Story 6.2 Code Review, 2026-02-23)

Dev agents default to **NumPy docstring style** (`Parameters\n----------`, `Returns\n-------`, `Raises\n------`) when writing docstrings for data-science-oriented code. This project mandates **Google style** (`Args:`, `Returns:`, `Raises:`) per STYLE_GUIDE Section 1 and `pyproject.toml` pydocstyle convention.

**Drift is invisible to Ruff** — the Google convention in `[tool.ruff.lint.pydocstyle]` only activates when `D` rules are added to `extend-select`. Until then, NumPy-style docstrings pass all linters silently.

**Template requirement:** Code review must explicitly verify all new docstrings in `src/` use Google style. Flag NumPy-style docs as a MEDIUM finding. Add a reminder to Dev Agent story notes: *"Use Google docstring style (Args:, Returns:, Raises:), NOT NumPy style (Parameters\n----------)"*.

### `mode: str` Public APIs Must Validate at Entry Point (Discovered Story 6.2 Code Review, 2026-02-23)

When a public function accepts a `mode: str` (or any str standing in for an enum), validation must happen **at the function's entry point**, not delegated to a downstream internal call. The delegation pattern causes:
1. Cryptic stack traces pointing into library internals rather than the user's call site
2. Potential for partial side effects (e.g., cache population) before the error surfaces
3. Violation of the "fail fast" principle

**Pattern:** Always validate enum-like strings immediately before any other logic:
```python
_VALID_MODES: frozenset[str] = frozenset({"batch", "stateful"})

def walk_forward_splits(seasons, feature_server, *, mode: str = "batch") -> ...:
    if mode not in _VALID_MODES:
        msg = f"mode must be 'batch' or 'stateful', got {mode!r}"
        raise ValueError(msg)
    # ... rest of function
```

Consider `typing.Literal["batch", "stateful"]` as the type annotation when the valid set is small and stable — mypy catches invalid literals at call sites without runtime overhead.

### Exception Guard Breadth in Per-Item try/except Blocks (Discovered Story 6.3 Code Review, 2026-02-23)

When wrapping individual callable invocations to store `NaN` on failure (e.g., metric functions per fold), catching only `ValueError, ZeroDivisionError` is **too narrow**. Metric libraries can raise `RuntimeError`, `OverflowError`, `FloatingPointError`, `numpy.exceptions.FloatingPointError`, or other exceptions depending on edge-case inputs (single-class test sets, all-NaN inputs, etc.). An uncaught exception propagates out of the worker and crashes the entire parallel job.

**Pattern:** Use `except Exception:  # noqa: BLE001` for per-item error isolation:
```python
for name, fn in metric_fns.items():
    try:
        metrics[name] = fn(y_true_np, y_prob_np)
    except Exception:  # noqa: BLE001
        metrics[name] = float("nan")
```

The `# noqa: BLE001` suppresses Ruff's "blind exception" warning with explicit acknowledgement. This is appropriate when the intent is "any failure → NaN, continue".

### Module-Level Default Dicts Must Use `MappingProxyType` (Discovered Story 6.3 Code Review, 2026-02-23)

Module-level `dict` objects used as defaults (e.g., `DEFAULT_METRICS = {"log_loss": log_loss, ...}`) are mutable — callers can silently mutate them (`DEFAULT_METRICS["injected"] = ...`), affecting all subsequent users in the same process. This is particularly dangerous in long-running data science workflows where modules are imported once.

**Pattern:** Wrap with `types.MappingProxyType` and annotate as `Mapping[...]`:
```python
import types
from collections.abc import Mapping

DEFAULT_METRICS: Mapping[str, Callable[...]] = types.MappingProxyType({
    "log_loss": log_loss,
    "brier_score": brier_score,
    ...
})
```

The `run_backtest` caller still does `dict(DEFAULT_METRICS)` to get a mutable working copy — `MappingProxyType` is shallow-copied correctly by `dict()`.

### Frozen Dataclass Result Fields: Use `Mapping` Not `dict` (Discovered Story 6.3 Code Review, 2026-02-23)

`frozen=True` on a dataclass prevents reassigning fields but does NOT prevent mutating mutable field values. A `dict` field on a frozen dataclass can have new keys injected by any consumer.

**Pattern:** Annotate metric/config fields as `Mapping[str, float]` instead of `dict[str, float]`. Python's type system (and mypy `--strict`) will then flag mutation attempts at call sites:
```python
@dataclasses.dataclass(frozen=True)
class FoldResult:
    metrics: Mapping[str, float]   # ✅ read-only contract
    # NOT: metrics: dict[str, float]   # ❌ mutable despite frozen
```

### AC Performance Targets Need Explicit Test Stubs (Discovered Story 6.3 Code Review, 2026-02-23)

When an AC specifies a performance target (e.g., "10-year backtest completes in < 60 seconds"), the dev agent often omits a test because the target requires end-to-end infrastructure not available during unit testing. This creates a silent gap: the AC is marked [x] in the story but has zero test coverage.

**Pattern:** Always add a `@pytest.mark.skip(reason="requires real data pipeline")` stub in `TestPerformance` class that:
1. Documents the exact assertion (`result.elapsed_seconds < 60.0`)
2. Explains what infrastructure is needed to un-skip it
3. Shows the exact `run_backtest(RealModel(), real_server, ...)` invocation

This makes the coverage gap visible, tracks the intent, and gives future engineers a ready-to-enable test.

### Determinism Tests Require Data-Dependent Models (Discovered Story 6.3 Code Review, 2026-02-23)

A determinism test comparing parallel vs. sequential results using a **constant-prediction model** (e.g., always returns 0.5) is not meaningful. Constant predictions produce identical metric values regardless of parallelism ordering bugs — the test passes even if fold results are returned in the wrong order or associated with the wrong year.

**Pattern:** Add a `_DataDependentModel` alongside the `_FakeStatelessModel` in test helpers. The data-dependent model should return predictions that actually vary with input feature values (e.g., based on column means). Use it in the determinism test to provide meaningful coverage:
```python
class _DataDependentModel(_FakeStatelessModel):
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        feat = _feature_cols(X)
        col_mean = X[feat[0]].mean() if feat else 0.5
        prob = float(np.clip(col_mean / (col_mean + 1.0), 0.01, 0.99))
        return pd.Series(prob, index=X.index)
```

### Test Helper DataFrames Must Include Real Feature Columns (Discovered Story 6.3 2nd Code Review, 2026-02-23)

A synthetic test DataFrame that contains **only metadata columns** (all in `METADATA_COLS`) will produce an empty feature list from `_feature_cols()`. Stateless model tests will then exercise `model.fit(df[[]])` — a zero-column slice — which passes trivially if the fake model ignores `X`. The column-filtering code path is never actually validated.

**Pattern:** Always add at least two non-metadata numeric columns to `_make_season_df()` helpers (e.g., `elo_diff`, `win_pct_diff`). This ensures `_feature_cols()` returns a non-empty list, column-filtering is exercised, and data-dependent models have values to work with:
```python
"elo_diff": rng.normal(0.0, 50.0, size=total),
"win_pct_diff": rng.uniform(-0.5, 0.5, size=total),
```

### Public API Promotion for Shared Helpers (Discovered Story 6.3 2nd Code Review, 2026-02-23)

When a private helper function (single underscore prefix, e.g., `_feature_cols`) is imported by an external module (e.g., `cli/train.py`), it should be promoted to a public API. Options:
1. Remove the underscore prefix: `feature_cols` → add to `__init__.py` `__all__`.
2. If the function is truly internal to the module (logic-sharing only), expose a public wrapper or re-export from `__init__.py`.

Importing private functions from external modules violates the module encapsulation contract and creates implicit coupling that mypy cannot enforce. This is a common drift issue when moving shared logic between modules (e.g., moving `METADATA_COLS` from `cli` to `evaluation.backtest`).

### Propagated Exceptions Must Be Documented in Raises: Section (Discovered Story 6.3 2nd Code Review, 2026-02-23)

When a public function calls another function that raises `ValueError` (or other exceptions) with no try/except, the caller's docstring `Raises:` section must document it. "Propagated from X" is sufficient. Missing this creates a contract gap — callers have no API-level warning that the exception can occur.

**Pattern:** After implementing a function, grep for all uncaught exceptions from called functions and ensure they appear in `Raises:`.

### Docstring Style Drift in Modified Files (Discovered Story 6.3 2nd Code Review, 2026-02-23)

When a story modifies an existing file (e.g., to update an import), the dev agent should inspect the file's docstrings for style compliance. Files written before the Google-docstring mandate was established often still use NumPy style (`Parameters\n----------`). Any file **touched** by a story is an opportunity to correct docstring drift — failing to do so perpetuates the style inconsistency and will be flagged in every subsequent code review.
