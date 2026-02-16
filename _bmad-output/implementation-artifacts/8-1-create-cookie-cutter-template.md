# Story 8.1: create-cookie-cutter-template

Status: ready-for-dev

---

## Story

As a **developer starting new Python projects**,
I want **a cookie-cutter template derived from NCAA_eval that includes all learnings, dev stack, and BMAD integration**,
so that **I can bootstrap future projects with proven patterns and avoid re-learning/re-implementing toolchain decisions**.

---

## Acceptance Criteria

1. **Cookiecutter Structure Created**
   - GIVEN the NCAA_eval project is complete
   - WHEN I run `cookiecutter <template-repo>`
   - THEN a new project is generated with all base configurations, BMAD integration, and documented patterns
   - AND the generated project matches the NCAA_eval structure and tooling decisions

2. **Cruft Support Implemented**
   - GIVEN a project generated from the template
   - WHEN template updates are released
   - THEN I can run `cruft update` to merge non-conflicting improvements
   - AND conflicts are clearly identified for manual resolution

3. **BMAD Integration Updateable**
   - GIVEN BMAD releases a new version
   - WHEN the template includes BMAD update hooks/documentation
   - THEN users can update BMAD independently of other template changes
   - AND custom agent modifications are preserved

4. **Complete Dev Stack Templated**
   - GIVEN the template is used
   - THEN Poetry configuration, Python 3.12+ requirement, mypy strict, Ruff linting are pre-configured
   - AND all dev dependencies (pytest, hypothesis, mutmut, pre-commit, nox, etc.) are included
   - AND pyproject.toml reflects all NCAA_eval quality tooling decisions

5. **Testing Strategy Templated**
   - GIVEN a new project from template
   - THEN the 4-tier testing strategy is documented and test structure exists
   - AND smoke/integration/property test markers are configured
   - AND coverage targets and mutation testing setup are ready to use

6. **Documentation Preserved**
   - GIVEN the template repository
   - THEN STYLE_GUIDE.md, TESTING_STRATEGY.md, and template-requirements.md are included
   - AND a comprehensive README explains template usage, customization points, and update procedures

---

## Tasks / Subtasks

### Phase 1: Extract & Analyze (AC: 1)
- [ ] Finalize template-requirements.md with all project decisions (AC: 1)
  - [ ] Complete Dev Stack section with final dependency rationale
  - [ ] Complete Testing Strategy section with patterns and examples
  - [ ] Document all BMAD agent modifications made during NCAA_eval
  - [ ] Document project structure decisions and evolution
  - [ ] Capture lessons learned throughout implementation
- [ ] Identify all parameterizable values (project_name, author, etc.) (AC: 1)
- [ ] Extract configuration files to be templated (AC: 4)
  - [ ] pyproject.toml with all tool configurations
  - [ ] .pre-commit-config.yaml if created
  - [ ] Any custom nox sessions
  - [ ] GitHub workflows/actions if created

### Phase 2: Cookiecutter Creation (AC: 1, 4, 5)
- [ ] Create cookiecutter.json with template variables (AC: 1)
  - [ ] project_name, project_slug, author_name, author_email
  - [ ] python_version_min, bmad_version
  - [ ] Optional toggles: use_streamlit, use_xgboost, etc.
- [ ] Create {{cookiecutter.project_slug}} template directory structure (AC: 1, 4)
  - [ ] src/{{cookiecutter.project_slug}}/ with __init__.py, py.typed
  - [ ] tests/ with conftest.py and marker structure
  - [ ] docs/ with STYLE_GUIDE.md, TESTING_STRATEGY.md templates
  - [ ] _bmad/ directory structure for BMAD integration
  - [ ] .github/ with PR template
- [ ] Parameterize pyproject.toml (AC: 4)
  - [ ] Replace project name, version, author with Jinja2 variables
  - [ ] Include all Poetry dependencies from NCAA_eval
  - [ ] Include all tool configurations (mypy, ruff, pytest, coverage, commitizen)
- [ ] Create test structure templates (AC: 5)
  - [ ] tests/unit/ skeleton with example test
  - [ ] tests/integration/ skeleton with example test
  - [ ] conftest.py with basic fixtures
  - [ ] Example property-based test using Hypothesis

### Phase 3: Cruft Configuration (AC: 2)
- [ ] Create .cruft.json for generated projects (AC: 2)
  - [ ] Specify template URL and version tracking
  - [ ] Define skip patterns for files that should never update
  - [ ] Document update conflict resolution strategy
- [ ] Test cruft update workflow (AC: 2)
  - [ ] Generate test project from template
  - [ ] Modify template and version it
  - [ ] Run `cruft update` on test project
  - [ ] Verify non-conflicting changes merge cleanly
  - [ ] Verify conflicts are identified correctly
- [ ] Document cruft usage in template README (AC: 2)

### Phase 4: BMAD Integration (AC: 3)
- [ ] Document BMAD version compatibility (AC: 3)
  - [ ] Specify minimum/maximum BMAD versions tested
  - [ ] Document breaking changes between versions
- [ ] Preserve BMAD customizations (AC: 3)
  - [ ] Include modified agents in template with documentation
  - [ ] Mark which modifications should be preserved vs. reset
  - [ ] Create hooks for BMAD version updates
- [ ] Create BMAD update procedure documentation (AC: 3)
  - [ ] Step-by-step guide for updating BMAD in generated project
  - [ ] How to preserve custom agent modifications
  - [ ] How to merge new BMAD features

### Phase 5: Documentation (AC: 6)
- [ ] Write comprehensive template README (AC: 6)
  - [ ] Installation: `cookiecutter <template-repo>`
  - [ ] Template variable descriptions
  - [ ] Project structure overview
  - [ ] Customization points and how to modify
  - [ ] Cruft update procedures
  - [ ] BMAD integration and update instructions
- [ ] Include STYLE_GUIDE.md in template (AC: 6)
  - [ ] Copy from NCAA_eval with generic project references
- [ ] Include TESTING_STRATEGY.md in template (AC: 6)
  - [ ] Copy from NCAA_eval with generic project references
- [ ] Create TEMPLATE_USAGE.md guide (AC: 6)
  - [ ] Quick start guide
  - [ ] Common customizations
  - [ ] Troubleshooting guide
  - [ ] FAQ

### Phase 6: Validation (AC: 1-6)
- [ ] Generate test project from template (AC: 1)
  - [ ] Verify directory structure matches NCAA_eval patterns
  - [ ] Verify all parameterized values are correctly substituted
- [ ] Verify tooling works in generated project (AC: 4, 5)
  - [ ] Run `poetry install` successfully
  - [ ] Run `ruff check .` passes on clean project
  - [ ] Run `mypy` passes with strict mode
  - [ ] Run `pytest -m smoke` passes (if smoke tests included)
  - [ ] Run `pre-commit install` and hooks work
- [ ] Test cruft update flow (AC: 2)
  - [ ] Make template change, test cruft update
- [ ] Test BMAD initialization in generated project (AC: 3)
  - [ ] Verify BMAD agents work
  - [ ] Verify custom modifications preserved
- [ ] Validate documentation completeness (AC: 6)
  - [ ] README is clear and comprehensive
  - [ ] All links work
  - [ ] Customization instructions are accurate

### Phase 7: Release
- [ ] Create template repository (GitHub/GitLab)
- [ ] Tag initial release (v1.0.0)
- [ ] Write release notes documenting template features
- [ ] Update NCAA_eval template-requirements.md with template repo link

---

## Dev Notes

### Context: Why This Story Exists

This story exists because the NCAA_eval project established a robust Python development workflow including:
- Strict type checking (mypy --strict)
- Comprehensive linting (Ruff with custom rules)
- 4-tier testing strategy (pre-commit → PR/CI → AI review → owner review)
- Property-based testing with Hypothesis
- Mutation testing for critical modules
- Poetry for dependency management
- BMAD integration for AI-assisted development

Rather than re-learning and re-implementing these decisions for every new project, this template captures all learnings so future projects start with these proven patterns.

### Key Technical Requirements

**1. Dev Stack (from pyproject.toml)**
- **Language:** Python 3.12+ (supports modern syntax: match, type statement, X | None)
- **Package Manager:** Poetry 1.x
- **Project Layout:** src layout (src/{{project_slug}}/)
- **Type Checking:** mypy with strict = true
- **Linting:** Ruff with extended rules (I, UP, PT, TID25)
- **Formatting:** Ruff formatter with line-length = 110
- **Testing:** pytest + hypothesis + mutmut + pytest-cov
- **Dev Tools:** pre-commit, nox, commitizen, check-manifest, edgetest
- **Docs:** Sphinx with Furo theme

**2. Quality Standards (from STYLE_GUIDE.md)**
- Google-style docstrings (convention = "google" in pyproject.toml)
- Mandatory `from __future__ import annotations` in every file (isort config)
- Naming: snake_case (functions/vars), PascalCase (classes), UPPER_SNAKE_CASE (constants)
- Vectorization-first: REJECT PRs with for loops over DataFrames for calculations
- Line length: 110 characters
- Conventional commits enforced by commitizen

**3. Testing Strategy (from TESTING_STRATEGY.md)**
- **4-tier execution model:**
  - Tier 1 (Pre-commit < 10s): Lint, format, type-check, smoke tests, check-manifest
  - Tier 2 (PR/CI): Full test suite, integration, property-based, performance, coverage, mutation
  - Tier 3: AI code review
  - Tier 4: Owner review
- **Test markers:** smoke, slow, integration, property, performance, regression, mutation
- **Coverage targets:** Overall 80% line, 75% branch (varies by module criticality)
- **Mutation testing:** Critical modules only (e.g., evaluation/metrics.py: 95% line, 90% branch)

**4. BMAD Integration**
- Config structure: _bmad/bmm/config.yaml with user_name, communication_language, output paths
- Agent modifications: Document any customizations made during NCAA_eval
- Workflow customizations: Document any workflow.yaml changes
- Update strategy: Hooks for updating BMAD without breaking customizations

### Architecture Compliance

**Project Structure (Must Match NCAA_eval Pattern):**
```
{{cookiecutter.project_slug}}/
├── _bmad/                         # BMAD framework integration
│   ├── bmm/
│   │   ├── config.yaml           # Project config (parameterized)
│   │   ├── agents/               # Custom agent modifications
│   │   └── workflows/            # Custom workflow modifications
│   └── core/                     # BMAD core (managed by BMAD updates)
├── _bmad-output/                 # Generated artifacts (git-ignored)
│   ├── planning-artifacts/
│   └── implementation-artifacts/
├── src/
│   └── {{cookiecutter.project_slug}}/
│       ├── __init__.py           # Package root with re-exports
│       ├── py.typed              # PEP 561 marker
│       └── [domain modules]/    # Project-specific structure
├── tests/
│   ├── __init__.py
│   ├── conftest.py               # Shared fixtures
│   ├── unit/
│   └── integration/
├── docs/
│   ├── STYLE_GUIDE.md            # Copied from NCAA_eval template
│   ├── TESTING_STRATEGY.md       # Copied from NCAA_eval template
│   └── testing/                  # Detailed testing guides
├── .github/
│   └── pull_request_template.md  # Quality gates checklist
├── pyproject.toml                # Single source of truth for config
├── README.md                     # Template usage guide
└── [other standard files]        # .gitignore, LICENSE, etc.
```

### Library & Framework Requirements

**Core Dependencies (Poetry):**
```toml
[tool.poetry.dependencies]
python = ">=3.12,<4.0"
# Additional dependencies parameterized based on template use case
# e.g., pandas, numpy, xgboost for data science projects

[tool.poetry.group.dev.dependencies]
pytest = "*"
hypothesis = "*"
mutmut = "*"
ruff = "*"
mypy = "*"
pre-commit = "*"
nox = "*"
commitizen = "*"
check-manifest = "*"
edgetest = "*"
sphinx = "*"
furo = "*"
```

**Tool Configurations (All in pyproject.toml):**
- mypy: strict = true, files = ["src/{{project_slug}}", "tests"]
- ruff: line-length = 110, extend-select = ["I", "UP", "PT", "TID25"]
- ruff.lint.isort: required-imports = ["from __future__ import annotations"]
- pytest: minversion = "8.0.0", testpaths = ["tests"]
- coverage: show_missing = true, omit test directories
- commitizen: conventional commits format

### File Structure Requirements

**Files That Must Be Templated:**
1. **pyproject.toml** - All tool configs with project name/author parameterized
2. **src/{{project_slug}}/__init__.py** - Package initialization
3. **src/{{project_slug}}/py.typed** - PEP 561 marker (empty file)
4. **tests/conftest.py** - Basic pytest fixtures
5. **docs/STYLE_GUIDE.md** - Copy from NCAA_eval with generic references
6. **docs/TESTING_STRATEGY.md** - Copy from NCAA_eval with generic references
7. **.github/pull_request_template.md** - Quality gates checklist
8. **README.md** - Template-specific README with usage instructions
9. **_bmad/bmm/config.yaml** - BMAD config with parameterized values
10. **.cruft.json** - Cruft configuration for template tracking

**Files That Should Never Update via Cruft:**
- User-modified agent files in _bmad/bmm/agents/
- Project-specific domain code in src/
- Project-specific tests
- Project-specific documentation beyond style/testing guides

### Testing Requirements

**Test Structure Template:**
```
tests/
├── conftest.py                    # Shared fixtures
├── unit/
│   └── test_example.py           # Example unit test with markers
└── integration/
    └── test_example_integration.py  # Example integration test
```

**Example Smoke Test (Include in Template):**
```python
from __future__ import annotations

import pytest

@pytest.mark.smoke
def test_package_imports():
    """Smoke test: verify package imports successfully."""
    import {{cookiecutter.project_slug}}
    assert {{cookiecutter.project_slug}}.__name__ == "{{cookiecutter.project_slug}}"
```

**Example Property-Based Test (Include in Template):**
```python
from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

@pytest.mark.property
@given(st.integers())
def test_example_property(value: int):
    """Example property-based test using Hypothesis."""
    # Property: some invariant that should always hold
    assert isinstance(value, int)
```

**Pre-commit Hook Configuration:**
Must run Tier 1 checks (< 10s total):
- ruff check .
- ruff format --check .
- mypy
- pytest -m smoke
- check-manifest

### Previous Story Intelligence

**N/A** - This is the first story in Epic 8 (Project Templates & Maintenance).

However, this story builds on ALL previous work in Epic 1 (stories 1-1 through 1-3):
- **1-1:** Established Poetry, src layout, pyproject.toml structure
- **1-2:** Defined STYLE_GUIDE.md with vectorization rule, mypy strict, Ruff config
- **1-3:** Defined TESTING_STRATEGY.md with 4-tier model, Hypothesis, mutation testing

These stories inform the template structure and configuration defaults.

### Git Intelligence Summary

Recent commits establish the foundation this template will preserve:
- **b93bad1:** docs: define code quality standards and style guide
- **3598700:** feat: initialize Poetry project with src layout and strict type checking
- **aad16fb:** initialize sprint

These commits show the progression from empty repository → Poetry setup → quality standards → sprint planning.

The template must capture this end-state configuration so future projects start at the "ready to implement features" stage, not the "configure toolchain" stage.

### Latest Technical Information

**Key Technology Versions (as of Feb 2026):**

1. **Python 3.12+**
   - Latest stable: 3.12.x
   - Required for: match statements, type statement, improved typing features
   - Template should parameterize minimum version (default: 3.12, allow override)

2. **Poetry**
   - Latest stable: 1.8.x
   - Template uses Poetry for dependency management
   - Consider adding poetry.lock to template or documenting when to commit it

3. **Ruff**
   - Rapidly evolving linter/formatter
   - Template should use `ruff = "*"` (latest) unless specific version needed
   - Breaking changes rare due to opt-in rule system

4. **Mypy**
   - Strict mode stable across recent versions
   - Template uses `mypy = "*"` for latest type system improvements

5. **Pytest**
   - minversion = "8.0.0" in pyproject.toml
   - Hypothesis and mutation testing plugins stable

6. **Cruft**
   - Current version: Check latest stable release
   - Ensure template documents minimum Cruft version required
   - Syntax: `cruft create <template-url>` and `cruft update`

7. **Cookiecutter**
   - Current version: Check latest stable release
   - Jinja2 templating syntax for parameterization

**Best Practices to Include:**
- Use `tool.poetry.dependencies` with `*` for flexibility, or pin majors only
- Lock files (poetry.lock) should be committed for applications, optional for libraries
- Pre-commit hooks should be fast (< 10s) to avoid developer friction
- Documentation should be in project (docs/) not external wiki

### Project Context Reference

**Primary Context Documents:**
- [Template Requirements](_bmad-output/planning-artifacts/template-requirements.md) - Living document tracking all learnings
- [pyproject.toml](pyproject.toml) - Complete tool configuration reference
- [STYLE_GUIDE.md](docs/STYLE_GUIDE.md) - Coding standards to preserve in template
- [TESTING_STRATEGY.md](docs/TESTING_STRATEGY.md) - Testing approach to preserve in template

**Related Epic Context:**
- Epic 1 (Stories 1-1 through 1-3) established the foundation this template preserves

### References

**Configuration Sources:**
- [Source: pyproject.toml] - All tool configurations, dependencies, settings
- [Source: docs/STYLE_GUIDE.md] - Coding conventions, vectorization rule, naming standards
- [Source: docs/TESTING_STRATEGY.md] - 4-tier testing model, markers, coverage targets

**Template Structure References:**
- [Source: _bmad-output/planning-artifacts/template-requirements.md] - Complete requirements and implementation checklist
- [Source: _bmad/bmm/config.yaml] - BMAD configuration structure to preserve

**Key Design Decisions:**
- **Src Layout:** [Source: pyproject.toml] packages = [{include = "ncaa_eval", from = "src"}]
- **Strict Typing:** [Source: pyproject.toml] [tool.mypy] strict = true
- **110 Line Length:** [Source: pyproject.toml] [tool.ruff] line-length = 110
- **Required Future Import:** [Source: pyproject.toml] required-imports = ["from __future__ import annotations"]
- **Vectorization Rule:** [Source: docs/STYLE_GUIDE.md#5-vectorization-first] Reject PRs with DataFrame loops

---

## Dev Agent Record

### Agent Model Used

(To be filled by dev agent during implementation)

### Implementation Notes

**Key Implementation Decisions:**

1. **Cookiecutter vs Copier:**
   - Choose Cookiecutter for maturity and ecosystem
   - Cruft provides update functionality on top of Cookiecutter
   - Alternative: Copier has built-in updates but smaller ecosystem

2. **Parameterization Strategy:**
   - Parameterize: project name, author, Python version, optional features
   - Do NOT parameterize: tool configurations (preserve NCAA_eval decisions)
   - Conditional features: Streamlit, XGBoost, NetworkX (project-specific deps)

3. **BMAD Integration Approach:**
   - Include full _bmad directory structure in template
   - Document which agent modifications are template defaults vs. user customizations
   - Provide separate BMAD update documentation (independent of cruft updates)

4. **Cruft Update Strategy:**
   - Skip patterns: src/, tests/ (user code), custom agents
   - Auto-update: pyproject.toml (merge), docs/STYLE_GUIDE.md, docs/TESTING_STRATEGY.md
   - Conflict resolution: Document manual merge procedures in README

5. **Validation Approach:**
   - Generate test project and run full Tier 1 + Tier 2 checks
   - Verify generated project passes: ruff, mypy, pytest -m smoke
   - Test cruft update by making template change and applying to test project

### Debug Log References

(To be filled by dev agent during implementation)

### Completion Notes List

(To be filled by dev agent during implementation)

### File List

(To be filled by dev agent - all files created/modified during implementation)

---

**Story Ready for Development**

This story has been prepared with comprehensive context including:
- ✅ Complete dev stack configuration (Python 3.12+, Poetry, mypy strict, Ruff)
- ✅ Quality standards (STYLE_GUIDE.md, vectorization rule, 110 line length)
- ✅ Testing strategy (4-tier model, Hypothesis, mutation testing)
- ✅ BMAD integration requirements
- ✅ Cruft update strategy
- ✅ Clear acceptance criteria and task breakdown
- ✅ Architecture compliance guidelines
- ✅ Validation procedures

**Next Steps After Story Completion:**
1. Use template to generate a test project
2. Verify all tooling works correctly
3. Document any issues found in template-requirements.md
4. Update template based on feedback
5. Tag official release (v1.0.0)
