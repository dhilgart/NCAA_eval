# Cookie-Cutter Template Requirements

**Purpose:** Capture all learnings and decisions from NCAA_eval to create a reusable project template with Cruft support and BMAD integration.

**Status:** Living Document - Update throughout implementation
**Target:** Post-project completion - see related backlog story

---

## 1. Dev Stack & Architecture

### Core Technologies
- **Language/Runtime:** [Update as decisions solidify]
- **Package Manager:** Poetry (confirmed - see pyproject.toml)
- **Python Version:** [Document chosen version and rationale]
- **Project Layout:** src layout (confirmed - see project structure)

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

### Configuration Management
- Type checking: mypy with strict settings (see pyproject.toml)
- Linting: [Document chosen tools]
- Formatting: [Document chosen tools]

**Template Action Items:**
- [ ] Export final pyproject.toml as template base
- [ ] Document rationale for each major dependency
- [ ] Create default configurations for all tooling

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
- [Capture anti-patterns and pitfalls]

### Would Do Differently

**Story 1.3 - Testing Strategy (2026-02-16):**
- ⚠️ **Missing dependency caught in review** - pytest-cov was referenced extensively but not added to pyproject.toml until code review. Template should include essential test dependencies upfront.
- ⚠️ **Timing inconsistency across docs** - pyproject.toml said "< 5 seconds" while main doc said "< 10 seconds". Template should establish single source of truth for constraints.

### Process Improvements
- [Capture workflow or process refinements]

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

*Last Updated: 2026-02-16 (Story 1.3 - Testing Strategy learnings captured)*
*Next Review: [Set cadence - weekly? sprint boundaries?]*
