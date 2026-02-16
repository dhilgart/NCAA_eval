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
  - [dependency]: [version] # [reason for inclusion]
```

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
- Framework: [Pytest/unittest/other - document choice]
- Coverage Target: [Document target %]
- Coverage Tool: [Document tool]

### Test Organization
```
tests/
  ├── unit/          # [Document structure decisions]
  ├── integration/   # [Document structure decisions]
  └── e2e/          # [Document structure decisions]
```

### Key Testing Patterns
- [Pattern 1]: [When to use, example]
- [Pattern 2]: [When to use, example]

### CI/CD Integration
- [Document test automation approach]
- [Document quality gates]

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
**Reference:** [QUALITY_GATES.md](./QUALITY_GATES.md) (if exists)

- Type checking: [Document requirements]
- Test coverage: [Document thresholds]
- Linting: [Document rules]
- Documentation: [Document requirements]

### Style Guide
**Reference:** [STYLE_GUIDE.md](./STYLE_GUIDE.md)

**Template Action Items:**
- [ ] Export final style guide
- [ ] Export quality gate configurations
- [ ] Create pre-commit hooks template

---

## 7. Lessons Learned

### What Worked Well
- [Capture positive patterns as you discover them]

### What Didn't Work
- [Capture anti-patterns and pitfalls]

### Would Do Differently
- [Capture hindsight insights]

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

*Last Updated: 2026-02-16*
*Next Review: [Set cadence - weekly? sprint boundaries?]*
