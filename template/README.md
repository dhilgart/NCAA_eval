# bmad-python-template

A production-ready Python project template with strict type checking, comprehensive testing, and optional BMAD AI development framework integration.

Derived from the [NCAA_eval](https://github.com/dhilgart/NCAA_eval) project's proven development workflow.

## Features

- **Python 3.12+** with modern syntax support (match, type statement, `X | None`)
- **Poetry** for dependency management (src layout)
- **mypy --strict** type checking on all source and test files
- **Ruff** linting and formatting (110-char line length, Google docstrings)
- **pytest** with Hypothesis (property-based) and mutmut (mutation testing)
- **Nox** session management (lint -> typecheck -> tests pipeline)
- **pre-commit** hooks for fast feedback (< 10s)
- **Commitizen** for conventional commits
- **Sphinx + Furo** documentation with myst-parser for Markdown
- **GitHub Actions** CI/CD (PR checks + auto version bump)
- **Cruft** support for template updates
- **BMAD** AI development framework integration (optional)

## Quick Start

### Using Cookiecutter

```sh
pip install cookiecutter
cookiecutter /path/to/template
# or from a git repo:
# cookiecutter gh:your-username/bmad-python-template
```

### Using Cruft (recommended -- enables template updates)

```sh
pip install cruft
cruft create /path/to/template
# or from a git repo:
# cruft create gh:your-username/bmad-python-template
```

### Template Variables

| Variable | Default | Description |
|---|---|---|
| `project_name` | `My Project` | Human-readable project name |
| `project_slug` | (auto) | Python package name (auto-generated from project_name) |
| `project_description` | | Short project description |
| `author_name` | | Your full name |
| `author_email` | | Your email address |
| `github_username` | | Your GitHub username |
| `python_version_min` | `3.12` | Minimum Python version |
| `open_source_license` | `MIT` | License type (MIT, GPLv3, Apache 2.0, or None) |
| `use_bmad` | `y` | Include BMAD AI dev framework integration |
| `bmad_user_name` | (auto) | BMAD display name (defaults to author_name) |

## After Generation

```sh
cd your-project-slug

# Set up development environment
conda create -n your_project python=3.12 -y
conda activate your_project
pip install poetry
POETRY_VIRTUALENVS_CREATE=false poetry install --with dev
pre-commit install

# Verify everything works
nox
```

## Updating Generated Projects

If you created your project with `cruft create`:

```sh
# Check if template has updates
cruft check

# Apply template updates
cruft update
```

### Files that auto-update via cruft

- `pyproject.toml` (tool configurations)
- `.pre-commit-config.yaml`
- `.github/workflows/`
- `docs/STYLE_GUIDE.md`, `docs/TESTING_STRATEGY.md`
- `noxfile.py`

### Files that are never overwritten

- `src/` (your source code)
- `tests/` (your tests)
- `_bmad/bmm/` (your BMAD customizations)
- `README.md` (your project README)

## Template Structure

```
template/
├── cookiecutter.json                    # Template variables
├── hooks/
│   └── post_gen_project.py             # Post-generation setup
├── {{cookiecutter.project_slug}}/      # Generated project root
│   ├── src/{{cookiecutter.project_slug}}/
│   ├── tests/
│   ├── docs/
│   ├── .github/workflows/
│   ├── pyproject.toml
│   ├── noxfile.py
│   ├── .pre-commit-config.yaml
│   └── ...
└── README.md                           # This file
```

## Design Decisions

This template encodes the following opinions (all configurable):

1. **src layout** over flat layout (cleaner imports, PEP 561 compliance)
2. **Poetry** over pip/setuptools (lockfile, groups, single pyproject.toml)
3. **mypy --strict** from day one (easier than retrofitting later)
4. **Ruff** for both linting and formatting (replaces black, isort, flake8)
5. **Google-style docstrings** (concise, works with napoleon)
6. **4-tier quality gates** (pre-commit -> CI -> AI review -> owner review)
7. **Property-based testing** with Hypothesis (catch edge cases automatically)
8. **Mutation testing** with mutmut (verify test quality)
9. **Conventional commits** (enables automated changelog + version bumps)

## Requirements

- Python 3.12+
- [conda](https://docs.conda.io/) (recommended for environment management)
- [Poetry](https://python-poetry.org/)
- [cookiecutter](https://cookiecutter.readthedocs.io/) or [cruft](https://cruft.github.io/cruft/)

## License

This template is available under the MIT License.
