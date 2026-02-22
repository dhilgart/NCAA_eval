# {{ cookiecutter.project_name }}

{{ cookiecutter.project_description }}

## Quick Start

### Prerequisites

- Python {{ cookiecutter.python_version_min }}+
- [conda](https://docs.conda.io/) (recommended) or virtualenv
- [Poetry](https://python-poetry.org/) for dependency management

### Installation

```sh
# Create conda environment
conda create -n {{ cookiecutter.project_slug }} python={{ cookiecutter.python_version_min }} -y
conda activate {{ cookiecutter.project_slug }}

# Install Poetry inside conda env
pip install poetry

# Install project dependencies
POETRY_VIRTUALENVS_CREATE=false poetry install --with dev

# Install pre-commit hooks
pre-commit install
```

### Running Quality Gates

```sh
# Full pipeline: lint -> typecheck -> tests
nox

# Individual sessions
nox -s lint       # Ruff linting + format check
nox -s typecheck  # mypy --strict
nox -s tests      # Full pytest suite
nox -s docs       # Build Sphinx documentation
```

### Running Tests

```sh
# All tests
pytest

# Smoke tests only (fast, < 10s)
pytest -m smoke

# With coverage
pytest --cov=src/{{ cookiecutter.project_slug }} --cov-report=term-missing
```

## Project Structure

```
{{ cookiecutter.project_slug }}/
├── src/
│   └── {{ cookiecutter.project_slug }}/
│       ├── __init__.py        # Package root
│       └── py.typed           # PEP 561 marker
├── tests/
│   ├── conftest.py            # Shared fixtures
│   ├── unit/                  # Unit tests
│   └── integration/           # Integration tests
├── docs/
│   ├── STYLE_GUIDE.md         # Coding standards
│   ├── TESTING_STRATEGY.md    # Testing approach
│   └── conf.py                # Sphinx configuration
├── .github/
│   └── workflows/             # CI/CD pipelines
├── pyproject.toml             # Single source of truth for config
├── noxfile.py                 # Quality pipeline sessions
└── .pre-commit-config.yaml    # Git hooks
```

## Development Workflow

1. Create a feature branch: `git checkout -b feat/my-feature`
2. Write failing tests first (red-green-refactor)
3. Implement the feature
4. Run `nox` to validate all quality gates
5. Commit with conventional format: `git commit -m "feat(scope): description"`
6. Open a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed instructions.

## Quality Standards

- **Type checking:** mypy with `--strict` mode
- **Linting:** Ruff with extended rules (110-char line length)
- **Testing:** pytest + Hypothesis (property-based) + mutmut (mutation)
- **Docs:** Google-style docstrings, Sphinx with Furo theme
- **Commits:** Conventional commits enforced by Commitizen

See [STYLE_GUIDE.md](docs/STYLE_GUIDE.md) and [TESTING_STRATEGY.md](docs/TESTING_STRATEGY.md) for details.
{% if cookiecutter.open_source_license != 'None' %}
## License

This project is licensed under the {{ cookiecutter.open_source_license }}. See [LICENSE](LICENSE) for details.
{% endif %}
