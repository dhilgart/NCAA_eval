# Claude Code -- Workspace Instructions

## Bash Rules (Mandatory)

- **Never chain bash commands with `&&`, `|`, or `;`** -- always issue commands as separate tool calls.
- **Never use heredocs or `printf ... > file` for commit messages** -- use the `Write` tool to create `/tmp/commit_msg.txt`, then `git commit -F /tmp/commit_msg.txt`.

## Project Conventions

- `from __future__ import annotations` required in all Python files (enforced by Ruff)
- `mypy --strict` mandatory for all files in `src/` and `tests/`
- Conventional commits: `type(scope): description`
- Line length: 110 characters
- Google-style docstrings

## Running Quality Gates

```bash
nox                # Full pipeline: lint -> typecheck -> tests
nox -s lint        # Ruff only
nox -s typecheck   # mypy only
nox -s tests       # pytest only
nox -s docs        # Build Sphinx documentation
```

## Running Tests

```bash
pytest                                    # All tests
pytest -m smoke                           # Fast smoke tests only (< 10s)
pytest -m "not slow"                      # Skip slow tests
pytest --cov=src/{{ cookiecutter.project_slug }} --cov-report=term-missing  # With coverage
```

## Environment Setup

```bash
conda create -n {{ cookiecutter.project_slug }} python={{ cookiecutter.python_version_min }} -y
conda activate {{ cookiecutter.project_slug }}
pip install poetry
POETRY_VIRTUALENVS_CREATE=false poetry install --with dev
pre-commit install
```
