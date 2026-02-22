"""Nox session management for the {{ cookiecutter.project_slug }} quality pipeline.

Running `nox` executes Ruff (lint/format) -> Mypy (type check) -> Pytest (tests).
Individual sessions can be invoked with `nox -s <session>`.
"""

from __future__ import annotations

import nox

# Default sessions and execution order
nox.options.sessions = ["lint", "typecheck", "tests"]


@nox.session(python=False)
def lint(session: nox.Session) -> None:
    """Run Ruff linting with auto-fix and format checking."""
    session.run("ruff", "check", ".", "--fix")
    session.run("ruff", "format", "--check", ".")


@nox.session(python=False)
def typecheck(session: nox.Session) -> None:
    """Run mypy strict type checking on source and test files."""
    session.run(
        "mypy",
        "--strict",
        "--show-error-codes",
        "--namespace-packages",
        "src/{{ cookiecutter.project_slug }}",
        "tests",
        "noxfile.py",
    )


@nox.session(python=False)
def tests(session: nox.Session) -> None:
    """Run the full pytest test suite."""
    session.run("pytest", "--tb=short")


@nox.session(python=False)
def docs(session: nox.Session) -> None:
    """Generate Sphinx HTML documentation from source docstrings."""
    session.run("sphinx-apidoc", "-f", "-e", "-o", "docs/api", "src/{{ cookiecutter.project_slug }}")
    session.run("sphinx-build", "-b", "html", "docs", "docs/_build/html")
