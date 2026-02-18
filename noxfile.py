"""Nox session management for the NCAA_eval quality pipeline.

Running ``nox`` executes Ruff (lint/format) -> Mypy (type check) -> Pytest (tests).
Individual sessions can be invoked with ``nox -s <session>``.
"""

from __future__ import annotations

import nox

# Default sessions and execution order (Architecture Section 10.2)
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
        "src/ncaa_eval",
        "tests",
    )


@nox.session(python=False)
def tests(session: nox.Session) -> None:
    """Run the full pytest test suite."""
    session.run("pytest", "--tb=short")
