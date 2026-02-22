"""Smoke tests verifying basic package setup."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.smoke
@pytest.mark.xfail(
    raises=ImportError,
    strict=False,
    reason="Package not installed yet — run 'poetry install --with dev' first",
)
def test_package_imports() -> None:
    """Smoke test: verify package imports successfully.

    Requires the package to be installed (via `poetry install --with dev`).
    This test is expected to fail in a fresh environment before installation.
    """
    import {{ cookiecutter.project_slug }}

    assert {{ cookiecutter.project_slug }}.__name__ == "{{ cookiecutter.project_slug }}"


@pytest.mark.smoke
@pytest.mark.no_mutation
def test_src_directory_structure() -> None:
    """Verify project follows src layout conventions."""
    project_root = Path(__file__).parent.parent.parent
    src_dir = project_root / "src" / "{{ cookiecutter.project_slug }}"
    assert src_dir.is_dir(), f"Missing src directory: {src_dir}"
    assert (src_dir / "__init__.py").is_file(), "Missing __init__.py"
    assert (src_dir / "py.typed").is_file(), "Missing py.typed PEP 561 marker"
