"""Smoke tests for package structure and basic contracts.

These tests run in pre-commit hooks to catch structural issues quickly.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import pytest


@pytest.mark.smoke
def test_package_metadata_accessible() -> None:
    """Verify package metadata is registered correctly.

    This smoke test catches:
    - Poetry build/install configuration issues
    - Missing pyproject.toml metadata
    - Package name mismatches (ncaa-eval vs ncaa_eval)
    """
    version = importlib.metadata.version("ncaa-eval")
    assert version is not None
    assert len(version) > 0


@pytest.mark.smoke
def test_src_directory_structure() -> None:
    """Verify expected src/ directory structure exists.

    This smoke test catches:
    - Missing critical subdirectories
    - Incorrect package layout
    - Build/installation problems
    """
    test_file = Path(__file__)
    project_root = test_file.parent.parent.parent

    src_dir = project_root / "src" / "ncaa_eval"
    assert src_dir.exists(), f"Package directory not found: {src_dir}"
    assert src_dir.is_dir(), f"Package path is not a directory: {src_dir}"

    init_file = src_dir / "__init__.py"
    assert init_file.exists(), f"Package __init__.py not found: {init_file}"
