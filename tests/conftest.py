"""Shared pytest fixtures for the ncaa_eval test suite.

Fixtures defined here are available to all tests without explicit imports.
See docs/testing/conventions.md for fixture naming conventions.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Provide an isolated temporary directory for test data.

    Uses pytest's built-in tmp_path fixture which handles cleanup
    automatically (no manual teardown needed). Windows-safe: avoids
    manual shutil.rmtree which can fail if file handles are not released.

    Args:
        tmp_path: pytest built-in temporary directory fixture.

    Returns:
        Path: A temporary directory that exists for the duration of the test.
    """
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir
