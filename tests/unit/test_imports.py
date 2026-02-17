"""Smoke tests for package imports.

These tests run in pre-commit hooks to catch import errors quickly.
"""

from __future__ import annotations

import pytest


@pytest.mark.smoke
def test_can_import_ncaa_eval() -> None:
    """Verify package is importable without errors.

    This smoke test catches:
    - Import errors from syntax issues
    - Circular import problems
    - Missing dependencies
    - Module initialization failures
    """
    import ncaa_eval

    assert ncaa_eval is not None
