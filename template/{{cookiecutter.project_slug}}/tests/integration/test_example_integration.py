"""Example integration test with I/O.

Demonstrates the pattern for integration tests that interact with
the file system. Replace with real integration tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.integration
def test_write_and_read_data(temp_data_dir: Path) -> None:
    """Example: verify data round-trips through the file system."""
    test_file = temp_data_dir / "sample.txt"
    test_file.write_text("hello world")
    assert test_file.read_text() == "hello world"
