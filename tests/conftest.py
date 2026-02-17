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


@pytest.fixture
def sample_game_records() -> list[dict[str, object]]:
    """Return representative sample game records for testing.

    Returns a list of dicts representing game records. Using plain dicts
    until the Game schema is defined in Story 2.2, at which point this
    fixture will be updated to return typed objects.

    Returns:
        list[dict[str, object]]: A list of sample game record dictionaries.
    """
    return [
        {
            "season": 2023,
            "day_num": 134,
            "w_team_id": 1101,
            "l_team_id": 1102,
            "w_score": 78,
            "l_score": 65,
            "w_loc": "H",
            "num_ot": 0,
        },
        {
            "season": 2023,
            "day_num": 136,
            "w_team_id": 1103,
            "l_team_id": 1104,
            "w_score": 82,
            "l_score": 79,
            "w_loc": "N",
            "num_ot": 1,
        },
        {
            "season": 2022,
            "day_num": 132,
            "w_team_id": 1101,
            "l_team_id": 1103,
            "w_score": 91,
            "l_score": 88,
            "w_loc": "A",
            "num_ot": 2,
        },
    ]
