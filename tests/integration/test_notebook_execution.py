"""Smoke test: ensemble tutorial notebook executes without error.

Runs the notebook via ``jupyter nbconvert --execute`` and asserts a zero
exit code.  Skipped when synced data is not present (CI environments).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
NOTEBOOK = PROJECT_ROOT / "notebooks" / "tutorials" / "03_ensemble_model.ipynb"
DATA_MARKER = PROJECT_ROOT / "data" / "MTeams.csv"


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(
    not DATA_MARKER.exists(),
    reason="Synced NCAA data not present (run `python -m ncaa_eval.cli sync` first)",
)
def test_ensemble_tutorial_notebook_executes() -> None:
    """Notebook runs end-to-end without error."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=600",
            "--output-dir",
            str(NOTEBOOK.parent),
            str(NOTEBOOK),
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0, (
        f"Notebook execution failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
