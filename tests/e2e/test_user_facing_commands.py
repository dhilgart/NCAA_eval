"""E2E startup checks for all user-facing entry points.

Each test invokes a documented command via ``subprocess.run()`` with
``cwd=REPO_ROOT``, exactly as a user would from the repo root.  Tests
assert clean startup (returncode 0 and no ``ModuleNotFoundError`` /
``ImportError`` in stderr).

These are *execution-context* tests — they deliberately avoid
``importlib`` / Python ``import`` to ensure the test harness's
``sys.path`` injection does not mask real failures.

All tests are marked ``@pytest.mark.integration`` because they spawn
subprocesses and are too slow for the smoke tier.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _clean_env() -> dict[str, str]:
    """Build an environment dict without FORCE_COLOR or PYTHONPATH to avoid ANSI noise and path injection."""
    env = {k: v for k, v in os.environ.items() if k not in {"FORCE_COLOR", "FORCE_COLORS", "PYTHONPATH"}}
    env["NO_COLOR"] = "1"
    return env


def _assert_no_import_errors(result: subprocess.CompletedProcess[str]) -> None:
    """Assert that stderr contains no ModuleNotFoundError or ImportError."""
    assert "ModuleNotFoundError" not in result.stderr, f"ModuleNotFoundError in stderr:\n{result.stderr}"
    assert "ImportError" not in result.stderr, f"ImportError in stderr:\n{result.stderr}"


# ---------------------------------------------------------------------------
# sync.py
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_sync_help() -> None:
    """sync.py --help must exit 0 with no import errors."""
    result = subprocess.run(
        [sys.executable, "sync.py", "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=30,
        env=_clean_env(),
    )
    _assert_no_import_errors(result)
    assert result.returncode == 0, f"sync.py --help failed (rc={result.returncode}):\n{result.stderr}"


# ---------------------------------------------------------------------------
# ncaa_eval.cli
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_cli_help() -> None:
    """Python -m ncaa_eval.cli --help must exit 0."""
    result = subprocess.run(
        [sys.executable, "-m", "ncaa_eval.cli", "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=30,
        env=_clean_env(),
    )
    _assert_no_import_errors(result)
    assert result.returncode == 0, f"CLI --help failed (rc={result.returncode}):\n{result.stderr}"


@pytest.mark.integration
def test_cli_train_help() -> None:
    """Python -m ncaa_eval.cli train --help must exit 0."""
    result = subprocess.run(
        [sys.executable, "-m", "ncaa_eval.cli", "train", "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=30,
        env=_clean_env(),
    )
    _assert_no_import_errors(result)
    assert result.returncode == 0, f"CLI train --help failed (rc={result.returncode}):\n{result.stderr}"


@pytest.mark.integration
def test_cli_predict_help() -> None:
    """Python -m ncaa_eval.cli predict --help must exit 0."""
    result = subprocess.run(
        [sys.executable, "-m", "ncaa_eval.cli", "predict", "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=30,
        env=_clean_env(),
    )
    _assert_no_import_errors(result)
    assert result.returncode == 0, f"CLI predict --help failed (rc={result.returncode}):\n{result.stderr}"


# ---------------------------------------------------------------------------
# Model listing (documented in getting-started.md and user-guide.md)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_list_models() -> None:
    """Python -c 'from ncaa_eval.model import list_models; print(list_models())'."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from ncaa_eval.model import list_models; print(list_models())",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=30,
        env=_clean_env(),
    )
    _assert_no_import_errors(result)
    assert result.returncode == 0, f"list_models() failed (rc={result.returncode}):\n{result.stderr}"
    assert "elo" in result.stdout, f"'elo' not in list_models() output:\n{result.stdout}"


# ---------------------------------------------------------------------------
# Streamlit dashboard startup
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
def test_streamlit_startup() -> None:
    """Streamlit run dashboard/app.py must start without import errors.

    Launches Streamlit in headless mode on a non-standard port, waits a
    few seconds, then checks that stderr contains no import errors — whether
    the process exited early (crash) or is still running (successful startup).
    """
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "dashboard/app.py",
            "--server.headless",
            "true",
            "--server.port",
            "18501",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_clean_env(),
    )
    poll_result: int | None = None
    try:
        # Give Streamlit time to start (or crash).
        time.sleep(8)
        poll_result = proc.poll()
    finally:
        proc.terminate()

    try:
        _, stderr = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        pytest.fail(
            "Streamlit process did not terminate cleanly within 5 s after SIGTERM — stderr unavailable"
        )

    # Check stderr regardless of whether the process exited or was terminated —
    # a running process that logged import errors would otherwise pass silently.
    assert "ModuleNotFoundError" not in stderr, f"ModuleNotFoundError on streamlit startup:\n{stderr}"
    assert "ImportError" not in stderr, f"ImportError on streamlit startup:\n{stderr}"
    if poll_result is not None:
        assert poll_result == 0, f"Streamlit exited with code {poll_result}:\n{stderr}"
