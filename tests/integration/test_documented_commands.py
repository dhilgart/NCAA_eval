"""E2E integration tests for every documented toolchain command.

These tests invoke each documented CLI command via ``subprocess.run()`` to
verify that documentation does not rot.  Every command listed in
``docs/tutorials/getting-started.md``, the README quality-gate table, and
``noxfile.py`` sessions is covered.

All tests are marked ``@pytest.mark.integration`` and ``@pytest.mark.slow``
because they spawn subprocesses.  To avoid infinite recursion (since some
tests run ``pytest`` itself), the subprocess calls use
``--ignore=tests/integration/test_documented_commands.py``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Project root — all subprocess commands run from here.
# Anchored to this file's location so the path stays valid regardless of cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Reusable ignore flag to prevent recursive test discovery.
_SELF_IGNORE = f"--ignore={PROJECT_ROOT / 'tests' / 'integration' / 'test_documented_commands.py'}"

# Regex that matches any ANSI/VT100 escape sequence.
# Used to strip color codes from subprocess output so text-content assertions
# are not broken by FORCE_COLOR=1 (set by GitHub Actions).
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _run(
    cmd: list[str],
    *,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    """Run *cmd* as a subprocess from the project root and return the result.

    Non-zero exit codes are intentionally NOT raised — callers assert
    ``result.returncode`` themselves to produce helpful failure messages.

    Args:
        cmd: Command tokens (e.g. ``["pytest", "-m", "smoke"]``).
        timeout: Maximum wall-clock seconds before the process is killed.

    Returns:
        Completed process with captured stdout/stderr.
    """
    # Build an environment that suppresses ANSI output.
    # GitHub Actions sets FORCE_COLOR=1, which causes Rich/Typer to emit ANSI
    # escape sequences even when stdout is a captured pipe.  Removing
    # FORCE_COLOR and setting NO_COLOR=1 is the belt-and-suspenders approach;
    # _ANSI_ESCAPE stripping below is the final fallback.
    env = {k: v for k, v in os.environ.items() if k not in {"FORCE_COLOR", "FORCE_COLORS"}}
    env["NO_COLOR"] = "1"
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
        env=env,
    )
    # Strip any remaining ANSI sequences so callers' text assertions are robust.
    return subprocess.CompletedProcess(
        args=result.args,
        returncode=result.returncode,
        stdout=_ANSI_ESCAPE.sub("", result.stdout),
        stderr=_ANSI_ESCAPE.sub("", result.stderr),
    )


# ---------------------------------------------------------------------------
# Quality-gate command tests (AC #2–#8)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
def test_pytest_smoke() -> None:
    """``pytest -m smoke`` exits 0 and completes in under 10 s (AC #2)."""
    start = time.monotonic()
    result = _run(["pytest", "-m", "smoke", _SELF_IGNORE], timeout=30)
    elapsed = time.monotonic() - start
    assert result.returncode == 0, (
        f"pytest -m smoke failed (rc={result.returncode}):\n{result.stdout}\n{result.stderr}"
    )
    assert elapsed < 10, f"pytest -m smoke took {elapsed:.1f}s — must complete in under 10s (AC #2)"


@pytest.mark.integration
@pytest.mark.slow
def test_pytest_full_suite() -> None:
    """``pytest`` (full suite) exits 0 (AC #3)."""
    result = _run(["pytest", _SELF_IGNORE], timeout=300)
    assert result.returncode == 0, (
        f"pytest full suite failed (rc={result.returncode}):\n{result.stdout}\n{result.stderr}"
    )


@pytest.mark.integration
@pytest.mark.slow
def test_pytest_coverage() -> None:
    """``pytest --cov`` exits 0 and stdout contains 'TOTAL' (AC #4)."""
    result = _run(
        [
            "pytest",
            "--cov=src/ncaa_eval",
            "--cov-report=term-missing",
            _SELF_IGNORE,
        ],
        timeout=300,
    )
    assert result.returncode == 0, (
        f"pytest --cov failed (rc={result.returncode}):\n{result.stdout}\n{result.stderr}"
    )
    assert "TOTAL" in result.stdout, f"Coverage output missing 'TOTAL':\n{result.stdout}"


@pytest.mark.integration
@pytest.mark.slow
def test_ruff_check() -> None:
    """``ruff check .`` exits 0 (AC #5)."""
    result = _run(["ruff", "check", "."], timeout=60)
    assert result.returncode == 0, (
        f"ruff check failed (rc={result.returncode}):\n{result.stdout}\n{result.stderr}"
    )


@pytest.mark.integration
@pytest.mark.slow
def test_ruff_format_check() -> None:
    """``ruff format --check .`` exits 0 (AC #6)."""
    result = _run(["ruff", "format", "--check", "."], timeout=60)
    assert result.returncode == 0, (
        f"ruff format --check failed (rc={result.returncode}):\n{result.stdout}\n{result.stderr}"
    )


@pytest.mark.integration
@pytest.mark.slow
def test_mypy_strict() -> None:
    """``mypy --strict src/ncaa_eval tests`` exits 0 (AC #7)."""
    result = _run(
        ["mypy", "--strict", "src/ncaa_eval", "tests"],
        timeout=120,
    )
    assert result.returncode == 0, (
        f"mypy --strict failed (rc={result.returncode}):\n{result.stdout}\n{result.stderr}"
    )


@pytest.mark.integration
@pytest.mark.slow
def test_nox_lint() -> None:
    """``nox -s lint`` exits 0 (AC #8)."""
    result = _run(["nox", "-s", "lint"], timeout=60)
    assert result.returncode == 0, (
        f"nox -s lint failed (rc={result.returncode}):\n{result.stdout}\n{result.stderr}"
    )


@pytest.mark.integration
@pytest.mark.slow
def test_nox_typecheck() -> None:
    """``nox -s typecheck`` exits 0 (AC #8)."""
    result = _run(["nox", "-s", "typecheck"], timeout=120)
    assert result.returncode == 0, (
        f"nox -s typecheck failed (rc={result.returncode}):\n{result.stdout}\n{result.stderr}"
    )


@pytest.mark.integration
@pytest.mark.slow
def test_nox_tests() -> None:
    """``nox -s tests`` exits 0 (AC #8)."""
    result = _run(["nox", "-s", "tests", "--", _SELF_IGNORE], timeout=300)
    assert result.returncode == 0, (
        f"nox -s tests failed (rc={result.returncode}):\n{result.stdout}\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# CLI help tests (AC #9–#10)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
def test_cli_help() -> None:
    """``python -m ncaa_eval.cli --help`` exits 0 and prints usage (AC #9)."""
    result = _run(
        [sys.executable, "-m", "ncaa_eval.cli", "--help"],
        timeout=30,
    )
    assert result.returncode == 0, (
        f"CLI --help failed (rc={result.returncode}):\n{result.stdout}\n{result.stderr}"
    )
    assert "Usage" in result.stdout, f"CLI --help missing 'Usage':\n{result.stdout}"


@pytest.mark.integration
@pytest.mark.slow
def test_cli_train_help() -> None:
    """``python -m ncaa_eval.cli train --help`` exits 0 and shows --model (AC #10)."""
    result = _run(
        [sys.executable, "-m", "ncaa_eval.cli", "train", "--help"],
        timeout=30,
    )
    assert result.returncode == 0, (
        f"CLI train --help failed (rc={result.returncode}):\n{result.stdout}\n{result.stderr}"
    )
    assert "--model" in result.stdout, f"CLI train --help missing '--model':\n{result.stdout}"


# ---------------------------------------------------------------------------
# check-manifest test (AC #11)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
def test_check_manifest() -> None:
    """``check-manifest`` exits 0 (AC #11)."""
    result = _run(["check-manifest"], timeout=60)
    assert result.returncode == 0, (
        f"check-manifest failed (rc={result.returncode}):\n{result.stdout}\n{result.stderr}"
    )
