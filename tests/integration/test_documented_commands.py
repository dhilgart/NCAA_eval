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

import datetime
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


# ---------------------------------------------------------------------------
# CLI train execution fixture and tests (AC #14 — tutorial command validation)
# ---------------------------------------------------------------------------


@pytest.fixture
def train_data_dir(tmp_path: Path) -> Path:
    """Minimal Parquet data directory for CLI train execution tests.

    Creates teams, seasons, and 24 regular-season games per season (2023 and
    2024) so that ``python -m ncaa_eval.cli train`` can run end-to-end without
    requiring a real data download.

    Returns:
        Path to the populated data directory.
    """
    from ncaa_eval.ingest.repository import ParquetRepository
    from ncaa_eval.ingest.schema import Game, Season, Team

    data_dir = tmp_path / "train_data"
    repo = ParquetRepository(base_path=data_dir)

    repo.save_teams(
        [
            Team(team_id=101, team_name="Team A", canonical_name="team_a"),
            Team(team_id=102, team_name="Team B", canonical_name="team_b"),
            Team(team_id=103, team_name="Team C", canonical_name="team_c"),
            Team(team_id=104, team_name="Team D", canonical_name="team_d"),
        ]
    )
    repo.save_seasons([Season(year=2023), Season(year=2024)])

    # 4 teams → 6 unique pairs × 4 rounds = 24 regular-season games per season.
    # Enough data for SRS batch-rating computation and stateless model training.
    team_ids = [101, 102, 103, 104]
    pairs = [(w, l_id) for i, w in enumerate(team_ids) for l_id in team_ids[i + 1 :]]
    all_games: list[Game] = []
    for season_year in (2023, 2024):
        for rnd, (w_id, l_id) in enumerate(pairs * 4):
            day = 10 + rnd * 4
            all_games.append(
                Game(
                    game_id=f"g{season_year}_{rnd + 1:03d}",
                    season=season_year,
                    day_num=day,
                    date=datetime.date(season_year - 1, 11, 1) + datetime.timedelta(days=day),
                    w_team_id=w_id,
                    l_team_id=l_id,
                    w_score=75,
                    l_score=60,
                    loc="N",
                )
            )
    repo.save_games(all_games)
    return data_dir


@pytest.mark.integration
@pytest.mark.slow
def test_cli_train_elo(train_data_dir: Path, tmp_path: Path) -> None:
    """``python -m ncaa_eval.cli train --model elo`` exits 0 (AC #14)."""
    result = _run(
        [
            sys.executable,
            "-m",
            "ncaa_eval.cli",
            "train",
            "--model",
            "elo",
            "--start-year",
            "2023",
            "--end-year",
            "2024",
            "--data-dir",
            str(train_data_dir),
            "--output-dir",
            str(tmp_path / "output"),
        ],
        timeout=120,
    )
    assert result.returncode == 0, (
        f"CLI train --model elo failed (rc={result.returncode}):\n{result.stdout}\n{result.stderr}"
    )


@pytest.mark.integration
@pytest.mark.slow
def test_cli_train_xgboost(train_data_dir: Path, tmp_path: Path) -> None:
    """``python -m ncaa_eval.cli train --model xgboost`` exits 0 (AC #14)."""
    result = _run(
        [
            sys.executable,
            "-m",
            "ncaa_eval.cli",
            "train",
            "--model",
            "xgboost",
            "--start-year",
            "2023",
            "--end-year",
            "2024",
            "--data-dir",
            str(train_data_dir),
            "--output-dir",
            str(tmp_path / "output"),
        ],
        timeout=120,
    )
    assert result.returncode == 0, (
        f"CLI train --model xgboost failed (rc={result.returncode}):\n{result.stdout}\n{result.stderr}"
    )


@pytest.mark.integration
@pytest.mark.slow
def test_cli_train_logistic_regression(train_data_dir: Path, tmp_path: Path) -> None:
    """``python -m ncaa_eval.cli train --model logistic_regression`` exits 0 (AC #14)."""
    result = _run(
        [
            sys.executable,
            "-m",
            "ncaa_eval.cli",
            "train",
            "--model",
            "logistic_regression",
            "--start-year",
            "2023",
            "--end-year",
            "2024",
            "--data-dir",
            str(train_data_dir),
            "--output-dir",
            str(tmp_path / "output"),
        ],
        timeout=120,
    )
    assert result.returncode == 0, (
        f"CLI train --model logistic_regression failed "
        f"(rc={result.returncode}):\n{result.stdout}\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# predict / export — shared fixtures and tests
#
# StackedEnsemble cannot be trained via ``cli train --model ensemble``
# (the registry entry is a sentinel placeholder). All three model types are
# trained programmatically so the same Parquet data directory doubles as the
# RunStore root — which is what the documented ``cli predict / export``
# workflow requires (``--data-dir`` serves both roles).
# ---------------------------------------------------------------------------


def _populate_data_dir(data_dir: Path) -> None:
    """Write minimal Parquet data (4 teams × 2 seasons × 24 games) to *data_dir*."""
    from ncaa_eval.ingest.repository import ParquetRepository
    from ncaa_eval.ingest.schema import Game, Season, Team

    repo = ParquetRepository(base_path=data_dir)
    repo.save_teams(
        [
            Team(team_id=101, team_name="Team A", canonical_name="team_a"),
            Team(team_id=102, team_name="Team B", canonical_name="team_b"),
            Team(team_id=103, team_name="Team C", canonical_name="team_c"),
            Team(team_id=104, team_name="Team D", canonical_name="team_d"),
        ]
    )
    repo.save_seasons([Season(year=2023), Season(year=2024)])
    team_ids = [101, 102, 103, 104]
    pairs = [(w, l_id) for i, w in enumerate(team_ids) for l_id in team_ids[i + 1 :]]
    all_games: list[Game] = []
    for season_year in (2023, 2024):
        for rnd, (w_id, l_id) in enumerate(pairs * 4):
            day = 10 + rnd * 4
            all_games.append(
                Game(
                    game_id=f"g{season_year}_{rnd + 1:03d}",
                    season=season_year,
                    day_num=day,
                    date=datetime.date(season_year - 1, 11, 1) + datetime.timedelta(days=day),
                    w_team_id=w_id,
                    l_team_id=l_id,
                    w_score=75,
                    l_score=60,
                    loc="N",
                )
            )
    repo.save_games(all_games)


@pytest.fixture(scope="module")
def elo_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, Path]:
    """Train an Elo model once for all predict/export tests.

    ``data_dir`` and ``output_dir`` are the same directory so that
    ``cli predict/export --data-dir <dir>`` can locate both Parquet data
    and RunStore artifacts — matching the documented user workflow.

    Returns:
        (run_id, data_dir)
    """
    from rich.console import Console

    from ncaa_eval.cli.train import run_training
    from ncaa_eval.model import get_model

    data_dir = tmp_path_factory.mktemp("elo_run")
    _populate_data_dir(data_dir)
    model = get_model("elo")()
    run = run_training(
        model,
        start_year=2023,
        end_year=2024,
        data_dir=data_dir,
        output_dir=data_dir,
        model_name="elo",
        console=Console(quiet=True),
    )
    return run.run_id, data_dir


@pytest.fixture(scope="module")
def xgboost_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, Path]:
    """Train an XGBoost model once for all predict/export tests.

    Returns:
        (run_id, data_dir)
    """
    from rich.console import Console

    from ncaa_eval.cli.train import run_training
    from ncaa_eval.model import get_model

    data_dir = tmp_path_factory.mktemp("xgboost_run")
    _populate_data_dir(data_dir)
    model = get_model("xgboost")()
    run = run_training(
        model,
        start_year=2023,
        end_year=2024,
        data_dir=data_dir,
        output_dir=data_dir,
        model_name="xgboost",
        console=Console(quiet=True),
    )
    return run.run_id, data_dir


@pytest.fixture(scope="module")
def ensemble_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, Path]:
    """Train a StackedEnsemble (Elo + LogisticRegression base, LR meta) once.

    StackedEnsemble is not trainable via ``cli train --model ensemble``
    because the registry entry is a sentinel.  Programmatic construction
    is the documented path for ensemble training.

    Returns:
        (run_id, data_dir)
    """
    from rich.console import Console

    from ncaa_eval.cli.train import run_training
    from ncaa_eval.ingest.repository import ParquetRepository
    from ncaa_eval.ingest.schema import Game
    from ncaa_eval.model import get_model
    from ncaa_eval.model.ensemble import StackedEnsemble

    data_dir = tmp_path_factory.mktemp("ensemble_run")
    _populate_data_dir(data_dir)

    # Add tournament games to 2024 so walk_forward_splits produces a non-empty
    # test fold — the splitter uses is_tournament=True as the test-fold filter,
    # so without these games all OOF fold results would be empty and the
    # ensemble training would abort before saving the model.
    repo = ParquetRepository(base_path=data_dir)
    existing = repo.get_games(2024)
    tournament_games = [
        Game(
            game_id=f"t2024_{i:02d}",
            season=2024,
            day_num=134 + i,
            w_team_id=101 + (i % 2),
            l_team_id=103 + (i % 2),
            w_score=75,
            l_score=60,
            loc="N",
            is_tournament=True,
        )
        for i in range(4)
    ]
    repo.save_games(existing + tournament_games)

    ensemble = StackedEnsemble(
        base_models=[get_model("elo")(), get_model("elo")()],
        meta_learner=get_model("logistic_regression")(),
    )
    run = run_training(
        ensemble,
        start_year=2023,
        end_year=2024,
        data_dir=data_dir,
        output_dir=data_dir,
        model_name="ensemble",
        console=Console(quiet=True),
    )
    return run.run_id, data_dir


# ---------------------------------------------------------------------------
# CLI predict tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.no_mutation
def test_cli_predict_elo(elo_run: tuple[str, Path], tmp_path: Path) -> None:
    """``cli predict`` with a stateful Elo model produces a valid CSV."""
    from typer.testing import CliRunner

    from ncaa_eval.cli.main import app

    run_id, data_dir = elo_run
    output_csv = tmp_path / "predictions.csv"
    result = CliRunner().invoke(
        app,
        [
            "predict",
            "--run-id",
            run_id,
            "--season",
            "2024",
            "--data-dir",
            str(data_dir),
            "--output",
            str(output_csv),
        ],
    )
    assert result.exit_code == 0, f"predict (elo) failed:\n{result.output}"
    content = output_csv.read_text()
    assert content.startswith("season,team_a_id,team_b_id,pred_win_prob"), content[:200]
    assert len(content.strip().splitlines()) > 1, "expected data rows"


@pytest.mark.integration
@pytest.mark.no_mutation
def test_cli_predict_xgboost(xgboost_run: tuple[str, Path], tmp_path: Path) -> None:
    """``cli predict`` with a stateless XGBoost model produces a valid CSV."""
    from typer.testing import CliRunner

    from ncaa_eval.cli.main import app

    run_id, data_dir = xgboost_run
    output_csv = tmp_path / "predictions.csv"
    result = CliRunner().invoke(
        app,
        [
            "predict",
            "--run-id",
            run_id,
            "--season",
            "2024",
            "--data-dir",
            str(data_dir),
            "--output",
            str(output_csv),
        ],
    )
    assert result.exit_code == 0, f"predict (xgboost) failed:\n{result.output}"
    content = output_csv.read_text()
    assert content.startswith("season,team_a_id,team_b_id,pred_win_prob"), content[:200]
    assert len(content.strip().splitlines()) > 1, "expected data rows"


@pytest.mark.integration
@pytest.mark.no_mutation
def test_cli_predict_ensemble(ensemble_run: tuple[str, Path], tmp_path: Path) -> None:
    """``cli predict`` with a StackedEnsemble produces a valid CSV."""
    from typer.testing import CliRunner

    from ncaa_eval.cli.main import app

    run_id, data_dir = ensemble_run
    output_csv = tmp_path / "predictions.csv"
    result = CliRunner().invoke(
        app,
        [
            "predict",
            "--run-id",
            run_id,
            "--season",
            "2024",
            "--data-dir",
            str(data_dir),
            "--output",
            str(output_csv),
        ],
    )
    assert result.exit_code == 0, f"predict (ensemble) failed:\n{result.output}"
    content = output_csv.read_text()
    assert content.startswith("season,team_a_id,team_b_id,pred_win_prob"), content[:200]
    assert len(content.strip().splitlines()) > 1, "expected data rows"


@pytest.mark.integration
@pytest.mark.no_mutation
def test_cli_predict_invalid_run_id(tmp_path: Path) -> None:
    """``cli predict`` exits 1 when run_id cannot be found."""
    from typer.testing import CliRunner

    from ncaa_eval.cli.main import app

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    result = CliRunner().invoke(
        app,
        [
            "predict",
            "--run-id",
            "nonexistent-run",
            "--season",
            "2024",
            "--data-dir",
            str(data_dir),
            "--output",
            str(tmp_path / "out.csv"),
        ],
    )
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# CLI export tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.no_mutation
def test_cli_export_elo(elo_run: tuple[str, Path], tmp_path: Path) -> None:
    """``cli export`` with an Elo model produces a Kaggle submission CSV."""
    from typer.testing import CliRunner

    from ncaa_eval.cli.main import app

    run_id, data_dir = elo_run
    output_csv = tmp_path / "submission.csv"
    result = CliRunner().invoke(
        app,
        [
            "export",
            "--run-id",
            run_id,
            "--season",
            "2024",
            "--data-dir",
            str(data_dir),
            "--output",
            str(output_csv),
        ],
    )
    assert result.exit_code == 0, f"export (elo) failed:\n{result.output}"
    content = output_csv.read_text()
    assert content.startswith("ID,Pred"), content[:200]
    assert len(content.strip().splitlines()) > 1, "expected data rows"


@pytest.mark.integration
@pytest.mark.no_mutation
def test_cli_export_stateless_rejected(xgboost_run: tuple[str, Path], tmp_path: Path) -> None:
    """``cli export`` exits 1 for a stateless (XGBoost) model — unsupported."""
    from typer.testing import CliRunner

    from ncaa_eval.cli.main import app

    run_id, data_dir = xgboost_run
    result = CliRunner().invoke(
        app,
        [
            "export",
            "--run-id",
            run_id,
            "--season",
            "2024",
            "--data-dir",
            str(data_dir),
            "--output",
            str(tmp_path / "out.csv"),
        ],
    )
    assert result.exit_code == 1


@pytest.mark.integration
@pytest.mark.no_mutation
def test_cli_export_invalid_run_id(tmp_path: Path) -> None:
    """``cli export`` exits 1 when run_id cannot be found."""
    from typer.testing import CliRunner

    from ncaa_eval.cli.main import app

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    result = CliRunner().invoke(
        app,
        [
            "export",
            "--run-id",
            "nonexistent-run",
            "--season",
            "2024",
            "--data-dir",
            str(data_dir),
            "--output",
            str(tmp_path / "out.csv"),
        ],
    )
    assert result.exit_code == 1
