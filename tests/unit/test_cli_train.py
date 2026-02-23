"""Integration tests for the training CLI (``python -m ncaa_eval.cli train``)."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from typer.testing import CliRunner

from ncaa_eval.cli.main import app
from ncaa_eval.model.tracking import RunStore

runner = CliRunner()


def _make_synthetic_season(
    year: int,
    n_games: int = 20,
    n_tournament: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic feature DataFrame mimicking StatefulFeatureServer output.

    Produces metadata columns plus two numeric feature columns (``feat_a``,
    ``feat_b``).  The first ``n_games - n_tournament`` rows are regular season;
    the remaining are tournament games.
    """
    rng = np.random.default_rng(seed + year)
    n_reg = n_games - n_tournament

    data: dict[str, object] = {
        "game_id": [f"{year}_game_{i}" for i in range(n_games)],
        "season": [year] * n_games,
        "day_num": list(range(10, 10 + n_games)),
        "date": [datetime.date(year, 1, 10 + i) for i in range(n_games)],
        "team_a_id": [100 + i for i in range(n_games)],
        "team_b_id": [200 + i for i in range(n_games)],
        "is_tournament": [*([False] * n_reg), *([True] * n_tournament)],
        "loc_encoding": [rng.choice([0, 1, -1]) for _ in range(n_games)],
        "team_a_won": [bool(rng.choice([True, False])) for _ in range(n_games)],
        "w_score": [70 + int(rng.integers(0, 20)) for _ in range(n_games)],
        "l_score": [60 + int(rng.integers(0, 15)) for _ in range(n_games)],
        "num_ot": [0] * n_games,
        "feat_a": rng.standard_normal(n_games).tolist(),
        "feat_b": rng.standard_normal(n_games).tolist(),
    }
    return pd.DataFrame(data)


def _mock_serve_season_features(self: object, year: int, mode: str = "batch") -> pd.DataFrame:
    """Replacement for ``StatefulFeatureServer.serve_season_features``."""
    return _make_synthetic_season(year)


# ---------------------------------------------------------------------------
# Task 6.1: CLI train with logistic_regression on synthetic data
# ---------------------------------------------------------------------------


class TestCLITrain:
    @patch(
        "ncaa_eval.cli.train.StatefulFeatureServer.serve_season_features",
        _mock_serve_season_features,
    )
    def test_train_logistic_regression(self, tmp_path: Path) -> None:
        """CLI invocation with logistic_regression creates a run with correct model_type."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "train",
                "--model",
                "logistic_regression",
                "--start-year",
                "2020",
                "--end-year",
                "2021",
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(output_dir),
            ],
        )
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Verify the persisted run records the correct model_type and year range
        store = RunStore(base_path=output_dir)
        runs = store.list_runs()
        assert len(runs) == 1
        assert runs[0].model_type == "logistic_regression"
        assert runs[0].start_year == 2020
        assert runs[0].end_year == 2021

    # -----------------------------------------------------------------------
    # Task 6.2: CLI produces run.json and predictions.parquet output files
    # -----------------------------------------------------------------------

    @patch(
        "ncaa_eval.cli.train.StatefulFeatureServer.serve_season_features",
        _mock_serve_season_features,
    )
    def test_train_produces_output_files(self, tmp_path: Path) -> None:
        """run.json and predictions.parquet are created in the output dir."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "train",
                "--model",
                "logistic_regression",
                "--start-year",
                "2020",
                "--end-year",
                "2021",
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(output_dir),
            ],
        )
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Discover the run
        store = RunStore(base_path=output_dir)
        runs = store.list_runs()
        assert len(runs) == 1

        run = runs[0]
        run_dir = output_dir / "runs" / run.run_id
        assert (run_dir / "run.json").exists()
        assert (run_dir / "predictions.parquet").exists()

    # -----------------------------------------------------------------------
    # Task 6.3: Invalid model name prints error with available models
    # -----------------------------------------------------------------------

    def test_train_invalid_model(self) -> None:
        """CLI with an invalid model name exits with error and lists available."""
        result = runner.invoke(
            app,
            ["train", "--model", "nonexistent_model"],
        )
        assert result.exit_code != 0
        assert "Available models" in result.output

    # -----------------------------------------------------------------------
    # Task 6.4: Config override applies custom hyperparameters
    # -----------------------------------------------------------------------

    @patch(
        "ncaa_eval.cli.train.StatefulFeatureServer.serve_season_features",
        _mock_serve_season_features,
    )
    def test_train_with_config_override(self, tmp_path: Path) -> None:
        """--config flag applies custom hyperparameters from a JSON file."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        config_file = tmp_path / "custom_config.json"
        config_file.write_text(json.dumps({"C": 10.0, "max_iter": 500}))

        result = runner.invoke(
            app,
            [
                "train",
                "--model",
                "logistic_regression",
                "--start-year",
                "2020",
                "--end-year",
                "2021",
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(output_dir),
                "--config",
                str(config_file),
            ],
        )
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Verify config was applied
        store = RunStore(base_path=output_dir)
        runs = store.list_runs()
        assert len(runs) == 1
        run = runs[0]
        assert run.hyperparameters["C"] == 10.0
        assert run.hyperparameters["max_iter"] == 500
