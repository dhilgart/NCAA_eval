"""Unit tests for the predict CLI command (``ncaa-eval predict``)."""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd  # type: ignore[import-untyped]
import pytest
from typer.testing import CliRunner

from ncaa_eval.cli.main import app
from ncaa_eval.ingest.schema import Game
from ncaa_eval.model.base import StatefulModel
from ncaa_eval.model.ensemble import StackedEnsemble
from ncaa_eval.model.logistic_regression import LogisticRegressionModel

runner = CliRunner()


def _make_mock_stateful_model() -> MagicMock:
    """Create a mock StatefulModel with a predict_matchup method."""
    model = MagicMock(spec=StatefulModel)
    model.predict_matchup.return_value = 0.6
    return model


def _make_games(season: int) -> list[Game]:
    """Create a small list of games for 3 teams."""
    return [
        Game(
            game_id=f"{season}_001",
            season=season,
            day_num=10,
            date=None,
            w_team_id=1101,
            l_team_id=1102,
            w_score=75,
            l_score=70,
            loc="N",
            num_ot=0,
            is_tournament=False,
        ),
        Game(
            game_id=f"{season}_002",
            season=season,
            day_num=11,
            date=None,
            w_team_id=1103,
            l_team_id=1101,
            w_score=80,
            l_score=75,
            loc="N",
            num_ot=0,
            is_tournament=False,
        ),
        Game(
            game_id=f"{season}_003",
            season=season,
            day_num=12,
            date=None,
            w_team_id=1102,
            l_team_id=1103,
            w_score=72,
            l_score=68,
            loc="N",
            num_ot=0,
            is_tournament=False,
        ),
    ]


class TestCLIPredict:
    """Tests for the CLI predict command."""

    @patch("ncaa_eval.cli.predict.ParquetRepository")
    @patch("ncaa_eval.cli.predict.RunStore")
    def test_stateful_predict_writes_csv_to_file(
        self,
        mock_store_cls: MagicMock,
        mock_repo_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """CLI predict with stateful model writes valid CSV to file."""
        mock_store = mock_store_cls.return_value
        mock_store.load_model.return_value = _make_mock_stateful_model()

        mock_repo = mock_repo_cls.return_value
        mock_repo.get_games.return_value = _make_games(2025)

        output_file = tmp_path / "predictions.csv"
        result = runner.invoke(
            app,
            [
                "predict",
                "--run-id",
                "test-run-001",
                "--season",
                "2025",
                "--data-dir",
                str(tmp_path),
                "--output",
                str(output_file),
            ],
        )
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert output_file.exists()

        content = output_file.read_text()
        reader = csv.DictReader(io.StringIO(content))
        rows = list(reader)
        # 3 teams → C(3,2) = 3 rows
        assert len(rows) == 3
        for row in rows:
            assert "season" in row
            assert "team_a_id" in row
            assert "team_b_id" in row
            assert "pred_win_prob" in row
            assert row["season"] == "2025"
            assert int(row["team_a_id"]) < int(row["team_b_id"])

    @patch("ncaa_eval.cli.predict.ParquetRepository")
    @patch("ncaa_eval.cli.predict.RunStore")
    def test_stateful_predict_writes_csv_to_stdout(
        self,
        mock_store_cls: MagicMock,
        mock_repo_cls: MagicMock,
    ) -> None:
        """CLI predict outputs CSV to stdout when --output is not provided."""
        mock_store = mock_store_cls.return_value
        mock_store.load_model.return_value = _make_mock_stateful_model()

        mock_repo = mock_repo_cls.return_value
        mock_repo.get_games.return_value = _make_games(2025)

        result = runner.invoke(
            app,
            [
                "predict",
                "--run-id",
                "test-run-001",
                "--season",
                "2025",
            ],
        )
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        # CliRunner merges all output streams; extract CSV lines starting from
        # the header so status messages (on real stderr) don't interfere.
        csv_header = "season,team_a_id,team_b_id,pred_win_prob"
        csv_start = result.output.find(csv_header)
        assert csv_start != -1, "CSV header not found in output"
        csv_text = result.output[csv_start:]
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)
        # 3 teams → C(3,2) = 3 rows
        assert len(rows) == 3
        for row in rows:
            assert row["season"] == "2025"
            assert int(row["team_a_id"]) < int(row["team_b_id"])

    @patch("ncaa_eval.cli.predict._setup_feature_server")
    @patch("ncaa_eval.cli.predict.RunStore")
    def test_stateless_predict_writes_csv(
        self,
        mock_store_cls: MagicMock,
        mock_setup_server: MagicMock,
        tmp_path: Path,
    ) -> None:
        """CLI predict with stateless model writes valid CSV with game-level rows."""
        # Stateless model mock — use LogisticRegressionModel spec so that
        # isinstance(mock, StatefulModel) is False without __class__ hacks.
        mock_model = MagicMock(spec=LogisticRegressionModel)
        mock_model.feature_config = MagicMock()
        mock_model.predict_proba.return_value = pd.Series([0.65, 0.42, 0.58])

        mock_store = mock_store_cls.return_value
        mock_store.load_model.return_value = mock_model
        mock_store.load_feature_names.return_value = ["feat_a", "feat_b"]

        # Mock feature server returning a DataFrame with game metadata + features
        season_df = pd.DataFrame(
            {
                "team_a_id": [1101, 1103, 1102],
                "team_b_id": [1102, 1101, 1103],
                "feat_a": [1.0, 2.0, 3.0],
                "feat_b": [4.0, 5.0, 6.0],
            }
        )
        mock_server = MagicMock()
        mock_server.serve_season_features.return_value = season_df
        mock_setup_server.return_value = mock_server

        output_file = tmp_path / "predictions.csv"
        result = runner.invoke(
            app,
            [
                "predict",
                "--run-id",
                "test-run-002",
                "--season",
                "2025",
                "--data-dir",
                str(tmp_path),
                "--output",
                str(output_file),
            ],
        )
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert output_file.exists()

        content = output_file.read_text()
        reader = csv.DictReader(io.StringIO(content))
        rows = list(reader)
        # 3 games → 3 rows
        assert len(rows) == 3
        for row in rows:
            assert "season" in row
            assert "team_a_id" in row
            assert "team_b_id" in row
            assert "pred_win_prob" in row
            prob = float(row["pred_win_prob"])
            assert 0.0 <= prob <= 1.0

    @patch("ncaa_eval.cli.predict._setup_feature_server")
    @patch("ncaa_eval.cli.predict.RunStore")
    def test_stateless_predict_no_feature_names_exits_with_error(
        self,
        mock_store_cls: MagicMock,
        mock_setup_server: MagicMock,
        tmp_path: Path,
    ) -> None:
        """CLI predict exits 1 when stateless model has no saved feature names.

        Regression: legacy runs trained before feature-name persistence was added
        have load_feature_names() == None.  This should raise FileNotFoundError
        at predict.py:78 and translate to exit code 1 via the CLI handler.
        """
        mock_model = MagicMock(spec=LogisticRegressionModel)
        mock_model.feature_config = MagicMock()

        mock_store = mock_store_cls.return_value
        mock_store.load_model.return_value = mock_model
        mock_store.load_feature_names.return_value = None  # ← the gap under test

        result = runner.invoke(
            app,
            [
                "predict",
                "--run-id",
                "legacy-run-no-feats",
                "--season",
                "2025",
                "--data-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code != 0
        assert "Error" in result.output

    @patch("ncaa_eval.cli.predict.ParquetRepository")
    @patch("ncaa_eval.cli.predict.RunStore")
    def test_nonexistent_run_exits_with_error(
        self,
        mock_store_cls: MagicMock,
        mock_repo_cls: MagicMock,
    ) -> None:
        """CLI predict exits with code 1 when no model is found."""
        mock_store = mock_store_cls.return_value
        mock_store.load_model.return_value = None

        result = runner.invoke(
            app,
            [
                "predict",
                "--run-id",
                "nonexistent",
                "--season",
                "2025",
            ],
        )
        assert result.exit_code != 0
        assert "Error" in result.output

    @patch("ncaa_eval.cli.predict.ParquetRepository")
    @patch("ncaa_eval.cli.predict.RunStore")
    def test_missing_season_data_exits_with_error(
        self,
        mock_store_cls: MagicMock,
        mock_repo_cls: MagicMock,
    ) -> None:
        """CLI predict exits with error when no games exist for the season."""
        mock_store = mock_store_cls.return_value
        mock_store.load_model.return_value = _make_mock_stateful_model()

        mock_repo = mock_repo_cls.return_value
        mock_repo.get_games.return_value = []

        result = runner.invoke(
            app,
            [
                "predict",
                "--run-id",
                "test-run-001",
                "--season",
                "1900",
            ],
        )
        assert result.exit_code != 0
        assert "Error" in result.output

    @patch("ncaa_eval.cli.predict.ParquetRepository")
    @patch("ncaa_eval.cli.predict.RunStore")
    def test_stdout_contains_only_csv_when_no_output_arg(
        self,
        mock_store_cls: MagicMock,
        mock_repo_cls: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """sys.stdout must be pure CSV — no status messages — when --output is omitted.

        Verifies pipe-safety: calls run_predict() directly (bypassing CliRunner's
        stream-merging) and asserts every line written to sys.stdout is CSV.
        """
        from ncaa_eval.cli.predict import run_predict

        mock_store = mock_store_cls.return_value
        mock_store.load_model.return_value = _make_mock_stateful_model()
        mock_repo = mock_repo_cls.return_value
        mock_repo.get_games.return_value = _make_games(2025)

        captured_stdout = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured_stdout)

        run_predict(
            run_id="test-run-001",
            season=2025,
            data_dir=tmp_path,
            output=None,
        )

        csv_output = captured_stdout.getvalue()
        lines = [ln for ln in csv_output.splitlines() if ln.strip()]
        assert lines[0] == "season,team_a_id,team_b_id,pred_win_prob", (
            f"First line of stdout is not CSV header: {lines[0]!r}"
        )
        for line in lines[1:]:
            parts = line.split(",")
            assert len(parts) == 4, f"Non-CSV line found in stdout: {line!r}"


class TestEnsemblePredict:
    """Tests for ensemble prediction through the CLI predict path."""

    @patch("ncaa_eval.cli.predict._build_ensemble_predictions")
    @patch("ncaa_eval.cli.predict.RunStore")
    def test_ensemble_predict_writes_csv(
        self,
        mock_store_cls: MagicMock,
        mock_build_ensemble: MagicMock,
        tmp_path: Path,
    ) -> None:
        """CLI predict with ensemble model produces valid CSV output."""
        # Mock RunStore to return an ensemble
        mock_ensemble = MagicMock(spec=StackedEnsemble)
        mock_store = mock_store_cls.return_value
        mock_store.load_model.return_value = mock_ensemble

        # Mock ensemble predictions: 3 teams → C(3,2)=3 rows
        mock_build_ensemble.return_value = [
            (2025, 1101, 1102, 0.65),
            (2025, 1101, 1103, 0.58),
            (2025, 1102, 1103, 0.52),
        ]

        output_file = tmp_path / "predictions.csv"
        result = runner.invoke(
            app,
            [
                "predict",
                "--run-id",
                "test-ensemble-001",
                "--season",
                "2025",
                "--data-dir",
                str(tmp_path),
                "--output",
                str(output_file),
            ],
        )
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert output_file.exists()

        content = output_file.read_text()
        reader = csv.DictReader(io.StringIO(content))
        rows = list(reader)
        assert len(rows) == 3
        for row in rows:
            assert row["season"] == "2025"
            assert "team_a_id" in row
            assert "team_b_id" in row
            assert "pred_win_prob" in row
            prob = float(row["pred_win_prob"])
            assert 0.0 <= prob <= 1.0

    @patch("ncaa_eval.cli.predict.RunStore")
    def test_build_ensemble_predictions_calls_predict_bracket(
        self,
        mock_store_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """_build_ensemble_predictions calls ensemble.predict_bracket and formats output correctly.

        This test patches at the ensemble.predict_bracket level (not the helper
        function) to exercise the actual _build_ensemble_predictions logic including
        upper-triangle extraction and probability clamping.
        """
        import numpy as np

        from ncaa_eval.cli.predict import _build_ensemble_predictions

        mock_ensemble = MagicMock(spec=StackedEnsemble)
        team_ids = [1101, 1102, 1103]
        n = len(team_ids)

        # Build a 3×3 probability matrix
        prob_matrix = np.zeros((n, n), dtype=np.float64)
        prob_matrix[0, 1] = 0.65  # 1101 vs 1102
        prob_matrix[1, 0] = 0.35
        prob_matrix[0, 2] = 0.58  # 1101 vs 1103
        prob_matrix[2, 0] = 0.42
        prob_matrix[1, 2] = 0.52  # 1102 vs 1103
        prob_matrix[2, 1] = 0.48

        mock_ensemble.predict_bracket.return_value = pd.DataFrame(
            prob_matrix, index=team_ids, columns=team_ids
        )
        mock_store = mock_store_cls.return_value
        mock_store.load_model.return_value = mock_ensemble

        rows = _build_ensemble_predictions(ensemble=mock_ensemble, season=2025, data_dir=tmp_path)

        # Should call predict_bracket with correct args
        mock_ensemble.predict_bracket.assert_called_once_with(tmp_path, 2025)

        # Upper-triangle: C(3,2) = 3 rows
        assert len(rows) == 3
        # All rows: season correct, team_a_id < team_b_id enforced by ordering
        for season_val, a, b, prob in rows:
            assert season_val == 2025
            assert 0.0 <= prob <= 1.0
        # Verify specific probabilities match the matrix
        row_dict = {(a, b): p for _, a, b, p in rows}
        assert row_dict[(1101, 1102)] == pytest.approx(0.65)
        assert row_dict[(1101, 1103)] == pytest.approx(0.58)
        assert row_dict[(1102, 1103)] == pytest.approx(0.52)
