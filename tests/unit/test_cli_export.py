"""Unit tests for the export CLI command (``python -m ncaa_eval.cli export``)."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from ncaa_eval.cli.main import app
from ncaa_eval.ingest.schema import Game
from ncaa_eval.model.base import StatefulModel

runner = CliRunner()


def _make_mock_stateful_model() -> MagicMock:
    """Create a mock StatefulModel with a predict_matchup method."""
    model = MagicMock(spec=StatefulModel)
    # predict_matchup returns 0.5 for all matchups (simple Elo-like stub)
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


class TestCLIExport:
    """Tests for the CLI export command."""

    @patch("ncaa_eval.cli.export.ParquetRepository")
    @patch("ncaa_eval.cli.export.RunStore")
    def test_export_writes_csv(
        self,
        mock_store_cls: MagicMock,
        mock_repo_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """CLI export writes a valid Kaggle CSV to the output path."""
        # Set up mocks
        mock_store = mock_store_cls.return_value
        mock_store.load_model.return_value = _make_mock_stateful_model()

        mock_repo = mock_repo_cls.return_value
        mock_repo.get_games.return_value = _make_games(2025)

        output_file = tmp_path / "submission.csv"
        result = runner.invoke(
            app,
            [
                "export",
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

        # Verify CSV content
        content = output_file.read_text()
        reader = csv.DictReader(io.StringIO(content))
        rows = list(reader)
        # 3 teams → C(3,2) = 3 rows
        assert len(rows) == 3
        for row in rows:
            assert "ID" in row
            assert "Pred" in row
            parts = row["ID"].split("_")
            assert parts[0] == "2025"
            assert int(parts[1]) < int(parts[2])

    @patch("ncaa_eval.cli.export.ParquetRepository")
    @patch("ncaa_eval.cli.export.RunStore")
    def test_export_no_model_exits_with_error(
        self,
        mock_store_cls: MagicMock,
        mock_repo_cls: MagicMock,
    ) -> None:
        """CLI export exits with code 1 when no model is found."""
        mock_store = mock_store_cls.return_value
        mock_store.load_model.return_value = None

        result = runner.invoke(
            app,
            [
                "export",
                "--run-id",
                "nonexistent",
                "--season",
                "2025",
            ],
        )
        assert result.exit_code != 0
        assert "Error" in result.output

    @patch("ncaa_eval.cli.export.ParquetRepository")
    @patch("ncaa_eval.cli.export.RunStore")
    def test_export_stateless_model_exits_with_error(
        self,
        mock_store_cls: MagicMock,
        mock_repo_cls: MagicMock,
    ) -> None:
        """CLI export rejects stateless models with a clear error."""
        mock_store = mock_store_cls.return_value
        # Return a non-StatefulModel mock
        mock_model = MagicMock()
        mock_model.__class__ = type("FakeModel", (), {})
        mock_store.load_model.return_value = mock_model

        result = runner.invoke(
            app,
            [
                "export",
                "--run-id",
                "test-run",
                "--season",
                "2025",
            ],
        )
        assert result.exit_code != 0
        assert "stateful" in result.output.lower() or "Elo" in result.output

    @patch("ncaa_eval.cli.export.ParquetRepository")
    @patch("ncaa_eval.cli.export.RunStore")
    def test_export_missing_season_data_exits_with_error(
        self,
        mock_store_cls: MagicMock,
        mock_repo_cls: MagicMock,
    ) -> None:
        """CLI export exits with error when no games exist for the season."""
        mock_store = mock_store_cls.return_value
        mock_store.load_model.return_value = _make_mock_stateful_model()

        mock_repo = mock_repo_cls.return_value
        mock_repo.get_games.return_value = []

        result = runner.invoke(
            app,
            [
                "export",
                "--run-id",
                "test-run-001",
                "--season",
                "1900",
            ],
        )
        assert result.exit_code != 0
        assert "Error" in result.output

    @patch("ncaa_eval.cli.export.ParquetRepository")
    @patch("ncaa_eval.cli.export.RunStore")
    def test_export_stdout_when_no_output(
        self,
        mock_store_cls: MagicMock,
        mock_repo_cls: MagicMock,
    ) -> None:
        """CLI export outputs CSV to stdout when --output is not provided."""
        mock_store = mock_store_cls.return_value
        mock_store.load_model.return_value = _make_mock_stateful_model()

        mock_repo = mock_repo_cls.return_value
        mock_repo.get_games.return_value = _make_games(2025)

        result = runner.invoke(
            app,
            [
                "export",
                "--run-id",
                "test-run-001",
                "--season",
                "2025",
            ],
        )
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        # CSV content appears in stdout
        assert "ID,Pred" in result.output
        assert "2025_1101_1102" in result.output
