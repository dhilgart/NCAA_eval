"""Tests for Kaggle submission CSV export."""

from __future__ import annotations

import csv
import io
from itertools import combinations

import numpy as np
import pytest

from ncaa_eval.evaluation.kaggle_export import format_kaggle_submission


class TestFormatKaggleSubmission:
    """Tests for the format_kaggle_submission pure function."""

    @pytest.fixture
    def small_matrix(self) -> tuple[list[int], np.ndarray]:
        """4-team probability matrix with known values.

        Teams: 1101, 1102, 1103, 1104
        P[i,j] = probability team_ids[i] beats team_ids[j]
        """
        team_ids = [1101, 1102, 1103, 1104]
        n = len(team_ids)
        P = np.zeros((n, n), dtype=np.float64)
        # 1101 beats 1102 with 60%, 1103 with 70%, 1104 with 80%
        P[0, 1] = 0.6
        P[0, 2] = 0.7
        P[0, 3] = 0.8
        # 1102 beats 1103 with 55%, 1104 with 65%
        P[1, 2] = 0.55
        P[1, 3] = 0.65
        # 1103 beats 1104 with 50%
        P[2, 3] = 0.5
        # Complementarity
        P[1, 0] = 0.4
        P[2, 0] = 0.3
        P[3, 0] = 0.2
        P[2, 1] = 0.45
        P[3, 1] = 0.35
        P[3, 2] = 0.5
        return team_ids, P

    def test_csv_header(self, small_matrix: tuple[list[int], np.ndarray]) -> None:
        """Output starts with exactly 'ID,Pred' header."""
        team_ids, P = small_matrix
        result = format_kaggle_submission(2025, team_ids, P)
        lines = result.strip().split("\n")
        assert lines[0] == "ID,Pred"

    def test_row_count(self, small_matrix: tuple[list[int], np.ndarray]) -> None:
        """Output has C(4,2)=6 data rows."""
        team_ids, P = small_matrix
        result = format_kaggle_submission(2025, team_ids, P)
        lines = result.strip().split("\n")
        assert len(lines) == 7  # 1 header + 6 data rows

    def test_id_format_lower_first(self, small_matrix: tuple[list[int], np.ndarray]) -> None:
        """All IDs have lower team ID first: YYYY_LowerID_HigherID."""
        team_ids, P = small_matrix
        result = format_kaggle_submission(2025, team_ids, P)
        reader = csv.DictReader(io.StringIO(result))
        for row in reader:
            parts = row["ID"].split("_")
            assert len(parts) == 3
            assert parts[0] == "2025"
            assert int(parts[1]) < int(parts[2])

    def test_probability_values(self, small_matrix: tuple[list[int], np.ndarray]) -> None:
        """Pred column matches expected probabilities for lower-ID team."""
        team_ids, P = small_matrix
        result = format_kaggle_submission(2025, team_ids, P)
        reader = csv.DictReader(io.StringIO(result))
        rows = list(reader)

        # Build expected: for sorted pairs, P(lower beats higher)
        idx = {tid: i for i, tid in enumerate(team_ids)}
        expected = {}
        for low, high in combinations(sorted(team_ids), 2):
            expected[f"2025_{low}_{high}"] = P[idx[low], idx[high]]

        for row in rows:
            assert row["ID"] in expected
            assert float(row["Pred"]) == pytest.approx(expected[row["ID"]])

    def test_probabilities_in_valid_range(self, small_matrix: tuple[list[int], np.ndarray]) -> None:
        """All Pred values are in [0, 1]."""
        team_ids, P = small_matrix
        result = format_kaggle_submission(2025, team_ids, P)
        reader = csv.DictReader(io.StringIO(result))
        for row in reader:
            pred = float(row["Pred"])
            assert 0.0 <= pred <= 1.0

    def test_complementarity(self, small_matrix: tuple[list[int], np.ndarray]) -> None:
        """P(A beats B) in CSV = 1 - matrix P(B beats A)."""
        team_ids, P = small_matrix
        result = format_kaggle_submission(2025, team_ids, P)
        reader = csv.DictReader(io.StringIO(result))
        idx = {tid: i for i, tid in enumerate(team_ids)}

        for row in reader:
            parts = row["ID"].split("_")
            low_id = int(parts[1])
            high_id = int(parts[2])
            pred = float(row["Pred"])
            # Pred = P(low beats high) = matrix[low_idx, high_idx]
            # Complement: 1 - P(high beats low) = 1 - matrix[high_idx, low_idx]
            assert pred == pytest.approx(1.0 - P[idx[high_id], idx[low_id]])

    def test_unsorted_team_ids(self) -> None:
        """Output is correct even when input team_ids are not sorted."""
        team_ids = [1104, 1101, 1103]  # not sorted
        n = len(team_ids)
        P = np.full((n, n), 0.5, dtype=np.float64)
        np.fill_diagonal(P, 0.0)
        # 1104 is index 0, 1101 is index 1, 1103 is index 2
        P[1, 0] = 0.3  # P(1101 beats 1104)
        P[0, 1] = 0.7  # P(1104 beats 1101)
        P[1, 2] = 0.6  # P(1101 beats 1103)
        P[2, 1] = 0.4  # P(1103 beats 1101)
        P[0, 2] = 0.55  # P(1104 beats 1103)
        P[2, 0] = 0.45  # P(1103 beats 1104)

        result = format_kaggle_submission(2025, team_ids, P)
        reader = csv.DictReader(io.StringIO(result))
        rows = {r["ID"]: float(r["Pred"]) for r in reader}

        # ID uses lower ID first
        assert "2025_1101_1103" in rows
        assert "2025_1101_1104" in rows
        assert "2025_1103_1104" in rows
        # P(1101 beats 1103) = P[1,2] = 0.6
        assert rows["2025_1101_1103"] == pytest.approx(0.6)
        # P(1101 beats 1104) = P[1,0] = 0.3
        assert rows["2025_1101_1104"] == pytest.approx(0.3)
        # P(1103 beats 1104) = P[2,0] = 0.45
        assert rows["2025_1103_1104"] == pytest.approx(0.45)

    def test_matrix_shape_mismatch_raises(self) -> None:
        """ValueError raised when matrix dimensions don't match team count."""
        team_ids = [1101, 1102, 1103]
        P = np.zeros((4, 4), dtype=np.float64)
        with pytest.raises(ValueError, match="prob_matrix shape"):
            format_kaggle_submission(2025, team_ids, P)

    def test_two_teams(self) -> None:
        """Minimal case: 2 teams produce exactly 1 row."""
        team_ids = [1101, 1102]
        P = np.array([[0.0, 0.6], [0.4, 0.0]], dtype=np.float64)
        result = format_kaggle_submission(2025, team_ids, P)
        lines = result.strip().split("\n")
        assert len(lines) == 2  # header + 1 row
        assert lines[1] == "2025_1101_1102,0.6"
