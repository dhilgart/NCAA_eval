"""Unit tests for the DataFrame assertions module."""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from ncaa_eval.utils.assertions import (
    assert_columns,
    assert_dtypes,
    assert_no_nulls,
    assert_shape,
    assert_value_range,
)

# ---------------------------------------------------------------------------
# assert_shape
# ---------------------------------------------------------------------------


@pytest.mark.smoke
class TestAssertShape:
    """Tests for `assert_shape`."""

    def test_exact_match_passes(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert_shape(df, expected_rows=3, expected_cols=2)

    def test_rows_only_passes(self) -> None:
        df = pd.DataFrame({"a": [1, 2]})
        assert_shape(df, expected_rows=2)

    def test_cols_only_passes(self) -> None:
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        assert_shape(df, expected_cols=3)

    def test_no_constraints_passes(self) -> None:
        df = pd.DataFrame({"a": [1]})
        assert_shape(df)

    def test_wrong_rows_raises(self) -> None:
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError, match="assert_shape failed"):
            assert_shape(df, expected_rows=5)

    def test_wrong_cols_raises(self) -> None:
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="assert_shape failed"):
            assert_shape(df, expected_cols=10)

    def test_error_message_contains_actual_and_expected(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(ValueError, match=r"expected \(5, 2\), got \(2, 2\)"):
            assert_shape(df, expected_rows=5, expected_cols=2)


# ---------------------------------------------------------------------------
# assert_columns
# ---------------------------------------------------------------------------


@pytest.mark.smoke
class TestAssertColumns:
    """Tests for `assert_columns`."""

    def test_all_present_passes(self) -> None:
        df = pd.DataFrame({"TeamID": [1], "Score": [70]})
        assert_columns(df, ["TeamID", "Score"])

    def test_subset_passes(self) -> None:
        df = pd.DataFrame({"TeamID": [1], "Score": [70], "Extra": [0]})
        assert_columns(df, ["TeamID"])

    def test_missing_column_raises(self) -> None:
        df = pd.DataFrame({"TeamID": [1]})
        with pytest.raises(ValueError, match="assert_columns failed"):
            assert_columns(df, ["TeamID", "Score"])

    def test_error_message_lists_missing(self) -> None:
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Score"):
            assert_columns(df, ["Score"])

    def test_empty_required_passes(self) -> None:
        df = pd.DataFrame({"a": [1]})
        assert_columns(df, [])


# ---------------------------------------------------------------------------
# assert_dtypes
# ---------------------------------------------------------------------------


@pytest.mark.smoke
class TestAssertDtypes:
    """Tests for `assert_dtypes`."""

    def test_matching_dtypes_passes(self) -> None:
        df = pd.DataFrame({"Score": [70, 85]})
        assert_dtypes(df, {"Score": "int64"})

    def test_multiple_columns_passes(self) -> None:
        df = pd.DataFrame({"a": [1], "b": [1.0], "c": ["x"]})
        assert_dtypes(df, {"a": "int64", "b": "float64", "c": "object"})

    def test_wrong_dtype_raises(self) -> None:
        df = pd.DataFrame({"Score": [70.0, 85.0]})
        with pytest.raises(ValueError, match="assert_dtypes failed"):
            assert_dtypes(df, {"Score": "int64"})

    def test_error_message_contains_column_and_types(self) -> None:
        df = pd.DataFrame({"Score": [1.0]})
        with pytest.raises(ValueError, match=r"column 'Score': expected int64, got float64"):
            assert_dtypes(df, {"Score": "int64"})

    def test_nonexistent_column_raises_valueerror(self) -> None:
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="assert_dtypes failed"):
            assert_dtypes(df, {"nonexistent": "int64"})


# ---------------------------------------------------------------------------
# assert_no_nulls
# ---------------------------------------------------------------------------


@pytest.mark.smoke
class TestAssertNoNulls:
    """Tests for `assert_no_nulls`."""

    def test_no_nulls_all_columns_passes(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        assert_no_nulls(df)

    def test_no_nulls_specific_columns_passes(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [None, 4]})
        assert_no_nulls(df, columns=["a"])

    def test_nulls_in_all_columns_mode_raises(self) -> None:
        df = pd.DataFrame({"a": [1, None], "b": [3, 4]})
        with pytest.raises(ValueError, match="assert_no_nulls failed"):
            assert_no_nulls(df)

    def test_nulls_in_specific_column_raises(self) -> None:
        df = pd.DataFrame({"a": [1, None], "b": [3, 4]})
        with pytest.raises(ValueError, match="assert_no_nulls failed"):
            assert_no_nulls(df, columns=["a"])

    def test_error_message_lists_columns_and_counts(self) -> None:
        df = pd.DataFrame({"TeamName": [None, None, "A"]})
        with pytest.raises(ValueError, match=r"'TeamName' \(2 nulls\)"):
            assert_no_nulls(df)

    def test_nan_detected_as_null(self) -> None:
        df = pd.DataFrame({"a": [1.0, np.nan]})
        with pytest.raises(ValueError, match="assert_no_nulls failed"):
            assert_no_nulls(df)

    def test_nonexistent_column_raises_valueerror(self) -> None:
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="assert_no_nulls failed"):
            assert_no_nulls(df, columns=["nonexistent"])


# ---------------------------------------------------------------------------
# assert_value_range
# ---------------------------------------------------------------------------


@pytest.mark.smoke
class TestAssertValueRange:
    """Tests for `assert_value_range`."""

    def test_values_within_range_passes(self) -> None:
        df = pd.DataFrame({"Score": [60, 70, 80]})
        assert_value_range(df, "Score", min_val=0, max_val=200)

    def test_boundary_values_pass(self) -> None:
        df = pd.DataFrame({"Score": [0, 200]})
        assert_value_range(df, "Score", min_val=0, max_val=200)

    def test_min_only_passes(self) -> None:
        df = pd.DataFrame({"Score": [10, 20]})
        assert_value_range(df, "Score", min_val=0)

    def test_max_only_passes(self) -> None:
        df = pd.DataFrame({"Score": [10, 20]})
        assert_value_range(df, "Score", max_val=100)

    def test_below_min_raises(self) -> None:
        df = pd.DataFrame({"Score": [-3, 50, 100]})
        with pytest.raises(ValueError, match="assert_value_range failed"):
            assert_value_range(df, "Score", min_val=0, max_val=200)

    def test_above_max_raises(self) -> None:
        df = pd.DataFrame({"Score": [50, 250]})
        with pytest.raises(ValueError, match="assert_value_range failed"):
            assert_value_range(df, "Score", min_val=0, max_val=200)

    def test_error_message_contains_violation_count_and_range(self) -> None:
        df = pd.DataFrame({"Score": [-3, 50, 250]})
        with pytest.raises(ValueError, match=r"2 values outside range \[0, 200\]"):
            assert_value_range(df, "Score", min_val=0, max_val=200)

    def test_error_message_contains_actual_min_max(self) -> None:
        df = pd.DataFrame({"Score": [-3, 50, 250]})
        with pytest.raises(ValueError, match=r"min=-3.*max=250"):
            assert_value_range(df, "Score", min_val=0, max_val=200)

    def test_no_constraints_passes(self) -> None:
        df = pd.DataFrame({"Score": [-999, 999]})
        assert_value_range(df, "Score")

    def test_nonexistent_column_raises_valueerror(self) -> None:
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="assert_value_range failed"):
            assert_value_range(df, "nonexistent")
