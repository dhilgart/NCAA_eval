"""DataFrame validation helpers for data integrity checks.

Provides a small set of assertion functions that validate common
DataFrame properties — shape, column presence, dtypes, null values,
and value ranges.  Every function raises `ValueError` (never
`AssertionError`) so that validations cannot be silently disabled
by ``python -O``.

Usage:
    >>> import pandas as pd
    >>> from ncaa_eval.utils.assertions import assert_columns, assert_no_nulls
    >>> df = pd.DataFrame({"TeamID": [1, 2], "Score": [70, 85]})
    >>> assert_columns(df, ["TeamID", "Score"])
    >>> assert_no_nulls(df)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd  # type: ignore[import-untyped]


def assert_shape(
    df: pd.DataFrame,
    expected_rows: int | None = None,
    expected_cols: int | None = None,
) -> None:
    """Validate DataFrame dimensions.

    Args:
        df: DataFrame to check.
        expected_rows: Expected number of rows (``None`` to skip).
        expected_cols: Expected number of columns (``None`` to skip).

    Raises:
        ValueError: If actual dimensions do not match expectations.

    Example:
        >>> import pandas as pd
        >>> from ncaa_eval.utils.assertions import assert_shape
        >>> df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> assert_shape(df, expected_rows=3, expected_cols=2)
    """
    actual_rows, actual_cols = df.shape
    if expected_rows is not None and actual_rows != expected_rows:
        msg = (
            f"assert_shape failed: expected ({expected_rows}, "
            f"{expected_cols if expected_cols is not None else actual_cols}), "
            f"got ({actual_rows}, {actual_cols})"
        )
        raise ValueError(msg)
    if expected_cols is not None and actual_cols != expected_cols:
        msg = (
            f"assert_shape failed: expected ("
            f"{expected_rows if expected_rows is not None else actual_rows}, "
            f"{expected_cols}), got ({actual_rows}, {actual_cols})"
        )
        raise ValueError(msg)


def assert_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    """Validate that all required columns exist in the DataFrame.

    Args:
        df: DataFrame to check.
        required: Column names that must be present.

    Raises:
        ValueError: If any required columns are missing.

    Example:
        >>> import pandas as pd
        >>> from ncaa_eval.utils.assertions import assert_columns
        >>> df = pd.DataFrame({"TeamID": [1], "Score": [70]})
        >>> assert_columns(df, ["TeamID", "Score"])
    """
    missing = set(required) - set(df.columns)
    if missing:
        msg = f"assert_columns failed: missing columns {missing}"
        raise ValueError(msg)


def assert_dtypes(df: pd.DataFrame, expected: Mapping[str, str | type]) -> None:
    """Validate column dtype mapping.

    Args:
        df: DataFrame to check.
        expected: Mapping of column name to expected dtype (as string or
            type, e.g. ``{"Score": "int64"}``).

    Raises:
        ValueError: If any column's dtype does not match the expectation.

    Example:
        >>> import pandas as pd
        >>> from ncaa_eval.utils.assertions import assert_dtypes
        >>> df = pd.DataFrame({"Score": [70, 85]})
        >>> assert_dtypes(df, {"Score": "int64"})
    """
    missing_cols = set(expected.keys()) - set(df.columns)
    if missing_cols:
        msg = f"assert_dtypes failed: columns not found in DataFrame: {missing_cols}"
        raise ValueError(msg)
    for col, dtype in expected.items():
        actual = str(df[col].dtype)
        expected_str = dtype if isinstance(dtype, str) else dtype.__name__
        if actual != expected_str:
            msg = f"assert_dtypes failed for column {col!r}: expected {expected_str}, got {actual}"
            raise ValueError(msg)


def assert_no_nulls(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> None:
    """Validate no null values in specified or all columns.

    Args:
        df: DataFrame to check.
        columns: Specific columns to check.  ``None`` checks all columns.

    Raises:
        ValueError: If null values are found.

    Example:
        >>> import pandas as pd
        >>> from ncaa_eval.utils.assertions import assert_no_nulls
        >>> df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        >>> assert_no_nulls(df)
    """
    cols_to_check: Sequence[str] = list(df.columns) if columns is None else columns
    if columns is not None:
        missing_cols = set(columns) - set(df.columns)
        if missing_cols:
            msg = f"assert_no_nulls failed: columns not found in DataFrame: {missing_cols}"
            raise ValueError(msg)
    null_counts = df[list(cols_to_check)].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        detail = ", ".join(f"{col!r} ({count} nulls)" for col, count in cols_with_nulls.items())
        msg = f"assert_no_nulls failed: null values found in columns [{detail}]"
        raise ValueError(msg)


def assert_value_range(
    df: pd.DataFrame,
    column: str,
    *,
    min_val: float | None = None,
    max_val: float | None = None,
) -> None:
    """Validate that column values fall within the given bounds.

    Args:
        df: DataFrame to check.
        column: Column whose values to validate.
        min_val: Minimum allowed value (inclusive).  ``None`` to skip.
        max_val: Maximum allowed value (inclusive).  ``None`` to skip.

    Raises:
        ValueError: If any values fall outside the specified range.

    Example:
        >>> import pandas as pd
        >>> from ncaa_eval.utils.assertions import assert_value_range
        >>> df = pd.DataFrame({"Score": [60, 70, 80]})
        >>> assert_value_range(df, "Score", min_val=0, max_val=200)
    """
    if column not in df.columns:
        msg = f"assert_value_range failed: column {column!r} not found in DataFrame"
        raise ValueError(msg)
    series = df[column]
    violations = pd.Series(False, index=series.index)

    if min_val is not None:
        violations = violations | (series < min_val)
    if max_val is not None:
        violations = violations | (series > max_val)

    n_violations = int(violations.sum())
    if n_violations > 0:
        actual_min = series.min()
        actual_max = series.max()
        range_str = (
            f"[{min_val if min_val is not None else '-inf'}, {max_val if max_val is not None else 'inf'}]"
        )
        msg = (
            f"assert_value_range failed for column {column!r}: "
            f"{n_violations} values outside range {range_str}, "
            f"min={actual_min}, max={actual_max}"
        )
        raise ValueError(msg)
