"""DataFrame validation helpers backed by Pandera.

Provides a functional API for validating common DataFrame properties —
shape, column presence, dtypes, null values, and value ranges.

All functions except `assert_shape` delegate to Pandera and propagate
`pandera.errors.SchemaError` on failure. `assert_shape` raises `ValueError`
because Pandera does not support partial-dimension (one-sided) shape checks.

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
import pandera.pandas as pa


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
        pa.errors.SchemaError: If any required columns are missing.

    Example:
        >>> import pandas as pd
        >>> from ncaa_eval.utils.assertions import assert_columns
        >>> df = pd.DataFrame({"TeamID": [1], "Score": [70]})
        >>> assert_columns(df, ["TeamID", "Score"])
    """
    if not required:
        return
    pa.DataFrameSchema(
        {col: pa.Column() for col in required},
        strict=False,
    ).validate(df)


def assert_dtypes(df: pd.DataFrame, expected: Mapping[str, str | type]) -> None:
    """Validate column dtype mapping.

    Args:
        df: DataFrame to check.
        expected: Mapping of column name to expected dtype (as string or
            type, e.g. ``{"Score": "int64"}``).

    Raises:
        pa.errors.SchemaError: If any column's dtype does not match the
            expectation, or a specified column is not present.

    Example:
        >>> import pandas as pd
        >>> from ncaa_eval.utils.assertions import assert_dtypes
        >>> df = pd.DataFrame({"Score": [70, 85]})
        >>> assert_dtypes(df, {"Score": "int64"})
    """
    if not expected:
        return
    pa.DataFrameSchema(
        {col: pa.Column(dtype=dtype) for col, dtype in expected.items()},
        strict=False,
    ).validate(df)


def assert_no_nulls(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> None:
    """Validate no null values in specified or all columns.

    Args:
        df: DataFrame to check.
        columns: Specific columns to check.  ``None`` checks all columns.

    Raises:
        pa.errors.SchemaError: If null values are found, or a specified
            column is not present.

    Example:
        >>> import pandas as pd
        >>> from ncaa_eval.utils.assertions import assert_no_nulls
        >>> df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        >>> assert_no_nulls(df)
    """
    cols = list(df.columns) if columns is None else list(columns)
    if not cols:
        return
    pa.DataFrameSchema(
        {col: pa.Column(nullable=False) for col in cols},
        strict=False,
    ).validate(df)


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
        pa.errors.SchemaError: If any values fall outside the specified range,
            or the column is not present.

    Example:
        >>> import pandas as pd
        >>> from ncaa_eval.utils.assertions import assert_value_range
        >>> df = pd.DataFrame({"Score": [60, 70, 80]})
        >>> assert_value_range(df, "Score", min_val=0, max_val=200)
    """
    checks: list[pa.Check] = []
    if min_val is not None:
        checks.append(pa.Check.ge(min_val))
    if max_val is not None:
        checks.append(pa.Check.le(max_val))
    # Always build the schema — validates column existence even when no bounds given.
    pa.DataFrameSchema(
        {column: pa.Column(checks=checks or None)},
        strict=False,
    ).validate(df)
