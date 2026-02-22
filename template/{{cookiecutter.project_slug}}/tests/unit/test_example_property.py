"""Example property-based test using Hypothesis.

Demonstrates the pattern for invariant testing. Replace with
real property tests as you implement business logic.
"""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st


@pytest.mark.property
@given(st.integers())
def test_example_integer_property(value: int) -> None:
    """Example: integers are always instances of int."""
    assert isinstance(value, int)
