"""Task 1 tests: each frame transform on three points (task1_cases.py).
Ran with:  python -m pytest -q tests

For every point: the forward transform must give the hand-predicted result,
and the inverse transform must bring it back to the original point.
"""
import sys
import pathlib

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import pytest
from numpy.testing import assert_allclose

from task1_cases import CASES

ATOL = 1e-9


@pytest.mark.parametrize("name, forward, inverse, p_in, predicted, reason", CASES,
                         ids=[f"{c[0]}:{i % 3 + 1}" for i, c in enumerate(CASES)])
def test_transform_point(name, forward, inverse, p_in, predicted, reason):
    computed = forward(p_in)
    assert_allclose(computed, predicted, atol=ATOL, err_msg=reason)
    assert_allclose(inverse(computed), p_in, atol=ATOL, err_msg="inverse: " + reason)
