"""Task 7 test (handout): feed the known marker locations in the Detector frames
into the reconstruction and show that it returns the known marker locations in
the CK frame with (near) zero residual error.
Run with:  python -m pytest -q tests
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import numpy as np
from numpy.testing import assert_allclose

import drr
import reconstruction as R


def test_known_markers_reconstruct_with_zero_rem():
    """Known marker locations in Detector A and B (forward projection of the
    true CK markers) -> reconstruction must give the true CK markers, REM ~ 0."""
    markers_ck = drr.marker_centres_ck()
    uv_A = drr.project_point_to_detector(markers_ck, "A")
    uv_B = drr.project_point_to_detector(markers_ck, "B")
    p_ck, rem = R.reconstruct_point(uv_A, uv_B)
    assert_allclose(p_ck, markers_ck, atol=1e-9)
    assert np.all(rem < 1e-9)
