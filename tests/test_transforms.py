
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
import numpy as np
from numpy.testing import assert_allclose
import transforms as T

S = np.sqrt(0.5)

def test_md_ck_identity():
    p = [-19, 28, 11]
    assert_allclose(T.md_to_ck(p), p)
    assert_allclose(T.ck_to_md(T.md_to_ck(p)), p)

def test_isocenter_maps_to_w_1000_both_poses():
    for pose in "AB":
        assert_allclose(T.ck_to_detector([0, 0, 0], pose), [0, 0, 1000], atol=1e-9)

def test_source_maps_to_w_equals_SDD():
    for pose in "AB":
        src = T.source_position_ck(pose)
        assert_allclose(T.ck_to_detector(src, pose), [0, 0, 2000], atol=1e-9)

def test_source_A_position():
    assert_allclose(T.source_position_ck("A"), [-1000 * S, 1000 * S, 0], atol=1e-9)

def test_z_axis_point_maps_to_u():
    for pose in "AB":
        assert_allclose(T.ck_to_detector([0, 0, 50], pose), [50, 0, 1000], atol=1e-9)

def test_x_axis_point_pose_A_and_B():
    assert_allclose(T.ck_to_detector([100, 0, 0], "A"), [0, 100 * S, 1000 - 100 * S], atol=1e-9)
    assert_allclose(T.ck_to_detector([100, 0, 0], "B"), [0, 100 * S, 1000 + 100 * S], atol=1e-9)

def test_roundtrip_markers():
    from config import MARKERS_MD_MM
    for pose in "AB":
        d = T.ck_to_detector(MARKERS_MD_MM, pose)
        assert_allclose(T.detector_to_ck(d, pose), MARKERS_MD_MM, atol=1e-9)

def test_image_frame():
    assert_allclose(T.detector_to_image([0, 0]), [999.5, 999.5])
    assert_allclose(T.detector_to_image([-99.95, -99.95]), [0, 0], atol=1e-9)
    assert_allclose(T.image_to_detector(T.detector_to_image([12.3, -45.6])), [12.3, -45.6])


def test_homogeneous_matrices_are_inverses():
    for pose in "AB":
        prod = T.F_det_from_ck(pose) @ T.F_ck_from_det(pose)
        assert_allclose(prod, np.eye(4), atol=1e-9)
