"""
test_registration.py — ground-truth tests for the registration module.


Tool model (tool frame, mm):
    M1 = (2, 18, 0)   M2 = (8, 18, 0)   M3 = (2, 24, 0)
    P  = (0, 0, 0)    v  = (0, -1, 0)

Run:  python test_registration.py
"""

import numpy as np
from registration import register_tool, M_TOOL, P_TOOL, V_TOOL
from geometry import rot_x, rot_y, rot_z, make_F, frame_from_points, transform_point

TOL = 1e-9  # numerical-precision tolerance 
np.set_printoptions(suppress=True)


def check(name, markers_ct, p_expected, v_expected, F_expected):
    p_ct, v_ct, F = register_tool(M_TOOL, P_TOOL, V_TOOL, markers_ct)
    mapped = np.array([transform_point(F, m) for m in M_TOOL])
    errs = {
        "tip error (mm)":  np.linalg.norm(p_ct - p_expected),
        "axis error":      np.linalg.norm(v_ct - v_expected),
        "F error":         np.linalg.norm(F - F_expected),
        "marker RMS (mm)": np.sqrt(np.mean(np.sum((mapped - markers_ct) ** 2, axis=1))),
    }
    ok = all(e < TOL for e in errs.values())
    print(f"\n=== {name} ===")
    print(f"  M_ct = {markers_ct.tolist()}")
    print(f"  expected Pct = {p_expected},  computed Pct = {np.round(p_ct, 9) + 0.0}")
    print(f"  expected vct = {v_expected},  computed vct = {np.round(v_ct, 9) + 0.0}")
    for k, e in errs.items():
        print(f"  {k:<16} {e:.1e}")
    print("  PASS" if ok else "  FAIL")
    return ok



# Test 1: PURE TRANSLATION by t = (10, 20, 30), R = I
#
#   On paper: every point shifts by t; directions are unchanged.
#     M1ct = (2,18,0)+(10,20,30) = (12, 38, 30)
#     M2ct = (8,18,0)+(10,20,30) = (18, 38, 30)
#     M3ct = (2,24,0)+(10,20,30) = (12, 44, 30)
#     Pct  = (0,0,0) +(10,20,30) = (10, 20, 30)
#     vct  = (0,-1,0)    translation does not affect a direction (w = 0)

def test_pure_translation():
    markers_ct = np.array([[12, 38, 30], [18, 38, 30], [12, 44, 30]], float)
    return check("Test 1: pure translation, t = (10, 20, 30)", markers_ct,
                 p_expected=np.array([10, 20, 30.0]),
                 v_expected=np.array([0, -1, 0.0]),
                 F_expected=make_F(np.eye(3), [10, 20, 30]))



# Test 2: PURE ROTATION of 90 deg about z, t = 0
#
#   Rz(90) = [[0,-1,0],[1,0,0],[0,0,1]]   maps (x, y, z) -> (-y, x, z)
#     M1ct = (-18,  2, 0)
#     M2ct = (-18,  8, 0)
#     M3ct = (-24,  2, 0)
#     Pct  = (0, 0, 0)  
#     vct  = (1, 0, 0)     (0,-1,0) -> (-(-1), 0, 0)

def test_pure_rotation():
    markers_ct = np.array([[-18, 2, 0], [-18, 8, 0], [-24, 2, 0]], float)
    return check("Test 2: pure rotation, 90 deg about z", markers_ct,
                 p_expected=np.array([0, 0, 0.0]),
                 v_expected=np.array([1, 0, 0.0]),
                 F_expected=make_F(rot_z(90), [0, 0, 0]))



# Test 3: ROTATION + TRANSLATION: 90 deg about x, then t = (100, -50, 200)
#   (F = T R: rotate first, then translate)
#
#   Rx(90) = [[1,0,0],[0,0,-1],[0,1,0]]   maps (x, y, z) -> (x, -z, y)
#     M1: (2,18,0) -> (2,0,18) -> +t -> (102, -50, 218)
#     M2: (8,18,0) -> (8,0,18) -> +t -> (108, -50, 218)
#     M3: (2,24,0) -> (2,0,24) -> +t -> (102, -50, 224)
#     Pct = (0,0,0) -> (0,0,0) -> +t -> (100, -50, 200)
#     vct = (0,-1,0) -> (0, 0, -1)    rotation only, no +t 

def test_rotation_and_translation():
    markers_ct = np.array([[102, -50, 218], [108, -50, 218], [102, -50, 224]], float)
    return check("Test 3: 90 deg about x, then t = (100, -50, 200)", markers_ct,
                 p_expected=np.array([100, -50, 200.0]),
                 v_expected=np.array([0, 0, -1.0]),
                 F_expected=make_F(rot_x(90), [100, -50, 200]))



if __name__ == "__main__":
    results = [test_pure_translation(), test_pure_rotation(),
               test_rotation_and_translation()]
    print(f"\n{sum(results)}/{len(results)} tests passed")
