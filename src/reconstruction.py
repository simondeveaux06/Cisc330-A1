"""Task 7: reconstruct a marker in the CK frame from its two X-ray images.

Method (Discrete Tomography lecture, "Reconstruction of point marker from 2 images")
-----------------------------------------------------------------------------------
1. Each image gives the marker's shadow T_d = (u, v) in its Detector frame.
   Transform the shadow (u, v, 0) to the CK frame (Task 1).
2. Back-projection: mathematically "reverse" the X-ray beam. The marker must lie
   somewhere on the line from its shadow to the source:
       L = T_d + t*v,   v = (S - T_d) / |S - T_d|
3. Do this for pose A and pose B, giving two lines in CK.
4. Symbolic intersection (geometry.closest_points_between_lines): the
   midpoint of the common perpendicular is the reconstructed marker; the
   length of the common perpendicular (the shortest distance between the two
   back-projection lines) is the residual error metric, REM.
   With perfect data the two lines meet exactly and REM = 0.

All inputs can be single points (2,) or batches (N, 2): Task 10 reconstructs
hundreds of thousands of perturbed pairs at once.
"""
import numpy as np

import transforms as T
from geometry import normalize, closest_points_between_lines


def back_projection_line(uv_det_mm, pose):
    """Back-projection line(s) of detector point(s) (u, v) mm for `pose`.
    Returns (T_d in CK (N, 3), unit direction toward the source (N, 3))."""
    uv = np.atleast_2d(np.asarray(uv_det_mm, float))
    shadow_ck = T.detector_to_ck(np.c_[uv, np.zeros(len(uv))], pose)
    source_ck = T.source_position_ck(pose)
    return shadow_ck, normalize(source_ck - shadow_ck)


def reconstruct_point(uv_A_mm, uv_B_mm):
    """Reconstruct marker(s) in CK from their shadows on detector A and B.

    uv_A_mm, uv_B_mm: (2,) or (N, 2) detector coordinates (mm) of the SAME
    marker(s) in the two images (correspondence must be known; Task 8).
    Returns (p_ck (N, 3) mm, rem (N,) mm).
    """
    P_A, d_A = back_projection_line(uv_A_mm, "A")
    P_B, d_B = back_projection_line(uv_B_mm, "B")
    midpoint, rem, _, _ = closest_points_between_lines(P_A, d_A, P_B, d_B)
    return midpoint, rem
