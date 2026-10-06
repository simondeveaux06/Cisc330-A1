
"""Frame transforms as 4x4 homogeneous matrices (course convention).

Column-vector-on-the-right: p_to = F_{to<-from} @ p_from.
CK: +x patient right, +y anterior, +z superior, origin = isocenter (mm).
MD frame coincides with CK (identity).
Detector frame (u, v, w): u = +z_CK, v = x_CK rotated by pose angle about z,
w = y_CK rotated by pose angle (points to the source).
Origin = detector center, (SDD - SAD) behind the isocenter along -w.
Image frame: axes aligned with detector (row <-> u, col <-> v), pixel centers
at integers, (0,0) = center of corner pixel (u = v = -100 mm + half pixel).

From math primer transformations
"""
import numpy as np
from config import SDD_MM, SAD_MM, POSE_ANGLE_DEG, DETECTOR_SIZE_MM, PIXEL_PITCH_MM

ISO_TO_DET_MM = SDD_MM - SAD_MM

def translation_matrix(d_mm):
    T = np.eye(4); T[:3, 3] = d_mm
    return T

def rotation_matrix_from_base_rows(u, v, w):
    """R_{frame<-home}: the frame's base vectors (in home coords) as ROWS."""
    R = np.eye(4); R[:3, :3] = np.vstack([u, v, w])
    return R

def detector_axes_in_ck(pose):
    t = np.radians(POSE_ANGLE_DEG[pose])
    return (np.array([0., 0., 1.]),
            np.array([np.cos(t), np.sin(t), 0.]),
            np.array([-np.sin(t), np.cos(t), 0.]))

def detector_origin_in_ck(pose):
    return -ISO_TO_DET_MM * detector_axes_in_ck(pose)[2]

def F_det_from_ck(pose):
    """F_{Det<-CK} = R_{Det<-CK} T(-O_det)."""
    R = rotation_matrix_from_base_rows(*detector_axes_in_ck(pose))
    return R @ translation_matrix(-detector_origin_in_ck(pose))

def F_ck_from_det(pose):
    """F_{CK<-Det} = T(O_det) R_{Det<-CK}^T (inverse of the above)."""
    R = rotation_matrix_from_base_rows(*detector_axes_in_ck(pose))
    return translation_matrix(detector_origin_in_ck(pose)) @ R.T

def transform_points(F, pts):
    pts = np.atleast_2d(np.asarray(pts, float))
    hom = np.hstack([pts, np.ones((len(pts), 1))])
    return (hom @ F.T)[:, :3]

def _apply(F, p):
    p = np.asarray(p, float)
    return transform_points(F, p).reshape(p.shape)

# Wrappers (same names as before, so existing tests still work)
def md_to_ck(p_md): return _apply(np.eye(4), p_md)
def ck_to_md(p_ck): return _apply(np.eye(4), p_ck)
def ck_to_detector(p_ck, pose): return _apply(F_det_from_ck(pose), p_ck)
def detector_to_ck(p_det, pose): return _apply(F_ck_from_det(pose), p_det)
def source_position_ck(pose): return SAD_MM * detector_axes_in_ck(pose)[2]

def detector_to_image(uv_mm):
    return (np.asarray(uv_mm, float) + DETECTOR_SIZE_MM / 2) / PIXEL_PITCH_MM - 0.5

def image_to_detector(rc_pix):
    return (np.asarray(rc_pix, float) + 0.5) * PIXEL_PITCH_MM - DETECTOR_SIZE_MM / 2
