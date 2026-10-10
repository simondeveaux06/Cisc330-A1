"""

Conventions (Math Primer Parts 2-3)
-----------------------------------
 Every F is a 4x4 homogeneous matrix; points are padded with a trailing 1.

Frames
------
MD   Vertebra model frame (mm). Coincides with CK for this set-up (config).
CK   CyberKnife frame (mm). Origin = isocenter. +x patient right,
     +y anterior (ceiling), +z superior. Patient supine.
Det  Detector frame of pose A or B (mm). Origin = detector centre.
       u = +z_CK                         (always, handout p.3)
       v = Rz(angle) * x_CK
       w = Rz(angle) * y_CK              (points to the X-ray source)
     (u, v, w) is right-handed: u x v = w.
     In the zero-angle home pose v = +x_CK and w = +y_CK (handout p.2).
     The detector centre is ISO_TO_DET_MM behind the isocenter along -w,
     and the source is SAD_MM in front of it along +w.
Img  Image frame of pose A or B, in pixels. Axes parallel to the detector:
       row index  <->  u     (superior direction)
       col index  <->  v
"""
import numpy as np
from config import (ISO_TO_DET_MM, SAD_MM, POSE_ANGLE_DEG, N_PIXELS,
                    PIXEL_PITCH_MM, MD_ORIGIN_IN_CK_MM, MD_AXES_IN_CK,
                    ON_DETECTOR_PLANE_TOL_MM)


# Generic homogeneous-matrix utilities 
# ===========================================================================
def translation_matrix(d_mm):
    """4x4 homogeneous translation by vector d_mm (3,)."""
    T = np.eye(4)
    T[:3, 3] = d_mm
    return T


def scale_matrix(s):
    """4x4 homogeneous isotropic scaling by factor s."""
    S = np.eye(4)
    S[:3, :3] *= s
    return S


def rotation_matrix_from_base_rows(b1, b2, b3):
    R = np.eye(4)
    R[:3, :3] = np.vstack([b1, b2, b3])
    return R


def frame_from_home(origin_in_home, b1, b2, b3):
    return rotation_matrix_from_base_rows(b1, b2, b3) @ translation_matrix(-np.asarray(origin_in_home))


def home_from_frame(origin_in_home, b1, b2, b3):
    return translation_matrix(origin_in_home) @ rotation_matrix_from_base_rows(b1, b2, b3).T


def transform_points(F, pts):
    """Apply 4x4 transform F to one point (3,) or many points (N, 3).
    Returns an array with the same shape as the input."""
    pts = np.asarray(pts, float)
    pts2d = np.atleast_2d(pts)
    hom = np.hstack([pts2d, np.ones((len(pts2d), 1))])
    out = (hom @ F.T)[:, :3]           # row-vector form of F @ p for each point
    return out.reshape(pts.shape)


def _check_pose(pose):
    if pose not in POSE_ANGLE_DEG:
        raise ValueError(f"Unknown pose {pose!r}; expected one of {tuple(POSE_ANGLE_DEG)}")


# MD <-> CK
# ===========================================================================
def F_ck_from_md():
    return home_from_frame(MD_ORIGIN_IN_CK_MM, *MD_AXES_IN_CK)


def F_md_from_ck():
    return frame_from_home(MD_ORIGIN_IN_CK_MM, *MD_AXES_IN_CK)


# CK <-> Detector (pose A or B)
# ===========================================================================
def detector_axes_in_ck(pose):
    """Unit base vectors (u, v, w) of the pose's detector, in CK coordinates."""
    _check_pose(pose)
    t = np.radians(POSE_ANGLE_DEG[pose])
    u = np.array([0.0, 0.0, 1.0])
    v = np.array([np.cos(t), np.sin(t), 0.0])     
    w = np.array([-np.sin(t), np.cos(t), 0.0])    
    return u, v, w


def detector_origin_in_ck(pose):
    return -ISO_TO_DET_MM * detector_axes_in_ck(pose)[2]


def source_position_ck(pose):
    return SAD_MM * detector_axes_in_ck(pose)[2]


def F_det_from_ck(pose):
    return frame_from_home(detector_origin_in_ck(pose), *detector_axes_in_ck(pose))


def F_ck_from_det(pose):
    return home_from_frame(detector_origin_in_ck(pose), *detector_axes_in_ck(pose))


# Detector <-> Image (pose A or B)
# ===========================================================================
# Both detectors are identical, so the two maps are numerically the same

def F_img_from_det(pose):
    """F_Img<-Det: Step 1 scale mm -> pixels, Step 2 shift the origin from the
    detector centre to the corner pixel centre (Discrete Tomography slides).
    The third coordinate is scaled too, so a point on the plane keeps 0."""
    _check_pose(pose)
    shift = translation_matrix([_PIXEL_CENTRE_SHIFT, _PIXEL_CENTRE_SHIFT, 0.0])
    return shift @ scale_matrix(1.0 / PIXEL_PITCH_MM)


def F_det_from_img(pose):
    """F_Det<-Img, the inverse: undo the shift, then undo the scale."""
    _check_pose(pose)
    unshift = translation_matrix([-_PIXEL_CENTRE_SHIFT, -_PIXEL_CENTRE_SHIFT, 0.0])
    return scale_matrix(PIXEL_PITCH_MM) @ unshift


def _as_detector_plane_points(p_det_mm):
    """Accept (u, v) or (u, v, w) points; return (u, v, 0) after checking w ~ 0."""
    p = np.asarray(p_det_mm, float)
    if p.shape[-1] == 2:
        return np.concatenate([p, np.zeros(p.shape[:-1] + (1,))], axis=-1)
    if np.any(np.abs(p[..., 2]) > ON_DETECTOR_PLANE_TOL_MM):
        raise ValueError("Point is not on the detector plane (w != 0). "
                         "Forward-project it onto the detector first.")
    return p


# Point-level wrapper
# ===========================================================================
def md_to_ck(p_md_mm):
    return transform_points(F_ck_from_md(), p_md_mm)


def ck_to_md(p_ck_mm):
    return transform_points(F_md_from_ck(), p_ck_mm)


def ck_to_detector(p_ck_mm, pose):
    return transform_points(F_det_from_ck(pose), p_ck_mm)


def detector_to_ck(p_det_mm, pose):
    return transform_points(F_ck_from_det(pose), p_det_mm)


def detector_to_image(p_det_mm, pose):
    """Detector-plane point(s), (u, v) or (u, v, 0) in mm -> (row, col) pixels."""
    p3 = _as_detector_plane_points(p_det_mm)
    return transform_points(F_img_from_det(pose), p3)[..., :2]


def image_to_detector(rc_pix, pose):
    """(row, col) pixel coordinates -> detector-plane point (u, v, 0) in mm."""
    rc = np.asarray(rc_pix, float)
    rc3 = np.concatenate([rc, np.zeros(rc.shape[:-1] + (1,))], axis=-1)
    return transform_points(F_det_from_img(pose), rc3)
