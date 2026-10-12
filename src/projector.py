"""Q3 CyberKnife perspective forward projector.

CK input points and detector output coordinates are in millimetres.
Detector image pixel coordinates are NOT computed here (see Q4).
The detector plane is w=0 and the X-ray source is (0,0,SDD_MM)
when expressed in the corresponding detector coordinate frame.
"""
import numpy as np
from config import SDD_MM
from transforms import ck_to_detector


def project_ck_to_detector(points_ck_mm, pose):
    """Project CK point(s) onto detector A or B.

    Args:
        points_ck_mm: 3-vector or (N,3) array in CK coordinates (mm).
        pose: 'A' or 'B'.
    Returns:
        Projected detector-plane (u,v) in mm; shape (2,) or (N,2).
    Raises:
        ValueError: invalid point shape or nonfinite point, or if any point
            is at/behind the source plane (no forward-ray intersection).
    """
    points = np.asarray(points_ck_mm, dtype=float)
    if points.ndim == 1 and points.shape == (3,):
        single = True
        points = points[None, :]
    elif points.ndim == 2 and points.shape[1] == 3:
        single = False
    else:
        raise ValueError('Expected CK point shape (3,) or (N,3).')
    if not np.isfinite(points).all():
        raise ValueError('CK points must have finite coordinates.')
    detector_pts = ck_to_detector(points, pose)
    u, v, w = detector_pts[:, 0], detector_pts[:, 1], detector_pts[:, 2]
    # Source=(0,0,SDD), ray=S+t*(P-S), detector plane w=0.
    # Thus t=SDD/(SDD-w), and (u_proj,v_proj)=t*(u,v).
    denominator = SDD_MM - w
    if np.any(denominator <= 1e-9):
        raise ValueError('Point is at or behind the source plane.')
    t = SDD_MM / denominator
    projected_uv = np.column_stack((t * u, t * v))
    return projected_uv[0] if single else projected_uv


def project_both_detectors(points_ck_mm):
    """Return {'A': uv_A, 'B': uv_B} in detector-plane millimetres."""
    return {pose: project_ck_to_detector(points_ck_mm, pose)
            for pose in ('A', 'B')}
