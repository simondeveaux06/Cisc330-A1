"""
geometry.py — shared geometry utilities for the CT-guided bone biopsy project.
"""

import numpy as np


# Homogeneous building blocks

def make_F(R, t):
    """Pack a 3x3 rotation R and 3-vector t into a 4x4 homogeneous F = T R."""
    F = np.eye(4)
    F[:3, :3] = R
    F[:3, 3] = t
    return F


def invert_F(F):
    """
    Inverse of a rigid homogeneous transform.
    Since R^-1 = R^T for a rotation (Primer Part 2), the inverse of
    F = T R is R^-1 T^-1, which is  [R^T | -R^T t].
    """
    R, t = F[:3, :3], F[:3, 3]
    return make_F(R.T, -R.T @ t)


def transform_point(F, p):
    """Map a POINT: pad with w = 1, so both rotation and translation apply."""
    ph = np.append(np.asarray(p, dtype=float), 1.0)
    return (F @ ph)[:3]


def transform_vector(F, v):
    """
    Map a DIRECTION VECTOR: pad with w = 0, so only rotation applies.
    """
    vh = np.append(np.asarray(v, dtype=float), 0.0)
    w = (F @ vh)[:3]
    return w / np.linalg.norm(w)


# Cartesian frame from three points 

def frame_from_points(A, B, C):
    """
    Build the unique orthonormal body frame of three non-collinear markers,
    and return it as the 4x4 matrix F_h<body> (body coordinates -> home).

    Primer test case: A(0,0,0), B(1,0,0), C(0,1,0) -> i(1,0,0), j(0,1,0), k(0,0,1)
    """
    A, B, C = (np.asarray(x, dtype=float) for x in (A, B, C))
    ab = B - A
    if np.linalg.norm(ab) < 1e-12:
        raise ValueError("markers A and B coincide")
    i = ab / np.linalg.norm(ab)
    n = np.cross(i, C - A)
    if np.linalg.norm(n) < 1e-12:
        raise ValueError("markers are collinear; the frame is undefined")
    k = n / np.linalg.norm(n)
    j = np.cross(k, i)
    O = (A + B + C) / 3.0
    R = np.column_stack([i, j, k])   
    return make_F(R, O)


# Rigid registration


def register(model_pts, observed_pts):
    """
    Find the rigid transform F that maps model points onto observed points:
        observed_i = F @ model_i

    Parameters: two (3, 3) arrays of the three markers, as rows, in matching
    order. Returns F (4x4).
    """
    P = np.asarray(model_pts, dtype=float)
    Q = np.asarray(observed_pts, dtype=float)
    if P.shape != (3, 3) or Q.shape != (3, 3):
        raise ValueError("model_pts and observed_pts must both be (3, 3): three markers as rows")

    F_he = frame_from_points(*P)
    F_hv = frame_from_points(*Q)
    return F_hv @ invert_F(F_he)


# Elementary rotations

def rot_x(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
