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


# Cartesian frame from three points (Primer Part 1)

def frame_from_points(A, B, C):
    """
    Build the unique orthonormal body frame of three non-collinear markers,
    and return it as the 4x4 matrix F_h<body> (body coordinates -> home).

      Step 1, orthonormal base (right-hand rule):
          i = normalize(B - A)            in the ABC plane
          k = normalize(i x (C - A))      perpendicular to the ABC plane
          j = k x i                       in the ABC plane, already unit length
      Step 2, centre: the centroid O = (A + B + C) / 3, the least-squares best
          fitting point, which makes the frame less sensitive to noise in any
          single marker.
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
    R = np.column_stack([i, j, k])   # columns = base vectors (R_hv in the Primer)
    return make_F(R, O)


# Rigid registration

def register(model_pts, observed_pts, method="frame"):
    """
    Find the rigid transform F that maps model points onto observed points:
        observed_i ~= F @ model_i

    method = "frame" 
        Build the body frame from the markers in each observation:
            F_he = frame_from_points(model markers)      (observation e)
            F_hv = frame_from_points(observed markers)   (observation v)
        A point fixed on the body has the same body-frame coordinates in both
        observations, so
            p_v = F_hv @ F_eh @ p_e,     with F_eh = inv(F_he)
        and the registration matrix is F = F_hv @ F_eh.
        Uses exactly three markers, in matching order.

    method = "svd" 
        Least-squares fit over all N >= 3 markers. Gives the same F as the
        frame method on exact data. With noisy data the two differ slightly,
        because the frame method trusts the A->B direction completely,
        while SVD spreads the error over all the markers.

    Returns F (4x4).
    """
    P = np.asarray(model_pts, dtype=float)
    Q = np.asarray(observed_pts, dtype=float)
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("model_pts and observed_pts must both be (N, 3) arrays")

    if method == "frame":
        if P.shape[0] != 3:
            raise ValueError("the frame method uses exactly 3 markers")
        F_he = frame_from_points(*P)
        F_hv = frame_from_points(*Q)
        return F_hv @ invert_F(F_he)

    if method == "svd":
        if P.shape[0] < 3:
            raise ValueError("at least 3 point pairs are needed")
        cp, cq = P.mean(axis=0), Q.mean(axis=0)
        H = (P - cp).T @ (Q - cq)                 # 3x3 cross-covariance
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))    # -1 would mean a reflection
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        return make_F(R, cq - R @ cp)

    raise ValueError(f"unknown method {method!r}")


# Elementary rotations (Primer Part 2), as 3x3; wrap with make_F for 4x4

def rot_x(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
