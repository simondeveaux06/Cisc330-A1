"""Reusable vector-geometry utilities (Math Primer Part 1), vectorised with numpy.

Lines are written L = P + t*v with v a unit vector, so the parameter t is the
distance (mm) from the fixed point P along the line.
"""
import numpy as np


def normalize(vectors):
    vectors = np.asarray(vectors, float)
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def line_sphere_intersection_params(P, v, centre, radius):

    P = np.atleast_2d(np.asarray(P, float))
    v = np.atleast_2d(np.asarray(v, float))
    c = P - np.asarray(centre, float)
    half_b = np.einsum("ij,ij->i", v, c)            
    c_term = np.einsum("ij,ij->i", c, c) - radius ** 2
    quarter_disc = half_b ** 2 - c_term                 
    hits = quarter_disc > 0.0                       
    root = np.sqrt(np.where(hits, quarter_disc, np.nan))
    return -half_b - root, -half_b + root, hits


def chord_length_through_sphere(P, v, centre, radius):
    """Length (mm) of each line's segment inside the sphere; 0 for misses.

    Uses the two intersection roots: chord = t_exit - t_entry = 2*sqrt(disc/4).
    Assumes the line starts outside the sphere (true for our X-ray source).
    """
    t_entry, t_exit, hits = line_sphere_intersection_params(P, v, centre, radius)
    return np.where(hits, t_exit - t_entry, 0.0)


def line_plane_intersection(P, v, plane_point, plane_normal):
    """Intersection of line(s) P + t*v with the plane (plane_point, plane_normal):
    t = ((A - P) . n) / (v . n)   (Primer 1, line-plane intersection)."""
    P = np.atleast_2d(np.asarray(P, float))
    v = np.atleast_2d(np.asarray(v, float))
    n = np.asarray(plane_normal, float)
    t = ((np.asarray(plane_point, float) - P) @ n) / (v @ n)
    return P + t[:, None] * v


def closest_points_between_lines(P1, v1, P2, v2, parallel_tol=1e-12):

    P1, v1, P2, v2 = (np.atleast_2d(np.asarray(a, float)) for a in (P1, v1, P2, v2))
    P1, v1, P2, v2 = np.broadcast_arrays(P1, v1, P2, v2)
    v3 = np.cross(v1, v2)
    sin_angle = np.linalg.norm(v3, axis=-1)
    if np.any(sin_angle < parallel_tol):
        raise ValueError("Lines are (nearly) parallel: no unique closest points")
    v3 = v3 / sin_angle[:, None]
    A = np.stack([-v1, v2, v3], axis=-1)                    # (N, 3, 3), columns -v1, v2, v3
    t = np.linalg.solve(A, (P1 - P2)[..., None])[..., 0]    # (N, 3): t1, t2, t3
    L1 = P1 + t[:, :1] * v1
    L2 = P2 + t[:, 1:2] * v2
    return (L1 + L2) / 2, np.abs(t[:, 2]), L1, L2
