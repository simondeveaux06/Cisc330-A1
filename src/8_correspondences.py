"""
Question 8: Marker Correspondences.

The three markers are identical. Therefore, their labels are unknown in the
two detector images. This module determines which Detector B point matches
each Detector A point.

Primary method: combinatorial search using the REM matrix.

First, pair every Detector A point with every Detector B point. Reconstruct
each pair in the CyberKnife frame and calculate its residual error metric
(REM). REM is the distance between the two back-projection lines
(Discrete Tomography, Slide 24). Store the nine values in a 3-by-3 matrix:

    rem_matrix[i, j] = REM(A_i, B_j)

A correct pair should produce a small REM because its two back-projection
lines should meet near the same physical marker
(Discrete Tomography, Slide 28).

A small REM in one row is not enough. The same Detector B point must not be
assigned to more than one Detector A point. Therefore, this method tests all
six one-to-one permutations of the three Detector B points. Each permutation
represents one possible complete correspondence
(Discrete Tomography, Slide 28).

For each permutation, calculate the total score:

    score = REM(A_1, B_p1)
          + REM(A_2, B_p2)
          + REM(A_3, B_p3)

The permutation with the smallest total score is selected as the estimated
correspondence. The method also reports the second-best score and the score
gap. A zero or very small score gap means that two correspondence hypotheses
are almost equally plausible. This may indicate an ambiguous correspondence
(Discrete Tomography, Slides 29-31).

Alternative method: epipolar constraint.

For each Detector A point, create its 3D back-projection line and project that
line onto Detector B. This produces an epipolar line. Measure the 2D
perpendicular distance from each Detector B point to each epipolar line.
Use these distances to form a second cost matrix and determine the
one-to-one correspondence (Discrete Tomography, Slide 27).

All detector coordinates, REM values, and epipolar distances are in
millimetres. Matrix rows follow the input order in Detector A. Matrix columns
follow the input order in Detector B.
"""

from dataclasses import dataclass
from itertools import permutations

import numpy as np

F = __import__('3_forward_projector')
R = __import__('7_reconstruction')
T = __import__('1_transforms')


PERMUTATIONS = tuple(permutations(range(3)))
DEFAULT_ABS_TOL_MM = 1e-12
DEFAULT_REL_TOL = 1e-12


@dataclass(frozen=True)
class RemCorrespondenceResult:
    """Result of the combinatorial REM-matrix method."""

    mapping: tuple
    rem_matrix_mm: np.ndarray
    selected_residuals_mm: np.ndarray
    total_score_mm: float
    second_best_score_mm: float
    score_gap_mm: float


@dataclass(frozen=True)
class EpipolarCorrespondenceResult:
    """Result of the epipolar-distance method."""

    mapping: tuple
    epipolar_distance_matrix_mm: np.ndarray
    selected_distances_mm: np.ndarray
    total_score_mm: float
    second_best_score_mm: float
    score_gap_mm: float


def _three_detector_points(points, name):
    """Return a finite (3, 2) detector-coordinate array in mm."""
    points = np.asarray(points, dtype=float)
    if points.shape != (3, 2):
        raise ValueError(f"{name} must have shape (3, 2); got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain only finite coordinates")
    return points


def _select_one_to_one(cost_matrix_mm, abs_tol_mm, rel_tol):
    """Select one matrix entry per row and per column.

    Each permutation p maps A row i to B column p[i]. The score is
        C(p) = sum(cost_matrix_mm[i, p[i]] for i in range(3)).

    The permutations have lexicographic order. If multiple scores agree with
    the minimum within the given tolerances, select the first permutation.
    Report a zero gap for that numerical tie.
    """
    if not np.isfinite(abs_tol_mm) or abs_tol_mm < 0.0:
        raise ValueError("abs_tol_mm must be finite and non-negative")
    if not np.isfinite(rel_tol) or rel_tol < 0.0:
        raise ValueError("rel_tol must be finite and non-negative")

    rows = np.arange(3)
    scores = np.array([
        np.sum(cost_matrix_mm[rows, permutation])
        for permutation in PERMUTATIONS
    ])
    minimum = np.min(scores)
    tied = np.flatnonzero(np.isclose(scores, minimum,
                                     atol=abs_tol_mm, rtol=rel_tol))

    # PERMUTATIONS has lexicographic order, so the first tied item is stable.
    selected_index = int(tied[0])
    mapping = PERMUTATIONS[selected_index]
    selected = cost_matrix_mm[rows, mapping].copy()
    total = float(scores[selected_index])

    if len(tied) > 1:
        # A zero gap exposes ambiguity at the requested numerical precision.
        second_best = total
        gap = 0.0
    else:
        second_best = float(np.min(np.delete(scores, selected_index)))
        gap = second_best - total
    return mapping, selected, total, second_best, gap


def rem_matrix(uv_A_mm, uv_B_mm):
    """Reconstruct all nine A-B pairs and return their Task 7 REM values."""
    uv_A = _three_detector_points(uv_A_mm, "uv_A_mm")
    uv_B = _three_detector_points(uv_B_mm, "uv_B_mm")

    # Repeat A by row and tile B so reshape keeps rows=A and columns=B.
    pairs_A = np.repeat(uv_A, 3, axis=0)
    pairs_B = np.tile(uv_B, (3, 1))
    _, rem = R.reconstruct_point(pairs_A, pairs_B)
    return rem.reshape(3, 3)


def match_markers_by_rem(uv_A_mm, uv_B_mm,
                         abs_tol_mm=DEFAULT_ABS_TOL_MM,
                         rel_tol=DEFAULT_REL_TOL):
    """Match three unlabeled points with the combinatorial REM-matrix method."""
    matrix = rem_matrix(uv_A_mm, uv_B_mm)
    mapping, selected, total, second_best, gap = _select_one_to_one(
        matrix, abs_tol_mm, rel_tol
    )
    return RemCorrespondenceResult(mapping, matrix, selected, total,
                                   second_best, gap)


def _epipolar_line_on_detector_b(uv_A_mm, line_tol_mm):
    """Return two Detector B points on the epipolar line for one A point.

    The A shadow and source A are two points on the 3-D back-projection line.
    Project both points onto Detector B. Their two projections define the
    epipolar line in Detector B coordinates.
    """
    shadow_A_ck, _ = R.back_projection_line(uv_A_mm, "A")
    source_A_ck = T.source_position_ck("A")
    points_B = F.project_ck_to_detector(
        np.vstack([shadow_A_ck[0], source_A_ck]), "B"
    )
    if np.linalg.norm(points_B[1] - points_B[0]) <= line_tol_mm:
        raise ValueError("Epipolar line is degenerate in Detector B")
    return points_B


def epipolar_distance_matrix(uv_A_mm, uv_B_mm,
                             line_tol_mm=DEFAULT_ABS_TOL_MM):
    """Return B-point distances from the three A epipolar lines, in mm.

    For projected line points q0 and q1, let e = q1 - q0. For candidate b,
        distance = abs(e_u*(b-q0)_v - e_v*(b-q0)_u) / norm(e).
    This is the perpendicular distance from b to the 2-D line.
    """
    uv_A = _three_detector_points(uv_A_mm, "uv_A_mm")
    uv_B = _three_detector_points(uv_B_mm, "uv_B_mm")
    if not np.isfinite(line_tol_mm) or line_tol_mm < 0.0:
        raise ValueError("line_tol_mm must be finite and non-negative")

    matrix = np.empty((3, 3), dtype=float)
    for row, point_A in enumerate(uv_A):
        q0, q1 = _epipolar_line_on_detector_b(point_A, line_tol_mm)
        direction = q1 - q0
        offset = uv_B - q0
        # The 2-D cross product is the signed parallelogram area.
        cross = direction[0] * offset[:, 1] - direction[1] * offset[:, 0]
        matrix[row] = np.abs(cross) / np.linalg.norm(direction)
    return matrix


def match_markers_by_epipolar(uv_A_mm, uv_B_mm,
                              abs_tol_mm=DEFAULT_ABS_TOL_MM,
                              rel_tol=DEFAULT_REL_TOL):
    """Match three unlabeled points with epipolar distances on Detector B."""
    matrix = epipolar_distance_matrix(uv_A_mm, uv_B_mm)
    mapping, selected, total, second_best, gap = _select_one_to_one(
        matrix, abs_tol_mm, rel_tol
    )
    return EpipolarCorrespondenceResult(mapping, matrix, selected, total,
                                        second_best, gap)
