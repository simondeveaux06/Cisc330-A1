"""Question 9: Detect ambiguous marker correspondences.

The REM method reuses the all-pairs matrix from Task 8. A complete
permutation is valid only when each selected REM is at or below the valid REM
tolerance. Two valid permutations are ambiguous only when their total scores
are equal within the score tolerance.

The epipolar method also reuses the Task 8 matrix. It thresholds each matrix
entry and counts the complete one-to-one matchings in the resulting Boolean
matrix.

All detector coordinates, REM values, and epipolar distances are in mm.
"""

from dataclasses import dataclass

import numpy as np

C = __import__('8_correspondences')


DEFAULT_VALID_REM_TOL_MM = 1e-6
DEFAULT_SCORE_TIE_TOL_MM = 1e-9
DEFAULT_LOW_CONFIDENCE_GAP_MM = 1e-3
DEFAULT_EPIPOLAR_TOL_MM = 1e-6

# The first two markers are in the CK xy-plane. Any crossed pairing of their
# rays also meets in that plane. The third marker has a different z value, so
# it has one valid partner and leaves two valid complete permutations.
ENGINEERED_AMBIGUOUS_MARKERS_CK_MM = np.array([
    [-30.0, 0.0, 0.0],
    [30.0, 0.0, 0.0],
    [5.0, 25.0, 20.0],
])


@dataclass(frozen=True)
class PermutationScore:
    """Score and selected matrix values for one A-to-B permutation."""

    permutation: tuple
    selected_values_mm: np.ndarray
    total_score_mm: float
    is_valid: bool


@dataclass(frozen=True)
class RemAmbiguityResult:
    """Diagnostics from the REM-based ambiguity detector.

    The best and second-best fields rank all six raw permutation scores.
    ``threshold_valid_permutations`` passed the per-pair REM limit.
    ``valid_permutations`` also passed the score-tie test against the best
    threshold-valid permutation.
    """

    rem_matrix_mm: np.ndarray
    permutation_scores: tuple
    threshold_valid_permutations: tuple
    valid_permutations: tuple
    indistinguishable_permutations: tuple
    classification: str
    confidence: str
    best_permutation: tuple
    best_score_mm: float
    second_best_score_mm: float
    score_gap_mm: float
    valid_rem_tol_mm: float
    score_tie_tol_mm: float
    low_confidence_gap_mm: float


@dataclass(frozen=True)
class EpipolarAmbiguityResult:
    """Diagnostics from the thresholded epipolar-distance method."""

    epipolar_distance_matrix_mm: np.ndarray
    valid_pairings: np.ndarray
    valid_permutations: tuple
    classification: str
    distance_tol_mm: float


def _non_negative_tolerance(value, name):
    """Return a finite, non-negative tolerance as a float."""
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _permutation_diagnostics(matrix_mm, valid_value_tol_mm):
    """Evaluate every one-to-one permutation of a 3-by-3 matrix."""
    rows = np.arange(3)
    diagnostics = []
    for permutation in C.PERMUTATIONS:
        selected = matrix_mm[rows, permutation].copy()
        diagnostics.append(PermutationScore(
            permutation=permutation,
            selected_values_mm=selected,
            total_score_mm=float(np.sum(selected)),
            is_valid=bool(np.all(selected <= valid_value_tol_mm)),
        ))
    return tuple(diagnostics)


def detect_ambiguity_by_rem(
        uv_A_mm, uv_B_mm,
        valid_rem_tol_mm=DEFAULT_VALID_REM_TOL_MM,
        score_tie_tol_mm=DEFAULT_SCORE_TIE_TOL_MM,
        low_confidence_gap_mm=DEFAULT_LOW_CONFIDENCE_GAP_MM):
    """Classify a complete correspondence with REM values.

    ``unique`` means that one valid permutation is distinguishable from all
    other valid permutations. ``ambiguous`` means that at least two valid
    permutations have scores within ``score_tie_tol_mm`` of the best valid
    score. ``unresolved`` means that no complete permutation passes the
    per-pair REM threshold.

    A unique result has low confidence when its next valid score is outside
    the tie tolerance but no farther away than ``low_confidence_gap_mm``.
    """
    valid_tol = _non_negative_tolerance(valid_rem_tol_mm,
                                        "valid_rem_tol_mm")
    tie_tol = _non_negative_tolerance(score_tie_tol_mm,
                                      "score_tie_tol_mm")
    confidence_tol = _non_negative_tolerance(
        low_confidence_gap_mm, "low_confidence_gap_mm"
    )
    if confidence_tol < tie_tol:
        raise ValueError("low_confidence_gap_mm must be at least score_tie_tol_mm")

    matrix = C.rem_matrix(uv_A_mm, uv_B_mm)
    scores = _permutation_diagnostics(matrix, valid_tol)
    all_ranked = sorted(scores, key=lambda item: (
        item.total_score_mm, item.permutation
    ))
    overall_best = all_ranked[0]
    overall_second = all_ranked[1]
    overall_gap = overall_second.total_score_mm - overall_best.total_score_mm
    valid = tuple(score for score in scores if score.is_valid)

    if not valid:
        return RemAmbiguityResult(
            rem_matrix_mm=matrix,
            permutation_scores=scores,
            threshold_valid_permutations=(),
            valid_permutations=(),
            indistinguishable_permutations=(),
            classification="unresolved",
            confidence="none",
            best_permutation=overall_best.permutation,
            best_score_mm=overall_best.total_score_mm,
            second_best_score_mm=overall_second.total_score_mm,
            score_gap_mm=overall_gap,
            valid_rem_tol_mm=valid_tol,
            score_tie_tol_mm=tie_tol,
            low_confidence_gap_mm=confidence_tol,
        )

    ranked = sorted(valid, key=lambda item: (
        item.total_score_mm, item.permutation
    ))
    best = ranked[0]
    tied = tuple(
        item.permutation for item in ranked
        if abs(item.total_score_mm - best.total_score_mm) <= tie_tol
    )
    valid_second = ranked[1].total_score_mm if len(ranked) > 1 else None
    valid_gap = (valid_second - best.total_score_mm
                 if valid_second is not None else None)

    if len(tied) > 1:
        classification = "ambiguous"
        confidence = "indistinguishable"
    else:
        classification = "unique"
        confidence = ("low" if valid_gap is not None
                      and valid_gap <= confidence_tol
                      else "high")

    return RemAmbiguityResult(
        rem_matrix_mm=matrix,
        permutation_scores=scores,
        threshold_valid_permutations=tuple(
            item.permutation for item in valid
        ),
        valid_permutations=tied,
        indistinguishable_permutations=tied,
        classification=classification,
        confidence=confidence,
        best_permutation=overall_best.permutation,
        best_score_mm=overall_best.total_score_mm,
        second_best_score_mm=overall_second.total_score_mm,
        score_gap_mm=overall_gap,
        valid_rem_tol_mm=valid_tol,
        score_tie_tol_mm=tie_tol,
        low_confidence_gap_mm=confidence_tol,
    )


def detect_ambiguity_by_epipolar(
        uv_A_mm, uv_B_mm,
        distance_tol_mm=DEFAULT_EPIPOLAR_TOL_MM):
    """Threshold epipolar distances and count complete matchings."""
    distance_tol = _non_negative_tolerance(distance_tol_mm,
                                           "distance_tol_mm")
    matrix = C.epipolar_distance_matrix(uv_A_mm, uv_B_mm)
    valid_pairings = matrix <= distance_tol
    rows = np.arange(3)
    valid_permutations = tuple(
        permutation for permutation in C.PERMUTATIONS
        if np.all(valid_pairings[rows, permutation])
    )

    if len(valid_permutations) == 0:
        classification = "unresolved"
    elif len(valid_permutations) == 1:
        classification = "unique"
    else:
        classification = "ambiguous"

    return EpipolarAmbiguityResult(
        epipolar_distance_matrix_mm=matrix,
        valid_pairings=valid_pairings,
        valid_permutations=valid_permutations,
        classification=classification,
        distance_tol_mm=distance_tol,
    )
