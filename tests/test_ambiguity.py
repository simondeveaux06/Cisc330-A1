"""Tests for Task 9 correspondence ambiguity detection."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import numpy as np

A = __import__('9_ambiguity')
D = __import__('5_drr')
F = __import__('3_forward_projector')


def _project(markers_ck):
    """Project CK points with the existing Task 3 projector."""
    return (F.project_ck_to_detector(markers_ck, "A"),
            F.project_ck_to_detector(markers_ck, "B"))


def _known_projections():
    return _project(D.marker_centres_ck())


def _ambiguous_projections():
    return _project(A.ENGINEERED_AMBIGUOUS_MARKERS_CK_MM)


def test_unique_correspondence_has_one_valid_complete_assignment():
    """The assignment markers have one complete near-zero REM matching."""
    uv_A, uv_B = _known_projections()
    result = A.detect_ambiguity_by_rem(uv_A, uv_B)

    assert result.classification == "unique"
    assert result.confidence == "high"
    assert result.valid_permutations == ((0, 1, 2),)
    assert result.best_permutation == (0, 1, 2)
    assert len(result.permutation_scores) == 6


def test_engineered_same_z_plane_case_is_ambiguous():
    """Two markers at CK z=0 permit both direct and crossed pairings."""
    uv_A, uv_B = _ambiguous_projections()
    result = A.detect_ambiguity_by_rem(uv_A, uv_B)

    assert result.classification == "ambiguous"
    assert result.valid_permutations == ((0, 1, 2), (1, 0, 2))
    assert result.indistinguishable_permutations == ((0, 1, 2), (1, 0, 2))
    assert result.score_gap_mm <= result.score_tie_tol_mm


def test_shifted_detector_points_are_unresolved():
    """A 1 mm u error makes every complete assignment fail its REM limit."""
    uv_A, uv_B = _known_projections()
    uv_B[:, 0] += 1.0
    result = A.detect_ambiguity_by_rem(uv_A, uv_B)

    assert result.classification == "unresolved"
    assert result.valid_permutations == ()
    assert np.isfinite(result.best_score_mm)
    assert result.second_best_score_mm > result.best_score_mm


def test_valid_rem_tolerance_changes_acceptance_not_geometry():
    """A small localization error is rejected or accepted by an explicit limit."""
    uv_A, uv_B = _known_projections()
    uv_B[0, 0] += 1e-4

    strict = A.detect_ambiguity_by_rem(uv_A, uv_B,
                                       valid_rem_tol_mm=1e-6)
    relaxed = A.detect_ambiguity_by_rem(uv_A, uv_B,
                                        valid_rem_tol_mm=1e-3)

    assert strict.classification == "unresolved"
    assert relaxed.classification == "unique"
    assert relaxed.best_permutation == (0, 1, 2)


def test_near_tie_is_low_confidence_but_not_ambiguous():
    """Two close valid scores stay distinct when their gap exceeds tie tolerance."""
    markers_ck = A.ENGINEERED_AMBIGUOUS_MARKERS_CK_MM.copy()
    markers_ck[1, 2] = 1e-4
    uv_A, uv_B = _project(markers_ck)

    result = A.detect_ambiguity_by_rem(
        uv_A, uv_B,
        valid_rem_tol_mm=2e-4,
        score_tie_tol_mm=1e-5,
        low_confidence_gap_mm=1e-3,
    )
    relaxed_tie = A.detect_ambiguity_by_rem(
        uv_A, uv_B,
        valid_rem_tol_mm=2e-4,
        score_tie_tol_mm=3e-4,
        low_confidence_gap_mm=1e-3,
    )

    assert result.classification == "unique"
    assert result.confidence == "low"
    assert result.threshold_valid_permutations == (
        (0, 1, 2), (1, 0, 2)
    )
    assert result.valid_permutations == ((0, 1, 2),)
    assert result.score_tie_tol_mm < result.score_gap_mm < 1e-3
    assert relaxed_tie.classification == "ambiguous"


def test_rem_and_epipolar_methods_agree_on_all_three_classes():
    """Both independent methods classify the same three test situations."""
    cases = []
    cases.append((*_known_projections(), "unique"))
    cases.append((*_ambiguous_projections(), "ambiguous"))
    uv_A, uv_B = _known_projections()
    uv_B[:, 0] += 1.0
    cases.append((uv_A, uv_B, "unresolved"))

    for points_A, points_B, expected in cases:
        rem = A.detect_ambiguity_by_rem(points_A, points_B)
        epi = A.detect_ambiguity_by_epipolar(points_A, points_B)
        assert rem.classification == epi.classification == expected
        assert rem.valid_permutations == epi.valid_permutations
