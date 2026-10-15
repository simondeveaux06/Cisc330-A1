"""Tests for Task 8 marker correspondences."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import numpy as np

C = __import__('8_correspondences')
D = __import__('5_drr')


def _known_projections():
    markers_ck = D.marker_centres_ck()
    return (D.project_point_to_detector(markers_ck, "A"),
            D.project_point_to_detector(markers_ck, "B"))


def test_normal_projections_recover_physical_correspondences_and_zero_rem():
    """Keep the normal order and verify identity matching with near-zero REM."""
    uv_A, uv_B = _known_projections()
    result = C.match_markers_by_rem(uv_A, uv_B)

    assert result.mapping == (0, 1, 2)
    assert result.rem_matrix_mm.shape == (3, 3)
    assert np.all(result.selected_residuals_mm < 1e-9)
    assert result.total_score_mm < 1e-9
    assert result.second_best_score_mm > result.total_score_mm
    assert result.score_gap_mm > 0.0


def test_shuffled_inputs_recover_physical_correspondences():
    """Shuffle both views and verify that the physical markers still match."""
    uv_A, uv_B = _known_projections()
    order_A = np.array([2, 0, 1])
    order_B = np.array([1, 2, 0])

    result = C.match_markers_by_rem(uv_A[order_A], uv_B[order_B])

    assert result.mapping == (1, 2, 0)
    assert np.all(result.selected_residuals_mm < 1e-9)


def test_swapping_two_detector_b_markers_recovers_the_swap():
    """Swap M2 and M3 in Detector B and verify the recovered B indices."""
    uv_A, uv_B = _known_projections()
    order_B = np.array([0, 2, 1])

    result = C.match_markers_by_rem(uv_A, uv_B[order_B])

    assert result.mapping == (0, 2, 1)
    assert np.all(result.selected_residuals_mm < 1e-9)


def test_epipolar_and_rem_methods_agree_after_both_inputs_are_shuffled():
    """Verify that the REM and epipolar methods agree after both shuffles."""
    uv_A, uv_B = _known_projections()
    order_A = np.array([2, 0, 1])
    order_B = np.array([1, 2, 0])
    shuffled_A, shuffled_B = uv_A[order_A], uv_B[order_B]

    rem_result = C.match_markers_by_rem(shuffled_A, shuffled_B)
    epi_result = C.match_markers_by_epipolar(shuffled_A, shuffled_B)

    assert epi_result.mapping == rem_result.mapping == (1, 2, 0)
    assert epi_result.epipolar_distance_matrix_mm.shape == (3, 3)
    assert np.all(epi_result.selected_distances_mm < 1e-9)


def test_one_to_one_assignment_does_not_reuse_a_row_minimum_column():
    """Verify that two A rows cannot select the same minimum-cost B column."""
    costs = np.array([[0.0, 100.0, 100.0],
                      [0.0,   1.0, 100.0],
                      [100.0, 0.0,   1.0]])

    mapping, selected, total, second_best, gap = C._select_one_to_one(
        costs, abs_tol_mm=1e-12, rel_tol=1e-12
    )

    assert mapping == (0, 1, 2)
    np.testing.assert_array_equal(selected, np.array([0.0, 1.0, 1.0]))
    assert len(set(mapping)) == 3
    assert total == 2.0
    assert second_best > total
    assert gap > 0.0


def test_tolerance_tie_uses_lexicographically_first_permutation_and_zero_gap():
    """Verify stable lexicographic selection when scores tie within tolerance."""
    costs = np.array([[1e-13, 0.0, 10.0],
                      [0.0,   0.0, 10.0],
                      [10.0, 10.0, 0.0]])

    mapping, selected, total, second_best, gap = C._select_one_to_one(
        costs, abs_tol_mm=1e-12, rel_tol=0.0
    )

    # (1, 0, 2) has exact score zero, but (0, 1, 2) is within tolerance.
    assert mapping == (0, 1, 2)
    np.testing.assert_array_equal(selected, np.array([1e-13, 0.0, 0.0]))
    assert total == second_best == 1e-13
    assert gap == 0.0
