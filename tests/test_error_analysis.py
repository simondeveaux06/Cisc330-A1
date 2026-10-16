"""Task 10 error metrics, independent perturbations, and strict MLE limit."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / 'src'))

import numpy as np
import pytest
from numpy.testing import assert_allclose

from geometry import closest_points_between_lines

E = __import__('10_error_analysis')


def test_mre_is_euclidean_distance_for_single_points_and_batches():
    true = np.array([1., 2., 3.])
    assert E.marker_reconstruction_error(true + [3, 4, 0], true) == 5
    assert_allclose(E.marker_reconstruction_error(
        np.array([true, true + [0, 0, -2]]), true), [0, 2])


def test_rem_is_shortest_perpendicular_distance_for_known_skew_lines():
    # x-axis and a y-directed line at z=3 have a 3 mm common perpendicular.
    _, rem, on_A, on_B = closest_points_between_lines(
        [0, 0, 0], [1, 0, 0], [0, 0, 3], [0, 1, 0])
    assert_allclose(rem, [3])
    assert_allclose(on_B - on_A, [[0, 0, 3]])


def test_perturbations_have_exact_radius_and_all_integer_directions():
    offsets = E.perturbations(2.5)
    assert offsets.shape == (360, 2)
    assert_allclose(np.linalg.norm(offsets, axis=1), 2.5, atol=1e-14)
    assert_allclose(offsets[[0, 90, 180, 270]],
                    [[2.5, 0], [0, 2.5], [-2.5, 0], [0, -2.5]], atol=1e-14)
    assert_allclose(np.mod(np.rad2deg(np.arctan2(
        offsets[:, 1], offsets[:, 0])), 360), np.arange(360), atol=1e-12)
    assert_allclose(E.perturbations(0), 0)


@pytest.mark.parametrize('mle', [-1, np.inf, np.nan])
def test_invalid_perturbation_radius_is_rejected(mle):
    with pytest.raises(ValueError):
        E.perturbations(mle)


def test_max_mle_uses_strict_limit_and_handles_unsorted_values():
    assert E.max_mle([8, 2, 7, 0], [6, 1, 5, 0]) == 2
    assert E.max_mle([1, 2], [5, 6]) is None
    assert E.max_mle([], []) is None
    with pytest.raises(ValueError):
        E.max_mle([0, 1], [0])


def test_simulation_covers_every_pair_and_reuses_reconstruction_metrics():
    samples = E.simulate_mle(1.25, batch_size=10000)
    assert samples.shape == (3 * 360 * 360,)
    offsets = E.perturbations(1.25)
    markers = E.D.marker_centres_ck()
    uv_A = E.F.project_ck_to_detector(markers, 'A')
    uv_B = E.F.project_ck_to_detector(markers, 'B')
    for i, name in enumerate(('M1', 'M2', 'M3')):
        rows = samples[samples['marker_id'] == name]
        assert_allclose(rows['mle_mm'], 1.25)
        assert np.array_equal(rows['angle_A_deg'].astype(int) * 360
                              + rows['angle_B_deg'].astype(int),
                              np.arange(360 * 360))
        # Includes independent directions and points around batch boundaries.
        for index in [0, 359, 360, 9999, 10000, 129599]:
            row = rows[index]
            a = uv_A[i] + offsets[row['angle_A_deg']]
            b = uv_B[i] + offsets[row['angle_B_deg']]
            point, rem = E.R.reconstruct_point(a, b)
            assert_allclose(row['mre_mm'], np.linalg.norm(point[0] - markers[i]))
            assert_allclose(row['rem_mm'], rem[0])
            P, u = E.R.back_projection_line(a, 'A')
            Q, v = E.R.back_projection_line(b, 'B')
            normal = np.cross(u[0], v[0])
            independent_rem = abs(np.dot(Q[0] - P[0], normal)) / np.linalg.norm(normal)
            assert_allclose(row['rem_mm'], independent_rem, atol=1e-10)
    assert np.isfinite(samples['mre_mm']).all()
    assert np.isfinite(samples['rem_mm']).all()


def test_zero_mle_recovers_all_known_markers():
    samples = E.simulate_mle(0)
    assert_allclose(samples['mre_mm'], 0, atol=1e-9)
    assert_allclose(samples['rem_mm'], 0, atol=1e-9)
