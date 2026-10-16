"""Task 10: exhaustively perturb both detector views and measure MRE and REM.

Every MLE uses all 360 x 360 independent integer-degree direction pairs for
each known CK marker. Geometry comes from Tasks 3 and 7; batches bound memory.
"""
import numpy as np

from config import MARKER_NAMES

D = __import__('5_drr')
F = __import__('3_forward_projector')
R = __import__('7_reconstruction')

SAMPLE_DTYPE = np.dtype([
    ('mle_mm', 'f8'), ('marker_id', 'U2'),
    ('angle_A_deg', 'u2'), ('angle_B_deg', 'u2'),
    ('mre_mm', 'f8'), ('rem_mm', 'f8'),
])


def marker_reconstruction_error(reconstructed_ck, true_ck):
    """Euclidean CK distance in mm, supporting batches."""
    return np.linalg.norm(np.asarray(reconstructed_ck) - true_ck, axis=-1)


def perturbations(mle_mm):
    """Offsets of exactly MLE mm; angle 0 points along +u toward +v."""
    if not np.isfinite(mle_mm) or mle_mm < 0:
        raise ValueError('MLE must be finite and non-negative')
    angles = np.deg2rad(np.arange(360))
    return mle_mm * np.column_stack((np.cos(angles), np.sin(angles)))


def max_mle(mles_mm, worst_mres_mm, limit_mm=5.0):
    """Largest tested MLE with worst MRE strictly below the limit, or None."""
    mles = np.asarray(mles_mm, dtype=float)
    errors = np.asarray(worst_mres_mm, dtype=float)
    if (mles.ndim != 1 or mles.shape != errors.shape
            or not np.isfinite(mles).all() or not np.isfinite(errors).all()):
        raise ValueError('Expected equally sized finite 1-D MLE and MRE arrays')
    valid = mles[errors < limit_mm]
    return float(valid.max()) if valid.size else None


def simulate_mle(mle_mm, batch_size=16384):
    """Return all 388,800 reconstructions' errors and direction metadata."""
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')
    offsets = perturbations(mle_mm)
    markers = D.marker_centres_ck()
    uv_A = F.project_ck_to_detector(markers, 'A')
    uv_B = F.project_ck_to_detector(markers, 'B')
    angle_A = np.repeat(np.arange(360), 360)
    angle_B = np.tile(np.arange(360), 360)
    count = len(angle_A)
    samples = np.empty(len(markers) * count, dtype=SAMPLE_DTYPE)
    for marker, name in enumerate(MARKER_NAMES):
        rows = samples[marker * count:(marker + 1) * count]
        rows['mle_mm'] = mle_mm
        rows['marker_id'] = name
        rows['angle_A_deg'], rows['angle_B_deg'] = angle_A, angle_B
        for start in range(0, count, batch_size):
            sl = slice(start, min(start + batch_size, count))
            reconstructed, rem = R.reconstruct_point(
                uv_A[marker] + offsets[angle_A[sl]],
                uv_B[marker] + offsets[angle_B[sl]],
            )
            rows['mre_mm'][sl] = marker_reconstruction_error(
                reconstructed, markers[marker]
            )
            rows['rem_mm'][sl] = rem
    return samples
