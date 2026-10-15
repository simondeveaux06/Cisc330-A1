"""
Question 4: Marker Projector

Uses the existing Q1 transforms and Q3 forward projector. Physical coordinates
are millimetres; image coordinates are zero-based (row, column) pixel-center
indices (possibly fractional). A 200 mm detector spans [-100,100] mm in u,v.
Run: python3 marker_projector.py
"""
from pathlib import Path
import csv
import numpy as np
from config import MARKERS_MD_MM, DETECTOR_SIZE_MM, N_PIXELS
from transforms import md_to_ck, ck_to_detector, detector_to_image
from projector import project_ck_to_detector

OUT = Path(__file__).resolve().parent / 'q4_outputs'


def project_markers():
    """Return arrays (3,3) CK, (3,2) detector and (3,2) image for each pose."""
    ck = md_to_ck(MARKERS_MD_MM)
    result = {'markers_md_mm': np.asarray(MARKERS_MD_MM, float).copy(),
              'markers_ck_mm': ck}
    for pose in ('A', 'B'):
        uv = project_ck_to_detector(ck, pose)
        rc = detector_to_image(uv)
        inside = np.all(np.abs(uv) <= DETECTOR_SIZE_MM / 2, axis=1)
        result[f'detector_{pose}_uv_mm'] = uv
        result[f'image_{pose}_row_col_px'] = rc
        result[f'detector_{pose}_inside'] = inside
    return result


def hand_estimates():
    """Independent approximate predictions using cos(45°)≈sin(45°)≈0.7071.

    For CK (x,y,z), detector u=z, v=c*x+s*y, w=1000-s*x+c*y,
    with s=+0.7071 for A and -0.7071 for B, c=0.7071.
    Projected (u',v') = 2000*(u,v)/(2000-w).
    """
    predictions = {}
    for pose, s in (('A', 0.7071), ('B', -0.7071)):
        estimates = []
        for x, y, z in MARKERS_MD_MM:
            u = z
            v = 0.7071*x + s*y
            w = 1000 - s*x + 0.7071*y
            estimates.append([2000*u/(2000-w), 2000*v/(2000-w)])
        predictions[pose] = np.array(estimates)
    return predictions


def validate(results, predictions, tolerance_mm=0.01):
    """Check approximate hand estimates, frame conversion, and pixel convention."""
    np.testing.assert_allclose(results['markers_ck_mm'], MARKERS_MD_MM, atol=1e-9, rtol=0)
    for pose in ('A', 'B'):
        actual = results[f'detector_{pose}_uv_mm']
        expected = predictions[pose]
        np.testing.assert_allclose(actual, expected, atol=tolerance_mm, rtol=0,
                                   err_msg=f'{pose}: hand estimate disagreement')
        det_3d = ck_to_detector(results['markers_ck_mm'], pose)
        # All marker projections must be collinear with source=(0,0,2000).
        scale = 2000 / (2000 - det_3d[:, 2])
        np.testing.assert_allclose(actual, det_3d[:, :2] * scale[:, None], atol=1e-9)
        rc = results[f'image_{pose}_row_col_px']
        np.testing.assert_allclose(rc, detector_to_image(actual), atol=1e-9)
        inside_pixels = np.all((rc >= -0.5) & (rc <= N_PIXELS - 0.5), axis=1)
        np.testing.assert_array_equal(inside_pixels, results[f'detector_{pose}_inside'])
    return True


def save_results(results, predictions, output_dir=OUT):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / 'marker_projections.npz', **results)
    with (output_dir / 'marker_projections.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['marker', 'pose', 'ck_x_mm', 'ck_y_mm', 'ck_z_mm',
                         'detector_u_mm', 'detector_v_mm', 'image_row_px',
                         'image_col_px', 'inside_detector'])
        for pose in ('A', 'B'):
            for i, (xyz, uv, rc, inside) in enumerate(zip(
                results['markers_ck_mm'], results[f'detector_{pose}_uv_mm'],
                results[f'image_{pose}_row_col_px'], results[f'detector_{pose}_inside'])):
                writer.writerow([f'M{i+1}', pose, *[f'{v:.8f}' for v in xyz],
                                 *[f'{v:.8f}' for v in uv], *[f'{v:.8f}' for v in rc], bool(inside)])
    with (output_dir / 'hand_estimate_comparison.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['marker', 'pose', 'hand_u_mm', 'computed_u_mm', 'u_error_mm',
                         'hand_v_mm', 'computed_v_mm', 'v_error_mm'])
        for pose in ('A', 'B'):
            for i, (hand, actual) in enumerate(zip(predictions[pose], results[f'detector_{pose}_uv_mm'])):
                writer.writerow([f'M{i+1}', pose, f'{hand[0]:.6f}', f'{actual[0]:.6f}',
                                 f'{actual[0]-hand[0]:.6f}', f'{hand[1]:.6f}',
                                 f'{actual[1]:.6f}', f'{actual[1]-hand[1]:.6f}'])
    return output_dir


def main():
    results = project_markers()
    estimates = hand_estimates()
    validate(results, estimates)
    output_dir = save_results(results, estimates)
    print('Q4 validation PASSED (hand-estimate tolerance: 0.01 mm)')
    for pose in ('A', 'B'):
        print(f'\nDetector {pose}:')
        for i, (uv, rc, hand, inside) in enumerate(zip(
            results[f'detector_{pose}_uv_mm'], results[f'image_{pose}_row_col_px'],
            estimates[pose], results[f'detector_{pose}_inside']), start=1):
            print(f'M{i}: detector={np.round(uv, 4)} mm; image={np.round(rc, 3)} px; '
                  f'hand≈{np.round(hand, 4)} mm; inside={inside}')
    print(f'\nSaved files in {output_dir}')


if __name__ == '__main__':
    main()
