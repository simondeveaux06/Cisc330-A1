"""Question 2: transform, save, plot, and validate the CyberKnife markers.

Run from anywhere: python path/to/markers.py
Results are written to a 'q2_outputs' folder next to this file.
All positions and pairwise distances are in millimetres.
Detector positions are 3D frame coordinates, NOT perspective projections.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import MARKERS_MD_MM
from transforms import md_to_ck, ck_to_detector, detector_to_ck, ck_to_md

OUTPUT_DIR = Path(__file__).resolve().parent / 'q2_outputs'
TOL_MM = 1e-9


def calculate_marker_frames():
    """Return one (3, 3) array per coordinate frame, in millimetres."""
    md = np.asarray(MARKERS_MD_MM, dtype=float).copy()
    ck = md_to_ck(md)
    det_a = ck_to_detector(ck, 'A')
    det_b = ck_to_detector(ck, 'B')
    return {'MD': md, 'CK': ck, 'Detector_A': det_a, 'Detector_B': det_b}


def pairwise_distances(points):
    """Distances for M1-M2, M1-M3, M2-M3 (mm)."""
    return np.array([np.linalg.norm(points[i] - points[j])
                     for i, j in ((0, 1), (0, 2), (1, 2))])


def validate_marker_frames(frames, tol_mm=TOL_MM):
    """Raise AssertionError if a transform violates expected rigid geometry."""
    md, ck = frames['MD'], frames['CK']
    if md.shape != (3, 3) or any(p.shape != (3, 3) for p in frames.values()):
        raise AssertionError('Expected three 3D marker coordinates per frame.')
    if not all(np.isfinite(p).all() for p in frames.values()):
        raise AssertionError('All marker coordinates must be finite.')
    np.testing.assert_allclose(ck, md, rtol=0, atol=tol_mm,
                               err_msg='MD and CK should coincide.')
    np.testing.assert_allclose(ck_to_md(ck), md, rtol=0, atol=tol_mm)
    baseline = pairwise_distances(md)
    for name, points in frames.items():
        np.testing.assert_allclose(pairwise_distances(points), baseline,
                                   rtol=0, atol=tol_mm,
                                   err_msg=f'Pairwise distances changed in {name}.')
    for pose in ('A', 'B'):
        recovered = detector_to_ck(frames[f'Detector_{pose}'], pose)
        np.testing.assert_allclose(recovered, ck, rtol=0, atol=tol_mm,
                                   err_msg=f'Detector {pose} round trip failed.')
    return baseline


def save_coordinates(frames, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / 'marker_coordinates_mm.npz', **frames)
    with (output_dir / 'marker_coordinates_mm.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['marker', 'frame', 'x_or_u_mm', 'y_or_v_mm', 'z_or_w_mm'])
        for frame, points in frames.items():
            for i, point in enumerate(points, start=1):
                writer.writerow([f'M{i}', frame, *[f'{value:.10f}' for value in point]])


def plot_markers(frames, output_dir):
    """Create separate 3D plots; MD and CK overlap because their frames coincide."""
    fig = plt.figure(figsize=(11, 9))
    for idx, (name, points) in enumerate(frames.items(), start=1):
        ax = fig.add_subplot(2, 2, idx, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=55)
        for marker_num, (x, y, z) in enumerate(points, start=1):
            ax.text(x, y, z, f' M{marker_num}')
        ax.set_title(f'{name} frame')
        if name in ('MD', 'CK'):
            ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)'); ax.set_zlabel('z (mm)')
        else:
            ax.set_xlabel('u (mm)'); ax.set_ylabel('v (mm)'); ax.set_zlabel('w (mm)')
        ax.set_box_aspect((1, 1, 1))
    fig.suptitle('CyberKnife marker positions in four coordinate frames')
    fig.tight_layout()
    fig.savefig(output_dir / 'marker_frames_3d.png', dpi=180)
    plt.close(fig)


def main():
    frames = calculate_marker_frames()
    distances = validate_marker_frames(frames)
    save_coordinates(frames, OUTPUT_DIR)
    plot_markers(frames, OUTPUT_DIR)
    with (OUTPUT_DIR / 'validation.txt').open('w') as f:
        f.write('Q2 MARKER TRANSFORM VALIDATION\n')
        f.write('All coordinates and distances are in mm.\n')
        f.write(f'Tolerance: {TOL_MM:g} mm\n')
        f.write('PASS: MD and CK marker coordinates agree.\n')
        f.write('PASS: All frame transformations preserve pairwise distances.\n')
        f.write('PASS: Detector A/B to CK round trips recover the original points.\n')
        for label, d in zip(('M1-M2', 'M1-M3', 'M2-M3'), distances):
            f.write(f'{label}: {d:.6f} mm\n')
    print('Q2 validation PASSED')
    for name, pts in frames.items():
        print(f'\n{name} (mm):\n{np.array2string(pts, precision=4)}')
    print(f'\nSaved outputs to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
