"""Q4: reproducible CyberKnife geometry sketch, using existing group modules.

Run: python make_task4_sketch.py
Requires: numpy, matplotlib; config.py, transforms.py, projector.py alongside.
Creates: task4_sketch.png
AI-assisted plotting script; numerical coordinates come from the imported modules.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from config import MARKERS_MD_MM, SAD_MM, SDD_MM, DETECTOR_SIZE_MM
_transforms = __import__('1_transforms', fromlist=['*'])
md_to_ck = _transforms.md_to_ck
detector_to_ck = _transforms.detector_to_ck
detector_origin_in_ck = _transforms.detector_origin_in_ck
detector_axes_in_ck = _transforms.detector_axes_in_ck
source_position_ck = _transforms.source_position_ck
ck_to_detector = _transforms.ck_to_detector
project_ck_to_detector = __import__(
    '3_forward_projector', fromlist=['project_ck_to_detector']
).project_ck_to_detector

COLORS = {'A': '#dc4c4c', 'B': '#2563eb'}
MARKER_COLORS = ['#e6a13a', '#9a60c6', '#159b81']


def geometry():
    markers = md_to_ck(MARKERS_MD_MM)
    result = {}
    half = DETECTOR_SIZE_MM / 2
    for pose in ('A', 'B'):
        source = source_position_ck(pose)
        center = detector_origin_in_ck(pose)
        u_axis, v_axis, w_axis = detector_axes_in_ck(pose)
        uv = project_ck_to_detector(markers, pose)
        # Plane points: w=0; inverse transform is the existing implementation.
        hits = detector_to_ck(np.column_stack((uv, np.zeros(len(uv)))), pose)
        corners_uv = np.array([[-half, -half], [half, -half],
                               [half, half], [-half, half]])
        corners = detector_to_ck(np.column_stack((corners_uv, np.zeros(4))), pose)
        # Independent consistency checks, not just a rendered visual inspection.
        np.testing.assert_allclose(ck_to_detector(hits, pose)[:, :2], uv, atol=1e-9)
        np.testing.assert_allclose(ck_to_detector(hits, pose)[:, 2], 0, atol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(source), SAD_MM, atol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(source - center), SDD_MM, atol=1e-9)
        for marker, hit in zip(markers, hits):
            # Ray source -> marker -> hit: all collinear, marker between endpoints.
            direction = hit - source
            t = np.dot(marker - source, direction) / np.dot(direction, direction)
            assert 0 < t < 1
            np.testing.assert_allclose(source + t * direction, marker, atol=1e-8)
        result[pose] = dict(source=source, center=center, u_axis=u_axis,
                            v_axis=v_axis, w_axis=w_axis, uv=uv, hits=hits,
                            corners=corners)
    return markers, result


def rough_estimates(markers):
    x, y, z = markers.T
    return {'A': np.column_stack((2*z, 2*(x+y)/np.sqrt(2))),
            'B': np.column_stack((2*z, 2*(x-y)/np.sqrt(2)))}


def draw_vector(ax, a, b, color, width=1.5, style='-', alpha=1):
    a, b = np.asarray(a), np.asarray(b)
    ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
            color=color, lw=width, ls=style, alpha=alpha)


def draw_main(ax, markers, geom):
    ax.scatter([0], [0], [0], c='black', s=60, depthshade=False, zorder=9)
    ax.text(25, -65, -90, 'CK isocenter (0,0,0)', fontsize=9, weight='bold')
    axis_colors = ['#a53232', '#23894a', '#345db3']
    for idx, (label, col) in enumerate(zip(('CK +x', 'CK +y', 'CK +z'), axis_colors)):
        endpoint = np.zeros(3); endpoint[idx] = 370
        ax.quiver(0, 0, 0, *endpoint, color=col, linewidth=2.2, arrow_length_ratio=.09)
        ax.text(*(endpoint * 1.11), label, color=col, fontsize=10, weight='bold')
    for pose in ('A', 'B'):
        g = geom[pose]; col = COLORS[pose]
        ax.add_collection3d(Poly3DCollection([g['corners']], facecolors=col,
                                             edgecolors=col, alpha=.19, linewidths=1.8))
        ax.scatter(*g['source'], s=105, c=col, marker='^', depthshade=False)
        ax.text(*(g['source'] + np.array([30, 10, 75])), f'Source {pose}',
                fontsize=10, color=col, weight='bold')
        ax.text(*(g['center'] + np.array([45, 0, -110])), f'Detector {pose}',
                fontsize=10, color=col, weight='bold')
        # Dashed central beam from source through isocenter to detector center.
        draw_vector(ax, g['source'], g['center'], col, 1.1, '--', .6)
        for i, (marker, hit) in enumerate(zip(markers, g['hits'])):
            draw_vector(ax, g['source'], hit, col, 1.15, '-', .63)
            ax.scatter(*hit, s=33, c=col, edgecolors='white', linewidths=.6,
                       depthshade=False)
        # SAD and SDD label each pose's actual source-isocenter and source-detector distances.
        mid_sad = g['source'] / 2
        ax.text(*(mid_sad + np.array([0, 0, 170 if pose == 'A' else -170])),
                f'{pose}: SAD = {SAD_MM:.0f} mm', color=col, fontsize=8)
        mid_sdd = (g['source'] + g['center']) / 2
        ax.text(*(mid_sdd + np.array([0, 0, -240 if pose == 'A' else 240])),
                f'{pose}: SDD = {SDD_MM:.0f} mm', color=col, fontsize=8)
    for i, marker in enumerate(markers):
        ax.scatter(*marker, s=88, color=MARKER_COLORS[i], edgecolors='black',
                   linewidths=.5, depthshade=False)
    ax.set_xlabel('CK x (mm)', labelpad=8)
    ax.set_ylabel('CK y (mm)', labelpad=8)
    ax.set_zlabel('CK z (mm)', labelpad=8)
    ax.set_title('Full 3D geometry (physical coordinates, mm)', fontsize=12, pad=13)
    ax.set_xlim(-1300, 1300); ax.set_ylim(-1300, 1300); ax.set_zlim(-330, 490)
    ax.set_box_aspect((2.6, 2.6, 1.6), zoom=1.28)
    ax.view_init(elev=24, azim=-65)
    ax.tick_params(labelsize=8)


def draw_zoom(ax, markers):
    ax.scatter([0], [0], [0], c='black', s=65, label='Isocenter')
    for i, p in enumerate(markers):
        ax.scatter(*p, s=115, color=MARKER_COLORS[i], edgecolors='black', linewidths=.6)
        ax.text(p[0]+2, p[1]+2, p[2]+2, f'M{i+1} ({p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f})',
                fontsize=10, weight='bold')
    for j, col in enumerate(('#a53232','#23894a','#345db3')):
        end = np.zeros(3); end[j] = 35
        ax.quiver(0, 0, 0, *end, color=col, arrow_length_ratio=.12, linewidth=2)
    ax.set_xlim(-37, 68); ax.set_ylim(-8, 77); ax.set_zlim(-25, 43)
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)'); ax.set_zlabel('z (mm)')
    ax.set_title('Enlarged CK marker positions', fontsize=12, pad=13)
    ax.view_init(elev=22, azim=-64)
    ax.set_box_aspect((1.1, 1, .75))
    ax.tick_params(labelsize=8)


def draw_detector(ax, pose, exact, approx):
    half = DETECTOR_SIZE_MM / 2
    ax.add_patch(Rectangle((-half,-half), 2*half, 2*half,
                           facecolor=COLORS[pose], edgecolor=COLORS[pose],
                           alpha=.06, linewidth=1.5))
    ax.axhline(0, color='gray', lw=.7, alpha=.5)
    ax.axvline(0, color='gray', lw=.7, alpha=.5)
    for i, (e, a) in enumerate(zip(exact, approx)):
        ax.plot([e[0], a[0]], [e[1], a[1]], color=MARKER_COLORS[i], lw=1.1, alpha=.75)
        ax.scatter(*e, marker='o', s=75, color=MARKER_COLORS[i],
                   edgecolors='black', linewidths=.7, zorder=5)
        ax.scatter(*a, marker='x', s=85, color='black', linewidths=1.5, zorder=6)
        ax.annotate(f'M{i+1}', xy=e, xytext=(7, 6 if i != 2 else -14),
                    textcoords='offset points', fontsize=10, weight='bold')
    ax.set(xlim=(-110,110), ylim=(-110,110), xlabel='Detector u (mm)',
           ylabel='Detector v (mm)', title=f'Detector {pose}: exact vs ~2× estimates')
    ax.set_aspect('equal', adjustable='box'); ax.grid(alpha=.22)
    ax.tick_params(labelsize=9)


def main():
    markers, geom = geometry()
    approx = rough_estimates(markers)
    # Ensure the approximation and exact coordinates are distinct data sources.
    fig = plt.figure(figsize=(17, 12.5), facecolor='white', constrained_layout=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], left=.055, right=.96,
                          top=.91, bottom=.095, hspace=.29, wspace=.20)
    draw_main(fig.add_subplot(gs[0, 0], projection='3d'), markers, geom)
    draw_zoom(fig.add_subplot(gs[0, 1], projection='3d'), markers)
    for col, pose in enumerate(('A','B')):
        draw_detector(fig.add_subplot(gs[1, col]), pose, geom[pose]['uv'], approx[pose])
    fig.suptitle('Q4 | CyberKnife marker projections — computational 3D sketch',
                 fontsize=19, weight='bold', y=.985)
    handles = [Line2D([0],[0], marker='o', color='none', markerfacecolor='#777',
                      markeredgecolor='black', markersize=9, label='Exact projected marker'),
               Line2D([0],[0], marker='x', color='black', linestyle='None',
                      markersize=9, label='Rough ~2× estimate')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5,.055),
               ncol=2, frameon=False, fontsize=10)
    fig.text(.5,.023, 'AI-assisted Matplotlib diagram | CK axes and all coordinates in mm | '
             'Approximation valid near isocenter; exact rays use perspective projection.',
             ha='center', fontsize=9, color='#444')
    out = Path(__file__).with_name('task4_sketch.png')
    fig.savefig(out, dpi=185, facecolor='white')
    plt.close(fig)
    print(f'Saved {out}')
    for pose in ('A','B'):
        print(f'\nDetector {pose} — exact vs rough (u,v) mm')
        for i, (e, a) in enumerate(zip(geom[pose]['uv'], approx[pose])):
            print(f'M{i+1}: exact=({e[0]:8.3f}, {e[1]:8.3f}), '
                  f'rough=({a[0]:8.3f}, {a[1]:8.3f}), '
                  f'error={np.linalg.norm(e-a):.3f} mm')
    print('\nAll 3D detector hits validated against existing transforms and projector.')


if __name__ == '__main__':
    main()
