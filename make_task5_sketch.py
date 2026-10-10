
"""Draw the DRR imaging geometry seen from the patient's feet (CK x-y plane),
to accompany the Task 5 discussion: sources, detectors, sphere phantom,
markers and a few pixel beams. Output: output/drr/task5_geometry_topview.png
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import drr
import transforms as T
from config import POSES, DETECTOR_SIZE_MM, SPHERE_PHANTOM_RADIUS_MM, MARKER_NAMES

COLOURS = {"A": "tab:blue", "B": "tab:orange"}


def main():
    fig, ax = plt.subplots(figsize=(7, 7))
    for pose in POSES:
        S = T.source_position_ck(pose)
        u, v, w = T.detector_axes_in_ck(pose)
        C = T.detector_origin_in_ck(pose)
        edge = C + np.outer([-1, 1], v) * DETECTOR_SIZE_MM / 2
        ax.plot(edge[:, 0], edge[:, 1], lw=4, color=COLOURS[pose])
        ax.plot(*S[:2], "o", ms=10, color=COLOURS[pose])
        ax.annotate(f"source {pose}", S[:2], xytext=(5, 5), textcoords="offset points")
        ax.annotate(f"detector {pose}", C[:2], xytext=(5, -15), textcoords="offset points")
        for s in (-1, 0, 1):        # central beam and the two edge beams (u = 0)
            end = C + s * v * DETECTOR_SIZE_MM / 2
            ax.plot([S[0], end[0]], [S[1], end[1]], lw=0.8, ls="-" if s == 0 else "--",
                    color=COLOURS[pose])
    ax.add_patch(plt.Circle((0, 0), SPHERE_PHANTOM_RADIUS_MM, fill=False, lw=1.5))
    for name, m in zip(MARKER_NAMES, drr.marker_centres_ck()):
        ax.plot(m[0], m[1], "k.", ms=6)
        ax.annotate(name, m[:2], xytext=(4, 4), textcoords="offset points", fontsize=8)
    # Zoomed inset around the isocenter: phantom, markers and beam crossing
    inset = ax.inset_axes([0.33, 0.36, 0.34, 0.30])
    for pose in POSES:
        S, C = T.source_position_ck(pose), T.detector_origin_in_ck(pose)
        inset.plot([S[0], C[0]], [S[1], C[1]], lw=0.8, color=COLOURS[pose])
    inset.add_patch(plt.Circle((0, 0), SPHERE_PHANTOM_RADIUS_MM, fill=False, lw=1.2))
    for name, m in zip(MARKER_NAMES, drr.marker_centres_ck()):
        inset.plot(m[0], m[1], "k.", ms=6)
        inset.annotate(name, m[:2], xytext=(3, 3), textcoords="offset points", fontsize=7)
    inset.set_xlim(70, -70)
    inset.set_ylim(-70, 70)
    inset.set_aspect("equal")
    inset.tick_params(labelsize=7)
    inset.set_title("zoom: R = 50 mm phantom", fontsize=8)

    ax.set_aspect("equal")
    ax.invert_xaxis()        # viewed from the feet: patient's right (+x) on the left
    ax.set_xlabel("x_CK [mm]  (+x = patient right)")
    ax.set_ylabel("y_CK [mm]  (+y = anterior / ceiling)")
    ax.set_title("DRR geometry, viewed from the feet (z = 0 plane)\n"
                 "solid: central beams; dashed: beams to detector edges")
    ax.grid(alpha=0.3)
    out = ROOT / "output" / "drr" / "task5_geometry_topview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
