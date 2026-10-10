"""Task 6: localize the markers in the four full-resolution DRRs (from
run_task5.py) and compare segmented, true and visually picked positions.

Run from the repository root:   python run_task6.py

Inputs
  output/drr/drr_<phantom>_<pose>.npz      DRRs from Task 5
  docs/visual_localization.csv             YOUR visual readings (see below)
Outputs in output/localization/
  segmented_markers.csv     detected centres (image + detector coords), for Tasks 7/8
  task6_results.txt         segmented vs true (and vs visual, once filled in)
  task6_zoom_<phantom>.png  close-ups: true (+), segmented (x), visual (o)

Visual localization (the handout's test, done by a person)
  Open each output/drr/drr_<phantom>_<pose>.png (2000 x 2000) in a viewer that
  shows the cursor's pixel coordinates (e.g. Fiji/ImageJ, GIMP, Paint), put
  the cursor in the middle of each of the three bright discs and type the
  (x, y) the viewer shows into docs/visual_localization.csv, then re-run this
  script. Order of the markers does not matter (they are matched to the
  nearest true projection).
  The PNGs were saved with superior (+u) at the TOP, so the viewer's y axis
  runs opposite to our row index:   row = 1999 - y_viewer,   col = x_viewer.
"""
import csv
import sys
import pathlib

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import drr
import localization as L
import transforms as T
from config import POSES, N_PIXELS, MARKER_NAMES, PIXEL_PITCH_MM

DRR_DIR = ROOT / "output" / "drr"
OUT_DIR = ROOT / "output" / "localization"
VISUAL_CSV = ROOT / "docs" / "visual_localization.csv"
PHANTOMS = ("sphere", "vertebra")
ZOOM_HALF_MM = 5.0


def load_drr(phantom, pose):
    return np.load(DRR_DIR / f"drr_{phantom}_{pose}.npz")["attenuation"]


def viewer_xy_to_detector_uv(x_viewer, y_viewer, pose):
    """Pixel picked in an image viewer on the saved PNG (origin top-left,
    superior at the top) -> Detector (u, v) mm via the Task 1 image frame."""
    row = (N_PIXELS - 1) - y_viewer
    col = x_viewer
    return T.image_to_detector(np.array([row, col], float), pose)[:2]


def read_visual_csv():
    """{(phantom, pose): (N, 2) detector uv} for the rows that are filled in."""
    picks = {}
    if not VISUAL_CSV.exists():
        return picks
    with open(VISUAL_CSV, newline="") as f:
        for rec in csv.DictReader(f):
            if not rec["x_viewer_px"].strip() or not rec["y_viewer_px"].strip():
                continue
            key = (rec["phantom"], rec["pose"])
            uv = viewer_xy_to_detector_uv(float(rec["x_viewer_px"]), float(rec["y_viewer_px"]), rec["pose"])
            picks.setdefault(key, []).append(uv)
    return {k: np.array(v) for k, v in picks.items()}


def write_visual_template():
    """Create the empty sheet once, so it is never overwritten after filling."""
    if VISUAL_CSV.exists():
        return
    VISUAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(VISUAL_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phantom", "pose", "png_file", "x_viewer_px", "y_viewer_px", "note"])
        for phantom in PHANTOMS:
            for pose in POSES:
                for _ in range(3):
                    w.writerow([phantom, pose, f"output/drr/drr_{phantom}_{pose}.png", "", "", ""])


def fmt_uv(uv):
    return f"({uv[0]:8.3f}, {uv[1]:8.3f})"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_visual_template()
    visual = read_visual_csv()

    lines = ["TASK 6 - MARKER LOCALIZATION (full resolution, 0.1 mm pixels)",
             f"threshold = {L.THRESHOLD_FRACTION} x expected peak contrast "
             f"{L.EXPECTED_PEAK_CONTRAST:.3f} = {L.THRESHOLD_FRACTION * L.EXPECTED_PEAK_CONTRAST:.3f}"
             f"   background window = {L.BACKGROUND_WINDOW_MM} mm", "=" * 92]
    csv_rows = [["phantom", "pose", "row_px", "col_px", "u_mm", "v_mm", "area_px", "peak_contrast"]]
    all_seg_err, all_vis_err, all_vis_seg = [], [], []

    for phantom in PHANTOMS:
        fig, axes = plt.subplots(len(POSES), 3, figsize=(10, 7), constrained_layout=True)
        for i, pose in enumerate(POSES):
            image = load_drr(phantom, pose)
            result = L.localize_markers(image, pose)
            seg_uv = L.detections_uv_mm(result)
            true_uv = drr.project_point_to_detector(drr.marker_centres_ck(), pose)
            seg_matched = L.match_to_reference(seg_uv, true_uv)     # ordered M1, M2, M3
            for m in result.markers:
                csv_rows.append([phantom, pose, f"{m.row:.3f}", f"{m.col:.3f}",
                                 f"{m.u_mm:.4f}", f"{m.v_mm:.4f}", m.area_px, f"{m.peak_contrast:.4f}"])

            vis_matched = None
            if (phantom, pose) in visual and len(visual[(phantom, pose)]) == 3:
                vis_matched = L.match_to_reference(visual[(phantom, pose)], true_uv)

            lines.append(f"\n{phantom} pose {pose}")
            lines.append(f"  {'':3} {'true (u,v) mm':>22} {'segmented (u,v) mm':>22} {'seg err mm':>10} "
                         f"{'visual (u,v) mm':>22} {'vis-true':>9} {'vis-seg':>8}")
            for k, name in enumerate(MARKER_NAMES):
                e_seg = np.linalg.norm(seg_matched[k] - true_uv[k])
                all_seg_err.append(e_seg)
                row = f"  {name:3} {fmt_uv(true_uv[k]):>22} {fmt_uv(seg_matched[k]):>22} {e_seg:10.4f} "
                if vis_matched is not None:
                    e_vis = np.linalg.norm(vis_matched[k] - true_uv[k])
                    e_vs = np.linalg.norm(vis_matched[k] - seg_matched[k])
                    all_vis_err.append(e_vis)
                    all_vis_seg.append(e_vs)
                    row += f"{fmt_uv(vis_matched[k]):>22} {e_vis:9.3f} {e_vs:8.3f}"
                else:
                    row += f"{'(not filled in)':>22}"
                lines.append(row)

                # Close-up of this marker
                ax = axes[i, k]
                r0, c0 = T.detector_to_image(true_uv[k], pose)
                half_px = int(ZOOM_HALF_MM / PIXEL_PITCH_MM)
                r_lo, c_lo = int(r0) - half_px, int(c0) - half_px
                crop = image[r_lo:r_lo + 2 * half_px + 1, c_lo:c_lo + 2 * half_px + 1]
                ext = [c_lo - 0.5, c_lo + 2 * half_px + 0.5, r_lo - 0.5, r_lo + 2 * half_px + 0.5]
                ax.imshow(crop, cmap="gray", origin="lower", extent=ext)
                rs, cs = T.detector_to_image(seg_matched[k], pose)
                ax.plot(c0, r0, "+", color="lime", ms=16, mew=1.5, label="true")
                ax.plot(cs, rs, "x", color="red", ms=10, mew=1.5, label="segmented")
                if vis_matched is not None:
                    rv, cv = T.detector_to_image(vis_matched[k], pose)
                    ax.plot(cv, rv, "o", mfc="none", color="cyan", ms=10, mew=1.5, label="visual")
                ax.set_title(f"pose {pose}, {name}: seg err {e_seg * 1000:.0f} µm", fontsize=9)
                ax.set_xlabel("col [px]", fontsize=8)
                ax.set_ylabel("row [px]", fontsize=8)
                ax.tick_params(labelsize=7)
        axes[0, 0].legend(fontsize=7, loc="lower left")
        fig.suptitle(f"{phantom} DRR: marker close-ups (±{ZOOM_HALF_MM:.0f} mm on the detector)")
        fig.savefig(OUT_DIR / f"task6_zoom_{phantom}.png", dpi=110)
        plt.close(fig)

    lines.append("\nSUMMARY over all 12 marker images (detector mm; 1 px = 0.1 mm)")
    lines.append(f"  segmented vs true : mean {np.mean(all_seg_err):.4f}   max {np.max(all_seg_err):.4f}")
    if all_vis_err:
        lines.append(f"  visual vs true    : mean {np.mean(all_vis_err):.3f}   max {np.max(all_vis_err):.3f}"
                     f"   ({len(all_vis_err)} markers filled in)")
        lines.append(f"  visual vs segm.   : mean {np.mean(all_vis_seg):.3f}   max {np.max(all_vis_seg):.3f}")
    else:
        lines.append(f"  visual            : not filled in yet -> edit {VISUAL_CSV.relative_to(ROOT)} and re-run")

    with open(OUT_DIR / "segmented_markers.csv", "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    text = "\n".join(lines)
    print(text)
    (OUT_DIR / "task6_results.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
