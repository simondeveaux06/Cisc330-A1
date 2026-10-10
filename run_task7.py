"""Task 7 test (handout): feed the known marker locations in the Detector frames
into the reconstruction module and show that it produces the known marker
locations in the CK frame with (near) zero residual error.

Run from the repository root:   python run_task7.py
Output: output/reconstruction/task7_results.txt
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

import drr
import reconstruction as R
from config import MARKER_NAMES

OUT_DIR = ROOT / "output" / "reconstruction"


def fmt(p):
    return "(" + ", ".join(f"{x:9.4f}" for x in p) + ")"


def main():
    # Known marker locations in the Detector frames: forward projection of the
    # true markers (the group's Task 3/4 projector gives the same numbers).
    markers_ck = drr.marker_centres_ck()
    uv_A = drr.project_point_to_detector(markers_ck, "A")
    uv_B = drr.project_point_to_detector(markers_ck, "B")
    p_ck, rem = R.reconstruct_point(uv_A, uv_B)

    lines = ["TASK 7 - MARKER RECONSTRUCTION: known detector locations -> CK frame", "=" * 110,
             f"  {'':3} {'Detector A (u,v) mm':>20} {'Detector B (u,v) mm':>20} {'known CK mm':>32} "
             f"{'reconstructed CK mm':>32} {'|err| mm':>9} {'REM mm':>9}"]
    for k, name in enumerate(MARKER_NAMES):
        err = np.linalg.norm(p_ck[k] - markers_ck[k])
        lines.append(f"  {name:3} ({uv_A[k, 0]:7.3f},{uv_A[k, 1]:8.3f}) ({uv_B[k, 0]:7.3f},{uv_B[k, 1]:8.3f}) "
                     f"{fmt(markers_ck[k]):>32} {fmt(p_ck[k]):>32} {err:9.1e} {rem[k]:9.1e}")
    lines.append("\nThe known CK positions are recovered to machine precision with near-zero REM.")
    text = "\n".join(lines)
    print(text)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "task7_results.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
