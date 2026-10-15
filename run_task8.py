"""Run Task 8 on the three known markers and print both diagnostics."""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

C = __import__('8_correspondences')
D = __import__('5_drr')


def main():
    markers_ck = D.marker_centres_ck()
    uv_A = D.project_point_to_detector(markers_ck, "A")
    uv_B = D.project_point_to_detector(markers_ck, "B")
    rem = C.match_markers_by_rem(uv_A, uv_B)
    epi = C.match_markers_by_epipolar(uv_A, uv_B)

    np.set_printoptions(precision=10, suppress=True)
    print("REM matrix (mm):")
    print(rem.rem_matrix_mm)
    print(f"mapping A->B: {rem.mapping}")
    print(f"selected REMs (mm): {rem.selected_residuals_mm}")
    print(f"total / second / gap (mm): {rem.total_score_mm:.12g} / "
          f"{rem.second_best_score_mm:.12g} / {rem.score_gap_mm:.12g}")
    print("\nEpipolar-distance matrix on Detector B (mm):")
    print(epi.epipolar_distance_matrix_mm)
    print(f"mapping A->B: {epi.mapping}")
    print(f"selected distances (mm): {epi.selected_distances_mm}")


if __name__ == "__main__":
    main()
