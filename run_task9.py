"""Run Task 9 ambiguity detection on three engineered cases."""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

A = __import__('9_ambiguity')
D = __import__('5_drr')
F = __import__('3_forward_projector')


def project(markers_ck):
    """Project CK markers onto both detector poses with Task 3."""
    return (F.project_ck_to_detector(markers_ck, "A"),
            F.project_ck_to_detector(markers_ck, "B"))


def print_case(name, uv_A, uv_B):
    """Print both methods and the main REM diagnostics for one case."""
    rem = A.detect_ambiguity_by_rem(uv_A, uv_B)
    epi = A.detect_ambiguity_by_epipolar(uv_A, uv_B)

    print(f"\n{name}")
    print("-" * len(name))
    print("REM matrix (mm):")
    print(rem.rem_matrix_mm)
    print("permutation scores (mm):")
    for item in rem.permutation_scores:
        print(f"  {item.permutation}: {item.total_score_mm:.12g} "
              f"valid={item.is_valid}")
    print(f"threshold-valid permutations: "
          f"{rem.threshold_valid_permutations}")
    print(f"valid permutations: {rem.valid_permutations}")
    print(f"indistinguishable permutations: "
          f"{rem.indistinguishable_permutations}")
    print(f"REM classification / confidence: "
          f"{rem.classification} / {rem.confidence}")
    print(f"best / second-best (mm): "
          f"{rem.best_score_mm} / {rem.second_best_score_mm}")
    print("epipolar-distance matrix (mm):")
    print(epi.epipolar_distance_matrix_mm)
    print(f"epipolar valid permutations: {epi.valid_permutations}")
    print(f"epipolar classification: {epi.classification}")


def main():
    np.set_printoptions(precision=10, suppress=True)
    print("TASK 9 - REM-BASED AMBIGUITY DETECTION")
    print(f"valid REM tolerance: {A.DEFAULT_VALID_REM_TOL_MM:g} mm")
    print(f"score tie tolerance: {A.DEFAULT_SCORE_TIE_TOL_MM:g} mm")
    print(f"low-confidence gap: {A.DEFAULT_LOW_CONFIDENCE_GAP_MM:g} mm")
    print(f"epipolar-distance tolerance: {A.DEFAULT_EPIPOLAR_TOL_MM:g} mm")

    unique_A, unique_B = project(D.marker_centres_ck())
    print_case("Unique assignment markers", unique_A, unique_B)

    ambiguous_A, ambiguous_B = project(
        A.ENGINEERED_AMBIGUOUS_MARKERS_CK_MM
    )
    print_case("Engineered same-z-plane ambiguity",
               ambiguous_A, ambiguous_B)

    unresolved_B = unique_B.copy()
    unresolved_B[:, 0] += 1.0
    print_case("Unresolved 1 mm detector-B u shift",
               unique_A, unresolved_B)


if __name__ == "__main__":
    main()
