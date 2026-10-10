"""Task 1 demo: for each frame transform, print the three test points with the
reason, the hand-predicted result, the computed result, and the inverse
(which must return the original point). Saved to output/task1_results.txt.

Run from the repository root:  python demo_task1.py
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np
from task1_cases import TRANSFORMS

TOL = 1e-9


def fmt(v):
    return "(" + ", ".join(f"{x:9.3f}" for x in np.ravel(v)) + ")"


def main():
    lines = ["TASK 1 - FRAME TRANSFORMS: three test points per transform", "=" * 78]
    n_pass = n_total = 0
    for name, forward, inverse, points in TRANSFORMS:
        lines += ["", f"{name}   (inverse: {name.split(' -> ')[1]} -> {name.split(' -> ')[0]})", "-" * 78]
        for i, (p_in, predicted, reason) in enumerate(points, 1):
            computed = forward(p_in)
            back = inverse(computed)
            ok = np.allclose(computed, predicted, atol=TOL) and np.allclose(back, p_in, atol=TOL)
            n_pass += ok
            n_total += 1
            lines += [f" point {i} [{'PASS' if ok else 'FAIL'}]  why: {reason}",
                      f"    input:     {fmt(p_in)}",
                      f"    predicted: {fmt(predicted)}",
                      f"    computed:  {fmt(computed)}",
                      f"    inverse:   {fmt(back)}"]
    lines.append(f"\n{n_pass}/{n_total} points passed (forward = prediction, inverse = original point).")
    text = "\n".join(lines)
    print(text)
    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "task1_results.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
