
"""Task 1 test points: three per frame transform, with the hand-predicted
result and the reason each point was chosen.

Shared by test_transforms.py (automatic pass/fail) and demo_task1.py (prints
the predicted-vs-computed table for the report), so the cases are defined once.

Each transform is tested in both directions on the same three points: the
forward map must give the predicted result, and the inverse map must bring
that result back to the original point.

Predictions were worked out by hand from the handout geometry (p.2-4)
BEFORE running the code. Useful numbers:
    s = sin(45 deg) = cos(45 deg) = 0.70711
    Pose A axes in CK: u=(0,0,1), v=( s, s,0), w=(-s, s,0)
    Pose B axes in CK: u=(0,0,1), v=( s,-s,0), w=( s, s,0)
    Detector centre = -1000*w, source = +1000*w   (mm)
"""
import numpy as np
import transforms as T

s = np.sqrt(0.5)

# (transform name, forward function, inverse function,
#  [(input point, predicted forward result, reason), ... three points ...])
TRANSFORMS = [
    ("MD -> CK", T.md_to_ck, T.ck_to_md, [
        ([0, 0, 0], [0, 0, 0],
         "MD origin: the two frames share their origin, so it must map to the CK origin."),
        ([-19, 28, 11], [-19, 28, 11],
         "Marker M1: same origin and parallel axes, so the coordinates do not change."),
        ([10, -20, 30], [10, -20, 30],
         "All three components non-zero and different, so a swapped or flipped axis would show."),
    ]),
    ("CK -> Det A", lambda p: T.ck_to_detector(p, "A"), lambda p: T.detector_to_ck(p, "A"), [
        ([0, 0, 0], [0, 0, 1000],
         "Isocenter lies on the central beam, SDD - SAD = 1000 mm in front of the detector centre."),
        ([-1000 * s, 1000 * s, 0], [0, 0, 2000],
         "Source A lies on the central beam, SDD = 2000 mm in front of the detector."),
        ([100, 0, 0], [0, 100 * s, 1000 - 100 * s],
         "100 mm to the patient's right: detector A is right-posterior, so the point gets closer "
         "to it (w < 1000); v = 100*cos45."),
    ]),
    ("CK -> Det B", lambda p: T.ck_to_detector(p, "B"), lambda p: T.detector_to_ck(p, "B"), [
        ([0, 0, 0], [0, 0, 1000],
         "Isocenter lies on the central beam of pose B too."),
        ([1000 * s, 1000 * s, 0], [0, 0, 2000],
         "Source B lies on the central beam, SDD = 2000 mm in front of detector B."),
        ([100, 0, 0], [0, 100 * s, 1000 + 100 * s],
         "Same point as for A: detector B is left-posterior, so the point moves away from it (w > 1000)."),
    ]),
    ("Det A -> Img A", lambda p: T.detector_to_image(p, "A"), lambda p: T.image_to_detector(p, "A")[..., :2], [
        ([0, 0], [999.5, 999.5],
         "Detector centre falls between the two middle pixels of a 2000-pixel row."),
        ([-99.95, -99.95], [0, 0],
         "Centre of the corner pixel (half a pixel in from the -u, -v edge) is pixel (0, 0)."),
        ([10, 0], [1099.5, 999.5],
         "10 mm along u = 100 pixels along the row index only (checks row <-> u)."),
    ]),
    ("Det B -> Img B", lambda p: T.detector_to_image(p, "B"), lambda p: T.image_to_detector(p, "B")[..., :2], [
        ([0, 0], [999.5, 999.5],
         "Identical detectors: B's centre maps like A's."),
        ([99.95, 99.95], [1999, 1999],
         "Centre of the opposite corner pixel (+u, +v edge) is the last pixel (1999, 1999)."),
        ([0, -10], [999.5, 899.5],
         "-10 mm along v = -100 pixels along the column index only (checks col <-> v)."),
    ]),
]

# Flat list (transform name, forward, inverse, input, predicted, reason)
CASES = [(name, fwd, inv, p_in, pred, why)
         for name, fwd, inv, points in TRANSFORMS for p_in, pred, why in points]
