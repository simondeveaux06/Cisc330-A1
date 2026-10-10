"""Task 6: localize the centres of the spherical markers in a DRR.

"""
from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi

import transforms as T
from config import (DETECTOR_SIZE_MM, N_PIXELS, MARKER_RADIUS_MM, SDD_MM, SAD_MM,
                    DENSITY_STEEL_G_CM3, DENSITY_VERTEBRA_G_CM3,
                    K_ATTENUATION_CM2_PER_G, MM_PER_CM)

N_MARKERS_EXPECTED = 3

# Expected marker appearance 
MAGNIFICATION_AT_ISO = SDD_MM / SAD_MM
EXPECTED_PEAK_CONTRAST = (K_ATTENUATION_CM2_PER_G
                          * (DENSITY_STEEL_G_CM3 - DENSITY_VERTEBRA_G_CM3)
                          * 2 * MARKER_RADIUS_MM / MM_PER_CM)         
THRESHOLD_FRACTION = 0.5                                              # half maximum
BACKGROUND_WINDOW_MM = 8.1      # > marker shadow diameter (~6.3 mm on the detector)
AREA_TOLERANCE = (0.4, 2.5)     # accepted area range, as multiples of the expected area


@dataclass
class MarkerDetection:
    """One localized marker."""
    row: float          # sub-pixel image row (u direction)
    col: float          # sub-pixel image column (v direction)
    u_mm: float         # Detector frame
    v_mm: float
    area_px: int        # pixels above threshold
    peak_contrast: float  # max top-hat value inside the blob


@dataclass
class LocalizationResult:
    markers: list           # accepted MarkerDetection objects
    rejected_areas_px: list  # areas of rejected components
    threshold: float


def pixel_pitch_mm(image):
    """Pixel size of a square DRR covering the whole 200 mm detector."""
    return DETECTOR_SIZE_MM / image.shape[0]


def expected_marker_area_px(image):
    """Area above half maximum of a marker shadow: 0.75 * pi * r^2, with r the
    shadow radius at the isocenter magnification."""
    shadow_radius_px = MARKER_RADIUS_MM * MAGNIFICATION_AT_ISO / pixel_pitch_mm(image)
    return 0.75 * np.pi * shadow_radius_px ** 2


def remove_background(image, window_mm=BACKGROUND_WINDOW_MM):
    """White top-hat: image minus its grey-scale opening (step 1)."""
    window_px = int(np.ceil(window_mm / pixel_pitch_mm(image))) | 1      # odd size
    return image - ndi.grey_opening(image, size=(window_px, window_px))


def localize_markers(image, pose, n_expected=N_MARKERS_EXPECTED):
    """Find the marker centres in one DRR (steps 1-5 in the module docstring).

    image: (n, n) attenuation image ln(I0/I), [row, col]; any n (coarse or full).
    pose:  'A' or 'B' (needed for the pixel -> detector transform).
    Returns a LocalizationResult. Raises RuntimeError if the number of
    accepted markers differs from n_expected, because silently continuing
    with a missing or spurious marker would be unsafe for guidance.
    """
    image = np.asarray(image, float)
    signal = remove_background(image)
    threshold = THRESHOLD_FRACTION * EXPECTED_PEAK_CONTRAST
    labels, n_components = ndi.label(signal > threshold, structure=np.ones((3, 3)))

    expected_area = expected_marker_area_px(image)
    lo, hi = AREA_TOLERANCE[0] * expected_area, AREA_TOLERANCE[1] * expected_area
    markers, rejected = [], []
    scale = image.shape[0] / N_PIXELS                     # 1 at full resolution
    for label in range(1, n_components + 1):
        blob = labels == label
        area = int(blob.sum())
        if not lo <= area <= hi:
            rejected.append(area)
            continue
        row, col = ndi.center_of_mass(np.clip(signal - threshold, 0, None) * blob)
        # Pixel -> detector: the Task 1 image transform assumes the full
        # 0.1 mm grid, so rescale coarse indices to full-resolution indices.
        full_rc = (np.array([row, col]) + 0.5) / scale - 0.5
        u_mm, v_mm, _ = T.image_to_detector(full_rc, pose)
        markers.append(MarkerDetection(row, col, u_mm, v_mm, area, float(signal[blob].max())))

    markers.sort(key=lambda m: (m.col, m.row))
    if len(markers) != n_expected:
        raise RuntimeError(f"pose {pose}: found {len(markers)} markers, expected {n_expected} "
                           f"(rejected component areas: {rejected})")
    return LocalizationResult(markers, rejected, threshold)


def detections_uv_mm(result):
    """(N, 2) array of detected (u, v) in mm."""
    return np.array([[m.u_mm, m.v_mm] for m in result.markers])


def match_to_reference(detected_uv, reference_uv):
    """Pair each reference point with its nearest detection (evaluation only:
    the reference is the known true projection). Markers are > 10 mm apart on
    the detector and errors are < 1 mm, so nearest-neighbour is unambiguous;
    we still check that the pairing is one-to-one."""
    detected_uv, reference_uv = np.atleast_2d(detected_uv), np.atleast_2d(reference_uv)
    dists = np.linalg.norm(reference_uv[:, None, :] - detected_uv[None, :, :], axis=-1)
    order = dists.argmin(axis=1)
    if len(set(order)) != len(order):
        raise RuntimeError("Nearest-neighbour matching is not one-to-one")
    return detected_uv[order]
