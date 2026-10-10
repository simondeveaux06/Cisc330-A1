"""Task 5: Digitally Reconstructed Radiographs (DRRs).

Approach (one X-ray beam per detector pixel)
--------------------------------------------
1. Pixel rays. For each pixel centre of the detector of pose A or B, take its
   position in the Detector frame (u, v, w=0) and transform it to the CK frame
   (Task 1 transforms). The beam is the line from the X-ray source S through
   that pixel: L = S + t*d, with d = normalize(pixel - S). Because d is a unit
   vector, t is distance in mm from the source.
2. Path lengths. For each object (material) we compute how many mm of the beam
   lie inside it:
     * analytic spheres (sphere phantom, steel markers): line-sphere chord;
     * the vertebra surface mesh: all ray/triangle hits are sorted along the
       ray and paired (entry, exit); the path is the sum of (exit - entry).
3. Line integral. Sum over materials of density * path length:
       mass_path = sum_i rho_i * L_i      [g/cm^2]
   This is the "total atomic mass in the way of the beam" from the lectures.
4. Attenuation (Beer-Lambert, lecture "Beam Attenuation Equation"):
       I = I0 * exp(-k * mass_path)
   and the attenuation image ln(I0 / I) = k * mass_path, which is what we
   display (bright = more material), as in clinical radiographs.

Assumptions (also listed in the report)
---------------------------------------
* Point source, monochromatic beam with one k for all materials: no scatter,
  no beam hardening, no noise, ideal detector (no blur).
* Each pixel's value is the single ray through its centre (no supersampling).
* Markers are modelled analytically as steel spheres of the handout's size
  (3 mm) at the handout's coordinates. The STL file also contains marker
  spheres, but they are 2 mm in diameter and offset by up to ~0.4 mm from the
  handout table, so we drop them (and two tiny stray fragments) and keep only
  the largest body, the vertebra. This keeps the DRR consistent with the
  "true" marker positions used in Tasks 4, 6 and 7.
* Sphere phantom: markers lie fully inside the phantom, so their chord is
  removed from the phantom path (no material is counted twice).
  Vertebra: the markers are glued to the bone surface and only partly overlap
  it; the overlap is ignored. Worst case this over-counts 3 mm of 0.25 g/cm^3
  bone (0.075 g/cm^2) against 2.36 g/cm^2 of steel, i.e. < 3.2 % of the
  marker signal.
"""
import numpy as np

import transforms as T
from config import (DETECTOR_SIZE_MM, N_PIXELS, MARKERS_MD_MM, MARKER_RADIUS_MM,
                    SPHERE_PHANTOM_CENTRE_CK_MM, SPHERE_PHANTOM_RADIUS_MM,
                    DENSITY_VERTEBRA_G_CM3, DENSITY_STEEL_G_CM3,
                    K_ATTENUATION_CM2_PER_G, I0_UNMOLESTED, MM_PER_CM,
                    VERTEBRA_STL_PATH)
from geometry import normalize, chord_length_through_sphere


# 1. Pixel rays
# ===========================================================================
def detector_pixel_centres_mm(n_pixels=N_PIXELS):
    """(n, n, 3) array of pixel-centre positions (u, v, 0) in the Detector
    frame for an n x n sampling of the physical 200 mm detector.

    n = N_PIXELS gives the real 0.1 mm image (identical to Task 1's
    image_to_detector); a smaller n gives a coarse image for fast debugging.
    Index [row, col] follows the image convention: row <-> u, col <-> v.
    """
    pitch_mm = DETECTOR_SIZE_MM / n_pixels
    centres_1d = (np.arange(n_pixels) + 0.5) * pitch_mm - DETECTOR_SIZE_MM / 2
    u_grid, v_grid = np.meshgrid(centres_1d, centres_1d, indexing="ij")
    return np.stack([u_grid, v_grid, np.zeros_like(u_grid)], axis=-1)


def pixel_rays_ck(pose, n_pixels=N_PIXELS):
    """Beams from the source to every pixel centre, in the CK frame.

    Returns (source_ck (3,), directions_ck (n*n, 3) unit vectors), with the
    directions in row-major (row, col) order of the image.
    """
    pixels_det = detector_pixel_centres_mm(n_pixels).reshape(-1, 3)
    pixels_ck = T.detector_to_ck(pixels_det, pose)
    source_ck = T.source_position_ck(pose)
    return source_ck, normalize(pixels_ck - source_ck)


# 2. Path lengths per material (mm)
# ===========================================================================
def marker_centres_ck():
    """True marker centres in CK (mm), from the MD-frame table (Task 1/2)."""
    return T.md_to_ck(MARKERS_MD_MM)


def marker_path_mm(source_ck, dirs_ck):
    """Total steel path (mm) of each ray through the three marker spheres.
    (The markers are far apart, so one ray cannot pass through two of them
    in a way that double counts; we simply sum the three chords.)"""
    return sum(chord_length_through_sphere(source_ck, dirs_ck, c, MARKER_RADIUS_MM)
               for c in marker_centres_ck())


def sphere_phantom_paths_mm(source_ck, dirs_ck):
    """Path lengths (mm) through the sphere phantom: {'bone': ..., 'steel': ...}.
    Markers sit inside the phantom, so their chord replaces phantom material."""
    phantom = chord_length_through_sphere(source_ck, dirs_ck,
                                          SPHERE_PHANTOM_CENTRE_CK_MM, SPHERE_PHANTOM_RADIUS_MM)
    steel = marker_path_mm(source_ck, dirs_ck)
    return {"bone": phantom - steel, "steel": steel}


def load_vertebra_mesh(stl_path=VERTEBRA_STL_PATH):
    """Load the STL (MD frame, mm) and keep only its largest body (the
    vertebra), dropping the STL's own marker spheres and stray fragments.
    The mesh is then placed in the CK frame with F_CK<-MD."""
    import trimesh   # third-party; imported here so the sphere DRR works without it
    mesh = trimesh.load(stl_path, force="mesh")
    vertebra = max(mesh.split(only_watertight=False), key=lambda body: len(body.faces))
    vertebra.apply_transform(T.F_ck_from_md())
    return vertebra


def path_length_through_mesh(mesh, source, dirs, chunk=50_000, dedupe_tol_mm=1e-6):

    path_mm = np.zeros(len(dirs))
    n_odd = 0
    for start in range(0, len(dirs), chunk):
        d = dirs[start:start + chunk]
        origins = np.broadcast_to(source, d.shape)
        hit_xyz, ray_idx, _ = mesh.ray.intersects_location(origins, d, multiple_hits=True)
        if len(ray_idx) == 0:
            continue
        t = np.einsum("ij,ij->i", hit_xyz - source, d[ray_idx])

        order = np.lexsort((t, ray_idx))                 # sort by ray, then by t
        ray_idx, t = ray_idx[order], t[order]
        keep = np.ones(len(t), bool)
        keep[1:] = ~((ray_idx[1:] == ray_idx[:-1]) & (np.diff(t) < dedupe_tol_mm))
        ray_idx, t = ray_idx[keep], t[keep]

        counts = np.bincount(ray_idx, minlength=len(d))
        first = np.cumsum(counts) - counts               # index of each ray's first hit
        k_in_ray = np.arange(len(t)) - first[ray_idx]    # 0, 1, 2, ... within each ray
        sign = np.where(k_in_ray % 2 == 0, -1.0, 1.0)    # entries -t, exits +t
        unpaired = (counts[ray_idx] % 2 == 1) & (k_in_ray == counts[ray_idx] - 1)
        sign[unpaired] = 0.0
        path_mm[start:start + chunk] = np.bincount(ray_idx, weights=sign * t, minlength=len(d))
        n_odd += int(np.sum(counts % 2 == 1))
    return path_mm, n_odd


def vertebra_paths_mm(pose, source_ck, dirs_ck, mesh_ck=None):

    mesh_ck = load_vertebra_mesh() if mesh_ck is None else mesh_ck
    F = T.F_det_from_ck(pose)
    mesh_det = mesh_ck.copy()
    mesh_det.apply_transform(F)
    source_det = T.ck_to_detector(source_ck, pose)
    dirs_det = dirs_ck @ F[:3, :3].T                   # directions rotate only
    bone, n_odd = path_length_through_mesh(mesh_det, source_det, dirs_det)
    return {"bone": bone, "steel": marker_path_mm(source_ck, dirs_ck)}, n_odd


# 3-4. Line integral and attenuation
# ===========================================================================
DENSITY_G_CM3 = {"bone": DENSITY_VERTEBRA_G_CM3, "steel": DENSITY_STEEL_G_CM3}


def mass_path_g_cm2(paths_mm):
    """Line integral sum_i rho_i * L_i in g/cm^2 (path converted mm -> cm)."""
    return sum(DENSITY_G_CM3[material] * L_mm / MM_PER_CM for material, L_mm in paths_mm.items())


def detected_intensity(mass_path):
    """Beer-Lambert: I = I0 * exp(-k * mass_path)."""
    return I0_UNMOLESTED * np.exp(-K_ATTENUATION_CM2_PER_G * mass_path)


def attenuation_image(intensity):
    """ln(I0 / I): the quantity the detector 'measures' relative to the
    unmolested beam (lecture: we only care about I/I0). Bright = dense."""
    return np.log(I0_UNMOLESTED / intensity)


# Complete DRR
# ===========================================================================
def compute_drr(pose, phantom="sphere", n_pixels=N_PIXELS, mesh=None):

    source_ck, dirs_ck = pixel_rays_ck(pose, n_pixels)
    n_odd = 0
    if phantom == "sphere":
        paths = sphere_phantom_paths_mm(source_ck, dirs_ck)
    elif phantom == "vertebra":
        paths, n_odd = vertebra_paths_mm(pose, source_ck, dirs_ck, mesh)
    else:
        raise ValueError(f"Unknown phantom {phantom!r}")

    mass = mass_path_g_cm2(paths).reshape(n_pixels, n_pixels)
    intensity = detected_intensity(mass)
    return {"intensity": intensity, "attenuation": attenuation_image(intensity),
            "mass_path": mass, "n_odd_rays": n_odd,
            "pose": pose, "phantom": phantom, "n_pixels": n_pixels}


def project_point_to_detector(p_ck, pose):
    """Where the beam from the source through CK point(s) hits the detector,
    in Detector coordinates (u, v) mm. Used here only to overlay the expected
    marker positions on the DRR and to test it; the group's Task 3 forward
    projector should give the same numbers."""
    from geometry import line_plane_intersection
    S = T.source_position_ck(pose)
    p_ck = np.atleast_2d(p_ck)
    hit_ck = line_plane_intersection(S, normalize(p_ck - S),
                                     T.detector_origin_in_ck(pose), T.detector_axes_in_ck(pose)[2])
    return T.ck_to_detector(hit_ck, pose)[:, :2]
