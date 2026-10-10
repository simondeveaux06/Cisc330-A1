"""System-wide constants for the CyberKnife assignment (CISC/CMPE 330, A1).

All lengths are in millimetres (mm), all angles in degrees, unless the
variable name says otherwise. Values come from the assignment handout.
"""

import pathlib

import numpy as np

# X-ray imaging geometry (handout p.3)
SDD_MM = 2000.0           # source-to-detector distance
SAD_MM = 1000.0           # source-to-axis distance
ISO_TO_DET_MM = SDD_MM - SAD_MM   


# pose (source on +y, detector on -y). Positive = right-hand rule about +z.
POSE_ANGLE_DEG = {"A": +45.0, "B": -45.0}
POSES = tuple(POSE_ANGLE_DEG.keys())


# ---------------------------------------------------------------------------
DETECTOR_SIZE_MM = 200.0  # square detector, side length
PIXEL_PITCH_MM = 0.1      # pixel size
N_PIXELS = int(round(DETECTOR_SIZE_MM / PIXEL_PITCH_MM))  

# A detector-frame point must lie on the detector plane (w = 0) before it can

# ---------------------------------------------------------------------------
MD_ORIGIN_IN_CK_MM = np.array([0.0, 0.0, 0.0])
MD_AXES_IN_CK = np.eye(3)    # rows = MD base vectors (e1, e2, e3) in CK coords

# Marker centres in the MD frame (mm), handout p.4
# ---------------------------------------------------------------------------
MARKER_NAMES = ("M1", "M2", "M3")
MARKERS_MD_MM = np.array([[-19.0, 28.0, 11.0],  
                          [ 43.0, 14.0,  2.0], 
                          [  2.0, 48.0, -6.0]]) 
MARKER_DIAMETER_MM = 3.0


# Stuff used for task 5
# ---------------------------------------------------------------------------
MARKER_RADIUS_MM = MARKER_DIAMETER_MM / 2.0


SPHERE_PHANTOM_CENTRE_CK_MM = np.array([0.0, 0.0, 0.0])
SPHERE_PHANTOM_RADIUS_MM = 50.0

# Densities (g/cm^3)
DENSITY_VERTEBRA_G_CM3 = 0.25   # 3-D printed model, 20% infill (handout p.3)
DENSITY_STEEL_G_CM3 = 7.85      # typical carbon marker sphere density 


K_ATTENUATION_CM2_PER_G = 0.2
I0_UNMOLESTED = 1.0

MM_PER_CM = 10.0

# Vertebra surface model 
VERTEBRA_STL_PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "LumbarVertebrae.-witrh-Markers.stl"
