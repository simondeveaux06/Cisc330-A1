'''Constants given from the assignment'''

import numpy as np

SDD_MM = 2000.0           # source-detector distance
SAD_MM = 1000.0           # source-axis distance
POSE_ANGLE_DEG = {"A": +45.0, "B": -45.0}   # rotation about +z
DETECTOR_SIZE_MM = 200.0  # square detector, side length
PIXEL_PITCH_MM = 0.1
N_PIXELS = int(round(DETECTOR_SIZE_MM / PIXEL_PITCH_MM)) 

# Marker coordinates in MD frame (mm)
MARKERS_MD_MM = np.array([[-19.0, 28.0, 11.0],   # M1
                          [ 43.0, 14.0,  2.0],   # M2
                          [  2.0, 48.0, -6.0]])  # M3
