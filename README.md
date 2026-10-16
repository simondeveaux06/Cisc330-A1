# CISC/CMPE 330 Assignment 1

CyberKnife coordinate transforms, marker projection, digitally reconstructed
radiographs (DRRs), marker localization, 3-D reconstruction, correspondence,
ambiguity detection, and reconstruction error analysis.

## Reproduce the computational results

Run all commands from the repository root, which contains this README and
`requirements.txt`. The supplied model must be present at
`data/LumbarVertebrae.-witrh-Markers.stl` (the spelling is intentional).
No external repositories, image viewer, or manual measurements are needed to
reproduce the automated results. Generated files under `output/` are ignored
by Git and are created by the commands below.

### 1. Set up Python

Python 3.9 or newer is required. The results were checked with Python 3.9.6.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, use `python` instead of `python3` to create the environment, then
activate it with `.venv\Scripts\activate.bat` in Command Prompt or
`.venv\Scripts\Activate.ps1` in PowerShell. All subsequent commands use the
activated environment's `python`.

The verified package versions were NumPy 2.0.2, Matplotlib 3.9.4, SciPy 1.13.1,
Trimesh 4.12.2, Rtree 1.4.1, and pytest 8.4.2. To use those exact versions
instead of the unpinned versions in `requirements.txt`:

```bash
python -m pip install numpy==2.0.2 matplotlib==3.9.4 scipy==1.13.1 trimesh==4.12.2 rtree==1.4.1 pytest==8.4.2
```

### 2. Run the automated tests

```bash
python -m pytest -q
```

Expected result: **37 passed**. The suite covers Task 1 transforms, Task 7
reconstruction and REM, Task 8 correspondence, Task 9 ambiguity, and Task 10
error metrics, perturbations, direction-pair coverage, and strict threshold
selection. It does not require pre-generated DRRs. Tasks 2 and 4 have additional
validation in their runners; the 37-test count does not include those checks.

To run only Task 10's tests:

```bash
python -m pytest -q tests/test_error_analysis.py
```

Expected result: **9 passed**. The verified macOS/NumPy environment emits
`RuntimeWarning` messages about divide-by-zero, overflow, and invalid values
in `src/1_transforms.py` during some matrix multiplications. The tests pass,
including checks that the simulated MRE and REM values are finite. These
warnings are recorded here so graders can distinguish the observed output
from a test failure.

### 3. Regenerate Tasks 1–9

Run this sequence in order. Task 6 reads the four numerical DRRs written by
Task 5. Task 5 uses the full **2000 × 2000** detector with **0.1 mm** pixels and
may take several minutes. The scripts save figures without opening windows.

```bash
python demo_task1.py
python src/2_markers.py
python src/4_marker_projector.py
python run_task5.py
python run_task6.py
python run_task7.py
python run_task8.py > output/task8_results.txt
python run_task9.py > output/task9_results.txt
python print_results.py
```

Task 3 is exercised by the Task 4 projector. The redirected Task 8 and Task 9
results can be read in the files below; omit the redirection to see them in the
terminal instead. Task 5 creates `output/` before these redirects are used.

| Task | Expected check or result | Where to inspect it |
| --- | --- | --- |
| 1 | Hand-predicted transforms and inverse checks pass. | Terminal; `output/ALL_RESULTS.txt` |
| 2 | Prints `Q2 validation PASSED`; transforms preserve inter-marker distances. | `src/q2_outputs/`, including `validation.txt` and marker coordinates/figure |
| 3–4 | Prints `Q4 validation PASSED`; projects all three markers onto A and B. | Terminal; `src/q4_outputs/` |
| 5 | Four full-resolution DRRs: sphere and vertebra, each in poses A and B. Sphere center mass path is approximately 2.500 g/cm². | `output/drr/`, including `task5_summary.txt` and both overlay figures |
| 6 | Automated localization over 12 marker images: mean error approximately **0.0121 mm**, maximum approximately **0.0481 mm**. | `output/localization/task6_results.txt`, `segmented_markers.csv`, and `task6_zoom_*.png` |
| 7 | Recovers the three known CK marker positions with MRE and REM near floating-point precision. | `output/reconstruction/task7_results.txt` |
| 8 | Both methods return the identity mapping `(0, 1, 2)` for the supplied ordered exact projections. | `output/task8_results.txt` |
| 9 | Original markers: `unique`; engineered case: `ambiguous`; 1 mm detector-B u shift: `unresolved`. | `output/task9_results.txt` |

`print_results.py` collects Tasks 1, 5, 6, and 7 into `output/ALL_RESULTS.txt`.
It reads Task 5's existing summary and reruns Tasks 1, 6, and 7; it does not
regenerate Task 5 or include Tasks 8–10. Runtime measurements and the last few
floating-point digits can vary by machine.

Optional computational geometry figures can also be regenerated:

```bash
python src/make_task4_sketch.py
python make_task5_sketch.py
```

These write `src/task4_sketch.png` and
`output/drr/task5_geometry_topview.png`, respectively.

### Task 6 manual comparison limitation

The Word report includes manual localization summary statistics, but the
individual cursor readings are not available in the submitted measurement
CSV, `docs/visual_localization.csv`. Those manual summary statistics cannot
be independently reproduced from the submitted files. They are separate from
the automated localization errors, which the commands above recompute.

### 4. Regenerate Task 10 and Graphs

Task 10 runs independently of Tasks 5–6 and needs no saved DRRs:

```bash
python run_task10.py
```

For each MLE in detector-plane millimetres, the script evaluates all
**360 × 360 = 129,600 independent direction pairs** for each of the three
original markers. Each detector observation is displaced by exactly that
MLE; the A and B directions vary independently in 1° increments. Correct
correspondences are held fixed.

The runner sweeps MLE in 0.5 mm steps until the worst MRE reaches at least
6 mm, then refines the first 5 mm crossing to a bracket narrower than 0.01 mm.
It evaluates **24 MLE values and 9,331,200 reconstructions**. 

| Output under `output/error_analysis/` | Contents |
| --- | --- |
| `MLE_vs_MRE.png` | Maximum MRE for each marker and overall, including a zoom around the 5 mm crossing. |
| `REM_vs_MRE.png` | Density histogram accumulated from every simulated REM/MRE pair. |
| `task10_samples.npz` | All samples, approximately 157 MiB compressed; one named array per tested MLE. |
| `task10_results.txt` | Per-MLE maxima, worst marker/directions, threshold bracket, and total sample count. |

Expected numerical results:

| MLE (mm) | Overall maximum MRE (mm) | Every MRE strictly below 5 mm? |
| ---: | ---: | --- |
| 7.000000 | 4.955636 | Yes |
| 7.062500 | 4.999882 | Yes |
| 7.0703125 | 5.005413 | No |
| 8.000000 | 5.663577 | No |
| 8.500000 | 6.017546 | No |

The largest passing **tested** MLE is **7.0625 mm**, with a crossing bracket
of **[7.0625, 7.0703125] mm**. The largest observed MRE is approximately
6.017546 mm for M3 at MLE 8.5 mm, with directions A = B = 270°.
These are finite-grid results, rather than guarantees over continuous angles,
untested MLEs, unequal detector error magnitudes, or other marker layouts.

## Conventions used to interpret the results

- MD → CK is the identity: the configured origins coincide and axes align.
- Detector A/B poses are +45°/−45° about CK z. Lengths are in mm unless
  explicitly labelled as pixels.
- Image arrays use rows along detector u and columns along v. The detector
  center maps to pixel coordinates `(999.5, 999.5)`, using integer pixel-center
  indices. A saved PNG viewer's top-left y coordinate is converted with
  `row = 1999 - y_viewer`.
- Task 5 uses the documented simplified attenuation model with a common
  coefficient of 0.2 cm²/g, phantom/vertebra density 0.25 g/cm³, and steel
  density 7.85 g/cm³. Numerical localization uses the saved attenuation arrays
  in `.npz` files; the `.png` images are for display and manual inspection.
- Task 7 and the Task 8 runner use exact projected marker coordinates. They
  do not automatically read Task 6's detected centers; the near-zero Task 7
  errors validate exact geometry rather than the complete image-processing
  pipeline.
- In Task 9, `threshold-valid permutations` lists every REM-feasible assignment.
  The REM classification `ambiguous` additionally requires a score tie;
  `unique` means a distinguishable preferred score. The epipolar method counts
  feasible complete assignments without requiring a score tie. Both methods
  agree on the three supplied demonstration cases.
