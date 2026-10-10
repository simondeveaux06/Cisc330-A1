
import sys
import time
import pathlib

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import drr
from config import POSES, N_PIXELS, DETECTOR_SIZE_MM, MARKER_NAMES

N_PIXELS_RUN = N_PIXELS            # 2000 x 2000 = the real 0.1 mm detector
OUT_DIR = ROOT / "output" / "drr"
HALF_MM = DETECTOR_SIZE_MM / 2
EXTENT_MM = [-HALF_MM, HALF_MM, -HALF_MM, HALF_MM]   # imshow extent: [v_min, v_max, u_min, u_max]


def to_png_uint8(attenuation, max_att):
    return np.clip(np.round(attenuation / max_att * 255), 0, 255).astype(np.uint8)


def save_overlay(images, phantom, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    vmax = max(img["attenuation"].max() for img in images.values())
    for ax, pose in zip(axes, POSES):
        # Array is [row=u, col=v]: imshow puts columns (v) horizontally and rows (u)
        # vertically; origin="lower" puts row 0 at the bottom so +u (superior) is up.
        ax.imshow(images[pose]["attenuation"], cmap="gray", vmin=0, vmax=vmax,
                  origin="lower", extent=EXTENT_MM)
        uv = drr.project_point_to_detector(drr.marker_centres_ck(), pose)
        for name, (u, v) in zip(MARKER_NAMES, uv):
            ax.plot(v, u, "+", color="red", ms=14, mew=1.2)
            ax.annotate(name, (v, u), xytext=(8, 8), textcoords="offset points", color="red")
        ax.set_title(f"{phantom} DRR, pose {pose}  (ln I0/I, k=0.2 cm²/g)")
        ax.set_xlabel("v  [mm on detector]")
        ax.set_ylabel("u  [mm on detector]  (+u = superior)")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mesh = drr.load_vertebra_mesh()
    summary = [f"Task 5 DRR summary   ({N_PIXELS_RUN} x {N_PIXELS_RUN} pixels, "
               f"{DETECTOR_SIZE_MM / N_PIXELS_RUN:.2f} mm pitch)", "=" * 70]

    for phantom in ("sphere", "vertebra"):
        images = {}
        for pose in POSES:
            t0 = time.time()
            images[pose] = drr.compute_drr(pose, phantom, N_PIXELS_RUN, mesh=mesh)
            att = images[pose]["attenuation"]
            # float32 keeps ~7 significant digits, far more than the image needs,
            # and compression shrinks the empty (zero) background: ~32 MB -> a few MB.
            np.savez_compressed(OUT_DIR / f"drr_{phantom}_{pose}.npz",
                                attenuation=att.astype(np.float32))

            mass = images[pose]["mass_path"]
            centre = slice(N_PIXELS_RUN // 2 - 1, N_PIXELS_RUN // 2 + 1)
            uv = drr.project_point_to_detector(drr.marker_centres_ck(), pose)
            summary += [
                f"\n{phantom} pose {pose}   ({time.time() - t0:.1f} s)",
                f"  max mass path        {mass.max():.3f} g/cm^2   (min I/I0 = {images[pose]['intensity'].min():.3f})",
                f"  centre mass path     {mass[centre, centre].mean():.3f} g/cm^2",
                f"  unattenuated pixels  {np.mean(mass == 0) * 100:.1f} %",
                "  expected marker centres on detector (u, v) mm:",
                *[f"    {n}: ({u:8.2f}, {v:8.2f})" for n, (u, v) in zip(MARKER_NAMES, uv)],
            ]
        vmax = max(img["attenuation"].max() for img in images.values())
        for pose in POSES:
            plt.imsave(OUT_DIR / f"drr_{phantom}_{pose}.png",
                       to_png_uint8(images[pose]["attenuation"], vmax), cmap="gray", origin="lower")
        save_overlay(images, phantom, OUT_DIR / f"task5_{phantom}_overlay.png")

    text = "\n".join(summary)
    print(text)
    (OUT_DIR / "task5_summary.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
