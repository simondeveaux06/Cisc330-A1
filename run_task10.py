"""Run Task 10: full direction sweeps, threshold refinement, and saved plots.

Run ``python run_task10.py`` from the repository root. Results are saved in
output/error_analysis. Load task10_samples.npz with numpy.load; each named
array contains one MLE's complete structured rows (angles in degrees, mm).
"""
import pathlib
import sys
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
E = __import__('10_error_analysis')
from config import MARKER_NAMES

OUT_DIR = ROOT / 'output' / 'error_analysis'
LIMIT_MM = 5.0
COARSE_STEP_MM = 0.5
THRESHOLD_RESOLUTION_MM = 0.01
CLEAR_EXCEEDANCE_MM = 6.0


def save_plots(path, summaries):
    """Plot worst cases and a density histogram of every simulated case."""
    mles = np.array([rows[0]['mle_mm'] for rows in summaries])
    maxima = np.array([rows['mre_mm'] for rows in summaries])
    order = np.argsort(mles)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(MARKER_NAMES):
        ax.plot(mles[order], maxima[order, i], '.-', label=name)
    ax.plot(mles[order], maxima.max(axis=1)[order], 'k--',
            label='Overall worst case', linewidth=2)
    ax.axhline(LIMIT_MM, color='red', linestyle=':', label='MRE = 5 mm')
    ax.set(xlabel='MLE (mm)', ylabel='Worst-case MRE (mm)',
           title='Exhaustive detector error directions')
    ax.legend()
    ax.grid(alpha=0.25)
    # The marker maxima differ by hundredths of a millimetre, so show the
    # 5 mm crossing at a scale where all three curves can be distinguished.
    zoom = ax.inset_axes([0.10, 0.51, 0.40, 0.38])
    for i, name in enumerate(MARKER_NAMES):
        zoom.plot(mles[order], maxima[order, i], '.-', label=name)
    zoom.plot(mles[order], maxima.max(axis=1)[order], 'k--', linewidth=1)
    zoom.axhline(LIMIT_MM, color='red', linestyle=':')
    zoom.set(xlim=(7.04, 7.10), ylim=(4.97, 5.03),
             title='Zoom near MRE = 5 mm')
    zoom.tick_params(labelsize=8)
    zoom.grid(alpha=0.25)
    ax.indicate_inset_zoom(zoom, edgecolor='0.4')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'MLE_vs_MRE.png', dpi=180)
    plt.close(fig)

    # A fixed common grid accumulates all cases without retaining all MLEs.
    with np.load(path) as archive:
        max_rem = max(archive[key]['rem_mm'].max() for key in archive.files)
        rem_edges = np.linspace(0, max_rem * 1.001, 301)
        mre_edges = np.linspace(0, maxima.max() * 1.001, 301)
        counts = np.zeros((300, 300))
        for key in archive.files:
            rows = archive[key]
            counts += np.histogram2d(rows['rem_mm'], rows['mre_mm'],
                                     bins=(rem_edges, mre_edges))[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    mesh = ax.pcolormesh(rem_edges, mre_edges, np.ma.masked_equal(counts.T, 0),
                         norm=LogNorm(), cmap='viridis', shading='auto')
    fig.colorbar(mesh, ax=ax, label='Reconstruction count (log scale)')
    ax.axhline(LIMIT_MM, color='red', linestyle='--', label='MRE = 5 mm')
    ax.set(xlabel='REM (mm)', ylabel='MRE (mm)',
           title='All reconstructions: REM versus MRE')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'REM_vs_MRE.png', dpi=180)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / 'task10_samples.npz'
    summaries = []
    print('TASK 10 - MARKER RECONSTRUCTION ERROR ANALYSIS', flush=True)
    print('129,600 direction pairs per marker per MLE; all lengths in mm.',
          flush=True)
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as archive:
        def evaluate(mle):
            samples = E.simulate_mle(mle)
            worst_rows = []
            for name in MARKER_NAMES:
                rows = samples[samples['marker_id'] == name]
                worst_rows.append(rows[np.argmax(rows['mre_mm'])])
            worst = np.array(worst_rows, dtype=E.SAMPLE_DTYPE)
            with archive.open(f'mle_{len(summaries):03d}.npy', 'w',
                              force_zip64=True) as stream:
                np.lib.format.write_array(stream, samples, allow_pickle=False)
            summaries.append(worst)
            maximum = float(worst['mre_mm'].max())
            print(f'MLE {mle:.6f}: worst MRE {maximum:.6f}', flush=True)
            return maximum

        mle = 0.0
        while evaluate(mle) < CLEAR_EXCEEDANCE_MM:
            mle += COARSE_STEP_MM

        # Refine the first sampled strict-safe/unsafe bracket with full sweeps.
        unsafe = min(rows[0]['mle_mm'] for rows in summaries
                     if rows['mre_mm'].max() >= LIMIT_MM)
        safe = max(rows[0]['mle_mm'] for rows in summaries
                   if rows[0]['mle_mm'] < unsafe
                   and rows['mre_mm'].max() < LIMIT_MM)
        while unsafe - safe > THRESHOLD_RESOLUTION_MM:
            mid = (safe + unsafe) / 2
            if evaluate(mid) < LIMIT_MM:
                safe = mid
            else:
                unsafe = mid

    ordered = sorted(summaries, key=lambda rows: rows[0]['mle_mm'])
    all_worst = np.concatenate(ordered)
    worst = all_worst[np.argmax(all_worst['mre_mm'])]
    largest = E.max_mle([rows[0]['mle_mm'] for rows in ordered],
                        [rows['mre_mm'].max() for rows in ordered])
    lines = ['TASK 10 SUMMARY',
             'MLE (mm) | M1 max MRE | M2 max MRE | M3 max MRE | overall (mm)']
    for rows in ordered:
        values = rows['mre_mm']
        lines.append(f"{rows[0]['mle_mm']:.6f} | "
                     + ' | '.join(f'{v:.6f}' for v in values)
                     + f' | {values.max():.6f}')
    lines.extend([
        f"Overall worst: {worst['marker_id']}, MLE {worst['mle_mm']:.6f} mm, "
        f"A {worst['angle_A_deg']} deg, B {worst['angle_B_deg']} deg, "
        f"MRE {worst['mre_mm']:.6f} mm, REM {worst['rem_mm']:.6f} mm.",
        f'Largest tested MLE with every MRE < 5 mm: {largest:.6f} mm.',
        f'Any tested MLE produces MRE >= 5 mm: '
        f'{bool(np.any(all_worst["mre_mm"] >= LIMIT_MM))}.',
        f'Tested threshold bracket: [{safe:.6f}, {unsafe:.6f}] mm '
        f'(width {unsafe - safe:.6f} mm).',
        'The bracket describes tested integer-degree directions, not a '
        'guarantee over continuous angles or untested MLEs.',
        f'{len(ordered) * 3 * 360 * 360:,} reconstructions saved in {path.name}.',
    ])
    text = '\n'.join(lines)
    print('\n' + text, flush=True)
    (OUT_DIR / 'task10_results.txt').write_text(text + '\n')
    save_plots(path, ordered)
    print(f'Outputs saved in {OUT_DIR}', flush=True)


if __name__ == '__main__':
    main()
