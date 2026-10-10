"""Print the results of Tasks 1, 5, 6 and 7 in order (for the report) and save
them to output/ALL_RESULTS.txt.

Run from the repository root:   python print_results.py

Task 5 takes ~4 minutes to recompute, so by default its saved summary
(output/drr/task5_summary.txt) is printed; set RERUN_TASK5 = True to rebuild
the DRRs first. Tasks 1, 6 and 7 are recomputed (a few seconds), and the
automatic tests of Tasks 1 and 7 are run and summarised.
"""
import contextlib
import io
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

RERUN_TASK5 = False

# Automated tests exist only where the handout's test is numeric (Tasks 1 and 7);
# Tasks 5 and 6 are tested by visual inspection, as the handout specifies.
TASK_TESTS = {
    "Task 1": ["tests/test_transforms.py"],
    "Task 5": [],
    "Task 6": [],
    "Task 7": ["tests/test_reconstruction.py"],
}


def run_tests(test_files):
    """Run pytest on the given files and return its one-line summary."""
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", *test_files],
                         cwd=ROOT, capture_output=True, text=True).stdout.strip().splitlines()
    return out[-1] if out else "no output"


def captured(func):
    """Run func() and return what it printed."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        func()
    return buffer.getvalue().rstrip()


def main():
    import demo_task1
    import run_task5
    import run_task6
    import run_task7

    if RERUN_TASK5:
        captured(run_task5.main)
    task5_text = (ROOT / "output" / "drr" / "task5_summary.txt").read_text().rstrip()

    sections = [
        ("Task 1 - Frame transforms", captured(demo_task1.main)),
        ("Task 5 - Digitally Reconstructed Radiographs", task5_text),
        ("Task 6 - Marker localization", captured(run_task6.main)),
        ("Task 7 - Marker reconstruction and REM", captured(run_task7.main)),
    ]
    blocks = []
    for (title, body), (task, files) in zip(sections, TASK_TESTS.items()):
        test_line = (f"Automatic test ({', '.join(files)}): {run_tests(files)}" if files
                     else "Test: visual inspection (see figures and discussion)")
        blocks.append("#" * 100 + f"\n# {title}\n# {test_line}\n" + "#" * 100 + "\n\n" + body + "\n")
    text = "\n".join(blocks)
    print(text)
    (ROOT / "output" / "ALL_RESULTS.txt").write_text(text)


if __name__ == "__main__":
    main()
