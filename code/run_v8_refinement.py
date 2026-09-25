from __future__ import annotations

import subprocess
import sys
from pathlib import Path


# These are the only first-pass regressions large enough to justify a full
# portfolio search.  Each full search is a superset of the screening pool, so
# it can only retain or improve the saved result for that combination.
REFINEMENT_CASES = {
    1: [2, 3, 10, 28, 34, 61, 62, 63, 66, 69, 78],
    2: [15, 24, 51, 62, 79],
    3: [34, 79],
}


def run(command: list[str], root: Path) -> None:
    print("RUN:", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode:
        raise RuntimeError(f"command failed with exit code {completed.returncode}")


def main() -> int:
    root = Path.cwd()
    runner = root / "code" / "run_batch_parallel_v7.py"
    solver = root / "code" / "solve_multicore_v7.py"
    results = root / "results_v8_n5"

    for problem, cases in REFINEMENT_CASES.items():
        run([
            sys.executable, "-u", str(runner),
            "--workers", "4",
            "--search", "full",
            "--problems", str(problem),
            "--cores", "5",
            "--case-ids", *map(str, cases),
            "--results-dir", str(results),
            "--solver", str(solver),
            "--force",
        ], root)

    run([
        sys.executable, str(root / "code" / "summarize_n5.py"),
        "--results-dir", str(results),
        "--baseline-dir", str(root / "results"),
    ], root)
    print("REFINEMENT_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
