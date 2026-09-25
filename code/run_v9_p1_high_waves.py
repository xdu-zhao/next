from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def key(score: dict) -> tuple[int, int]:
    return (
        int(score["makespan"]),
        int(score.get("data_movement_bytes", {}).get("added_copy_bytes", 0)),
    )


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    root = Path.cwd()
    trial = root / "results_v9_p1_high_waves"
    incumbent = root / "results_v8_n5"
    runner = root / "code" / "run_batch_parallel_v7.py"
    solver = root / "code" / "solve_multicore_v7.py"
    cmd = [
        sys.executable, "-u", str(runner),
        "--workers", "8", "--search", "quick",
        "--problems", "1", "--cores", "5",
        "--results-dir", str(trial), "--solver", str(solver),
    ]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=root, check=True)

    improved = []
    for case_id in range(1, 101):
        stem = f"case_{case_id:03d}_p1_n5"
        trial_score = trial / f"{stem}_score.json"
        incumbent_score = incumbent / f"{stem}_score.json"
        if not trial_score.is_file() or not incumbent_score.is_file():
            continue
        if key(load(trial_score)) < key(load(incumbent_score)):
            for suffix in ("_score.json", "_multicore_res.json"):
                shutil.copy2(trial / f"{stem}{suffix}", incumbent / f"{stem}{suffix}")
            improved.append(case_id)

    (incumbent / "v9_p1_high_waves_improved_cases.json").write_text(
        json.dumps(improved, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    subprocess.run([
        sys.executable, str(root / "code" / "summarize_n5.py"),
        "--results-dir", str(incumbent), "--baseline-dir", str(root / "results"),
    ], cwd=root, check=True)
    print(f"V9_P1_HIGH_WAVES_COMPLETE improved={len(improved)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
