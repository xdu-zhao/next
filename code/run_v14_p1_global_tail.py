"""Global confirmation of the validated P1 high-wave family on five cores."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def key(score: dict) -> tuple[int, int]:
    return (int(score["makespan"]), int(score.get("data_movement_bytes", {}).get("added_copy_bytes", 0)))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    trial, incumbent = root / "results_v14_p1_global_tail", root / "results_v8_n5"
    command = [sys.executable, "-u", str(root / "code" / "run_batch_parallel_v7.py"),
               "--workers", "5", "--search", "quick", "--problems", "1", "--cores", "5",
               "--results-dir", str(trial), "--solver", str(root / "code" / "solve_multicore_v7.py")]
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)
    improved = []
    for case in range(1, 101):
        stem = f"case_{case:03d}_p1_n5"
        if key(load(trial / f"{stem}_score.json")) < key(load(incumbent / f"{stem}_score.json")):
            for suffix in ("_score.json", "_multicore_res.json"):
                shutil.copy2(trial / f"{stem}{suffix}", incumbent / f"{stem}{suffix}")
            improved.append(case)
    (incumbent / "v14_p1_global_tail_improved_cases.json").write_text(json.dumps(improved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, str(root / "code" / "summarize_n5.py"), "--results-dir", str(incumbent), "--baseline-dir", str(root / "results")], cwd=root, check=True)
    print(f"V14_P1_GLOBAL_TAIL_COMPLETE improved={len(improved)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
