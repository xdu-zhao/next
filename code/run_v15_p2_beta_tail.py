"""Check P2's affinity-weight upper boundary on its six boundary winners."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


CASES = [24, 51, 62, 66, 82, 86]


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def key(score: dict) -> tuple[int, int]:
    return (int(score["makespan"]), int(score.get("data_movement_bytes", {}).get("added_copy_bytes", 0)))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    trial, incumbent = root / "results_v15_p2_beta_tail", root / "results_v8_n5"
    command = [sys.executable, "-u", str(root / "code" / "run_batch_parallel_v7.py"),
               "--workers", "5", "--search", "quick", "--problems", "2", "--cores", "5",
               "--case-ids", *(str(case) for case in CASES), "--results-dir", str(trial),
               "--solver", str(root / "code" / "solve_multicore_v7.py")]
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)
    improved = []
    for case in CASES:
        stem = f"case_{case:03d}_p2_n5"
        if key(load(trial / f"{stem}_score.json")) < key(load(incumbent / f"{stem}_score.json")):
            for suffix in ("_score.json", "_multicore_res.json"):
                shutil.copy2(trial / f"{stem}{suffix}", incumbent / f"{stem}{suffix}")
            improved.append(case)
    (incumbent / "v15_p2_beta_tail_improved_cases.json").write_text(json.dumps(improved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, str(root / "code" / "summarize_n5.py"), "--results-dir", str(incumbent), "--baseline-dir", str(root / "results")], cwd=root, check=True)
    print(f"V15_P2_BETA_TAIL_COMPLETE improved={len(improved)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
