"""Probe the high-wave tail only where the previous winner was wave 4."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


CASES = [2, 3, 14, 31, 41, 47, 62, 63, 69, 75, 77, 82, 85]


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def key(score: dict) -> tuple[int, int]:
    return (int(score["makespan"]), int(score.get("data_movement_bytes", {}).get("added_copy_bytes", 0)))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    trial, incumbent = root / "results_v11_p1_ultra_waves", root / "results_v8_n5"
    command = [
        sys.executable, "-u", str(root / "code" / "run_batch_parallel_v7.py"),
        "--workers", "5", "--search", "quick", "--problems", "1", "--cores", "5",
        "--case-ids", *(str(case) for case in CASES),
        "--results-dir", str(trial), "--solver", str(root / "code" / "solve_multicore_v7.py"),
    ]
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)
    improved = []
    for case in CASES:
        stem = f"case_{case:03d}_p1_n5"
        if key(load(trial / f"{stem}_score.json")) < key(load(incumbent / f"{stem}_score.json")):
            for suffix in ("_score.json", "_multicore_res.json"):
                shutil.copy2(trial / f"{stem}{suffix}", incumbent / f"{stem}{suffix}")
            improved.append(case)
    (incumbent / "v11_p1_ultra_waves_improved_cases.json").write_text(
        json.dumps(improved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    subprocess.run([sys.executable, str(root / "code" / "summarize_n5.py"), "--results-dir", str(incumbent), "--baseline-dir", str(root / "results")], cwd=root, check=True)
    print(f"V11_P1_ULTRA_WAVES_COMPLETE improved={len(improved)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
