"""Evaluator check of the only P3 winner at the old cache-value boundary."""

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
    trial, incumbent, stem = root / "results_v13_p3_cache_tail", root / "results_v8_n5", "case_027_p3_n5"
    command = [sys.executable, "-u", str(root / "code" / "run_batch_parallel_v7.py"),
               "--workers", "1", "--search", "quick", "--problems", "3", "--cores", "5",
               "--case-ids", "27", "--results-dir", str(trial), "--solver", str(root / "code" / "solve_multicore_v7.py")]
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)
    improved = key(load(trial / f"{stem}_score.json")) < key(load(incumbent / f"{stem}_score.json"))
    if improved:
        for suffix in ("_score.json", "_multicore_res.json"):
            shutil.copy2(trial / f"{stem}{suffix}", incumbent / f"{stem}{suffix}")
    (incumbent / "v13_p3_cache_tail.json").write_text(json.dumps({"case": 27, "improved": improved}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, str(root / "code" / "summarize_n5.py"), "--results-dir", str(incumbent), "--baseline-dir", str(root / "results")], cwd=root, check=True)
    print(f"V13_P3_CACHE_TAIL_COMPLETE improved={improved}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
