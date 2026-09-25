from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def score_key(score: dict) -> tuple[int, int]:
    return (
        int(score["makespan"]),
        int(score.get("data_movement_bytes", {}).get("added_copy_bytes", 0)),
    )


def main() -> int:
    root = Path.cwd()
    trial = root / "results_v10_p23_full"
    incumbent = root / "results_v8_n5"
    runner = root / "code" / "run_batch_parallel_v7.py"
    solver = root / "code" / "solve_multicore_v7.py"
    command = [
        sys.executable, "-u", str(runner),
        "--workers", "8", "--search", "full",
        "--problems", "2", "3", "--cores", "5",
        "--results-dir", str(trial), "--solver", str(solver),
    ]
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)

    improved: dict[str, list[int]] = {"p2": [], "p3": []}
    for problem in (2, 3):
        for case_id in range(1, 101):
            stem = f"case_{case_id:03d}_p{problem}_n5"
            candidate_score = trial / f"{stem}_score.json"
            incumbent_score = incumbent / f"{stem}_score.json"
            if not candidate_score.is_file() or not incumbent_score.is_file():
                continue
            if score_key(load(candidate_score)) < score_key(load(incumbent_score)):
                for suffix in ("_score.json", "_multicore_res.json"):
                    shutil.copy2(trial / f"{stem}{suffix}", incumbent / f"{stem}{suffix}")
                improved[f"p{problem}"].append(case_id)

    (incumbent / "v10_p23_full_improved_cases.json").write_text(
        json.dumps(improved, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    subprocess.run([
        sys.executable, str(root / "code" / "summarize_n5.py"),
        "--results-dir", str(incumbent), "--baseline-dir", str(root / "results"),
    ], cwd=root, check=True)
    print(
        "V10_P23_FULL_COMPLETE "
        f"p2={len(improved['p2'])} p3={len(improved['p3'])}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
