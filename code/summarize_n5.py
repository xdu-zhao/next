from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize five-core search results against single-core baselines."
    )
    parser.add_argument("--results-dir", default="results_v8_n5")
    parser.add_argument("--baseline-dir", default="results")
    parser.add_argument("--start-case", type=int, default=1)
    parser.add_argument("--end-case", type=int, default=100)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    baseline_dir = Path(args.baseline_dir)
    rows = []
    missing = []

    for case_id in range(args.start_case, args.end_case + 1):
        name = f"case_{case_id:03d}"
        baseline_path = baseline_dir / f"{name}_singlecore_score.json"
        if not baseline_path.is_file():
            baseline_path = (
                baseline_dir / "singlecore" / f"{name}_singlecore_score.json"
            )
        if not baseline_path.is_file():
            missing.append(str(baseline_path))
            continue
        single = int(load(baseline_path)["makespan"])
        scores = {}
        for problem in (1, 2, 3):
            path = results_dir / f"{name}_p{problem}_n5_score.json"
            if not path.is_file():
                missing.append(str(path))
                continue
            scores[problem] = load(path)
        for problem, score in scores.items():
            makespan = int(score["makespan"])
            movement = score.get("data_movement_bytes", {})
            cache = score.get("cache_stats", {})
            rows.append({
                "case": name,
                "problem": problem,
                "singlecore_makespan": single,
                "fivecore_makespan": makespan,
                "speedup": single / makespan,
                "added_copy_bytes": movement.get("added_copy_bytes", 0),
                "cache_hit_rate": cache.get("hit_rate", ""),
                "p3_over_p2": (
                    int(scores[2]["makespan"]) / makespan
                    if problem == 3 and 2 in scores else ""
                ),
            })

    output = results_dir / "n5_summary.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["case"])
        writer.writeheader()
        writer.writerows(rows)

    stats = {}
    for problem in (1, 2, 3):
        values = [row["speedup"] for row in rows if row["problem"] == problem]
        if values:
            stats[f"p{problem}"] = {
                "completed_cases": len(values),
                "average_speedup": sum(values) / len(values),
                "minimum_speedup": min(values),
                "maximum_speedup": max(values),
            }
    stats["missing_files"] = len(missing)
    stats_path = results_dir / "n5_summary.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
