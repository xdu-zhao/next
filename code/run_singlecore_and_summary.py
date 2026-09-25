from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PRINT_LOCK = threading.Lock()


def valid_json(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return isinstance(obj, dict) and "makespan" in obj
    except Exception:
        return False


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class Progress:
    def __init__(self, total: int):
        self.total = total
        self.done = 0
        self.lock = threading.Lock()
        self.started = time.perf_counter()

    def update(self, case_name: str, ok: bool, duration: float, makespan=None):
        with self.lock:
            self.done += 1
            elapsed = time.perf_counter() - self.started
            rate = self.done / elapsed if elapsed > 0 else 0
            eta = (self.total - self.done) / rate if rate > 0 else 0
            status = "OK" if ok else "FAIL"
            extra = f" makespan={makespan}" if makespan is not None else ""
            with PRINT_LOCK:
                print(
                    f"[{self.done:3d}/{self.total} "
                    f"{100*self.done/self.total:5.1f}%] "
                    f"{case_name} {status} {fmt_seconds(duration)} "
                    f"ETA~{fmt_seconds(eta)}{extra}",
                    flush=True,
                )


def run_single_case(
    case_id: int,
    root: Path,
    data_dir: Path,
    results_dir: Path,
    config: Path,
    evaluator: Path,
    force: bool,
    progress: Progress,
):
    case_name = f"case_{case_id:03d}"
    graph = data_dir / f"{case_name}.json"

    score_dir = results_dir / "singlecore"
    log_dir = score_dir / "logs"
    trace_dir = score_dir / "traces"

    score = score_dir / f"{case_name}_singlecore_score.json"
    log = log_dir / f"{case_name}_singlecore.log"
    trace = trace_dir / f"{case_name}_singlecore_trace.json"

    score_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    if not force and valid_json(score):
        result = load_json(score)
        progress.update(case_name, True, 0.0, result.get("makespan"))
        return None

    cmd = [
        sys.executable,
        "-u",
        str(evaluator),
        str(graph),
        "--config",
        str(config),
        "-o",
        str(score),
        "--trace-output",
        str(trace),
        "--log-output",
        str(log),
    ]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    started = time.perf_counter()

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            check=False,
        )
        duration = time.perf_counter() - started
        ok = completed.returncode == 0 and valid_json(score)

        if ok:
            result = load_json(score)
            progress.update(case_name, True, duration, result.get("makespan"))
            return None

        progress.update(case_name, False, duration)
        return (
            f"{case_name}: exit={completed.returncode}\n"
            f"{completed.stdout[-3000:]}"
        )

    except Exception as exc:
        duration = time.perf_counter() - started
        progress.update(case_name, False, duration)
        return f"{case_name}: {exc!r}"


def find_multicore_score(results_dir: Path, case_name: str, problem: int, cores: int) -> Path:
    return results_dir / f"{case_name}_p{problem}_n{cores}_score.json"


def summarize(results_dir: Path, start_case: int, end_case: int):
    single_dir = results_dir / "singlecore"
    rows = []
    missing = []

    for case_id in range(start_case, end_case + 1):
        case_name = f"case_{case_id:03d}"
        single_path = single_dir / f"{case_name}_singlecore_score.json"

        if not valid_json(single_path):
            missing.append(str(single_path))
            continue

        t1 = int(load_json(single_path)["makespan"])

        for problem in (1, 2, 3):
            for cores in (2, 3, 4, 5):
                multi_path = find_multicore_score(
                    results_dir, case_name, problem, cores
                )

                if not valid_json(multi_path):
                    missing.append(str(multi_path))
                    continue

                result = load_json(multi_path)
                tn = int(result["makespan"])
                speedup = (t1 / tn) if tn else float("inf")

                movement = result.get("data_movement_bytes", {})
                cache = result.get("cache_stats", {})

                rows.append({
                    "case": case_name,
                    "problem": problem,
                    "cores": cores,
                    "singlecore_makespan": t1,
                    "multicore_makespan": tn,
                    "speedup": speedup,
                    "added_copy_bytes": movement.get("added_copy_bytes", ""),
                    "cache_hit_rate": cache.get("hit_rate", ""),
                })

    if missing:
        print("\nWARNING: missing/invalid files:", flush=True)
        for path in missing[:30]:
            print("  " + path, flush=True)
        if len(missing) > 30:
            print(f"  ... and {len(missing) - 30} more", flush=True)

    per_case_csv = results_dir / "speedup_per_case.csv"
    with per_case_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "problem",
                "cores",
                "singlecore_makespan",
                "multicore_makespan",
                "speedup",
                "added_copy_bytes",
                "cache_hit_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    averages = []
    for problem in (1, 2, 3):
        for cores in (2, 3, 4, 5):
            values = [
                r["speedup"]
                for r in rows
                if r["problem"] == problem and r["cores"] == cores
            ]
            if values:
                averages.append({
                    "problem": problem,
                    "cores": cores,
                    "num_cases": len(values),
                    "average_speedup": sum(values) / len(values),
                    "min_speedup": min(values),
                    "max_speedup": max(values),
                })

    avg_csv = results_dir / "average_speedup.csv"
    with avg_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "problem",
                "cores",
                "num_cases",
                "average_speedup",
                "min_speedup",
                "max_speedup",
            ],
        )
        writer.writeheader()
        writer.writerows(averages)

    # Also compare P3 (L2) to P2 (no L2) at the same core count.
    cache_rows = []
    index = {
        (r["case"], r["problem"], r["cores"]): r
        for r in rows
    }

    for case_id in range(start_case, end_case + 1):
        case_name = f"case_{case_id:03d}"
        for cores in (2, 3, 4, 5):
            p2 = index.get((case_name, 2, cores))
            p3 = index.get((case_name, 3, cores))
            if not p2 or not p3:
                continue
            cache_rows.append({
                "case": case_name,
                "cores": cores,
                "p2_no_l2_makespan": p2["multicore_makespan"],
                "p3_l2_makespan": p3["multicore_makespan"],
                "l2_speedup": (
                    p2["multicore_makespan"] / p3["multicore_makespan"]
                    if p3["multicore_makespan"]
                    else float("inf")
                ),
                "p3_cache_hit_rate": p3["cache_hit_rate"],
            })

    cache_csv = results_dir / "l2_speedup_per_case.csv"
    with cache_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "cores",
                "p2_no_l2_makespan",
                "p3_l2_makespan",
                "l2_speedup",
                "p3_cache_hit_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(cache_rows)

    print("\n=== Average speedup over cases ===")
    print("Problem   2 cores   3 cores   4 cores   5 cores")
    for problem in (1, 2, 3):
        vals = {}
        for row in averages:
            if row["problem"] == problem:
                vals[row["cores"]] = row["average_speedup"]
        print(
            f"P{problem:<7}"
            f"{vals.get(2, float('nan')):8.4f}x"
            f"{vals.get(3, float('nan')):10.4f}x"
            f"{vals.get(4, float('nan')):10.4f}x"
            f"{vals.get(5, float('nan')):10.4f}x"
        )

    print("\n=== Average L2 speedup (P2 makespan / P3 makespan) ===")
    for cores in (2, 3, 4, 5):
        vals = [r["l2_speedup"] for r in cache_rows if r["cores"] == cores]
        if vals:
            print(f"{cores} cores: {sum(vals)/len(vals):.4f}x  ({len(vals)} cases)")

    print(f"\nWrote: {per_case_csv}")
    print(f"Wrote: {avg_csv}")
    print(f"Wrote: {cache_csv}")

    return 0 if not missing else 2


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run 100 official single-core baselines in parallel, then compute "
            "average speedups from existing multicore score JSONs."
        )
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--start-case", type=int, default=1)
    parser.add_argument("--end-case", type=int, default=100)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--config", default="data/config.txt")
    parser.add_argument(
        "--evaluator",
        default="code/singlecore_evaluate.py",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Do not run single-core evaluation; only summarize existing files.",
    )
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    root = Path.cwd()
    data_dir = (root / args.data_dir).resolve()
    results_dir = (root / args.results_dir).resolve()
    config = (root / args.config).resolve()
    evaluator = (root / args.evaluator).resolve()

    if not args.summary_only:
        if not evaluator.is_file():
            raise FileNotFoundError(f"evaluator not found: {evaluator}")
        if not config.is_file():
            raise FileNotFoundError(f"config not found: {config}")

        pending_ids = []
        score_dir = results_dir / "singlecore"

        for case_id in range(args.start_case, args.end_case + 1):
            case_name = f"case_{case_id:03d}"
            score = score_dir / f"{case_name}_singlecore_score.json"
            graph = data_dir / f"{case_name}.json"

            if not graph.is_file():
                print(f"[WARN] missing graph: {graph}", flush=True)
                continue

            if args.force or not valid_json(score):
                pending_ids.append(case_id)

        if pending_ids:
            print(
                f"Single-core pending={len(pending_ids)} "
                f"workers={args.workers}",
                flush=True,
            )

            progress = Progress(len(pending_ids))
            failures = []

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(
                        run_single_case,
                        case_id,
                        root,
                        data_dir,
                        results_dir,
                        config,
                        evaluator,
                        args.force,
                        progress,
                    )
                    for case_id in pending_ids
                ]

                for future in as_completed(futures):
                    failure = future.result()
                    if failure:
                        failures.append(failure)

            if failures:
                fail_path = results_dir / "singlecore_failures.txt"
                fail_path.write_text(
                    "\n\n".join(failures) + "\n",
                    encoding="utf-8",
                )
                print(
                    f"Single-core failures={len(failures)}; "
                    f"see {fail_path}",
                    flush=True,
                )
            else:
                print("All requested single-core baselines finished.", flush=True)
        else:
            print("All requested single-core baselines already exist.", flush=True)

    return summarize(results_dir, args.start_case, args.end_case)


if __name__ == "__main__":
    raise SystemExit(main())
