from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PRINT_LOCK = threading.Lock()


def fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def valid_json(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            json.load(f)
        return True
    except Exception:
        return False


class Progress:
    def __init__(self, total: int):
        self.total = total
        self.done = 0
        self.failed = 0
        self.started = time.perf_counter()
        self.lock = threading.Lock()

    def update(self, label: str, ok: bool, duration: float, best_line: str = ""):
        with self.lock:
            self.done += 1
            if not ok:
                self.failed += 1

            elapsed = time.perf_counter() - self.started
            rate = self.done / elapsed if elapsed > 0 else 0.0
            remaining = self.total - self.done
            eta = remaining / rate if rate > 0 else 0.0
            pct = 100.0 * self.done / self.total if self.total else 100.0

            status = "OK" if ok else "FAIL"
            msg = (
                f"[{self.done:4d}/{self.total} {pct:5.1f}%] "
                f"{label} {status} {fmt_seconds(duration)} "
                f"ETA~{fmt_seconds(eta)}"
            )
            if best_line:
                msg += f" | {best_line}"

            with PRINT_LOCK:
                print(msg, flush=True)


def result_paths(results_dir: Path, case_name: str, problem: int, cores: int):
    stem = f"{case_name}_p{problem}_n{cores}"
    plan = results_dir / f"{stem}_multicore_res.json"
    score = results_dir / f"{stem}_score.json"
    log = results_dir / "logs" / f"{stem}.log"
    return plan, score, log


def is_done(results_dir: Path, case_name: str, problem: int, cores: int) -> bool:
    plan, score, _ = result_paths(results_dir, case_name, problem, cores)
    return valid_json(plan) and valid_json(score)


def read_best_line(log_path: Path) -> str:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""

    for line in reversed(lines):
        line = line.strip()
        if line.startswith("BEST:"):
            return line
    return ""


def run_one_combo(
    solver: Path,
    graph: Path,
    config: Path,
    results_dir: Path,
    problem: int,
    cores: int,
    search: str,
    force: bool,
    progress: Progress,
):
    case_name = graph.stem
    label = f"{case_name} p={problem} n={cores}"

    plan, score, log_path = result_paths(
        results_dir, case_name, problem, cores
    )

    if not force and is_done(results_dir, case_name, problem, cores):
        return None

    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        str(solver),
        str(graph),
        "-p", str(problem),
        "-n", str(cores),
        "--search", search,
        "--config", str(config),
        "-o", str(plan),
        "--result-output", str(score),
    ]

    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")

    started = time.perf_counter()

    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            completed = subprocess.run(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                check=False,
            )

        duration = time.perf_counter() - started
        ok = (
            completed.returncode == 0
            and valid_json(plan)
            and valid_json(score)
        )
        best = read_best_line(log_path)

        progress.update(label, ok, duration, best)

        if ok:
            return None

        return f"{label}: exit={completed.returncode}; log={log_path}"

    except Exception as exc:
        duration = time.perf_counter() - started
        progress.update(label, False, duration, str(exc))
        return f"{label}: {exc!r}"


def run_case(
    graph: Path,
    solver: Path,
    config: Path,
    results_dir: Path,
    problems: list[int],
    core_counts: list[int],
    search: str,
    force: bool,
    progress: Progress,
):
    failures = []

    # One worker owns one case at a time. This avoids running several huge
    # configurations of the same graph simultaneously and keeps memory steadier.
    for problem in problems:
        for cores in core_counts:
            if not force and is_done(results_dir, graph.stem, problem, cores):
                continue

            failure = run_one_combo(
                solver=solver,
                graph=graph,
                config=config,
                results_dir=results_dir,
                problem=problem,
                cores=cores,
                search=search,
                force=force,
                progress=progress,
            )
            if failure:
                failures.append(failure)

    return failures


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Parallel batch runner for solve_multicore.py. "
            "Designed for multi-core desktop CPUs such as Ryzen 7 9700X."
        )
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent cases. V3 default: 8; reduce to 6 if RAM pressure is high.",
    )
    parser.add_argument(
        "--search",
        choices=("none", "quick", "full"),
        default="full",
    )
    parser.add_argument("--start-case", type=int, default=1)
    parser.add_argument("--end-case", type=int, default=100)
    parser.add_argument(
        "--problems",
        nargs="+",
        type=int,
        choices=(1, 2, 3),
        default=[1, 2, 3],
    )
    parser.add_argument(
        "--cores",
        nargs="+",
        type=int,
        choices=(2, 3, 4, 5),
        default=[2, 3, 4, 5],
    )
    parser.add_argument(
        "--data-dir",
        default="data",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
    )
    parser.add_argument(
        "--config",
        default="data/config.txt",
    )
    parser.add_argument(
        "--solver",
        default="code/solve_multicore_v4.py",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even when both plan and score JSON already exist and are valid.",
    )
    args = parser.parse_args()

    root = Path.cwd()
    data_dir = (root / args.data_dir).resolve()
    results_dir = (root / args.results_dir).resolve()
    config = (root / args.config).resolve()
    solver = (root / args.solver).resolve()

    if not solver.is_file():
        raise FileNotFoundError(f"solver not found: {solver}")
    if not config.is_file():
        raise FileNotFoundError(f"config not found: {config}")

    graphs = []
    for i in range(args.start_case, args.end_case + 1):
        graph = data_dir / f"case_{i:03d}.json"
        if graph.is_file():
            graphs.append(graph)
        else:
            with PRINT_LOCK:
                print(f"[WARN] missing {graph}", flush=True)

    # Large cases first so they do not form a long tail at the end.
    graphs.sort(key=lambda p: p.stat().st_size, reverse=True)

    total_all = len(graphs) * len(args.problems) * len(args.cores)
    pending = 0
    already_done = 0

    for graph in graphs:
        for problem in args.problems:
            for cores in args.cores:
                if not args.force and is_done(
                    results_dir, graph.stem, problem, cores
                ):
                    already_done += 1
                else:
                    pending += 1

    print(
        f"Cases={len(graphs)} | combinations={total_all} | "
        f"already_done={already_done} | pending={pending} | "
        f"workers={args.workers} | search={args.search}",
        flush=True,
    )

    if pending == 0:
        print("Nothing to do: all requested results are already valid.", flush=True)
        return 0

    print(
        "Scheduling larger cases first; each worker processes one case sequentially.",
        flush=True,
    )

    progress = Progress(pending)
    all_failures = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                run_case,
                graph,
                solver,
                config,
                results_dir,
                args.problems,
                args.cores,
                args.search,
                args.force,
                progress,
            )
            for graph in graphs
            if any(
                args.force
                or not is_done(results_dir, graph.stem, p, n)
                for p in args.problems
                for n in args.cores
            )
        ]

        for future in as_completed(futures):
            try:
                all_failures.extend(future.result())
            except Exception as exc:
                all_failures.append(f"worker exception: {exc!r}")

    elapsed = time.perf_counter() - progress.started

    print(
        f"Finished pending={pending}, failed={len(all_failures)}, "
        f"elapsed={fmt_seconds(elapsed)}",
        flush=True,
    )

    failure_path = results_dir / "parallel_failures.txt"

    if all_failures:
        failure_path.write_text(
            "\n".join(all_failures) + "\n",
            encoding="utf-8",
        )
        print(f"Failure list: {failure_path}", flush=True)
        return 1

    if failure_path.exists():
        failure_path.unlink()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())