"""Small, evaluator-backed neighbourhood search for V7 five-core plans.

The portfolio search chooses a whole-graph policy.  This script complements it
with a deliberately bounded refinement: it moves a contiguous window of the
current topological assignment to another core, and accepts only a strict
official-evaluator improvement.  Windows preserve the V7 plan invariant that
subgraphs are global-topology-contiguous, so every candidate remains valid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from solve_multicore_v7 import (
    assignment_to_plan,
    build_view,
    dump_json,
    load_json,
    official_eval,
    op_work,
    result_key,
    topo_order,
)


def plan_assignment(plan: dict) -> dict[int, int]:
    owner = {
        int(sgid): core
        for core, schedule in enumerate(plan["core_schedules"])
        for sgid in schedule
    }
    return {
        int(node): owner[int(sgid)]
        for node, sgid in plan["node_to_subgraph"].items()
    }


def make_windows(order, assignment, view, per_run: int):
    """Split each same-core run into a few work-balanced movable windows."""
    runs, current = [], []
    last = None
    for op in order:
        core = assignment[op]
        if current and core != last:
            runs.append(current)
            current = []
        current.append(op)
        last = core
    if current:
        runs.append(current)

    windows = []
    for run in runs:
        total = sum(op_work(view["op_by_id"][op]) for op in run)
        target = max(1, total / per_run)
        current, used = [], 0
        for op in run:
            work = op_work(view["op_by_id"][op])
            if current and used + work > target and len(windows) < 100000:
                windows.append(current)
                current, used = [], 0
            current.append(op)
            used += work
        if current:
            windows.append(current)
    return windows


def refine_case(root: Path, case_id: int, problem: int, rounds: int, max_windows: int):
    code = root / "code"
    data = root / "data" / f"case_{case_id:03d}.json"
    results = root / "results_v8_n5"
    stem = f"case_{case_id:03d}_p{problem}_n5"
    plan_path = results / f"{stem}_multicore_res.json"
    score_path = results / f"{stem}_score.json"
    graph, incumbent = load_json(data), load_json(plan_path)
    incumbent_score = load_json(score_path)
    best_key = result_key(incumbent_score)
    view, order = build_view(graph), topo_order(build_view(graph))
    assignment = plan_assignment(incumbent)
    config = root / "data" / "config.txt"
    tried = 0

    for _ in range(rounds):
        windows = make_windows(order, assignment, view, per_run=6)
        # Prioritise substantial moves, but keep each representative run cheap.
        windows.sort(
            key=lambda block: sum(op_work(view["op_by_id"][op]) for op in block),
            reverse=True,
        )
        best_trial = None
        for block in windows[:max_windows]:
            source = assignment[block[0]]
            for target in range(5):
                if target == source:
                    continue
                candidate_assignment = assignment.copy()
                candidate_assignment.update({op: target for op in block})
                candidate = assignment_to_plan(
                    graph, view, order, candidate_assignment, 5
                )
                score = official_eval(graph, candidate, problem, config)
                tried += 1
                key = result_key(score)
                if key < best_key and (
                    best_trial is None or key < best_trial[0]
                ):
                    best_trial = (key, candidate, score, candidate_assignment)
        if best_trial is None:
            break
        best_key, incumbent, incumbent_score, assignment = best_trial

    improved = best_key < result_key(load_json(score_path))
    print(
        f"case={case_id:03d} p={problem} tried={tried} "
        f"best={best_key} improved={improved}", flush=True
    )
    if improved:
        trial = root / "results_v11_block_refine"
        dump_json(trial / f"{stem}_multicore_res.json", incumbent)
        dump_json(trial / f"{stem}_score.json", incumbent_score)
    return improved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-ids", nargs="+", type=int, required=True)
    parser.add_argument("--problems", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-windows", type=int, default=12)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    improved = []
    for problem in args.problems:
        for case_id in args.case_ids:
            if refine_case(root, case_id, problem, args.rounds, args.max_windows):
                improved.append({"case": case_id, "problem": problem})
    (root / "results_v11_block_refine" / "representative_summary.json").parent.mkdir(
        parents=True, exist_ok=True
    )
    (root / "results_v11_block_refine" / "representative_summary.json").write_text(
        json.dumps({"improved": improved}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
