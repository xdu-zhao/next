from __future__ import annotations

import argparse
import heapq
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

from evaluation_validation import read_evaluation_config
from multicore_cut_evaluate_problem_1 import evaluate_scene_a, read_scene_a_config
from multicore_cut_evaluate_problem_2 import evaluate_scene_b, read_scene_b_config
from multicore_cut_evaluate_problem_3 import evaluate_problem_3, read_cache_config
from stub_multicore_cut_and_schedule import (
    _build_op_adjacency,
    _contract_excluded_copy_nodes,
    validate_multicore_plan,
)

COPY_TYPES = {"COPY_IN", "COPY_OUT"}

SOLVER_VERSION = "V2.1-fallback-cache-safe"


def pad_plan_cores(graph, plan, target_cores):
    """
    Reuse a valid k-core plan under a larger advertised core count by appending
    empty core schedules. The mapping/subgraph order is unchanged, so the
    execution is identical while the evaluator sees target_cores cores.
    """
    schedules = [list(order) for order in plan["core_schedules"]]

    if len(schedules) > target_cores:
        raise ValueError(
            f"cannot shrink {len(schedules)}-core plan to {target_cores} cores"
        )

    schedules.extend(
        [] for _ in range(target_cores - len(schedules))
    )

    padded = {
        "node_to_subgraph": dict(plan["node_to_subgraph"]),
        "core_schedules": schedules,
    }

    validate_multicore_plan(graph, padded)
    return padded


def plan_signature(plan):
    """Stable signature used to avoid evaluating duplicate candidates."""
    return json.dumps(
        plan,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def load_reusable_plan(graph, path, target_cores):
    """
    Load a previously saved BEST plan and pad it to target_cores.

    Returns None when the file is absent/invalid. This makes reuse an optional
    optimization rather than a hard dependency.
    """
    path = Path(path)

    if not path.is_file():
        return None

    try:
        plan = load_json(path)

        if (
            not isinstance(plan, dict)
            or set(plan) != {"node_to_subgraph", "core_schedules"}
        ):
            return None

        if len(plan["core_schedules"]) > target_cores:
            return None

        return pad_plan_cores(
            graph,
            plan,
            target_cores,
        )

    except Exception:
        return None


def discover_reuse_candidates(
    graph,
    output_path,
    problem,
    num_cores,
):
    """
    Discover already-computed sibling BEST plans in the same results directory.

    Expected filename:
      case_XXX_p{problem}_n{num_cores}_multicore_res.json

    Reuse policy:
      1. For N>2, reuse the best same-problem 2..N-1 core plans and pad with
         empty cores. This removes avoidable multi-core regressions.
      2. For Problem 3, also reuse Problem 2 BEST plans (same/lower core count)
         as cache-safe candidates. The P3 evaluator decides whether L2 helps.

    The normal batch runners evaluate cores in 2,3,4,5 order and problems in
    1,2,3 order, so these files already exist when they are useful.
    """
    output_path = Path(output_path)
    suffix = (
        f"_p{problem}_n{num_cores}_multicore_res.json"
    )

    if not output_path.name.endswith(suffix):
        return []

    case_prefix = output_path.name[:-len(suffix)]
    parent = output_path.parent
    found = []

    # Same-problem lower-core fallback.
    for used_cores in range(2, num_cores):
        path = parent / (
            f"{case_prefix}_p{problem}_n{used_cores}_multicore_res.json"
        )
        plan = load_reusable_plan(
            graph,
            path,
            num_cores,
        )
        if plan is not None:
            found.append((
                f"reuse-p{problem}-n{used_cores}-best",
                plan,
            ))

    # Problem 3 must never throw away a strong Scene-B (P2) partition.
    if problem == 3:
        for used_cores in range(2, num_cores + 1):
            path = parent / (
                f"{case_prefix}_p2_n{used_cores}_multicore_res.json"
            )
            plan = load_reusable_plan(
                graph,
                path,
                num_cores,
            )
            if plan is not None:
                found.append((
                    f"reuse-p2-n{used_cores}-best",
                    plan,
                ))

    return found


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def build_view(graph):
    op_by_id = {op["id"]: op for op in graph["ops"]}
    all_ops = set(op_by_id)

    eligible = sorted(
        oid for oid, op in op_by_id.items()
        if op.get("op") not in COPY_TYPES
    )
    eligible_set = set(eligible)

    _, full_succs = _build_op_adjacency(graph)
    preds, succs = _contract_excluded_copy_nodes(
        eligible, full_succs
    )

    tensor_by_id = {t["id"]: t for t in graph["tensors"]}
    producers = defaultdict(set)
    consumers = defaultdict(set)

    for e in graph["edges"]:
        a, b = e["source"], e["target"]

        if a in all_ops and b not in all_ops:
            producers[b].add(a)

        elif a not in all_ops and b in all_ops:
            consumers[a].add(b)

    # eligible op -> eligible op 的近似通信量
    comm_bytes = defaultdict(int)

    # 原图输入张量，用于判断同核复用
    input_tensors = defaultdict(list)

    for tid, tensor in tensor_by_id.items():
        size = int(tensor["size"])

        ps = [
            p for p in producers.get(tid, ())
            if p in eligible_set
        ]

        cs = [
            c for c in consumers.get(tid, ())
            if c in eligible_set
        ]

        if ps and cs:
            for p in ps:
                for c in cs:
                    if p != c:
                        comm_bytes[(p, c)] += size

        elif cs and not ps:
            for c in cs:
                input_tensors[c].append((tid, size))

    return {
        "op_by_id": op_by_id,
        "eligible": eligible,
        "eligible_set": eligible_set,
        "preds": preds,
        "succs": succs,
        "tensor_by_id": tensor_by_id,
        "producers": producers,
        "consumers": consumers,
        "comm_bytes": comm_bytes,
        "input_tensors": input_tensors,
    }


def topo_order(view):
    indeg = {
        u: len(view["preds"][u])
        for u in view["eligible"]
    }

    ready = [
        u for u in view["eligible"]
        if indeg[u] == 0
    ]

    heapq.heapify(ready)

    order = []

    while ready:
        u = heapq.heappop(ready)
        order.append(u)

        for v in view["succs"][u]:
            indeg[v] -= 1

            if indeg[v] == 0:
                heapq.heappush(ready, v)

    if len(order) != len(view["eligible"]):
        raise RuntimeError(
            "contracted graph contains a cycle"
        )

    return order


def weak_components(view):
    seen = set()
    components = []

    for u in view["eligible"]:

        if u in seen:
            continue

        seen.add(u)
        stack = [u]
        comp = []

        while stack:

            x = stack.pop()
            comp.append(x)

            for y in (
                view["preds"][x]
                | view["succs"][x]
            ):

                if y not in seen:
                    seen.add(y)
                    stack.append(y)

        components.append(comp)

    return components


def op_work(op):
    return max(
        1,
        int(op.get("cycles", 0))
    )


def split_component(
    comp,
    topo_pos,
    op_by_id,
    target_work,
):
    """
    大弱连通分量只沿拓扑序切割，
    保证 quotient graph 仍然容易保持 DAG。
    """

    seq = sorted(
        comp,
        key=topo_pos.__getitem__,
    )

    total = sum(
        op_work(op_by_id[u])
        for u in seq
    )

    if (
        total <= 1.25 * target_work
        or len(seq) <= 4
    ):
        return [seq]

    parts = []

    cur = []
    cur_work = 0.0

    for u in seq:

        w = op_work(op_by_id[u])

        if (
            cur
            and cur_work + w > target_work
            and cur_work >= 0.55 * target_work
        ):
            parts.append(cur)
            cur = []
            cur_work = 0.0

        cur.append(u)
        cur_work += w

    if cur:

        tail_work = sum(
            op_work(op_by_id[u])
            for u in cur
        )

        if (
            parts
            and tail_work < 0.30 * target_work
        ):
            parts[-1].extend(cur)

        else:
            parts.append(cur)

    return parts


def build_chunks(
    graph,
    num_cores,
    waves,
):
    """
    Problem 1 的核心切图。

    小弱连通分量尽量整体保留；
    只有太大的弱连通分量才切分。

    waves:
        1.0 -> 约 num_cores 个子图
        2.0 -> 约 2*num_cores 个子图
    """

    view = build_view(graph)

    topo = topo_order(view)

    pos = {
        u: i
        for i, u in enumerate(topo)
    }

    op_by_id = view["op_by_id"]

    total_work = sum(
        op_work(op_by_id[u])
        for u in view["eligible"]
    )

    desired = max(
        1,
        int(round(num_cores * waves))
    )

    target = max(
        1.0,
        total_work / desired,
    )

    big_parts = []
    small_components = []

    for comp in weak_components(view):

        cw = sum(
            op_work(op_by_id[u])
            for u in comp
        )

        if cw > 1.25 * target:

            big_parts.extend(
                split_component(
                    comp,
                    pos,
                    op_by_id,
                    target,
                )
            )

        else:

            small_components.append(
                (cw, comp)
            )

    # 已被拆开的“大分量”
    chunks = [
        list(part)
        for part in big_parts
    ]

    chunk_work = [
        sum(
            op_work(op_by_id[u])
            for u in part
        )
        for part in chunks
    ]

    # 补足目标 chunk 数
    while len(chunks) < desired:
        chunks.append([])
        chunk_work.append(0.0)

    # 小弱连通分量彼此独立，可安全 LPT 装箱
    for cw, comp in sorted(
        small_components,
        key=lambda x: x[0],
        reverse=True,
    ):

        j = min(
            range(len(chunks)),
            key=lambda i: chunk_work[i],
        )

        chunks[j].extend(comp)
        chunk_work[j] += cw

    chunks = [
        sorted(c, key=pos.__getitem__)
        for c in chunks
        if c
    ]

    chunks.sort(
        key=lambda c: min(
            pos[u] for u in c
        )
    )

    node_to_sg = {}

    for sg, nodes in enumerate(chunks):

        for u in nodes:
            node_to_sg[u] = sg

    m = len(chunks)

    sg_preds = [
        set()
        for _ in range(m)
    ]

    sg_succs = [
        set()
        for _ in range(m)
    ]

    # 子图 DAG
    for u in view["eligible"]:

        a = node_to_sg[u]

        for v in view["succs"][u]:

            b = node_to_sg[v]

            if a != b:
                sg_succs[a].add(b)
                sg_preds[b].add(a)

    # 估算各 Task 的边界 COPY 量
    copy_in_bytes = [0] * m
    copy_out_bytes = [0] * m

    eligible_set = view["eligible_set"]

    for tid, tensor in view["tensor_by_id"].items():

        size = int(tensor["size"])

        src_sgs = {
            node_to_sg[p]
            for p in view["producers"].get(
                tid, ()
            )
            if p in eligible_set
        }

        dst_sgs = {
            node_to_sg[c]
            for c in view["consumers"].get(
                tid, ()
            )
            if c in eligible_set
        }

        # 原图输入
        if not src_sgs and dst_sgs:

            for b in dst_sgs:
                copy_in_bytes[b] += size

        # 原图输出
        elif src_sgs and not dst_sgs:

            for a in src_sgs:
                copy_out_bytes[a] += size

        # 子图之间 tensor
        else:

            for a in src_sgs:

                if any(
                    b != a
                    for b in dst_sgs
                ):
                    copy_out_bytes[a] += size

            for b in dst_sgs:

                if any(
                    a != b
                    for a in src_sgs
                ):
                    copy_in_bytes[b] += size

    costs = []

    for sg, nodes in enumerate(chunks):

        pipe_work = Counter()

        for u in nodes:

            op = op_by_id[u]

            if op["pipe"] in (
                "PIPE_M",
                "PIPE_V",
            ):

                pipe_work[
                    op["pipe"]
                ] += op_work(op)

        # MTE2、MTE3 是不同 Pipe
        mte2 = (
            copy_in_bytes[sg] / 60.0
        )

        mte3 = (
            copy_out_bytes[sg] / 60.0
        )

        costs.append(
            max(
                1.0,
                float(
                    pipe_work["PIPE_M"]
                ),
                float(
                    pipe_work["PIPE_V"]
                ),
                mte2,
                mte3,
            )
        )

    return (
        view,
        topo,
        node_to_sg,
        sg_preds,
        sg_succs,
        costs,
    )


def upward_rank(
    preds,
    succs,
    costs,
):
    """
    类 HEFT upward rank，
    这里只用于 Problem 1 Task 选择。
    """

    remain = [
        len(succs[i])
        for i in range(len(costs))
    ]

    q = deque(
        i
        for i, x in enumerate(remain)
        if x == 0
    )

    rank = [
        0.0
        for _ in costs
    ]

    while q:

        u = q.popleft()

        rank[u] = (
            costs[u]
            + max(
                (
                    rank[v]
                    for v in succs[u]
                ),
                default=0.0,
            )
        )

        for p in preds[u]:

            remain[p] -= 1

            if remain[p] == 0:
                q.append(p)

    return rank


def plan_scene_a(
    graph,
    num_cores,
    waves=1.0,
):
    """
    Problem 1:
    通信友好的 Task 划分 +
    100/1000 cycles 等待感知分核。
    """

    (
        view,
        topo,
        node_to_sg,
        sg_preds,
        sg_succs,
        costs,
    ) = build_chunks(
        graph,
        num_cores,
        waves,
    )

    m = len(costs)

    rank = upward_rank(
        sg_preds,
        sg_succs,
        costs,
    )

    indeg = [
        len(sg_preds[i])
        for i in range(m)
    ]

    ready = [
        (-rank[i], i)
        for i in range(m)
        if indeg[i] == 0
    ]

    heapq.heapify(ready)

    sg_core = [None] * m
    finish = [0.0] * m

    core_free = [
        0.0
        for _ in range(num_cores)
    ]

    core_orders = [
        []
        for _ in range(num_cores)
    ]

    while ready:

        _, sg = heapq.heappop(ready)

        best = None

        for core in range(num_cores):

            dep_ready = 0.0

            for p in sg_preds[sg]:

                if sg_core[p] == core:
                    wait = 100.0
                else:
                    wait = 1000.0

                dep_ready = max(
                    dep_ready,
                    finish[p] + wait,
                )

            own_ready = core_free[core]

            if core_orders[core]:
                own_ready += 100.0

            start = max(
                dep_ready,
                own_ready,
            )

            end = (
                start
                + costs[sg]
            )

            if (
                best is None
                or end < best[0]
            ):
                best = (
                    end,
                    core,
                )

        end, core = best

        sg_core[sg] = core
        finish[sg] = end
        core_free[core] = end

        core_orders[
            core
        ].append(sg)

        for v in sg_succs[sg]:

            indeg[v] -= 1

            if indeg[v] == 0:

                heapq.heappush(
                    ready,
                    (-rank[v], v),
                )

    plan = {
        "node_to_subgraph": {
            str(u): int(
                node_to_sg[u]
            )
            for u in view["eligible"]
        },

        "core_schedules":
            core_orders,
    }

    validate_multicore_plan(
        graph,
        plan,
    )

    return plan


def assignment_to_plan(
    graph,
    view,
    topo,
    assignment,
    num_cores,
):
    """
    任意 op->core 分配
    转换为“全局拓扑连续段”。

    每个连续段作为一个 sgid，
    可以避免随意聚类产生 quotient cycle。
    """

    node_to_sg = {}

    core_schedules = [
        []
        for _ in range(num_cores)
    ]

    sg = -1
    last_core = None

    for u in topo:

        core = int(
            assignment[u]
        )

        if core != last_core:

            sg += 1

            core_schedules[
                core
            ].append(sg)

            last_core = core

        node_to_sg[u] = sg

    plan = {
        "node_to_subgraph": {
            str(u): int(
                node_to_sg[u]
            )
            for u in view["eligible"]
        },

        "core_schedules":
            core_schedules,
    }

    validate_multicore_plan(
        graph,
        plan,
    )

    return plan


def plan_contiguous(
    graph,
    num_cores,
):
    """
    Problem 2/3 候选 A：

    把全局拓扑序切成 num_cores 个
    计算量近似相同的连续块。

    优势：跨核边通常较少。
    """

    view = build_view(graph)

    topo = topo_order(view)

    op_by_id = view[
        "op_by_id"
    ]

    total = sum(
        op_work(op_by_id[u])
        for u in topo
    )

    assignment = {}

    core = 0
    used = 0.0

    for u in topo:

        while (
            core < num_cores - 1
            and used
            >= total
            * (core + 1)
            / num_cores
        ):

            core += 1

        assignment[u] = core

        used += op_work(
            op_by_id[u]
        )

    return assignment_to_plan(
        graph,
        view,
        topo,
        assignment,
        num_cores,
    )


def plan_affinity(
    graph,
    num_cores,
    beta=0.15,
    comm_scale=1.0,
    input_scale=1.0,
):
    """
    Problem 2/3 候选 B：

    同时考虑：
    1. PIPE_M 负载
    2. PIPE_V 负载
    3. 前驱跨核代价
    4. 共享输入重复读取
    """

    view = build_view(graph)

    topo = topo_order(view)

    op_by_id = view[
        "op_by_id"
    ]

    total = sum(
        op_work(op_by_id[u])
        for u in view["eligible"]
    )

    target_total = max(
        1.0,
        total / num_cores,
    )

    pipe_total = Counter()

    for u in view["eligible"]:

        pipe_total[
            op_by_id[u]["pipe"]
        ] += op_work(
            op_by_id[u]
        )

    loads = [
        Counter()
        for _ in range(num_cores)
    ]

    seen_inputs = [
        set()
        for _ in range(num_cores)
    ]

    assignment = {}

    for u in topo:

        op = op_by_id[u]

        pipe = op["pipe"]
        duration = op_work(op)

        best = None

        for core in range(num_cores):

            # 跨核内部 tensor
            cross_cost = 0.0

            for p in view["preds"][u]:

                if assignment[p] != core:

                    bytes_ = (
                        view["comm_bytes"]
                        .get((p, u), 0)
                    )

                    cross_cost += (
                        500.0
                        + 2.0
                        * bytes_
                        / 60.0
                    )

            # 原始共享输入：
            # 同核第一次读才计惩罚
            first_read = sum(
                size / 60.0
                for tid, size
                in view[
                    "input_tensors"
                ].get(u, ())
                if tid
                not in seen_inputs[
                    core
                ]
            )

            projected_pipe = (
                loads[core][pipe]
                + duration
            )

            pipe_target = max(
                1.0,
                pipe_total[pipe]
                / num_cores,
            )

            norm_pipe = (
                projected_pipe
                / pipe_target
            )

            projected_total = (
                sum(
                    loads[
                        core
                    ].values()
                )
                + duration
            )

            norm_total = (
                projected_total
                / target_total
            )

            load_penalty = (
                beta
                * target_total
                * (
                    0.65 * norm_pipe
                    + 0.35 * norm_total
                )
            )

            score = (
                comm_scale
                * cross_cost
                + input_scale
                * first_read
                + load_penalty
            )

            if (
                best is None
                or score < best[0]
            ):

                best = (
                    score,
                    core,
                )

        core = best[1]

        assignment[u] = core

        loads[
            core
        ][pipe] += duration

        for tid, _ in (
            view[
                "input_tensors"
            ].get(u, ())
        ):

            seen_inputs[
                core
            ].add(tid)

    return assignment_to_plan(
        graph,
        view,
        topo,
        assignment,
        num_cores,
    )


def official_eval(
    graph,
    plan,
    problem,
    config_path,
):
    """
    直接调用官方 evaluator 函数。

    这样搜索候选时不产生大量 Trace，
    速度比每次 subprocess 跑脚本快很多。
    """

    settings = (
        read_evaluation_config(
            str(config_path)
        )
    )

    bandwidth = settings[
        "bandwidth"
    ]

    capacity = settings[
        "capacity"
    ]

    if problem == 1:

        scene = (
            read_scene_a_config(
                str(config_path)
            )
        )

        return evaluate_scene_a(
            graph,
            plan,
            bandwidth=bandwidth,
            capacity=capacity,
            cross_core_wait=scene[
                "task_cross_core_wait_cycles"
            ],
            same_core_wait=scene[
                "task_same_core_wait_cycles"
            ],
        )

    scene = (
        read_scene_b_config(
            str(config_path)
        )
    )

    if problem == 2:

        return evaluate_scene_b(
            graph,
            plan,
            bandwidth=bandwidth,
            capacity=capacity,
            cross_core_copy_delay=scene[
                "cross_core_copy_delay_cycles"
            ],
        )

    cache = (
        read_cache_config(
            str(config_path)
        )
    )

    return evaluate_problem_3(
        graph,
        plan,
        bandwidth=bandwidth,
        capacity=capacity,
        cross_core_copy_delay=scene[
            "cross_core_copy_delay_cycles"
        ],
        **cache,
    )


def candidate_plans(
    graph,
    problem,
    num_cores,
    mode,
):
    """
    Generate candidates.

    V2 changes:
    - P1 keeps the proven waves search.
    - P2 keeps the proven Scene-B candidates.
    - P3 is a strict superset of the old P3 heuristic family: it evaluates
      both cache-aware affinity candidates and P2-style affinity candidates.
      This avoids losing a strong no-L2 partition merely because cache-aware
      weights changed the partition.
    """

    if problem == 1:

        if mode == "none":
            wave_values = [1.0]

        elif mode == "quick":
            wave_values = [
                1.0,
                2.0,
            ]

        else:
            wave_values = [
                0.75,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
            ]

        for w in wave_values:

            yield (
                f"sceneA-waves={w:g}",
                plan_scene_a(
                    graph,
                    num_cores,
                    waves=w,
                ),
            )

        return

    if mode == "none":
        if problem == 2:
            yield (
                "affinity",
                plan_affinity(
                    graph,
                    num_cores,
                    beta=0.15,
                    comm_scale=1.0,
                    input_scale=1.0,
                ),
            )
        else:
            yield (
                "cache-affinity",
                plan_affinity(
                    graph,
                    num_cores,
                    beta=0.15,
                    comm_scale=0.80,
                    input_scale=0.20,
                ),
            )
        return

    if mode == "quick":
        beta_values = [
            0.12,
            0.25,
        ]
    else:
        beta_values = [
            0.06,
            0.12,
            0.20,
            0.35,
            0.60,
        ]

    # Common low-cross-core-edge candidates.
    yield (
        "contiguous",
        plan_contiguous(
            graph,
            num_cores,
        ),
    )

    yield (
        "coarse-w1",
        plan_scene_a(
            graph,
            num_cores,
            waves=1.0,
        ),
    )

    if mode == "full":
        yield (
            "coarse-w2",
            plan_scene_a(
                graph,
                num_cores,
                waves=2.0,
            ),
        )

    if problem == 2:
        for beta in beta_values:
            yield (
                f"affinity-b={beta:g}",
                plan_affinity(
                    graph,
                    num_cores,
                    beta=beta,
                    comm_scale=1.0,
                    input_scale=1.0,
                ),
            )
        return

    # Problem 3:
    # First keep every P2-style affinity candidate. This makes the candidate
    # family cache-safe: a good Scene-B partition is never discarded merely
    # because cache-aware weights prefer another partition.
    for beta in beta_values:
        yield (
            f"p2-affinity-b={beta:g}",
            plan_affinity(
                graph,
                num_cores,
                beta=beta,
                comm_scale=1.0,
                input_scale=1.0,
            ),
        )

    # Then add cache-aware variants. L2 can reward more sharing, so communication
    # and repeated-input penalties are deliberately softened.
    for beta in beta_values:
        yield (
            f"cache-affinity-b={beta:g}",
            plan_affinity(
                graph,
                num_cores,
                beta=beta,
                comm_scale=0.80,
                input_scale=0.20,
            ),
        )



def result_key(result):
    """
    第一目标：Makespan
    第二目标：额外 DDR 搬运量
    """

    movement = result.get(
        "data_movement_bytes",
        {},
    )

    return (
        int(
            result["makespan"]
        ),
        int(
            movement.get(
                "added_copy_bytes",
                0,
            )
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "communication-aware "
            "multicore scheduler"
        )
    )

    parser.add_argument(
        "graph"
    )

    parser.add_argument(
        "-p",
        "--problem",
        type=int,
        choices=(1, 2, 3),
        required=True,
    )

    parser.add_argument(
        "-n",
        "--num-cores",
        type=int,
        choices=(1, 2, 3, 4, 5),
        required=True,
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
    )

    parser.add_argument(
        "--search",
        choices=(
            "none",
            "quick",
            "full",
        ),
        default="quick",
    )

    parser.add_argument(
        "--config"
    )

    parser.add_argument(
        "--result-output"
    )

    parser.add_argument(
        "--no-reuse-sibling-best",
        action="store_true",
        help=(
            "Do not reuse already-computed lower-core/P2 BEST plans "
            "from the output directory."
        ),
    )

    args = parser.parse_args()

    print(
        f"solve_multicore {SOLVER_VERSION}; "
        f"problem={args.problem}; cores={args.num_cores}; "
        f"search={args.search}",
        flush=True,
    )

    graph_path = Path(
        args.graph
    )

    if args.config:

        config_path = Path(
            args.config
        )

    else:

        config_path = (
            graph_path.parent
            / "config.txt"
        )

    graph = load_json(
        graph_path
    )

    # 不搜索：直接生成一个方案
    if args.search == "none":

        name, plan = next(
            candidate_plans(
                graph,
                args.problem,
                args.num_cores,
                "none",
            )
        )

        validate_multicore_plan(
            graph,
            plan,
        )

        dump_json(
            args.output,
            plan,
        )

        print(
            f"OK: {name}; "
            f"subgraphs="
            f"{len(set(plan['node_to_subgraph'].values()))}; "
            f"plan={args.output}"
        )

        if args.result_output:

            result = official_eval(
                graph,
                plan,
                args.problem,
                config_path,
            )

            dump_json(
                args.result_output,
                result,
            )

            print(
                f"makespan="
                f"{result['makespan']}; "
                f"added_copy_bytes="
                f"{result.get('data_movement_bytes', {}).get('added_copy_bytes', 0)}"
            )

        return 0

    best = None

    candidates = list(
        candidate_plans(
            graph,
            args.problem,
            args.num_cores,
            args.search,
        )
    )

    if not args.no_reuse_sibling_best:
        reuse = discover_reuse_candidates(
            graph,
            args.output,
            args.problem,
            args.num_cores,
        )
        if reuse:
            print(
                "reuse candidates: "
                + ", ".join(name for name, _ in reuse),
                flush=True,
            )
            candidates.extend(reuse)

    # Deduplicate identical plans. This matters because coarse/contiguous or
    # reused plans can collapse to the same partition on small graphs.
    unique_candidates = []
    seen_signatures = set()

    for name, plan in candidates:
        signature = plan_signature(plan)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        unique_candidates.append((name, plan))

    print(
        f"candidate_count={len(unique_candidates)}",
        flush=True,
    )

    for index, (
        name,
        plan,
    ) in enumerate(
        unique_candidates,
        1,
    ):

        try:

            result = official_eval(
                graph,
                plan,
                args.problem,
                config_path,
            )

        except Exception as error:

            print(
                f"[{index:02d}] "
                f"{name}: FAIL: "
                f"{error}"
            )

            continue

        movement = result.get(
            "data_movement_bytes",
            {},
        )

        extra = ""

        if "cache_stats" in result:

            cache = result[
                "cache_stats"
            ]

            extra = (
                f"; cache_hits="
                f"{cache.get('hits', 0)}/"
                f"{cache.get('accesses', 0)}"
            )

        print(
            f"[{index:02d}] "
            f"{name}: "
            f"makespan="
            f"{result['makespan']}; "
            f"added_copy_bytes="
            f"{movement.get('added_copy_bytes', 0)}"
            f"{extra}"
        )

        item = (
            result_key(result),
            name,
            plan,
            result,
        )

        if (
            best is None
            or item[0] < best[0]
        ):

            best = item

    if best is None:

        raise RuntimeError(
            "all candidates failed"
        )

    _, name, plan, result = best

    dump_json(
        args.output,
        plan,
    )

    if args.result_output:

        dump_json(
            args.result_output,
            result,
        )

    print(
        f"BEST: {name}; "
        f"makespan="
        f"{result['makespan']}; "
        f"added_copy_bytes="
        f"{result.get('data_movement_bytes', {}).get('added_copy_bytes', 0)}; "
        f"plan={args.output}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())