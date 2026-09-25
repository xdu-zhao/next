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



def op_upward_rank(view, comm_scale=1.0):
    """
    Op-level HEFT-like upward rank.

    Rank includes local op work plus an approximate communication term.
    It is only used to choose which ready op is assigned first; the final
    winner is still selected by the official evaluator.
    """
    op_by_id = view["op_by_id"]
    topo = topo_order(view)
    rank = {}

    for u in reversed(topo):
        best_succ = 0.0
        for v in view["succs"][u]:
            comm = (
                500.0
                + 2.0 * view["comm_bytes"].get((u, v), 0) / 60.0
            )
            best_succ = max(
                best_succ,
                comm_scale * comm + rank[v],
            )

        rank[u] = op_work(op_by_id[u]) + best_succ

    return rank


def plan_affinity_ranked(
    graph,
    num_cores,
    beta=0.15,
    comm_scale=1.0,
    input_scale=1.0,
    rank_weight=1.0,
):
    """
    V3 candidate:
    affinity placement + critical-path-first ready queue.

    Compared with plan_affinity(), placement cost is intentionally kept
    compatible; the main change is that ready ops on the critical path are
    assigned first, reducing the chance that non-critical work consumes the
    best core placement too early.
    """
    view = build_view(graph)
    op_by_id = view["op_by_id"]
    rank = op_upward_rank(view, comm_scale=comm_scale)

    total = sum(op_work(op_by_id[u]) for u in view["eligible"])
    target_total = max(1.0, total / num_cores)

    pipe_total = Counter()
    for u in view["eligible"]:
        pipe_total[op_by_id[u]["pipe"]] += op_work(op_by_id[u])

    loads = [Counter() for _ in range(num_cores)]
    seen_inputs = [set() for _ in range(num_cores)]
    assignment = {}
    order = []

    indeg = {u: len(view["preds"][u]) for u in view["eligible"]}
    ready = [
        (-rank_weight * rank[u], u)
        for u in view["eligible"]
        if indeg[u] == 0
    ]
    heapq.heapify(ready)

    while ready:
        _, u = heapq.heappop(ready)
        op = op_by_id[u]
        pipe = op["pipe"]
        duration = op_work(op)

        best = None
        for core in range(num_cores):
            cross_cost = 0.0
            for p in view["preds"][u]:
                if assignment[p] != core:
                    bytes_ = view["comm_bytes"].get((p, u), 0)
                    cross_cost += 500.0 + 2.0 * bytes_ / 60.0

            first_read = sum(
                size / 60.0
                for tid, size in view["input_tensors"].get(u, ())
                if tid not in seen_inputs[core]
            )

            projected_pipe = loads[core][pipe] + duration
            pipe_target = max(1.0, pipe_total[pipe] / num_cores)
            norm_pipe = projected_pipe / pipe_target

            projected_total = sum(loads[core].values()) + duration
            norm_total = projected_total / target_total

            load_penalty = (
                beta
                * target_total
                * (0.65 * norm_pipe + 0.35 * norm_total)
            )

            score = (
                comm_scale * cross_cost
                + input_scale * first_read
                + load_penalty
            )

            if best is None or score < best[0]:
                best = (score, core)

        core = best[1]
        assignment[u] = core
        order.append(u)
        loads[core][pipe] += duration

        for tid, _ in view["input_tensors"].get(u, ()):
            seen_inputs[core].add(tid)

        for v in view["succs"][u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(
                    ready,
                    (-rank_weight * rank[v], v),
                )

    return assignment_to_plan(
        graph,
        view,
        order,
        assignment,
        num_cores,
    )



def dag_shape_metrics(graph):
    view = build_view(graph)
    op_by_id = view["op_by_id"]
    eligible = view["eligible"]
    if not eligible:
        return {"cp_ratio": 1.0, "avg_width": 1.0, "max_width": 1}
    topo = topo_order(view)
    cp = {}
    for u in reversed(topo):
        cp[u] = op_work(op_by_id[u]) + max(
            (cp[v] for v in view["succs"][u]), default=0.0
        )
    total = sum(op_work(op_by_id[u]) for u in eligible)
    indeg = {u: len(view["preds"][u]) for u in eligible}
    ready = [u for u in eligible if indeg[u] == 0]
    widths = []
    while ready:
        layer = list(ready)
        widths.append(len(layer))
        ready = []
        for u in layer:
            for v in view["succs"][u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    ready.append(v)
    return {
        "cp_ratio": max(cp.values(), default=total) / max(1.0, total),
        "avg_width": sum(widths) / max(1, len(widths)),
        "max_width": max(widths, default=1),
    }


def adaptive_scene_a_wave_values(graph, num_cores):
    m = dag_shape_metrics(graph)
    if m["cp_ratio"] >= 0.72 or m["max_width"] <= 2:
        vals = [0.22, 0.30, 0.45, 0.65]
    elif m["avg_width"] >= max(3.0, 0.8*num_cores) or m["max_width"] >= 2*num_cores:
        vals = [0.28, 0.40, 0.60, 0.90, 1.15]
    else:
        vals = [0.30, 0.45, 0.65, 0.90]
    return vals, m


def _tensor_reuse_values(view):
    consumers, sizes = Counter(), {}
    for tensors in view["input_tensors"].values():
        for tid, size in tensors:
            consumers[tid] += 1
            sizes[tid] = max(sizes.get(tid, 0), size)
    return {tid: (cnt-1)*sizes.get(tid,0)/60.0
            for tid,cnt in consumers.items() if cnt > 1}


def plan_cluster_rank_affinity(
    graph, num_cores, beta=0.12, comm_scale=1.0,
    input_scale=1.0, cluster_scale=0.35, cache_value_scale=0.0
):
    view=build_view(graph); op_by_id=view["op_by_id"]
    rank=op_upward_rank(view, comm_scale=comm_scale)
    reuse_value=_tensor_reuse_values(view)
    total=sum(op_work(op_by_id[u]) for u in view["eligible"])
    target=max(1.0,total/num_cores)
    pipe_total=Counter()
    for u in view["eligible"]:
        pipe_total[op_by_id[u]["pipe"]]+=op_work(op_by_id[u])
    loads=[Counter() for _ in range(num_cores)]
    seen=[set() for _ in range(num_cores)]
    assignment={}; order=[]
    indeg={u:len(view["preds"][u]) for u in view["eligible"]}
    ready=[(-rank[u],u) for u in view["eligible"] if indeg[u]==0]
    heapq.heapify(ready)
    while ready:
        _,u=heapq.heappop(ready); op=op_by_id[u]
        pipe=op["pipe"]; dur=op_work(op); best=None
        for core in range(num_cores):
            cross=cluster=first=cache=0.0
            for p in view["preds"][u]:
                nb=view["comm_bytes"].get((p,u),0)
                ec=500.0+2.0*nb/60.0
                if assignment[p]!=core: cross+=ec
                else: cluster+=ec
            for tid,size in view["input_tensors"].get(u,()):
                if tid not in seen[core]: first+=size/60.0
                else: cache+=reuse_value.get(tid,0.0)
            pt=max(1.0,pipe_total[pipe]/num_cores)
            np=(loads[core][pipe]+dur)/pt
            nt=(sum(loads[core].values())+dur)/target
            lp=beta*target*(0.68*np+0.32*nt)
            score=(comm_scale*cross+input_scale*first+lp
                   -cluster_scale*cluster-cache_value_scale*cache)
            if best is None or score<best[0]: best=(score,core)
        core=best[1]; assignment[u]=core; order.append(u)
        loads[core][pipe]+=dur
        for tid,_ in view["input_tensors"].get(u,()): seen[core].add(tid)
        for v in view["succs"][u]:
            indeg[v]-=1
            if indeg[v]==0: heapq.heappush(ready,(-rank[v],v))
    return assignment_to_plan(graph,view,order,assignment,num_cores)


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
    产生多个候选。

    最后不相信估计成本，
    而是让官方 evaluator 真跑一次，
    选真实 makespan 最好的。
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
            # V3: keep every V2 candidate and add finer partitions.
            # V5: preserve V4/V3 candidates and densify the proven
            # Scene-A winner neighborhood around waves ~= 0.60.
            wave_values = [
                0.35,
                0.45,
                0.50,
                0.55,
                0.60,
                0.65,
                0.70,
                0.75,
                1.0,
                1.25,
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

        if mode == "full":
            adaptive_waves, shape = adaptive_scene_a_wave_values(graph, num_cores)
            for w in adaptive_waves:
                yield (
                    f"adaptive-sceneA-w={w:g}-cp={shape['cp_ratio']:.2f}-mw={shape['max_width']}",
                    plan_scene_a(graph, num_cores, waves=w),
                )

        return

    # Problem 3 有共享 L2，
    # 因此可以降低重复读/通信惩罚
    if problem == 2:

        comm_scale = 1.0
        input_scale = 1.0

    else:

        comm_scale = 0.80
        input_scale = 0.20

    if mode == "none":

        yield (
            "affinity",
            plan_affinity(
                graph,
                num_cores,
                beta=0.15,
                comm_scale=comm_scale,
                input_scale=input_scale,
            ),
        )

        return

    # 低跨核边候选
    yield (
        "contiguous",
        plan_contiguous(
            graph,
            num_cores,
        ),
    )

    # component-aware 粗粒度候选
    yield (
        "coarse-w1",
        plan_scene_a(
            graph,
            num_cores,
            waves=1.0,
        ),
    )

    if mode == "full":

        # V5: coarse-w1 was one of the most frequent winners in V4.
        # Search its local neighborhood instead of jumping directly 1 -> 2.
        for coarse_w in [
            0.65,
            0.80,
            0.90,
            1.10,
            1.25,
            1.50,
        ]:
            yield (
                f"coarse-w{coarse_w:g}",
                plan_scene_a(
                    graph,
                    num_cores,
                    waves=coarse_w,
                ),
            )

        # Preserve the original V4/V3 coarse-w2 candidate.
        yield (
            "coarse-w2",
            plan_scene_a(
                graph,
                num_cores,
                waves=2.0,
            ),
        )

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

    for beta in beta_values:

        yield (
            f"affinity-b={beta:g}",
            plan_affinity(
                graph,
                num_cores,
                beta=beta,
                comm_scale=comm_scale,
                input_scale=input_scale,
            ),
        )

    # V3: critical-path-first affinity candidates.
    # Existing V2 candidates above are preserved, so the official evaluator
    # can always fall back to the old winner if these do not help.
    if mode == "quick":
        ranked_betas = [0.12]
    else:
        ranked_betas = [0.06, 0.12, 0.20, 0.35]

    for beta in ranked_betas:
        yield (
            f"rank-affinity-b={beta:g}",
            plan_affinity_ranked(
                graph,
                num_cores,
                beta=beta,
                comm_scale=comm_scale,
                input_scale=input_scale,
            ),
        )



    # V4: communication clustering + critical-path scheduling.
    cluster_params = (
        [(0.12, 0.35)]
        if mode == "quick"
        else [
            # V4 points are retained; V5 fills their neighborhoods.
            (0.06, 0.15),
            (0.08, 0.20),
            (0.10, 0.28),
            (0.12, 0.35),
            (0.15, 0.42),
            (0.18, 0.50),
            (0.22, 0.62),
        ]
    )
    for beta, cscale in cluster_params:
        cv = 0.0 if problem == 2 else 0.20
        yield (
            f"cluster-rank-b={beta:g}-c={cscale:g}"
            + ("" if problem == 2 else f"-cv={cv:g}"),
            plan_cluster_rank_affinity(
                graph, num_cores, beta=beta,
                comm_scale=comm_scale, input_scale=input_scale,
                cluster_scale=cscale, cache_value_scale=cv,
            ),
        )

    if problem == 3 and mode == "full":
        # V5: preserve V4 values and search around the winning .55/.80 region.
        for cv in (0.35, 0.45, 0.55, 0.65, 0.80, 0.95):
            yield (
                f"cache-value-rank-cv={cv:g}",
                plan_cluster_rank_affinity(
                    graph, num_cores, beta=0.12,
                    comm_scale=comm_scale, input_scale=input_scale,
                    cluster_scale=0.30, cache_value_scale=cv,
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

    args = parser.parse_args()

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

    for index, (
        name,
        plan,
    ) in enumerate(
        candidate_plans(
            graph,
            args.problem,
            args.num_cores,
            args.search,
        ),
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