"""Inference-only candidate selectors with explicit shared endpoint budget."""
from __future__ import annotations
import numpy as np
from emomm_sequence.inference.sequence_planner import beam_search


def round_robin(lists, selected, budget):
    result = list(dict.fromkeys(map(int, selected)))
    seen = set(result)
    positions = [0] * len(lists)
    while len(result) < budget:
        changed = False
        for k, values in enumerate(lists):
            while positions[k] < len(values) and int(values[positions[k]]) in seen:
                positions[k] += 1
            if positions[k] < len(values):
                idx = int(values[positions[k]])
                seen.add(idx)
                result.append(idx)
                positions[k] += 1
                changed = True
                if len(result) == budget:
                    break
        if not changed:
            break
    return np.asarray(result, dtype=np.int64)


def select_candidates(problem, budget, method="a_grid", seed=42):
    budget = min(budget, problem.m)
    if budget < problem.n:
        raise ValueError("candidate budget must be >= requested length")
    anchors = [problem.start, problem.target] if problem.n > 1 else [problem.start]
    if method == "random":
        orders = [np.random.default_rng(seed).permutation(problem.m)]
    elif method == "three_route":
        orders = [np.lexsort((problem.ids, problem.emotion[0])), np.lexsort((problem.ids, problem.emotion[-1]))]
        if np.linalg.norm(problem.query[2]) > 0:
            orders.append(np.lexsort((problem.ids, problem.preference)))
    elif method in {"a_grid", "a_refine"}:
        orders = [np.lexsort((problem.ids, row)) for row in problem.nodes]
    else:
        raise ValueError(f"unknown candidate method: {method}")
    return round_robin(orders, anchors, budget)


def independent_path(problem):
    selected = [problem.start]
    if problem.n == 1:
        return tuple(selected)
    used = {problem.start, problem.target}
    for row in problem.nodes[1:-1]:
        i = next(int(i) for i in np.lexsort((problem.ids, row)) if int(i) not in used)
        selected.append(i)
        used.add(i)
    return tuple([*selected, problem.target])


def refine(problem, candidates, width=50, rounds=2, proposals=8):
    candidates = np.asarray(candidates)
    current = beam_search(problem, candidates, width=width)
    expansions = current.expansions
    visited = set(candidates.tolist())
    if current.status != "ok":
        return candidates, current, {"solver_calls": 1, "cumulative_candidates": len(visited), "full_scans": 0}
    scans = 0
    calls = 1
    for _ in range(rounds):
        orders = []
        path = current.path
        for k in range(1, problem.n - 1):
            score = problem.nodes[k].copy()
            for adjacent in (path[k - 1], path[k + 1]):
                score += problem.edge_weight * np.clip((1 - problem.x @ problem.x[adjacent]) / 2, 0, 1)
            score[list(set(path) - {path[k]})] = np.inf
            orders.append(np.lexsort((problem.ids, score))[:proposals])
            scans += 1
        proposed = round_robin(orders, path, len(candidates))
        proposed = round_robin([candidates], proposed, len(candidates))
        result = beam_search(problem, proposed, width=width)
        expansions += result.expansions
        visited.update(proposed.tolist())
        calls += 1
        if result.status == "ok" and result.objective <= current.objective:
            current, candidates = result, proposed
    current.expansions = expansions
    return candidates, current, {"solver_calls": calls, "cumulative_candidates": len(visited), "full_scans": scans}
