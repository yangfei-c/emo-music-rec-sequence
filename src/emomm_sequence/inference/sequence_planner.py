"""Fixed-length unique-song beam search and certified small-bank subset DP."""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
from emomm_sequence.losses.sequence_cost import SequenceProblem


@dataclass
class Plan:
    path: tuple[int, ...]
    objective: float
    status: str = "ok"
    certified: bool = False
    expansions: int = 0


def beam_search(problem: SequenceProblem, candidates, width=50, timeout=30.0,
                perturb_seed=None, perturb_scale=0.0) -> Plan:
    begin = time.monotonic()
    c = np.asarray(sorted(set(map(int, candidates)), key=lambda i: problem.ids[i]), dtype=np.int64)
    if len(c) < problem.n or problem.start not in c or problem.target not in c:
        return Plan((), 1.0, "insufficient_candidates")
    if problem.n == 1:
        p = (problem.start,)
        return Plan(p, problem.cost(p))
    s, t = int(np.flatnonzero(c == problem.start)[0]), int(np.flatnonzero(c == problem.target)[0])
    edge = problem.edges(c)
    node = problem.nodes[:, c].astype(np.float64)
    noise = np.zeros_like(node)
    if perturb_seed is not None and perturb_scale:
        noise = np.random.default_rng(perturb_seed).gumbel(0, perturb_scale, node.shape) / problem.n
    paths = np.array([[s]], dtype=np.int64)
    costs = np.array([node[0, s]])
    expansions = 0
    for k in range(1, problem.n - 1):
        if time.monotonic() - begin > timeout:
            return Plan((), 1.0, "timeout", expansions=expansions)
        options = costs[:, None] + edge[paths[:, -1]] + node[k][None, :] + noise[k][None, :]
        options[np.arange(len(paths))[:, None], paths] = np.inf
        options[:, t] = np.inf
        if k == problem.n - 2:
            options += edge[:, t][None, :] + node[-1, t]
        expansions += int(np.isfinite(options).sum())
        # Stable ordering follows stable parent paths and canonical song IDs.
        flat = options.ravel()
        count = min(width, int(np.isfinite(flat).sum()))
        if not count:
            return Plan((), 1.0, "no_unique_path", expansions=expansions)
        ix = np.argsort(flat, kind="stable")[:count]
        rows, nxt = np.divmod(ix, len(c))
        costs = flat[ix]
        paths = np.column_stack((paths[rows], nxt))
    completed = [tuple(c[np.append(p, t)].tolist()) for p in paths]
    # Score the actual objective, never the perturbed proposal score.
    best = min(completed, key=lambda p: (problem.cost(p), tuple(problem.ids[list(p)])))
    return Plan(best, problem.cost(best), expansions=expansions)


def exact_search(problem: SequenceProblem, candidates=None, timeout=60.0) -> Plan:
    """Position-specific Held-Karp recurrence, with visited set in each state."""
    begin = time.monotonic()
    c = np.arange(problem.m) if candidates is None else np.asarray(sorted(set(map(int, candidates))))
    if len(c) < problem.n or problem.start not in c or problem.target not in c:
        return Plan((), 1.0, "insufficient_candidates", certified=True)
    if problem.n <= 2:
        result = beam_search(problem, c, width=1, timeout=timeout)
        result.certified = True
        return result
    inner = [int(i) for i in c if i not in (problem.start, problem.target)]
    if len(inner) > 20:
        raise ValueError("exact solver is restricted to <=20 interior nodes")
    local = np.array([problem.start, *inner, problem.target])
    edge = problem.edges(local)
    node = problem.nodes[:, local]
    states = {(0, 0): (float(node[0, 0]), (problem.start,))}
    expansions = 0
    for k in range(1, problem.n - 1):
        nxt_states = {}
        for (mask, last), (cost, path) in states.items():
            if expansions % 2048 == 0 and time.monotonic() - begin > timeout:
                return Plan((), 1.0, "uncertified_timeout", expansions=expansions)
            for bit, idx in enumerate(inner):
                if mask & (1 << bit):
                    continue
                nxt = bit + 1
                key = (mask | (1 << bit), nxt)
                proposal = (cost + float(edge[last, nxt]) + float(node[k, nxt]), path + (idx,))
                if key not in nxt_states or proposal < nxt_states[key]:
                    nxt_states[key] = proposal
                expansions += 1
        states = nxt_states
    _, path = min((cost + float(edge[last, -1]) + float(node[-1, -1]), path + (problem.target,))
                  for (_, last), (cost, path) in states.items())
    return Plan(path, problem.cost(path), certified=True, expansions=expansions)
