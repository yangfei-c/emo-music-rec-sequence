"""VA extension: joint positional/step interval adversary and path beam.

No song labels are consumed here. Fixed-path maximization is exact for a
rectangular uncertainty set; beam search over unique playlists is approximate.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from emomm_sequence.inference.sequence_planner import Plan


@dataclass
class VAProblem:
    ids: np.ndarray
    median: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    target_path: np.ndarray
    preference: np.ndarray
    start: int
    target: int
    has_preference: bool = True

    def __post_init__(self):
        self.ids = np.asarray(self.ids, dtype=str)
        self.median, self.lower, self.upper = [np.asarray(v, dtype=np.float64) for v in (self.median, self.lower, self.upper)]
        self.target_path = np.asarray(self.target_path, dtype=np.float64)
        self.preference = np.asarray(self.preference, dtype=np.float64)
        if any(v.shape != (len(self.ids), 2) for v in (self.median, self.lower, self.upper)):
            raise ValueError("VA predictions must align with IDs")
        if self.target_path.ndim != 2 or self.target_path.shape[1] != 2 or len(self.target_path) < 2:
            raise ValueError("VA path evaluation needs at least two positions")
        if self.preference.shape != (len(self.ids),):
            raise ValueError("preference scores must align")
        for v in (self.median, self.lower, self.upper, self.target_path, self.preference):
            if not np.isfinite(v).all() or np.any((v < 0) | (v > 1)):
                raise ValueError("VA/preference values must lie in [0,1]")
        if np.any(self.lower > self.median) or np.any(self.median > self.upper):
            raise ValueError("unordered intervals")
        if self.start == self.target or len(set(self.ids.tolist())) != len(self.ids):
            raise ValueError("unique IDs and distinct endpoints required")

    @property
    def n(self):
        return len(self.target_path)

    @property
    def step(self):
        return (self.target_path[-1] - self.target_path[0]) / (self.n - 1)

    @property
    def weights(self):
        return (0.5, 0.25, 0.25) if self.has_preference else (2/3, 0.0, 1/3)

    def ends(self, indices):
        return np.stack((self.lower[indices], self.upper[indices]), axis=-1)

    def point_cost(self, path):
        path = np.asarray(path, dtype=int)
        wpos, wpref, wstep = self.weights
        y = self.median[path]
        return float(wpos * np.abs(y - self.target_path).mean() + wpref * self.preference[path].mean()
                     + wstep * np.abs(np.diff(y, axis=0) - self.step).mean() / 2)

    def advance(self, dp, previous_ends, next_ends, position):
        # [W,C,axis,previous-state,next-state]. Each song variable is shared.
        wpos, _, wstep = self.weights
        transition = np.abs(next_ends[None, :, :, None, :] - previous_ends[:, None, :, :, None]
                            - self.step[None, None, :, None, None])
        node = np.abs(next_ends - self.target_path[position][None, :, None])
        return np.max(dp[:, None, :, :, None] + transition * wstep / (4 * (self.n - 1)), axis=3) + node[None] * wpos / (2 * self.n)

    def cost(self, path):
        path = np.asarray(path, dtype=int)
        if len(path) != self.n or len(set(path.tolist())) != self.n or path[0] != self.start or path[-1] != self.target:
            raise ValueError("invalid complete path")
        wpos, wpref, _ = self.weights
        ends = self.ends(path)
        dp = np.abs(ends[0] - self.target_path[0, :, None]) * wpos / (2 * self.n)
        for k in range(1, self.n):
            dp = self.advance(dp[None], ends[k-1:k], ends[k:k+1], k)[0, 0]
        return float(dp.max(axis=-1).sum() + wpref * self.preference[path].mean())


def va_beam_search(problem: VAProblem, candidates, width=50, timeout=30.0):
    begin = time.monotonic()
    c = np.asarray(sorted(set(map(int, candidates)), key=lambda i: problem.ids[i]))
    if len(c) < problem.n or problem.start not in c or problem.target not in c:
        return Plan((), 1.0, "insufficient_candidates")
    if problem.n == 2:
        path = (problem.start, problem.target)
        return Plan(path, problem.cost(path))
    s, t = int(np.flatnonzero(c == problem.start)[0]), int(np.flatnonzero(c == problem.target)[0])
    ends = problem.ends(c)
    wpos, wpref, _ = problem.weights
    node_pref = wpref * problem.preference[c] / problem.n
    paths = np.array([[s]], dtype=np.int64)
    dp = (np.abs(ends[s] - problem.target_path[0, :, None]) * wpos / (2 * problem.n))[None]
    pref = np.array([node_pref[s]])
    expansions = 0
    for k in range(1, problem.n - 1):
        if time.monotonic() - begin > timeout:
            return Plan((), 1.0, "timeout", expansions=expansions)
        values = problem.advance(dp, ends[paths[:, -1]], ends, k)
        next_pref = pref[:, None] + node_pref[None]
        scored = values
        if k == problem.n - 2:
            scored = problem.advance(values.reshape(-1, 2, 2), np.tile(ends, (len(paths), 1, 1)), ends[t:t+1], k + 1)
            scored = scored[:, 0].reshape(len(paths), len(c), 2, 2)
            next_pref = next_pref + node_pref[t]
        scores = scored.max(axis=-1).sum(axis=-1) + next_pref
        scores[np.arange(len(paths))[:, None], paths] = np.inf
        scores[:, t] = np.inf
        expansions += int(np.isfinite(scores).sum())
        count = min(width, int(np.isfinite(scores).sum()))
        if not count:
            return Plan((), 1.0, "no_unique_path", expansions=expansions)
        ix = np.argsort(scores.ravel(), kind="stable")[:count]
        parent, song = np.divmod(ix, len(c))
        paths = np.column_stack((paths[parent], song))
        dp, pref = values[parent, song], next_pref[parent, song]
    proposals = [tuple(c[np.append(path, t)].tolist()) for path in paths]
    best = min(proposals, key=lambda path: (problem.cost(path), tuple(problem.ids[list(path)])))
    return Plan(best, problem.cost(best), expansions=expansions)
