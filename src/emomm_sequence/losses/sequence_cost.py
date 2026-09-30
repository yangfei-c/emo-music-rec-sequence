"""Pool-independent objective shared by inference, training and evaluation."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.isfinite(x).all() or np.any(norm < 1e-10):
        raise ValueError("nonfinite or zero embedding")
    return x / norm


@dataclass
class SequenceProblem:
    x: np.ndarray
    ids: np.ndarray
    query: np.ndarray  # start, target, preference (zero if absent)
    emotion: np.ndarray  # [N,M], unweighted distances
    preference: np.ndarray  # [M]
    nodes: np.ndarray  # [N,M], weighted AND divided by N
    edge_weight: float  # already divided by N-1
    start: int
    target: int

    @property
    def n(self) -> int:
        return len(self.nodes)

    @property
    def m(self) -> int:
        return len(self.x)

    def edges(self, indices: np.ndarray) -> np.ndarray:
        xx = self.x[indices]
        return np.clip((1 - xx @ xx.T) * 0.5, 0, 1) * self.edge_weight

    def cost(self, path: list[int] | tuple[int, ...] | np.ndarray) -> float:
        p = np.asarray(path, dtype=np.int64)
        if len(p) != self.n or len(set(p.tolist())) != self.n:
            raise ValueError("sequence must be full length and unique")
        if p[0] != self.start or (self.n > 1 and p[-1] != self.target):
            raise ValueError("fixed endpoints changed")
        unary = self.nodes[np.arange(self.n), p].sum(dtype=np.float64)
        edge = np.clip((1 - np.sum(self.x[p[:-1]] * self.x[p[1:]], axis=1)) * 0.5, 0, 1)
        return float(unary + self.edge_weight * edge.sum(dtype=np.float64))

    def parts(self, path: list[int] | tuple[int, ...]) -> dict:
        p = np.asarray(path)
        return {
            "emotion_distance": float(self.emotion[np.arange(self.n), p].mean()),
            "preference_distance": float(self.preference[p].mean()),
            "smooth_distance": float(np.clip((1 - np.sum(self.x[p[:-1]] * self.x[p[1:]], axis=1)) / 2, 0, 1).mean()) if self.n > 1 else 0.0,
            "objective": self.cost(path),
        }


def make_problem(x, ids, start_text, target_text, preference_text, n, endpoints=None,
                 weights=(0.5, 0.25, 0.25)) -> SequenceProblem:
    x = normalize(x)
    ids = np.asarray(ids, dtype=str)
    if len(ids) != len(x) or len(set(ids.tolist())) != len(ids):
        raise ValueError("IDs must align and be unique (deduplicate content before this layer)")
    if not isinstance(n, int) or n < 1 or n > len(x):
        raise ValueError("invalid requested length")
    es, et = normalize(start_text), normalize(target_text)
    ep = np.zeros_like(es) if preference_text is None else normalize(preference_text)
    u = np.linspace(0, 1, n, dtype=np.float32) if n > 1 else np.array([0.5], dtype=np.float32)
    z = (1 - u[:, None]) * es + u[:, None] * et
    zero = np.linalg.norm(z, axis=1) < 1e-8
    for k in np.flatnonzero(zero):
        z[k] = es if u[k] <= 0.5 else et
    emotion = np.clip((1 - normalize(z) @ x.T) / 2, 0, 1)
    pref = np.zeros(len(x), dtype=np.float32) if preference_text is None else np.clip((1 - x @ ep) / 2, 0, 1)
    w = np.asarray(weights, dtype=float)
    if w.shape != (3,) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("invalid objective weights")
    if preference_text is None:
        w[1] = 0
    w /= w.sum()
    if endpoints is None:
        if n == 1:
            start = int(np.lexsort((ids, w[0] * emotion[0] + w[1] * pref))[0])
            target = start
        else:
            ew = 0.75 if preference_text is not None else 1.0
            ec = ew * emotion[[0, -1]] + (1 - ew) * pref
            top = [np.lexsort((ids, c))[:20] for c in ec]
            _, _, _, start, target = min((float(ec[0, i] + ec[1, j]), ids[i], ids[j], int(i), int(j))
                                        for i in top[0] for j in top[1] if i != j)
    else:
        start, target = map(int, endpoints)
        if not (0 <= start < len(x) and 0 <= target < len(x)) or (n > 1 and start == target):
            raise ValueError("invalid endpoints")
    return SequenceProblem(x, ids, np.stack((es, et, ep)), emotion, pref,
                           (w[0] * emotion + w[1] * pref) / n,
                           float(w[2] / max(n - 1, 1)), start, target)
