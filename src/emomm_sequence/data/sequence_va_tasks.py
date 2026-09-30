"""Supervised extension inputs; caches contain predictions, never evaluation truth."""
from __future__ import annotations
import json
import numpy as np
from emomm_sequence.data.sequence_tasks import Bank, make_task_problem
from emomm_sequence.inference.sequence_va import VAProblem, va_beam_search
from emomm_sequence.inference.trajectory_candidates import round_robin

VA_ENDPOINT_PROTOCOL = "frozen_anchors_on_eligible_bank_v1"


class VASequenceTask(VAProblem):
    @property
    def m(self):
        return len(self.ids)


class VAInputs:
    def __init__(self, root, mode):
        if mode not in ("point", "raw_interval", "cqr_interval"):
            raise ValueError(mode)
        self.mode = mode
        self.directory = root / "cache" / "va_extension"
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        with np.load(self.directory / "text.npz", allow_pickle=False) as z:
            self.text_va = z["predicted_va"]
        self.cache = {}

    def eligible_task(self, task):
        task = dict(task)
        if task["split"] == "validation":
            allowed = np.asarray(self.manifest["validation_eligibility"][task["source"]]["allowed_bank_indices"])
            if task.get("indices") is not None:
                allowed = allowed[np.isin(allowed, task["indices"])]
            if len(allowed) < task["num_tracks"]:
                raise ValueError("insufficient policy-only validation songs")
            task["indices"] = allowed.tolist()
            task["budget"] = min(task["budget"], len(allowed))
        return task

    def make(self, bank, task, texts):
        task = self.eligible_task(task)
        selected, frozen = make_task_problem(bank, task, texts)
        key = (bank.source, bank.split)
        if key not in self.cache:
            with np.load(self.directory / f"{bank.source}_{bank.split}.npz", allow_pickle=False) as z:
                self.cache[key] = {k: z[k] for k in z.files}
            if not np.array_equal(self.cache[key]["track_ids"], bank.ids):
                raise ValueError("supervised prediction ID order mismatch")
        data = self.cache[key]
        ix = np.arange(len(bank.ids)) if task.get("indices") is None else np.asarray(task["indices"])
        median = data["quantiles"][ix, :, 1]
        if self.mode == "point":
            lower, upper = median, median
        elif self.mode == "raw_interval":
            lower, upper = data["quantiles"][ix, :, 0], data["quantiles"][ix, :, 2]
        else:
            lower, upper = data["cqr_lower"][ix], data["cqr_upper"][ix]
        has_pref = task["preference_text_id"] is not None
        pref = np.zeros(len(ix))
        if has_pref:
            col = int(np.flatnonzero(data["preference_text_ids"] == task["preference_text_id"])[0])
            pref = data["preference_distances"][ix, col]
        ends = self.text_va[[task["start_text_id"], task["target_text_id"]]]
        targets = np.linspace(*ends, task["num_tracks"])
        # Blueprint section 5.3: retain Frozen anchors on the SAME eligible bank.
        # VA predictions and interval widths must not change endpoint selection.
        problem = VASequenceTask(selected.ids, median, lower, upper, targets, pref,
                                 frozen.start, frozen.target, has_pref)
        # Both A and B have the same frozen preference score as well as the VA predictions.
        extra = np.concatenate((median, lower, upper, pref[:, None]), axis=1).astype("float32")
        problem.x = np.concatenate((frozen.x, extra), axis=1).astype("float32")
        q_extra = np.zeros((3, 7), dtype="float32")
        q_extra[:2, :6] = np.tile(ends, (1, 3))
        q_extra[2, 6] = float(has_pref)
        problem.query = np.concatenate((frozen.query, q_extra), axis=1).astype("float32")
        positional = np.maximum(np.abs(lower[None] - targets[:, None]), np.abs(upper[None] - targets[:, None])).mean(axis=-1)
        wp, wr, _ = problem.weights
        problem.nodes = (wp * positional + wr * pref) / problem.n
        return selected, problem, task


def va_candidates(problem, budget):
    orders = [np.lexsort((problem.ids, row)) for row in problem.nodes]
    return round_robin(orders, [problem.start, problem.target], min(budget, problem.m))


def va_refine(problem, candidates, width=50, rounds=2, proposals=8):
    current = va_beam_search(problem, candidates, width=width)
    if current.status != "ok":
        return candidates, current
    for _ in range(rounds):
        orders = []
        # Exact fixed-path robust objective guides each replacement; no marginal-sum shortcut.
        for k in range(1, problem.n - 1):
            possible = np.lexsort((problem.ids, problem.nodes[k]))
            possible = [int(i) for i in possible if i not in set(current.path) - {current.path[k]}][:proposals]
            ordered = sorted(possible, key=lambda i: (problem.cost((*current.path[:k], i, *current.path[k+1:])), problem.ids[i]))
            orders.append(ordered)
        proposal = round_robin(orders, current.path, len(candidates))
        proposal = round_robin([candidates], proposal, len(candidates))
        result = va_beam_search(problem, proposal, width=width)
        if result.status == "ok" and result.objective <= current.objective:
            candidates, current = proposal, result
    return candidates, current
