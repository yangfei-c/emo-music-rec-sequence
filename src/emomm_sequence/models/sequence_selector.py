"""Small condition/set policy; backbone and music bank are never optimized."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Selection:
    indices: list[int]
    log_prob: torch.Tensor
    entropy: torch.Tensor
    warm_loss: torch.Tensor


class SequenceSelector(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=128, layers=2, heads=4,
                 variant="set", no_set=False, no_length=False):
        super().__init__()
        if variant not in {"set", "direct_path", "dynamic_reduce"}:
            raise ValueError(variant)
        self.variant, self.no_set, self.no_length = variant, no_set, no_length
        self.hidden_dim = hidden_dim
        self.project = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.condition = nn.Sequential(nn.Linear(embedding_dim * 3 + 3, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        layer = nn.TransformerEncoderLayer(hidden_dim, heads, hidden_dim * 2, dropout=0.0,
                                          activation="gelu", batch_first=True, norm_first=True)
        self.set_encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.pool = nn.MultiheadAttention(hidden_dim, heads, dropout=0.0, batch_first=True)
        self.query = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relevance = nn.Sequential(nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.reduce_query = nn.Linear(hidden_dim * 2, hidden_dim) if variant == "dynamic_reduce" else None

    def _scores(self, bank, condition, selected, similarities):
        if self.no_set:
            context = torch.zeros_like(condition)
        else:
            tokens = bank[selected].unsqueeze(0)
            if self.variant != "set":
                # Analytic positions support untrained lengths without a finite lookup table.
                position = torch.arange(len(selected), device=bank.device, dtype=bank.dtype)[:, None]
                freq = torch.exp(torch.arange(0, self.hidden_dim, 2, device=bank.device, dtype=bank.dtype) * (-math.log(10000) / self.hidden_dim))
                pe = torch.zeros_like(tokens[0])
                pe[:, 0::2] = torch.sin(position * freq)
                pe[:, 1::2] = torch.cos(position * freq)
                tokens = tokens + pe.unsqueeze(0)
            memory = self.set_encoder(tokens)
            context = self.pool(condition[None, None, :], memory, memory, need_weights=False)[0][0, 0]
        joint = torch.cat((condition, context))
        q = self.query(joint)
        score = (bank @ q) / math.sqrt(self.hidden_dim) + self.relevance(similarities).squeeze(-1)
        return score, joint

    def select(self, x, text, n, budget, start, target, *, sample=False, teacher=None,
               dynamic_k=16, generator=None):
        if x.ndim != 2 or text.shape != (3, x.shape[1]):
            raise ValueError("bad feature shapes")
        budget = min(int(budget), len(x))
        if budget < n or (n > 1 and start == target):
            raise ValueError("invalid sequence budget/endpoints")
        projected = self.project(x)
        similarities = x @ text.T
        selected = [int(start)]
        if self.variant == "set" and n > 1:
            selected.append(int(target))
        visited = torch.zeros(len(x), dtype=torch.bool, device=x.device)
        visited[selected] = True
        if self.variant != "set" and n > 1:
            visited[target] = True
        actions = (budget - len(selected)) if self.variant == "set" else max(n - 2, 0)
        teacher_remaining = [] if teacher is None else [int(i) for i in teacher[1:-1]]
        if teacher is not None:
            actions = len(teacher_remaining)
        logs, ents, warm = [], [], []
        zero = projected.sum() * 0
        for step in range(actions):
            nums = x.new_tensor([0.0 if self.no_length else n / 20, budget / max(len(x), 1), (actions - step) / max(actions, 1)])
            cond = self.condition(torch.cat((text.flatten(), nums)))
            score, joint = self._scores(projected, cond, selected, similarities)
            score = score.masked_fill(visited, -torch.inf)
            # Dynamic reduction is an explicitly adapted comparator, not an exact L2R reproduction.
            reducer_log = zero
            reducer_entropy = zero
            if self.reduce_query is not None:
                reduction_score = (projected @ self.reduce_query(joint)) / math.sqrt(self.hidden_dim)
                reduction_score = reduction_score.masked_fill(visited, -torch.inf)
                k = min(dynamic_k, int((~visited).sum().item()))
                if sample and teacher is None:
                    # Unordered local proposal is sampled without replacement; PL log-probs are explicit.
                    temp = reduction_score.clone()
                    local = []
                    for _ in range(k):
                        lp = F.log_softmax(temp, dim=0)
                        choice = int(torch.multinomial(lp.exp(), 1, generator=generator).item())
                        reducer_log = reducer_log + lp[choice]
                        reducer_entropy = reducer_entropy - (lp[torch.isfinite(lp)].exp() * lp[torch.isfinite(lp)]).sum() / k
                        local.append(choice)
                        temp = temp.masked_fill(F.one_hot(torch.tensor(choice, device=x.device), len(x)).bool(), -torch.inf)
                    local = torch.tensor(local, device=x.device)
                else:
                    local = torch.topk(reduction_score, k).indices
                if teacher is not None:
                    # Supervised reduction warm-up; the target is exposed only in training.
                    warm.append(-F.log_softmax(reduction_score, dim=0)[teacher_remaining[0]])
                    if teacher_remaining[0] not in local.tolist():
                        local = torch.cat((local[:-1], torch.tensor([teacher_remaining[0]], device=x.device)))
                keep = torch.zeros_like(visited)
                keep[local] = True
                score = score.masked_fill(~keep, -torch.inf)
            lp = F.log_softmax(score, dim=0)
            finite = torch.isfinite(lp)
            ents.append(-(lp[finite].exp() * lp[finite]).sum() + reducer_entropy)
            if teacher is not None:
                if self.variant == "set":
                    warm.append(-torch.logsumexp(lp[teacher_remaining], dim=0))
                    j = int(torch.randint(len(teacher_remaining), (1,), generator=generator, device=x.device).item())
                    choice = teacher_remaining.pop(j)
                else:
                    choice = teacher_remaining.pop(0)
                    warm.append(-lp[choice])
            elif sample:
                choice = int(torch.multinomial(lp.exp(), 1, generator=generator).item())
            else:
                choice = int(torch.argmax(lp).item())
            logs.append(lp[choice] + reducer_log)
            selected.append(choice)
            visited = visited.clone()
            visited[choice] = True
        if self.variant != "set" and n > 1:
            selected.append(int(target))
        return Selection(selected, torch.stack(logs).sum() if logs else zero,
                         torch.stack(ents).mean() if ents else zero,
                         torch.stack(warm).mean() if warm else zero)


def build_selector(config, variant="set", **overrides):
    c = config["route_b"]
    return SequenceSelector(config["assets"]["embedding_dim"], c["hidden_dim"], c["attention_layers"],
                            c["attention_heads"], variant=variant, **overrides)
