"""Dynamic reduction comparator: cheap full-bank policy, heavy local attention.

This adapts the reduction/construction principle to conditional music paths.
It is not a reproduction of L2R's distance-assisted modules or PRC inference.
"""
from __future__ import annotations
import math
from numbers import Integral
import torch
from torch import nn
from torch.nn import functional as F
from emomm_sequence.models.sequence_selector import SequenceSelector, Selection


def positive_dynamic_k(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError("dynamic_k must be a positive integer")
    return int(value)


class DynamicReductionSelector(SequenceSelector):
    def __init__(self, embedding_dim=768, hidden_dim=128, layers=2, heads=4, dynamic_k=16, **overrides):
        self.dynamic_k = positive_dynamic_k(dynamic_k)
        super().__init__(embedding_dim, hidden_dim, layers, heads, variant="dynamic_reduce", **overrides)
        layer = nn.TransformerEncoderLayer(hidden_dim, heads, hidden_dim * 2, dropout=0.0,
                                          activation="gelu", batch_first=True, norm_first=True)
        self.local_encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def _context(self, bank, condition, selected):
        if self.no_set:
            return torch.cat((condition, torch.zeros_like(condition)))
        tokens = bank[selected].unsqueeze(0)
        position = torch.arange(len(selected), device=bank.device, dtype=bank.dtype)[:, None]
        freq = torch.exp(torch.arange(0, self.hidden_dim, 2, device=bank.device, dtype=bank.dtype) * (-math.log(10000) / self.hidden_dim))
        pe = torch.zeros_like(tokens[0])
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * freq), torch.cos(position * freq)
        memory = self.set_encoder(tokens + pe.unsqueeze(0))
        context = self.pool(condition[None, None], memory, memory, need_weights=False)[0][0, 0]
        return torch.cat((condition, context))

    def select(self, x, text, n, budget, start, target, *, sample=False, teacher=None, dynamic_k=None, generator=None):
        if n < 2 or n > len(x) or start == target:
            raise ValueError("dynamic routing needs distinct endpoints and a feasible N>=2")
        dynamic_k = self.dynamic_k if dynamic_k is None else positive_dynamic_k(dynamic_k)
        projected = self.project(x)
        similarity = x @ text.T
        visited = torch.zeros(len(x), dtype=torch.bool, device=x.device)
        visited[start] = visited[target] = True
        selected = [int(start)]
        proposed = {int(start), int(target)}
        local_occurrences = 0
        logs, entropies, warm = [], [], []
        zero = projected.sum() * 0
        for step in range(n - 2):
            scalars = x.new_tensor([0.0 if self.no_length else n / 20, min(budget, len(x)) / len(x), (n - 2 - step) / max(n - 2, 1)])
            condition = self.condition(torch.cat((text.flatten(), scalars)))
            joint = self._context(projected, condition, selected)
            score = (projected @ self.reduce_query(joint)) / math.sqrt(self.hidden_dim)
            score = score.masked_fill(visited, -torch.inf)
            k = min(dynamic_k, int((~visited).sum().item()))
            proposal_log, proposal_entropy = zero, zero
            if sample and teacher is None:
                remaining = score
                local_values = []
                for _ in range(k):
                    lp = F.log_softmax(remaining, dim=0)
                    chosen = int(torch.multinomial(lp.exp(), 1, generator=generator).item())
                    proposal_log = proposal_log + lp[chosen]
                    finite = torch.isfinite(lp)
                    proposal_entropy = proposal_entropy - (lp[finite].exp() * lp[finite]).sum() / k
                    local_values.append(chosen)
                    remaining = remaining.masked_fill(F.one_hot(torch.tensor(chosen, device=x.device), len(x)).bool(), -torch.inf)
                local = torch.tensor(local_values, device=x.device)
            else:
                local = torch.topk(score, k).indices
            if teacher is not None:
                wanted = int(teacher[step + 1])
                warm.append(-F.log_softmax(score, dim=0)[wanted])
                if wanted not in local.tolist():
                    local = torch.cat((local[:-1], torch.tensor([wanted], device=x.device)))
            proposed.update(local.tolist())
            local_occurrences += len(local)
            # Only the reduced set receives candidate-to-candidate self-attention.
            query = self.query(joint)
            tokens = torch.cat((query[None], projected[local]), dim=0)[None]
            encoded = self.local_encoder(tokens)[0]
            local_score = (encoded[1:] @ encoded[0]) / math.sqrt(self.hidden_dim) + self.relevance(similarity[local]).squeeze(-1)
            lp = F.log_softmax(local_score, dim=0)
            entropies.append(-(lp.exp() * lp).sum() + proposal_entropy)
            if teacher is not None:
                action = int(torch.nonzero(local == wanted)[0, 0].item())
                warm.append(-lp[action])
            elif sample:
                action = int(torch.multinomial(lp.exp(), 1, generator=generator).item())
            else:
                action = int(torch.argmax(lp).item())
            choice = int(local[action].item())
            logs.append(lp[action] + proposal_log)
            selected.append(choice)
            visited = visited.clone()
            visited[choice] = True
        selected.append(int(target))
        result = Selection(selected, torch.stack(logs).sum() if logs else zero,
                           torch.stack(entropies).mean() if entropies else zero,
                           torch.stack(warm).mean() if warm else zero)
        result.proposed_indices = sorted(proposed)
        result.local_scored_occurrences = local_occurrences
        result.dynamic_k = dynamic_k
        return result


def build_dynamic_selector(config, **overrides):
    c = config["route_b"]
    return DynamicReductionSelector(config["assets"]["embedding_dim"], c["hidden_dim"], c["attention_layers"],
                                    c["attention_heads"], dynamic_k=c.get("dynamic_k", 16), **overrides)
