import torch
from emomm_sequence.models.sequence_selector import SequenceSelector, Selection
from emomm_sequence.losses.sequence_policy import policy_loss


def test_masks_and_gradient_update():
    torch.manual_seed(1)
    torch.set_num_threads(2)
    model = SequenceSelector(embedding_dim=8, hidden_dim=16, layers=1, heads=2)
    x = torch.nn.functional.normalize(torch.randn(12, 8), dim=-1)
    text = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
    out = model.select(x, text, 4, 7, 0, 11, sample=True)
    assert len(out.indices) == len(set(out.indices)) == 7
    assert out.indices[:2] == [0, 11]
    warm = model.select(x, text, 4, 7, 0, 11, teacher=[0, 4, 5, 11])
    loss = policy_loss(out, 0.2, 0.4, warm.warm_loss)
    loss.backward()
    assert sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None) > 0


def test_lower_cost_action_probability_increases():
    logits = torch.tensor([0.0, 0.0], requires_grad=True)
    logp = torch.log_softmax(logits, 0)[0]
    z = logp * 0
    loss = policy_loss(Selection([0], logp, z, z), 0.1, 0.5, entropy_weight=0)
    loss.backward()
    updated = logits.detach() - 0.1 * logits.grad
    assert torch.softmax(updated, 0)[0] > 0.5


def test_direct_path_respects_endpoint_and_unique_length():
    model = SequenceSelector(8, 16, 1, 2, variant="direct_path")
    x, text = torch.randn(9, 8), torch.randn(3, 8)
    out = model.select(x, text, 5, 8, 0, 8)
    assert out.indices[0] == 0 and out.indices[-1] == 8
    assert len(set(out.indices)) == 5
