"""Small independent wiring/derivative checks for the closing gradient graph."""
from types import SimpleNamespace
import torch
from torch import nn

from experiments.permworld_closing_audit_20261110.gradient_graph import early_logits, margin


def toy():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(80311)
        model = nn.Module()
        model.config = SimpleNamespace(d_model=4)
        model.recurrent_layers = nn.ModuleList([nn.GRU(4, 4, batch_first=True) for _ in range(4)])
        model.interlayer_dropout = nn.ModuleList([nn.Dropout(.1) for _ in range(3)])
        model.embedding_dropout = nn.Dropout(.1)
        model.token_embedding = nn.Embedding(31, 4)
        model.position_embedding = nn.Embedding(32, 4)
        model.final_norm = nn.LayerNorm(4)
        model.lm_head = nn.Linear(4, 31)
    return model.double().eval().requires_grad_(False)


def gate(layer, x, h):
    a = x @ layer.weight_ih_l0.T + layer.bias_ih_l0
    b = h @ layer.weight_hh_l0.T + layer.bias_hh_l0
    ar, az, an = a.chunk(3, -1); br, bz, bn = b.chunk(3, -1)
    r = (ar + br).sigmoid(); z = (az + bz).sigmoid()
    n = (an + r * bn).tanh()
    return (1 - z) * n + z * h


def test_graph_matches_independent_manual_gate_unroll():
    model = toy()
    prev = torch.arange(32, dtype=torch.float64).reshape(2, 4, 4) / 17
    h = prev[:, 0] + .2
    ids = torch.tensor([[3, 4, 5], [6, 7, 8]])
    actual = early_logits(model, prev, h, ids, 11)
    states = [h]
    for layer in range(1, 4): states.append(gate(model.recurrent_layers[layer], states[-1], prev[:, layer]))
    for offset in range(3):
        value = model.token_embedding(ids[:, offset]) + model.position_embedding(torch.tensor(11 + offset))
        for layer in range(4):
            states[layer] = gate(model.recurrent_layers[layer], value, states[layer]); value = states[layer]
    expected = model.lm_head(model.final_norm(value))
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)


def test_derivative_reaches_raw_first_state_and_does_not_train_parameters():
    model = toy()
    previous = torch.ones((2, 4, 4), dtype=torch.float64) / 3
    h = torch.full((2, 4), .13, dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([[3, 4, 5], [6, 7, 8]])
    result = early_logits(model, previous, h, ids, 11)
    gradient, = torch.autograd.grad(margin(result, [2, 4], [1, -1]).sum(), h)
    assert bool(torch.isfinite(gradient).all()) and float(gradient.norm()) > 0
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters())


def test_raw_gradient_matches_fixed_direction_finite_difference():
    model = toy()
    previous = torch.ones((2, 4, 4), dtype=torch.float64) / 3
    h = torch.full((2, 4), .13, dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([[3, 4, 5], [6, 7, 8]])
    direction = torch.tensor([[1., 2, 3, 4], [-1., 3, 2, 1]], dtype=torch.float64)
    direction = direction / direction.norm(dim=1, keepdim=True)
    gradient, = torch.autograd.grad(margin(early_logits(model, previous, h, ids, 11), [2, 4], [1, -1]).sum(), h)
    plus = margin(early_logits(model, previous, h + 1e-5 * direction, ids, 11), [2, 4], [1, -1])
    minus = margin(early_logits(model, previous, h - 1e-5 * direction, ids, 11), [2, 4], [1, -1])
    torch.testing.assert_close((gradient * direction).sum(1), (plus - minus) / 2e-5, atol=1e-9, rtol=0)


def test_margin_uses_recipient_local_goal_for_both_signs():
    logits = torch.arange(62).reshape(2, 31).double()
    torch.testing.assert_close(margin(logits, [7, 7], [1, -1]), torch.tensor([1., -1.], dtype=torch.float64))
