import copy

import pytest
import torch

from experiments.decision_stability.answer_mixing import AnswerMixer, mix_proposal
from experiments.decision_stability.model import Adapter, smoke_config
from models.losses import ACTLossHead
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1


def test_directional_update_steers_into_another_feature():
    answer = torch.zeros(1, 1, 2)
    proposal = torch.tensor([[[1.0, 0.0]]])
    u = torch.tensor([[[0.0, 1.0]]])
    v = proposal
    gate = torch.tensor([[[0.25]]])
    assert torch.equal(mix_proposal(answer, proposal, gate, (u, v)), torch.tensor([[[1.0, 0.25]]]))
    assert torch.equal(mix_proposal(answer, answer, gate, (u, v)), answer)
    assert torch.equal(mix_proposal(answer, proposal, gate), torch.tensor([[[1.25, 0.0]]]))


@pytest.mark.parametrize("method", ["scalar", "directional"])
def test_zero_gate_native_parity_and_act_training_gradient(method):
    torch.manual_seed(13)
    cfg = smoke_config()
    cfg.update(H_cycles=3, L_cycles=6, halt_max_steps=4)
    native = TinyRecursiveReasoningModel_ACTV1(cfg).eval()
    mixed = copy.deepcopy(native)
    mixer = AnswerMixer(cfg["hidden_size"], method)
    mixed.inner.answer_mixer = mixer
    batch = {"inputs": torch.full((2, cfg["seq_len"]), 2, dtype=torch.long),
             "labels": torch.full((2, cfg["seq_len"]), 2, dtype=torch.long),
             "puzzle_identifiers": torch.zeros(2, dtype=torch.long)}
    native_carry, mixed_carry = native.initial_carry(batch), mixed.initial_carry(batch)
    for _ in range(3):
        native_carry, expected = native(native_carry, batch)
        mixed_carry, actual = mixed(mixed_carry, batch)
        assert all(torch.equal(actual[k], expected[k]) for k in actual)
        assert torch.equal(native_carry.inner_carry.z_H, mixed_carry.inner_carry.z_H)
        assert torch.equal(native_carry.inner_carry.z_L, mixed_carry.inner_carry.z_L)

    head = ACTLossHead(mixed, "stablemax_cross_entropy").train()
    _, loss, _, _, _ = head(carry=head.initial_carry(batch), batch=batch, return_keys=[])
    loss.backward()
    gate_grad = mixer.gate.weight.grad
    assert torch.isfinite(gate_grad).all() and torch.count_nonzero(gate_grad)
    torch.optim.SGD(mixer.parameters(), lr=0.1).step()
    mixed.eval()
    adapter = Adapter(mixed)
    state = adapter.initial(batch)
    _, native_forward = adapter.step(state, batch)
    _, full_forward = adapter.step(state, batch, differentiable=True)
    assert torch.equal(native_forward, full_forward)
