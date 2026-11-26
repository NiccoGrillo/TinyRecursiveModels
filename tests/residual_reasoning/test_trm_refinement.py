import pytest
import torch

from experiments.residual_reasoning.model import Architecture
from experiments.residual_reasoning.trm_refinement import TRMRefiner


@pytest.mark.parametrize("method", ("unrestricted", "stateless", "recurrent"))
def test_refiner_preserves_native_logits_and_learns(method):
    torch.manual_seed(11)
    refiner = TRMRefiner(Architecture(width=12, heads=2, layers=1), 16, method)
    inputs = torch.randint(0, 11, (2, 5))
    z_h, z_l = (torch.randn(2, 7, 16, requires_grad=True) for _ in range(2))
    logits = torch.randn(2, 5, 11, requires_grad=True)
    labels = torch.randint(2, 11, (2, 5))
    context, base, delta, memory = refiner.prepare(inputs, z_h, z_l, logits)
    assert torch.equal(refiner.readout(base, delta), logits)
    updated, _ = refiner.advance(context, delta, memory, train=False)
    assert torch.equal(refiner.readout(base, updated), logits)

    optimizer = torch.optim.AdamW(refiner.parameters(), lr=0.001)
    loss = refiner.training_loss(inputs, z_h, z_l, logits, labels, loops=2)
    loss.backward()
    assert refiner.gain.grad.abs() > 0
    optimizer.step()
    optimizer.zero_grad()
    loss = refiner.training_loss(inputs, z_h, z_l, logits, labels, loops=2)
    loss.backward()
    assert z_h.grad is None and z_l.grad is None and logits.grad is None
    assert all(p.grad is not None and p.grad.isfinite().all() for p in refiner.parameters())
