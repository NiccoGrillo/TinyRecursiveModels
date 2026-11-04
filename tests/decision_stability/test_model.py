import json

import pytest
import torch

from experiments.decision_stability.model import State, load_model, read_config, smoke_config
from experiments.decision_stability.interventions import future_objective, gradient_direction, perturb, random_direction


@pytest.fixture
def case():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    config = smoke_config()
    config.update(seq_len=9, hidden_size=16, puzzle_emb_ndim=16)
    model = load_model(config)
    batch = {"inputs": torch.randint(1, 11, (2, 9)), "puzzle_identifiers": torch.zeros(2, dtype=torch.long)}
    state = model.initial(batch)
    state, _ = model.step(state, batch)
    return model, batch, state


@pytest.mark.parametrize("mlp", [True, False])
def test_full_macro_matches_upstream_at_original_cycle_counts(mlp):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    config = smoke_config(mlp)
    config.update(seq_len=9, hidden_size=16, puzzle_emb_ndim=16, H_cycles=3, L_cycles=6, L_layers=2)
    adapter = load_model(config)
    batch = {"inputs": torch.randint(1, 11, (2, 9)), "puzzle_identifiers": torch.zeros(2, dtype=torch.long)}
    state = adapter.initial(batch)
    for _ in range(2):
        native, native_logits = adapter.step(state, batch)
        differentiable, diff_logits = adapter.step(state, batch, differentiable=True)
        torch.testing.assert_close(native.z_H, differentiable.z_H, rtol=0, atol=0)
        torch.testing.assert_close(native.z_L, differentiable.z_L, rtol=0, atol=0)
        torch.testing.assert_close(native_logits, diff_logits, rtol=0, atol=0)
        state = native
    assert adapter.block_calls_per_macro == 42


def test_multimacro_gradient_agrees_with_finite_difference(case):
    adapter, batch, state = case
    anchor = adapter.readout(state).argmax(-1)
    low = state.z_L.clone().requires_grad_(True)
    value = future_objective(adapter, State(state.z_H, low), batch, anchor, horizon=2).sum()
    gradient, = torch.autograd.grad(value, low)
    assert gradient.norm() > 1e-7
    direction = gradient / gradient.norm()
    epsilon = 1e-3
    with torch.no_grad():
        plus = future_objective(adapter, State(state.z_H, low + epsilon * direction), batch, anchor, 2).sum()
        minus = future_objective(adapter, State(state.z_H, low - epsilon * direction), batch, anchor, 2).sum()
    estimate = (plus - minus) / (2 * epsilon)
    torch.testing.assert_close(estimate, (gradient * direction).sum(), rtol=0.03, atol=2e-4)
    targeted = gradient_direction(adapter, state, batch, horizon=2)
    torch.testing.assert_close(targeted, -gradient)


def test_readout_invariance_and_per_example_budget(case):
    adapter, batch, state = case
    direction = random_direction(state, 3)
    assert torch.equal(direction, random_direction(state, 3))
    for sign in (1, -1):
        branch, reports = perturb(state, sign * direction, 0.03)
        assert torch.equal(branch.z_H, state.z_H)
        assert torch.equal(adapter.readout(branch), adapter.readout(state))
        for report in reports:
            assert report["actual_relative_rms"] == pytest.approx(0.03, rel=1e-5)
            assert report["budget_ok"] and not report["zero_delta"]
    zero, reports = perturb(state, torch.zeros_like(direction), 0.03)
    assert all(r["zero_delta"] and not r["budget_ok"] for r in reports)
    assert torch.equal(zero.z_L, state.z_L)


def test_zero_perturbation_entire_trajectory_identity(case):
    adapter, batch, clean = case
    branch, _ = perturb(clean, random_direction(clean, 0), 0)
    for _ in range(7):
        clean, logits = adapter.step(clean, batch)
        branch, perturbed_logits = adapter.step(branch, batch)
        assert torch.equal(clean.z_L, branch.z_L)
        assert torch.equal(clean.z_H, branch.z_H)
        assert torch.equal(logits, perturbed_logits)


def test_no_reset_after_native_halting_horizon(case):
    adapter, batch, _ = case
    state = adapter.initial(batch)
    for _ in range(adapter.config.halt_max_steps):
        state, _ = adapter.step(state, batch)
    expected, logits = adapter.inner(state, batch)[:2]
    actual, actual_logits = adapter.step(state, batch)
    reset, reset_logits = adapter.step(adapter.initial(batch), batch)
    assert torch.equal(actual.z_L, expected.z_L)
    assert torch.equal(logits, actual_logits)
    assert not torch.allclose(reset_logits, actual_logits)


@pytest.mark.parametrize("prefix", ["", "model.", "_orig_mod.model."])
def test_checkpoint_wrappers_and_strict_loading(case, tmp_path, prefix):
    adapter, batch, state = case
    weights = {prefix + k: v for k, v in adapter.model.state_dict().items()}
    checkpoint = tmp_path / "weights.pt"
    torch.save(weights, checkpoint)
    loaded = load_model(adapter.config.model_dump(), checkpoint)
    assert torch.equal(loaded.readout(state), adapter.readout(state))
    weights.pop(next(iter(weights)))
    torch.save(weights, checkpoint)
    with pytest.raises(RuntimeError):
        load_model(adapter.config.model_dump(), checkpoint)


def test_config_rejects_wrong_backend_and_unresolved_hydra(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("arch:\n  name: other@Model\n")
    with pytest.raises(ValueError, match="original TRM"):
        read_config(path, {}, 2)
    path.write_text("hidden_size: ${.hidden_size}\n")
    with pytest.raises(ValueError, match="resolved"):
        read_config(path, {}, 2)


def test_realized_bfloat16_rounding_is_visible(case):
    _, _, state = case
    state.z_L = state.z_L.to(torch.bfloat16)
    _, reports = perturb(state, random_direction(state, 4), 1e-8)
    assert all(r["zero_delta"] and not r["budget_ok"] for r in reports)
