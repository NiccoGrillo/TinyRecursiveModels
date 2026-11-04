"""Paired perturbations of z_L, leaving the immediate readout exactly invariant."""
import torch

from .model import State, clone_state, reference_margins, rms


def future_objective(adapter, state, batch, reference, horizon, temperature=0.5):
    for _ in range(horizon):
        state, logits = adapter.step(state, batch, differentiable=True)
    margins = reference_margins(logits, reference)
    # Smooth minimum across cells; competitor max is piecewise differentiable.
    return -temperature * torch.logsumexp(-margins / temperature, dim=-1)


def gradient_direction(adapter, state, batch, horizon):
    if horizon < 1:
        raise ValueError("Probe horizon must be positive")
    low = state.z_L.detach().clone().requires_grad_(True)
    reference = adapter.readout(state).detach().argmax(-1)
    objective = future_objective(adapter, State(state.z_H.detach(), low), batch, reference, horizon)
    gradient, = torch.autograd.grad(objective.sum(), low)
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("Non-finite future-margin gradient")
    return -gradient.detach()  # reduces future margin of the currently decoded answer


def perturb(state, direction, epsilon, tolerance=0.05):
    """Relative per-example RMS budget, with measured rounding and budget flags."""
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    base_rms = rms(state.z_L)
    direction_rms = rms(direction)
    scale = epsilon * base_rms / direction_rms.clamp_min(1e-30)
    requested = direction.float() * scale[:, None, None]
    result = clone_state(state)
    result.z_L = (state.z_L.float() + requested).to(state.z_L.dtype)
    actual = result.z_L.float() - state.z_L.float()
    actual_rms = rms(actual)
    relative = actual_rms / base_rms.clamp_min(1e-30)
    error = (relative - epsilon).abs() / max(epsilon, 1e-30)
    reports = []
    for i in range(len(base_rms)):
        reports.append(dict(requested_relative_rms=epsilon, actual_relative_rms=relative[i].item(),
                            actual_delta_l2=actual[i].norm().item(),
                            base_z_l_rms=base_rms[i].item(), direction_rms=direction_rms[i].item(),
                            relative_budget_error=error[i].item(),
                            budget_ok=bool(error[i] <= tolerance),
                            zero_delta=bool(actual_rms[i] == 0)))
    return result, reports


def random_direction(state, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(state.z_L.shape, generator=generator).to(state.z_L.device)
