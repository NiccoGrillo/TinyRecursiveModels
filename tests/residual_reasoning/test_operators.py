"""Analytic/autograd operator checks; none are trained-model experiments."""
import pytest
import torch

from experiments.residual_reasoning.operators import (
    bounded_residual_mix,
    residual_mixing_force,
)

DTYPE = torch.float64


def residual_pair(q):
    return torch.stack((q.square() - 1, 2 * q * (q - 1)), dim=-1)


def test_offdiagonal_mix_moves_a_strict_wrong_minimum():
    q = torch.tensor(0.0, dtype=DTYPE, requires_grad=True)
    c = residual_pair(q)
    energy = 0.5 * c.square().sum()
    gradient = torch.autograd.grad(energy, q, create_graph=True, retain_graph=True)[0]
    hessian = torch.autograd.grad(gradient, q, retain_graph=True)[0]
    assert gradient.item() == 0
    assert hessian.item() == 2  # A strict positive-energy local minimum.
    jacobian = torch.stack([
        torch.autograd.grad(c[i], q, retain_graph=True)[0] for i in range(2)
    ])
    for weights in (torch.tensor([0.1, 10.0]), torch.tensor([9.0, 0.2])):
        assert (jacobian * weights * c).sum().item() == 0
    factors = torch.tensor([[1.0], [-1.0]], dtype=DTYPE)
    force = residual_mixing_force(q, c, factors, m0=1.0, b=1.0)
    torch.testing.assert_close(force, torch.tensor(-2.0 / 3.0, dtype=DTYPE))
    assert (q - 0.1 * force).item() > 0


def test_common_zero_is_preserved_for_arbitrary_factors():
    q = torch.tensor([1.0, 1.0, 1.0], dtype=DTYPE, requires_grad=True)
    c = residual_pair(q)
    factors = torch.randn((3, 2, 4), dtype=DTYPE, generator=torch.Generator().manual_seed(3))
    force = residual_mixing_force(q, c, factors)
    torch.testing.assert_close(force, torch.zeros_like(q), atol=0, rtol=0)


def test_zero_jacobian_cannot_move_a_nonzero_residual():
    q = torch.tensor(0.0, dtype=DTYPE, requires_grad=True)
    c = torch.stack((q.square() + 1, 2 * q.square() - 3))
    factors = torch.tensor([[2.0], [-1.0]], dtype=DTYPE)
    force = residual_mixing_force(q, c, factors)
    assert force.item() == 0


def test_batched_mix_matches_dense_matrices_and_spectral_bounds():
    generator = torch.Generator().manual_seed(9)
    c = torch.randn((2, 3, 5), generator=generator, dtype=DTYPE)
    u = torch.randn((2, 3, 5, 2), generator=generator, dtype=DTYPE)
    m0, b = 0.3, 1.2
    w = m0 * torch.eye(5, dtype=DTYPE) + b * (u @ u.transpose(-1, -2)) / (
        1 + u.square().sum(dim=(-2, -1))
    )[..., None, None]
    torch.testing.assert_close(bounded_residual_mix(c, u, m0=m0, b=b), (w @ c[..., None]).squeeze(-1))
    eigenvalues = torch.linalg.eigvalsh(w)
    assert eigenvalues.min() >= m0 - 1e-12
    assert eigenvalues.max() <= m0 + b + 1e-12


def test_forward_vjp_omits_metric_derivative_but_training_keeps_it():
    def loss(a, u, *, graph):
        q = torch.tensor([0.2, -0.4], dtype=DTYPE, requires_grad=True)
        c = torch.stack((a * q.square() - 1, 2 * q * (q - 1)), dim=-1)
        factors = torch.stack((u + q, 0.3 - q), dim=-1).unsqueeze(-1)
        force = residual_mixing_force(q, c, factors, create_graph=graph)
        # Reference J^T(Wc), with no derivative of W in the forward expression.
        mixed = bounded_residual_mix(c, factors)
        reference = (2 * a * q) * mixed[..., 0] + (4 * q - 2) * mixed[..., 1]
        torch.testing.assert_close(force, reference)
        return (q - 0.15 * force).square().sum()

    a = torch.tensor(0.7, dtype=DTYPE, requires_grad=True)
    u = torch.tensor(0.9, dtype=DTYPE, requires_grad=True)
    da, du = torch.autograd.grad(loss(a, u, graph=True), (a, u))
    eps = 1e-6
    fd_a = (loss(a.detach() + eps, u.detach(), graph=False) - loss(a.detach() - eps, u.detach(), graph=False)) / (2 * eps)
    fd_u = (loss(a.detach(), u.detach() + eps, graph=False) - loss(a.detach(), u.detach() - eps, graph=False)) / (2 * eps)
    torch.testing.assert_close(da, fd_a, atol=1e-8, rtol=1e-6)
    torch.testing.assert_close(du, fd_u, atol=1e-8, rtol=1e-6)
    assert abs(du.item()) > 1e-5  # A detached cotangent would fail this check.


def test_shape_and_bound_errors_are_explicit():
    c = torch.zeros((3, 2), dtype=DTYPE)
    with pytest.raises(ValueError, match="Expected|batch"):
        bounded_residual_mix(c, torch.zeros((2, 1), dtype=DTYPE))
    with pytest.raises(ValueError, match="m0"):
        bounded_residual_mix(c, torch.zeros((3, 2, 1), dtype=DTYPE), m0=0)


def test_disconnected_residual_is_rejected():
    state = torch.tensor([0.3], dtype=DTYPE, requires_grad=True)
    different_state = torch.tensor([0.3], dtype=DTYPE, requires_grad=True)
    residual = different_state.square() - 1
    factors = torch.ones((1, 1), dtype=DTYPE)
    with pytest.raises(ValueError, match="disconnected"):
        residual_mixing_force(state, residual, factors)


def test_graph_constant_residual_has_zero_force_and_outer_gradients():
    state = torch.tensor([0.3], dtype=DTYPE, requires_grad=True)
    parameter = torch.tensor(2.0, dtype=DTYPE, requires_grad=True)
    residual = state * 0 + parameter
    factors = parameter.reshape(1, 1)
    force = residual_mixing_force(state, residual, factors, create_graph=True)
    torch.testing.assert_close(force, torch.zeros_like(state), atol=0, rtol=0)
    state_gradient, parameter_gradient = torch.autograd.grad(force.sum(), (state, parameter))
    torch.testing.assert_close(state_gradient, torch.zeros_like(state), atol=0, rtol=0)
    torch.testing.assert_close(parameter_gradient, torch.zeros_like(parameter), atol=0, rtol=0)
