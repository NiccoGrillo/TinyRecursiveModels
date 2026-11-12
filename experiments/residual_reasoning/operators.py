"""Experimental residual-space mixing operators, not a trained reasoner.

For residual c(q) and bounded factors U, the semantic force is J_c(q)^T W c,
where W = m0 I + b U U^T / (1 + ||U||_F^2). This is a VJP with a
state-dependent cotangent, not the total derivative of c^T W(q) c / 2.

Leading dimensions index independent metric blocks. A residual network may in
principle couple those blocks: autograd then returns the corresponding full
VJP, not a silently block-diagonal approximation of its Jacobian.
"""
from __future__ import annotations

import math
from numbers import Real

import torch
from torch import Tensor


def _check_inputs(residual: Tensor, factors: Tensor, m0: float, b: float) -> None:
    if residual.ndim < 1 or factors.ndim != residual.ndim + 1:
        raise ValueError("Expected residual [..., r] and factors [..., r, rank].")
    if factors.shape[:-2] != residual.shape[:-1] or factors.shape[-2] != residual.shape[-1]:
        raise ValueError("Residual and factors must have identical batch and residual dimensions.")
    if residual.shape[-1] == 0 or factors.shape[-1] == 0:
        raise ValueError("Residual dimension and factor rank must be positive.")
    if not residual.is_floating_point() or not factors.is_floating_point():
        raise TypeError("Residual and factors must be real floating-point tensors.")
    if residual.dtype != factors.dtype or residual.device != factors.device:
        raise ValueError("Residual and factors must share dtype and device.")
    if not isinstance(m0, Real) or not isinstance(b, Real):
        raise TypeError("m0 and b must be fixed real scalar hyperparameters.")
    if not math.isfinite(m0) or m0 <= 0 or not math.isfinite(b) or b < 0:
        raise ValueError("Require finite fixed m0 > 0 and b >= 0.")


def bounded_residual_mix(
    residual: Tensor, factors: Tensor, *, m0: float = 1.0, b: float = 1.0
) -> Tensor:
    """Apply W without materializing it; its eigenvalues lie in [m0, m0+b].

    ``m0`` and ``b`` are fixed scalar hyperparameters, not unconstrained learned
    scales. The Frobenius normalization is per leading-index metric block.
    Extreme inputs can overflow in low precision; this operator is not a
    finite-precision certificate. Validate practical dtype/range separately.
    """
    _check_inputs(residual, factors, m0, b)
    projected = torch.matmul(factors.transpose(-1, -2), residual.unsqueeze(-1))
    correction = torch.matmul(factors, projected).squeeze(-1)
    denominator = 1.0 + factors.square().sum(dim=(-2, -1))
    return m0 * residual + b * correction / denominator.unsqueeze(-1)


def residual_mixing_force(
    state: Tensor,
    residual: Tensor,
    factors: Tensor,
    *,
    m0: float = 1.0,
    b: float = 1.0,
    create_graph: bool = False,
) -> Tensor:
    """Return J_residual(state)^T W residual using an already-built graph.

    ``state`` must require gradients and ``residual`` must have been computed
    from that same tensor. ``factors`` may depend on state, residual, parameters
    and a workspace. Set ``create_graph=True`` during unrolled training: the
    cotangent remains differentiable, so outer derivatives include dependence
    of W and c while the *forward force* omits the spurious derivative of W.

    No update, energy decrease, controller policy or correctness is implied.
    A constant residual map represented on the graph (e.g. state * 0 + c0)
    returns zero, illustrating the J=0 limitation. Detached residual tensors
    are rejected because they can silently turn a real derivative into zero.
    """
    if not state.requires_grad:
        raise ValueError("state must require gradients before computing residual.")
    if state.device != residual.device or state.dtype != residual.dtype:
        raise ValueError("State and residual must share dtype and device.")
    cotangent = bounded_residual_mix(residual, factors, m0=m0, b=b)
    if not residual.requires_grad:
        raise ValueError("residual must be computed with autograd enabled; do not detach it.")
    force = torch.autograd.grad(
        residual,
        state,
        grad_outputs=cotangent,
        create_graph=create_graph,
        retain_graph=create_graph,
        allow_unused=True,
    )[0]
    if force is None:
        raise ValueError("residual is disconnected from state; compute it from the supplied state tensor.")
    if create_graph and not force.requires_grad:
        # Keep a zero-valued path so an outer loss can differentiate a constant
        # force without inventing a nonzero derivative.
        force = force + 0.0 * state + 0.0 * cotangent.sum()
    return force
