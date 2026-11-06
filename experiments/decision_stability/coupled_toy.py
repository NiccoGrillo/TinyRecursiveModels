"""Analytic shared-additive-core counterexample; NOT a trained TRM experiment.

The update wiring matches TRM's additive injection and shared core. The scalar
cubic is not a post-normalized Transformer weight construction. See
docs/RESEARCH_DIRECTIONS.md for proofs, limitations, and empirical gates.
"""
import json

import numpy as np


def shared_core(value):
    return value * (value - 1) * (0.7 * value + 0.2)


def outer(state, low_cycles, high_scale=1.0):
    if low_cycles < 1 or not 0 < high_scale <= 1:
        raise ValueError("Use positive cycles and 0 < high_scale <= 1")
    high, low = np.asarray(state, dtype=np.float64)
    for _ in range(low_cycles):
        low = shared_core(low + high + 1.0)
    high = high + high_scale * (shared_core(high + low) - high)
    return np.array([high, low])


def jacobian(low_cycles, high_scale=1.0):
    """Exact derivative at the common origin; order is (high, low)."""
    q = 0.9**low_cycles
    coupling = 9.0 * (1 - q)
    return np.array([[1 - high_scale - 0.2 * high_scale * (1 + coupling),
                      -0.2 * high_scale * q], [coupling, q]])


def period_jacobian(schedule):
    result = np.eye(2)
    for low_cycles in schedule:
        result = jacobian(low_cycles) @ result
    return result


def describe(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    return dict(jacobian=matrix.tolist(),
                eigenvalues=[dict(real=float(v.real), imag=float(v.imag)) for v in eigenvalues],
                spectral_radius=float(max(abs(eigenvalues))),
                largest_singular_value=float(np.linalg.norm(matrix, 2)))


def report():
    q = 0.9**12
    return dict(
        evidence_kind="analytic_counterexample_not_trained_model_evidence",
        shared_core="G(v)=v(v-1)(0.7v+0.2); low=G(low+high+1), high=G(high+low)",
        origin_common_equilibrium=True,
        conditional_derivatives=dict(low=0.9, high=-0.2),
        outer_maps=[dict(low_cycles=l, high_scale=eta, trm_equivalent_block_evals=2 * (l + 1),
                         **describe(jacobian(l, eta))) for l, eta in ((6, 1.0), (12, 1.0), (12, 0.8))],
        equal_count_periods=[dict(schedule=list(s), low_updates=sum(s), high_updates=len(s),
                                 trm_equivalent_block_evals=2 * (sum(s) + len(s)), **describe(period_jacobian(s)))
                            for s in ((6, 6), (3, 9), (9, 3))],
        undamped_stable_condition="0.9**L > 5/13; integer L=1,...,9",
        high_scale_stability_upper_at_l12=2 * (1 + q) / (3 - 0.6 * q),
        caveats=["Block-equivalent costs price each scalar core call as two TRM layers; they are not measured toy FLOPs.",
                 "Local latent stability is not decoded-answer retention or task correctness.",
                 "An exact common equilibrium stays fixed even when unstable.",
                 "Reversed two-phase schedules have the same eigenvalues, not the same transient gain.",
                 "Damping adds tensor arithmetic; recurrent-block counts alone are not full runtime."])


if __name__ == "__main__":
    print(json.dumps(report(), indent=2, allow_nan=False))
