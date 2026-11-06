import numpy as np
import pytest

from experiments.decision_stability.coupled_toy import jacobian, outer, period_jacobian


@pytest.mark.parametrize("cycles,scale", [(6, 1.0), (12, 1.0), (12, 0.8)])
def test_derived_jacobian_matches_direct_shared_core_recurrence(cycles, scale):
    epsilon = 1e-6
    numerical = np.column_stack([
        (outer(epsilon * e, cycles, scale) - outer(-epsilon * e, cycles, scale)) / (2 * epsilon)
        for e in np.eye(2)])
    assert np.allclose(jacobian(cycles, scale), numerical, rtol=1e-6, atol=1e-7)
    assert np.array_equal(outer(np.zeros(2), cycles, scale), np.zeros(2))


def test_coupled_stability_rescue_and_equal_count_phase_distinction():
    rho = lambda matrix: max(abs(np.linalg.eigvals(matrix)))
    assert rho(jacobian(6)) < 1 < rho(jacobian(12))
    assert rho(jacobian(12, 0.8)) < 1
    native = period_jacobian((6, 6))
    forward = period_jacobian((3, 9))
    reverse = period_jacobian((9, 3))
    assert rho(forward) == pytest.approx(rho(reverse))
    assert rho(forward) < rho(native)
    # Same asymptotic eigenvalues do not imply the same finite-state effect.
    assert np.linalg.norm(forward, 2) > np.linalg.norm(native, 2) > np.linalg.norm(reverse, 2)
