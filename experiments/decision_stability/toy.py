"""Stable eigenvalues can coexist with temporary decision loss (illustration only)."""
import json
import numpy as np


def trajectory(steps=20):
    matrix = np.array([[0.8, 5.0], [0.0, 0.9]])
    state = np.array([0.0, -0.1])
    rows = []
    for step in range(steps + 1):
        rows.append(dict(step=step, error=float(state[0]), hidden=float(state[1]),
                         margin=float(1 + state[0]), correct=bool(1 + state[0] > 0)))
        state = matrix @ state
    return rows


if __name__ == "__main__":
    print(json.dumps(dict(kind="analytic_illustration_not_trained_evidence", eigenvalues=[0.8, 0.9],
                          trajectory=trajectory()), indent=2))
