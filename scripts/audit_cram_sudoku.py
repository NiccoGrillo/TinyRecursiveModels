"""Certify discrete Sudoku energy bounds from the W and B tensors in CRAM checkpoints."""

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch


def audit(weights, biases):
    # CRAM stores one 9x9 table per unordered cell pair, in torch.triu_indices order.
    left, right = np.triu_indices(81, k=1)
    if weights.shape != (len(left), 9, 9) or biases.shape != (81, 9):
        raise ValueError("Expected CRAM's uncompressed Sudoku W and B tensors")
    if not np.isfinite(weights).all() or not np.isfinite(biases).all():
        raise ValueError("Weights and biases must be finite")
    # All checkpoint floats are dyadic rationals. Scale to integers before any sums.
    unique = np.unique(np.concatenate((weights.ravel(), biases.ravel())))
    unit = max(float(value).as_integer_ratio()[1] for value in unique)
    largest_weight = max(abs(Fraction(float(value))) for value in unique)
    if 8 * 81**2 * largest_weight * unit >= 2**63:
        raise ValueError("Checkpoint requires wider integer arithmetic")
    weights = (weights * unit).astype(np.int64)
    biases = (biases * unit).astype(np.int64)
    same_box = (left // 27 == right // 27) & (left % 9 // 3 == right % 9 // 3)
    peers = (left // 9 == right // 9) | (left % 9 == right % 9) | same_box
    equal_digit = np.eye(9, dtype=bool)
    peer_terms = peers[:, None, None] & equal_digit
    min_peer_cost = int(weights[peer_terms].min())
    if min_peer_cost <= 0:
        raise ValueError("The peer-penalty proof requires positive costs on all peer diagonals")

    remainder = weights.copy()
    remainder[peer_terms] = 0
    tables = np.zeros((81, 81, 9, 9), dtype=np.int64)
    tables[left, right] = remainder
    tables[right, left] = remainder.transpose(0, 2, 1)
    sensitivity = []
    for cell, incident in enumerate(tables):
        # Bound each neighbour's contribution for fixed old and new digits.
        changes = incident[:, :, None, :] - incident[:, None, :, :]
        edge_bound = np.abs(changes).max(axis=-1).sum(axis=0)
        bias_bound = np.abs(biases[cell, :, None] - biases[cell, None, :])
        sensitivity.append(int((bias_bound + edge_bound).max()))
    largest, second = sorted(sensitivity, reverse=True)[:2]
    diagonal = weights.diagonal(axis1=1, axis2=2)
    # Share each pair cost equally between its incident cells. At cell i, fix
    # its digit a, relax all neighbour digits independently, then optimize a.
    twice_lower = int((2 * biases + tables.min(axis=3).sum(axis=1)).min(axis=1).sum())
    twice_upper = int((2 * biases + tables.max(axis=3).sum(axis=1)).max(axis=1).sum())
    invalid_lower = Fraction(4 * min_peer_cost + twice_lower, 2 * unit)
    valid_upper = Fraction(twice_upper, 2 * unit)
    return {
        "positive_equal_digit_pairs": int((diagonal > 0).any(axis=1).sum()),
        "negative_off_peer_equal_digit_entries": int((diagonal[~peers] < 0).sum()),
        "nonzero_unequal_digit_entries": int(np.count_nonzero(weights[:, ~equal_digit])),
        "nonzero_biases": int(np.count_nonzero(biases)),
        "min_peer_cost": min_peer_cost / unit,
        "largest_residual_sensitivity": largest / unit,
        "single_replacement_lower_bound": (2 * min_peer_cost - largest) / unit,
        "distinct_digit_swap_lower_bound": (2 * min_peer_cost - largest - second) / unit,
        "all_valid_upper_exact": str(valid_upper),
        "all_invalid_lower_exact": str(invalid_lower),
        "all_invalid_lower": float(invalid_lower),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+", help="Sudoku seed model.pt paths")
    args = parser.parse_args()
    for path in args.checkpoints:
        state = torch.load(path, map_location="cpu", weights_only=True)
        result = audit(state["W"].double().numpy(), state["B"].double().numpy())
        print(json.dumps({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **result}))


if __name__ == "__main__":
    main()
