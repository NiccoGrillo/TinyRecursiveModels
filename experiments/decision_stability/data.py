"""Strict reader for unaugmented Sudoku test arrays produced by upstream."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

from .model import sha256


@dataclass
class SudokuData:
    inputs: np.ndarray
    labels: np.ndarray
    identifiers: np.ndarray
    sample_ids: list
    metadata: dict
    provenance: dict

    def batches(self, size, device):
        for start in range(0, len(self.inputs), size):
            stop = start + size
            batch = {"inputs": torch.tensor(self.inputs[start:stop], dtype=torch.long, device=device),
                     "labels": torch.tensor(self.labels[start:stop], dtype=torch.long, device=device),
                     "puzzle_identifiers": torch.tensor(self.identifiers[start:stop], dtype=torch.long, device=device)}
            yield self.sample_ids[start:stop], batch


def sudoku_valid(predictions, inputs):
    """All 27 units, legal digits, AND original givens; no reference solutions used."""
    grid = predictions.reshape(-1, 9, 9)
    digits = torch.arange(2, 11, device=grid.device)
    rows = grid.sort(-1).values.eq(digits).all(-1).all(-1)
    cols = grid.transpose(1, 2).sort(-1).values.eq(digits).all(-1).all(-1)
    boxes = grid.reshape(-1, 3, 3, 3, 3).permute(0, 1, 3, 2, 4).reshape(-1, 9, 9)
    boxes_ok = boxes.sort(-1).values.eq(digits).all(-1).all(-1)
    givens = ((inputs == 1) | (predictions == inputs)).all(-1)
    return rows & cols & boxes_ok & givens


def load_sudoku(root, partition="dev", seed=0, limit=128):
    root = Path(root)
    folder = root / "test"
    files = {"metadata": folder / "dataset.json"}
    names = ("inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices")
    files.update({k: folder / f"all__{k}.npy" for k in names})
    metadata = json.loads(files["metadata"].read_text())
    if any(metadata[k] != v for k, v in {"seq_len": 81, "vocab_size": 11, "pad_id": 0}.items()):
        raise ValueError("This first backend supports upstream 9x9 Sudoku tokenization only")
    arrays = {k: np.load(files[k], mmap_mode="r", allow_pickle=False) for k in names}
    if any(array.dtype.kind not in "iu" for array in arrays.values()):
        raise ValueError("Prepared tokens, identifiers, and indices must have integer dtype")
    n = len(arrays["inputs"])
    if arrays["inputs"].shape != (n, 81) or arrays["labels"].shape != (n, 81):
        raise ValueError("Expected arrays with shape [N, 81]")
    if arrays["puzzle_identifiers"].shape != (n,):
        raise ValueError("Expected one identifier per unaugmented example")
    for key in ("puzzle_indices", "group_indices"):
        if not np.array_equal(arrays[key], np.arange(n + 1)):
            raise ValueError("Use the unaugmented test set, with one example per original group")
    # Deduplicate exact input puzzles before splitting. Augmentation-equivalent
    # duplicates are NOT canonicalized; the upstream grouping is still required.
    unique = {}
    for i, row in enumerate(arrays["inputs"]):
        key = hashlib.sha256(np.asarray(row, dtype="<i8").tobytes()).hexdigest()
        if key in unique:
            if not np.array_equal(arrays["labels"][i], arrays["labels"][unique[key]]):
                raise ValueError("Duplicate puzzle with conflicting reference solutions")
        else:
            unique[key] = i
    ordered = sorted(unique, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
    cut = len(ordered) // 2
    pool = ordered[:cut] if partition == "dev" else ordered[cut:]
    if partition not in ("dev", "test") or not 0 < limit <= len(pool):
        raise ValueError(f"Invalid partition/limit: {partition}, {limit}; available {len(pool)}")
    keys = pool[:limit]
    indices = [unique[key] for key in keys]
    inp = np.asarray(arrays["inputs"][indices], dtype=np.int64)
    lab = np.asarray(arrays["labels"][indices], dtype=np.int64)
    ids = np.asarray(arrays["puzzle_identifiers"][indices], dtype=np.int64)
    if not ((inp >= 1) & (inp <= 10)).all() or not ((lab >= 2) & (lab <= 10)).all():
        raise ValueError("Expected blank=1, digits=2..10, complete unpadded Sudoku labels")
    if not ((ids >= 0) & (ids < metadata["num_puzzle_identifiers"])).all():
        raise ValueError("Puzzle identifier out of range")
    if not sudoku_valid(torch.tensor(lab), torch.tensor(inp)).all():
        raise ValueError("Reference Sudoku violates rules or givens")
    return SudokuData(inp, lab, ids, keys, metadata, dict(
        source="upstream_prepared_test", partition=partition, split_seed=seed,
        selected_indices=indices, selected_ids=keys, original_count=n,
        unique_count=len(unique), partition_available=len(pool),
        source_sha256={k: sha256(v) for k, v in files.items()}))


def smoke_data():
    """Two valid completed grids with blanks; synthetic, not a reasoning benchmark."""
    grid = np.array([[(r * 3 + r // 3 + c) % 9 + 2 for c in range(9)] for r in range(9)])
    labels = np.stack([grid.reshape(-1), ((grid - 2 + 1) % 9 + 2).reshape(-1)])
    inputs = labels.copy()
    inputs[:, ::2] = 1
    return SudokuData(inputs, labels, np.zeros(2, dtype=np.int64), ["smoke-0", "smoke-1"],
                      dict(seq_len=81, vocab_size=11, num_puzzle_identifiers=1),
                      dict(source="synthetic_smoke_only", partition="smoke"))
