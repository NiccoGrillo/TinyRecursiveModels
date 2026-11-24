"""Prepare unambiguous, fixed-length supervised examples. No verifier enters inference."""

import argparse
import csv
import hashlib
import heapq
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from .common import digest, fingerprint, read, write


def unique_sudoku(question, answer):
    """Verify the supplied solution, then search for any distinct completion (MRV)."""
    board = list(question)
    groups = (
        [[9 * r + c for c in range(9)] for r in range(9)]
        + [[9 * r + c for r in range(9)] for c in range(9)]
        + [
            [9 * (r + i) + c + j for i in range(3) for j in range(3)]
            for r in (0, 3, 6)
            for c in (0, 3, 6)
        ]
    )
    if any({answer[i] for i in group} != set("123456789") for group in groups):
        raise ValueError("Invalid stored solution")
    if any(v != "0" and v != answer[i] for i, v in enumerate(board)):
        raise ValueError("Given disagrees with solution")
    # Bitmasks avoid allocating sets at each branch in the uniqueness proof.
    masks = [0] * 27
    units = [(i // 9, 9 + i % 9, 18 + i // 27 * 3 + i % 9 // 3) for i in range(81)]
    empty = []
    for i, v in enumerate(board):
        if v == "0":
            empty.append(i)
        else:
            for u in units[i]:
                masks[u] |= 1 << int(v)

    def alternative(depth, changed):
        if depth == len(empty):
            return changed
        best, options, count = depth, 0, 10
        for j in range(depth, len(empty)):
            r, c, b = units[empty[j]]
            available = 1022 & ~(masks[r] | masks[c] | masks[b])
            n = available.bit_count()
            if n < count:
                best, options, count = j, available, n
                if n <= 1:
                    break
        empty[depth], empty[best] = empty[best], empty[depth]
        i = empty[depth]
        while options:
            bit = options & -options
            options ^= bit
            for u in units[i]:
                masks[u] ^= bit
            found = alternative(depth + 1, changed or bit != 1 << int(answer[i]))
            for u in units[i]:
                masks[u] ^= bit
            if found:
                empty[depth], empty[best] = empty[best], empty[depth]
                return True
        empty[depth], empty[best] = empty[best], empty[depth]
        return False

    return not alternative(0, False)


def select_csv(path, count, seed, exclude=()):
    """Exact bottom-k of distinct original input hashes, independent of CSV ordering."""
    heap, selected = [], {}
    exclude = set(exclude)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            q, a = row["question"].replace(".", "0"), row["answer"]
            if (
                len(q) != 81
                or len(a) != 81
                or set(q) - set("0123456789")
                or set(a) - set("123456789")
            ):
                raise ValueError(f"Malformed example in {path}")
            if q in exclude:
                continue
            if q in selected:
                if selected[q] != a:
                    raise ValueError("Conflicting answers to selected input")
                continue
            priority = int.from_bytes(
                hashlib.sha256(f"{seed}/{q}".encode()).digest(), "big"
            )
            if len(heap) == count and priority >= -heap[0][0]:
                continue
            if len(heap) == count:
                _, removed = heapq.heappop(heap)
                del selected[removed]
            heapq.heappush(heap, (-priority, q))
            selected[q] = a
    if len(heap) < count:
        raise ValueError(f"Need {count} distinct inputs in {path}; found {len(heap)}")
    rows = [(q, selected[q]) for _, q in sorted(heap, reverse=True)]
    if any(not unique_sudoku(q, a) for q, a in rows):
        raise ValueError(
            "Selected data contain an ambiguous puzzle. Use a verified-unique source."
        )
    return rows


def prepare(out, task, counts, seed, length, vocab, train_csv=None, test_csv=None):
    out = Path(out)
    if out.exists() or min(counts) <= 0 or min(length, vocab) < 2 or vocab > 255:
        raise ValueError(
            "Need a fresh directory, positive counts, length >= 2 and 2..255 symbols"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    rng = np.random.default_rng(seed)
    splits, sources = {}, {}
    try:
        if task == "sudoku":
            if train_csv is None or test_csv is None:
                raise ValueError("Supply both original train and test CSVs")
            rows = select_csv(train_csv, sum(counts[:2]), seed)
            held = select_csv(test_csv, counts[2], seed, (q for q, _ in rows))
            for k, part in zip(
                ("train", "dev", "test"), (rows[: counts[0]], rows[counts[0] :], held)
            ):
                x = np.array([[int(t) for t in q] for q, a in part], np.uint8)
                y = np.array([[int(t) - 1 for t in a] for q, a in part], np.uint8)
                splits[k] = (x, y, x == 0)
            vin, vout = 10, 9
            sources = {"train.csv": digest(train_csv), "test.csv": digest(test_csv)}
        else:
            vin = vout = vocab
            if sum(counts) > vocab**length:
                raise ValueError(
                    "Requested more distinct inputs than the sequence domain"
                )
            seen = set()
            for k, n, size in zip(
                ("train", "dev", "test", "long"),
                (*counts, counts[2]),
                (length, length, length, 2 * length),
            ):
                x = np.empty((n, size), np.uint8)
                i = 0
                while i < n:
                    row = rng.integers(vocab, size=size, dtype=np.uint8)
                    if row.tobytes() not in seen:
                        x[i] = row
                        seen.add(row.tobytes())
                        i += 1
                y = (np.cumsum(x, axis=1) % vocab).astype(np.uint8)
                splits[k] = (x, y, np.ones_like(x, dtype=bool))
        meta = dict(
            schema=1,
            task=task,
            input_vocab=vin,
            output_vocab=vout,
            seed=seed,
            sources=sources,
            splits={},
        )
        for k, (x, y, editable) in splits.items():
            ids = np.array([hashlib.sha256(row.tobytes()).hexdigest() for row in x])
            path = tmp / f"{k}.npz"
            np.savez(path, x=x, y=y, editable=editable, ids=ids)
            meta["splits"][k] = dict(
                count=len(x), length=x.shape[1], sha256=digest(path)
            )
        write(tmp / "dataset.json", meta)
        os.rename(tmp, out)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)


class Dataset:
    def __init__(self, path, split):
        self.path = Path(path)
        self.meta = read(self.path / "dataset.json")
        info = self.meta["splits"][split]
        path = self.path / f"{split}.npz"
        if digest(path) != info["sha256"]:
            raise ValueError(f"Dataset checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as z:
            self.x, self.y = torch.from_numpy(z["x"]), torch.from_numpy(z["y"])
            self.editable, self.ids = torch.from_numpy(z["editable"]), z["ids"]
        self.fingerprint = fingerprint(self.meta)

    def batch(self, indices, device):
        return (
            self.x[indices].to(device, torch.long),
            self.y[indices].to(device, torch.long),
            self.editable[indices].to(device),
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=["sudoku", "scan"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--train-csv", type=Path)
    p.add_argument("--test-csv", type=Path)
    p.add_argument(
        "--counts",
        type=int,
        nargs=3,
        default=[8192, 512, 1024],
        metavar=("TRAIN", "DEV", "TEST"),
    )
    p.add_argument("--length", type=int, default=16)
    p.add_argument("--vocab", type=int, default=8)
    p.add_argument("--seed", type=int, default=2809)
    a = p.parse_args()
    prepare(a.out, a.task, a.counts, a.seed, a.length, a.vocab, a.train_csv, a.test_csv)


if __name__ == "__main__":
    main()
