"""Export one Sudoku cohort and one augmentation stream for both trainers."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from build_sudoku_dataset import shuffle_sudoku
from experiments.residual_reasoning.common import digest, read, write
from experiments.residual_reasoning.data import Dataset


def original_split(root, split, limit=None):
    data = Dataset(root, split)
    return data.x.numpy()[:limit], data.y.numpy()[:limit], data.ids[:limit]


def augmented_rows(x, y, repeats, seed):
    np.random.seed(seed)
    width = 1 + repeats
    xs = np.empty((len(x) * width, 81), np.uint8)
    ys = np.empty_like(xs)
    xs[::width], ys[::width] = x, y
    for index, (board, labels) in enumerate(zip(x, y)):
        for variant in range(1, width):
            tokens, solutions = shuffle_sudoku(board.reshape(9, 9), (labels + 1).reshape(9, 9))
            xs[index * width + variant] = tokens.reshape(-1)
            ys[index * width + variant] = solutions.reshape(-1) - 1
    return xs, ys


def encode_split(root, split, x, y, ids, width):
    native_name = {"train": "train", "dev": "test", "test": "heldout"}[split]
    native = root / "native" / native_name
    native.mkdir(parents=True)
    flat = root / "residual" / f"{split}.npz"
    np.savez(flat, x=x, y=y, editable=x == 0, ids=ids)
    count = len(x)
    arrays = dict(inputs=x + 1, labels=y + 2, puzzle_identifiers=np.zeros(count, np.int32),
                  puzzle_indices=np.arange(count + 1, dtype=np.int32),
                  group_indices=np.arange(0, count + 1, width, dtype=np.int32))
    for name, array in arrays.items():
        np.save(native / f"all__{name}.npy", array)
    write(native / "dataset.json", dict(seq_len=81, vocab_size=11, pad_id=0, ignore_label_id=0,
          blank_identifier_id=0, num_puzzle_identifiers=1, total_groups=count // width,
          mean_puzzle_examples=1, total_puzzles=count, sets=["all"]))
    np.testing.assert_array_equal(np.load(native / "all__inputs.npy") - 1, x)
    np.testing.assert_array_equal(np.load(native / "all__labels.npy") - 2, y)
    return dict(count=count, length=81, sha256=digest(flat))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--augments", type=int, default=32)
    args = p.parse_args()
    source = read(args.source / "dataset.json")
    train_x, train_y, originals = original_split(args.source, "train", 1000)
    train_x, train_y = augmented_rows(train_x, train_y, args.augments, 11)
    augmented_ids = np.array([hashlib.sha256(row.tobytes()).hexdigest() for row in train_x])
    dev_x, dev_y, dev_ids = original_split(args.source, "dev")
    test_x, test_y, test_ids = original_split(args.source, "test")
    if (len(set(augmented_ids)) != len(augmented_ids)
        or set(augmented_ids) & (set(dev_ids) | set(test_ids))
        or set(dev_ids) & set(test_ids)):
        raise ValueError("Expected unique inputs and exact disjointness between splits")
    assert np.all((train_x == 0) | (train_x == train_y + 1))
    args.root.mkdir()
    (args.root / "residual").mkdir()
    encoding = {
        "train": encode_split(args.root, "train", train_x, train_y, augmented_ids, args.augments + 1),
        "dev": encode_split(args.root, "dev", dev_x, dev_y, dev_ids, 1),
        "test": encode_split(args.root, "test", test_x, test_y, test_ids, 1),
    }
    write(args.root / "residual" / "dataset.json", dict(schema=1, task="sudoku",
          input_vocab=10, output_vocab=9, seed=11, sources=source["sources"], splits=encoding))
    write(args.root / "native" / "identifiers.json", ["<blank>"])
    write(args.root / "cohort.json", dict(original_ids=originals.tolist(), augments=args.augments,
          augmentation_source="pinned upstream shuffle_sudoku", augmentation_seed=11,
          NumPy=np.__version__, source_manifest=digest(args.source / "dataset.json"),
          dev_ids=dev_ids.tolist(), heldout_ids=test_ids.tolist(), splits=encoding))
    print({name: values["count"] for name, values in encoding.items()}, flush=True)


if __name__ == "__main__":
    main()
