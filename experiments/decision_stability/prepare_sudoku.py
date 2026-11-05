"""Prepare only the unaugmented evaluation CSV, without downloading training data."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .model import sha256


def prepare(source, out, source_url):
    source, out = Path(source), Path(out)
    if out.exists():
        raise FileExistsError(out)
    inputs, labels = [], []
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["source", "question", "answer", "rating"]:
            raise ValueError("Expected upstream Sudoku CSV columns")
        for row in reader:
            question, answer = row["question"].replace(".", "0"), row["answer"]
            if (len(question) != 81 or len(answer) != 81 or
                    set(question) - set("0123456789") or set(answer) - set("123456789")):
                raise ValueError(f"Invalid Sudoku tokens at CSV line {reader.line_num}")
            inputs.append(np.frombuffer(question.encode(), dtype=np.uint8) - ord("0") + 1)
            labels.append(np.frombuffer(answer.encode(), dtype=np.uint8) - ord("0") + 1)
    if not inputs:
        raise ValueError("Empty Sudoku CSV")
    n = len(inputs)
    arrays = dict(inputs=np.stack(inputs), labels=np.stack(labels),
                  puzzle_identifiers=np.zeros(n, dtype=np.int32),
                  puzzle_indices=np.arange(n + 1, dtype=np.int32),
                  group_indices=np.arange(n + 1, dtype=np.int32))
    # Same token values, ordering, and unaugmented groups as the upstream builder.
    # The diagnostic reader additionally validates selected solutions and givens.
    folder = out / "test"
    folder.mkdir(parents=True)
    metadata = dict(seq_len=81, vocab_size=11, pad_id=0, ignore_label_id=0,
                    blank_identifier_id=0, num_puzzle_identifiers=1, total_groups=n,
                    mean_puzzle_examples=1, total_puzzles=n, sets=["all"])
    (folder / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for name, values in arrays.items():
        np.save(folder / f"all__{name}.npy", values)
    provenance = dict(source_url=source_url, csv_sha256=sha256(source), examples=n,
                      transformation="CSV row order; blank=1, digits=2..10; no augmentation or filtering")
    (out / "preparation.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-url", required=True, help="Use a revision-pinned source URL")
    args = parser.parse_args()
    print(json.dumps(prepare(args.csv, args.out, args.source_url)))


if __name__ == "__main__":
    main()
