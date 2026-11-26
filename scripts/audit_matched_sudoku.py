"""Audit the saved residual Sudoku checkpoint without additional training."""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from evaluate_matched_sudoku import model_from_checkpoint, read_boards, score_residual, sha


def verify_original_labels(inputs, labels, csv_path):
    givens = ["".join(map(str, board)) for board in inputs.tolist()]
    solutions = {board: None for board in givens}
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            board = row["question"].replace(".", "0")
            if board in solutions:
                solutions[board] = row["answer"]
    for board, tokens in zip(givens, labels.tolist()):
        if solutions[board] != "".join(str(token + 1) for token in tokens):
            raise ValueError(f"Original solution differs for {board}")


def replay(model, inputs, labels):
    by_batch = []
    for start in range(0, len(inputs), 16):
        x, y = inputs[start : start + 16], labels[start : start + 16]
        with torch.no_grad():
            encoded = model.encode(x)
            answer, workspace = model.start(encoded)
        outcomes = [score_residual(model, answer, y)]
        for _ in range(16):
            answer, workspace = model.step(encoded, answer, workspace, train=False)
            outcomes.append(score_residual(model, answer, y))
        by_batch.append(torch.stack(outcomes, dim=1))
    return torch.cat(by_batch)


def summarize(correct, blank):
    wrong_blanks = (blank & ~correct).sum(dim=1).float()
    return dict(
        exact=int(correct.all(1).sum()),
        n=len(correct),
        clue_accuracy=float(correct[~blank].double().mean()),
        blank_accuracy=float(correct[blank].double().mean()),
        median_wrong_blanks=float(torch.quantile(wrong_blanks, 0.5)),
        wrong_blank_quartiles=torch.quantile(wrong_blanks, torch.tensor([0.25, 0.75])).tolist(),
        at_most_one_wrong_blank=int((wrong_blanks <= 1).sum()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "cohort", "weights", "evaluation", "original-test-csv", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    cohort = json.loads(args.cohort.read_text())
    archived = json.loads(args.evaluation.read_text())
    if archived["checkpoint_sha256"] != sha(args.weights):
        raise ValueError("Checkpoint differs from the original GPU run")
    device = torch.device("cpu")
    model, _ = model_from_checkpoint(SimpleNamespace(family="residual", weights=args.weights), device)
    selections = {"train": slice(0, 128 * (cohort["augments"] + 1), cohort["augments"] + 1),
                  "test": slice(128)}
    report = {"checkpoint_sha256": archived["checkpoint_sha256"], "splits": {}}
    for split, selection in selections.items():
        inputs, labels, ids, digest = read_boards(args.data, split, "residual", selection, device)
        blank = inputs.eq(0)
        known = score_residual(model, model.codes[labels], labels)
        if not known.all():
            raise ValueError("Known correct codewords did not score as correct")
        correct = replay(model, inputs, labels)
        rows = {str(d): summarize(correct[:, d], blank) for d in (0, 1, 4, 8, 16)}
        if split == "test":
            verify_original_labels(inputs, labels, args.original_test_csv)
            _, native_labels, native_ids, _ = read_boards(args.data, split, "trm", selection, device)
            if native_ids != ids or not torch.equal(native_labels - 2, labels):
                raise ValueError("The native and residual target encodings do not round-trip")
            trace = archived["rows"]
            if digest != archived["data_sha256"] or ids != [r["id"] for r in trace]:
                raise ValueError("Input cohort differs from the original GPU run")
            for observed, original in zip(correct.all(2).tolist(), trace):
                if observed[:len(original["correct"])] != original["correct"]:
                    raise ValueError("CPU replay did not reproduce the GPU exact trace")
            for budget in (0.1, 0.75):
                indices = [sum(t <= budget for t in row["seconds"]) - 1 for row in trace]
                if min(indices) < 0:
                    raise ValueError(f"No available prediction at {budget}s on at least one input")
                rows[f"at_{budget}s"] = summarize(correct[torch.arange(len(indices)), indices], blank)
            report["ever_exact"] = int(correct.all(2).any(1).sum())
            report["example_original_solution"] = (labels[0] + 1).reshape(9, 9).tolist()
            report["example_decoded_solution_codewords"] = (
                model.decode(model.codes[labels[0]]).argmax(-1) + 1
            ).reshape(9, 9).tolist()
        report["splits"][split] = dict(known_correct=int(known.all(1).sum()), endpoints=rows)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    for split, data in report["splits"].items():
        print(split, "known correct", data["known_correct"])
        for endpoint, stats in data["endpoints"].items():
            print(endpoint, stats)


if __name__ == "__main__":
    main()
