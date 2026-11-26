"""Calibrate a shared native TRM fork and continuation budget on Sudoku dev."""

import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from evaluate_matched_sudoku import model_from_checkpoint, read_boards, sha

FORKS = (2, 4, 6, 8, 12)
REFERENCE_STEPS = 4
MIN_COHORT = 32
MIN_NATIVE_REPAIRS = 8


def select_fork(traces):
    """Choose the eligible fork with most native repairs, then the earliest."""
    candidates = []
    for fork in FORKS:
        budget = statistics.median(
            row["seconds"][fork + REFERENCE_STEPS - 1] - row["seconds"][fork - 1]
            for row in traces
        )
        correct = [row["correct"][fork - 1] for row in traces]
        endpoints = []
        for row in traces:
            origin = row["seconds"][fork - 1]
            affordable = [ok for t, ok in zip(row["seconds"][fork:], row["correct"][fork:])
                          if t - origin <= budget]
            endpoints.append(affordable[-1] if affordable else row["correct"][fork - 1])
        right = sum(correct)
        wrong = len(correct) - right
        repaired = sum(not a and b for a, b in zip(correct, endpoints))
        broken = sum(a and not b for a, b in zip(correct, endpoints))
        candidates.append(dict(
            fork=fork, seconds=budget, right=right, wrong=wrong, repaired=repaired,
            broken=broken, final=sum(endpoints),
            eligible=min(right, wrong) >= MIN_COHORT and repaired >= MIN_NATIVE_REPAIRS
            and repaired > broken,
        ))
    eligible = [row for row in candidates if row["eligible"]]
    choice = max(eligible, key=lambda x: (x["repaired"], -x["fork"])) if eligible else None
    return choice, candidates


@torch.inference_mode()
def rollout(model, board, label, device, steps, measure):
    batch = {"inputs": board[None], "puzzle_identifiers": torch.zeros(1, dtype=torch.long, device=device)}
    with torch.device(device):
        carry = model.initial_carry(batch)
        predictions, times = [], []
        if measure and device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for t in range(1, steps + 1):
            carry, output = model(carry, batch)
            if measure:
                predictions.append(output["logits"].argmax(-1))
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                times.append(time.perf_counter() - start)
            # The released native wrapper halts exactly at its configured cap.
            # Its step count is checked after the completed open-loop prefix.
        if int(carry.steps.item()) != steps:
            raise ValueError("Native rollout reset before the requested macro depth")
    if not measure:
        return carry.inner_carry, output["logits"]
    correct = torch.cat(predictions).eq(label).all(1).tolist()
    return dict(seconds=times, correct=correct)


def write_fork_bank(path, model, x, y, ids, traces, choice, device):
    h, l, logits = [], [], []
    fork = choice["fork"]
    for board, label, trace in zip(x, y, traces):
        carry, score = rollout(model, board, label, device, fork, measure=False)
        # Verify against the independent timing pass on the same inputs.
        if bool(score.argmax(-1).eq(label).all()) != trace["correct"][fork - 1]:
            raise ValueError("Recreated fork disagreed with the measured trajectory")
        h.append(carry.z_H.cpu())
        l.append(carry.z_L.cpu())
        logits.append(score.cpu())
    torch.save(dict(ids=ids, fork=fork, z_H=torch.cat(h), z_L=torch.cat(l),
                    logits=torch.cat(logits), inputs=x.cpu(), labels=y.cpu()), path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "config", "weights", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--limit", type=int, default=256)
    args = p.parse_args()
    if args.limit < 1 or sha(args.weights) != args.checkpoint_sha256:
        p.error("Expected a positive cohort size and the sealed checkpoint")
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    x, y, ids, data_sha = read_boards(args.data, "dev", "trm", slice(args.limit), device)
    if len(ids) != args.limit or len(set(ids)) != len(ids):
        raise ValueError("Development cohort size or input IDs differ")
    model, config = model_from_checkpoint(SimpleNamespace(family="trm", weights=args.weights, config=args.config), device)
    steps = config["halt_max_steps"]
    if steps != max(FORKS) + REFERENCE_STEPS:
        raise ValueError("Expected the matched Sudoku reference's sixteen macro steps")
    args.output.mkdir()
    for board, label in zip(x[:4], y[:4]):
        rollout(model, board, label, device, steps, measure=True)
    traces = [dict(id=key, **rollout(model, board, label, device, steps, measure=True))
              for key, board, label in zip(ids, x, y)]
    choice, candidates = select_fork(traces)
    bank = args.output / "fork_states.pt"
    if choice:
        write_fork_bank(bank, model, x, y, ids, traces, choice, device)
    payload = dict(
        checkpoint_sha256=args.checkpoint_sha256, config_sha256=sha(args.config),
        data_sha256=data_sha, cohort="dev", n=len(ids),
        device=torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        torch=str(torch.__version__), policy=dict(reference_steps=REFERENCE_STEPS,
        minimum_per_cohort=MIN_COHORT, minimum_native_repairs=MIN_NATIVE_REPAIRS),
        choice=choice, candidates=candidates, bank_sha256=sha(bank) if choice else None,
        traces=traces,
    )
    with (args.output / "calibration.json").open("x") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(json.dumps(dict(choice=choice, candidates=candidates)), flush=True)


if __name__ == "__main__":
    main()
