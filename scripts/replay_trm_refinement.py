"""Audit the frozen Sudoku refiners on their original native TRM fork states."""

import argparse
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from evaluate_matched_sudoku import model_from_checkpoint, read_boards, sha
from experiments.residual_reasoning.model import Architecture
from experiments.residual_reasoning.trm_refinement import TRMRefiner
from run_trm_refinement_dev import BATCH, METHODS, native_training_bank, timed_refinement

BANK_KEYS = ("inputs", "z_H", "z_L", "logits", "labels")


def fixed_depth(refiner, bank, ids, loops, device):
    totals = [dict(cell_nll=0.0, blank_nll=0.0, exact=[]) for _ in range(loops + 1)]
    initial = bank[3].argmax(-1).eq(bank[4]).all(-1).tolist()
    for start in range(0, len(ids), BATCH):
        inputs, h, l, logits, labels = [t[start : start + BATCH].to(device) for t in bank]
        blank = inputs.eq(1)
        with torch.no_grad():
            context, base, correction, workspace = refiner.prepare(inputs, h, l, logits)
        for step, row in enumerate(totals):
            if step:
                correction, workspace = refiner.advance(context, correction, workspace, train=False)
            with torch.no_grad():
                scores = refiner.readout(base, correction)
                loss = F.cross_entropy(scores.transpose(1, 2), labels, reduction="none")
                if not bool(loss.isfinite().all()) or (step == 0 and not torch.equal(scores, logits.float())):
                    raise ValueError(f"Invalid scores at depth {step}, batch starting at {start}")
                row["cell_nll"] += loss.sum(dtype=torch.float64).item()
                row["blank_nll"] += loss[blank].sum(dtype=torch.float64).item()
                row["exact"].extend(scores.argmax(-1).eq(labels).all(-1).tolist())

    cells = bank[4].numel()
    blanks = int(bank[0].eq(1).sum())
    rows = []
    for step, row in enumerate(totals):
        exact = row["exact"]
        repaired = [i for i, a, b in zip(ids, initial, exact, strict=True) if not a and b]
        broken = [i for i, a, b in zip(ids, initial, exact, strict=True) if a and not b]
        rows.append(dict(step=step, exact=sum(exact), repaired=len(repaired), broken=len(broken),
                         cell_nll=row["cell_nll"] / cells, blank_nll=row["blank_nll"] / blanks,
                         repaired_ids=repaired, broken_ids=broken))
    return rows


def zero_gain_control(refiner, bank, ids, expected, budget, device):
    def predict(i):
        return timed_refinement(refiner, *(t[i : i + 1] for t in bank), budget, device)

    with torch.no_grad():
        saved_gain = refiner.gain.clone()
        refiner.gain.zero_()
    try:
        for i in range(4):
            predict(i)
        rows = [predict(i) for i in range(len(ids))]
    finally:
        with torch.no_grad():
            refiner.gain.copy_(saved_gain)
    solved = [i for i, row in zip(ids, rows, strict=True) if row["correct"]]
    steps = [row["steps"] for row in rows]
    return dict(passed=solved == expected and min(steps) > 0 and not any(r["nonfinite"] for r in rows),
                exact=len(solved), solved_ids=solved, min_steps=min(steps),
                median_steps=statistics.median(steps), max_steps=max(steps),
                nonfinite=sum(r["nonfinite"] for r in rows),
                cap_hits=sum(r["cap_hit"] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    root = args.root
    device = torch.device(args.device)
    hardware = torch.cuda.get_device_name() if device.type == "cuda" else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    calibration_path = root / "dev-calibration/calibration.json"
    bank_path = root / "dev-calibration/fork_states.pt"
    result_path = root / "dev-refiners/dev-results.json"
    plan = json.loads(calibration_path.read_text())
    prior = json.loads(result_path.read_text())
    fork, budget = prior["fork"], prior["budget_seconds"]
    if (plan["cohort"] != "dev" or plan["choice"]["fork"] != fork or plan["choice"]["seconds"] != budget
            or (fork, prior["loops"], prior["batch"]) != (2, 4, BATCH)
            or sha(calibration_path) != prior["calibration_sha256"] or sha(bank_path) != plan["bank_sha256"]
            or plan["checkpoint_sha256"] != prior["checkpoint_sha256"]):
        raise ValueError("Calibration or saved fork bank differs from the development run")
    dev = torch.load(bank_path, map_location="cpu", weights_only=True)
    ids = dev["ids"]
    stop = prior["outcomes"]["stop"]
    if (len(ids) != 256 or ids != prior["ids"] or ids != [t["id"] for t in plan["traces"]]
            or dev["fork"] != fork or sum(stop) != 60
            or stop != dev["logits"].argmax(-1).eq(dev["labels"]).all(-1).tolist()):
        raise ValueError("Development fork IDs or original exact mask differ from the saved run")
    expected = [i for i, correct in zip(ids, stop, strict=True) if correct]
    dev_bank = tuple(dev[key].to(device) for key in BANK_KEYS)
    banks = {"dev": (dev_bank, ids)}
    if device.type == "cuda":
        weights = root / "checkpoint/checkpoint.pt"
        config = root / "checkpoint/all_config.yaml"
        data = root / "train-only"
        cohort = json.loads((data / "cohort.json").read_text())
        if (hardware != plan["device"] or str(torch.__version__) != plan["torch"]
                or sha(weights) != plan["checkpoint_sha256"] or sha(config) != plan["config_sha256"]
                or cohort["splits"]["dev"]["sha256"] != plan["data_sha256"]):
            raise ValueError("Expected the same hardware, dataset manifest and native checkpoint as training")
        inputs, labels, train_ids, train_sha = read_boards(
            data, "train", "trm", slice(0, None, cohort["augments"] + 1), device
        )
        if (len(train_ids) != 1000 or train_ids != cohort["original_ids"] or train_ids != prior["train_ids"]
                or train_sha != prior["train_sha256"] or set(train_ids) & set(ids)):
            raise ValueError("Training inputs differ from the 1,000 original training boards")
        native, _ = model_from_checkpoint(SimpleNamespace(family="trm", weights=weights, config=config), device)
        banks["train"] = (native_training_bank(native, inputs, labels, fork), train_ids)
        print("Recreated original 1,000 training fork states", flush=True)

    args.output.mkdir()
    payload = dict(fork=fork, budget_seconds=budget, device=hardware, torch=str(torch.__version__),
                   calibration_sha256=prior["calibration_sha256"], bank_sha256=plan["bank_sha256"],
                   checkpoint_sha256=prior["checkpoint_sha256"], config_sha256=plan["config_sha256"],
                   train_sha256=prior["train_sha256"] if "train" in banks else None,
                   original_result_sha256=sha(result_path), audit_script_sha256=sha(Path(__file__)),
                   zero_gain_expected_ids=expected, splits={}, methods={})
    for split, (bank, split_ids) in banks.items():
        ok = bank[3].argmax(-1).eq(bank[4]).all(-1).tolist()
        payload["splits"][split] = dict(ids=split_ids,
                                       fork_exact_ids=[i for i, a in zip(split_ids, ok, strict=True) if a])
    for method in METHODS:
        ckpt = root / "dev-refiners" / f"{method}.pt"
        digest = sha(ckpt)
        if digest != prior["arms"][method]["checkpoint_sha256"]:
            raise ValueError(f"{method} checkpoint hash mismatch")
        refiner = TRMRefiner(Architecture(), dev_bank[1].shape[-1], method).to(device).eval()
        refiner.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        gain = refiner.gain.item()
        control = zero_gain_control(refiner, dev_bank, ids, expected, budget, device)
        print(f"{method}: zero-gain exact IDs {'match' if control['passed'] else 'FAIL'}", flush=True)
        payload["methods"][method] = dict(checkpoint_sha256=digest, gain=gain, zero_gain=control)
        if control["passed"]:
            payload["methods"][method]["fixed_depth"] = {
                split: fixed_depth(refiner, bank, split_ids, prior["loops"], device)
                for split, (bank, split_ids) in banks.items()
            }
    (args.output / "replay.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    if not all(arm["zero_gain"]["passed"] for arm in payload["methods"].values()):
        raise RuntimeError("Production zero-gain control failed. See replay.json for the exact board IDs.")


if __name__ == "__main__":
    main()
