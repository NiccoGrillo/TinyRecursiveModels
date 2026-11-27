"""Replay the original frozen-refiner training and extend its optimizer trajectory."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from evaluate_matched_sudoku import model_from_checkpoint, read_boards, sha
from experiments.residual_reasoning.model import Architecture
from experiments.residual_reasoning.trm_refinement import TRMRefiner
from replay_trm_refinement import BANK_KEYS, fixed_depth
from run_trm_refinement_dev import BATCH, LOOPS, SEED, STEPS, native_training_bank, train

METHODS = ("unrestricted", "recurrent")
MILESTONES = (128, 512, 1024, 2048, 4096)


def run(root, output):
    device = torch.device("cuda")
    hardware = torch.cuda.get_device_name()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    calibration_path = root / "dev-calibration/calibration.json"
    bank_path = root / "dev-calibration/fork_states.pt"
    result_path = root / "dev-refiners/dev-results.json"
    weights = root / "checkpoint/checkpoint.pt"
    config = root / "checkpoint/all_config.yaml"
    data = root / "train-only"
    calibration = json.loads(calibration_path.read_text())
    prior = json.loads(result_path.read_text())
    cohort = json.loads((data / "cohort.json").read_text())
    fork = prior["fork"]
    hashes = {name: sha(filename) for name, filename in
              dict(calibration=calibration_path, bank=bank_path, result=result_path,
                   weights=weights, config=config).items()}
    if (calibration["cohort"] != "dev" or calibration["choice"]["fork"] != fork
            or (fork, prior["seed"], prior["batch"], prior["loops"], prior["train_steps"])
            != (2, SEED, BATCH, LOOPS, STEPS)
            or hashes["calibration"] != prior["calibration_sha256"]
            or hashes["bank"] != calibration["bank_sha256"]
            or hashes["weights"] != prior["checkpoint_sha256"]
            or hashes["config"] != calibration["config_sha256"]
            or hardware != calibration["device"] or str(torch.__version__) != calibration["torch"]
            or cohort["splits"]["dev"]["sha256"] != calibration["data_sha256"]):
        raise ValueError("Expected the original frozen-refiner runtime, data manifest and checkpoint")
    dev = torch.load(bank_path, map_location="cpu", weights_only=True)
    dev_ids = dev["ids"]
    if (len(dev_ids) != 256 or dev_ids != prior["ids"] or dev["fork"] != fork
            or dev["logits"].argmax(-1).eq(dev["labels"]).all(-1).tolist() != prior["outcomes"]["stop"]):
        raise ValueError("Development IDs or original fork predictions differ")
    inputs, labels, train_ids, train_sha = read_boards(
        data, "train", "trm", slice(0, None, cohort["augments"] + 1), device
    )
    if (len(train_ids) != 1000 or train_ids != prior["train_ids"] or train_ids != cohort["original_ids"]
            or train_sha != prior["train_sha256"] or set(train_ids) & set(dev_ids)):
        raise ValueError("Training IDs or input bytes differ")
    native, _ = model_from_checkpoint(SimpleNamespace(family="trm", weights=weights, config=config), device)
    train_bank = native_training_bank(native, inputs, labels, fork)
    banks = {"train": (train_bank, train_ids), "dev": (tuple(dev[k].to(device) for k in BANK_KEYS), dev_ids)}
    output.mkdir()
    architecture = Architecture()
    torch.manual_seed(SEED)
    shared = TRMRefiner(architecture, train_bank[1].shape[-1], "unrestricted").state_dict()
    report = dict(fork=fork, inference_steps=LOOPS, milestones=MILESTONES, batch=BATCH, seed=SEED,
                  device=hardware, torch=str(torch.__version__), train_sha256=train_sha,
                  input_sha256=hashes, source_sha256=sha(Path(__file__)), splits={}, arms={})
    for split, (bank, ids) in banks.items():
        flags = bank[3].argmax(-1).eq(bank[4]).all(-1).tolist()
        report["splits"][split] = dict(ids=ids, fork_exact_ids=[i for i, ok in zip(ids, flags, strict=True) if ok])

    for method in METHODS:
        archived = root / "dev-refiners" / f"{method}.pt"
        archived_sha = sha(archived)
        if archived_sha != prior["arms"][method]["checkpoint_sha256"]:
            raise ValueError(f"Archived {method} checkpoint checksum mismatch")
        reference = torch.load(archived, map_location="cpu", weights_only=True)
        snapshots = []

        def snapshot(step, refiner, optimizer):
            state = refiner.state_dict()
            if step == STEPS:
                mismatched = [key for key, value in state.items() if not torch.equal(value.cpu(), reference[key])]
                if mismatched:
                    diffs = {key: (state[key].cpu() - reference[key]).abs().max().item() for key in mismatched}
                    raise ValueError(f"{method} step {STEPS} replay differs from the archive: {diffs}")
                print(f"{method} step {STEPS} matches every archived weight bitwise", flush=True)
            ckpt = output / f"{method}-{step}.pt"
            temporary = ckpt.with_suffix(".tmp")
            torch.save(dict(step=step, model=state, optimizer=optimizer.state_dict()), temporary)
            temporary.replace(ckpt)
            refiner.eval()
            try:
                metrics = {split: fixed_depth(refiner, bank, ids, LOOPS, device)
                           for split, (bank, ids) in banks.items()}
            finally:
                refiner.train()
            row = dict(optimizer_steps=step, sample_presentations=step * BATCH, checkpoint_sha256=sha(ckpt),
                       gain=refiner.gain.detach().item(), fixed_depth=metrics)
            snapshots.append(row)
            (output / f"{method}-{step}.json").write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
            print(method, step, {split: {k: rows[-1][k] for k in ("exact", "repaired", "broken", "cell_nll")}
                                 for split, rows in metrics.items()}, flush=True)

        _, elapsed = train(method, architecture, train_bank, shared, device, steps=MILESTONES[-1],
                           checkpoint_steps=MILESTONES, on_checkpoint=snapshot)
        report["arms"][method] = dict(archived_checkpoint_sha256=archived_sha, step_512_bitwise_match=True,
                                      seconds_including_checkpoints_and_evaluation=elapsed, snapshots=snapshots)
    (output / "duration.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.root, args.output)
