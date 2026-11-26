"""Continue a sealed prefix-sum run with an explicit unroll depth."""

import argparse
import os
from dataclasses import asdict
from pathlib import Path

import torch

from experiments.residual_reasoning.common import digest, exclusive, read, source, write
from experiments.residual_reasoning.data import Dataset
from experiments.residual_reasoning.model import Architecture
from experiments.residual_reasoning.objective import Training
from experiments.residual_reasoning.train import hardware, runtime
from run_prefix_curriculum import fit

ARMS = {"direct-4": ("direct", 4), "curriculum-4": ("curriculum", 4),
        "curriculum-8": ("curriculum", 8)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--steps", type=int, default=1024)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    device, world, _ = runtime(args.device)
    if world != 1 or (device.type == "cuda" and torch.cuda.device_count() != 1):
        parser.error("use one process and at most one GPU")

    prior_arm, loops = ARMS[args.arm]
    prior = read(args.prior / "study.json")
    milestones = args.prior / prior_arm / "milestones.json"
    if digest(milestones) != read(args.prior / "complete.json")[prior_arm]:
        raise ValueError("Prior milestone checksum mismatch")
    last = read(milestones)[-1]
    checkpoint = milestones.with_name(f"step-{last['steps']}.pt")
    if digest(checkpoint) != last["checkpoint"] or last["last_training_length"] != 16:
        raise ValueError("Expected an intact length-16 training checkpoint")

    train = Dataset(args.scan, "train")
    datasets = {4: args.prior / "data-length-4", 8: args.prior / "data-length-8", 16: args.scan}
    dev = {n: Dataset(path, "dev") for n, path in datasets.items()}
    for n, dataset in dev.items():
        expected = prior["data"][str(n)]
        if dataset.fingerprint != expected["dataset"] or dataset.ids.tolist() != expected["dev_ids"]:
            raise ValueError(f"Length {n} development cohort differs from the prior run")
    cfg = Training(**(prior["training"] | {"steps": last["steps"] + args.steps, "loops": loops}))
    arch = Architecture(**prior["architecture"])

    with exclusive(args.root):
        study_path = args.root / "study.json"
        if study_path.exists():
            raise FileExistsError("Select a fresh output directory for each arm")
        write(study_path, dict(
            arm=args.arm, start_step=last["steps"], start_checkpoint=last["checkpoint"],
            source=source(), scripts={
                name: digest(Path(__file__).with_name(name))
                for name in ("run_prefix_curriculum.py", "run_prefix_continuation.py")
            },
            hardware=hardware(device), container_sha256=os.getenv("RR_IMAGE_SHA256"),
            architecture=asdict(arch), training=asdict(cfg), data=prior["data"],
        ))
        fit(args.arm, (16,), args.root, arch, cfg, {16: train}, dev,
            args.steps, device, initial_checkpoint=checkpoint)
        write(args.root / "complete.json", {args.arm: digest(args.root / args.arm / "milestones.json")})


if __name__ == "__main__":
    main()
