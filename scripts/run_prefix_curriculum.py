"""Run matched-update prefix-sum length schedules on one process."""
import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch

from experiments.residual_reasoning.common import atomic, digest, exclusive, read, source, write
from experiments.residual_reasoning.data import Dataset, prepare
from experiments.residual_reasoning.model import Architecture, Reasoner
from experiments.residual_reasoning.objective import Objective, Training
from experiments.residual_reasoning.train import checkpoint, hardware, runtime, sync

DEPTHS = (0, 1, 2, 4, 8, 16)
LENGTHS = (4, 8, 16)
SEED = 11
DATA_SEED = 2809


def evaluate(model, data, device):
    counts = torch.zeros((len(DEPTHS), data.x.size(1)), dtype=torch.int64, device=device)
    exact, nonfinite = [], []
    for start in range(0, len(data.ids), 16):
        inputs, labels, _ = data.batch(slice(start, start + 16), device)
        with torch.no_grad():
            encoded = model.encode(inputs)
            answer, memory = model.start(encoded)
        batch_exact, batch_nonfinite = [], []
        finite = torch.ones(len(inputs), dtype=torch.bool, device=device)
        for depth in range(DEPTHS[-1] + 1):
            if depth:
                answer, memory = model.step(encoded, answer, memory, train=False)
            finite &= answer.isfinite().flatten(1).all(1) & memory.isfinite().flatten(1).all(1)
            if depth in DEPTHS:
                with torch.no_grad():
                    correct = model.decode(answer).argmax(-1).eq(labels) & finite[:, None]
                counts[DEPTHS.index(depth)] += correct.sum(0)
                batch_exact.append(correct.all(1))
                batch_nonfinite.append((~finite).sum())
        exact.append(torch.stack(batch_exact))
        nonfinite.append(torch.stack(batch_nonfinite))
    return dict(
        depths=DEPTHS,
        correct_per_position=counts.cpu().tolist(),
        exact_by_input=torch.cat(exact, dim=1).cpu().tolist(),
        nonfinite=torch.stack(nonfinite).sum(0).cpu().tolist(),
    )


def fit(arm, schedule, root, arch, cfg, train, dev, steps_per_stage, device, *, initial_checkpoint=None):
    torch.manual_seed(SEED)
    model = Reasoner(arch, 8, 8, "identity").to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    initial_step = 0
    if initial_checkpoint is not None:
        initial = checkpoint(initial_checkpoint)
        model.load_state_dict(initial["model"])
        optimizer.load_state_dict(initial["optimizer"])
        initial_step = initial["step"]
    objective = Objective(model, cfg, False)
    folder = root / arm
    folder.mkdir()
    milestones = []
    for stage, length in enumerate(schedule, 1):
        data = train[length]
        first_step = initial_step + (stage - 1) * steps_per_stage + 1
        last_step = initial_step + stage * steps_per_stage
        model.train()
        sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for step in range(first_step, last_step + 1):
            step_seed = SEED * 10000019 + step
            indices = torch.randint(
                len(data.ids), (cfg.batch,), generator=torch.Generator().manual_seed(step_seed)
            )
            x, y, editable = data.batch(indices, device)
            torch.manual_seed(step_seed)
            optimizer.zero_grad(set_to_none=True)
            loss, stats = objective(x, y, editable)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.clip, error_if_nonfinite=True
            )
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            optimizer.step()
            if step in (first_step, last_step) or step % 256 == 0:
                record = {key: value.item() for key, value in stats.items()}
                record.update(step=step, length=length, gradient_norm=norm.item())
                with (folder / "train.jsonl").open("a") as log:
                    log.write(json.dumps(record, allow_nan=False) + "\n")
        sync(device)
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
        model_path = folder / f"step-{last_step}.pt"
        with atomic(model_path) as temp:
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=last_step), temp)
        model.eval()
        val = {str(n): evaluate(model, dataset, device) for n, dataset in dev.items()}
        milestone = dict(
            steps=last_step, last_training_length=length, training_seconds=elapsed,
            peak_allocated_bytes=peak, checkpoint=digest(model_path), dev=val,
        )
        milestones.append(milestone)
        write(folder / "milestones.json", milestones)
        index = DEPTHS.index(4)
        summary = {n: sum(values["exact_by_input"][index]) for n, values in val.items()}
        print(f"{arm} step {last_step}, length {length}, dev exact at depth 4: {summary}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--steps-per-stage", type=int, default=1024)
    args = parser.parse_args()
    if args.steps_per_stage < 1:
        parser.error("--steps-per-stage must be positive")
    device, world, _ = runtime(args.device)
    if world != 1 or (device.type == "cuda" and torch.cuda.device_count() != 1):
        parser.error("this diagnostic uses one process and at most one GPU")
    cfg = read(args.config)
    arch = Architecture(**cfg["architecture"])
    training = Training(**(cfg["training"] | {"steps": args.steps_per_stage * len(LENGTHS)}))
    schedules = {"direct": (16, 16, 16), "curriculum": LENGTHS}
    with exclusive(args.root):
        if (args.root / "study.json").exists():
            raise FileExistsError("Select a fresh experiment root")
        datasets = {16: args.scan}
        for length in LENGTHS[:-1]:
            path = args.root / f"data-length-{length}"
            prepare(path, "scan", [2048, 512, 512], DATA_SEED, length, 8)
            datasets[length] = path
        train = {n: Dataset(path, "train") for n, path in datasets.items()}
        dev = {n: Dataset(path, "dev") for n, path in datasets.items()}
        for n in LENGTHS:
            assert train[n].x.size(1) == dev[n].x.size(1) == n
            assert not set(train[n].ids).intersection(dev[n].ids)
        write(
            args.root / "study.json",
            dict(
                source=source(), script_sha256=digest(Path(__file__)),
                container_sha256=os.getenv("RR_IMAGE_SHA256"), hardware=hardware(device),
                seed=SEED, steps_per_stage=args.steps_per_stage, schedules=schedules,
                architecture=asdict(arch), training=asdict(training),
                data={str(n): {"dataset": dev[n].fingerprint, "dev_ids": dev[n].ids.tolist()}
                      for n in LENGTHS},
            ),
        )
        for arm, schedule in schedules.items():
            fit(arm, schedule, args.root, arch, training, train, dev, args.steps_per_stage, device)
        write(args.root / "complete.json", {arm: digest(args.root / arm / "milestones.json")
                                             for arm in schedules})


if __name__ == "__main__":
    main()
