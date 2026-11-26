"""Train one Sudoku system under a declared single-GPU wall-time budget."""
import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from experiments.residual_reasoning.common import atomic, digest, read, source, write


def train_native(args, start):
    from models.ema import EMAHelper
    from pretrain import PretrainConfig, init_train_state, train_batch
    from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig

    native_data = args.data / "native"
    recipe = yaml.safe_load(args.native_config.read_text())
    recipe.update(
        data_paths=[str(native_data)], global_batch_size=768, seed=11, epochs=100000,
        load_checkpoint=None, checkpoint_path=str(args.root), ema=True,
    )
    config = PretrainConfig(**recipe)
    with (args.root / "all_config.yaml").open("x") as fp:
        yaml.safe_dump(config.model_dump(), fp)
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(seed=11, dataset_paths=[str(native_data)],
                            global_batch_size=config.global_batch_size,
                            test_set_mode=False, epochs_per_iter=16, rank=0, num_replicas=1),
        split="train",
    )
    model = init_train_state(config, dataset.metadata, rank=0, world_size=1)
    model.model.train()
    ema = EMAHelper(mu=config.ema_rate)
    ema.register(model.model)
    batch_iterator = iter(dataset)
    next_snapshot = 900
    seen = 0
    while True:
        try:
            _, batch, size = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(dataset)
            _, batch, size = next(batch_iterator)
        metrics = train_batch(config, model, batch, size, rank=0, world_size=1)
        ema.update(model.model)
        seen += size
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        if model.step == 1 or model.step % 100 == 0 or elapsed >= args.seconds:
            row = {k: float(v) for k, v in metrics.items()}
            row.update(step=model.step, sample_presentations=seen, elapsed_s=elapsed)
            with (args.root / "train.jsonl").open("a") as log:
                log.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row, allow_nan=False), flush=True)
        if elapsed >= next_snapshot or elapsed >= args.seconds:
            path = args.root / ("checkpoint.pt" if elapsed >= args.seconds else f"step_{model.step}.pt")
            state = model.model.state_dict()
            with atomic(path) as tmp:
                torch.save({k: ema.shadow.get(k, v).detach().cpu() for k, v in state.items()}, tmp)
            next_snapshot += 900
        if elapsed >= args.seconds:
            return dict(steps=model.step, examples=seen, elapsed_s=elapsed,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        checkpoint=digest(args.root / "checkpoint.pt"))


def train_residual(args, start):
    from experiments.residual_reasoning.train import run

    config = read(args.residual_config)
    config["training"].update(steps=1_000_000, save_every=1000, log_every=100,
                              max_seconds=max(1.0, args.seconds - (time.monotonic() - start)))
    write(args.root / "residual_config.json", config)
    run(cfg=config, method="recurrent", seed=11, data_path=args.data / "residual",
        out=args.root / "model", device="cuda", resume=True)
    summary = read(args.root / "model" / "complete.json")
    summary.update(training_seconds=summary["elapsed_s"], elapsed_s=time.monotonic() - start)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("trm", "residual"), required=True)
    for name in ("root", "data", "native-config", "residual-config"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    args = parser.parse_args()
    if args.seconds <= 0 or torch.cuda.device_count() != 1:
        parser.error("Expected a positive budget and a single visible GPU")
    args.root.mkdir(parents=True)
    write(args.root / "manifest.json", dict(
        arm=args.arm, source=source(), budget_s=args.seconds,
        cohort_sha256=digest(args.data / "cohort.json"), device=torch.cuda.get_device_name(),
        torch=str(torch.__version__), script=digest(Path(__file__)), seed=11,
    ))
    torch.manual_seed(11)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.monotonic()
    summary = train_native(args, start) if args.arm == "trm" else train_residual(args, start)
    write(args.root / "complete.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
