"""Exercise a complete miniature campaign plus the requested single-node DDP topology."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import torch

from .common import exclusive, read, write
from .model import Architecture, Reasoner
from .objective import Objective, Training
from .train import hardware
from .data import prepare
from .report import aggregate
from .study import execute, jobs


def production_shape(config, workers):
    """One synthetic per-rank training update on the real Sudoku tensor shapes."""
    device = torch.device("cuda")
    torch.manual_seed(11)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = Training(**config["training"])
    if cfg.batch % workers:
        raise ValueError("The configured global batch must divide the GPU count")
    batch = cfg.batch // workers
    x = torch.randint(10, (batch, 81), device=device)
    y = torch.randint(9, (batch, 81), device=device)
    m = Reasoner(Architecture(**config["architecture"]), 10, 9, "recurrent").to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg.lr)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    loss, _ = Objective(m, cfg, False)(x, y, x == 0)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        m.parameters(), cfg.clip, error_if_nonfinite=True
    )
    opt.step()
    torch.cuda.synchronize()
    return dict(
        hardware=hardware(device),
        local_batch=batch,
        length=81,
        loops=cfg.loops,
        loss=loss.item(),
        grad_norm=norm.item(),
        elapsed_s=time.perf_counter() - start,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument(
        "--gpus", type=int, default=1, help="Number of local workers for the DDP check"
    )
    a = p.parse_args()
    if a.gpus < 1 or (a.device == "cuda" and torch.cuda.device_count() != a.gpus):
        raise ValueError("Visible GPU count differs from the requested smoke topology")
    cfgpath = Path(__file__).resolve().parents[2] / "config/residual/smoke.json"
    cfg = read(cfgpath)
    root = a.root / "smoke"
    data = root / "data"
    with exclusive(root):
        if not (data / "scan").exists():
            prepare(data / "scan", "scan", [24, 8, 8], 2809, 5, 5)
        if a.gpus > 1:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nnodes=1",
                    f"--nproc-per-node={a.gpus}",
                    "-m",
                    "experiments.residual_reasoning.train",
                    "--config",
                    str(cfgpath),
                    "--method",
                    "recurrent",
                    "--seed",
                    "11",
                    "--data",
                    str(data / "scan"),
                    "--out",
                    str(root / "distributed"),
                    "--device",
                    a.device,
                    "--resume",
                ],
                check=True,
            )
        if a.device == "cuda":
            write(
                root / "production-shape.json",
                production_shape(read(cfgpath.with_name("study.json")), a.gpus),
            )
        target = root / "campaign"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.residual_reasoning.study",
                "seal",
                "--config",
                str(cfgpath),
                "--root",
                str(target),
                "--data",
                str(data),
            ],
            check=True,
        )
        for stage, items in jobs(cfg).items():
            for i in range(len(items)):
                execute(cfg, target, data, stage, i, a.device, ["dev", "test", "long"])
        aggregate(target, ["dev", "test", "long"], require_complete=True)


if __name__ == "__main__":
    main()
