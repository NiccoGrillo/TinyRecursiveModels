"""Train one joint model or fit one controller on a pinned reference checkpoint."""

import argparse
import contextlib
import json
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .common import atomic, digest, exclusive, fingerprint, read, source, write
from .data import Dataset
from .model import Architecture, METHODS, NO_FIELD, Reasoner
from .objective import Objective, Training


def runtime(device):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    world, rank = int(os.getenv("WORLD_SIZE", "1")), int(os.getenv("RANK", "0"))
    if device == "cuda":
        torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if world > 1:
        dist.init_process_group(backend="nccl" if device == "cuda" else "gloo")
    return torch.device(device), world, rank


def hardware(device):
    return dict(
        torch=str(torch.__version__),
        numpy=np.__version__,
        python=platform.python_version(),
        platform=platform.machine(),
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        precision="fp32",
        tf32=False,
    )


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def from_checkpoint(path, device):
    ck = checkpoint(path)
    spec = ck["spec"]
    m = Reasoner(Architecture(**spec["architecture"]), *spec["vocab"], spec["method"])
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), spec


def transfer(model, path, data, arch):
    ck = checkpoint(path)
    ref = ck["spec"]
    if (
        ref["dataset"] != data.fingerprint
        or ref["architecture"] != asdict(arch)
        or ref["reference"] is not None
        or ref["method"] in NO_FIELD
    ):
        raise ValueError(
            "Reference data, architecture or training condition is incompatible"
        )
    keys = {
        k: v
        for k, v in ck["model"].items()
        if k.startswith(("field.", "initial.", "input.")) or k in {"position", "codes"}
    }
    required = {
        k
        for k in model.state_dict()
        if k.startswith(("field.", "initial.", "input.")) or k in {"position", "codes"}
    }
    if set(keys) != required:
        raise ValueError("Incomplete shared field or initializer")
    model.load_state_dict(keys, strict=False)
    for module in (model.input, model.initial, model.field):
        module.requires_grad_(False)


def trim_log(path, step):
    if path.exists():
        lines = path.read_text().splitlines()
        with atomic(path) as tmp:
            with tmp.open("w") as out:
                for i, line in enumerate(lines):
                    try:
                        item = json.loads(line)
                    except ValueError:
                        if i == len(lines) - 1:
                            break
                        raise
                    if item["step"] <= step:
                        out.write(line + "\n")


def run(
    *,
    cfg,
    method,
    seed,
    data_path,
    out,
    reference=None,
    override=None,
    device="auto",
    resume=False,
):
    dev, world, rank = runtime(device)
    try:
        _run(
            cfg,
            method,
            seed,
            data_path,
            Path(out),
            reference,
            override or {},
            dev,
            world,
            rank,
            resume,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run(
    cfg, method, seed, data_path, out, reference, override, dev, world, rank, resume
):
    arch = Architecture(**cfg["architecture"])
    train = Training(**(cfg["training"] | override))
    data = Dataset(data_path, "train")
    if train.batch % world or train.batch // world < 2:
        raise ValueError(
            "Global batch must divide world size with at least two examples per rank"
        )
    if reference and method in NO_FIELD | {"free_energy"}:
        raise ValueError("This method requires joint training")
    torch.manual_seed(seed)
    vocab = [data.meta["input_vocab"], data.meta["output_vocab"]]
    m = Reasoner(arch, *vocab, method)
    if reference:
        transfer(m, reference, data, arch)
    m = m.to(dev)
    params = [p for p in m.parameters() if p.requires_grad]
    optim = (
        torch.optim.AdamW(params, lr=train.lr, weight_decay=train.weight_decay)
        if params
        else None
    )
    spec = dict(
        schema=1,
        architecture=asdict(arch),
        training=asdict(train),
        method=method,
        seed=seed,
        dataset=data.fingerprint,
        vocab=vocab,
        world=world,
        source=source(),
        reference=digest(reference) if reference else None,
        hardware=hardware(dev),
        container_sha256=os.getenv("RR_IMAGE_SHA256"),
    )
    if world > 1 and optim:
        objective = DistributedDataParallel(
            Objective(m, train, bool(reference)),
            device_ids=[torch.cuda.current_device()] if dev.type == "cuda" else None,
            find_unused_parameters=True,
        )
    else:
        objective = Objective(m, train, bool(reference))
    log = out / "train.jsonl"
    dest = out / "checkpoint.pt"
    with exclusive(out) if rank == 0 else contextlib.nullcontext():
        if world > 1:
            dist.barrier()
        step, elapsed = 0, 0.0
        manifest = out / "manifest.json"
        if manifest.exists():
            if not resume or read(manifest)["spec"] != spec:
                raise ValueError(
                    f"Existing run or resume configuration mismatch: {out}"
                )
            if (out / "complete.json").exists():
                if digest(dest) != read(out / "complete.json")["checkpoint"]:
                    raise ValueError("Completed checkpoint checksum mismatch")
                return
            if dest.exists():
                ck = checkpoint(dest)
                if ck["spec"] != spec:
                    raise ValueError("Checkpoint provenance mismatch")
                m.load_state_dict(ck["model"])
                if optim:
                    optim.load_state_dict(ck["optimizer"])
                step, elapsed = ck["step"], ck["elapsed"]
        elif dest.exists() or log.exists():
            raise ValueError(f"Unowned output in {out}")
        if world > 1:
            dist.barrier()  # All ranks finished reading prior state before rank zero writes.
        if rank == 0:
            write(
                manifest,
                dict(
                    spec=spec,
                    id=fingerprint(spec),
                    parameters=sum(p.numel() for p in m.parameters()),
                    trainable=sum(p.numel() for p in params),
                    reference_path=str(reference) if reference else None,
                    scheduler_job=os.getenv("SLURM_JOB_ID"),
                    container_sha256=os.getenv("RR_IMAGE_SHA256"),
                ),
            )
            trim_log(log, step)
        if world > 1:
            dist.barrier()
        start = time.perf_counter()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        # Parameter-free shared arms need only the transferred checkpoint.
        limit = train.steps if optim else 0
        budget_hit = train.max_seconds is not None and elapsed >= train.max_seconds
        while step < limit and not budget_hit:
            step += 1
            # Global minibatch shared across methods. Other random draws are rank/step-local.
            indices = torch.randint(
                len(data.x),
                (train.batch,),
                generator=torch.Generator().manual_seed(seed * 10000019 + step),
            )
            x, y, editable = data.batch(indices.chunk(world)[rank], dev)
            torch.manual_seed((seed * 10000019 + step) * world + rank)
            optim.zero_grad(set_to_none=True)
            loss, stats = objective(x, y, editable)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                params, train.clip, error_if_nonfinite=True
            )
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            optim.step()
            if train.max_seconds is not None:
                sync(dev)
                hit = torch.tensor(
                    elapsed + time.perf_counter() - start >= train.max_seconds,
                    device=dev,
                )
                if world > 1:
                    dist.broadcast(hit, 0)
                budget_hit = bool(hit)
            if budget_hit or step % train.log_every == 0 or step == limit or step == 1:
                metrics = torch.stack([v.detach() for v in stats.values()] + [grad])
                if world > 1:
                    dist.all_reduce(metrics)
                    metrics /= world
                if rank == 0:
                    record = dict(zip([*stats, "gradient_norm"], metrics.tolist()))
                    record.update(
                        step=step,
                        examples=step * train.batch,
                        elapsed_s=elapsed + time.perf_counter() - start,
                    )
                    with log.open("a") as handle:
                        handle.write(json.dumps(record, allow_nan=False) + "\n")
            if budget_hit or step % train.save_every == 0 or step == limit:
                if rank == 0:
                    with atomic(dest) as tmp:
                        torch.save(
                            dict(
                                spec=spec,
                                model=m.state_dict(),
                                optimizer=optim.state_dict(),
                                step=step,
                                elapsed=elapsed + time.perf_counter() - start,
                            ),
                            tmp,
                        )
                if world > 1:
                    dist.barrier()
        if rank == 0:
            if not optim:
                with atomic(dest) as tmp:
                    torch.save(
                        dict(
                            spec=spec,
                            model=m.state_dict(),
                            optimizer=None,
                            step=0,
                            elapsed=0.0,
                        ),
                        tmp,
                    )
            summary = dict(
                stop_reason=(
                    "no_controller"
                    if not optim
                    else "time_budget"
                    if budget_hit
                    else "step_budget"
                ),
                step=step,
                examples=step * train.batch,
                checkpoint=digest(dest),
                elapsed_s=elapsed + time.perf_counter() - start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated()
                if dev.type == "cuda"
                else None,
            )
            write(out / "complete.json", summary)
            print(f"Completed {out} at step {step}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    run(
        cfg=read(a.config),
        method=a.method,
        seed=a.seed,
        data_path=a.data,
        out=a.out,
        reference=a.reference,
        device=a.device,
        resume=a.resume,
    )


if __name__ == "__main__":
    main()
