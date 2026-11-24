"""Expand the study into idempotent jobs. Use the same runner on a laptop and Slurm."""

import argparse
import os
from pathlib import Path

import torch

from .common import digest, exclusive, fingerprint, read, source, write
from .data import Dataset
from .evaluate import checked_output, evaluate_arm, make_bank, price_reference
from .model import Architecture, METHODS, NO_FIELD
from .objective import Training
from .train import hardware, run


def jobs(cfg):
    Architecture(**cfg["architecture"])
    Training(**cfg["training"])
    ev = cfg["evaluation"]
    if (
        min(
            ev["batch"],
            ev["time_reference_steps"],
            ev["timing_batches"],
            ev["patience"],
        )
        < 1
        or not 0 <= ev["uphill"] <= 1
        or min(ev["restart_every"], ev["diagnostic_examples"]) < 0
        or min(ev["forks"]) != 0
        or min(ev["horizons"]) != 0
        or max(ev["horizons"]) > ev["max_cost_steps"]
    ):
        raise ValueError("Invalid evaluation protocol")
    if not cfg["references"] or set(cfg["references"]) - {"direct", "identity"}:
        raise ValueError("References must use direct or identity updates")
    plan = {"reference": [], "joint": [], "shared": [], "bank": [], "eval": []}
    for task in cfg["tasks"]:
        for seed in cfg["seeds"]:
            base = f"{task}/seed-{seed}"

            def row(tag, method, ref=None, override=None):
                return dict(
                    id=f"{base}/{tag}",
                    task=task,
                    seed=seed,
                    method=method,
                    ref=ref,
                    override=override or {},
                )

            for ref in cfg["references"]:
                prior = f"{base}/reference/{ref}"
                plan["reference"].append(row(f"reference/{ref}", ref))
                plan["bank"].append(row(f"bank/{ref}", ref, prior))
                for method in cfg["shared"]:
                    if method in NO_FIELD | {"free_energy"}:
                        raise ValueError(f"Invalid frozen method {method}")
                    plan["shared"].append(row(f"shared/{ref}/{method}", method, prior))
                for edits in (1, 2):
                    plan["eval"].append(
                        row(f"shared/{ref}/search{edits}", f"search{edits}", prior)
                    )
            for method in cfg["joint"]:
                if method in cfg["references"]:
                    raise ValueError("Reference models are already jointly trained")
                plan["joint"].append(row(f"joint/{method}", method))
            for name, override in cfg["ablations"].items():
                Training(**(cfg["training"] | override))
                plan["joint"].append(
                    row(f"ablation/{name}", "recurrent", override=override)
                )
    for group in ["reference", "joint", "shared"]:
        plan["eval"].extend(plan[group])
    for stage, items in plan.items():
        ids = [i["id"] for i in items]
        if len(set(ids)) != len(ids) or not all(items):
            raise ValueError(f"Empty or duplicate jobs in {stage}")
    for m in [*cfg["joint"], *cfg["shared"]]:
        if m not in METHODS:
            raise ValueError(f"Unknown architecture {m}")
    return plan


def ready(root, key):
    run_dir = root / "models" / key
    ck = run_dir / "checkpoint.pt"
    if digest(ck) != read(run_dir / "complete.json")["checkpoint"]:
        raise ValueError(f"Incomplete or corrupt model: {key}")
    return ck


def budget(path, ref_path, dev, cfg, length, device):
    spec = dict(
        reference=digest(ref_path),
        dev=dev.fingerprint,
        cfg=cfg,
        length=length,
        hardware=hardware(device),
        source=source(),
    )
    # Bank jobs normally populate the timing cache before any arm is evaluated.
    with exclusive(path):
        if checked_output(path, spec):
            return read(path / "price.json")
        price = price_reference(ref_path, dev, cfg, length, device)
        write(path / "price.json", price)
        write(
            path / "complete.json",
            dict(spec=spec, files={"price.json": digest(path / "price.json")}),
        )
        return price


def execute(cfg, root, data_dir, stage, i, device, splits):
    plan = jobs(cfg)
    if not 0 <= i < len(plan[stage]):
        raise ValueError(f"Index {i} outside stage {stage}")
    job = plan[stage][i]
    task, key, seed = job["task"], job["id"], job["seed"]
    dataset = data_dir / task
    ref = ready(root, job["ref"]) if job["ref"] else None
    if stage in {"reference", "joint", "shared"}:
        run(
            cfg=cfg,
            method=job["method"],
            seed=seed,
            data_path=dataset,
            out=root / "models" / key,
            reference=ref,
            override=job["override"],
            device=device,
            resume=True,
        )
        return
    if int(os.getenv("WORLD_SIZE", "1")) != 1:
        raise ValueError("Bank and evaluation jobs run in one process")
    dev = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    )
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    ev = cfg["evaluation"]
    meta = read(dataset / "dataset.json")
    refname = job["ref"].split("/")[-1] if ref else cfg["references"][0]
    base = Path(task) / f"seed-{seed}"
    bankroot = root / "banks" / base / refname
    timer = root / "prices" / base / refname
    price_ref = ref if ref else ready(root, str(base / "reference" / refname))
    is_search = job["method"].startswith("search")
    ckpath = None if stage == "bank" else (ref if is_search else ready(root, key))
    for split in splits:
        if split == "long" and split not in meta["splits"]:
            continue
        ds = Dataset(dataset, split)
        bpath = bankroot / split
        length = ds.x.shape[1]
        if stage == "bank":
            budget(
                timer / str(length), price_ref, Dataset(dataset, "dev"), ev, length, dev
            )
            make_bank(bpath, ref, ds, ev["forks"], ev["batch"], dev)
        else:
            price_dir = timer / str(length)
            price_file = price_dir / "price.json"
            seal = read(price_dir / "complete.json")
            if (
                seal["files"]["price.json"] != digest(price_file)
                or seal["spec"]["hardware"] != hardware(dev)
                or seal["spec"]["reference"] != digest(price_ref)
                or seal["spec"]["cfg"] != ev
            ):
                raise ValueError("Timing cache or evaluation hardware mismatch")
            if ref:
                sealed = read(bpath / "complete.json")
                if sealed["files"]["bank.pt"] != digest(bpath / "bank.pt"):
                    raise ValueError("Corrupt fork bank")
            evaluate_arm(
                root / "evaluations" / key / split,
                ckpath,
                ds,
                ev,
                dev,
                read(price_file),
                bank_path=bpath if ref else None,
                search_edits=int(job["method"][-1]) if is_search else 0,
                seed=seed,
            )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["plan", "run", "count", "seal"])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--root", type=Path, default=Path("runs/residual"))
    p.add_argument("--data", type=Path, default=Path("data/residual"))
    p.add_argument("--stage", choices=["reference", "joint", "shared", "bank", "eval"])
    p.add_argument("--index", type=int)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    p.add_argument(
        "--splits",
        choices=["dev", "test", "long"],
        nargs="+",
        default=None,
    )
    args = p.parse_args()
    cfg = read(args.config)
    plan = jobs(cfg)
    if args.action == "plan":
        for stage in plan:
            for i, job in enumerate(plan[stage]):
                print(f"{stage:10s} {i:4d}  {job['id']}")
    elif args.action == "count":
        if not args.stage:
            p.error("count requires --stage")
        print(len(plan[args.stage]))
    elif args.action == "seal":
        # Locks protocol + actual dataset identities before scheduling. No training here.
        with exclusive(args.root):
            spec = dict(
                schema=1,
                config=cfg,
                source=source(),
                plan=plan,
                data={
                    task: fingerprint(read(args.data / task / "dataset.json"))
                    for task in cfg["tasks"]
                },
            )
            where = args.root / "study.json"
            if where.exists() and read(where) != spec:
                raise ValueError("Study identity changed. Select a fresh result root.")
            write(where, spec)
    else:
        if args.stage is None or args.index is None:
            p.error("run requires --stage and --index (see plan)")
        spec = read(args.root / "study.json")
        if cfg != spec["config"] or source() != spec["source"] or plan != spec["plan"]:
            raise ValueError("Source or config differs from sealed study")
        job = plan[args.stage][args.index]
        if (
            fingerprint(read(args.data / job["task"] / "dataset.json"))
            != spec["data"][job["task"]]
        ):
            raise ValueError("Dataset differs from sealed study")
        execute(
            cfg,
            args.root,
            args.data,
            args.stage,
            args.index,
            args.device,
            args.splits or cfg.get("splits", ["dev", "test", "long"]),
        )


if __name__ == "__main__":
    main()
