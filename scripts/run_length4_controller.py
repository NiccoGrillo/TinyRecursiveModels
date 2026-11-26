"""Compare frozen-field controllers at a sealed length-four reference fork."""
import argparse
import json
from pathlib import Path

from experiments.residual_reasoning.common import atomic, digest, exclusive, read, source, write
from experiments.residual_reasoning.data import Dataset
from experiments.residual_reasoning.evaluate import evaluate_arm, make_bank, price_reference
from experiments.residual_reasoning.report import paired_rows, summarize, table
from experiments.residual_reasoning.train import checkpoint, hardware, run, runtime

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = p.parse_args()
    cfg = read(args.config)
    dev = Dataset(args.data, "dev")
    train = Dataset(args.data, "train")
    if digest(args.checkpoint) != args.checkpoint_sha256 or dev.x.size(1) != 4:
        raise ValueError("Expected the sealed curriculum checkpoint and length-four data")
    if len(set(train.ids) & set(dev.ids)):
        raise ValueError("Training and development IDs overlap")
    device, world, _ = runtime(args.device)
    if world != 1:
        raise ValueError("Expected one worker")
    ev = cfg["evaluation"] | {"forks": [4]}
    arms = ("identity", "direct", "gated", "stateless", "recurrent", "search1", "search2")
    root = args.root
    with exclusive(root):
        write(root / "protocol.json", dict(
            source=source(), runner_sha256=digest(Path(__file__)),
            starting_checkpoint=args.checkpoint_sha256, reference_training="4, 8, 16, 16",
            seed=11, dataset=dev.fingerprint, config=cfg, evaluation=ev,
            hardware=hardware(device), arms=("stop", *arms), primary="test/wall/fork4",
        ))
        prior = checkpoint(args.checkpoint)
        ref = root / "reference.pt"
        ref_spec = dict(architecture=cfg["architecture"], vocab=(8, 8), method="identity",
                        dataset=dev.fingerprint, reference=None,
                        upstream_checkpoint=args.checkpoint_sha256,
                        source_training_schedule=(4, 8, 16, 16))
        with atomic(ref) as tmp:
            torch.save(dict(model=prior["model"], spec=ref_spec, step=prior["step"]), tmp)
        models = {"identity": ref, "search1": ref, "search2": ref}
        for arm in ("direct", "gated", "stateless", "recurrent"):
            out = root / "models" / arm
            run(cfg=cfg, method=arm, seed=11, data_path=args.data,
                out=out, reference=ref, device=args.device, resume=True)
            models[arm] = out / "checkpoint.pt"
        price = price_reference(ref, dev, ev, 4, device)
        write(root / "price.json", price)
        summary, contrasts = [], []
        for split in ("dev", "test"):
            data = Dataset(args.data, split)
            if split == "test" and (set(data.ids) & (set(train.ids) | set(dev.ids))):
                raise ValueError("Held-out inputs overlap training or development")
            bank = root / "banks" / split
            make_bank(bank, ref, data, [4], ev["batch"], device)
            pairs = {}
            for arm in arms:
                out = root / "evaluations" / split / arm
                evaluate_arm(out, models[arm], data, ev, device, price, bank_path=bank,
                             search_edits=int(arm[-1]) if arm.startswith("search") else 0,
                             seed=11)
                rows, paired = summarize(out, {"split": split, "arm": arm}, [4])
                pairs[arm] = paired[4]
                for row in rows:
                    row["broken"] = row["right"] - row["retained"]
                    row["net_solves"] = row["recovered"] - row["broken"]
                    if arm == "identity" and row["policy"] == "step_0":
                        summary.append(row | {"arm": "stop", "policy": "wall"})
                    summary.append(row)
            ids, initial, _ = pairs["identity"]
            pairs["stop"] = (ids, initial, initial)
            contrasts.extend(paired_rows(dict(split=split, fork=4, policy="wall"), pairs))
        table(root / "summary.csv", summary)
        table(root / "paired.csv", contrasts)
        write(root / "complete.json", dict(
            summary=digest(root / "summary.csv"), paired=digest(root / "paired.csv"),
            checkpoints={arm: digest(path) for arm, path in models.items()},
        ))
        for row in summary:
            if row["split"] == "test" and row["policy"] == "wall":
                print(json.dumps({k: row[k] for k in ("arm", "n", "right", "wrong",
                                                       "recovered", "broken", "solved", "mean_t")}),
                      flush=True)


if __name__ == "__main__":
    main()
