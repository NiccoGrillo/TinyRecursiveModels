"""PH1 serial control, two learned ordered reads over the same supplied tables."""
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.parallel_hypotheses.study import BATCH, SEEDS, make_data

HERE = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class SerialLookup(nn.Module):
    def __init__(self):
        super().__init__()
        self.address = nn.Parameter(torch.randn(4, 4) * 0.01)
        self.selector = nn.Parameter(torch.randn(2, 2) * 0.01)

    def forward(self, x):
        tables = x[:, :32].reshape(-1, 2, 4, 4)
        state = x[:, 32:]
        addressing = self.address.softmax(dim=-1)
        selections = self.selector.softmax(dim=-1)
        for selection in selections:
            address = state @ addressing
            reads = (tables * address[:, None, :, None]).sum(dim=2)
            state = (reads * selection[None, :, None]).sum(dim=1)
        return state


def nll(probabilities, targets):
    chosen = probabilities.gather(1, targets[:, None])
    return -torch.log(chosen.clamp_min(torch.finfo(chosen.dtype).tiny)).mean()


@torch.inference_mode()
def timings(model, tensor):
    measurements = {}
    for batch in (1, 64):
        samples = []
        x = tensor[:batch]
        for _ in range(10):
            model(x)
        for _ in range(9):
            start = time.perf_counter_ns()
            for _ in range(50):
                model(x)
            samples.append((time.perf_counter_ns() - start) / 50000)
        measurements[str(batch)] = float(np.median(samples))
    return measurements


def run(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _, data = make_data()
    x, y, _ = data["train"]
    xd, yd, keys = data["dev"]
    x, y, xd = map(torch.from_numpy, (x, y, xd))
    primary = keys[:, 1] != 0
    runs, records = [], []
    args.out.mkdir(exist_ok=True, parents=True)
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = SerialLookup()
        opt = torch.optim.Adam(model.parameters(), lr=0.03)
        rng = np.random.default_rng(seed + 1729)
        order_hash = hashlib.sha256()
        curve = []
        begin = time.perf_counter()
        for step in range(1, 1501):
            idx = rng.integers(len(x), size=BATCH)
            order_hash.update(idx.astype("<i8").tobytes())
            opt.zero_grad(set_to_none=True)
            loss = nll(model(x[idx]), y[idx])
            loss.backward()
            opt.step()
            if step % 250 == 0:
                with torch.inference_mode():
                    prob = model(x)
                    curve.append(dict(step=step, exact=(prob.argmax(1)==y).float().mean().item(),
                                      nll=nll(prob, y).item()))
        seconds = time.perf_counter() - begin
        with torch.inference_mode():
            pred = model(xd).argmax(1).numpy()
            matrices = {name: p.detach().tolist() for name, p in model.named_parameters()}
            geometry = model.address.softmax(-1).tolist()
            policy = model.selector.softmax(-1).tolist()
        records.extend([seed, *key, int(reference), int(value)] for key, reference, value in zip(keys, yd, pred))
        runs.append(dict(seed=seed, curve=curve, train_seconds=seconds, weights=matrices,
                         batch_indices_sha256=order_hash.hexdigest(), address=geometry, selector=policy,
                         primary_correct=int((pred[primary] == yd[primary]).sum()),
                         primary_exact_groups=int((pred[primary] == yd[primary]).reshape(-1, 4).all(1).sum()),
                         negative_correct=int((pred[~primary] == yd[~primary]).sum()),
                         latency_us=timings(model, xd)))
    with (args.out/"predictions.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(("seed","F","G","query","target","prediction"))
        writer.writerows(records)
    summary = dict(parameters=20, dense_macs_per_query=112, elementary_work=
                   "Per round: 4x4 address transform (16 multiplications, 12 adds), two 4x4 table reads (32 multiplications, 24 adds), two-way mix (8 multiplications, 4 adds). Two rounds: 112 multiplications and 80 additions. Shared row softmax 4x4 and two round-selector softmaxes 2x2, argmax and memory excluded from MACs. Wall timing includes softmax and both rounds, excludes argmax and loading.",
                   primary_queries=int(primary.sum()), negative_queries=int((~primary).sum()), primary_groups=int(primary.sum()//4),
                   torch=torch.__version__, numpy=np.__version__, python=sys.version, platform=platform.platform(),
                   device="cpu", dtype="float32", cpu_threads=torch.get_num_threads(), gpu_hours=0,
                   source_sha256=sha(Path(__file__).read_bytes()), protocol_sha256=sha((HERE/"PROTOCOL.md").read_bytes()),
                   ph1_source_sha256=sha((HERE.parent/"parallel_hypotheses"/"study.py").read_bytes()),
                   base_revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=HERE,text=True).strip(),
                   splits={k: sha(a.tobytes()+b.tobytes()+c.tobytes()) for k,(a,b,c) in data.items()}, runs=runs)
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print([(r["seed"],r["primary_correct"], r["negative_correct"], r["latency_us"]) for r in runs])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE/"results")
    run(parser.parse_args())
