"""PH1. Run with VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1."""
import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SEEDS = (31, 47, 83)
HERE = Path(__file__).resolve().parent
D = 32
BATCH = 64


def digest(data):
    return hashlib.sha256(data).hexdigest()


def make_data():
    rng = np.random.default_rng(211)
    perms = list(itertools.permutations(range(4)))
    eye = np.eye(4, dtype=np.float32)
    rows = {"train": [], "dev": []}
    ids = {"train": [], "dev": [], "reserved": []}
    for g, second in enumerate(perms):
        for rank, f_index in enumerate(rng.permutation(24)):
            f = int(f_index)
            split = "train" if rank < 16 else "dev" if rank < 20 else "reserved"
            ids[split].append([f, g])
            if split == "reserved":
                continue
            first = perms[f]
            for q in range(4):
                x = np.concatenate([eye[list(first)].ravel(), eye[list(second)].ravel(), eye[q]])
                y = second[first[q]]
                # Independent direct executor operating only on supplied features.
                intermediate = x[:16].reshape(4, 4)[x[32:].argmax()].argmax()
                assert x[16:32].reshape(4, 4)[intermediate].argmax() == y
                rows[split].append((x, y, (f, g, q)))
    assert len({tuple(p) for split in ids.values() for p in split}) == 576
    data = {k: (np.stack([r[0] for r in v]), np.array([r[1] for r in v]),
                np.array([r[2] for r in v])) for k, v in rows.items()}
    return ids, data


def initialize(seed):
    rng = np.random.default_rng(seed)
    def matrix(m, n):
        return rng.normal(0, 1 / np.sqrt(m), (m, n)).astype(np.float32)
    return dict(E=matrix(36, 2 * D), e=np.zeros(2 * D, np.float32),
                U=matrix(D, D), V=matrix(D, D), b=np.zeros(D, np.float32),
                R=matrix(2 * D, 4), c=np.zeros(4, np.float32))


def forward(p, x, fresh):
    # Copy the two role states into dense row order before each matrix product.
    h = np.tanh(x @ p["E"] + p["e"]).reshape(-1, 2, D)
    states = [h]
    for t in range(2):
        peer = states[t if fresh else 0][:, ::-1].reshape(-1, D)
        h = np.tanh(h.reshape(-1, D) @ p["U"] + peer @ p["V"] + p["b"]).reshape(-1, 2, D)
        states.append(h)
    return h.reshape(-1, 2 * D) @ p["R"] + p["c"], states


def loss_grad(p, x, y, fresh):
    logits, hs = forward(p, x, fresh)
    shifted = logits - logits.max(axis=1, keepdims=True)
    prob = np.exp(shifted)
    prob /= prob.sum(axis=1, keepdims=True)
    loss = -np.log(prob[np.arange(len(y)), y]).mean()
    prob[np.arange(len(y)), y] -= 1
    dl = prob / len(y)
    grads = {k: np.zeros_like(v) for k, v in p.items()}
    grads["R"] = hs[-1].reshape(-1, 2 * D).T @ dl
    grads["c"] = dl.sum(axis=0)
    dh = [np.zeros_like(h) for h in hs]
    dh[2] = (dl @ p["R"].T).reshape(-1, 2, D)
    for t in (1, 0):
        source = t if fresh else 0
        local = (dh[t+1] * (1 - hs[t+1] ** 2)).reshape(-1, D)
        grads["U"] += hs[t].reshape(-1, D).T @ local
        grads["V"] += hs[source][:, ::-1].reshape(-1, D).T @ local
        grads["b"] += local.sum(axis=0)
        dh[t] += (local @ p["U"].T).reshape(-1, 2, D)
        dh[source] += (local @ p["V"].T).reshape(-1, 2, D)[:, ::-1]
    de = (dh[0] * (1 - hs[0] ** 2)).reshape(-1, 2 * D)
    grads["E"] = x.T @ de
    grads["e"] = de.sum(axis=0)
    return loss, grads


def gradient_check(x, y):
    errors = {}
    for fresh in (True, False):
        p = {k: v.astype(np.float64) for k, v in initialize(0).items()}
        x64 = x[:7].astype(np.float64)
        labels = y[:7]
        _, g = loss_grad(p, x64, labels, fresh)
        for k in p:
            # Test the largest derivative of each trained tensor, including recurrent peer V.
            i = np.unravel_index(np.abs(g[k]).argmax(), g[k].shape)
            initial = p[k][i]
            p[k][i] = initial + 1e-5
            high = loss_grad(p, x64, labels, fresh)[0]
            p[k][i] = initial - 1e-5
            low = loss_grad(p, x64, labels, fresh)[0]
            p[k][i] = initial
            err = abs((high - low) / 2e-5 - g[k][i])
            errors[f"{fresh}/{k}"] = float(err)
            assert err < 1e-7 and abs(g[k][i]) > 1e-8
    return errors


def train(state, x, y, stop):
    p = state["p"]
    fresh = state["arm"] == "dynamic"
    start = time.perf_counter()
    for step in range(state["step"] + 1, stop + 1):
        idx = state["rng"].integers(len(x), size=BATCH)
        _, gs = loss_grad(p, x[idx], y[idx], fresh)
        for k, g in gs.items():
            m, v = state["mom"][k]
            m *= 0.9
            m += 0.1 * g
            v *= 0.999
            v += 0.001 * g * g
            p[k] -= (0.003 / (1 - 0.9 ** step)) * m / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
        if step % 250 == 0:
            logits = forward(p, x, fresh)[0]
            shifted = logits - logits.max(axis=1, keepdims=True)
            loss = (np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(y)), y]).mean()
            state["curve"].append(dict(step=step, exact=float((logits.argmax(1) == y).mean()), loss=float(loss)))
    state["step"] = stop
    state["seconds"] += time.perf_counter() - start


def timings(p, x, fresh):
    ans = {}
    for batch in (1, 64):
        calls = 50
        inp = x[:batch]
        for _ in range(10):
            forward(p, inp, fresh)
        samples = []
        for _ in range(9):
            begin = time.perf_counter_ns()
            for _ in range(calls):
                forward(p, inp, fresh)
            samples.append((time.perf_counter_ns() - begin) / (1000 * calls))
        ans[str(batch)] = float(np.median(samples))
    return ans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    assert all(os.environ.get(k) == "1" for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"))
    for folder in (args.out, args.weights):
        folder.mkdir(parents=True, exist_ok=True)
    ids, data = make_data()
    x, y, _ = data["train"]
    errors = gradient_check(x, y)
    manifest = {"groups": ids, "fold_sha256": {k: digest(a.tobytes()+b.tobytes()+c.tobytes()) for k,(a,b,c) in data.items()}}
    (args.out/"split.json").write_text(json.dumps(manifest, separators=(",", ":"))+"\n")
    states = []
    for seed in SEEDS:
        for arm in ("dynamic", "static"):
            p = initialize(seed)
            zeros = {k: [np.zeros_like(v), np.zeros_like(v)] for k,v in p.items()}
            state = dict(p=p, arm=arm, seed=seed, step=0, seconds=0, mom=zeros,
                         rng=np.random.default_rng(seed+1729), curve=[])
            train(state, x, y, 1500)
            states.append(state)
    extend = any(s["curve"][-1]["exact"] - s["curve"][-3]["exact"] >= 0.02 for s in states)
    if extend:
        for state in states:
            train(state, x, y, 3000)
    xd, yd, keys = data["dev"]
    rows = []
    runs = []
    paired = {}
    for s in states:
        p, arm, seed = s["p"], s["arm"], s["seed"]
        fresh = arm == "dynamic"
        pred = forward(p, xd, fresh)[0].argmax(1)
        stale = forward(p, xd, False)[0].argmax(1) if fresh else pred
        paired[(seed, arm)] = pred
        file = args.weights / f"{arm}-{seed}.npz"
        np.savez(file, **p)
        rows.extend([seed, arm, *k, label, a, b] for k,label,a,b in zip(keys, yd, pred, stale))
        primary = keys[:,1] != 0
        runs.append(dict(seed=seed, arm=arm, endpoint=s["step"], training_seconds=s["seconds"],
                         checkpoint_sha256=digest(file.read_bytes()), curve=s["curve"], latency_us=timings(p, xd, fresh),
                         primary_correct=int((pred[primary]==yd[primary]).sum()),
                         negative_correct=int((pred[~primary]==yd[~primary]).sum()),
                         primary_stale_correct=int((stale[primary]==yd[primary]).sum())))
    diffs = []
    pairs = []
    for seed in SEEDS:
        a = paired[(seed,"dynamic")][primary] == yd[primary]
        b = paired[(seed,"static")][primary] == yd[primary]
        diffs.append((a.astype(float)-b).reshape(-1,4).mean(axis=1))
        pairs.append(dict(seed=seed, gains=int((a&~b).sum()), losses=int((b&~a).sum())))
    group_diff = np.mean(diffs, axis=0)
    boot = np.random.default_rng(713).choice(group_diff, (2000,len(group_diff)),replace=True).mean(axis=1)
    with (args.out/"predictions.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["seed","arm","F","G","query","target","prediction","stale_prediction"])
        writer.writerows(rows)
    p = states[0]["p"]
    summary = dict(source_sha256=digest(Path(__file__).read_bytes()), protocol_sha256=digest((HERE/"PROTOCOL.md").read_bytes()),
                   git_base=subprocess.check_output(["git","rev-parse","HEAD"], cwd=HERE,text=True).strip(),
                   python=sys.version, numpy=np.__version__, platform=platform.platform(), machine=platform.machine(),
                   peak_process_rss_as_reported=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   float_dtype="float32", inference_macs=36*2*D+2*2*2*D**2+2*D*4,
                   parameters=sum(v.size for v in p.values()), gradient_max_abs_error=max(errors.values()),
                   work_notes="MACs for dense forward per query, excludes bias, tanh, memory and argmax. Latency includes NumPy call overhead and both rounds.",
                   extended=extend, primary_queries=int(primary.sum()), primary_groups=int(primary.sum()//4),
                   negative_queries=int((~primary).sum()), primary_delta=float(group_diff.mean()),
                   group_percentile_95=np.quantile(boot,[.025,.975]).tolist(), pairs=pairs, runs=runs)
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({k:summary[k] for k in ["extended","primary_delta","group_percentile_95","pairs"]}),flush=True)


if __name__ == "__main__":
    main()
