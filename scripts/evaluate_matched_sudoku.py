"""Run native released Sudoku checkpoints on a common, bounded dev cohort."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_boards(root, split, family, selection, device):
    manifest = json.loads((root / "dataset.json").read_text())
    path = root / f"{split}.npz"
    data_sha = sha(path)
    if data_sha != manifest["splits"][split]["sha256"]:
        raise ValueError("Sudoku split checksum mismatch")
    shift = int(family != "residual")
    with np.load(path, allow_pickle=False) as data:
        x = torch.from_numpy(data["x"][selection]).to(device, dtype=torch.long) + shift
        y = torch.from_numpy(data["y"][selection]).to(device, dtype=torch.long) + 2 * shift
        ids = data["ids"][selection].tolist()
    if not ids or min(x.shape[1], y.shape[1]) != 81:
        raise ValueError("Expected original Sudoku givens and full-board labels")
    return x, y, ids, data_sha


def score_residual(model, answer, label):
    return model.decode(answer).argmax(-1).eq(label)


def model_from_checkpoint(args, device, *, batch_size=1):
    if args.family == "residual":
        from experiments.residual_reasoning.train import from_checkpoint

        return from_checkpoint(args.weights, device)
    weights = torch.load(args.weights, map_location="cpu", weights_only=True)
    for prefix in ("_orig_mod.", "model."):
        if weights and all(key.startswith(prefix) for key in weights):
            weights = {key[len(prefix):]: value for key, value in weights.items()}
    if weights["inner.puzzle_emb.weights"].shape[0] != 1:
        raise ValueError("Expected the one-identifier Sudoku checkpoint")
    cfg = yaml.safe_load(args.config.read_text())["arch"]
    cfg = {key: value for key, value in cfg.items() if key not in ("name", "loss")}
    cfg.update(batch_size=batch_size, seq_len=81, vocab_size=11, num_puzzle_identifiers=1)
    if args.family == "trm":
        from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1 as Native
    else:
        from models.fixed_point_reasoning.fprm import FixedPointReasoningModel_ACTV1 as Native
        cfg.update(max_iter_eval=35000, stepsize_decay_eval=0.997, decay_patience=10)
    model = Native(cfg).to(device).eval()
    model.load_state_dict(weights, strict=True)
    if args.family == "fprm":
        model.set_num_iters()
    return model, cfg


def evaluate(model, inputs, label, device, max_seconds, autocast):
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    batch = {"inputs": inputs[None], "puzzle_identifiers": torch.zeros(1, dtype=torch.long, device=device)}
    precision = torch.autocast(device.type, dtype=torch.bfloat16, cache_enabled=False) if autocast else nullcontext()
    with torch.inference_mode(), torch.device(device), precision:
        synchronize()
        start = time.perf_counter()
        carry = model.initial_carry(batch)
        elapsed, predictions = [], []
        while not elapsed or elapsed[-1] < max_seconds:
            carry, output = model(carry, batch)
            prediction = output["logits"].argmax(-1)
            synchronize()
            elapsed.append(time.perf_counter() - start)
            predictions.append(bool(prediction.eq(label).all()))
            if carry.halted.item():
                break
    return dict(seconds=elapsed, correct=predictions, native_halted=bool(carry.halted.item()))


def evaluate_residual(model, inputs, label, device, max_seconds):
    times, outcomes = [], []
    with torch.no_grad():
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        encoded = model.encode(inputs[None])
        answer, workspace = model.start(encoded)
    for depth in range(17):
        if depth:
            answer, workspace = model.step(encoded, answer, workspace, train=False)
        with torch.no_grad():
            correct = score_residual(model, answer, label).all()
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - start)
        outcomes.append(bool(correct))
        if times[-1] >= max_seconds:
            break
    return dict(seconds=times, correct=outcomes, native_halted=False,
                normal_readout_index=4 if len(times) > 4 else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "config", "weights", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--family", choices=("trm", "fprm", "residual"), required=True)
    p.add_argument("--source-rev", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--split", choices=("dev", "test"), default="test")
    p.add_argument("--limit", type=int, default=128)
    p.add_argument("--max-seconds", type=float, default=0.75)
    args = p.parse_args()
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    x, y, ids, data_sha = read_boards(args.data, args.split, args.family, slice(args.limit), device)
    torch.manual_seed(11)
    model, cfg = model_from_checkpoint(args, device)
    def predict(inputs, labels, cap):
        if args.family == "residual":
            return evaluate_residual(model, inputs, labels, device, cap)
        result = evaluate(model, inputs, labels, device, cap, args.family == "fprm")
        result["normal_readout_index"] = len(result["correct"]) - 1 if result["native_halted"] else None
        return result

    predict(x[0], y[0], min(0.03, args.max_seconds))
    rows = []
    for i, key in enumerate(ids):
        torch.manual_seed(11 + i)
        rows.append(dict(id=key, **predict(x[i], y[i], args.max_seconds)))
    payload = dict(family=args.family, source=args.source_rev, config=cfg, data_sha256=data_sha, split=args.split,
                   checkpoint_sha256=sha(args.weights), timing_cap_s=args.max_seconds,
                   device=torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
                   torch=str(torch.__version__), rows=rows)
    with args.output.open("x") as handle:
        json.dump(payload, handle, allow_nan=False)
    print(args.family, len(rows), "boards", sum(row["native_halted"] for row in rows), "native stops")


if __name__ == "__main__":
    main()
