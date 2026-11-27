"""Train and evaluate the three paired native answer-update arms on Sudoku development."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from evaluate_matched_sudoku import model_from_checkpoint, read_boards, sha
from experiments.decision_stability.answer_mixing import AnswerMixer
from experiments.decision_stability.clock import paired_binary_interval
from experiments.residual_reasoning.common import atomic, write
from models.ema import EMAHelper
from models.losses import ACTLossHead
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from probe_native_trm_branches import audit

ARMS = ("native", "scalar", "directional")
SEED, STEPS, BATCH, WARMUP = 11, 1024, 128, 64
MEANINGFUL = 0.02


def inputs_from(root):
    calibration, _ = audit(root / "dev-calibration/calibration.json", root / "dev-refiners/dev-results.json")
    weights = root / "checkpoint/checkpoint.pt"
    config = root / "checkpoint/all_config.yaml"
    if (sha(weights) != calibration["checkpoint_sha256"] or sha(config) != calibration["config_sha256"]
            or torch.cuda.get_device_name() != calibration["device"] or str(torch.__version__) != calibration["torch"]):
        raise ValueError("Expected the sealed native checkpoint, config and GH200 runtime")
    return SimpleNamespace(family="trm", weights=weights, config=config), calibration


def make_model(args, arm, device, batch_size):
    torch.manual_seed(SEED)
    with torch.device(device):
        model, config = model_from_checkpoint(args, device, batch_size=batch_size)
        if arm != "native":
            model.inner.answer_mixer = AnswerMixer(config["hidden_size"], arm)
    return model, config


def offered_batches(size):
    generator = np.random.default_rng(SEED)
    rows = np.concatenate([generator.permutation(size) for _ in range((STEPS * BATCH + size - 1) // size)])
    return rows[:STEPS * BATCH].reshape(STEPS, BATCH)


def make_batch(inputs, labels, rows):
    return {"inputs": inputs[rows], "labels": labels[rows], "puzzle_identifiers": torch.zeros_like(rows),
            "sample_rows": rows}


@torch.no_grad()
def verify_start(model, reference, batch):
    with torch.device(batch["inputs"].device):
        carry = model.initial_carry(batch)
        control = reference.initial_carry(batch)
    for _ in range(2):
        carry, observed = model(carry, batch)
        control, expected = reference(control, batch)
        if (not all(torch.equal(observed[key], expected[key]) for key in expected)
                or not torch.equal(carry.inner_carry.z_H, control.inner_carry.z_H)
                or not torch.equal(carry.inner_carry.z_L, control.inner_carry.z_L)):
            raise ValueError("Initial recurrence does not reproduce the frozen native control")


def train(root, output, arm, device):
    from adam_atan2 import AdamATan2

    args, calibration = inputs_from(root)
    recipe = yaml.safe_load(args.config.read_text())
    if not recipe["arch"].get("no_ACT_continue", True):
        raise ValueError("Expected the original single-Q ACT supervision")
    data = root / "train-only"
    cohort = json.loads((data / "cohort.json").read_text())
    prior = json.loads((root / "dev-refiners/dev-results.json").read_text())
    inputs, labels, ids, data_sha = read_boards(data, "train", "trm", slice(None), device)
    stride = cohort["augments"] + 1
    if (data_sha != prior["train_sha256"] or len(ids) != 33000
            or ids[::stride] != prior["train_ids"] or set(ids) & set(prior["ids"])):
        raise ValueError("Training rows differ from the sealed augmented stream")
    output.mkdir(exist_ok=True)
    arm_output = output / arm
    arm_output.mkdir()
    stream = offered_batches(len(ids))
    stream_sha = hashlib.sha256(stream.tobytes()).hexdigest()
    reuse = None
    if arm != "native":
        native_result = json.loads((output / "native/complete.json").read_text())
        masks = output / "native/act_schedule.npy"
        if (native_result["stream_sha256"] != stream_sha or native_result["train_sha256"] != data_sha
                or sha(masks) != native_result["act_schedule_sha256"]):
            raise ValueError("Native ACT schedule and offered stream differ")
        reuse = torch.from_numpy(np.load(masks, allow_pickle=False)).to(device)
        if reuse.shape != (STEPS, BATCH) or reuse.dtype != torch.bool:
            raise ValueError("Invalid native ACT schedule")
    model, cfg = make_model(args, arm, device, BATCH)
    if arm != "native":
        reference, _ = make_model(args, "native", device, BATCH)
        verify_start(model, reference, make_batch(inputs, labels, torch.as_tensor(stream[0], device=device)))
        del reference
    head = ACTLossHead(model, recipe["arch"]["loss"]["loss_type"]).train()
    ema = EMAHelper(mu=recipe["ema_rate"])
    ema.register(model)
    dense = AdamATan2(head.parameters(), lr=0.0, weight_decay=recipe["weight_decay"],
                      betas=(recipe["beta1"], recipe["beta2"]))
    sparse = CastedSparseEmbeddingSignSGD_Distributed(
        model.puzzle_emb.buffers(), lr=0.0, weight_decay=recipe["puzzle_emb_weight_decay"], world_size=1)
    torch.manual_seed(SEED)
    schedule, effective = [], []
    carry = None
    torch.cuda.reset_peak_memory_stats()
    begin = perf_counter()
    with (arm_output / "progress.jsonl").open("x", buffering=1) as log:
        for i, indices in enumerate(stream):
            rows = torch.as_tensor(indices, device=device)
            batch = make_batch(inputs, labels, rows)
            with torch.device(device):
                if carry is None:
                    carry = head.initial_carry(batch)
                carry, loss, _, _, _ = head(carry=carry, batch=batch, return_keys=[])
            (loss / BATCH).backward()
            step = i + 1
            scale = min(1.0, step / WARMUP)
            for optimizer, rate in ((dense, recipe["lr"]), (sparse, recipe["puzzle_emb_lr"])):
                for group in optimizer.param_groups:
                    group["lr"] = rate * scale
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            ema.update(model)
            if reuse is None:
                schedule.append(carry.halted.detach())
            else:
                carry.halted = reuse[i]
            effective.append(carry.current_data["sample_rows"])
            if step == 1 or step % 64 == 0:
                value = float(loss.detach()) / BATCH
                record = dict(step=step, loss=value, seconds=perf_counter() - begin)
                line = json.dumps(record, allow_nan=False)
                log.write(line + "\n")
                print(arm, line, flush=True)
    torch.cuda.synchronize()
    elapsed = perf_counter() - begin
    effective_rows = torch.stack(effective).cpu().numpy()
    effective_sha = hashlib.sha256(effective_rows.tobytes()).hexdigest()
    if reuse is None:
        masks = arm_output / "act_schedule.npy"
        np.save(masks, torch.stack(schedule).cpu().numpy(), allow_pickle=False)
    elif native_result["effective_sha256"] != effective_sha:
        raise ValueError("Actually supervised augmented rows differed from the native arm")
    checkpoint = arm_output / "checkpoint.pt"
    with atomic(checkpoint) as tmp:
        torch.save({key: ema.shadow.get(key, value).detach().cpu() for key, value in model.state_dict().items()},
                   tmp)
    write(arm_output / "complete.json", dict(
        arm=arm, steps=STEPS, batch=BATCH, seed=SEED, presentations=STEPS * BATCH,
        unique_supervised_rows=int(np.unique(effective_rows).size), source_sha256=sha(Path(__file__)),
        train_sha256=data_sha, stream_sha256=stream_sha, effective_sha256=effective_sha,
        act_schedule_sha256=sha(masks), initial_weights_sha256=calibration["checkpoint_sha256"],
        checkpoint_sha256=sha(checkpoint), parameters=sum(p.numel() for p in model.parameters()),
        forward_dtype=cfg["forward_dtype"], seconds=elapsed, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        dense_lr=recipe["lr"], sparse_lr=recipe["puzzle_emb_lr"], warmup=WARMUP))


@torch.inference_mode()
def predict(model, inputs, label, budget, device):
    batch = {"inputs": inputs[None], "puzzle_identifiers": inputs.new_zeros(1)}
    torch.cuda.synchronize()
    start = perf_counter()
    with torch.device(device):
        carry = model.initial_carry(batch)
    tokens, logits, times = [], [], []
    for _ in range(model.config.halt_max_steps):
        carry, out = model(carry, batch)
        score = out["logits"]
        logits.append(score)
        tokens.append(score.argmax(-1))
        torch.cuda.synchronize()
        times.append(perf_counter() - start)
    predictions = torch.cat(tokens)
    if not bool(torch.cat(logits).isfinite().all()):
        raise ValueError("Nonfinite development logits")
    exact = predictions.eq(label).all(-1).tolist()
    accept = sum(t <= budget for t in times)
    return dict(exact=exact, seconds=times, accepted=accept, timed_exact=accept > 0 and exact[accept - 1],
                timed_no_readout=accept == 0, timed_cap=accept == len(times))


def evaluate(root, output, device):
    args, calibration = inputs_from(root)
    path = root / "dev-calibration/fork_states.pt"
    if sha(path) != calibration["bank_sha256"]:
        raise ValueError("Saved development bank checksum mismatch")
    bank = torch.load(path, map_location="cpu", weights_only=True)
    traces = calibration["traces"]
    ids = [row["id"] for row in traces]
    if len(ids) != 256 or bank["ids"] != ids:
        raise ValueError("Unexpected development IDs")
    budget = median(row["seconds"][11] for row in traces)
    summaries = {arm: json.loads((output / arm / "complete.json").read_text()) for arm in ARMS}
    if any(len({x[key] for x in summaries.values()}) != 1 for key in (
            "steps", "batch", "seed", "train_sha256", "stream_sha256", "effective_sha256", "act_schedule_sha256",
            "initial_weights_sha256", "source_sha256", "dense_lr", "sparse_lr", "warmup")):
        raise ValueError("Architectural arms differ in training exposure or protocol")
    models = {}
    for arm in ARMS:
        model, _ = make_model(args, arm, device, 1)
        checkpoint = output / arm / "checkpoint.pt"
        if sha(checkpoint) != summaries[arm]["checkpoint_sha256"]:
            raise ValueError("Learned checkpoint checksum mismatch")
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        models[arm] = model.eval()
    inputs, labels = (bank[key].to(device) for key in ("inputs", "labels"))
    for i in range(4):
        for model in models.values():
            predict(model, inputs[i], labels[i], budget, device)
    rows = []
    for i, key in enumerate(ids):
        phase = i % len(ARMS)
        order = ARMS[phase:] + ARMS[:phase]
        rows.append(dict(id=key, arms={arm: predict(models[arm], inputs[i], labels[i], budget, device)
                                      for arm in order}))
    timed = {arm: [row["arms"][arm]["timed_exact"] for row in rows] for arm in ARMS}
    paired = {f"{arm}_vs_{control}": paired_binary_interval(timed[arm], timed[control])
              for arm, control in (("scalar", "native"), ("directional", "native"), ("directional", "scalar"))}
    results = {arm: dict(
        timed_exact=sum(timed[arm]),
        fixed_12=sum(r["arms"][arm]["exact"][11] for r in rows),
        fixed_16=sum(r["arms"][arm]["exact"][15] for r in rows),
        median_accepted=median(r["arms"][arm]["accepted"] for r in rows),
        cap_hits=sum(r["arms"][arm]["timed_cap"] for r in rows),
        no_readout=sum(r["arms"][arm]["timed_no_readout"] for r in rows),
        median_depth_12_seconds=median(r["arms"][arm]["seconds"][11] for r in rows))
        for arm in ARMS}
    promote = all(paired[f"directional_vs_{arm}"]["mean"] >= MEANINGFUL
                  and paired[f"directional_vs_{arm}"]["lower"] > 0 for arm in ("native", "scalar"))
    write(output / "evaluation.json", dict(budget_seconds=budget, board_count=len(ids), summaries=summaries,
                                          results=results, paired=paired, meaningful=MEANINGFUL,
                                          directional_promotes=promote, rows=rows))
    print(json.dumps(results), flush=True)


def preflight(root, output, device):
    global STEPS, BATCH
    STEPS, BATCH = 4, 8
    for arm in ARMS:
        train(root, output, arm, device)
    args, calibration = inputs_from(root)
    bank = torch.load(root / "dev-calibration/fork_states.pt", map_location="cpu", weights_only=True)
    x, y = (bank[key][0].to(device) for key in ("inputs", "labels"))
    results = {}
    for arm in ARMS:
        model, _ = make_model(args, arm, device, 1)
        state = torch.load(output / arm / "checkpoint.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        if arm != "native" and not torch.count_nonzero(state["inner.answer_mixer.gate.weight"]):
            raise ValueError(f"The {arm} gate did not update under ACT training")
        results[arm] = predict(model, x, y, median(row["seconds"][11] for row in calibration["traces"]), device)
    write(output / "preflight.json", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=(*ARMS, "evaluate", "preflight"), required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.arm == "evaluate":
        evaluate(args.root, args.output, device)
    elif args.arm == "preflight":
        preflight(args.root, args.output, device)
    else:
        train(args.root, args.output, args.arm, device)
