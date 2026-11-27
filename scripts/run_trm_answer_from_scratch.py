"""Bounded end-to-end native TRM answer-mixing comparison on Sudoku development."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch
import yaml

from evaluate_matched_sudoku import read_boards, sha
from experiments.decision_stability.answer_mixing import AnswerMixer
from experiments.decision_stability.clock import paired_binary_interval
from experiments.residual_reasoning.common import atomic, write
from models.ema import EMAHelper
from models.losses import ACTLossHead
from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1 as Native,
    TinyRecursiveReasoningModel_ACTV1Carry as Carry,
    TinyRecursiveReasoningModel_ACTV1InnerCarry as State,
)
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from probe_native_trm_branches import audit
from run_trm_answer_mixing import ARMS, make_batch, predict, verify_start

SEED, BATCH, FIRST, SECOND, INTERVAL, TUNE = 11, 768, 4096, 8192, 512, 512
STAGE_SECONDS = (2880, 5760)
MIN_GAIN = 6


def read(path):
    return json.loads(path.read_text())


def inputs(root, device, train=False):
    calibration, _ = audit(root / "dev-calibration/calibration.json", root / "dev-refiners/dev-results.json")
    config = root / "checkpoint/all_config.yaml"
    bank_path = root / "dev-calibration/fork_states.pt"
    if (sha(config) != calibration["config_sha256"] or sha(bank_path) != calibration["bank_sha256"]
            or calibration["device"] != torch.cuda.get_device_name() or calibration["torch"] != str(torch.__version__)):
        raise ValueError("Expected the original recipe, development bank and GH200 runtime")
    recipe = yaml.safe_load(config.read_text())
    bank = torch.load(bank_path, map_location="cpu", weights_only=True)
    if bank["ids"] != [r["id"] for r in calibration["traces"]] or len(bank["ids"]) != 256:
        raise ValueError("Wrong development cohort")
    if train:
        prior = read(root / "dev-refiners/dev-results.json")
        x, y, ids, digest = read_boards(root / "train-only", "train", "trm", slice(None), device)
        stride = 1 + read(root / "train-only/cohort.json")["augments"]
        if (len(ids) != 33000 or ids[::stride] != prior["train_ids"] or digest != prior["train_sha256"]
                or set(ids) & set(bank["ids"])):
            raise ValueError("Wrong augmented training cohort")
        return recipe, bank, (x, y, digest)
    return recipe, bank, calibration


def initialize(recipe, arm, device, batch):
    """Sample all shared parameters before an arm-specific module consumes any RNG."""
    torch.manual_seed(SEED)
    cfg = {k: v for k, v in recipe["arch"].items() if k not in ("name", "loss")}
    cfg.update(batch_size=batch, seq_len=81, vocab_size=11, num_puzzle_identifiers=1)
    with torch.device(device):
        model = Native(cfg)
        if arm != "native":
            model.inner.answer_mixer = AnswerMixer(cfg["hidden_size"], arm)
    return model


def shared_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if not name.startswith("inner.answer_mixer."):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def row_stream(size, stop, batch):
    rng = np.random.default_rng(SEED)
    rows = np.concatenate([rng.permutation(size) for _ in range((stop * batch + size - 1) // size)])
    return rows[:stop * batch].reshape(stop, batch)


def save(path, data):
    with atomic(path) as temporary:
        torch.save(data, temporary)


def pack_carry(carry):
    return (carry.inner_carry.z_H, carry.inner_carry.z_L, carry.steps, carry.halted, carry.current_data)


def unpack_carry(packed, device):
    h, l, steps, halted, batch = packed
    return Carry(State(h.to(device), l.to(device)), steps.to(device), halted.to(device),
                 {key: value.to(device) for key, value in batch.items()})


def rates(recipe, arm, choice, native_choice=0):
    """Each arm has two equally sized tuning trials."""
    base = recipe["lr"] * (2 ** native_choice)
    return (base * (2 ** choice), None) if arm == "native" else (base, base * (4 ** choice))


def fit(args, arm, directory, start, stop, interval, max_seconds, choice, previous=None, reference=None):
    from adam_atan2 import AdamATan2

    recipe, _, (x, y, data_sha) = inputs(args.root, args.device, train=True)
    directory.mkdir(parents=True)
    model = initialize(recipe, arm, args.device, args.batch)
    identity = shared_digest(model)
    stream = row_stream(len(x), SECOND, args.batch)[start:stop]
    stream_sha = hashlib.sha256(stream.tobytes()).hexdigest()
    reused = None
    reference_records = {}
    if reference is not None:
        ref = read(reference / "complete.json")
        masks = reference / "reset.npy"
        if (ref["shared_init"] != identity or ref["data_sha256"] != data_sha or ref["start"] != start
                or sha(masks) != ref["reset_sha256"]):
            raise ValueError("Mismatched native initialization, data or reset trace")
        reused = np.load(masks, allow_pickle=False)
        if reused.dtype != bool or reused.shape != (ref["end"] - start, args.batch):
            raise ValueError("Malformed native reset trace")
        stream = stream[:len(reused)]
        reference_records = {r["step"]: r for r in ref["snapshots"]}
    if arm != "native":
        reference_model = initialize(recipe, "native", args.device, args.batch).eval()
        verify_start(model.eval(), reference_model, make_batch(
            x, y, torch.as_tensor(stream[0, : min(args.batch, 2)], device=args.device)))
        del reference_model
    model.train()
    head = ACTLossHead(model, recipe["arch"]["loss"]["loss_type"])
    native_choice = 0 if reference is None or args.job == "preflight" else read(reference.parent / "tuning.json")["choice"]
    base_lr, mix_lr = rates(recipe, arm, choice, native_choice)
    mix = getattr(model.inner, "answer_mixer", None)
    mixer_params = list(mix.parameters()) if mix is not None else []
    mixer_keys = {id(p) for p in mixer_params}
    groups = [dict(params=[p for p in model.parameters() if id(p) not in mixer_keys], base_lr=base_lr)]
    if mix is not None:
        groups.append(dict(params=mixer_params, base_lr=mix_lr))
    dense = AdamATan2(groups, lr=0.0, weight_decay=recipe["weight_decay"],
                      betas=(recipe["beta1"], recipe["beta2"]))
    sparse = CastedSparseEmbeddingSignSGD_Distributed(
        model.puzzle_emb.buffers(), lr=0.0, weight_decay=recipe["puzzle_emb_weight_decay"], world_size=1)
    ema = EMAHelper(mu=recipe["ema_rate"])
    ema.register(model)
    torch.manual_seed(SEED)
    active, carry = 0.0, None
    if previous is not None:
        state = torch.load(previous / "resume.pt", map_location="cpu", weights_only=True)
        if (state["step"] != start or state["shared_init"] != identity or state["choice"] != choice
                or state["data_sha256"] != data_sha or state["arm"] != arm):
            raise ValueError("Resume provenance mismatch")
        model.load_state_dict(state["model"])
        dense.load_state_dict(state["dense"])
        sparse.load_state_dict(state["sparse"])
        ema.load_state_dict({k: v.to(args.device) for k, v in state["ema"].items()})
        carry = unpack_carry(state["carry"], args.device)
        active = state["seconds"]
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
    forward = head if args.eager else torch.compile(head, dynamic=False)
    snapshots, recorded, effective = [], [], []
    segment = []
    batch_losses = []
    end = start
    torch.cuda.reset_peak_memory_stats()
    begin = perf_counter()
    with (directory / "training.jsonl").open("x", buffering=1) as log:
        for i, indices in enumerate(stream):
            step = start + i + 1
            batch = make_batch(x, y, torch.as_tensor(indices, device=args.device))
            with torch.device(args.device):
                if carry is None:
                    carry = head.initial_carry(batch)
                carry, loss, metrics, _, _ = forward(carry=carry, batch=batch, return_keys=[])
            (loss / args.batch).backward()
            scale = min(1.0, step / int(recipe["lr_warmup_steps"]))
            for group in dense.param_groups:
                group["lr"] = group["base_lr"] * scale
            sparse.param_groups[0]["lr"] = recipe["puzzle_emb_lr"] * scale
            dense.step()
            sparse.step()
            dense.zero_grad(set_to_none=True)
            sparse.zero_grad(set_to_none=True)
            ema.update(model)
            if reused is None:
                recorded.append(carry.halted)
            else:
                carry.halted = torch.as_tensor(reused[i], device=args.device)
            segment.append(carry.current_data["sample_rows"])
            batch_losses.append(torch.stack((loss.detach(), metrics["lm_loss"], metrics["q_halt_loss"])).float())
            end = step
            if step % 64 == 0 or i + 1 == len(stream):
                torch.cuda.synchronize()
                wall = active + perf_counter() - begin
                values = (torch.stack(batch_losses).mean(0) / args.batch).tolist()
                record = dict(step=step, supervised_examples=step * args.batch, gpu_train_seconds=wall,
                              loss=values[0], lm_loss=values[1], q_halt_loss=values[2],
                              native_lr=base_lr * scale, mixer_lr=None if mix_lr is None else mix_lr * scale)
                print(arm, json.dumps(record, allow_nan=False), flush=True)
                log.write(json.dumps(record, allow_nan=False) + "\n")
                batch_losses.clear()
                if step % interval == 0 or i + 1 == len(stream) or wall >= max_seconds:
                    active = wall
                    effective.append(torch.stack(segment).cpu().numpy())
                    segment.clear()
                    digest = hashlib.sha256(np.concatenate(effective).tobytes()).hexdigest()
                    if step in reference_records and digest != reference_records[step]["supervised_sha256"]:
                        raise ValueError(f"Supervised stream differs at step {step}")
                    path = directory / f"step-{step:05d}.pt"
                    save(path, {k: ema.shadow.get(k, v).detach().cpu() for k, v in model.state_dict().items()})
                    snapshots.append(dict(step=step, supervised_examples=step * args.batch,
                                          gpu_train_seconds=active, file=path.name,
                                          checkpoint_sha256=sha(path), supervised_sha256=digest))
                    if active >= max_seconds:
                        break
                    begin = perf_counter()
    if end == start:
        raise ValueError("Training received no effective updates")
    masks_sha = None
    if reference is None:
        mask_path = directory / "reset.npy"
        np.save(mask_path, torch.stack(recorded).cpu().numpy(), allow_pickle=False)
        masks_sha = sha(mask_path)
    if not directory.name.startswith("trial-"):
        save(directory / "resume.pt", dict(step=end, arm=arm, choice=choice, seconds=active, model=model.state_dict(),
                                          ema=ema.state_dict(), dense=dense.state_dict(), sparse=sparse.state_dict(),
                                          carry=pack_carry(carry), cpu_rng=torch.get_rng_state(),
                                          cuda_rng=torch.cuda.get_rng_state(), shared_init=identity, data_sha256=data_sha))
    write(directory / "complete.json", dict(arm=arm, start=start, end=end, target=stop, choice=choice,
          batch=args.batch, seed=SEED, gpu_train_seconds=active, budget_cut=end < stop,
          base_lr=base_lr, mixer_lr=mix_lr, shared_init=identity, stream_sha256=stream_sha,
          reset_sha256=masks_sha, data_sha256=data_sha, snapshots=snapshots,
          peak_cuda_bytes=torch.cuda.max_memory_allocated(), source_sha256=sha(Path(__file__))))


@torch.inference_mode()
def fixed_depth(model, bank, device):
    mixer = getattr(model.inner, "answer_mixer", None)
    if mixer is not None:
        mixer.diagnostics = []
    per_depth, blanks, total = [0] * 16, 0, 0
    for x, y in zip(bank["inputs"].split(32), bank["labels"].split(32), strict=True):
        x, y = x.to(device), y.to(device)
        batch = dict(inputs=x, puzzle_identifiers=x.new_zeros(len(x)))
        carry = model.initial_carry(batch)
        for depth in range(16):
            carry, output = model(carry, batch)
            scores = output["logits"]
            if not bool(scores.isfinite().all()):
                raise ValueError("Nonfinite development readout")
            right = scores.argmax(-1).eq(y)
            per_depth[depth] += int(right.all(-1).sum())
        blank = x.eq(1)
        blanks += int((right & blank).sum())
        total += int(blank.sum())
    activity = {}
    if mixer is not None:
        g, edit, update, changed, n = torch.stack(mixer.diagnostics).sum(0).tolist()
        activity = dict(mean_abs_gate=g / n, correction_to_native_update=edit / max(update, 1e-30),
                        fraction_changed_post_cast=changed / n)
        mixer.diagnostics = None
    return dict(exact_by_depth=per_depth, exact_12=per_depth[11], exact_16=per_depth[-1],
                blank_accuracy_16=blanks / total, **activity)


def load_snapshot(model, directory, snapshot):
    path = directory / snapshot["file"]
    if sha(path) != snapshot["checkpoint_sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))


def phase_train(args, arm, phase):
    output = args.output / arm
    reference = lambda s: args.output / "native" / f"stage-{s}" if arm != "native" else None
    if phase == 2:
        decision = read(args.output / "extension.json")
        if not decision["extend"]:
            return
        previous = output / "stage-1"
        active = read(previous / "complete.json")["gpu_train_seconds"]
        fit(args, arm, output / "stage-2", FIRST, SECOND, INTERVAL, min(STAGE_SECONDS[1], active + STAGE_SECONDS[0]),
            read(output / "tuning.json")["choice"], previous=previous, reference=reference(2))
        return
    recipe, bank, _ = inputs(args.root, args.device)
    model = initialize(recipe, arm, args.device, 32).eval()
    trials = []
    for choice in (0, 1):
        trial = output / f"trial-{choice}"
        fit(args, arm, trial, 0, TUNE, TUNE, 240, choice, reference=reference(1))
        summary = read(trial / "complete.json")
        trials.append((trial, summary))
    # Compare both trials at an identical observed training step. Unequal incomplete trials
    # fall back to the declared base rate without borrowing information from later stages.
    scores = []
    for trial, summary in trials:
        snapshot = summary["snapshots"][-1]
        load_snapshot(model, trial, snapshot)
        scores.append(dict(choice=summary["choice"], step=summary["end"],
                           seconds=summary["gpu_train_seconds"], **fixed_depth(model, bank, args.device)))
    choice = 0
    if scores[0]["step"] == scores[1]["step"]:
        choice = max(range(2), key=lambda i: (scores[i]["exact_16"], scores[i]["blank_accuracy_16"], -i))
    del model
    write(output / "tuning.json", dict(choice=choice, trials=scores))
    fit(args, arm, output / "stage-1", 0, FIRST, INTERVAL, STAGE_SECONDS[0], choice, reference=reference(1))


def compare(args, phase):
    recipe, bank, calibration = inputs(args.root, args.device)
    output = args.output
    curves, final = {}, {}
    for arm in ARMS:
        model = initialize(recipe, arm, args.device, 32).eval()
        runs = [output / arm / f"stage-{i}" for i in range(1, phase + 1)]
        rows = []
        for run in runs:
            info = read(run / "complete.json")
            for snapshot in info["snapshots"]:
                load_snapshot(model, run, snapshot)
                rows.append(dict(**snapshot, **fixed_depth(model, bank, args.device), stage=run.name))
        curves[arm] = rows
        final[arm] = rows[-1]["step"]
    common = sorted(set.intersection(*({row["step"] for row in curves[a]} for a in ARMS)))
    if not common:
        raise ValueError("No complete equal-exposure checkpoint")
    for step in common:
        if len({next(r for r in curves[a] if r["step"] == step)["supervised_sha256"] for a in ARMS}) != 1:
            raise ValueError(f"Unmatched effective training stream at step {step}")
    times = sorted({r["gpu_train_seconds"] for rows in curves.values() for r in rows
                    if max(rows[0]["gpu_train_seconds"] for rows in curves.values()) <= r["gpu_train_seconds"]
                    <= min(rows[-1]["gpu_train_seconds"] for rows in curves.values())})
    equal_time = [dict(gpu_train_seconds=t, arms={
        a: next(r for r in reversed(rows) if r["gpu_train_seconds"] <= t) for a, rows in curves.items()})
        for t in times]
    write(output / f"curves-stage-{phase}.json", dict(curves=curves, equal_steps=common, equal_time=equal_time))
    with atomic(output / f"curves-stage-{phase}.csv") as temporary:
        with temporary.open("w", newline="") as handle:
            columns = ("arm", "stage", "step", "supervised_examples", "gpu_train_seconds",
                       "exact_12", "exact_16", "blank_accuracy_16", "mean_abs_gate",
                       "correction_to_native_update", "fraction_changed_post_cast")
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for arm, rows in curves.items():
                writer.writerows(dict(arm=arm, **row) for row in rows)
    if phase == 1:
        gains = {}
        if common[-1] == FIRST and FIRST - 2 * INTERVAL in common:
            for arm, rows in curves.items():
                earlier = next(r for r in rows if r["step"] == FIRST - 2 * INTERVAL)
                gains[arm] = rows[-1]["exact_16"] - earlier["exact_16"]
        extend = bool(gains) and max(gains.values()) >= MIN_GAIN
        write(output / "extension.json", dict(extend=extend, threshold=MIN_GAIN, gains=gains,
                                              stage_one_end=final))
        if extend:
            print("Extension triggered:", gains, flush=True)
            return
    budget = median(row["seconds"][11] for row in calibration["traces"])
    picks = {a: next(r for r in curves[a] if r["step"] == common[-1]) for a in ARMS}
    models = {a: initialize(recipe, a, args.device, 1).eval() for a in ARMS}
    for arm in ARMS:
        load_snapshot(models[arm], output / arm / picks[arm]["stage"], picks[arm])
    x, y = (bank[key].to(args.device) for key in ("inputs", "labels"))
    for i in range(4):
        for model in models.values():
            predict(model, x[i], y[i], budget, args.device)
    results = []
    for i, board_id in enumerate(bank["ids"]):
        offset = i % len(ARMS)
        arms = {a: predict(models[a], x[i], y[i], budget, args.device) for a in ARMS[offset:] + ARMS[:offset]}
        results.append(dict(id=board_id, arms=arms))
    flags = {a: [r["arms"][a]["timed_exact"] for r in results] for a in ARMS}
    paired = {f"{a}_vs_{b}": paired_binary_interval(flags[a], flags[b]) for a, b in
              (("directional", "native"), ("directional", "scalar"), ("scalar", "native"))}
    write(output / "evaluation.json", dict(
        train_step=common[-1], budget_seconds=budget, checkpoints=picks,
        timing={a: dict(exact=sum(flags[a]), cap_hits=sum(r["arms"][a]["timed_cap"] for r in results),
                        no_readout=sum(r["arms"][a]["timed_no_readout"] for r in results),
                        median_accepted=median(r["arms"][a]["accepted"] for r in results)) for a in ARMS},
        paired=paired, rows=results, stage_ends=final))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--job", required=True, choices=(*ARMS, "compare", "preflight"))
    parser.add_argument("--phase", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    args.device, args.batch, args.eager = torch.device("cuda"), BATCH, False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.job == "preflight":
        args.batch, args.eager = 8, True
        for arm in ARMS:
            ref = args.output / "native" if arm != "native" else None
            directory = args.output / arm
            fit(args, arm, directory / "initial", 0, 4, 4, 60, 0,
                reference=None if ref is None else ref / "initial")
            fit(args, arm, directory / "resumed", 4, 8, 4, 120, 0, previous=directory / "initial",
                reference=None if ref is None else ref / "resumed")
        recipe, bank, _ = inputs(args.root, args.device)
        rows = {}
        for arm in ARMS:
            model = initialize(recipe, arm, args.device, 2).eval()
            directory = args.output / arm / "resumed"
            load_snapshot(model, directory, read(directory / "complete.json")["snapshots"][-1])
            rows[arm] = fixed_depth(model, {k: bank[k][:2] for k in ("inputs", "labels")}, args.device)
            if arm != "native" and rows[arm]["mean_abs_gate"] == 0:
                raise ValueError("Zero gate after training")
        write(args.output / "preflight.json", rows)
    elif args.job == "compare":
        if args.phase == 1 or read(args.output / "extension.json")["extend"]:
            compare(args, args.phase)
    else:
        phase_train(args, args.job, args.phase)


if __name__ == "__main__":
    main()
