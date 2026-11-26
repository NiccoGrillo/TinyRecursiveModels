"""Train matched TRM refiners and compare continuation policies on dev only."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from evaluate_matched_sudoku import model_from_checkpoint, read_boards, sha
from experiments.residual_reasoning.common import source
from experiments.residual_reasoning.model import Architecture
from experiments.residual_reasoning.trm_refinement import TRMRefiner
from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1Carry as NativeCarry,
    TinyRecursiveReasoningModel_ACTV1InnerCarry as NativeInnerCarry,
)

METHODS = ("unrestricted", "stateless", "recurrent")
SEED = 11
BATCH = 16
LOOPS = 4
STEPS = 512
MAX_REFINEMENT_STEPS = 64


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def native_training_bank(native, inputs, labels, fork):
    h, l, logits = [], [], []
    for batch in inputs.split(BATCH):
        data = {"inputs": batch, "puzzle_identifiers": batch.new_zeros(len(batch))}
        with torch.device(batch.device):
            carry = native.initial_carry(data)
            for _ in range(fork):
                carry, outputs = native(carry, data)
        h.append(carry.inner_carry.z_H.cpu())
        l.append(carry.inner_carry.z_L.cpu())
        logits.append(outputs["logits"].cpu())
    return inputs.cpu(), torch.cat(h), torch.cat(l), torch.cat(logits), labels.cpu()


def train(method, architecture, bank, shared, device):
    torch.manual_seed(SEED)
    refiner = TRMRefiner(architecture, bank[1].shape[-1], method)
    initial = refiner.state_dict()
    for key in initial.keys() & shared.keys():
        if initial[key].shape == shared[key].shape:
            initial[key] = shared[key]
    refiner.load_state_dict(initial)
    refiner.to(device).train()
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=0.0003, weight_decay=0.01)
    sync(device)
    start = time.perf_counter()
    for step in range(1, STEPS + 1):
        indices = torch.randint(len(bank[0]), (BATCH,),
                                generator=torch.Generator().manual_seed(SEED * 10000019 + step))
        minibatch = [value[indices].to(device) for value in bank]
        optimizer.zero_grad(set_to_none=True)
        loss = refiner.training_loss(*minibatch, loops=LOOPS)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 128 == 0:
            print(method, step, "loss", loss.item(), flush=True)
    sync(device)
    return refiner.eval(), time.perf_counter() - start


def timed_refinement(refiner, inputs, h, l, logits, label, budget, device):
    sync(device)
    start = time.perf_counter()
    with torch.no_grad():
        context, base, delta, workspace = refiner.prepare(inputs, h, l, logits)
    sync(device)
    elapsed = time.perf_counter() - start
    setup = elapsed
    best = base.argmax(-1)
    accepted_logits = base
    completed = 0
    attempted = 0
    while elapsed < budget and attempted < MAX_REFINEMENT_STEPS:
        delta, workspace = refiner.advance(context, delta, workspace, train=False)
        with torch.no_grad():
            scores = refiner.readout(base, delta)
            prediction = scores.argmax(-1)
        sync(device)
        elapsed = time.perf_counter() - start
        attempted += 1
        if elapsed <= budget:
            best = prediction
            accepted_logits = scores
            completed += 1
    finite = bool(accepted_logits.isfinite().all())
    return dict(correct=finite and bool(best.eq(label).all()), steps=completed,
                cap_hit=completed == MAX_REFINEMENT_STEPS, setup_seconds=setup, nonfinite=not finite)


@torch.inference_mode()
def timed_native(model, inputs, h, l, logits, label, fork, budget, device):
    batch = {"inputs": inputs, "puzzle_identifiers": inputs.new_zeros(len(inputs))}
    carry = NativeCarry(NativeInnerCarry(h, l),
                        inputs.new_full((len(inputs),), fork, dtype=torch.int32),
                        inputs.new_zeros(len(inputs), dtype=torch.bool), batch)
    best = logits.argmax(-1)
    accepted_logits = logits
    completed = 0
    sync(device)
    start = time.perf_counter()
    for _ in range(fork, model.config.halt_max_steps):
        carry, output = model(carry, batch)
        prediction = output["logits"].argmax(-1)
        sync(device)
        if time.perf_counter() - start > budget:
            break
        best = prediction
        accepted_logits = output["logits"]
        completed += 1
    finite = bool(accepted_logits.isfinite().all())
    return dict(correct=finite and bool(best.eq(label).all()), steps=completed,
                cap_hit=completed == model.config.halt_max_steps - fork,
                setup_seconds=0.0, nonfinite=not finite)


def transitions(fork, end):
    return dict(repaired=sum(not a and b for a, b in zip(fork, end)),
                broken=sum(a and not b for a, b in zip(fork, end)), final=sum(end))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "cohort", "weights", "config", "calibration", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    hardware = torch.cuda.get_device_name() if device.type == "cuda" else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plan = json.loads((args.calibration / "calibration.json").read_text())
    chosen = plan["choice"]
    bank_path = args.calibration / "fork_states.pt"
    if (not chosen or plan["cohort"] != "dev" or sha(args.weights) != plan["checkpoint_sha256"]
            or sha(args.config) != plan["config_sha256"] or sha(bank_path) != plan["bank_sha256"]
            or plan["device"] != hardware or plan["torch"] != str(torch.__version__)):
        raise ValueError("Expected an eligible, intact native development calibration")
    fork, budget = chosen["fork"], chosen["seconds"]
    dev = torch.load(bank_path, map_location="cpu", weights_only=True)
    cohort = json.loads(args.cohort.read_text())
    x, y, train_ids, train_sha = read_boards(args.data, "train", "trm",
                                             slice(0, None, cohort["augments"] + 1), device)
    if (train_ids != cohort["original_ids"] or set(train_ids) & set(dev["ids"])
            or dev["ids"] != [row["id"] for row in plan["traces"]] or dev["fork"] != fork):
        raise ValueError("Training or development IDs differ from the frozen manifests")
    if plan["data_sha256"] != cohort["splits"]["dev"]["sha256"]:
        raise ValueError("Development data hash mismatch")
    native, _ = model_from_checkpoint(SimpleNamespace(family="trm", weights=args.weights, config=args.config), device)
    bank = native_training_bank(native, x, y, fork)
    args.output.mkdir()
    architecture = Architecture()
    torch.manual_seed(SEED)
    shared = TRMRefiner(architecture, bank[1].shape[-1], "unrestricted").state_dict()
    orig = [row["correct"][fork - 1] for row in plan["traces"]]
    arms = {}
    refiners = {}
    ordered = [dev[key].to(device) for key in ("inputs", "z_H", "z_L", "logits", "labels")]
    for method in METHODS:
        refiner, elapsed = train(method, architecture, bank, shared, device)
        checkpoint = args.output / f"{method}.pt"
        torch.save(refiner.state_dict(), checkpoint)
        arms[method] = dict(training_seconds=elapsed, parameters=sum(p.numel() for p in refiner.parameters()),
                            checkpoint_sha256=sha(checkpoint))
        refiners[method] = refiner

    def predict(method, i):
        tensors = [value[i : i + 1] for value in ordered]
        if method == "native":
            return timed_native(native, *tensors, fork, budget, device)
        return timed_refinement(refiners[method], *tensors, budget, device)

    names = ("native", *METHODS)
    results = {method: [] for method in names}
    for method in names:
        for i in range(4):
            predict(method, i)
    for i in range(len(orig)):
        offset = i % len(names)
        for method in names[offset:] + names[:offset]:
            results[method].append(predict(method, i))
    outcomes = {"stop": orig, **{name: [row["correct"] for row in history]
                                  for name, history in results.items()}}
    stats = {name: transitions(orig, values) for name, values in outcomes.items()}
    best_baseline = max(stats[name]["final"] for name in ("stop", "native"))
    promotion = [method for method in METHODS
                 if (stats[method]["final"] - best_baseline) / len(orig) >= 0.02
                 and (stats[method]["broken"] - stats["native"]["broken"]) / sum(orig) <= 0.01]
    pairs = {name: dict(helped=sum(not a and b for a, b in zip(values, outcomes["recurrent"])),
                        harmed=sum(a and not b for a, b in zip(values, outcomes["recurrent"])))
             for name, values in outcomes.items() if name != "recurrent"}
    payload = dict(source=source(), calibration_sha256=sha(args.calibration / "calibration.json"),
                   checkpoint_sha256=plan["checkpoint_sha256"], train_sha256=train_sha,
                   train_ids=train_ids, train_steps=STEPS, batch=BATCH, loops=LOOPS, seed=SEED,
                   fork=fork, budget_seconds=budget, ids=dev["ids"], starting_correct=sum(orig),
                   arms=arms, endpoints=stats, recurrent_vs_comparator=pairs, outcomes=outcomes,
                   timing_rows=results, passes_dev_gate=promotion)
    with (args.output / "dev-results.json").open("x") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
