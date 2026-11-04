"""Run natural trajectories and paired interventions; see docs/EXPERIMENTS.md."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from .data import load_sudoku, smoke_data
from .interventions import gradient_direction, perturb, random_direction
from .metrics import observe, summarize_interventions, summarize_rollouts
from .model import clone_state, load_model, read_config, sha256, smoke_config


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Random tiny TRM on synthetic inputs; never benchmark evidence")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data", type=Path, help="Prepared upstream Sudoku root containing test/")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--experiment", choices=["rollout", "intervene", "both"], default="both")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--partition", choices=["dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--fork-steps", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--targeted", action="store_true", help="Add offline future-gradient ± arms (extra forward/backward cost)")
    parser.add_argument("--probe-horizon", type=int, default=2)
    parser.add_argument("--norm-tolerance", type=float, default=0.05)
    return parser.parse_args()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False, default=str) + "\n")


def code_provenance():
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else None
    files = sorted(p for base in (root / "experiments", root / "models") for p in base.rglob("*.py"))
    hashes = {str(p.relative_to(root)): sha256(p) for p in files}
    return dict(git_commit=git("rev-parse", "HEAD"), git_status=git("status", "--porcelain"),
                source_sha256=hashes)


def run(args):
    if min(args.batch_size, args.threads, args.steps, args.horizon, args.probe_horizon) < 1:
        raise ValueError("Batch, threads, steps and horizons must be positive")
    if not 0 <= args.epsilon or not 0 <= args.norm_tolerance < 1:
        raise ValueError("Invalid epsilon or norm tolerance")
    intervene = args.experiment in ("both", "intervene")
    if intervene and (len(set(args.fork_steps)) != len(args.fork_steps) or
                      any(t < 1 or t > args.steps for t in args.fork_steps)):
        raise ValueError("Fork steps must be distinct, predetermined and within [1, steps]")
    if args.smoke and any(x is not None for x in (args.config, args.checkpoint, args.data)):
        raise ValueError("--smoke cannot be combined with real model/data paths")
    if not args.smoke and not all(x is not None for x in (args.config, args.checkpoint, args.data)):
        raise ValueError("Research runs require --config, --checkpoint and --data; use --smoke explicitly otherwise")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Deterministic CPU; GPU kernels may still depend on hardware/version. Log that limitation.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = smoke_data() if args.smoke else load_sudoku(args.data, args.partition, args.split_seed, args.limit)
    config = smoke_config() if args.smoke else read_config(args.config, data.metadata, args.batch_size, args.dtype)
    if args.smoke and args.dtype:
        config["forward_dtype"] = args.dtype
    adapter = load_model(config, args.checkpoint, args.device)
    out = args.out
    # No accidental replacement of earlier results, even failed runs.
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema_version=1, status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                    run_kind="random_weight_smoke" if args.smoke else "checkpoint_diagnostic",
                    arguments=vars(args), model_config=config, dataset=data.provenance,
                    checkpoint_sha256=None if args.smoke else sha256(args.checkpoint),
                    config_sha256=None if args.smoke else sha256(args.config),
                    software=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                                  cuda=torch.version.cuda, device=str(args.device)),
                    code=code_provenance(), upstream_commit="c01103738605ba39d1430519b1ee0c62f4c707f8",
                    precision_note="Explicit dtype override changes the numerical dynamics; compare against original precision.",
                    phase="after all H_cycles; never reset the inner carry during continuation",
                    norm_definition="epsilon = ||delta z_L||_2 / ||z_L||_2 per example; measured after dtype cast",
                    module_calls_per_macro=adapter.module_calls_per_macro,
                    block_calls_per_macro=adapter.block_calls_per_macro)
    write_json(out / "manifest.json", manifest)
    ledger = dict(natural_forward_sample_macros=0, continuation_forward_sample_macros=0,
                  probe_forward_sample_macros=0, probe_backward_batch_calls=0,
                  probe_backward_sample_objectives=0)
    natural_metrics, intervention_metrics = [], []
    begin = time.perf_counter()
    try:
        with (out / "rollouts.jsonl").open("w") as natural_file, (out / "interventions.jsonl").open("w") as paired_file:
            for ids, batch in data.batches(args.batch_size, args.device):
                state = adapter.initial(batch)
                forks = {}
                previous_prediction = None
                for step in range(1, args.steps + 1):
                    previous = state
                    state, logits = adapter.step(state, batch)
                    rows = observe(adapter, state, logits, previous, batch, ids, step, previous_prediction)
                    for row in rows:
                        natural_file.write(json.dumps(row, allow_nan=False) + "\n")
                        natural_metrics.append({k: row[k] for k in ("sample_id", "step", "correct", "rule_valid")})
                    ledger["natural_forward_sample_macros"] += len(ids)
                    previous_prediction = logits.argmax(-1)
                    if intervene and step in args.fork_steps:
                        forks[step] = (clone_state(state), rows)
                for fork_step, (fork, fork_rows) in sorted(forks.items()):
                    fork_logits = adapter.readout(fork).detach()
                    anchor = fork_logits.argmax(-1)
                    # Seed each example independently so changing batch size does not change random directions.
                    random_parts = []
                    for i, sample_id in enumerate(ids):
                        key = f"{args.seed}:{fork_step}:{sample_id}".encode()
                        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63 - 1)
                        part = clone_state(fork)
                        part.z_L = part.z_L[i:i + 1]
                        random_parts.append(random_direction(part, seed))
                    random = torch.cat(random_parts)
                    directions = {"clean": torch.zeros_like(random), "random_plus": random, "random_minus": -random}
                    if args.targeted:
                        gradient = gradient_direction(adapter, fork, batch, args.probe_horizon)
                        directions.update(gradient_plus=gradient, gradient_minus=-gradient)
                        ledger["probe_forward_sample_macros"] += len(ids) * args.probe_horizon
                        ledger["probe_backward_batch_calls"] += 1
                        ledger["probe_backward_sample_objectives"] += len(ids)
                    branches = {}
                    for arm, direction in directions.items():
                        perturbed, budgets = perturb(fork, direction, 0 if arm == "clean" else args.epsilon,
                                                    args.norm_tolerance)
                        if not torch.equal(perturbed.z_H, fork.z_H) or not torch.equal(adapter.readout(perturbed), fork_logits):
                            raise AssertionError("An intervention changed the immediate answer logits")
                        branches[arm] = (perturbed, budgets)
                    matched = [all(budgets[i]["budget_ok"] for _, budgets in branches.values()) for i in range(len(ids))]
                    for arm, (current, budgets) in branches.items():
                        previous_prediction = anchor
                        for offset in range(1, args.horizon + 1):
                            previous = current
                            current, logits = adapter.step(current, batch)
                            records = observe(adapter, current, logits, previous, batch, ids, fork_step + offset,
                                              previous_prediction)
                            previous_prediction = logits.argmax(-1)
                            for i, row in enumerate(records):
                                row.update(fork_step=fork_step, offset=offset, arm=arm,
                                           correct_at_fork=fork_rows[i]["correct"],
                                           min_margin_at_fork=fork_rows[i]["min_margin"],
                                           q_halt_logit_at_fork=fork_rows[i]["q_halt_logit"],
                                           changed_from_fork=bool((previous_prediction[i] != anchor[i]).any()),
                                           immediate_logits_equal=True, matched_budget_ok=matched[i], **budgets[i])
                                paired_file.write(json.dumps(row, allow_nan=False) + "\n")
                                intervention_metrics.append({k: v for k, v in row.items() if k != "prediction"})
                            ledger["continuation_forward_sample_macros"] += len(ids)
        if str(args.device).startswith("cuda"):
            torch.cuda.synchronize()
        summary = dict(run_kind=manifest["run_kind"], natural=summarize_rollouts(natural_metrics),
                       interventions=summarize_interventions(intervention_metrics) if intervene else [])
        write_json(out / "summary.json", summary)
        manifest.update(status="complete", wall_seconds=time.perf_counter() - begin, compute=ledger,
                        compute_note="Counts include all repeated branches and probe forwards. Backwards are separate; counts are not FLOPs.",
                        finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(out / "manifest.json", manifest)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}", compute=ledger)
        write_json(out / "manifest.json", manifest)
        raise
    print(json.dumps(dict(output=str(out), run_kind=manifest["run_kind"], examples=len(data.inputs),
                          final_accuracy=summary["natural"]["final_accuracy"], wall_seconds=manifest["wall_seconds"])))
    return summary


if __name__ == "__main__":
    run(arguments())
