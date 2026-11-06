"""Exact-budget continuation-schedule crossover from the identical native state.

One outer update = L low-state updates + one high-state update. Dense readout is
observational; changing L is a dynamics intervention. See docs/NEXT_EXPERIMENT.md.
"""
import argparse
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .data import load_sudoku, smoke_data, sudoku_valid
from .metrics import exact_correct, ratio
from .model import clone_state, load_model, read_config, reference_margins, rms, sha256, smoke_config
from .report import paired_interval
from .run import code_provenance, write_json


def budget_plan(low_cycles, layers, budget):
    if len(low_cycles) != len(set(low_cycles)) or any(n < 1 for n in low_cycles):
        raise ValueError("Schedules must be distinct positive L values")
    costs = {n: (n + 1) * layers for n in low_cycles}
    common = math.lcm(*costs.values())
    if budget < common or budget % common:
        raise ValueError(f"Branch budget must be a positive multiple of {common} block evaluations")
    return costs, common


def zero_event_upper(n, confidence=0.95):
    """Exact one-sided binomial upper limit when zero of n independent puzzles fail."""
    return 1 - (1 - confidence)**(1 / n) if n else None


def event_rate(values):
    result = ratio(sum(values), len(values))
    result["zero_event_upper_95_one_sided"] = zero_event_upper(len(values)) if not any(values) else None
    return result


@lru_cache(maxsize=16)
def _log_choose(n):
    return [math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) for j in range(n + 1)]


def _binomial_cdf(k, n, p):
    if p == 0:return 1.0
    if p == 1:return float(k == n)
    logs = [v + j * math.log(p) + (n - j) * math.log1p(-p) for j, v in enumerate(_log_choose(n)[:k + 1])]
    largest = max(logs)
    return min(1.0, math.exp(largest) * math.fsum(math.exp(v - largest) for v in logs))


@lru_cache(maxsize=4096)
def _binomial_interval(k, n, tail):
    def invert(index, target):
        low, high = 0.0, 1.0
        for _ in range(48):
            p = (low + high) / 2
            if _binomial_cdf(index, n, p) > target:low = p
            else:high = p
        return (low + high) / 2
    return (0.0 if k == 0 else invert(k - 1, 1 - tail),
            1.0 if k == n else invert(k, tail))


def paired_binary_interval(arm, native, confidence=0.95):
    """Conservative exact bound for p(arm=1,native=0)-p(arm=0,native=1).

    Two Clopper-Pearson intervals with Bonferroni simultaneous coverage. Unlike
    percentile bootstrap this does not collapse when no discordance is observed.
    Puzzle independence is still assumed; this is pointwise across comparisons.
    """
    if len(arm) != len(native):raise ValueError("Pairs must have equal length")
    n = len(arm)
    if not n:return dict(examples=0, mean=None, lower=None, upper=None, excess=0, reduced=0)
    excess = sum(bool(a) and not bool(b) for a, b in zip(arm, native))
    reduced = sum(not bool(a) and bool(b) for a, b in zip(arm, native))
    tail = (1 - confidence) / 4
    elo, ehi = _binomial_interval(excess, n, tail)
    rlo, rhi = _binomial_interval(reduced, n, tail)
    return dict(examples=n, mean=(excess - reduced) / n, lower=elo - rhi, upper=ehi - rlo,
                excess=excess, reduced=reduced, confidence=confidence, method="Bonferroni Clopper-Pearson discordance bounds")


@torch.no_grad()
def measure(adapter, state, previous, logits, batch, ids, outer, blocks):
    prediction = logits.argmax(-1)
    correct = exact_correct(prediction, batch["labels"])
    valid = sudoku_valid(prediction, batch["inputs"])
    margin = reference_margins(logits, prediction).min(-1).values
    q = adapter.q_logit(state)
    h_change = rms(state.z_H.float() - previous.z_H.float()) / rms(previous.z_H).clamp_min(1e-12)
    l_change = rms(state.z_L.float() - previous.z_L.float()) / rms(previous.z_L).clamp_min(1e-12)
    start = adapter.inner.puzzle_emb_len
    per_cell = (state.z_H[:, start:].float() - previous.z_H[:, start:].float()).norm(dim=-1)
    per_cell /= previous.z_H[:, start:].float().norm(dim=-1).clamp_min(1e-12)
    blank = batch["inputs"] == 1
    blank_mean = (per_cell * blank).sum(-1) / blank.sum(-1).clamp_min(1)
    rows = []
    for i, sample_id in enumerate(ids):
        pred = prediction[i].cpu().tolist()
        rows.append(dict(sample_id=sample_id, outer_update=outer, cumulative_block_evals=blocks,
                         correct=bool(correct[i]), rule_valid=bool(valid[i]), prediction=pred,
                         prediction_hash=hashlib.sha256(bytes(pred)).hexdigest(),
                         min_margin=margin[i].item(), q_halt_logit=q[i].item(),
                         z_h_relative_change=h_change[i].item(), z_l_relative_change=l_change[i].item(),
                         blank_z_h_mean_relative_change=blank_mean[i].item()))
    return rows


def observation_audit(prefix, branches, native_l, h_cycles, followup):
    """Same native trajectory: distinguish actual macro outputs from intermediate readouts."""
    timelines = defaultdict(list)
    for row in prefix + [r for r in branches if r["low_cycles"] == native_l]:
        timelines[row["sample_id"]].append(row)
    per_puzzle = []
    for sample, timeline in sorted(timelines.items()):
        t = sorted(timeline, key=lambda r: r["outer_update"])
        if [r["outer_update"] for r in t] != list(range(1, len(t) + 1)):
            raise ValueError("Native observations must form a complete, unique outer-update timeline")
        by_time = {r["outer_update"]: r for r in t}
        official = next((r["outer_update"] for r in t if r["outer_update"] % h_cycles == 0 and r["correct"]), None)
        dense = next((r["outer_update"] for r in t if r["correct"]), None)
        row = dict(sample_id=sample, first_native_macro_solve=official, first_dense_solve=dense,
                   official_anchor_eligible=official is not None and official + followup <= len(t),
                   dense_anchor_eligible=dense is not None and dense + followup <= len(t))
        if row["official_anchor_eligible"]:
            window = [by_time[k] for k in range(official + 1, official + followup + 1)]
            official_loss = any(not r["correct"] for r in window if r["outer_update"] % h_cycles == 0)
            dense_loss = any(not r["correct"] for r in window)
            row.update(native_macro_loss_after_official_anchor=official_loss,
                       dense_loss_after_official_anchor=dense_loss,
                       dense_excursion_missed_by_macro=dense_loss and not official_loss)
        if row["dense_anchor_eligible"]:
            row["dense_loss_after_dense_anchor"] = any(not by_time[k]["correct"] for k in range(dense + 1, dense + followup + 1))
        per_puzzle.append(row)
    off = [r for r in per_puzzle if r["official_anchor_eligible"]]
    dense = [r for r in per_puzzle if r["dense_anchor_eligible"]]
    return dict(followup_outer_updates=followup, followup_native_macro_intervals=followup // h_cycles,
                official_anchor_cohort=len(off), dense_anchor_cohort=len(dense),
                native_macro_loss_after_official_anchor=event_rate([r["native_macro_loss_after_official_anchor"] for r in off]),
                dense_loss_after_official_anchor=event_rate([r["dense_loss_after_official_anchor"] for r in off]),
                dense_excursion_missed_by_macro=event_rate([r["dense_excursion_missed_by_macro"] for r in off]),
                dense_loss_after_dense_anchor=event_rate([r["dense_loss_after_dense_anchor"] for r in dense]),
                not_eligible_official=len(per_puzzle) - len(off), not_eligible_dense=len(per_puzzle) - len(dense),
                per_puzzle=per_puzzle)


def summarize(prefix, branches, anchors, native_l, h_cycles, layers, budget, followup,
              bootstrap=2000, seed=0, min_solved=300):
    schedules = sorted({r["low_cycles"] for r in branches})
    costs, common = budget_plan(schedules, layers, budget)
    lookup = {(r["sample_id"], r["low_cycles"], r["branch_outer"]): r for r in branches}
    if len(lookup) != len(branches):
        raise ValueError("Duplicate branch observations")
    ids = sorted(anchors)
    if any((i, l, k) not in lookup for i in ids for l in schedules for k in range(1, budget // costs[l] + 1)):
        raise ValueError("Incomplete branch observations")
    if len(lookup) != sum(budget // costs[l] for l in schedules) * len(ids):
        raise ValueError("Unexpected branch observations")
    outcomes = {}
    for i in ids:
        for l in schedules:
            t = [lookup[i, l, k] for k in range(1, budget // costs[l] + 1)]
            if any(r["branch_block_evals"] != r["branch_outer"] * costs[l] for r in t):
                raise ValueError("Incorrect block accounting")
            initial = anchors[i]["correct"]
            # Left-continuous output: while doing low-state work, the last z_H
            # still determines the answer. The terminal update has zero duration.
            previous = [initial] + [r["correct"] for r in t[:-1]]
            outcomes[i, l] = dict(final_correct=t[-1]["correct"], ever_lost=initial and any(not r["correct"] for r in t),
                                  ever_recovered=not initial and any(r["correct"] for r in t),
                                  incorrect_block_fraction=sum(not c for c in previous) * costs[l] / budget)
    results = []
    for l in schedules:
        for cohort in ("all", "correct_at_fork", "wrong_at_fork"):
            chosen = [i for i in ids if cohort == "all" or anchors[i]["correct"] == (cohort == "correct_at_fork")]
            arm = [outcomes[i, l] for i in chosen]
            clean = [outcomes[i, native_l] for i in chosen]
            comparisons = {}
            for name in ("incorrect_block_fraction",):
                values = [float(a[name]) - float(b[name]) for a, b in zip(arm, clean)]
                comparisons[name] = paired_interval(values, bootstrap, seed)
            binary = {name: paired_binary_interval([a[name] for a in arm], [b[name] for b in clean])
                      for name in ("final_correct", "ever_lost", "ever_recovered")}
            if l == native_l:
                for v in binary.values():
                    v.update(lower=0.0 if chosen else None, upper=0.0 if chosen else None, method="Identical arm; difference is zero by construction")
            grid = []
            for c in range(common, budget + 1, common):
                grid.append(dict(branch_block_evals=c,
                                 outer_updates=c // costs[l],
                                 phase_mod_h=(c // costs[l]) % h_cycles,
                                 accuracy=ratio(sum(lookup[i, l, c // costs[l]]["correct"] for i in chosen), len(chosen)),
                                 correct_anchor_survival=ratio(sum(anchors[i]["correct"] and all(lookup[i, l, k]["correct"] for k in range(1, c // costs[l] + 1)) for i in chosen),
                                                               sum(anchors[i]["correct"] for i in chosen))))
            entry = dict(low_cycles=l, cohort=cohort, examples=len(chosen), outer_updates=budget // costs[l],
                         final_phase_mod_h=(budget // costs[l]) % h_cycles,
                         final_correct=event_rate([r["final_correct"] for r in arm]),
                         mean_incorrect_block_fraction=float(np.mean([r["incorrect_block_fraction"] for r in arm])) if arm else None,
                         paired_difference_vs_native=comparisons, common_budget_grid=grid)
            entry["paired_binary_difference_vs_native"] = binary
            # Retention/recovery rates only on their proper pre-treatment risk sets.
            if cohort == "correct_at_fork":entry["ever_lost"] = event_rate([r["ever_lost"] for r in arm])
            if cohort == "wrong_at_fork":entry["ever_recovered"] = event_rate([r["ever_recovered"] for r in arm])
            results.append(entry)
    solved = sum(anchors[i]["correct"] for i in ids)
    native_macro_loss = event_rate([
        any(not lookup[i, native_l, k]["correct"] for k in range(h_cycles, budget // costs[native_l] + 1, h_cycles))
        for i in ids if anchors[i]["correct"]])
    return dict(native_low_cycles=native_l, branch_budget=budget, common_block_interval=common,
                correct_at_fork=solved, wrong_at_fork=len(ids) - solved,
                native_macro_ever_loss_from_fork=native_macro_loss,
                precision_gate=dict(required_correct_at_fork=min_solved, reached=solved >= min_solved,
                                    note="A diagnostic adequacy flag, not an optional-stopping instruction or a power guarantee."),
                comparisons=results,
                observation_audit=observation_audit(prefix, branches, native_l, h_cycles, followup),
                uncertainty="Binary effects: conservative exact discordance bounds; continuous effects: paired puzzle bootstrap. Pointwise across comparisons; no checkpoint variance. Puzzle independence is assumed.")


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-attention", action="store_true")
    for name in ("config", "checkpoint", "data"):p.add_argument("--" + name, type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=["float32", "bfloat16"])
    p.add_argument("--attention-backend", choices=["math", "auto"], default="math")
    p.add_argument("--limit", type=int, default=1024)
    p.add_argument("--offset", type=int, default=32, help="Skip the old 32-puzzle exploratory prefix")
    p.add_argument("--partition", choices=["dev", "test"], default="dev")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--low-cycles", type=int, nargs="+", help="Must include native L; default native L and 12")
    p.add_argument("--fork-outer", type=int, help="Native prefix length; default nominal halt_max_steps * H_cycles")
    p.add_argument("--branch-blocks", type=int, help="Exact per-arm continuation budget; default 2184, or two common intervals for smoke")
    p.add_argument("--followup-outer", type=int, help="Native observation audit; default 36, or 4 for smoke")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--min-solved", type=int, default=300)
    return p.parse_args()


def run(args):
    if args.out.exists():raise FileExistsError(args.out)
    if args.smoke and any(getattr(args, name) is not None for name in ("config", "checkpoint", "data")):
        raise ValueError("Smoke cannot use real model/data paths")
    if not args.smoke and not all(getattr(args, name) is not None for name in ("config", "checkpoint", "data")):
        raise ValueError("Provide config, checkpoint and data, or explicitly use --smoke")
    if args.offset < 0 or min(args.limit, args.batch_size, args.threads, args.bootstrap, args.min_solved) < 1:
        raise ValueError("Invalid sample, batch, bootstrap or thread count")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = smoke_data() if args.smoke else load_sudoku(args.data, args.partition, args.split_seed, args.offset + args.limit)
    if not args.smoke:
        selection = slice(args.offset, args.offset + args.limit)
        data.inputs, data.labels, data.identifiers = data.inputs[selection], data.labels[selection], data.identifiers[selection]
        data.sample_ids = data.sample_ids[selection]
        data.provenance.update(selected_ids=data.sample_ids, selected_indices=data.provenance["selected_indices"][selection], selection_offset=args.offset)
    config = smoke_config(not args.smoke_attention) if args.smoke else read_config(args.config, data.metadata, args.batch_size, args.dtype)
    if args.dtype:config["forward_dtype"] = args.dtype
    adapter = load_model(config, args.checkpoint, args.device)
    native, h, layers = config["L_cycles"], config["H_cycles"], config["L_layers"]
    schedules = args.low_cycles if args.low_cycles is not None else sorted({native, 4 if args.smoke else 12})
    if native not in schedules or len(schedules) < 2:raise ValueError("Include native L and at least one distinct schedule")
    fork = args.fork_outer if args.fork_outer is not None else (4 if args.smoke else config["halt_max_steps"] * h)
    if fork < 0 or fork % h:raise ValueError("Fork at a native macro boundary (a nonnegative multiple of H_cycles)")
    common = math.lcm(*[(l + 1) * layers for l in schedules])
    budget = args.branch_blocks if args.branch_blocks is not None else (2 * common if args.smoke else 2184)
    costs, common = budget_plan(schedules, layers, budget)
    followup = args.followup_outer if args.followup_outer is not None else (4 if args.smoke else 36)
    if followup < h or followup % h or followup > fork + budget // costs[native]:
        raise ValueError("Follow-up must fit the native timeline and be a positive multiple of H_cycles")
    args.out.mkdir(parents=True)
    manifest = dict(schema_version=1, experiment="clock_crossover", status="running",
                    run_kind="random_weight_smoke" if args.smoke else "checkpoint_diagnostic",
                    started_utc=datetime.now(timezone.utc).isoformat(), arguments=vars(args), model_config=config,
                    dataset=data.provenance, code=code_provenance(), checkpoint_sha256=None if args.smoke else sha256(args.checkpoint),
                    config_sha256=None if args.smoke else sha256(args.config),
                    software=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                                  cuda=torch.version.cuda, device=args.device, attention_backend=args.attention_backend),
                    design=dict(native_l=native, low_cycles=schedules, fork_outer=fork, fork_native_macro=fork // h,
                                branch_blocks=budget, common_block_interval=common, blocks_per_outer=costs,
                                followup_outer=followup, followup_native_blocks=followup * costs[native],
                                estimand="Continuation-schedule effect on states reached by the native schedule; pre-fork cohorts only"))
    if args.checkpoint is not None and (args.checkpoint.parent / "provenance.json").exists():
        path = args.checkpoint.parent / "provenance.json"
        manifest["checkpoint_provenance"] = dict(sha256=sha256(path), record=json.loads(path.read_text()))
    write_json(args.out / "manifest.json", manifest)
    prefix, branches, anchors = [], [], {}
    ledger = dict(prefix_sample_block_evals=0, branch_sample_block_evals=0, readout_sample_evals=0,
                  input_embedding_sample_evals=0, backward_calls=0)
    start_time = time.perf_counter()
    backend = sdpa_kernel(SDPBackend.MATH) if args.attention_backend == "math" else nullcontext()
    try:
        with backend, (args.out / "prefix.jsonl").open("w") as pf, (args.out / "branches.jsonl").open("w") as bf:
            def emit(handle, memory, records):
                for row in records:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                    memory.append({k: v for k, v in row.items() if k != "prediction"})
            for ids, batch in data.batches(args.batch_size, args.device):
                state = adapter.initial(batch)
                context = adapter.outer_context(batch)
                ledger["input_embedding_sample_evals"] += len(ids)
                for step in range(1, fork + 1):
                    previous = state
                    state, logits = adapter.outer_step(state, batch, native, context)
                    records = measure(adapter, state, previous, logits, batch, ids, step, step * costs[native])
                    for row in records:row["native_macro_endpoint"] = step % h == 0
                    emit(pf, prefix, records)
                    ledger["prefix_sample_block_evals"] += len(ids) * costs[native]
                    ledger["readout_sample_evals"] += len(ids)
                anchor_logits = adapter.readout(state)
                ledger["readout_sample_evals"] += len(ids)
                anchor_rows = measure(adapter, state, state, anchor_logits, batch, ids, fork, fork * costs[native])
                for row in anchor_rows:anchors[row["sample_id"]] = row
                for l in schedules:
                    current = clone_state(state)
                    if not torch.equal(current.z_H, state.z_H) or not torch.equal(current.z_L, state.z_L):
                        raise AssertionError("Branches must start from the identical full state")
                    for offset in range(1, budget // costs[l] + 1):
                        previous = current
                        current, logits = adapter.outer_step(current, batch, l, context)
                        records = measure(adapter, current, previous, logits, batch, ids, fork + offset,
                                          fork * costs[native] + offset * costs[l])
                        for row in records:
                            row.update(low_cycles=l, branch_outer=offset, branch_block_evals=offset * costs[l],
                                       correct_at_fork=anchors[row["sample_id"]]["correct"],
                                       native_macro_endpoint=(fork + offset) % h == 0 if l == native else None)
                        emit(bf, branches, records)
                        ledger["branch_sample_block_evals"] += len(ids) * costs[l]
                        ledger["readout_sample_evals"] += len(ids)
                pf.flush(); bf.flush()
                print(json.dumps(dict(progress="batch_complete", completed_examples=len(anchors), total_examples=len(data.inputs))), flush=True)
        if str(args.device).startswith("cuda"):torch.cuda.synchronize()
        write_json(args.out / "anchors.json", anchors)
        summary = summarize(prefix, branches, anchors, native, h, layers, budget, followup,
                            args.bootstrap, args.seed, args.min_solved)
        write_json(args.out / "summary.json", summary)
        write_report(args.out, manifest, summary)
        manifest.update(status="complete", wall_seconds=time.perf_counter() - start_time, compute=ledger,
                        compute_note="Shared block counts are exact within a checkpoint, not FLOPs across architectures. Readout/embedding evaluations are separate; Q-head and analysis costs enter wall time.",
                        output_sha256={name: sha256(args.out / name) for name in ("prefix.jsonl", "branches.jsonl", "anchors.json", "summary.json")})
        write_json(args.out / "manifest.json", manifest)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", compute=ledger)
        write_json(args.out / "manifest.json", manifest)
        raise
    print(json.dumps(dict(output=str(args.out), status="complete", correct_at_fork=summary["correct_at_fork"],
                          adequacy_gate=summary["precision_gate"], wall_seconds=manifest["wall_seconds"])), flush=True)
    return summary


def write_report(out, manifest, summary):
    lines = ["# Clock crossover", "", f"Run kind: **{manifest['run_kind']}**.", "",
             "Each schedule starts from the same full state after the native prefix. Cohorts are defined before branching.", "",
             "Dense outcomes inspect every high-state update, including intermediate phases. They are not automatically errors at the native model's normal macro outputs. Common-budget grids also use dense readouts; endpoint phases are recorded in summary.json.", "",
             f"Continuation budget: {summary['branch_budget']} blocks per arm. Shared endpoints every {summary['common_block_interval']} blocks.", "",
             f"Correct at fork: {summary['correct_at_fork']}; wrong: {summary['wrong_at_fork']}. Adequacy flag: {summary['precision_gate']['reached']}.", "",
             "| L | Cohort | N | Final phase | Final correct | Excess dense ever-loss vs native, pp [95% interval] | Incorrect block-time |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for entry in summary["comparisons"]:
        delta = entry["paired_binary_difference_vs_native"]["ever_lost"]
        loss = "—" if entry["cohort"] != "correct_at_fork" or delta["mean"] is None else f"{100*delta['mean']:+.2f} [{100*delta['lower']:+.2f}, {100*delta['upper']:+.2f}]"
        f = entry["final_correct"]
        time_wrong = entry["mean_incorrect_block_fraction"]
        lines.append(f"| {entry['low_cycles']} | {entry['cohort']} | {entry['examples']} | {entry['final_phase_mod_h']} | {f['numerator']}/{f['denominator']} | {loss} | {'—' if time_wrong is None else f'{100*time_wrong:.2f}%'} |")
    v = summary["native_macro_ever_loss_from_fork"]
    lines += ["", f"Native macro-output ever-loss among correct-at-fork puzzles over the full branch budget: {v['numerator']}/{v['denominator']}; zero-event one-sided 95% upper limit: {v['zero_event_upper_95_one_sided']}."]
    lines += ["", "## Observation audit on the native trajectory", "",
              "Official-anchor comparisons use the same puzzles and the same fixed follow-up. Intermediate excursions are not failures of the native macro output policy.", ""]
    for key in ("native_macro_loss_after_official_anchor", "dense_loss_after_official_anchor", "dense_excursion_missed_by_macro", "dense_loss_after_dense_anchor"):
        v = summary["observation_audit"][key]
        lines.append(f"- {key}: {v['numerator']}/{v['denominator']}; zero-event one-sided upper limit: {v['zero_event_upper_95_one_sided']}")
    lines += ["", "Binary paired intervals use conservative Clopper-Pearson bounds on discordant events, so zero-event comparisons do not collapse to false certainty. Continuous block-time intervals use the puzzle bootstrap. All intervals are pointwise; no checkpoint variance or architecture-causal claim.", ""]
    (out / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    run(arguments())
