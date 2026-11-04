"""Metrics with explicit denominators; reference labels are used offline only."""
from collections import defaultdict

import torch

from .data import sudoku_valid
from .model import reference_margins, rms


def ratio(numerator, denominator):
    return {"numerator": int(numerator), "denominator": int(denominator),
            "rate": numerator / denominator if denominator else None}


def exact_correct(prediction, labels):
    mask = labels != -100
    return ((prediction == labels) | ~mask).all(-1) & mask.any(-1)


def observe(adapter, state, logits, previous, batch, ids, step, previous_prediction=None):
    with torch.no_grad():
        prediction = logits.argmax(-1)
        margins = reference_margins(logits, prediction)
        correct = exact_correct(prediction, batch["labels"])
        valid = sudoku_valid(prediction, batch["inputs"])
        high_residual = rms(state.z_H - previous.z_H)
        low_residual = rms(state.z_L - previous.z_L)
        q = adapter.q_logit(state)
        rows = []
        for i, sample_id in enumerate(ids):
            rows.append(dict(sample_id=sample_id, step=step, correct=bool(correct[i]),
                             rule_valid=bool(valid[i]), prediction=prediction[i].tolist(),
                             min_margin=margins[i].min().item(), mean_margin=margins[i].mean().item(),
                             q_halt_logit=q[i].item(), q_halt_sigmoid=q[i].sigmoid().item(),
                             z_h_delta_rms=high_residual[i].item(), z_l_delta_rms=low_residual[i].item(),
                             z_h_rms=rms(state.z_H)[i].item(), z_l_rms=rms(state.z_L)[i].item(),
                             changed_tokens=None if previous_prediction is None else
                             int((prediction[i] != previous_prediction[i]).sum()),
                             cumulative_block_calls=step * adapter.block_calls_per_macro))
    return rows


def summarize_rollouts(rows):
    timelines = defaultdict(list)
    for row in rows:
        timelines[row["sample_id"]].append(row)
    ordered = [sorted(x, key=lambda y: y["step"]) for x in timelines.values()]
    if not ordered:
        raise ValueError("No rollout records")
    steps = [r["step"] for r in ordered[0]]
    if any([r["step"] for r in t] != steps for t in ordered):
        raise ValueError("Incomplete or duplicated rollout timelines")
    losses = recovery = right_risk = wrong_risk = 0
    per_step = []
    for j, step in enumerate(steps):
        now = [t[j]["correct"] for t in ordered]
        entry = dict(step=step, accuracy=ratio(sum(now), len(now)))
        if j:
            before = [t[j - 1]["correct"] for t in ordered]
            loss = sum(a and not b for a, b in zip(before, now))
            gain = sum(not a and b for a, b in zip(before, now))
            entry.update(loss_hazard=ratio(loss, sum(before)),
                         recovery=ratio(gain, len(before) - sum(before)))
            losses += loss
            recovery += gain
            right_risk += sum(before)
            wrong_risk += len(before) - sum(before)
        per_step.append(entry)
    ever, lost_after_first, rule_stop = 0, 0, []
    for t in ordered:
        correct_indices = [j for j, r in enumerate(t) if r["correct"]]
        if correct_indices:
            ever += 1
            lost_after_first += any(not r["correct"] for r in t[correct_indices[0] + 1:])
        valid = next((r for r in t if r["rule_valid"]), None)
        rule_stop.append(valid)
    return dict(examples=len(ordered), per_step=per_step,
                ever_correct=ratio(ever, len(ordered)),
                final_accuracy=per_step[-1]["accuracy"],
                pooled_loss_hazard=ratio(losses, right_risk),
                pooled_recovery=ratio(recovery, wrong_risk),
                ever_lost_after_first_correct=ratio(lost_after_first, ever),
                first_correct_steps=dict(zip(sorted(timelines), [
                    next((r["step"] for r in sorted(timelines[k], key=lambda r: r["step"]) if r["correct"]), None)
                    for k in sorted(timelines)])),
                verifier_stop_coverage=ratio(sum(r is not None for r in rule_stop), len(ordered)),
                verifier_stop_reference_accuracy=ratio(sum(r is not None and r["correct"] for r in rule_stop),
                                                       sum(r is not None for r in rule_stop)))


def summarize_interventions(rows):
    """Paired harm/help and survival; separate every fork, arm, and budget stratum."""
    lookup = {(r["sample_id"], r["fork_step"], r["arm"], r["offset"]): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate intervention records")
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["fork_step"], r["arm"])].append(r)
    result = []
    for (fork, arm), group in sorted(grouped.items()):
        ids = sorted({r["sample_id"] for r in group})
        horizon = max(r["offset"] for r in group)
        if len(group) != len(ids) * horizon:
            raise ValueError("Incomplete intervention timelines")
        for stratum in ("all", "all_arms_within_norm_tolerance"):
            chosen = [i for i in ids if stratum == "all" or lookup[i, fork, arm, 1]["matched_budget_ok"]]
            harms = helps = right0 = lost = wrong0 = recovered = flips = 0
            survival = []
            for i in chosen:
                timeline = [lookup[i, fork, arm, h] for h in range(1, horizon + 1)]
                clean_final = lookup[i, fork, "clean", horizon]["correct"]
                last = timeline[-1]["correct"]
                initial = timeline[0]["correct_at_fork"]
                harms += clean_final and not last
                helps += not clean_final and last
                right0 += initial
                wrong0 += not initial
                lost += initial and not last
                recovered += not initial and last
                flips += any(r["changed_from_fork"] for r in timeline)
            for h in range(1, horizon + 1):
                survivors = sum(lookup[i, fork, arm, 1]["correct_at_fork"] and
                                all(lookup[i, fork, arm, k]["correct"] for k in range(1, h + 1))
                                for i in chosen)
                survival.append(dict(offset=h, survival=ratio(survivors, right0)))
            result.append(dict(fork_step=fork, arm=arm, stratum=stratum, examples=len(chosen),
                               final_harmed_vs_clean=ratio(harms, len(chosen)),
                               final_helped_vs_clean=ratio(helps, len(chosen)),
                               final_loss_given_correct_at_fork=ratio(lost, right0),
                               final_recovery_given_wrong_at_fork=ratio(recovered, wrong0),
                               any_prediction_change_from_fork=ratio(flips, len(chosen)),
                               first_loss_survival=survival,
                               zero_delta_count=sum(lookup[i, fork, arm, 1]["zero_delta"] for i in chosen)))
    return result
