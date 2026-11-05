"""Write readable Markdown/CSV and puzzle-paired uncertainty from completed runs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .metrics import ratio, summarize_interventions, summarize_rollouts
from .model import sha256


def paired_interval(differences, repetitions=2000, seed=0):
    """One difference per original puzzle, never one per time step or branch."""
    values = np.asarray(differences, dtype=float)
    if repetitions < 1:
        raise ValueError("Bootstrap repetitions must be positive")
    if not len(values):
        return dict(examples=0, mean=None, lower=None, upper=None)
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(repetitions)]
    low, high = np.quantile(means, [0.025, 0.975])
    return dict(examples=len(values), mean=float(values.mean()), lower=float(low), upper=float(high))


def rate(value):
    n, d = value["numerator"], value["denominator"]
    return f"{n}/{d} ({100 * n / d:.1f}%)" if d else "0/0 (undefined)"


def interval(value):
    if value["mean"] is None:
        return "undefined"
    return f"{100 * value['mean']:+.1f} [{100 * value['lower']:+.1f}, {100 * value['upper']:+.1f}] pp"


def write_report(run, repetitions=2000, seed=0):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Refusing to report an incomplete run")
    def read(name):
        return [json.loads(line) for line in (run / name).read_text().splitlines() if line.strip()]
    rows, interventions = read("rollouts.jsonl"), read("interventions.jsonl")
    summary = dict(run_kind=manifest["run_kind"], natural=summarize_rollouts(rows),
                   interventions=summarize_interventions(interventions) if interventions else [])
    if summary != json.loads((run / "summary.json").read_text()):
        raise ValueError("Saved summary disagrees with recomputed records")
    natural = summary["natural"]
    args, config = manifest["arguments"], manifest["model_config"]
    normal_step = config["halt_max_steps"]
    final_step = max(r["step"] for r in rows)
    lookup = {(r["sample_id"], r["step"]): r for r in rows}
    ids = sorted({r["sample_id"] for r in rows})
    baseline = None
    if normal_step <= final_step:
        differences = [int(lookup[i, final_step]["correct"]) - int(lookup[i, normal_step]["correct"]) for i in ids]
        baseline = dict(normal_step=normal_step, final_step=final_step,
                        accuracy=ratio(sum(lookup[i, normal_step]["correct"] for i in ids), len(ids)),
                        final_minus_normal=paired_interval(differences, repetitions, seed),
                        harmed=ratio(sum(x == -1 for x in differences), len(ids)),
                        helped=ratio(sum(x == 1 for x in differences), len(ids)))
    paired_lookup = {(r["sample_id"], r["fork_step"], r["arm"], r["offset"]): r for r in interventions}
    paired = []
    for metric in summary["interventions"]:
        fork, arm, stratum = metric["fork_step"], metric["arm"], metric["stratum"]
        first = [r for r in interventions if r["fork_step"] == fork and r["arm"] == arm and r["offset"] == 1]
        selected = [r for r in first if stratum == "all" or r["matched_budget_ok"]]
        horizon = len(metric["first_loss_survival"])
        last = [paired_lookup[r["sample_id"], fork, arm, horizon] for r in selected]
        differences = [int(r["correct"]) - int(paired_lookup[r["sample_id"], fork, "clean", horizon]["correct"]) for r in last]
        comparisons = [[(paired_lookup[r["sample_id"], fork, arm, t],
                         paired_lookup[r["sample_id"], fork, "clean", t])
                        for t in range(1, horizon + 1)] for r in selected]
        norms = [r["actual_relative_rms"] for r in selected]
        paired.append(dict(**metric, excluded=len(first) - len(selected),
                           final_accuracy=ratio(sum(r["correct"] for r in last), len(last)),
                           final_minus_clean=paired_interval(differences, repetitions, seed),
                           any_prediction_difference_vs_clean=ratio(sum(any(a["prediction"] != b["prediction"] for a, b in timeline) for timeline in comparisons), len(selected)),
                           final_prediction_difference_vs_clean=ratio(sum(timeline[-1][0]["prediction"] != timeline[-1][1]["prediction"] for timeline in comparisons), len(selected)),
                           ever_harmed_vs_clean=ratio(sum(any(b["correct"] and not a["correct"] for a, b in timeline) for timeline in comparisons), len(selected)),
                           ever_helped_vs_clean=ratio(sum(any(a["correct"] and not b["correct"] for a, b in timeline) for timeline in comparisons), len(selected)),
                           budget_failures=sum(not r["budget_ok"] for r in selected),
                           actual_relative_rms_min=min(norms) if norms else None,
                           actual_relative_rms_max=max(norms) if norms else None))
    result = dict(run_kind=manifest["run_kind"], normal_horizon=baseline, paired=paired,
                  analysis_source_sha256={name: sha256(Path(__file__).with_name(name))
                                          for name in ("report.py", "metrics.py")},
                  bootstrap=dict(repetitions=repetitions, seed=seed, confidence=0.95,
                                 unit="original puzzle, separately within each fork/arm/stratum",
                                 note="Pointwise percentile intervals, not simultaneous; no checkpoint variance. Degenerate intervals do not imply certainty."))
    (run / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (run / "accuracy_by_step.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "correct", "examples", "accuracy", "losses", "correct_at_previous_step", "recoveries", "wrong_at_previous_step"])
        for entry in natural["per_step"]:
            loss, recovery = entry.get("loss_hazard", {}), entry.get("recovery", {})
            writer.writerow([entry["step"], entry["accuracy"]["numerator"], entry["accuracy"]["denominator"], entry["accuracy"]["rate"],
                             loss.get("numerator"), loss.get("denominator"), recovery.get("numerator"), recovery.get("denominator")])
    lines = [f"# Experiment report: {run.name}", "",
             "**Random-weight implementation check — not trained-model evidence.**" if args["smoke"] else
             "**Trained-checkpoint exploratory diagnostic.** Checkpoint provenance and evaluation overlap limit interpretation.", "",
             f"- Examples: {natural['examples']}; partition: {manifest['dataset']['partition']}; split seed: {args['split_seed']}; noise seed: {args['seed']}.",
             f"- Precision: {config['forward_dtype']}; device: {args['device']}; steps: {args['steps']}; normal horizon: {normal_step}.",
             f"- Wall time: {manifest['wall_seconds']:.1f} s; checkpoint SHA256: `{manifest['checkpoint_sha256']}`.",
             "- Carry continues without ACT reset. Labels are used for offline evaluation only.", "", "## Natural continuation", "",
             "| Measure | Result |", "| --- | --- |",
             f"| Final exact solve rate | {rate(natural['final_accuracy'])} |",
             f"| Ever correct (offline oracle) | {rate(natural['ever_correct'])} |",
             f"| Ever lost after first correct (unequal follow-up) | {rate(natural['ever_lost_after_first_correct'])} |",
             f"| Correct → wrong transitions | {rate(natural['pooled_loss_hazard'])} |",
             f"| Wrong → correct transitions | {rate(natural['pooled_recovery'])} |",
             f"| First-valid verifier coverage | {rate(natural['verifier_stop_coverage'])} |",
             f"| Reference accuracy at verifier stop | {rate(natural['verifier_stop_reference_accuracy'])} |"]
    if baseline:
        lines += [f"| Normal horizon exact solve rate (step {normal_step}) | {rate(baseline['accuracy'])} |",
                  f"| Final − normal, paired 95% interval | {interval(baseline['final_minus_normal'])} |",
                  f"| Final harmed / helped versus normal | {rate(baseline['harmed'])} / {rate(baseline['helped'])} |"]
    lines += ["", "| Step | Exact accuracy | Loss given previously correct | Recovery given previously wrong |",
              "| --- | --- | --- | --- |"]
    for entry in natural["per_step"]:
        lines.append(f"| {entry['step']} | {rate(entry['accuracy'])} | {rate(entry['loss_hazard']) if 'loss_hazard' in entry else '—'} | {rate(entry['recovery']) if 'recovery' in entry else '—'} |")
    if paired:
        lines += ["", "## Paired continuations", "", "Harm/help compare final correctness against the same puzzle's clean branch. Loss/recovery condition on correctness at the fork. Stage cohorts can differ in correctness and margin.", ""]
        for stratum in ("all", "all_arms_within_norm_tolerance"):
            lines += ["", f"### Cohort: {stratum}", "", "| Fork | Arm | N (excluded) | Final solve | Harm | Help | Loss | Recovery | Δ vs clean [95% CI] |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
            for p in paired:
                if p["stratum"] == stratum:
                    lines.append(f"| {p['fork_step']} | {p['arm']} | {p['examples']} ({p['excluded']}) | {rate(p['final_accuracy'])} | {rate(p['final_harmed_vs_clean'])} | {rate(p['final_helped_vs_clean'])} | {rate(p['final_loss_given_correct_at_fork'])} | {rate(p['final_recovery_given_wrong_at_fork'])} | {interval(p['final_minus_clean'])} |")
        lines += ["", "### Prediction differences during continuation (full cohort)", "",
                  "Each count is a puzzle, compared with its clean branch at matching offsets. Any-time harm/help can both occur for one puzzle; neither is a natural-error predictor.", "",
                  "| Fork | Arm | Any prediction difference | Final prediction difference | Ever harmed | Ever helped |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for p in paired:
            if p["stratum"] == "all":
                lines.append(f"| {p['fork_step']} | {p['arm']} | {rate(p['any_prediction_difference_vs_clean'])} | {rate(p['final_prediction_difference_vs_clean'])} | {rate(p['ever_harmed_vs_clean'])} | {rate(p['ever_helped_vs_clean'])} |")
        lines += ["", "### Realized budgets (full cohort)", "", "| Fork | Arm | Actual relative norm range | Zero deltas | Budget failures |", "| --- | --- | --- | --- | --- |"]
        for p in paired:
            if p["stratum"] == "all":
                lines.append(f"| {p['fork_step']} | {p['arm']} | {p['actual_relative_rms_min']:.6g}–{p['actual_relative_rms_max']:.6g} | {p['zero_delta_count']} | {p['budget_failures']} |")
    lines += ["", "## Compute and interpretation", "", "```json", json.dumps(manifest["compute"], indent=2), "```", "",
              "Intervals resample original puzzles within each comparison; repeated steps are not independent observations. These are pointwise percentile intervals with no multiple-comparison correction or checkpoint-seed variance. Small samples and zero observed differences can give misleadingly narrow intervals. Runs with different seeds are not independent puzzle samples and must not be pooled naively.", "",
              "Gradient arms are offline attacks with extra probe forwards/backwards. Attack effects do not establish prediction or causation of natural failures. Current-output invariance does not establish a dynamically invariant manifold.", "",
              "Raw evidence: [manifest](manifest.json), [summary](summary.json), [uncertainty and paired results](report.json), [accuracy CSV](accuracy_by_step.csv), [rollouts](rollouts.jsonl), [interventions](interventions.jsonl).", ""]
    (run / "report.md").write_text("\n".join(lines))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    write_report(args.run, args.bootstrap, args.seed)
    print(args.run / "report.md")


if __name__ == "__main__":
    main()
