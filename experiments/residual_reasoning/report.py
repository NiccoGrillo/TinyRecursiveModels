"""Create a human-readable overview and tidy tables from sealed, per-example results."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import atomic, digest, exclusive, fingerprint, read, write


def table(path, rows):
    with atomic(path) as tmp:
        with tmp.open("w", newline="") as fp:
            if rows:
                writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


def ratio(x, n):
    return x / n if n else None


def records(path):
    with path.open() as fp:
        for line in fp:
            yield json.loads(line)


def summarize(folder, label, expected_forks):
    groups = defaultdict(list)
    for obj in records(folder / "examples.jsonl"):
        groups[obj["fork"]].append(obj)
    spec = read(folder / "complete.json")["spec"]
    if set(groups) != set(expected_forks):
        raise ValueError(f"Missing fork in {folder}")
    result, paired = [], {}
    for fork, data in groups.items():
        ids = [v["id"] for v in data]
        if len(set(ids)) != len(ids) or fingerprint(ids) != spec["ids"]:
            raise ValueError(f"Missing or duplicate evaluation IDs in {folder}")
        init = np.array([v["initial"] for v in data])
        decisions = {p for p in data[0]["policy"]}
        for p in sorted(decisions):
            rows = [v["policy"][p] for v in data]
            ok = np.array([v["ok"] for v in rows])
            wrong, right = int((~init).sum()), int(init.sum())
            recovered, retained = int((ok & ~init).sum()), int((ok & init).sum())
            row = dict(
                **label,
                fork=fork,
                policy=p,
                n=len(ids),
                wrong=wrong,
                right=right,
                recovered=recovered,
                retained=retained,
                solved=int(ok.sum()),
                solve=float(ok.mean()),
                repair=ratio(recovered, wrong),
                retention=ratio(retained, right),
                nonfinite=sum(
                    v["first_nonfinite_step"] is not None
                    and v["first_nonfinite_step"] <= v["policy"][p]["t"]
                    for v in data
                ),
                mean_setup_seconds=float(np.mean([v["setup_seconds"] for v in data])),
                cost_cap_fraction=float(np.mean([v["wall_cap_hit"] for v in data])),
                patience_hit_fraction=float(np.mean([v["patience_hit"] for v in data])),
            )
            for k in (
                "seconds",
                "t",
                "field",
                "vjp",
                "solves",
                "controller",
                "proposals",
                "restarts",
            ):
                row["mean_" + k] = float(np.mean([v[k] for v in rows]))
            if p == "wall":
                paired[fork] = (ids, init, ok)
            result.append(row)
    return result, paired


def paired_rows(group, arms):
    if "recurrent" not in arms:
        return []
    output = []
    base_ids, init, mix = arms["recurrent"]
    rng = np.random.default_rng(1819)
    for name, (ids, init2, ok) in sorted(arms.items()):
        if name == "recurrent":
            continue
        if ids != base_ids or not np.array_equal(init, init2):
            raise ValueError(
                "Paired evaluation requires identical IDs and initial correctness"
            )
        for cohort, mask in [
            ("all", np.ones_like(init)),
            ("wrong", ~init),
            ("correct", init),
        ]:
            delta = mix[mask].astype(int) - ok[mask]
            n = len(delta)
            # Paired bootstrap for {-1,0,1} differences via their multinomial counts.
            helps, harms = int((delta == 1).sum()), int((delta == -1).sum())
            lo = hi = None
            if n:
                draws = rng.multinomial(
                    n, [helps / n, harms / n, (n - helps - harms) / n], 2000
                )
                lo, hi = np.quantile(
                    (draws[:, 0] - draws[:, 1]) / n, [0.025, 0.975]
                ).tolist()
            output.append(
                dict(
                    **group,
                    comparison=name,
                    cohort=cohort,
                    n=n,
                    helped=helps,
                    harmed=harms,
                    difference=ratio(helps - harms, n),
                    lo95=lo,
                    hi95=hi,
                )
            )
    return output


def mechanism_cohorts(points, dev_points, label, fork):
    """Fix empirical thresholds on dev wrong states, then apply verbatim to this split."""
    dev = [
        r
        for r in dev_points
        if r["fork"] == fork
        and not r["current_correct"]
        and r["residual"] is not None
        and r["ordinary_force"] is not None
    ]
    high = float(np.quantile([r["residual"] for r in dev], 0.75)) if dev else None
    low = float(np.quantile([r["ordinary_force"] for r in dev], 0.25)) if dev else None
    wrong = [r for r in points if not r["current_correct"]]
    selected = [
        r
        for r in wrong
        if high is not None
        and r["residual"] is not None
        and r["ordinary_force"] is not None
        and r["residual"] >= high
        and r["ordinary_force"] <= low
    ]
    for name, group in (("all_wrong", wrong), ("high_residual_low_gradient", selected)):
        for mode in (
            "native",
            "identity",
            "random",
            "reset_memory",
            "recompute_memory",
        ):
            key = mode + "_endpoint"
            if key in points[0]:
                yield dict(
                    **label,
                    fork=fork,
                    cohort=name,
                    intervention=mode,
                    dev_n=len(dev),
                    residual_cut=high,
                    gradient_cut=low,
                    wrong_n=len(wrong),
                    n=len(group),
                    repaired=sum(r[key] is True for r in group),
                    repair=ratio(sum(r[key] is True for r in group), len(group)),
                )


def aggregate(root, splits, require_complete=False):
    spec = read(root / "study.json")
    cfg = spec["config"]
    out = root / "reports"
    rows, paired, missing, training, mechanism = [], defaultdict(dict), [], [], []
    for stage in ("reference", "shared", "joint"):
        for job in spec["plan"][stage]:
            path = root / "models" / job["id"]
            if not (path / "complete.json").exists():
                missing.append("model/" + job["id"])
                continue
            model = read(path / "manifest.json")
            summary = read(path / "complete.json")
            training.append(
                dict(
                    run=job["id"],
                    method=job["method"],
                    seed=job["seed"],
                    total_parameters=model["parameters"],
                    trainable=model["trainable"],
                    devices=model["spec"]["world"],
                    device=model["spec"]["hardware"]["device"],
                    **summary,
                )
            )
    interventions = []
    for job in spec["plan"]["eval"]:
        dev_probe_path = root / "evaluations" / job["id"] / "dev" / "diagnostics.jsonl"
        dev_probes = []
        dev_seal = dev_probe_path.parent / "complete.json"
        if dev_seal.exists():
            if digest(dev_probe_path) != read(dev_seal)["files"]["diagnostics.jsonl"]:
                raise ValueError("Development diagnostic checksum mismatch")
            dev_probes = list(records(dev_probe_path))
        for split in splits:
            if split == "long" and job["task"] == "sudoku":
                continue
            path = root / "evaluations" / job["id"] / split
            if not (path / "complete.json").exists():
                missing.append(job["id"] + "/" + split)
                continue
            seal = read(path / "complete.json")
            for name, sha in seal["files"].items():
                if digest(path / name) != sha:
                    raise ValueError(f"Result checksum mismatch: {path / name}")
            stage = job["id"].split("/")[2]
            arm = job["id"].split("/")[-1]
            ref = job["ref"].split("/")[-1] if job["ref"] else ""
            label = dict(
                task=job["task"],
                seed=job["seed"],
                split=split,
                condition=stage,
                reference=ref,
                arm=arm,
            )
            forks = cfg["evaluation"]["forks"] if ref else [0]
            result, pairs = summarize(path, label, forks)
            rows.extend(result)
            if stage == "shared":
                for f, pair in pairs.items():
                    key = (job["task"], job["seed"], split, ref, f)
                    paired[key][arm] = pair
            probe = list(records(path / "diagnostics.jsonl"))
            for f in sorted({r["fork"] for r in probe}):
                points = [r for r in probe if r["fork"] == f]
                interventions.extend(mechanism_cohorts(points, dev_probes, label, f))
                for k in points[0]:
                    if k not in ("id", "fork", "step"):
                        subset = [
                            r[k]
                            for r in points
                            if r[k] is not None
                            and (not k.startswith("swapped_") or r["distinct_answer"])
                        ]
                        mechanism.append(
                            dict(
                                **label,
                                fork=f,
                                metric=k,
                                n=len(subset),
                                mean=float(np.mean(subset)) if subset else None,
                            )
                        )
    contrasts = []
    for (task, seed, split, ref, fork), arms in paired.items():
        contrasts.extend(
            paired_rows(
                dict(
                    task=task,
                    seed=seed,
                    split=split,
                    reference=ref,
                    fork=fork,
                    policy="wall",
                ),
                arms,
            )
        )
    with exclusive(out):
        for name, contents in (
            ("metrics", rows),
            ("paired", contrasts),
            ("training", training),
            ("diagnostics", mechanism),
            ("interventions", interventions),
        ):
            table(out / f"{name}.csv", contents)
        write(
            out / "coverage.json",
            dict(
                study=fingerprint(spec),
                splits=splits,
                missing=missing,
                complete=not missing,
                completed_models=len(training),
            ),
        )
        md = [
            "# Residual reasoning results",
            "",
            f"Coverage: {len(training)} completed model records, {len(missing)} missing model or evaluation records.",
            "",
            "The table uses the development-calibrated continuation time budget. "
            "Rates are means over available training seeds with a defined denominator. "
            "The denominators column sums example observations over seeds. "
            "Inspect split and seed coverage before drawing conclusions.",
            "",
            "| Task | Split | Condition | Reference | Arm | Fork | Seeds | Solve | Repair | Retain | Wrong / right |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        focus = defaultdict(list)
        for r in rows:
            if (
                r["policy"] == "wall"
                and r["fork"]
                == (
                    max(cfg["evaluation"]["forks"]) if r["condition"] == "shared" else 0
                )
                and r["arm"]
                in {
                    "identity",
                    "direct",
                    "recurrent",
                    "gated",
                    "stateless",
                    "residual_map",
                    "search1",
                    "search2",
                    "unrestricted_free",
                }
            ):
                key = tuple(
                    r[k]
                    for k in ("task", "split", "condition", "reference", "arm", "fork")
                )
                focus[key].append(r)
        for key, rs in sorted(focus.items()):

            def pct(metric):
                vals = [r[metric] for r in rs if r[metric] is not None]
                return f"{100 * np.mean(vals):.1f}%" if vals else "—"

            totals = f"{sum(r['wrong'] for r in rs)} / {sum(r['right'] for r in rs)}"
            md.append(
                "| "
                + " | ".join(
                    map(
                        str,
                        [
                            *key,
                            len(rs),
                            pct("solve"),
                            pct("repair"),
                            pct("retention"),
                            totals,
                        ],
                    )
                )
                + " |"
            )
        md += [
            "",
            "[All policy endpoints and denominators](metrics.csv) · "
            "[Paired recurrent-minus-comparator differences](paired.csv) · "
            "[Training costs](training.csv) · [Mechanism probes](diagnostics.csv) · "
            "[Matched interventions](interventions.csv) · "
            "[Coverage and missing records](coverage.json)",
            "",
        ]
        with atomic(out / "README.md") as tmp:
            tmp.write_text("\n".join(md))
    if require_complete and missing:
        raise ValueError(
            f"{len(missing)} records still missing. See {out / 'coverage.json'}"
        )
    print(out / "README.md")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument(
        "--splits",
        nargs="+",
        choices=["dev", "test", "long"],
        default=None,
    )
    p.add_argument("--require-complete", action="store_true")
    a = p.parse_args()
    aggregate(
        a.root,
        a.splits
        or read(a.root / "study.json")["config"].get("splits", ["dev", "test", "long"]),
        a.require_complete,
    )


if __name__ == "__main__":
    main()
