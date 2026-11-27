"""Evaluate the locked D4/color-table solver on the public ARC training split."""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

REVISION = "399030444e0ab0cc8b4e199870fb20b863846f34"
HERE = Path(__file__).resolve().parent


def geometry(grid, index):
    rows = [row[::-1] for row in grid] if index >= 4 else grid
    for _ in range(index % 4):
        rows = [list(col) for col in zip(*reversed(rows))]
    return rows


def fit(demos, index):
    table = {}
    for pair in demos:
        view = geometry(pair["input"], index)
        target = pair["output"]
        if len(view) != len(target) or len(view[0]) != len(target[0]):
            return None
        for src, dst in zip(view, target):
            for a, b in zip(src, dst):
                if a in table and table[a] != b:
                    return None
                table[a] = b
    return [table.get(color, color) for color in range(10)]


def execute(grid, index, table):
    return [[table[color] for color in row] for row in geometry(grid, index)]


def choose(demos, cross_validate=True):
    candidates = []
    for index in range(8):
        table = fit(demos, index)
        if table is None:
            continue
        correct_folds = 0
        if cross_validate:
            for k, pair in enumerate(demos):
                partial = fit(demos[:k] + demos[k + 1:], index)
                if execute(pair["input"], index, partial) == pair["output"]:
                    correct_folds += 1
        changes = sum(i != val for i, val in enumerate(table))
        candidates.append((changes, index, table, correct_folds))
    if not candidates:
        return None, None, candidates
    full = min(candidates, key=lambda c: c[:2])
    loo = min(candidates, key=lambda c: (-c[3], c[0], c[1]))
    return full, loo, candidates


def digest(obj):
    return sha256(json.dumps(obj, separators=(",", ":")).encode()).hexdigest()


def apply_queries(candidate, inputs):
    if candidate is None:
        return None
    _, index, table, _ = candidate
    return [execute(grid, index, table) for grid in inputs]


def evaluate(task_id, task):
    demos = task["train"]
    full, loo, candidates = choose(demos)
    inputs = [pair["input"] for pair in task["test"]]
    predictions = [apply_queries(c, inputs) for c in (full, loo)]
    targets = [pair["output"] for pair in task["test"]]
    row = dict(task=task_id, demos=len(demos), queries=len(inputs),
               candidates=len(candidates), full_loo_folds=0)
    for label, selected, predicted in zip(("full", "loo"), (full, loo), predictions):
        row[label] = dict(geometry=selected[1] if selected else None,
                          table=selected[2] if selected else None,
                          exact=predicted == targets,
                          prediction_sha256=digest(predicted) if predicted else None)
    for k, held in enumerate(demos):
        partial, _, _ = choose(demos[:k] + demos[k + 1:], cross_validate=False)
        row["full_loo_folds"] += apply_queries(partial, [held["input"]]) == [held["output"]]
    return row


def interval(n, total):
    p = n / total
    z = 1.959963984540054
    denominator = 1 + z ** 2 / total
    center = (p + z ** 2 / (2 * total)) / denominator
    offset = z * math.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2)) / denominator
    return [center - offset, center + offset]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    git = ["git", "-C", str(args.repository)]
    revision = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        parser.error("ARC source revision mismatch")
    dirty = subprocess.check_output(git + ["status", "--porcelain", "--", "data/training"], text=True)
    if dirty:
        parser.error("ARC training files differ from source revision")
    files = sorted((args.repository / "data/training").glob("*.json"))
    if len(files) != 400:
        parser.error("Expected 400 unaugmented public training tasks")
    rows, manifest = [], {}
    start = time.perf_counter()
    for file in files:
        raw = file.read_bytes()
        manifest[file.name] = sha256(raw).hexdigest()
        rows.append(evaluate(file.stem, json.loads(raw)))
    seconds = time.perf_counter() - start
    full = sum(r["full"]["exact"] for r in rows)
    loo = sum(r["loo"]["exact"] for r in rows)
    summary = dict(dataset_revision=revision, split="data/training", tasks=len(rows),
                   task_hashes_sha256=digest(manifest), protocol_sha256=sha256((HERE / "PROTOCOL.md").read_bytes()).hexdigest(),
                   script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                   python=sys.version, machine=platform.machine(), system=platform.platform(),
                   wall_seconds=seconds, gpu_hours=0, coverage=sum(r["candidates"] > 0 for r in rows),
                   full_exact=full, loo_exact=loo, full_wilson95=interval(full, len(rows)),
                   loo_wins=sum(r["loo"]["exact"] and not r["full"]["exact"] for r in rows),
                   loo_losses=sum(not r["loo"]["exact"] and r["full"]["exact"] for r in rows),
                   loo_geometry_changed=sum(r["loo"]["geometry"] != r["full"]["geometry"] for r in rows),
                   full_solved_identity_geometry=sum(r["full"]["exact"] and r["full"]["geometry"] == 0 for r in rows),
                   full_solved_identity_table=sum(r["full"]["exact"] and r["full"]["table"] == list(range(10)) for r in rows),
                   full_solved_both_identity=sum(r["full"]["exact"] and r["full"]["geometry"] == 0 and r["full"]["table"] == list(range(10)) for r in rows),
                   full_held_demo_fraction_task_average=sum(r["full_loo_folds"]/r["demos"] for r in rows) / len(rows))
    args.out.mkdir(exist_ok=True, parents=True)
    for name, data in (("tasks", rows), ("manifest", manifest), ("summary", summary)):
        (args.out / (name + ".json")).write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
