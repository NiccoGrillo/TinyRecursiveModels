"""Compare sealed Sudoku timing traces on identical held-out input IDs."""

import argparse
import json
import statistics
from pathlib import Path


BUDGETS_MS = (10, 30, 100, 300, 750)


def at_budget(row, budget_ms):
    affordable = [ok for seconds, ok in zip(row["seconds"], row["correct"])
                  if seconds * 1000 <= budget_ms]
    return affordable[-1] if affordable else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("direct", "recurrent", "trm", "fprm", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    logs = {name: json.loads(getattr(args, name).read_text())
            for name in ("direct", "recurrent", "trm", "fprm")}
    direct = logs["direct"]
    ids = [row["id"] for row in direct["rows"]]
    if len(ids) != 128 or len(ids) != len(set(ids)):
        raise ValueError("Expected 128 distinct held-out boards")
    records = {}
    for name, log in logs.items():
        rows = log["rows"]
        if (log["data_sha256"] != direct["data_sha256"]
                or log["split"] != "test" or log["device"] != direct["device"]
                or [row["id"] for row in rows] != ids):
            raise ValueError(f"Mismatched input IDs, source data, or hardware for {name}")
        normal = [(row["correct"][i], row["seconds"][i]) for row in rows
                  if (i := row["normal_readout_index"]) is not None]
        record = dict(checkpoint=log["checkpoint_sha256"], n=len(ids),
                      normal_correct=sum(ok for ok, _ in normal), normal_observed=len(normal),
                      normal_median_ms=1000 * statistics.median(t for _, t in normal)
                      if normal else None, ever_correct=sum(any(row["correct"]) for row in rows))
        by_time = {}
        for budget in BUDGETS_MS:
            outcomes = [at_budget(row, budget) for row in rows]
            baseline = [at_budget(row, budget) for row in direct["rows"]]
            by_time[budget] = dict(correct=outcomes.count(True),
                                   observed=len(ids) - outcomes.count(None),
                                   direct_only=sum(bool(a) and not b for a, b in zip(baseline, outcomes)),
                                   comparator_only=sum(not a and bool(b) for a, b in zip(baseline, outcomes)))
        record["by_time_ms"] = by_time
        records[name] = record
    with args.output.open("x") as handle:
        json.dump(dict(data_sha256=direct["data_sha256"], ids=ids, arms=records), handle, indent=2)
        handle.write("\n")
    print(json.dumps(records))


if __name__ == "__main__":
    main()
