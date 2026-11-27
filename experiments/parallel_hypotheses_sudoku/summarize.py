"""Summarize the CPU replay after rechecking eligibility on the replayed trajectories."""
import json
import statistics
import sys
from pathlib import Path

rows = json.loads(Path(sys.argv[1]).read_text())["rows"]
assert len(rows) == len({r["id"] for r in rows}) == 76
eligible = [r for r in rows if not (r["a_exact"] or r["b_exact"] or r["serial_exact"] or r["agree"])]
def summarize(group):
    if not group:
        return {"n": 0}
    return dict(n=len(group), reciprocal=sum(bool(r["a_only"] and r["b_only"]) for r in group),
                oracle_exact=sum(r["both_wrong"] == 0 for r in group),
                pooled_exact=sum(r["pooled_exact"] for r in group),
                median_shared_wrong=statistics.median(r["both_wrong"] for r in group),
                median_a_only=statistics.median(r["a_only"] for r in group),
                median_b_only=statistics.median(r["b_only"] for r in group))
print(json.dumps(dict(archived_selection=summarize(rows), cpu_still_eligible=summarize(eligible),
                      site_and_digits_matched=sum(r["site_digits_match"] for r in rows),
                      cpu_serial_solved=sum(r["serial_exact"] for r in rows)),indent=2))
