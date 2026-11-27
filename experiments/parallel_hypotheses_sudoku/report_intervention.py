"""Paired exact-board accounting for a complete frozen intervention run."""
import json
import sys
from collections import Counter
from pathlib import Path
from experiments.decision_stability.clock import paired_binary_interval

rows=json.loads(Path(sys.argv[1]).read_text())["rows"]
assert len(rows)==len({r["id"] for r in rows})==256
scores={k:[r["exact"][k] for r in rows] for k in rows[0]["exact"]}
comparisons={k:paired_binary_interval(scores["aligned"],scores[k]) for k in ("shuffled","untouched","serial")}
new=sum(r["exact"]["aligned"] and not(r["exact"]["a"] or r["exact"]["b"]) for r in rows)
advance=(all(v["mean"]>0 for v in comparisons.values()) and comparisons["shuffled"]["lower"]>0
         and comparisons["untouched"]["lower"]>0 and new>0)
print(json.dumps(dict(exact={k:sum(v) for k,v in scores.items()},comparisons=comparisons,
    status=Counter(r["status"] for r in rows),new_from_branch_endpoints=new,advance=advance,
    aligned_first_step_changed=sum(r["first_changes"]["aligned"]>0 for r in rows)),indent=2))
