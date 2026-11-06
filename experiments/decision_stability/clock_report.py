"""Independently verify and regenerate a completed clock-crossover report."""
import argparse
import json
from pathlib import Path

from .clock import summarize, write_report
from .model import sha256


def verify(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest.get("experiment") != "clock_crossover" or manifest["status"] != "complete":
        raise ValueError("Expected a completed clock-crossover run")
    for name, expected in manifest["output_sha256"].items():
        if sha256(folder / name) != expected:raise ValueError(f"Output hash mismatch: {name}")
    def read(name):
        return [json.loads(line) for line in (folder / name).read_text().splitlines() if line.strip()]
    anchors = json.loads((folder / "anchors.json").read_text())
    d, a, c = manifest["design"], manifest["arguments"], manifest["model_config"]
    actual = summarize(read("prefix.jsonl"), read("branches.jsonl"), anchors, d["native_l"],
                       c["H_cycles"], c["L_layers"], d["branch_blocks"], d["followup_outer"],
                       a["bootstrap"], a["seed"], a["min_solved"])
    if actual != json.loads((folder / "summary.json").read_text()):
        raise ValueError("Summary disagrees with raw observations")
    write_report(folder, manifest, actual)
    return actual


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    args = p.parse_args()
    result = verify(args.run)
    print(json.dumps(dict(verified=True, correct_at_fork=result["correct_at_fork"],
                          precision_gate=result["precision_gate"])))
