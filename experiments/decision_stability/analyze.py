"""Recompute summaries from saved records without loading a model or checkpoint."""
import argparse
import json
from pathlib import Path

from .metrics import summarize_interventions, summarize_rollouts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Refusing to summarize an incomplete run")
    def read(name):
        with (args.run / name).open() as handle:
            return [json.loads(line) for line in handle if line.strip()]
    natural = summarize_rollouts(read("rollouts.jsonl"))
    paired = read("interventions.jsonl")
    print(json.dumps(dict(run_kind=manifest["run_kind"], natural=natural,
                          interventions=summarize_interventions(paired) if paired else []), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
