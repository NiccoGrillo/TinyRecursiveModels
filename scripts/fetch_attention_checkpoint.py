"""Fetch the pinned attention checkpoint for docs/NEXT_EXPERIMENT.md; no remote code execution."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


REPOSITORY = "gaoxin492/TinyRecursiveModels-Sudoku-Extreme-att"
REVISION = "33991b882368f425916d625a9ee614a7a85efe76"
FILES = {
    "step_21700": "932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3",
    "all_config.yaml": "9e595c896d6484e9dbb3c3963d11bac224dd91d70eb040f11de04ac8303be6b3",
    "trm.py": "3454818b64a1abb45c062782d380aa4bb560e2cb35db9cc309cf367ca3ac262c",
}


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fetch(out):
    out.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, expected in FILES.items():
        url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}"
        target = out / ("published_trm.py" if name == "trm.py" else name)
        if not target.exists():
            temp = target.with_name(target.name + ".partial")
            try:
                with urllib.request.urlopen(url, timeout=45) as src, temp.open("xb") as dst:
                    while block := src.read(1024 * 1024):
                        dst.write(block)
                if digest(temp) != expected:
                    raise ValueError(f"Downloaded file hash mismatch: {name}")
                temp.rename(target)
            except Exception:
                # Keep a partial file for inspection; do not silently retry over it.
                raise
        if digest(target) != expected:
            raise ValueError(f"Existing file hash mismatch: {target}")
        records[name] = dict(url=url, sha256=expected, bytes=target.stat().st_size)
        print(f"Verified {target}", flush=True)
    record = dict(repository=REPOSITORY, revision=REVISION, files=records,
                  note="Third-party reproduction; training/EMA provenance is not independently established. Published Python is data only and is never executed.")
    (out / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/gaoxin492-sudoku-att-33991b8"))
    fetch(parser.parse_args().out)
