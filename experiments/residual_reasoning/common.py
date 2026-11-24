"""Small, dependency-free run records and atomic file operations."""

import hashlib
import json
import os
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def json_text(obj):
    return json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n"


def fingerprint(obj):
    return hashlib.sha256(json_text(obj).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


@contextmanager
def atomic(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(fd)
    tmp = Path(tmp)
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write(path, obj):
    with atomic(path) as tmp:
        tmp.write_text(json_text(obj))


@contextmanager
def exclusive(path):
    import fcntl

    path.mkdir(parents=True, exist_ok=True)
    with (path / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another writer owns {path}") from exc
        yield


def source():
    root = Path(__file__).resolve().parents[2]
    files = sorted(Path(__file__).parent.glob("*.py"))
    return dict(
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        code_sha256=fingerprint({p.name: digest(p) for p in files}),
    )
