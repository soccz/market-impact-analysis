"""Versioned paths; older experiments are read-only dependencies."""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
V3 = ROOT.parent / "policy_reading_v3"
V4 = ROOT.parent / "policy_reading_v4"
sys.path.extend([str(V3), str(V4)])


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(2**20):
            h.update(block)
    return h.hexdigest()


def save(path, value, compact=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value, ensure_ascii=False, allow_nan=False, indent=None if compact else 2
        )
        + "\n"
    )


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def datasets():
    return {p.stem: read_rows(p) for p in sorted((ROOT / "data").glob("*.jsonl"))}


def check_lock():
    lock = json.loads((ROOT / "freeze.json").read_text())
    for path, digest in lock["files"].items():
        assert sha(ROOT / path) == digest, path
    return lock
