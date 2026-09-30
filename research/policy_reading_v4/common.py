"""Paths and immutable v3 dependencies for the next, explicitly post-hoc study."""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
V3 = ROOT.parent / "policy_reading_v3"
sys.path.insert(0, str(V3))


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
    return [json.loads(line) for line in path.read_text().splitlines()]


def datasets(scope=False):
    data = {
        n: read_rows(V3 / "data" / f"{n}.jsonl")
        for n in [
            "train",
            "validation",
            "evaluation",
            "units",
            "distractor",
            "official",
        ]
    }
    if scope:
        data["train"] = read_rows(V3 / "augmentation_data/scope.jsonl")
    data["combined"] = read_rows(ROOT / "data/combined.jsonl")
    return data


def check_lock():
    lock = json.loads((ROOT / "freeze.json").read_text())
    for rel, digest in lock["files"].items():
        assert sha(ROOT / rel) == digest, rel
    return lock
