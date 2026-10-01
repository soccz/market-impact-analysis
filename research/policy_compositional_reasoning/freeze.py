"""Write-once local protocol snapshots; hashes are not third-party registration."""

import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
CORE = [
    "fields.json",
    "logic.py",
    "worlds.py",
    "prompts.json",
    "infer.py",
    "test_logic.py",
    "protocol.json",
]


def freeze(name, files):
    dest = ROOT / (name + "_freeze.json")
    assert not dest.exists(), dest
    data = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        files={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files},
    )
    dest.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(dest.name)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("files", nargs="*")
    a = p.parse_args()
    freeze(a.name, CORE + a.files)
