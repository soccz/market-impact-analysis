"""Local write-once protocol snapshots; not an external preregistration."""

import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent


def freeze(name, files):
    path = ROOT / (name + "_freeze.json")
    assert not path.exists(), path
    data = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        files={f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files},
    )
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    return data


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("files", nargs="+")
    a = p.parse_args()
    freeze(a.name, a.files)
