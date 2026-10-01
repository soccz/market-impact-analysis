"""Download only pinned public originals to a user-selected private cache."""

import argparse
import json
from pathlib import Path

import requests
from infer import ROOT, digest


def fetch(cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    sources = json.loads((ROOT / "sources.json").read_text())
    replication = json.loads((ROOT / "replication_source.json").read_text())
    sources.append({**replication, "cache_file": "incheon-corrected.pdf"})
    for source in sources:
        path = cache / source["cache_file"]
        if path.exists():
            assert digest(path.read_bytes()) == source["sha256"], str(path)
            continue
        r = requests.get(source["url"], timeout=90)
        r.raise_for_status()
        assert (
            digest(r.content) == source["sha256"]
        ), "Source changed; preserve the old reference and investigate instead of replacing it."
        path.write_bytes(r.content)
    (cache / "incheon-retrieval.json").write_text(
        json.dumps(replication, ensure_ascii=False, indent=2) + "\n"
    )
    print(dict(source_files=len(sources), hashes_match=True))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    fetch(p.parse_args().cache)
