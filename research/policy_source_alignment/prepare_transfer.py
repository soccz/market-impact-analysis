"""Reconstruct source inputs from hashed official HTML text snapshots."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parent


def prepare(cache):
    data = json.loads((ROOT / "transfer_reference.json").read_text())
    for b in data["bundles"]:
        source = (Path(cache) / "sources" / (b["source"] + ".txt")).read_text()
        b["evidence"] = {}
        for key, span in b["source_spans"].items():
            text = source[span["start"] : span["end"]]
            assert hashlib.sha256(text.encode()).hexdigest() == span["sha256"]
            b["evidence"][key] = text
    return data


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    a = p.parse_args()
    (Path(a.cache) / "transfer_inputs.json").write_text(
        json.dumps(prepare(a.cache), ensure_ascii=False, indent=2) + "\n"
    )
