"""Rebuild experimental inputs from authored text and two primary HTML bodies.

HTML and reconstructed full official text remain in the user-selected local
directory. Changed content fails closed instead of silently changing evidence.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import requests
from prepare_sources import paragraphs

ROOT = Path(__file__).parent


def reconstruct(raw_directory):
    data = json.loads((ROOT / "reference.json").read_text())
    data = copy.deepcopy(data)
    for doc in data["documents"]:
        if doc["group"] != "official":
            continue
        raw = (Path(raw_directory) / (doc["id"] + ".html")).read_bytes()
        text, evidence = paragraphs(raw)
        source = doc["source"]
        assert hashlib.sha256(text.encode()).hexdigest() == source["text_sha256"], doc[
            "id"
        ]
        assert {
            k: hashlib.sha256(v.encode()).hexdigest() for k, v in evidence.items()
        } == source["paragraphs"]
        doc["evidence"] = evidence
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--download", action="store_true")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.download:
        for source in json.loads((ROOT / "sources.json").read_text()):
            path = out / (source["id"] + ".html")
            if path.exists():
                continue
            response = requests.get(source["url"], timeout=60)
            response.raise_for_status()
            path.write_bytes(response.content)
    data = reconstruct(out)
    value = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    expected = json.loads((ROOT / "evaluation_freeze.json").read_text())[
        "private_input_sha256"
    ]
    assert hashlib.sha256(value.encode()).hexdigest() == expected
    (out / "inputs.json").write_text(value)
    print("Input hash matches the frozen evaluation input.")


if __name__ == "__main__":
    main()
