"""Fetch eight official attachments into a user-chosen cache; never commit it."""

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

from reader import sha, read_source

ROOT = Path(__file__).resolve().parent


def load(cache):
    sources = json.loads((ROOT / "sources.json").read_text())
    docs = {}
    for source in sources:
        path = Path(cache) / (source["id"] + ".bin")
        if sha(path.read_bytes()) != source["sha256"]:
            raise ValueError("Original changed: " + source["id"])
        pages = read_source(path)
        if (
            "text_sha256" in source
            and sha(json.dumps(pages, ensure_ascii=False).encode())
            != source["text_sha256"]
        ):
            raise ValueError(
                "Text extraction changed; check PyMuPDF version: " + source["id"]
            )
        docs[source["id"]] = pages
    return docs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    args = parser.parse_args()
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    for source in json.loads((ROOT / "sources.json").read_text()):
        path = cache / (source["id"] + ".bin")
        if not path.exists():
            with urlopen(source["url"], timeout=60) as response:
                data = response.read()
            if sha(data) != source["sha256"]:
                raise ValueError("Remote attachment changed: " + source["id"])
            path.write_bytes(data)
    print("Verified", len(load(cache)), "source attachments and canonical texts")
