"""Rebuild exact private inputs from public source locators, without publishing text."""

import argparse
import json
from pathlib import Path
import fitz
import requests
from core import sha
from infer import ROOT


def canonical(path):
    with fitz.open(path) as doc:
        return [" ".join(page.get_text(sort=True).split()) for page in doc]


def build(cache, split):
    cache = Path(cache)
    sources = json.loads((ROOT / "sources.json").read_text())
    pages = {}
    for s in sources:
        path = cache / (s["id"] + ".pdf")
        assert sha(path.read_bytes()) == s["sha256"], s["id"]
        pages[s["id"]] = canonical(path)
        assert sha(pages[s["id"]]) == s["canonical_pages_sha256"], s["id"]
    ref = json.loads((ROOT / (split + "_reference.json")).read_text())
    bundles = []
    for b in ref["bundles"]:
        rebuilt = {k: b[k] for k in ["id", "kind"]}
        for phase in ["before", "after"]:
            rebuilt[phase] = {}
            for key, loc in b["source_locators"][phase].items():
                text = pages[loc["document"]][loc["page"] - 1][
                    loc["start"] : loc["end"]
                ]
                assert sha(text.encode()) == loc["text_sha256"], key
                rebuilt[phase][key] = text
        bundles.append(rebuilt)
    return {"bundles": bundles, "cases": ref["cases"]}


def public_stub(reference):
    """Text hashes preserve exact-match equivalence for source-free score replay."""
    data = {"cases": reference["cases"], "bundles": []}
    for b in reference["bundles"]:
        data["bundles"].append(
            {
                "id": b["id"],
                **{
                    phase: {
                        k: v["text_sha256"]
                        for k, v in b["source_locators"][phase].items()
                    }
                    for phase in ["before", "after"]
                },
            }
        )
    return data


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--download", action="store_true")
    a = p.parse_args()
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    if a.download:
        for s in json.loads((ROOT / "sources.json").read_text()):
            path = cache / (s["id"] + ".pdf")
            if not path.exists():
                response = requests.get(s["url"], timeout=60)
                response.raise_for_status()
                assert sha(response.content) == s["sha256"], "Attachment has changed"
                path.write_bytes(response.content)
    for split in ["curated", "retrieved"]:
        data = build(cache, split)
        (cache / (split + "_inputs.json")).write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        )
        print(split, len(data["cases"]), sha(data))
