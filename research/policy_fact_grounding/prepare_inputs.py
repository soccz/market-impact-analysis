"""Rebuild scoped inputs from official HWP files, including nested table text."""

import argparse
import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).parent


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def extract(path):
    from hwp5.xmlmodel import Hwp5File

    hwp = Hwp5File(str(path))
    try:
        root = ET.fromstring(b"".join(hwp.xmlevents().bytechunks()))
    finally:
        hwp.close()
    paragraphs = []
    for p in root.findall(".//Paragraph"):
        text = "".join(t.text or "" for t in p.findall("./LineSeg/Text"))
        if text.strip():
            paragraphs.append(text)
    return paragraphs


def build(source_dir, reference):
    texts = {}
    for source in json.loads((ROOT / "sources.json").read_text()):
        path = Path(source_dir) / source["file"]
        assert digest(path.read_bytes()) == source["sha256"]
        text = extract(path)
        assert (
            digest(json.dumps(text, ensure_ascii=False).encode())
            == source["extracted_sha256"]
        )
        texts[source["id"]] = text
    data = copy.deepcopy(reference)
    spans = json.loads((ROOT / "source_spans.json").read_text())
    for b in data["bundles"]:
        for key in b["evidence"]:
            s = spans[key]
            text = "\n".join(texts[s["source"]][i] for i in s["paragraph_indices"])
            assert digest(text.encode()) == s["sha256"]
            b["evidence"][key] = text
    return data


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sources", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--download", action="store_true")
    a = p.parse_args()
    if a.download:
        import requests

        Path(a.sources).mkdir(parents=True, exist_ok=True)
        for s in json.loads((ROOT / "sources.json").read_text()):
            response = requests.get(s["url"], timeout=60)
            response.raise_for_status()
            assert digest(response.content) == s["sha256"], s["id"]
            (Path(a.sources) / s["file"]).write_bytes(response.content)
    data = build(a.sources, json.loads((ROOT / "official_reference.json").read_text()))
    Path(a.out).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(dict(bundles=len(data["bundles"]), cases=len(data["cases"])))
