"""Parse source paragraphs without dropping HWP tables; keep full text private."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import unicodedata
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def paragraphs(path):
    executable = shutil.which("hwp5proc")
    if executable is None:
        raise RuntimeError("Install pyhwp and add hwp5proc to PATH")
    root = ET.fromstring(subprocess.check_output([executable, "xml", str(path)]))
    parents = {c: n for n in root.iter() for c in n}
    out = []
    for para in root.iter("Paragraph"):
        texts = []
        for node in para.iter("Text"):
            owner = parents[node]
            while owner.tag != "Paragraph":
                owner = parents[owner]
            if owner is para:
                texts.append(node.text or "")
        text = re.sub(r"\s+", " ", unicodedata.normalize("NFC", "".join(texts))).strip()
        if text:
            out.append(dict(id=f"P{len(out)+1:03}", text=text))
    return out


def prepare(cache):
    cache = Path(cache)
    docs = {}
    for source in json.loads((ROOT / "sources.json").read_text()):
        path = cache / source["cache_file"]
        assert sha(path.read_bytes()) == source["sha256"], source["id"]
        rows = paragraphs(path)
        text = "".join(r["text"] for r in rows)
        assert source["notice_number"] in re.sub(r"\s+", "", text).replace("–", "-")
        docs[source["id"]] = {r["id"]: r["text"] for r in rows}
        (cache / (source["id"] + "-paragraphs.json")).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
        )
    return docs


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    a = p.parse_args()
    print({k: len(v) for k, v in prepare(a.cache).items()})
