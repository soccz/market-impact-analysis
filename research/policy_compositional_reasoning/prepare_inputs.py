"""Rebuild scoped inputs from two versioned documents, without redistributing text."""

import argparse, copy, hashlib, json, zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import fitz

ROOT = Path(__file__).parent


def digest(b):
    return hashlib.sha256(b).hexdigest()


def extract(path, kind):
    if kind == "pdf":
        with fitz.open(path) as doc:
            return [p.get_text() for p in doc]
    with zipfile.ZipFile(path) as z:
        ns = {"hp": "http://www.hancom.co.kr/hwpml/2011/paragraph"}
        out = []
        for n in sorted(z.namelist()):
            if n.startswith("Contents/section") and n.endswith(".xml"):
                e = ET.fromstring(z.read(n))
                for p in e.iter("{" + ns["hp"] + "}p"):
                    t = "".join(
                        "".join(v.itertext()) for v in p.findall("./hp:run/hp:t", ns)
                    )
                    if t.strip():
                        out.append(t)
        return out


def build(source_dir, reference):
    sources = json.loads((ROOT / "sources.json").read_text())
    texts = {}
    for source in sources:
        p = Path(source_dir) / source["file"]
        assert digest(p.read_bytes()) == source["sha256"], p
        texts[source["id"]] = extract(p, source["kind"])
        assert (
            digest(json.dumps(texts[source["id"]], ensure_ascii=False).encode())
            == source["extracted_sha256"]
        )
    spans = json.loads((ROOT / "source_spans.json").read_text())
    data = copy.deepcopy(reference)
    for b in data["bundles"]:
        evidence = {}
        for key in b["evidence"]:
            s = spans[key]
            text = texts[s["source"]][s["unit_index"]][s["start"] : s["end"]]
            assert digest(text.encode()) == s["sha256"], key
            evidence[key] = text
        b["evidence"] = evidence
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", required=True)
    p.add_argument("--reference", default=str(ROOT / "official_reference.json"))
    p.add_argument("--out", required=True)
    p.add_argument("--download", action="store_true")
    a = p.parse_args()
    if a.download:
        import requests, warnings

        for s in json.loads((ROOT / "sources.json").read_text()):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = requests.get(s["url"], timeout=60, verify=s["tls_verified"])
            r.raise_for_status()
            assert digest(r.content) == s["sha256"], s["id"]
            dest = Path(a.sources) / s["file"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(r.content)
    data = build(a.sources, json.loads(Path(a.reference).read_text()))
    Path(a.out).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(len(data["bundles"]), len(data["cases"]))


if __name__ == "__main__":
    main()
