"""Save first natural-language revision responses and exact request identities."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import requests
from bridge import ROOT, MODELS, sha
from revision_chain import slot_request, direct_request


def evidence(previous_cache):
    data = json.loads((Path(previous_cache) / "curated_inputs.json").read_text())
    b = next(b for b in data["bundles"] if b["id"] == "seoul")
    return {
        phase: {k: v for k, v in b[phase].items() if k.endswith("CAP")}
        for phase in ["before", "after"]
    }


def jobs(source, model):
    cases = json.loads((ROOT / "revision_reference.json").read_text())["cases"]
    for c in cases:
        yield "slots", c["id"], slot_request(c["claim"], model)
        for phase in ["before", "after"]:
            yield phase, c["id"], direct_request(source[phase], c["claim"], model)


def run(previous_cache, cache):
    base = "http://127.0.0.1:11434"
    source = evidence(previous_cache)
    tags = requests.get(base + "/api/tags", timeout=20).json()["models"]
    version = requests.get(base + "/api/version", timeout=20).json()
    for model in MODELS:
        digest = next(t["digest"] for t in tags if t["name"] == MODELS[model])
        for stage, key, req in jobs(source, model):
            path = Path(cache) / model / stage / (key + ".json")
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                old = json.loads(path.read_text())
                assert (
                    old["request_sha256"] == sha(req) and old["model_digest"] == digest
                )
                continue
            start = time.monotonic()
            response = requests.post(base + "/api/chat", json=req, timeout=360)
            response.raise_for_status()
            raw = response.json()
            try:
                parsed = json.loads(raw.get("message", {}).get("content", ""))
            except json.JSONDecodeError:
                parsed = None
            record = dict(
                id=key,
                stage=stage,
                model=model,
                model_name=MODELS[model],
                model_digest=digest,
                server=version,
                created_utc=datetime.now(timezone.utc).isoformat(),
                request_sha256=sha(req),
                input_sha256=sha(req["messages"]),
                response=raw,
                parsed=parsed,
                elapsed_seconds=round(time.monotonic() - start, 3),
            )
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
            print(model, stage, key, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--previous-cache", required=True)
    p.add_argument("--cache", required=True)
    a = p.parse_args()
    run(a.previous_cache, a.cache)
