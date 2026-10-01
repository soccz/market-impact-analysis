"""Run only independent fresh reads for the two fixed retrieval repair conditions."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import requests
from core import sha
from infer import MODELS, request


def run(inputs, output, model, base="http://127.0.0.1:11434"):
    data = json.loads(Path(inputs).read_text())
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    digest = next(
        t["digest"]
        for t in requests.get(base + "/api/tags", timeout=20).json()["models"]
        if t["name"] == MODELS[model]
    )
    version = requests.get(base + "/api/version", timeout=20).json()
    bs = {b["id"]: b for b in data["bundles"]}
    for case in data["cases"]:
        req = request(bs[case["bundle"]], case, model, "after")
        path = out / (case["id"] + ".json")
        if path.exists():
            r = json.loads(path.read_text())
            assert r["request_sha256"] == sha(req) and r["model_digest"] == digest
            continue
        start = time.monotonic()
        r = requests.post(base + "/api/chat", json=req, timeout=360)
        r.raise_for_status()
        raw = r.json()
        try:
            parsed = json.loads(raw.get("message", {}).get("content", ""))
        except json.JSONDecodeError:
            parsed = None
        record = dict(
            id=case["id"],
            stage="after",
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
        print(json.dumps({"model": model, "id": case["id"]}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", choices=MODELS, required=True)
    a = p.parse_args()
    run(a.inputs, a.output, a.model)
