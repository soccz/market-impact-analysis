"""Development-informed structured reading; original baseline runner is frozen."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import requests
from infer import ROOT, MODELS, OPTIONS, SCHEMA, digest, messages

SCOPED_SCHEMA = {
    "type": "object",
    "properties": {
        "binding": {"type": "string"},
        "rule": {"type": "string"},
        "certainty": {"type": "string", "enum": ["definite", "possible", "missing"]},
        "evidence": SCHEMA["properties"]["evidence"],
        "decision": SCHEMA["properties"]["decision"],
    },
    "required": ["binding", "rule", "certainty", "evidence", "decision"],
    "additionalProperties": False,
}


def run(inputs, out, model, base):
    rows = json.loads(Path(inputs).read_text())
    prompt = (ROOT / "scoped_prompt.txt").read_text()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tags = requests.get(base + "/api/tags", timeout=20).json()["models"]
    tag = next(t for t in tags if t["name"] == MODELS[model])
    version = requests.get(base + "/api/version", timeout=20).json()
    for i, case in enumerate(rows):
        path = out / (case["id"] + ".json")
        request = dict(
            model=MODELS[model],
            messages=messages(case, prompt),
            stream=False,
            format=SCOPED_SCHEMA,
            options=OPTIONS,
            keep_alive="5m",
        )
        if model == "qwen":
            request["think"] = False
        request_hash = digest(
            json.dumps(request, ensure_ascii=False, sort_keys=True).encode()
        )
        if path.exists():
            prior = json.loads(path.read_text())
            assert (
                prior["request_sha256"] == request_hash
                and prior["model_digest"] == tag["digest"]
            )
            continue
        start = time.time()
        r = requests.post(base + "/api/chat", json=request, timeout=300)
        r.raise_for_status()
        response = r.json()
        raw = response.get("message", {}).get("content", "")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        record = dict(
            id=case["id"],
            method="scoped",
            model=model,
            model_name=MODELS[model],
            model_digest=tag["digest"],
            server=version,
            created_utc=datetime.now(timezone.utc).isoformat(),
            request_sha256=request_hash,
            input_sha256=digest(
                json.dumps(
                    {"sentences": case["sentences"], "claim": case["claim"]},
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode()
            ),
            response=response,
            parsed=parsed,
            elapsed_seconds=round(time.time() - start, 3),
        )
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "completed": i + 1,
                    "total": len(rows),
                    "id": case["id"],
                    "model": model,
                    "method": "scoped",
                    "seconds": record["elapsed_seconds"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--base", default="http://127.0.0.1:11434")
    a = p.parse_args()
    run(a.inputs, a.output, a.model, a.base)
