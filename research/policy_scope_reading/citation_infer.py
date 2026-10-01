"""Post-hoc citation-only ablation; baseline prompt and decoder remain unchanged."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import requests
from infer import ROOT, OPTIONS, SCHEMA, digest, messages

MODELS = {
    "kanana-public": "policy-kanana15-public:q4km-f7ae0cc1",
    "qwen": "qwen3.5:9b",
}


def make_request(case, model, method):
    prompt = json.loads((ROOT / "prompts.json").read_text())["baseline"]
    schema = json.loads(json.dumps(SCHEMA))
    schema["properties"]["evidence"]["items"]["enum"] = list(case["sentences"])
    schema["properties"]["evidence"]["minItems"] = 1
    request = dict(
        model=MODELS[model],
        messages=messages(case, prompt),
        stream=False,
        format=schema,
        options=OPTIONS,
        keep_alive="5m",
    )
    if model == "qwen":
        request["think"] = False
    return request


def run(inputs, output, model, method, base="http://127.0.0.1:11434"):
    rows = json.loads(Path(inputs).read_text())
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    tags = requests.get(base + "/api/tags", timeout=20).json()["models"]
    tag = next(t for t in tags if t["name"] == MODELS[model])
    version = requests.get(base + "/api/version", timeout=20).json()
    for i, case in enumerate(rows, 1):
        request = make_request(case, model, method)
        rh = digest(json.dumps(request, ensure_ascii=False, sort_keys=True).encode())
        path = out / (case["id"] + ".json")
        if path.exists():
            old = json.loads(path.read_text())
            assert old["request_sha256"] == rh and old["model_digest"] == tag["digest"]
            continue
        start = time.time()
        response = requests.post(base + "/api/chat", json=request, timeout=300)
        response.raise_for_status()
        response = response.json()
        try:
            parsed = json.loads(response.get("message", {}).get("content", ""))
        except json.JSONDecodeError:
            parsed = None
        record = dict(
            id=case["id"],
            method=method,
            model=model,
            model_name=MODELS[model],
            model_digest=tag["digest"],
            server=version,
            created_utc=datetime.now(timezone.utc).isoformat(),
            request_sha256=rh,
            input_sha256=digest(
                json.dumps(
                    {"sentences": case["sentences"], "claim": case["claim"]},
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode()
            ),
            parsed=parsed,
            response=response,
            elapsed_seconds=round(time.time() - start, 3),
        )
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        print(
            json.dumps(
                dict(
                    completed=i,
                    total=len(rows),
                    model=model,
                    method=method,
                    id=case["id"],
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--method", choices=["citation-only"], required=True)
    p.add_argument("--base", default="http://127.0.0.1:11434")
    a = p.parse_args()
    run(a.inputs, a.output, a.model, a.method, a.base)
