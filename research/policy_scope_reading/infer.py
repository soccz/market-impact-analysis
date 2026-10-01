"""Run frozen prompts, retain first outputs, and never send reference labels."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import requests

ROOT = Path(__file__).resolve().parent
SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["supported", "contradicted", "not_established"],
        },
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["decision", "evidence"],
    "additionalProperties": False,
}
MODELS = {"kanana": "ra-kanana15-8b:q4_k_m-20260927", "qwen": "qwen3.5:9b"}
OPTIONS = dict(
    temperature=0,
    seed=20261001,
    num_ctx=8192,
    num_predict=256,
    top_k=1,
    top_p=1,
    repeat_penalty=1,
    presence_penalty=0,
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def messages(case, prompt):
    text = (
        "근거 문장:\n"
        + "\n".join(f"[{k}] {v}" for k, v in case["sentences"].items())
        + "\n\n판독할 주장:\n"
        + case["claim"]
    )
    return [{"role": "system", "content": prompt}, {"role": "user", "content": text}]


def run(inputs, out, model, method, base):
    rows = json.loads(Path(inputs).read_text())
    prompts = json.loads((ROOT / "prompts.json").read_text())
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tags = requests.get(base + "/api/tags", timeout=20).json()["models"]
    tag = next(t for t in tags if t["name"] == MODELS[model])
    version = requests.get(base + "/api/version", timeout=20).json()
    for i, case in enumerate(rows):
        path = out / (case["id"] + ".json")
        request = dict(
            model=MODELS[model],
            messages=messages(case, prompts[method]),
            stream=False,
            format=SCHEMA,
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
            method=method,
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
                    "method": method,
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
    p.add_argument("--method", default="baseline")
    p.add_argument("--base", default="http://127.0.0.1:11434")
    a = p.parse_args()
    run(a.inputs, a.output, a.model, a.method, a.base)
