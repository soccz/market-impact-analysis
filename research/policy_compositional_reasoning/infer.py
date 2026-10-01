"""Independent policy extraction, natural-language profile parsing, direct control."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import requests
from logic import FIELDS, OPS

ROOT = Path(__file__).resolve().parent
MODELS = {"qwen": "qwen3.5:9b", "kanana-public": "policy-kanana15-public:q4km-f7ae0cc1"}
OPTIONS = dict(
    temperature=0,
    seed=20261001,
    num_ctx=8192,
    num_predict=512,
    top_k=1,
    top_p=1,
    repeat_penalty=1,
    presence_penalty=0,
)


def sha(v):
    return hashlib.sha256(
        v
        if isinstance(v, bytes)
        else json.dumps(v, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def obj(p):
    return {
        "type": "object",
        "properties": p,
        "required": list(p),
        "additionalProperties": False,
    }


def enum(v):
    return {"type": "string", "enum": list(v)}


def arr(item):
    return {"type": "array", "items": item}


SCALAR = {"type": ["number", "boolean", "string", "null"]}


def request(bundle, case, model, stage):
    prompts = json.loads((ROOT / "prompts.json").read_text())
    options = dict(OPTIONS)
    if stage == "policy":
        payload = {
            "scope": bundle["scope"],
            "fields": FIELDS,
            "evidence": bundle["evidence"],
        }
        value = {"anyOf": [SCALAR, arr(SCALAR)]}
        atom = obj(
            {
                "field": enum(FIELDS),
                "op": enum(OPS),
                "value": value,
                "evidence": arr(enum(bundle["evidence"])),
            }
        )
        schema = obj(
            {"status": enum(["ready", "insufficient"]), "clauses": arr(arr(atom))}
        )
        options["num_predict"] = 1536
    elif stage == "profile":
        payload = {"scope": bundle["scope"], "fields": FIELDS, "claim": case["claim"]}
        schema = obj(
            {
                "status": enum(["ready", "unsupported"]),
                "assertion": {"type": "boolean"},
                "values": arr(obj({"field": enum(FIELDS), "value": SCALAR})),
            }
        )
    else:
        payload = {
            "scope": bundle["scope"],
            "evidence": bundle["evidence"],
            "claim": case["claim"],
        }
        schema = obj(
            {
                "decision": enum(["supported", "contradicted", "not_established"]),
                "evidence": arr(enum(bundle["evidence"])),
            }
        )
        options["num_predict"] = 256
    req = {
        "model": MODELS[model],
        "messages": [
            {"role": "system", "content": prompts[stage]},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "options": options,
        "format": schema,
        "stream": False,
        "keep_alive": "5m",
    }
    if model == "qwen":
        req["think"] = False
    return req


def run(inputs, out, model, base="http://127.0.0.1:11434"):
    data = json.loads(Path(inputs).read_text())
    out = Path(out)
    bs = {b["id"]: b for b in data["bundles"]}
    digest = next(
        t["digest"]
        for t in requests.get(base + "/api/tags", timeout=20).json()["models"]
        if t["name"] == MODELS[model]
    )
    version = requests.get(base + "/api/version", timeout=20).json()
    jobs = [("policy", b["id"], b, None) for b in data["bundles"]] + [
        (s, c["id"], bs[c["bundle"]], c)
        for c in data["cases"]
        for s in ["profile", "direct"]
    ]
    for stage, key, b, c in jobs:
        req = request(b, c, model, stage)
        path = out / stage / (key + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            old = json.loads(path.read_text())
            assert old["request_sha256"] == sha(req) and old["model_digest"] == digest
            continue
        start = time.monotonic()
        response = requests.post(base + "/api/chat", json=req, timeout=360)
        response.raise_for_status()
        raw = response.json()
        try:
            parsed = json.loads(raw.get("message", {}).get("content", ""))
        except json.JSONDecodeError:
            parsed = None
        record = {
            "id": key,
            "stage": stage,
            "model": model,
            "model_name": MODELS[model],
            "model_digest": digest,
            "server": version,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "request_sha256": sha(req),
            "input_sha256": sha(req["messages"]),
            "response": raw,
            "parsed": parsed,
            "elapsed_seconds": round(time.monotonic() - start, 3),
        }
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"model": model, "stage": stage, "id": key}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", choices=MODELS, required=True)
    a = p.parse_args()
    run(a.inputs, a.out, a.model)
