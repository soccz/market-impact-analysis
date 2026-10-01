"""Frozen model requests; actual old cache and all first responses are retained."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import requests
from core import DECISIONS, ROUTES, decision, sha

ROOT = Path(__file__).resolve().parent
MODELS = {"qwen": "qwen3.5:9b", "kanana-public": "policy-kanana15-public:q4km-f7ae0cc1"}
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


def array(ids, nonempty=True):
    return {
        "type": "array",
        "items": {"type": "string", "enum": list(ids)},
        "minItems": int(nonempty),
    }


def obj(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def request(bundle, case, model, stage, old=None, change_map=None):
    prompts = json.loads((ROOT / "prompts.json").read_text())
    # Explicit allowlist: no annotation, reference, family or source locator enters a request.
    if stage == "before":
        payload = {"evidence": bundle["before"], "claim": case["claim"]}
        ids = bundle["before"]
    else:
        payload = {
            "previous_evidence": bundle["before"],
            "current_evidence": bundle["after"],
        }
        ids = {**bundle["before"], **bundle["after"]}
        if stage != "map":
            payload["claim"] = case["claim"]
    prompt = prompts[stage if stage in {"before", "after", "map"} else "gate"]
    options = dict(OPTIONS)
    if stage == "map":
        schema = obj(
            {
                "changes": {
                    "type": "array",
                    "items": obj(
                        {
                            "before_ids": array(bundle["before"], False),
                            "after_ids": array(bundle["after"], False),
                            "kind": {
                                "type": "string",
                                "enum": ["rule", "scope", "wording", "unresolved"],
                            },
                            "change": {"type": "string"},
                            "applies_to": {"type": "string"},
                        }
                    ),
                }
            }
        )
        options["num_predict"] = 1024
    elif stage in {"direct_gate", "map_gate"}:
        payload["previous_model_evidence_ids"] = old["evidence"]
        if stage == "map_gate":
            payload["provisional_model_change_map"] = change_map
            prompt += "\n" + prompts["map_gate_suffix"]
        schema = obj(
            {"route": {"type": "string", "enum": ROUTES}, "evidence": array(ids)}
        )
    else:
        schema = obj(
            {"decision": {"type": "string", "enum": DECISIONS}, "evidence": array(ids)}
        )
    result = dict(
        model=MODELS[model],
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        stream=False,
        format=schema,
        options=options,
        keep_alive="5m",
    )
    if model == "qwen":
        result["think"] = False
    return result


def run(inputs, output, model, base="http://127.0.0.1:11434"):
    data = json.loads(Path(inputs).read_text())
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    tag = next(
        t
        for t in requests.get(base + "/api/tags", timeout=20).json()["models"]
        if t["name"] == MODELS[model]
    )
    version = requests.get(base + "/api/version", timeout=20).json()

    def execute(key, stage, req):
        path = out / stage / (key + ".json")
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            record = json.loads(path.read_text())
            assert (
                record["request_sha256"] == sha(req)
                and record["model_digest"] == tag["digest"]
            )
            return record
        start = time.monotonic()
        response = requests.post(base + "/api/chat", json=req, timeout=360)
        response.raise_for_status()
        response = response.json()
        try:
            parsed = json.loads(response.get("message", {}).get("content", ""))
        except json.JSONDecodeError:
            parsed = None
        record = dict(
            id=key,
            stage=stage,
            model=model,
            model_name=MODELS[model],
            model_digest=tag["digest"],
            server=version,
            created_utc=datetime.now(timezone.utc).isoformat(),
            request_sha256=sha(req),
            input_sha256=sha(req["messages"]),
            response=response,
            parsed=parsed,
            elapsed_seconds=round(time.monotonic() - start, 3),
        )
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"model": model, "stage": stage, "id": key}), flush=True)
        return record

    maps = {}
    for bundle in data["bundles"]:
        result = execute(bundle["id"], "map", request(bundle, None, model, "map"))
        maps[bundle["id"]] = result["parsed"]
    bundles = {b["id"]: b for b in data["bundles"]}
    for case in data["cases"]:
        bundle = bundles[case["bundle"]]
        previous = execute(case["id"], "before", request(bundle, case, model, "before"))
        old = decision(previous["parsed"], bundle["before"])
        execute(case["id"], "after", request(bundle, case, model, "after"))
        for stage in ["direct_gate", "map_gate"]:
            execute(
                case["id"],
                stage,
                request(bundle, case, model, stage, old, maps[bundle["id"]]),
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--base", default="http://127.0.0.1:11434")
    a = p.parse_args()
    run(a.inputs, a.output, a.model, a.base)
