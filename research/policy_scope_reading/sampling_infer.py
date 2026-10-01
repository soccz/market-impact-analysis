"""Follow-up sampling comparison after early greedy-thinking truncations."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import requests
from extension_infer import make_request
from infer import digest


def request_for(case, thinking):
    request = deepcopy(make_request(case, "qwen", "baseline"))
    request["think"] = thinking
    request["options"].update(
        num_predict=4096,
        temperature=1.0,
        top_k=20,
        top_p=0.95,
        min_p=0.0,
        presence_penalty=1.5,
    )
    return request


def run(inputs, output, thinking=True, base="http://127.0.0.1:11434"):
    rows = json.loads(Path(inputs).read_text())
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    tag = next(
        t
        for t in requests.get(base + "/api/tags", timeout=20).json()["models"]
        if t["name"] == "qwen3.5:9b"
    )
    version = requests.get(base + "/api/version", timeout=20).json()
    for i, case in enumerate(rows, 1):
        request = request_for(case, thinking)
        rh = digest(json.dumps(request, ensure_ascii=False, sort_keys=True).encode())
        path = out / (case["id"] + ".json")
        if path.exists():
            prior = json.loads(path.read_text())
            assert (
                prior["request_sha256"] == rh and prior["model_digest"] == tag["digest"]
            )
            continue
        start = time.time()
        r = requests.post(base + "/api/chat", json=request, timeout=300)
        r.raise_for_status()
        response = r.json()
        try:
            parsed = json.loads(response.get("message", {}).get("content", ""))
        except json.JSONDecodeError:
            parsed = None
        record = dict(
            id=case["id"],
            method="sampled-thinking" if thinking else "sampled-control",
            model="qwen",
            model_name=tag["name"],
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
            response=response,
            parsed=parsed,
            elapsed_seconds=round(time.time() - start, 3),
        )
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        print(
            json.dumps(
                dict(
                    completed=i,
                    total=len(rows),
                    id=case["id"],
                    method=record["method"],
                    tokens=response.get("eval_count"),
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--no-thinking", action="store_true")
    a = p.parse_args()
    run(a.inputs, a.output, not a.no_thinking)
