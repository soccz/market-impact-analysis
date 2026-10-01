"""Development-informed task clarification; immutable baseline remains available."""

import argparse
from pathlib import Path
import json
import infer

BASE_REQUEST = infer.request
ROOT = Path(__file__).parent


def request(bundle, case, model, stage):
    req = BASE_REQUEST(bundle, case, model, stage)
    if stage in ["policy", "profile"]:
        extra = json.loads((ROOT / "explicit_prompts.json").read_text())[stage]
        req["messages"][0]["content"] += "\n" + extra
    return req


def run(inputs, out, model):
    saved = infer.request
    try:
        infer.request = request
        infer.run(inputs, out, model)
    finally:
        infer.request = saved


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", choices=infer.MODELS, required=True)
    a = p.parse_args()
    run(a.inputs, a.out, a.model)
