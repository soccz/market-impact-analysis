"""Development error repair: contrastive AND/OR and exclusion-to-pass examples."""

import argparse, json
from pathlib import Path
import infer
from infer_explicit import request as EXPLICIT_REQUEST

ROOT = Path(__file__).parent


def request(bundle, case, model, stage):
    req = EXPLICIT_REQUEST(bundle, case, model, stage)
    if stage == "policy":
        req["messages"][0]["content"] += (
            "\n"
            + json.loads((ROOT / "logic_examples_prompt.json").read_text())["policy"]
        )
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
