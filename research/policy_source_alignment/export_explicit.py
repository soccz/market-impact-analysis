"""Export explicit operator responses without rewriting saved model output."""

import argparse
import copy
import json
from pathlib import Path
from bridge import MODELS, sha
from explicit_logic import request, as_selection
from evaluate_v2 import development_data, previous_records, previous_rows, evaluate


def export(data, cache, dest):
    records = []
    for model in MODELS:
        for bundle in data["bundles"]:
            raw = json.loads(
                (Path(cache) / model / "policy" / (bundle["id"] + ".json")).read_text()
            )
            req = request(bundle, None, model, "policy")
            assert raw["request_sha256"] == sha(req)
            assert raw["input_sha256"] == sha(req["messages"])
            record = {
                k: raw[k]
                for k in [
                    "id",
                    "stage",
                    "model",
                    "model_name",
                    "model_digest",
                    "server",
                    "created_utc",
                    "request_sha256",
                    "input_sha256",
                    "elapsed_seconds",
                    "parsed",
                ]
            }
            record["response_sha256"] = sha(raw["response"])
            records.append(record)
    normalized = copy.deepcopy(records)
    for record in normalized:
        record["parsed"] = as_selection(record["parsed"])
    result = evaluate(data, normalized, previous_records(), previous_rows())
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name, value in dict(predictions=records, **result).items():
        (dest / (name + ".json")).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--previous-cache", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--dest", required=True)
    a = p.parse_args()
    print(
        json.dumps(
            export(development_data(a.previous_cache), a.cache, a.dest)["summary"],
            ensure_ascii=False,
        )
    )
