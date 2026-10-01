"""Export source-safe first responses and reproducible split-wise results."""

import argparse
import json
from pathlib import Path

from core import sha
from evaluate import score
from infer import ROOT, MODELS


def export(record):
    public = {k: v for k, v in record.items() if k not in {"response", "parsed"}}
    public["private_response_sha256"] = sha(record["response"])
    public["full_parsed_sha256"] = sha(record["parsed"])
    raw = record["response"]
    public["runtime"] = {
        k: raw[k]
        for k in [
            "done",
            "done_reason",
            "prompt_eval_count",
            "eval_count",
            "total_duration",
            "load_duration",
            "prompt_eval_duration",
            "eval_duration",
        ]
        if k in raw
    }
    public["response_redacted"] = True
    if record["stage"] == "map":
        parsed = record["parsed"]
        public["parsed"] = (
            {
                "changes": [
                    {k: x.get(k) for k in ["before_ids", "after_ids", "kind"]}
                    for x in parsed.get("changes", [])
                    if isinstance(x, dict)
                ]
            }
            if isinstance(parsed, dict)
            else None
        )
        public["map_text_omitted"] = True
    else:
        public["parsed"] = record["parsed"]
    return public


def load_records(folder):
    return {
        stage: {
            p.stem: json.loads(p.read_text()) for p in (folder / stage).glob("*.json")
        }
        for stage in ["map", "before", "after", "direct_gate", "map_gate"]
    }


def run(inputs, cache, split, output=None):
    data = json.loads(Path(inputs).read_text())
    cache = Path(cache)
    out = Path(output or ROOT / "results" / split)
    out.mkdir(parents=True, exist_ok=True)
    scores = {}
    runtime = {}
    for model in MODELS:
        records = load_records(cache / model)
        assert len(records["map"]) == len(data["bundles"])
        assert all(
            len(records[s]) == len(data["cases"])
            for s in ["before", "after", "direct_gate", "map_gate"]
        )
        scores[model] = score(data, records)
        public = {
            s: {k: export(v) for k, v in rows.items()} for s, rows in records.items()
        }
        (out / (model + ".json")).write_text(
            json.dumps(public, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        runtime[model] = {
            s: {
                "calls": len(rows),
                "sum_elapsed_seconds": round(
                    sum(r["elapsed_seconds"] for r in rows.values()), 3
                ),
                "truncated": sum(
                    r["response"].get("done_reason") == "length" for r in rows.values()
                ),
            }
            for s, rows in records.items()
        }
    summary = {
        m: {method: v["metrics"] for method, v in methods.items()}
        for m, methods in scores.items()
    }
    for name, value in [
        ("case_scores.json", scores),
        ("summary.json", summary),
        ("runtime.json", runtime),
    ]:
        (out / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
    print(
        json.dumps(
            {
                m: {
                    k: {
                        f: v[f]
                        for f in [
                            "n",
                            "correct",
                            "pair_correct",
                            "refreshed",
                            "false_flip",
                            "missed_flip",
                            "missed_recheck",
                            "retained_old_error",
                        ]
                    }
                    for k, v in methods.items()
                }
                for m, methods in summary.items()
            },
            ensure_ascii=False,
        )
    )
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--output")
    a = p.parse_args()
    run(a.inputs, a.cache, a.split, a.output)
