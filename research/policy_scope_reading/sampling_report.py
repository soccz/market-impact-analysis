"""Re-score every preselected follow-up claim under the shared sampled decoder."""

import argparse
import json
from pathlib import Path
import statistics

from evaluate import parsed
from infer import ROOT, digest
from report import enriched, export_official


def run(cache, output=None):
    cache = Path(cache)
    out = Path(output or ROOT / "results/sampling")
    out.mkdir(parents=True, exist_ok=True)
    rows = json.loads((cache / "followup_inputs.json").read_text())
    raw = {}
    predictions = {}
    runtime = {}
    for name in ["qwen-sampled-control", "qwen-sampled-thinking"]:
        folder = cache / "results/sampling/followup" / name
        raw[name] = {
            r["id"]: json.loads((folder / (r["id"] + ".json")).read_text())
            for r in rows
        }
        predictions[name] = {r["id"]: parsed(raw[name][r["id"]], r) for r in rows}
        records = list(raw[name].values())
        runtime[name] = dict(
            n=len(records),
            sum_reported_eval_count=sum(
                r["response"].get("eval_count", 0) for r in records
            ),
            median_reported_eval_count=statistics.median(
                r["response"].get("eval_count", 0) for r in records
            ),
            median_elapsed_seconds=statistics.median(
                r["elapsed_seconds"] for r in records
            ),
            truncated=sum(
                r["response"].get("done_reason") == "length" for r in records
            ),
            with_thinking_field=sum(
                bool(r["response"].get("message", {}).get("thinking")) for r in records
            ),
        )
    export_official(raw, out / "followup")
    for name, records in raw.items():
        for cid, r in records.items():
            path = out / "followup" / name / (cid + ".json")
            pub = json.loads(path.read_text())
            t = r["response"].get("message", {}).get("thinking", "")
            pub["thinking_trace_sha256"] = digest(t.encode())
            pub["thinking_trace_present"] = bool(t)
            pub["thinking_characters"] = len(t)
            path.write_text(json.dumps(pub, ensure_ascii=False, indent=2) + "\n")
    groups = {"followup": enriched(rows, predictions)}
    summary = {k: {m: s["metrics"] for m, s in v.items()} for k, v in groups.items()}
    a = {r["id"]: r for r in groups["followup"]["qwen-sampled-control"]["cases"]}
    b = {r["id"]: r for r in groups["followup"]["qwen-sampled-thinking"]["cases"]}
    pairs = dict(
        n=len(a),
        gained=sum(not a[k]["correct"] and b[k]["correct"] for k in a),
        lost=sum(a[k]["correct"] and not b[k]["correct"] for k in a),
        both_correct=sum(a[k]["correct"] and b[k]["correct"] for k in a),
    )
    for name, data in [
        ("summary.json", summary),
        ("case_scores.json", groups),
        ("runtime.json", runtime),
        ("paired_changes.json", pairs),
    ]:
        (out / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
    print(
        json.dumps(dict(summary=summary, runtime=runtime), ensure_ascii=False, indent=2)
    )
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--output")
    a = p.parse_args()
    run(a.cache, a.output)
