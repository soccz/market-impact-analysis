"""Aggregate frozen outputs, selective checks and paired evidence diagnostics."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from evaluate import consensus, parsed, score
from guard import apply
from infer import ROOT, digest


def read_predictions(rows, folder):
    predictions = {}
    records = {}
    for model in ["kanana", "qwen"]:
        for method in ["baseline", "scoped"]:
            name = model + "-" + method
            records[name] = {}
            predictions[name] = {}
            for case in rows:
                record = json.loads(
                    (Path(folder) / name / (case["id"] + ".json")).read_text()
                )
                records[name][case["id"]] = record
                predictions[name][case["id"]] = parsed(record, case)
            if method == "scoped":
                predictions[name + "-guard"] = {
                    c["id"]: apply(
                        c, records[name][c["id"]], predictions[name][c["id"]]
                    )
                    for c in rows
                }
    for method in ["baseline", "scoped", "scoped-guard"]:
        predictions["consensus-" + method] = {
            c["id"]: consensus(
                predictions["kanana-" + method][c["id"]],
                predictions["qwen-" + method][c["id"]],
            )
            for c in rows
        }
    return predictions, records


def enriched(rows, predictions, support=None):
    out = {}
    for name, prediction in predictions.items():
        s = score(rows, prediction, support)
        s["metrics"]["wrong_definitive"] = sum(
            c["prediction"]["decision"] in {"supported", "contradicted"}
            and not c["correct"]
            for c in s["cases"]
        )
        out[name] = s
    return out


def export_official(records, output):
    out = Path(output)
    for name, rows in records.items():
        (out / name).mkdir(parents=True, exist_ok=True)
        for cid, record in rows.items():
            source = record["response"]
            raw = source.get("message", {}).get("content", "")
            result = record.get("parsed")
            public = {
                k: v for k, v in record.items() if k not in {"response", "parsed"}
            }
            public["private_response_sha256"] = digest(raw.encode())
            public["response_redacted"] = True
            public["runtime"] = {
                k: source.get(k)
                for k in [
                    "done",
                    "done_reason",
                    "prompt_eval_count",
                    "prompt_eval_duration",
                    "eval_count",
                    "eval_duration",
                    "load_duration",
                    "total_duration",
                ]
            }
            if isinstance(result, dict):
                public["parsed"] = {
                    k: result[k]
                    for k in ["decision", "evidence", "certainty"]
                    if k in result
                }
                if "rule" in result:
                    trace = result["rule"]
                    witnesses = [
                        m.group()
                        for m in re.finditer(
                            r"(\d+(?:\.\d+)?)\s*(<=|>=|≤|≥|<|>|=)\s*(\d+(?:\.\d+)?)",
                            trace,
                        )
                    ]
                    witnesses += [w for w in ["불충족", "미충족"] if w in trace]
                    public["parsed"]["rule"] = "; ".join(witnesses)
                    public["trace_notice"] = (
                        "Numeric/failed-condition witness only; this is not the full generated trace."
                    )
                    public["full_trace_sha256"] = digest(trace.encode())
            else:
                public["parsed"] = None
            (out / name / (cid + ".json")).write_text(
                json.dumps(public, ensure_ascii=False, indent=2) + "\n"
            )


def run(cache, output=None):
    cache = Path(cache)
    out = Path(output or ROOT / "results")
    out.mkdir(parents=True, exist_ok=True)
    dev = json.loads((ROOT / "development.json").read_text())
    composition = json.loads((ROOT / "composition.json").read_text())
    official = json.loads((cache / "official_inputs.json").read_text())
    devpred, _ = read_predictions(dev, ROOT / "results/development")
    cpred, _ = read_predictions(composition, ROOT / "results/composition")
    opred, raw = read_predictions(official, cache / "results/official")
    export_official(raw, out / "official")
    publicpred, _ = read_predictions(official, out / "official")
    assert (
        publicpred == opred
    ), "Redacted witnesses must reproduce all guarded decisions exactly."
    groups = {
        "development": enriched(
            dev,
            devpred,
            json.loads((ROOT / "development_support.json").read_text())["support_sets"],
        ),
        "composition": enriched(composition, cpred),
    }
    for variant in ["base", "distractor", "removed"]:
        selected = [c for c in official if c.get("variant", "base") == variant]
        groups["official_" + variant] = enriched(selected, opred)
    pairs = {}
    base = [c for c in official if "variant" not in c]
    for method, preds in opred.items():
        pairs[method] = dict(
            pairs=len(base),
            distractor_decision_stable=sum(
                preds[c["id"]]["decision"] == preds[c["id"] + "-distractor"]["decision"]
                for c in base
            ),
            base_and_distractor_correct=sum(
                preds[c["id"]]["decision"]
                == c["reference"]["decision"]
                == preds[c["id"] + "-distractor"]["decision"]
                for c in base
            ),
            removed_returns_semantic_unknown=sum(
                preds[c["id"] + "-removed"]["decision"] == "not_established"
                for c in base
            ),
            removed_abstains=sum(
                preds[c["id"] + "-removed"]["decision"] == "abstain" for c in base
            ),
        )
    summary = {
        k: {method: value["metrics"] for method, value in methods.items()}
        for k, methods in groups.items()
    }
    for name, value in [
        ("summary.json", summary),
        ("case_scores.json", groups),
        ("paired_diagnostics.json", pairs),
    ]:
        (out / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--output")
    a = p.parse_args()
    run(a.cache, a.output)
