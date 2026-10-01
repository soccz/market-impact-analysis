"""Separate semantic unknown, model errors, citations and selective abstention."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from infer import ROOT

LABELS = {"supported", "contradicted", "not_established"}


def parsed(record, case):
    out = record.get("parsed")
    if (
        not isinstance(out, dict)
        or out.get("decision") not in LABELS
        or not isinstance(out.get("evidence"), list)
    ):
        return dict(decision="abstain", evidence=[], reason="invalid_schema")
    ids = out["evidence"]
    if not ids or any(
        not isinstance(i, str) or i not in case["sentences"] for i in ids
    ):
        return dict(decision="abstain", evidence=[], reason="invalid_citation")
    return dict(
        decision=out["decision"], evidence=sorted(set(ids)), reason="model_output"
    )


def consensus(a, b):
    if a["decision"] != b["decision"] or a["decision"] == "abstain":
        return dict(
            decision="abstain", evidence=[], reason="model_disagreement_or_invalid"
        )
    return dict(
        decision=a["decision"],
        evidence=sorted(set(a["evidence"]) | set(b["evidence"])),
        reason="two_model_agreement",
    )


def score(rows, predictions, supports=None):
    outputs = []
    counts = Counter()
    families = defaultdict(Counter)
    for row in rows:
        pred = predictions[row["id"]]
        ref = row["reference"]
        sets = (supports or {}).get(
            row["id"], ref.get("support_sets", [ref["evidence"]])
        )
        valid = set(pred["evidence"]) <= set(row["sentences"])
        supported = valid and any(set(s) <= set(pred["evidence"]) for s in sets)
        extra = min(len(set(pred["evidence"]) - set(s)) for s in sets)
        correct = pred["decision"] == ref["decision"]
        answered = pred["decision"] != "abstain"
        definitive = pred["decision"] in {"supported", "contradicted"}
        record = dict(
            id=row["id"],
            family=row["family"],
            reference=ref["decision"],
            prediction=pred,
            correct=correct,
            reference_evidence_covered=supported,
            joint=correct and supported,
            extra_citations_over_minimum=extra,
            unsupported_definitive=ref["decision"] == "not_established" and definitive,
        )
        outputs.append(record)
        metrics = dict(
            total=1,
            correct=correct,
            joint=record["joint"],
            answered=answered,
            wrong_answer=answered and not correct,
            abstain=not answered,
            reference_unknown=ref["decision"] == "not_established",
            semantic_unknown=pred["decision"] == "not_established",
            unsupported_definitive=record["unsupported_definitive"],
            extra_citations=extra,
            citations=len(pred["evidence"]),
        )
        for k, v in metrics.items():
            counts[k] += v
            families[row["family"]][k] += v
    return dict(
        metrics=dict(counts),
        families={k: dict(v) for k, v in families.items()},
        cases=outputs,
    )


def evaluate(inputs, results, output, development=False):
    rows = json.loads(Path(inputs).read_text())
    support = (
        json.loads((ROOT / "development_support.json").read_text())["support_sets"]
        if development
        else None
    )
    predictions = {}
    record_counts = {}
    for folder in sorted(Path(results).iterdir()):
        if not folder.is_dir():
            continue
        records = {}
        for row in rows:
            path = folder / (row["id"] + ".json")
            if not path.exists():
                break
            records[row["id"]] = parsed(json.loads(path.read_text()), row)
        if len(records) == len(rows):
            predictions[folder.name] = records
            record_counts[folder.name] = len(records)
    for method in ["baseline", "scoped"]:
        names = [f"{m}-{method}" for m in ["kanana", "qwen"]]
        if all(n in predictions for n in names):
            predictions["consensus-" + method] = {
                r["id"]: consensus(*(predictions[n][r["id"]] for n in names))
                for r in rows
            }
    summary = {k: score(rows, v, support) for k, v in predictions.items()}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {k: v["metrics"] for k, v in summary.items()}, ensure_ascii=False, indent=2
        )
    )
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--development", action="store_true")
    a = p.parse_args()
    evaluate(a.inputs, a.results, a.output, a.development)
