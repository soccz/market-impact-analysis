"""Export structured predictions and recompute metrics without inference."""

import argparse
import itertools
import json
import math
from pathlib import Path

from experiment import MODELS, sha
from method import complete, decide, literal_rules, retrieve, validate

ROOT = Path(__file__).parent
METHODS = ["literal_rules", "guarded_rules", "direct", "full", "retrieved"]


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def semantic_audit(prediction, reference, fields, evidence):
    """All cells induced by both graphs' unary numeric/boolean predicates.

    This checks graph equivalence in the declared nonnegative field domain,
    not correctness of the provisional natural-language reference.
    """
    try:
        p = validate(prediction, fields, evidence)
        r = validate(reference, fields, evidence)
    except (ValueError, TypeError, KeyError, RecursionError) as exc:
        return dict(status="invalid", reason=str(exc), checked=0)
    atoms = [n for n in list(p.values()) + list(r.values()) if n["op"] == "atom"]
    domains = {}
    for field in sorted({a["field"] for a in atoms}):
        if fields[field]["type"] == "boolean":
            domains[field] = [False, True]
        else:
            bounds = sorted(
                {0.0} | {float(a["value"]) for a in atoms if a["field"] == field}
            )
            values = bounds + [(x + y) / 2 for x, y in zip(bounds, bounds[1:])]
            values += [bounds[-1] + 1]
            domains[field] = sorted({x for x in values if x >= 0})
    total = math.prod(map(len, domains.values()))
    if total > 100000:
        return dict(status="budget_exceeded", cells=total, checked=0)
    mismatch = 0
    witness = None
    for row in itertools.product(*domains.values()):
        facts = dict(zip(domains, row))
        a, b = complete(prediction, facts, p), complete(reference, facts, r)
        if a != b:
            mismatch += 1
            if witness is None:
                witness = dict(facts=facts, predicted=a, reference=b)
    return dict(
        status="equivalent" if mismatch == 0 else "different",
        checked=total,
        mismatch_cells=mismatch,
        witness=witness,
    )


def score(data, predictions, rules, retrieval):
    docs = {d["id"]: d for d in data["documents"]}
    pred = {(r["model"], r["stage"], r["id"]): r for r in predictions}
    rows = []
    for model in MODELS:
        for case in data["cases"]:
            doc = docs[case["document"]]
            rule = rules[doc["id"]]
            row = dict(case, model=model, group=doc["group"])
            value = decide(
                rule["program"], case["facts"], doc["fields"], doc["evidence"]
            )
            row["literal_rules"] = value["decision"]
            row["guarded_rules"] = "abstain" if rule["guarded"] else value["decision"]
            direct = pred[model, "direct", case["id"]]
            parsed = direct["parsed"] or {}
            row["direct"] = (
                parsed.get("decision", "abstain") if direct["usable"] else "abstain"
            )
            row["reasons"] = {}
            for method in ["full", "retrieved"]:
                record = pred[model, method, doc["id"]]
                evidence = doc["evidence"]
                if method == "retrieved":
                    evidence = {
                        k: evidence[k] for k in retrieval[doc["id"]]["expanded"]
                    }
                result = decide(
                    record["parsed"] if record["usable"] else None,
                    case["facts"],
                    doc["fields"],
                    evidence,
                )
                row[method] = result["decision"]
                row["reasons"][method] = result["reason"]
            rows.append(row)
    summary = {}
    for group in ["authored", "official", "all"]:
        summary[group] = {}
        for model in MODELS:
            rr = [
                r
                for r in rows
                if r["model"] == model and (group == "all" or r["group"] == group)
            ]
            summary[group][model] = {}
            for method in METHODS:
                by_doc = {}
                for key in sorted({r["document"] for r in rr}):
                    dd = [r for r in rr if r["document"] == key]
                    by_doc[key] = dict(
                        correct=sum(r[method] == r["label"] for r in dd), n=len(dd)
                    )
                summary[group][model][method] = dict(
                    n=len(rr),
                    correct=sum(r[method] == r["label"] for r in rr),
                    abstain=sum(r[method] == "abstain" for r in rr),
                    undetermined=sum(r[method] == "undetermined" for r in rr),
                    fixes_vs_direct=sum(
                        r[method] == r["label"] and r["direct"] != r["label"]
                        for r in rr
                    ),
                    harms_vs_direct=sum(
                        r[method] != r["label"] and r["direct"] == r["label"]
                        for r in rr
                    ),
                    documents=by_doc,
                    document_macro_accuracy=sum(
                        v["correct"] / v["n"] for v in by_doc.values()
                    )
                    / len(by_doc),
                    complete_documents=sum(
                        v["correct"] == v["n"] for v in by_doc.values()
                    ),
                )
    pairs = []
    for group in ["exception", "subject"]:
        dd = [d for d in docs.values() if d.get("pair") == group]
        for model in MODELS:
            for method in METHODS:
                both, changed_both, changed_n = 0, 0, 0
                for i in range(1, 9):
                    a, b = [
                        next(
                            r
                            for r in rows
                            if r["model"] == model and r["id"] == d["id"] + f"-{i}"
                        )
                        for d in dd
                    ]
                    success = a[method] == a["label"] and b[method] == b["label"]
                    both += success
                    changed = a["label"] != b["label"]
                    changed_n += changed
                    changed_both += success and changed
                pairs.append(
                    dict(
                        pair=group,
                        model=model,
                        method=method,
                        n=8,
                        both_correct=both,
                        changed_n=changed_n,
                        changed_both_correct=changed_both,
                    )
                )
    return rows, summary, pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--cache", required=True)
    args = parser.parse_args()
    data = load(args.inputs)
    docs = {d["id"]: d for d in data["documents"]}
    predictions = []
    for path in sorted(Path(args.cache).glob("*/*/*.json")):
        record = load(path)
        raw = record.pop("response")
        record["response_sha256"] = sha(raw)
        record["done_reason"] = raw.get("done_reason")
        record["prompt_tokens"] = raw.get("prompt_eval_count", 0)
        record["output_tokens"] = raw.get("eval_count", 0)
        record["usable"] = (
            raw.get("done") is True
            and raw.get("done_reason") == "stop"
            and record["prompt_tokens"] + record["output_tokens"] < 16384
        )
        predictions.append(record)
    assert len(predictions) == 120, len(predictions)
    rules, retrieval, programs = {}, {}, []
    for doc in docs.values():
        rule, guarded = literal_rules(doc)
        rules[doc["id"]] = dict(program=rule, guarded=guarded)
        r = retrieve(doc)
        required = set(doc["required_evidence"])
        r.update(
            required=sorted(required),
            total_paragraphs=len(doc["evidence"]),
            ranked_recall=len(required & set(r["ranked"])) / len(required),
            expanded_recall=len(required & set(r["expanded"])) / len(required),
        )
        retrieval[doc["id"]] = r
        for model in MODELS:
            for stage in ["full", "retrieved"]:
                record = next(
                    p
                    for p in predictions
                    if (p["model"], p["stage"], p["id"]) == (model, stage, doc["id"])
                )
                evidence = (
                    doc["evidence"]
                    if stage == "full"
                    else {k: doc["evidence"][k] for k in r["expanded"]}
                )
                audit = semantic_audit(
                    record["parsed"] if record["usable"] else None,
                    doc["reference_program"],
                    doc["fields"],
                    evidence,
                )
                programs.append(
                    dict(document=doc["id"], model=model, stage=stage, **audit)
                )
    rows, summary, pairs = score(data, predictions, rules, retrieval)
    missing = missing_summary(rows)
    costs = {}
    for model in MODELS:
        costs[model] = {}
        for stage in ["direct", "full", "retrieved"]:
            pp = [p for p in predictions if p["model"] == model and p["stage"] == stage]
            costs[model][stage] = dict(
                requests=len(pp),
                prompt_tokens=sum(p["prompt_tokens"] for p in pp),
                output_tokens=sum(p["output_tokens"] for p in pp),
                elapsed_seconds=round(sum(p["elapsed_seconds"] for p in pp), 3),
                unusable=sum(not p["usable"] for p in pp),
            )
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    for name, value in dict(
        predictions=predictions,
        rules=rules,
        retrieval=retrieval,
        programs=programs,
        rows=rows,
        summary=summary,
        pairs=pairs,
        costs=costs,
        missing=missing,
    ).items():
        save(out / (name + ".json"), value)
    print(json.dumps(dict(results=summary, costs=costs), ensure_ascii=False))


def missing_summary(rows):
    result = []
    for model in MODELS:
        for group in ["all_known", "some_missing", "gold_undetermined"]:
            selected = [
                r
                for r in rows
                if r["model"] == model
                and (
                    all(v is not None for v in r["facts"].values())
                    if group == "all_known"
                    else (
                        any(v is None for v in r["facts"].values())
                        if group == "some_missing"
                        else r["label"] == "undetermined"
                    )
                )
            ]
            for method in METHODS:
                result.append(
                    dict(
                        model=model,
                        group=group,
                        method=method,
                        n=len(selected),
                        correct=sum(r[method] == r["label"] for r in selected),
                        abstain=sum(r[method] == "abstain" for r in selected),
                    )
                )
    return result


if __name__ == "__main__":
    main()
