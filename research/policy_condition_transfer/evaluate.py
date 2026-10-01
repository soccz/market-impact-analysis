"""Score saved predictions; no source extraction, model tuning or label changes."""

import argparse
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def covers(spans, groups):
    """Every required group needs one complete alternative span, without tolerance."""
    return bool(groups) and all(
        any(
            p["page"] == r["page"] and p["start"] <= r["start"] and p["end"] >= r["end"]
            for r in group
            for p in spans
        )
        for group in groups
    )


def score(predictions, refs, methods):
    rows = []
    for method in methods:
        for ref in refs:
            p = predictions[ref["document"]][method][ref["field"]]
            answerable = ref["value"] is not None
            exact = answerable and p["value"] == ref["value"]
            grounded = covers(p["evidence"], ref["support_groups"])
            if not answerable:
                outcome = (
                    "absence_abstain" if p["value"] is None else "unsupported_answer"
                )
            elif p["value"] is None:
                outcome = "abstain"
            elif not exact:
                outcome = "wrong_value"
            else:
                outcome = "correct_joint" if grounded else "value_only"
            rows.append(
                dict(
                    document=ref["document"],
                    field=ref["field"],
                    method=method,
                    reference=ref["value"],
                    prediction=p["value"],
                    answerable=answerable,
                    value_exact=exact,
                    evidence_complete=grounded,
                    outcome=outcome,
                    scope_branch_required=bool(ref.get("scope_requirements")),
                    reason=p["reason"],
                )
            )

    def counts(subset):
        c = Counter(
            answerable=0,
            not_established=0,
            correct_joint=0,
            value_only=0,
            wrong_value=0,
            abstain=0,
            absence_abstain=0,
            unsupported_answer=0,
        )
        for row in subset:
            c["answerable" if row["answerable"] else "not_established"] += 1
            c[row["outcome"]] += 1
        return dict(c)

    return (
        rows,
        {m: counts([r for r in rows if r["method"] == m]) for m in methods},
        {
            doc: {
                m: counts(
                    [r for r in rows if r["method"] == m and r["document"] == doc]
                )
                for m in methods
            }
            for doc in predictions
        },
    )


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    methods = json.loads((ROOT / "protocol.json").read_text())["methods"]
    pred = json.loads((ROOT / "results/predictions.json").read_text())
    refs = json.loads((ROOT / "reference.json").read_text())
    rows, metrics, by_doc = score(pred, refs, methods)
    summary = dict(
        documents=4,
        fields=16,
        answerable=13,
        not_established=3,
        independent_human_reviews=0,
        new_fits=0,
        methods=metrics,
        by_document=by_doc,
        interpretation="Unchanged-reader first application on four selected notices; AI provisional labels. Full support includes explicit branch evidence; not whole eligibility accuracy.",
    )
    for name, value in [("field_scores.json", rows), ("summary.json", summary)]:
        (output / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=ROOT / "results")
    args = parser.parse_args()
    print(json.dumps(run(args.output)["methods"], ensure_ascii=False, indent=2))
