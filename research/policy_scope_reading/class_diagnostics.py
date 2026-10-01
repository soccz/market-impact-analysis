"""Expose class collapse and selective coverage rather than only average accuracy."""

import argparse
import json
from pathlib import Path

from infer import ROOT

LABELS = ["supported", "contradicted", "not_established"]


def diagnose(cases):
    matrix = {label: {p: 0 for p in LABELS + ["abstain"]} for label in LABELS}
    for row in cases:
        matrix[row["reference"]][row["prediction"]["decision"]] += 1
    per_class = {}
    for label in LABELS:
        actual = sum(matrix[label].values())
        predicted = sum(matrix[a][label] for a in LABELS)
        tp = matrix[label][label]
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else None
        f1 = 2 * tp / (actual + predicted) if actual + predicted else 0.0
        per_class[label] = dict(
            reference_count=actual,
            predicted_count=predicted,
            precision=precision,
            recall=recall,
            f1=f1,
        )
    answered = sum(row["prediction"]["decision"] != "abstain" for row in cases)
    return dict(
        confusion=matrix,
        per_class=per_class,
        macro_f1=(
            sum(per_class[x]["f1"] for x in LABELS) / 3
            if all(per_class[x]["reference_count"] for x in LABELS)
            else None
        ),
        reference_known=sum(r["reference"] != "not_established" for r in cases),
        false_semantic_unknown=sum(
            r["reference"] != "not_established"
            and r["prediction"]["decision"] == "not_established"
            for r in cases
        ),
        output_coverage=answered / len(cases),
        selective_accuracy=(
            sum(r["correct"] for r in cases) / answered if answered else None
        ),
    )


def run(output=None):
    result = {}
    for stage in ["", "extension", "thinking", "sampling", "citation"]:
        groups = json.loads((ROOT / "results" / stage / "case_scores.json").read_text())
        result[stage or "first"] = {
            group: {
                method: diagnose(score["cases"]) for method, score in methods.items()
            }
            for group, methods in groups.items()
        }
    target = Path(output or ROOT / "results/class_diagnostics.json")
    target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output")
    a = p.parse_args()
    run(a.output)
