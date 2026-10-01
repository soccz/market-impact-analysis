"""Post-hoc structural intervention, kept separate from the frozen first run."""

import argparse
import json
import re
from pathlib import Path

from evaluate import covers
from prepare import load
from reader import FIELDS, predict

ROOT = Path(__file__).resolve().parent


def restrict(pages):
    text = "".join(pages)
    start = re.search(r"2\.?(?:신청자격(?:[·․]?요건)?|지원대상)", text)
    if start is None:
        return [" " * len(p) for p in pages], None
    end = re.search(
        r"3\.?(?:지원내용|신청방법|신청및접수|지원대상선정기준)", text[start.end() :]
    )
    if end is None:
        return [" " * len(p) for p in pages], None
    lo, hi = start.start(), start.end() + end.start()
    offset, result = 0, []
    for page in pages:
        result.append(
            "".join(c if lo <= offset + i < hi else " " for i, c in enumerate(page))
        )
        offset += len(page)
    return result, [lo, hi]


def read(pages):
    masked, region = restrict(pages)
    return predict(masked, "document_first"), region


def run(cache, output):
    docs = load(cache)
    predictions = {}
    for doc, pages in docs.items():
        parsed, region = read(pages)
        predictions[doc] = dict(section_range=region, fields=parsed)
    refs = json.loads((ROOT / "reference.json").read_text())
    counts = dict(
        answerable=0,
        correct_value=0,
        correct_joint=0,
        wrong_value=0,
        abstain=0,
        not_established=0,
        absence_abstain=0,
    )
    for r in refs:
        p = predictions[r["document"]]["fields"][r["field"]]
        if r["value"] is None:
            counts["not_established"] += 1
            counts["absence_abstain"] += p["value"] is None
        else:
            counts["answerable"] += 1
            counts["correct_value"] += p["value"] == r["value"]
            counts["correct_joint"] += p["value"] == r["value"] and covers(
                p["evidence"], r["support_groups"]
            )
            counts["wrong_value"] += p["value"] is not None and p["value"] != r["value"]
            counts["abstain"] += p["value"] is None
    stress = []
    for case in json.loads((ROOT / "stress_cases.json").read_text()):
        parsed, _ = read([case["text"]])
        value = parsed[case["field"]]["value"]
        stress.append(
            {**case, "prediction": value, "exact": value == case["reference"]}
        )
    result = dict(
        status="post-hoc, same-source adaptation; NOT held-out accuracy",
        predictions=predictions,
        metrics=counts,
        stress=stress,
    )
    Path(output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            dict(
                metrics=counts,
                stress_correct=sum(r["exact"] for r in stress),
                stress_total=len(stress),
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", default=ROOT / "results/posthoc.json")
    args = parser.parse_args()
    run(args.cache, args.output)
