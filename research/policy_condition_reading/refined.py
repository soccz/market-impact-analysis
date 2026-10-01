"""Second post-hoc iteration: heading aliases, nearest scope, allowed polarity.

Designed after seeing first-run and stress errors. Same-corpus adaptation only.
The frozen first-run parser and predictions remain untouched.
"""

import argparse
import json
import re
from pathlib import Path

from evaluate import covers
from prepare import load
from reader import FIELDS, evidence, parse

ROOT = Path(__file__).resolve().parent


def read(pages):
    joined = "".join(pages)
    start = re.search(r"2\.?(?:신청자격|지원대상|지원요건|참여요건)", joined)
    end = (
        re.search(
            r"3\.?(?:지원내용|신청방법|신청및접수|지원대상선정기준)",
            joined[start.end() :],
        )
        if start
        else None
    )
    region = [start.start(), start.end() + end.start()] if start and end else None
    output = {}
    for field in FIELDS:
        candidates = []
        offset = 0
        for index, text in enumerate(pages):
            lo = max(0, region[0] - offset) if region else len(text)
            hi = min(len(text), region[1] - offset) if region else 0
            offset += len(text)
            if hi <= lo:
                continue
            chunk = dict(page=index + 1, start=lo, end=hi, text=text[lo:hi])
            if field == "income":
                scoped, spans = {}, []
                generic = []
                for m in re.finditer(r"중위소득(\d+(?:\.\d+)?)%이하", chunk["text"]):
                    prefix = chunk["text"][max(0, m.start() - 45) : m.start()]
                    markers = list(
                        re.finditer(r"(?:생애)?최초수혜자|\d{4}년수혜자", prefix)
                    )
                    value = float(m.group(1))
                    if markers:
                        nearest = markers[-1]
                        scope = "first" if "최초" in nearest.group() else "returning"
                        scoped[scope] = value
                        spans.append(
                            evidence(
                                chunk, max(0, m.start() - 45) + nearest.start(), m.end()
                            )
                        )
                    else:
                        generic.append(
                            dict(
                                value={"all": value},
                                evidence=[evidence(chunk, m.start(), m.end())],
                            )
                        )
                if scoped:
                    candidates.append(dict(value=scoped, evidence=spans))
                elif generic:
                    candidates.append(generic[0])
            else:
                p = parse(chunk, field)
                if p and field == "welfare":
                    # If this local clause explicitly permits application, it
                    # cannot by itself establish an exclusion list.
                    e = p["evidence"][0]
                    clause = text[e["start"] : e["end"]]
                    if "신청가능" in clause or "지원가능" in clause:
                        p = None
                if p:
                    candidates.append(p)
        output[field] = (
            {**candidates[0], "reason": "parsed"}
            if candidates
            else dict(value=None, evidence=[], reason="no_supported_parse")
        )
    return output, region


def run(cache, output):
    predictions = {}
    for doc, pages in load(cache).items():
        fields, region = read(pages)
        predictions[doc] = dict(fields=fields, section_range=region)
    refs = json.loads((ROOT / "reference.json").read_text())
    metrics = dict(
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
            metrics["not_established"] += 1
            metrics["absence_abstain"] += p["value"] is None
        else:
            metrics["answerable"] += 1
            metrics["correct_value"] += p["value"] == r["value"]
            metrics["correct_joint"] += p["value"] == r["value"] and covers(
                p["evidence"], r["support_groups"]
            )
            metrics["wrong_value"] += (
                p["value"] is not None and p["value"] != r["value"]
            )
            metrics["abstain"] += p["value"] is None
    stress = []
    for case in json.loads((ROOT / "stress_cases.json").read_text()):
        fields, _ = read([case["text"]])
        stress.append(
            {
                **case,
                "prediction": fields[case["field"]]["value"],
                "exact": fields[case["field"]]["value"] == case["reference"],
            }
        )
    result = dict(
        status="post-hoc refinement on observed sources AND observed probes; not a held-out test",
        metrics=metrics,
        predictions=predictions,
        stress=stress,
    )
    Path(output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            dict(
                metrics=metrics,
                stress_correct=sum(r["exact"] for r in stress),
                stress_total=len(stress),
            )
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--output", default=ROOT / "results/refined.json")
    args = p.parse_args()
    run(args.cache, args.output)
