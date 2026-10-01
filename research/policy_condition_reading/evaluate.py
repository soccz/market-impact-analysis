"""Evaluate saved predictions separately from source reading and inference."""

import argparse
import itertools
import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from reader import FIELDS, execute

ROOT = Path(__file__).resolve().parent
METHODS = ["document_first", "tfidf_top1", "klue_top1", "tfidf_consensus3"]
PAIRS = [
    ("moving2022", "moving2023"),
    ("rent2023", "rent2024"),
    ("allowance2023", "allowance2024"),
    ("culture2023", "culture2024"),
]


def write(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def covers(spans, groups):
    # The union may contain overlapping retrieval windows. A complete individual
    # reference span must fit in a returned span; partial overlap is insufficient.
    return bool(groups) and all(
        any(
            any(
                p["page"] == r["page"]
                and p["start"] <= r["start"]
                and p["end"] >= r["end"] - 1
                for p in spans
            )
            for r in group
        )
        for group in groups
    )


def relation(a, b):
    return (
        "unknown" if a is None or b is None else ("unchanged" if a == b else "changed")
    )


def evaluate(prediction_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pred = json.loads(Path(prediction_path).read_text())
    refs = json.loads((ROOT / "reference.json").read_text())
    lookup = {(r["document"], r["field"]): r for r in refs}
    rows, metrics = [], {}
    for method in METHODS:
        count = Counter(
            answerable=0,
            correct_value=0,
            correct_joint=0,
            wrong_value=0,
            abstain=0,
            retrieval_complete=0,
            not_established=0,
            absence_abstain=0,
        )
        for r in refs:
            p = pred[r["document"]][method][r["field"]]
            exact = p["value"] == r["value"]
            grounded = covers(p["evidence"], r["support_groups"])
            retrieved = covers(p["retrieved"], r["support_groups"])
            if r["value"] is None:
                count["not_established"] += 1
                count["absence_abstain"] += p["value"] is None
                outcome = (
                    "absence_abstain" if p["value"] is None else "unsupported_answer"
                )
            else:
                count["answerable"] += 1
                count["correct_value"] += exact
                count["correct_joint"] += exact and grounded
                count["retrieval_complete"] += retrieved
                outcome = (
                    "correct"
                    if exact
                    else ("abstain" if p["value"] is None else "wrong_value")
                )
                if not exact:
                    count[outcome] += 1
            rows.append(
                dict(
                    document=r["document"],
                    field=r["field"],
                    method=method,
                    reference=r["value"],
                    prediction=p["value"],
                    value_exact=exact,
                    evidence_complete=grounded,
                    retrieval_complete=retrieved,
                    outcome=outcome,
                    reason=p["reason"],
                )
            )
        metrics[method] = dict(count)
    pair_rows = []
    for a, b in PAIRS:
        for field in FIELDS:
            ra, rb = lookup[a, field]["value"], lookup[b, field]["value"]
            expected = relation(ra, rb)
            for method in METHODS:
                actual = relation(
                    pred[a][method][field]["value"], pred[b][method][field]["value"]
                )
                pair_rows.append(
                    dict(
                        before=a,
                        after=b,
                        field=field,
                        method=method,
                        reference=expected,
                        prediction=actual,
                        exact=actual == expected,
                    )
                )
    profiles = {
        "age": [{"age": x} for x in [18, 19, 20, 22, 23, 24, 34, 35, 39, 40, None]],
        "income": [
            {"income_percent": x, "recipient": scope}
            for x, scope in itertools.product(
                [0, 60, 120, 120.01, 150, 150.01, 200, None], ["first", "returning"]
            )
        ],
        "welfare": [
            {"benefits": x}
            for x in [[], ["education"], ["housing"], ["livelihood"], ["medical"], None]
        ],
    }
    dates = {
        str(date.fromisoformat(d) + timedelta(days=delta))
        for r in refs
        if r["field"] == "birth"
        for d in r["value"]
        for delta in [-1, 0, 1]
    }
    profiles["birth"] = [{"birth": d} for d in sorted(dates)] + [{"birth": None}]
    cases, gate_counts = [], {}
    for method in METHODS:
        count = Counter(
            total=0,
            reference_decidable=0,
            agreed_decidable=0,
            wrong_decidable=0,
            abstained_decidable=0,
            reference_unknown=0,
            answered_unknown=0,
        )
        for r in refs:
            for i, profile in enumerate(profiles[r["field"]]):
                expected = execute(r["field"], r["value"], profile)
                actual = execute(
                    r["field"],
                    pred[r["document"]][method][r["field"]]["value"],
                    profile,
                )
                count["total"] += 1
                if expected == "unknown":
                    count["reference_unknown"] += 1
                    count["answered_unknown"] += actual != "unknown"
                else:
                    count["reference_decidable"] += 1
                    count["agreed_decidable"] += actual == expected
                    count["wrong_decidable"] += (
                        actual != expected and actual != "unknown"
                    )
                    count["abstained_decidable"] += actual == "unknown"
                cases.append(
                    dict(
                        document=r["document"],
                        field=r["field"],
                        case=i,
                        method=method,
                        input=profile,
                        reference=expected,
                        prediction=actual,
                    )
                )
        gate_counts[method] = dict(count)
    layout = []
    for doc in pred:
        before = sum(
            pred[doc]["unsorted_tfidf_top1"][f]["value"] == lookup[doc, f]["value"]
            for f in FIELDS
            if lookup[doc, f]["value"] is not None
        )
        after = sum(
            pred[doc]["tfidf_top1"][f]["value"] == lookup[doc, f]["value"]
            for f in FIELDS
            if lookup[doc, f]["value"] is not None
        )
        layout.append(
            dict(
                document=doc,
                unsorted_correct=before,
                sorted_correct=after,
                answerable=sum(lookup[doc, f]["value"] is not None for f in FIELDS),
            )
        )
    summary = dict(
        documents=8,
        programs=4,
        pairs=4,
        independent_human_reviews=0,
        metrics=metrics,
        gates=gate_counts,
        pair_metrics={
            m: dict(
                decidable=14,
                correct=sum(
                    r["exact"]
                    for r in pair_rows
                    if r["method"] == m and r["reference"] != "unknown"
                ),
                abstain=sum(
                    r["prediction"] == "unknown"
                    for r in pair_rows
                    if r["method"] == m and r["reference"] != "unknown"
                ),
            )
            for m in METHODS
        },
        layout_ablation=layout,
    )
    # Only one unavailable welfare pair: 16 fields minus 1 = 15 decidable pairs.
    for m in METHODS:
        summary["pair_metrics"][m]["decidable"] = sum(
            r["method"] == m and r["reference"] != "unknown" for r in pair_rows
        )
    write(output / "summary.json", summary)
    write(output / "errors.json", rows)
    write(output / "pairs.json", pair_rows)
    write(output / "profiles.json", profiles)
    write(output / "gates.json", cases)
    report = [
        "# 실제 공고의 자동 조건 판독: 탐색 결과",
        "",
        "독립 사람 검수 0명. 4개 사업·8개 공고의 잠정 기준값과 비교한 탐색 결과이며 블라인드 정확도가 아닙니다.",
        "",
        "| 방법 | 값 일치 / 30 | 값+근거 / 30 | 잘못된 값 | 보류 | 미확인 2항목의 보류 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, c in metrics.items():
        report.append(
            f"| {method} | {c['correct_value']} | {c['correct_joint']} | {c['wrong_value']} | {c['abstain']} | {c['absence_abstain']} / 2 |"
        )
    report += [
        "",
        "숫자가 같아도 근거가 대상 조건을 뒷받침하지 않으면 값+근거 일치로 세지 않습니다. 빈 값은 조건 폐지가 아닙니다. KLUE는 검색 학습 없이 평균 풀링한 약한 신경망 기준선입니다.",
        "",
        "## 고정 가상 입력의 조건 실행",
        "",
        "같은 추출값을 반복 적용한 민감도 분석이며 독립 실제 사례 수나 전체 자격 판정 정확도가 아닙니다.",
        "",
    ]
    for method, c in gate_counts.items():
        report.append(
            f"- {method}: 기준 판독 가능한 {c['reference_decidable']}개 중 일치 {c['agreed_decidable']}, 불일치 {c['wrong_decidable']}, 보류 {c['abstained_decidable']}; 기준 확인 필요 입력에 답변 {c['answered_unknown']}."
        )
    (output / "RESULTS.md").write_text("\n".join(report) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions", type=Path, default=ROOT / "results/predictions.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(args.predictions, args.output), ensure_ascii=False, indent=2
        )
    )
