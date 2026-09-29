"""Synthetic engineering examples, written for the September 2026 retrospective.

The annotation example executes the unchanged public consensus script inside a
temporary directory. The calendar and coverage examples are new explanations,
not client code, historical performance, or a model inference benchmark.
Only Python's standard library is required; no network or real data is used.
"""

import argparse
from bisect import bisect_left, bisect_right
import csv
from datetime import date
import hashlib
import json
from pathlib import Path
from statistics import mean
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
CONSENSUS = ROOT / "v2/code/train500/build_train_500_consensus.py"
LABEL_CASES = [
    ("agree", "세 판독이 같으면", ["support"] * 3, ["high"] * 3),
    (
        "majority",
        "다수결과 확신이 확보되면",
        ["support", "support", "neutral"],
        ["high", "medium", "high"],
    ),
    (
        "low",
        "다수결이어도 저확신이면",
        ["support", "support", "neutral"],
        ["high", "low", "high"],
    ),
    (
        "split",
        "세 판독이 모두 다르면",
        ["support", "contradict", "neutral"],
        ["high"] * 3,
    ),
    ("agree_low", "모두 저확신인데 일치하면", ["support"] * 3, ["low"] * 3),
    (
        "missing",
        "한 판독 파일에 행이 없으면",
        ["support", "support", None],
        ["high", "high", None],
    ),
]


def run_consensus():
    """Run the real historical program against invented, isolated CSV inputs."""
    with TemporaryDirectory(prefix="policy-engineering-demo-") as tmp:
        work = Path(tmp)
        (work / "외주").mkdir()
        (work / "splits").mkdir()
        for reader in range(3):
            with (work / "외주" / f"train_500_{reader + 1}_final.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "news_id",
                        "relevance",
                        "stance_label",
                        "confidence",
                        "reason",
                    ],
                )
                writer.writeheader()
                for key, _, labels, confidence in LABEL_CASES:
                    if key == "missing" and reader == 2:
                        continue
                    writer.writerow(
                        {
                            "news_id": key,
                            "relevance": "2",
                            "stance_label": labels[reader],
                            "confidence": confidence[reader],
                            "reason": "Invented input for documentation only",
                        }
                    )
        subprocess.run(
            [sys.executable, str(CONSENSUS)],
            cwd=work,
            check=True,
            capture_output=True,
            text=True,
        )
        with (work / "splits/train_500_consensus.csv").open(encoding="utf-8-sig") as f:
            output = {row["news_id"]: row for row in csv.DictReader(f)}
        with (work / "splits/train_500_review_queue.csv").open(
            encoding="utf-8-sig"
        ) as f:
            queue = {row["news_id"] for row in csv.DictReader(f)}
    cases = []
    for key, label, labels, confidence in LABEL_CASES:
        result = output.get(key)
        cases.append(
            {
                "id": key,
                "label": label,
                "votes": labels,
                "confidence": confidence,
                "status": (
                    result["consensus_status"] if result else "absent_from_output"
                ),
                "candidate": result["consensus_stance"] if result else None,
                "in_review_queue": key in queue,
            }
        )
    return {
        "mode": "unchanged_historical_script",
        "source": str(CONSENSUS.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(CONSENSUS.read_bytes()).hexdigest(),
        "cases": cases,
        "input_union": len(LABEL_CASES),
        "output_rows": len(output),
        "review_rows": len(queue),
    }


def pair_return(prices, news_date, target_date, start_rule):
    """A new explanatory function: compare two explicit calendar conventions."""
    if start_rule not in {"same_or_next", "strictly_next"}:
        raise ValueError(start_rule)
    dates = sorted(prices)
    search = bisect_left if start_rule == "same_or_next" else bisect_right
    start = search(dates, news_date)
    end = bisect_left(dates, target_date)
    if start >= len(dates) or end >= len(dates) or end <= start:
        return None
    return {
        "start": dates[start].isoformat(),
        "end": dates[end].isoformat(),
        "start_price": prices[dates[start]],
        "end_price": prices[dates[end]],
        "return_pct": round((prices[dates[end]] / prices[dates[start]] - 1) * 100, 6),
    }


def calendar_example():
    prices = {date(2026, 7, d): p for d, p in [(7, 100), (8, 120), (10, 110)]}
    return {
        "mode": "new_synthetic_explanation",
        "news_date": "2026-07-07",
        "target_date": "2026-07-10",
        "prices": [
            {"date": day.isoformat(), "price": value} for day, value in prices.items()
        ],
        "same_or_next": pair_return(
            prices, date(2026, 7, 7), date(2026, 7, 10), "same_or_next"
        ),
        "strictly_next": pair_return(
            prices, date(2026, 7, 7), date(2026, 7, 10), "strictly_next"
        ),
        "unavailable": pair_return(
            prices, date(2026, 7, 7), date(2026, 8, 10), "same_or_next"
        ),
    }


def coverage_example():
    policies = ["A", "B", "C", "D"]
    # Invented per-policy rates, not outputs from any project model.
    models = {"X": {"A": 90, "B": 90}, "Y": {"A": 95, "B": 95, "C": 10, "D": 10}}
    common = set.intersection(*(set(values) for values in models.values()))
    result = {
        "mode": "new_synthetic_explanation",
        "policies": policies,
        "models": models,
    }
    for scope in ["available", "common"]:
        summary = {}
        for model, values in models.items():
            selected = sorted(values if scope == "available" else common)
            summary[model] = {
                "policies": selected,
                "rate": mean(values[p] for p in selected),
                "n": len(selected),
            }
        result[scope] = summary
    return result


def build_report():
    return {
        "synthetic_only": True,
        "consensus": run_consensus(),
        "calendar": calendar_example(),
        "coverage": coverage_example(),
    }


def verify(report):
    cases = {case["id"]: case for case in report["consensus"]["cases"]}
    expected = {
        "agree": "unanimous",
        "majority": "majority_auto",
        "low": "needs_review_low",
        "split": "needs_review_split",
        "agree_low": "unanimous",
        "missing": "absent_from_output",
    }
    assert {key: case["status"] for key, case in cases.items()} == expected
    assert {key for key, case in cases.items() if case["in_review_queue"]} == {
        "low",
        "split",
    }
    assert (
        cases["low"]["candidate"] == "support"
    )  # A candidate is not an adjudicated answer.
    assert (
        report["consensus"]["input_union"] == 6
        and report["consensus"]["output_rows"] == 5
    )
    calendar = report["calendar"]
    assert calendar["same_or_next"]["return_pct"] == 10
    assert abs(calendar["strictly_next"]["return_pct"] + 8.333333) < 1e-6
    assert calendar["unavailable"] is None
    # A non-session publication must select the first available later date.
    prices = {date(2026, 7, 13): 100, date(2026, 7, 15): 105}
    for rule in ["same_or_next", "strictly_next"]:
        assert (
            pair_return(prices, date(2026, 7, 12), date(2026, 7, 15), rule)["start"]
            == "2026-07-13"
        )
        assert pair_return(prices, date(2026, 7, 16), date(2026, 7, 18), rule) is None
    coverage = report["coverage"]
    assert coverage["available"]["X"]["rate"] > coverage["available"]["Y"]["rate"]
    assert coverage["common"]["X"]["rate"] < coverage["common"]["Y"]["rate"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = build_report()
    if args.check:
        verify(report)
        print(
            "PASS: original consensus program (6 cases), calendar boundaries, coverage reversal; synthetic only"
        )
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
