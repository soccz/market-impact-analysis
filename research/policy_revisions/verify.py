"""Independently audit locked inputs, raw predictions, selective risk and spans."""

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
LABELS = ["correction", "policy_change", "equivalent", "undetermined"]


def load(name):
    return json.loads((ROOT / name).read_text())


def signature(spans):
    return sorted(
        (s["start"], s["end"], s["role"], s["won"])
        for s in spans
        if s["role"] != "other"
    )


def calculate(rows, ps):
    confusion = [[0] * 4 for _ in range(4)]
    argument_hits, joint_hits, accepted_errors, answered_hits = 0, 0, 0, 0
    accepted = 0
    classes = {c: {"n": 0, "accepted": 0, "correctly_answered": 0} for c in LABELS}
    for row, p in zip(rows, ps):
        truth, pred = LABELS.index(row["relation"]), LABELS.index(p["relation"])
        confusion[truth][pred] += 1
        same_args = signature(row["spans"]) == signature(p["arguments"])
        hit = same_args and truth == pred
        argument_hits += same_args
        joint_hits += hit
        accepted += p["accepted"]
        accepted_errors += p["accepted"] and not hit
        answered_hits += p["accepted"] and hit
        classes[row["relation"]]["n"] += 1
        classes[row["relation"]]["accepted"] += p["accepted"]
        classes[row["relation"]]["correctly_answered"] += p["accepted"] and hit
    f1 = []
    for i in range(4):
        denominator = sum(confusion[i]) + sum(r[i] for r in confusion)
        f1.append(2 * confusion[i][i] / denominator if denominator else 0)
    n = len(rows)
    return {
        "n": n,
        "relation_accuracy": sum(confusion[i][i] for i in range(4)) / n,
        "confusion": confusion,
        "macro_f1": sum(f1) / 4,
        "argument_exact": argument_hits / n,
        "joint_exact": joint_hits / n,
        "accepted": accepted,
        "coverage": accepted / n,
        "accepted_errors": accepted_errors,
        "selective_joint_error": accepted_errors / accepted if accepted else None,
        "correctly_answered_fraction": answered_hits / n,
    }, classes


def main():
    report = {
        "locks": {},
        "sets": {},
        "results": {},
        "limitations": [
            "Synthetic amounts/identifiers are masked; nominal row counts overstate linguistic diversity.",
            "Frame bootstrap is conditional on authored wording, not independent language/population uncertainty.",
            "Official case labels are provisional and selected; four of five sources were already seen in v1.",
            "Zero errors on a selected subset does not imply overall accuracy or a calibrated real-world guarantee.",
        ],
    }
    for lockfile in ["freeze.json", "encoder_freeze.json", "matched_freeze.json"]:
        for name, digest in load(lockfile)["sha256"].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, (
                lockfile,
                name,
            )
        report["locks"][lockfile] = "PASS"
    from frozen_encoder import normalized

    datasets = {}
    seen = set()
    for split in ["train", "validation", "evaluation", "official_cases"]:
        rows = [
            json.loads(s)
            for s in (ROOT / f"data/{split}.jsonl").read_text().splitlines()
        ]
        datasets[split] = rows
        assert len({r["id"] for r in rows}) == len(rows)
        texts = {r["text"] for r in rows}
        assert not seen & texts
        seen |= texts
        for r in rows:
            for span in r["spans"]:
                assert r["text"][span["start"] : span["end"]] == span["text"]
            if r["relation"] == "undetermined":
                assert not signature(r["spans"])
            else:
                assert Counter(
                    s["role"] for s in r["spans"] if s["role"] != "other"
                ) == {"previous": 1, "current": 1}
        report["sets"][split] = {
            "rows": len(rows),
            "unique_normalized_passages": len({normalized(r["text"])[0] for r in rows}),
            "label_counts": dict(Counter(r["relation"] for r in rows)),
        }
        if split != "official_cases":
            groups = defaultdict(list)
            for row in rows:
                groups[(row["frame"], row["family"])].append(row["relation"])
            assert all(sorted(v) == sorted(LABELS) for v in groups.values())
    for condition in ["character", "encoder", "matched_character"]:
        result = load(f"results/{condition}/results.json")
        report["results"][condition] = {}
        for split, recorded in result["sets"].items():
            rows = datasets[split]
            pred = load(f"results/{condition}/{split}_predictions.json")
            assert pred["ids"] == [r["id"] for r in rows]
            for row, raw in zip(rows, pred["raw"]):
                probs = raw["relation_probabilities"]
                assert abs(sum(probs.values()) - 1) < 1e-10
                assert max(probs, key=probs.get) == raw["relation"]
                for s in raw["candidates"]:
                    assert row["text"][s["start"] : s["end"]] == s["text"]
                    assert {k: s[k] for k in ["start", "end", "won", "text"]} in [
                        {k: g[k] for k in ["start", "end", "won", "text"]}
                        for g in row["spans"]
                    ]
            scores = recorded["metrics"] if "metrics" in recorded else recorded
            report["results"][condition][split] = {}
            for name, ps in pred["methods"].items():
                actual, classes = calculate(rows, ps)
                for key, value in actual.items():
                    expected = scores[name][key]
                    if isinstance(value, float):
                        assert abs(value - expected) < 1e-10, (
                            condition,
                            split,
                            name,
                            key,
                        )
                    else:
                        assert value == expected, (condition, split, name, key)
                for row, p in zip(rows, ps):
                    if p["accepted"]:
                        assert p["abstention_reason"] is None
                    for span in p["arguments"]:
                        assert row["text"][span["start"] : span["end"]] == span["text"]
                if name in ["constrained", "selective"]:
                    for p in ps:
                        if not p["accepted"] or p["relation"] == "undetermined":
                            continue
                        assert Counter(s["role"] for s in p["arguments"]) == {
                            "previous": 1,
                            "current": 1,
                        }
                        same = len({s["won"] for s in p["arguments"]}) == 1
                        assert same == (p["relation"] == "equivalent")
                        if name == "selective":
                            assert p["score"] >= result["threshold"]
                report["results"][condition][split][name] = {
                    "metrics": actual,
                    "by_true_relation": classes,
                }
    # Review one original numeric realization per distinct normalized passage.
    # Give reviewers only this file and the guide, not predictions or answer keys.
    representatives = {}
    for row in datasets["evaluation"]:
        representatives.setdefault(normalized(row["text"])[0], row)
    with (ROOT / "independent_review.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "id",
                "text",
                "relation",
                "previous_quote",
                "current_quote",
                "reason_quote",
                "ambiguity",
                "reviewer",
            ]
        )
        for r in representatives.values():
            writer.writerow([r["id"], r["text"], "", "", "", "", "", ""])
    (ROOT / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "locks": report["locks"],
                "sets": report["sets"],
                "review_rows": len(representatives),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
