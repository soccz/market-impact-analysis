"""Prepare blind-to-model review sheets; compare only genuinely supplied labels."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from collections import Counter
from build_data import RELATIONS, STATES

ROOT = Path(__file__).resolve().parent
FIELDS = [
    "review_id",
    "query",
    "text",
    "url",
    "source_date",
    "relation",
    "state",
    "before_quotes_json",
    "after_quotes_json",
    "relation_evidence_quotes_json",
    "state_evidence_quotes_json",
    "ambiguous",
    "notes",
]


def load(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def prepare(keypath):
    out = ROOT / "annotation"
    out.mkdir(exist_ok=True)
    assert not (out / "reviewer_a.csv").exists()
    rows = []
    keys = []
    seen = set()
    for name in ["evaluation", "official"]:
        for line in (ROOT / "data" / f"{name}.jsonl").read_text().splitlines():
            x = json.loads(line)
            pattern = re.sub(r"\d[\d,]*", "#", x["text"].replace(x["query"], "TARGET"))
            if name == "evaluation" and pattern in seen:
                continue
            seen.add(pattern)
            identifier = hashlib.sha256(("review-v3:" + x["id"]).encode()).hexdigest()[
                :12
            ]
            rows.append(
                dict(
                    review_id=identifier,
                    query=x["query"],
                    text=x["text"],
                    url=x.get("url", ""),
                    source_date=x.get("date", ""),
                )
            )
            keys.append(
                dict(
                    review_id=identifier,
                    source_file=name,
                    id=x["id"],
                    provisional_relation=x["relation"],
                    provisional_state=x["state"],
                )
            )
    rows.sort(key=lambda x: x["review_id"])
    for reviewer in ["a", "b"]:
        with (out / f"reviewer_{reviewer}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
    keypath.parent.mkdir(parents=True, exist_ok=True)
    assert not keypath.exists()
    keypath.write_text(json.dumps(keys, ensure_ascii=False, indent=2) + "\n")
    (out / "status.json").write_text(
        json.dumps(
            dict(
                status="pending",
                completed_human_reviews=0,
                rows=len(rows),
                synthetic_expressions=96,
                official_passages=12,
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        "Prepared", len(rows), "unannotated records per reviewer; key saved privately"
    )


def compare(a, b, out):
    left = load(a)
    right = load(b)
    assert len({x["review_id"] for x in left}) == len(left)
    assert len({x["review_id"] for x in right}) == len(right)
    right = {x["review_id"]: x for x in right}
    assert set(right) == {x["review_id"] for x in left}, "Review IDs must match"
    pairs = []
    for x in left:
        y = right[x["review_id"]]
        assert all(x[k] == y[k] for k in ["query", "text", "url", "source_date"])
        for row in [x, y]:
            assert row["relation"] in RELATIONS + [""], row["review_id"]
            assert row["state"] in STATES + [""], row["review_id"]
            for key in [
                "before_quotes_json",
                "after_quotes_json",
                "relation_evidence_quotes_json",
                "state_evidence_quotes_json",
            ]:
                if row[key]:
                    quotes = json.loads(row[key])
                    assert isinstance(quotes, list)
                    assert all(
                        isinstance(q, str) and q and q in row["text"] for q in quotes
                    )
        if all(x[k] and y[k] for k in ["relation", "state"]):
            pairs.append((x, y))
    result = dict(
        status=(
            "pending"
            if not pairs
            else (
                "partial"
                if len(pairs) < len(left)
                else "labels_complete_adjudication_pending"
            )
        ),
        total=len(left),
        paired_complete=len(pairs),
        unpaired=len(left) - len(pairs),
        axes={},
    )
    for key in ["relation", "state"]:
        n = len(pairs)
        if not n:
            result["axes"][key] = dict(agreement=None, kappa=None)
            continue
        ca = Counter(x[key] for x, y in pairs)
        cb = Counter(y[key] for x, y in pairs)
        observed = sum(x[key] == y[key] for x, y in pairs) / n
        expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n**2
        result["axes"][key] = dict(
            agreement=observed,
            kappa=(observed - expected) / (1 - expected) if expected < 1 else None,
        )
    result["disagreements"] = [
        dict(
            review_id=x["review_id"],
            a={k: x[k] for k in ["relation", "state", "notes"]},
            b={k: y[k] for k in ["relation", "state", "notes"]},
        )
        for x, y in pairs
        if any(x[k] != y[k] for k in ["relation", "state"])
    ]
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    subs = p.add_subparsers(dest="command", required=True)
    a = subs.add_parser("prepare")
    a.add_argument("--private-key", type=Path, required=True)
    b = subs.add_parser("compare")
    b.add_argument("a", type=Path)
    b.add_argument("b", type=Path)
    b.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args.private_key)
    else:
        compare(args.a, args.b, args.out)


if __name__ == "__main__":
    main()
