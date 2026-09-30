"""Independent artifact, probability, confusion, exact-match and selection audit."""

from collections import Counter
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
REL = ["correction", "policy_change", "restatement", "unspecified"]
STATE = ["asserted", "planned", "under_review", "denied", "hypothetical", "unspecified"]


def read(name):
    return [
        json.loads(s)
        for s in (ROOT / "data" / f"{name}.jsonl").read_text().splitlines()
    ]


def close(a, b):
    assert (a is None and b is None) or (
        a is not None and b is not None and abs(a - b) < 1e-6
    ), (a, b)


def probability(v, n):
    assert (
        len(v) == n
        and all(math.isfinite(x) and 0 <= x <= 1 for x in v)
        and abs(sum(v) - 1) < 1e-5
    )


def decision(raw, variant):
    rp = list(raw["relation_probabilities"])
    sp = raw["state_probabilities"]
    r, s = raw["relation"], raw["state"]
    if variant == "numeric":
        a = [x for x in raw["spans"] if x["role"] == "before"]
        b = [x for x in raw["spans"] if x["role"] == "after"]
        if len(a) == len(b) == 1 and a[0]["won"] != b[0]["won"]:
            rp[2] = 0
            total = sum(rp)
            rp = [x / total for x in rp]
    if variant != "native":
        r = REL[max(range(4), key=rp.__getitem__)]
        s = STATE[max(range(6), key=sp.__getitem__)]
    conf = min(
        [rp[REL.index(r)], sp[STATE.index(s)]]
        + [max(x["probabilities"]) for x in raw["spans"]]
    )
    return r, s, conf


def exact(g, p, r, s):
    a = [(x["start"], x["end"], x["role"], x["won"]) for x in g["spans"]]
    b = [(x["start"], x["end"], x["role"], x["won"]) for x in p["spans"]]
    return {
        "relation": g["relation"] == r,
        "state": g["state"] == s,
        "axes": g["relation"] == r and g["state"] == s,
        "roles": a == b,
        "joint": g["relation"] == r and g["state"] == s and a == b,
    }


def audit_metrics(rows, raw, variant, reported):
    ds = [decision(p, variant) for p in raw]
    es = [exact(g, p, r, s) for g, p, (r, s, c) in zip(rows, raw, ds)]
    assert reported["n"] == len(rows)
    for key in es[0]:
        close(sum(x[key] for x in es) / len(es), reported[key])
    for col, (key, labels) in enumerate([("relation", REL), ("state", STATE)]):
        cm = Counter((g[key], d[col]) for g, d in zip(rows, ds))
        f = []
        for label in labels:
            tp = cm[label, label]
            fp = sum(n for (a, b), n in cm.items() if b == label and a != label)
            fn = sum(n for (a, b), n in cm.items() if a == label and b != label)
            f.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
        close(sum(f) / len(f), reported[key + "_macro_f1"])
    counts = Counter()
    for g, p in zip(rows, raw):

        def chars(spans):
            return {
                (s["kind"], i)
                for s in spans
                for i in range(s["start"], s["end"])
                if not g["text"][i].isspace()
            }

        a = chars(g["evidence"])
        b = chars(p["evidence"])
        counts.update(tp=len(a & b), fp=len(b - a), fn=len(a - b))
    assert dict(counts) == reported["evidence_counts"]
    n = 2 * counts["tp"] + counts["fp"] + counts["fn"]
    close(2 * counts["tp"] / n if n else None, reported["evidence_character_f1"])
    return ds, es


def audit_gate(rows, ds, es, tau, reported):
    accepted = [i for i, d in enumerate(ds) if tau is not None and d[2] >= tau]
    wrong = sum(not es[i]["joint"] for i in accepted)
    assert (len(accepted), wrong) == (reported["accepted"], reported["wrong"])
    close(len(accepted) / len(rows), reported["coverage"])
    close(wrong / len(accepted) if accepted else None, reported["risk"])
    for key, labels in [("relation", REL), ("state", STATE)]:
        for label in labels:
            idx = [i for i in accepted if rows[i][key] == label]
            n = sum(x[key] == label for x in rows)
            bad = sum(not es[i]["joint"] for i in idx)
            v = reported["by_class"][key][label]
            assert (n, len(idx), bad) == (v["n"], v["accepted"], v["wrong"])
            close(len(idx) / n if n else None, v["coverage"])
            close(bad / len(idx) if idx else None, v["risk"])


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fit", action="store_true")
    mode.add_argument("--augmentation", choices=["units", "scope", "both"])
    args = parser.parse_args()
    result_root = ROOT / ("results_fit" if args.fit else "results")
    if args.augmentation:
        result_root = ROOT / "results_augmentation" / args.augmentation
    lock = json.loads(
        (
            ROOT
            / (
                "augmentation_freeze.json"
                if args.augmentation
                else "fit_freeze.json" if args.fit else "partial_freeze.json"
            )
        ).read_text()
    )
    for path, digest in lock["files"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    sets = {p.stem: read(p.stem) for p in (ROOT / "data").glob("*.jsonl")}
    seen = set()
    for name in ["train", "validation", "evaluation"]:
        texts = {x["text"] for x in sets[name]}
        assert not seen & texts
        seen |= texts
        assert len(texts) == len(sets[name])
    held = {(0, 1), (1, 3), (2, 4), (3, 2)}
    for name, rows in sets.items():
        assert len({x["id"] for x in rows}) == len(rows)
        for row in rows:
            for span in row["spans"] + row["evidence"]:
                assert row["text"][span["start"] : span["end"]] == span["text"]
            if name in ["train", "validation"]:
                assert (
                    REL.index(row["relation"]),
                    STATE.index(row["state"]),
                ) not in held
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    for name in ["train", "validation", "evaluation"]:
        patterns = {
            re.sub(r"\d[\d,]*", "#", x["text"].replace(x["query"], "TARGET"))
            for x in sets[name]
        }
        assert len(patterns) == manifest[name]["normalized_strings"]
    for name in ["units", "distractor", "evidence_removed"]:
        for a, b in zip(sets["evaluation"], sets[name]):
            assert all(
                a[k] == b[k]
                for k in ["pair_id", "query", "relation", "state", "composition"]
            )
            assert [
                (s["role"], s["won"]) for s in a["spans"] if s["role"] != "other"
            ] == [(s["role"], s["won"]) for s in b["spans"] if s["role"] != "other"]
    if args.augmentation:
        augmented = [
            json.loads(s)
            for s in (ROOT / "augmentation_data" / f"{args.augmentation}.jsonl")
            .read_text()
            .splitlines()
        ]
        for a, b in zip(sets["train"], augmented):
            assert all(
                a[k] == b[k]
                for k in ["query", "relation", "state", "family", "frame", "pair_id"]
            )
            for sp in b["spans"] + b["evidence"]:
                assert b["text"][sp["start"] : sp["end"]] == sp["text"]
            assert [
                (x["role"], x["won"]) for x in a["spans"] if x["role"] != "other"
            ] == [(x["role"], x["won"]) for x in b["spans"] if x["role"] != "other"]
        assert len(augmented) == 960
        sets["train"] = augmented
    summary = json.loads((result_root / "summary.json").read_text())
    checks = {}
    for tag, report in summary.items():
        name, variant = tag.split(":")
        directory = result_root / name
        for split, m in report["sets"].items():
            rows = sets[split]
            ps = json.loads((directory / f"{split}_predictions.json").read_text())
            assert [r["id"] for r in rows] == [p["id"] for p in ps]
            for row, p in zip(rows, ps):
                probability(p["relation_probabilities"], 4)
                probability(p["state_probabilities"], 6)
                if p.get("joint_probabilities"):
                    j = p["joint_probabilities"]
                    probability(j, 24)
                    for i in range(4):
                        close(sum(j[i * 6 : i * 6 + 6]), p["relation_probabilities"][i])
                    for i in range(6):
                        close(sum(j[i::6]), p["state_probabilities"][i])
                for s in p["spans"]:
                    probability(s["probabilities"], 3)
                    assert row["text"][s["start"] : s["end"]] == s["text"]
                for s in p["evidence"]:
                    assert row["text"][s["start"] : s["end"]] == s["text"]
            ds, es = audit_metrics(rows, ps, variant, m["metrics"])
            audit_gate(rows, ds, es, report["threshold"], m["selective"])
            if split == "validation":
                eligible = []
                for curve in report["validation_curve"]:
                    audit_gate(rows, ds, es, curve["threshold"], curve)
                    if curve["accepted"] >= 20 and curve["risk"] <= 0.05:
                        eligible.append(curve)
                selected = (
                    max(eligible, key=lambda v: (v["accepted"], -v["threshold"]))[
                        "threshold"
                    ]
                    if eligible
                    else None
                )
                assert selected == report["threshold"]
            for composition, cm in m.get("compositions", {}).items():
                idx = [i for i, r in enumerate(rows) if r["composition"] == composition]
                audit_metrics([rows[i] for i in idx], [ps[i] for i in idx], variant, cm)
        base = [
            decision(p, variant)
            for p in json.loads((directory / "evaluation_predictions.json").read_text())
        ]
        removed = [
            decision(p, variant)
            for p in json.loads(
                (directory / "evidence_removed_predictions.json").read_text()
            )
        ]
        diag = report["oracle_evidence_removal"]
        close(
            sum(a[:2] == b[:2] for a, b in zip(base, removed)) / len(base),
            diag["axis_decision_retention"],
        )
        close(
            sum(a[2] - b[2] for a, b in zip(base, removed)) / len(base),
            diag["confidence_drop"],
        )
        for stress in ["distractor", "units"]:
            transformed = [
                decision(p, variant)
                for p in json.loads(
                    (directory / f"{stress}_predictions.json").read_text()
                )
            ]
            close(
                sum(a[:2] == b[:2] for a, b in zip(base, transformed)) / len(base),
                report["sets"][stress]["axis_prediction_invariance"],
            )
        checks[tag] = "PASS"
    training = {}
    for seed in [17, 42, 2026]:
        for mode in (
            ["evidence"] if args.augmentation else ["flat", "factor", "evidence"]
        ):
            tag = f"{mode}-{seed}"
            log = json.loads((result_root / tag / "training.json").read_text())
            assert len(log["steps"]) == (300 if args.fit or args.augmentation else 90)
            assert all(math.isfinite(x["loss"]) for x in log["steps"])
            training[tag] = dict(
                attempted=len(log["steps"]),
                skipped=sum(x["skipped"] for x in log["steps"]),
                checkpoint_sha256=log["checkpoint_sha256"],
            )
    for name in ["a", "b"]:
        with (ROOT / f"annotation/reviewer_{name}.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 108 and all(
            not x["relation"] and not x["state"] for x in rows
        )
    out = dict(
        status="PASS",
        freeze_files=len(lock["files"]),
        variants=checks,
        training=training,
        human_reviews_completed=0,
        limits="Mechanical consistency, not linguistic gold validation or generalization proof.",
    )
    (
        ROOT
        / (
            f"verification_augmentation_{args.augmentation}.json"
            if args.augmentation
            else "verification_fit.json" if args.fit else "verification.json"
        )
    ).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
