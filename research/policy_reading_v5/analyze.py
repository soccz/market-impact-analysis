"""Pre-fixed paired-query evaluation, family slices and validation-only gating."""

from collections import defaultdict
import json
from statistics import mean
from common import ROOT, datasets, check_lock, save
from evaluate import correct, decode, metrics, choose, selective

CONDITIONS = ["legacy_plain", "legacy_marked", "plain", "marked", "blind"]
SEEDS = [17, 42, 2026]


def pairs(rows, ps, tau):
    groups = defaultdict(list)
    for r, p in zip(rows, ps):
        groups[r["document_id"]].append((r, p))
    if any(len(g) != 2 for g in groups.values()):
        return None
    counts = dict(
        both_joint=0,
        both_axes=0,
        both_roles=0,
        axes_changed=0,
        roles_changed=0,
        both_accepted=0,
        accepted_pair_wrong=0,
    )
    for g in groups.values():
        checks = [correct(r, p) for r, p in g]
        for key in ["joint", "axes", "roles"]:
            counts["both_" + key] += all(c[key] for c in checks)
        a, b = [p for r, p in g]
        counts["axes_changed"] += (a["relation"], a["state"]) != (
            b["relation"],
            b["state"],
        )
        counts["roles_changed"] += [s["role"] for s in a["spans"]] != [
            s["role"] for s in b["spans"]
        ]
        accepted = tau is not None and min(a["confidence"], b["confidence"]) >= tau
        counts["both_accepted"] += accepted
        counts["accepted_pair_wrong"] += accepted and not all(
            c["joint"] for c in checks
        )
    return dict(n=len(groups), **counts, pair_joint=counts["both_joint"] / len(groups))


def attachment(rows, ps):
    other_total = other_wrong = target_total = target_wrong = 0
    for r, p in zip(rows, ps):
        for g, s in zip(r["spans"], p["spans"]):
            if g["role"] == "other":
                other_total += 1
                other_wrong += s["role"] != "other"
            else:
                target_total += 1
                target_wrong += s["role"] != g["role"]
    return dict(
        other_total=other_total,
        other_wrong=other_wrong,
        target_total=target_total,
        target_wrong=target_wrong,
    )


def main():
    check_lock()
    sets = datasets()
    report = dict(runs={}, table={})
    for condition in CONDITIONS:
        for seed in SEEDS:
            folder = ROOT / "results" / condition / str(seed)
            raw = {
                name: json.loads((folder / (name + ".json")).read_text())
                for name in sets
            }
            predictions = {
                name: [decode(p) for p in rows] for name, rows in raw.items()
            }
            tau, curve = choose(sets["validation"], predictions["validation"])
            record = dict(threshold=tau, validation_curve=curve, sets={})
            for name, rows in sets.items():
                ps = predictions[name]
                record["sets"][name] = dict(
                    metrics=metrics(rows, ps),
                    pairs=pairs(rows, ps, tau),
                    attachment=attachment(rows, ps),
                    selective=selective(rows, ps, tau),
                    families={},
                )
                for family in sorted({r["family"] for r in rows}):
                    ids = [i for i, r in enumerate(rows) if r["family"] == family]
                    rr, pp = [rows[i] for i in ids], [ps[i] for i in ids]
                    record["sets"][name]["families"][str(family)] = dict(
                        metrics=metrics(rr, pp), pairs=pairs(rr, pp, tau)
                    )
            report["runs"][f"{condition}/{seed}"] = record
        report["table"][condition] = {}
        for name in sets:
            runs = [report["runs"][f"{condition}/{s}"]["sets"][name] for s in SEEDS]
            summary = {
                k: dict(
                    mean=(
                        mean(
                            r["metrics"][k] for r in runs if r["metrics"][k] is not None
                        )
                        if any(r["metrics"][k] is not None for r in runs)
                        else None
                    ),
                    values=[r["metrics"][k] for r in runs],
                )
                for k in ["joint", "axes", "roles", "evidence_character_f1"]
            }
            if runs[0]["pairs"]:
                summary["pair_joint"] = dict(
                    mean=mean(r["pairs"]["pair_joint"] for r in runs),
                    values=[r["pairs"]["pair_joint"] for r in runs],
                )
            summary["other_wrong"] = sum(r["attachment"]["other_wrong"] for r in runs)
            summary["other_total"] = sum(r["attachment"]["other_total"] for r in runs)
            report["table"][condition][name] = summary
    save(ROOT / "summary.json", report)
    print(
        json.dumps(
            {
                c: {
                    s: round(v["pair_joint"]["mean"] * 100, 2)
                    for s, v in ss.items()
                    if "pair_joint" in v
                }
                for c, ss in report["table"].items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
