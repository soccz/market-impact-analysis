"""Independently recompute every metric, paired success, attachment and threshold."""

from collections import Counter
import importlib.util
import json
from common import ROOT, V3, datasets, check_lock, save, sha
from build_data import amounts

spec = importlib.util.spec_from_file_location("previous_audit", V3 / "verify.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
CONDITIONS = ["legacy_plain", "legacy_marked", "plain", "marked", "blind"]
SEEDS = [17, 42, 2026]


def signature(p):
    return [
        p["relation_probabilities"],
        p["state_probabilities"],
        [s["probabilities"] for s in p["spans"]],
    ]


def verify_pairs(rows, ps, decisions, checks, tau, reported):
    if rows[0]["target"] is None:
        assert reported is None
        return
    count = Counter()
    for i in range(0, len(rows), 2):
        assert rows[i]["document_id"] == rows[i + 1]["document_id"]
        assert rows[i]["text"] == rows[i + 1]["text"]
        a, b = ps[i : i + 2]
        for key in ["joint", "axes", "roles"]:
            count["both_" + key] += checks[i][key] and checks[i + 1][key]
        count["axes_changed"] += (a["relation"], a["state"]) != (
            b["relation"],
            b["state"],
        )
        count["roles_changed"] += [s["role"] for s in a["spans"]] != [
            s["role"] for s in b["spans"]
        ]
        accepted = (
            tau is not None and decisions[i][2] >= tau and decisions[i + 1][2] >= tau
        )
        count["both_accepted"] += accepted
        count["accepted_pair_wrong"] += accepted and not (
            checks[i]["joint"] and checks[i + 1]["joint"]
        )
    assert reported["n"] == len(rows) // 2
    for key, value in count.items():
        assert reported[key] == value, (key, reported[key], value)
    audit.close(reported["pair_joint"], count["both_joint"] / (len(rows) // 2))


def main():
    check_lock()
    sets = datasets()
    report = json.loads((ROOT / "summary.json").read_text())
    checked = 0
    for condition in CONDITIONS:
        for seed in SEEDS:
            folder = ROOT / "results" / condition / str(seed)
            raw = {
                name: json.loads((folder / (name + ".json")).read_text())
                for name in sets
            }
            run = report["runs"][f"{condition}/{seed}"]
            tau = run["threshold"]
            for name, rows in sets.items():
                ps = raw[name]
                assert [r["id"] for r in rows] == [p["id"] for p in ps]
                for r, p in zip(rows, ps):
                    audit.probability(p["relation_probabilities"], 4)
                    audit.probability(p["state_probabilities"], 6)
                    assert (
                        p["relation"]
                        == audit.REL[
                            max(range(4), key=p["relation_probabilities"].__getitem__)
                        ]
                    )
                    assert (
                        p["state"]
                        == audit.STATE[
                            max(range(6), key=p["state_probabilities"].__getitem__)
                        ]
                    )
                    found = amounts(r["text"])
                    assert len(p["spans"]) == len(found) == 4
                    for s, f in zip(p["spans"], found):
                        assert {k: s[k] for k in f} == f
                        audit.probability(s["probabilities"], 3)
                        assert (
                            s["role"]
                            == ["other", "before", "after"][
                                max(range(3), key=s["probabilities"].__getitem__)
                            ]
                        )
                    for s in p["evidence"]:
                        assert 0 <= s["start"] < s["end"] <= len(r["text"])
                        assert r["text"][s["start"] : s["end"]] == s["text"]
                result = run["sets"][name]
                ds, es = audit.audit_metrics(rows, ps, "native", result["metrics"])
                audit.audit_gate(rows, ds, es, tau, result["selective"])
                verify_pairs(rows, ps, ds, es, tau, result["pairs"])
                other = [
                    (g, s)
                    for r, p in zip(rows, ps)
                    for g, s in zip(r["spans"], p["spans"])
                    if g["role"] == "other"
                ]
                target = [
                    (g, s)
                    for r, p in zip(rows, ps)
                    for g, s in zip(r["spans"], p["spans"])
                    if g["role"] != "other"
                ]
                assert result["attachment"] == dict(
                    other_total=len(other),
                    other_wrong=sum(s["role"] != "other" for g, s in other),
                    target_total=len(target),
                    target_wrong=sum(g["role"] != s["role"] for g, s in target),
                )
                for family, fragment in result["families"].items():
                    ids = [i for i, r in enumerate(rows) if r["family"] == int(family)]
                    rr, pp = [rows[i] for i in ids], [ps[i] for i in ids]
                    dd, ee = audit.audit_metrics(rr, pp, "native", fragment["metrics"])
                    verify_pairs(rr, pp, dd, ee, tau, fragment["pairs"])
                if name == "validation":
                    candidates = []
                    for t in range(100):
                        accepted = [i for i, d in enumerate(ds) if d[2] >= t / 100]
                        wrong = sum(not es[i]["joint"] for i in accepted)
                        curve = run["validation_curve"][t]
                        assert curve["threshold"] == t / 100
                        audit.audit_gate(rows, ds, es, t / 100, curve)
                        if len(accepted) >= 20 and wrong / len(accepted) <= 0.05:
                            candidates.append((len(accepted), -t, t / 100))
                    assert tau == (max(candidates)[2] if candidates else None)
                if condition == "blind" and name != "absent":
                    for a, b in zip(ps[::2], ps[1::2]):
                        assert signature(a) == signature(b), (
                            name,
                            "blind input must be identical",
                        )
                    assert result["pairs"]["both_joint"] == 0
                checked += 1
            for a, b in zip(raw["evaluation"], raw["units"]):
                assert signature(a) == signature(
                    b
                ), "unit rewrite changed normalized probabilities"
    for condition, table in report["table"].items():
        for name, values in table.items():
            runs = [report["runs"][f"{condition}/{s}"]["sets"][name] for s in SEEDS]
            for key in [
                "joint",
                "axes",
                "roles",
                "evidence_character_f1",
                "pair_joint",
            ]:
                if key not in values:
                    continue
                want = [
                    (
                        r["pairs"]["pair_joint"]
                        if key == "pair_joint"
                        else r["metrics"][key]
                    )
                    for r in runs
                ]
                assert values[key]["values"] == want
                valid = [x for x in want if x is not None]
                audit.close(
                    values[key]["mean"], sum(valid) / len(valid) if valid else None
                )
            for key in ["other_total", "other_wrong"]:
                assert values[key] == sum(r["attachment"][key] for r in runs)
    save(
        ROOT / "verification.json",
        dict(
            status="PASS",
            metric_sets=checked,
            metric_source="Independent v3 metric audit and separate v5 paired/attachment recalculation",
            paired_unit_probabilities_exact=True,
            blind_query_probabilities_exact=True,
            family_slices_and_validation_gates=True,
            summary_sha256=sha(ROOT / "summary.json"),
        ),
    )
    print(
        "PASS:",
        checked,
        "metric sets; family slices, paired-query counts, gates, unit and blind identity",
    )


if __name__ == "__main__":
    main()
