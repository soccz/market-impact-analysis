"""Independent audit of routing abstention, source restoration and full denominators."""

import json
from common import ROOT, datasets, check_lock, sha, save
from build_data import amounts
from verify import audit
from routing import retrieve


def main():
    check_lock()
    for name, digest in json.loads((ROOT / "routing_freeze.json").read_text())[
        "files"
    ].items():
        assert sha(ROOT / name) == digest, name
    sets = datasets()
    summary = json.loads((ROOT / "routing_summary.json").read_text())
    total = 0
    for seed in [17, 42, 2026]:
        run = summary["runs"][str(seed)]
        tau = run["threshold"]
        for name, rows in sets.items():
            records = json.loads(
                (ROOT / f"routing_results/{seed}/{name}.json").read_text()
            )
            result = run["sets"][name]
            ids = []
            raw = []
            for i, (r, rec) in enumerate(zip(rows, records)):
                assert r["id"] == rec["id"]
                v = retrieve(r["query"], r["text"])
                assert rec["selected_intervals"] == [list(x) for x in v["intervals"]]
                assert rec["answerable"] == v["answerable"]
                p = rec["prediction"]
                if not rec["answerable"]:
                    assert p is None
                    continue
                ids.append(i)
                raw.append(p)
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
                assert len(p["spans"]) == 4
                for s, found in zip(p["spans"], amounts(r["text"])):
                    assert {k: s[k] for k in found} == found
                    audit.probability(s["probabilities"], 3)
                    assert (
                        s["role"]
                        == ["other", "before", "after"][
                            max(range(3), key=s["probabilities"].__getitem__)
                        ]
                    )
                    if not any(
                        a <= s["start"] < s["end"] <= b for a, b in v["intervals"]
                    ):
                        assert s["role"] == "other" and s["probabilities"] == [
                            1.0,
                            0.0,
                            0.0,
                        ]
                for e in p["evidence"]:
                    assert r["text"][e["start"] : e["end"]] == e["text"]
                    assert any(
                        a <= e["start"] < e["end"] <= b for a, b in v["intervals"]
                    ), "Evidence crossed an excluded gap"
            rr = [rows[i] for i in ids]
            if rr:
                ds, es = audit.audit_metrics(
                    rr, raw, "native", result["routed_metrics"]
                )
                audit.audit_gate(rr, ds, es, tau, result["selective"])
                flags = {i: e["joint"] for i, e in zip(ids, es)}
            else:
                assert (
                    result["routed_metrics"] is None
                    and result["selective"]["accepted"] == 0
                )
                flags = {}
            assert result["n"] == len(rows) and result["routed"] == len(ids)
            assert result["joint_correct"] == sum(flags.values())
            audit.close(result["full_joint"], sum(flags.values()) / len(rows))
            audit.close(result["retrieval_coverage"], len(ids) / len(rows))
            audit.close(
                result["full_answer_coverage"],
                result["selective"]["accepted"] / len(rows),
            )
            if name != "absent":
                both = sum(
                    flags.get(i, False) and flags.get(i + 1, False)
                    for i in range(0, len(rows), 2)
                )
                assert result["both_correct"] == both
                audit.close(result["pair_joint"], both / (len(rows) // 2))
            else:
                assert result["pair_joint"] is None
            if name == "validation":
                candidates = []
                for t in range(100):
                    g = run["validation_curve"][t]
                    audit.audit_gate(rr, ds, es, t / 100, g)
                    if g["accepted"] >= 20 and g["risk"] <= 0.05:
                        candidates.append((g["accepted"], -t, t / 100))
                assert tau == (max(candidates)[2] if candidates else None)
            total += 1
    for name, table in summary["table"].items():
        for key, value in table.items():
            expected = [
                summary["runs"][str(s)]["sets"][name][key] for s in [17, 42, 2026]
            ]
            assert value["values"] == expected
            audit.close(value["mean"], sum(expected) / 3)
    save(
        ROOT / "routing_verification.json",
        dict(
            status="PASS",
            metric_sets=total,
            full_denominators=True,
            abstentions_not_counted_correct=True,
            source_spans_no_cross_gap=True,
            summary_sha256=sha(ROOT / "routing_summary.json"),
        ),
    )
    print(
        "PASS routing:",
        total,
        "sets; full denominators, explicit abstention, gaps and validation gates",
    )


if __name__ == "__main__":
    main()
