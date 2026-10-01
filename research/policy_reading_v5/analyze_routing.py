"""Report retrieval coverage separately; abstention counts as failure in full coverage-adjusted accuracy."""

import json
from statistics import mean
from common import ROOT, datasets, save
from evaluate import decode, correct, choose, metrics, selective


def main():
    sets = datasets()
    runs = {}
    for seed in [17, 42, 2026]:
        raw = {
            n: json.loads((ROOT / f"routing_results/{seed}/{n}.json").read_text())
            for n in sets
        }
        pairs = [
            (r, x["prediction"])
            for r, x in zip(sets["validation"], raw["validation"])
            if x["answerable"]
        ]
        tau, curve = (
            choose([r for r, p in pairs], [decode(p) for r, p in pairs])
            if pairs
            else (None, [])
        )
        report = dict(threshold=tau, validation_curve=curve, sets={})
        for name, rows in sets.items():
            records = raw[name]
            ids = [i for i, x in enumerate(records) if x["answerable"]]
            rr = [rows[i] for i in ids]
            pp = [decode(records[i]["prediction"]) for i in ids]
            checks = {
                i: correct(rows[i], records[i]["prediction"])["joint"] for i in ids
            }
            both = None
            if name != "absent":
                both = sum(
                    checks.get(i, False) and checks.get(i + 1, False)
                    for i in range(0, len(rows), 2)
                )
            g = (
                selective(rr, pp, tau)
                if rr
                else dict(
                    n=0, accepted=0, wrong=0, coverage=0.0, risk=None, by_class={}
                )
            )
            # Model metrics are conditional on the retrieval providing input. Full denominators remain explicit.
            report["sets"][name] = dict(
                n=len(rows),
                routed=len(ids),
                retrieval_coverage=len(ids) / len(rows),
                joint_correct=sum(checks.values()),
                full_joint=sum(checks.values()) / len(rows),
                both_correct=both,
                pair_joint=both / (len(rows) // 2) if both is not None else None,
                routed_metrics=metrics(rr, pp) if rr else None,
                selective=g,
                full_answer_coverage=g["accepted"] / len(rows),
            )
        runs[str(seed)] = report
    table = {}
    for name in sets:
        values = [runs[str(s)]["sets"][name] for s in [17, 42, 2026]]
        table[name] = {
            key: dict(
                mean=mean(r[key] for r in values), values=[r[key] for r in values]
            )
            for key in ["retrieval_coverage", "full_joint", "full_answer_coverage"]
            + (["pair_joint"] if name != "absent" else [])
        }
    save(
        ROOT / "routing_summary.json",
        dict(
            status="Post-result inference-only baseline, separately frozen after primary new-model results",
            runs=runs,
            table=table,
        ),
    )
    print(json.dumps({n: t.get("pair_joint") for n, t in table.items()}, indent=2))


if __name__ == "__main__":
    main()
