"""Audit complete witnesses with Python/Fraction, independent of Z3 execution."""

import copy
import json
import re
from fractions import Fraction
from bridge import ROOT, PREVIOUS, adapt
from rescore_previous import SPECS


def complete_decision(policy, profile):
    values = {v["field"]: v["value"] for v in profile["values"]}
    for d in policy["derived"]:
        values[d["field"]] = Fraction(str(d["constant"])) + sum(
            Fraction(str(values[t["field"]]))
            * Fraction(str(t["numerator"]))
            / Fraction(str(t["denominator"]))
            for t in d["terms"]
        )

    def check(atom):
        value = values[atom["field"]]
        other = atom["value"]
        op = atom["op"]
        if op == "in":
            return value in other
        if op == "not_in":
            return value not in other
        if type(value) in [int, float] or isinstance(value, Fraction):
            value, other = Fraction(str(value)), Fraction(str(other))
        if op == "eq":
            return value == other
        if op == "ne":
            return value != other
        if op == "le":
            return value <= other
        if op == "lt":
            return value < other
        if op == "ge":
            return value >= other
        assert op == "gt"
        return value > other

    return (
        "supported"
        if any(all(check(a) for a in c) for c in policy["clauses"])
        else "contradicted"
    )


def calculate():
    rows = copy.deepcopy(
        json.loads((ROOT / "results/counterexamples.json").read_text())
    )
    for r in rows:
        if r["status"] != "witness":
            continue
        # Z3 escapes non-ASCII strings in as_string(); normalize for JSON readers.
        for v in r["profile"]["values"]:
            domain = r["domains"][v["field"]]
            if (
                isinstance(v["value"], str)
                and isinstance(domain, list)
                and v["value"] not in domain
            ):
                v["value"] = re.sub(
                    r"\\u\{([0-9a-fA-F]+)\}", lambda m: chr(int(m[1], 16)), v["value"]
                )
            if isinstance(domain, list):
                assert v["value"] in domain
            else:
                assert (
                    type(v["value"]) is int
                    and domain["min"] <= v["value"] <= domain["max"]
                )
        ds = json.loads((PREVIOUS / SPECS[r["split"]]).read_text())
        b = next(b for b in ds["bundles"] if b["id"] == r["bundle"])
        records = json.loads(
            (PREVIOUS / f"results/{r['split']}/predictions.json").read_text()
        )
        predicted = next(
            p["parsed"]
            for p in records
            if p["stage"] == "policy"
            and p["model"] == r["model"]
            and p["id"] == r["bundle"]
        )
        for name, p in [("predicted", predicted), ("reference", b["reference_policy"])]:
            assert complete_decision(adapt(p), r["profile"]) == r[name], (
                r["bundle"],
                name,
            )
        r["independent_fraction_replay"] = True
    return rows


if __name__ == "__main__":
    data = calculate()
    (ROOT / "results/counterexamples_audited.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    )
    print(sum(r.get("independent_fraction_replay", False) for r in data))
