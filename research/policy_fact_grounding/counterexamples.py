"""Find bounded, replayable disagreement witnesses; never claim source truth."""

from datetime import date
import json
import bridge
from bridge import compile_program, variable, literal, FIELDS, adapt, interpret
import z3


def witness(predicted, reference, allowed):
    predicted, reference = adapt(predicted), adapt(reference)
    pf, _, ps = compile_program(predicted, allowed)
    rf, _, rs = compile_program(reference, allowed)
    if ps != "valid" or rs != "valid":
        return dict(status="unavailable", predicted_status=ps, reference_status=rs)
    fields = sorted(
        {
            a["field"]
            for p in [predicted, reference]
            for c in p["clauses"]
            for a in c
            if a["field"] in FIELDS
        }
        | {
            t["field"]
            for p in [predicted, reference]
            for d in p["derived"]
            for t in d["terms"]
        }
    )
    solver = z3.Solver()
    solver.set(timeout=3000)
    domains = {}
    for f in fields:
        kind = FIELDS[f]["type"]
        v = variable(f)
        if kind == "number":
            upper = (
                120
                if f == "age_years"
                else (
                    168
                    if f == "weekly_work_hours"
                    else 1000 if f == "income_pct" else 10**12
                )
            )
            solver.add(v >= 0, v <= upper, v == z3.ToReal(z3.ToInt(v)))
            domains[f] = dict(min=0, max=upper, integer=True)
        elif kind == "date":
            solver.add(
                v >= date(2000, 1, 1).toordinal(), v <= date(2100, 12, 31).toordinal()
            )
            domains[f] = ["2000-01-01", "2100-12-31"]
        elif kind == "string":
            vals = sorted(
                {
                    x
                    for p in [predicted, reference]
                    for c in p["clauses"]
                    for a in c
                    if a["field"] == f
                    for x in (
                        a["value"] if a["op"] in ["in", "not_in"] else [a["value"]]
                    )
                }
                | {"__other__"}
            )
            solver.add(z3.Or([v == z3.StringVal(x) for x in vals]))
            domains[f] = vals
        else:
            domains[f] = [False, True]
    solver.add(z3.Xor(pf, rf))
    status = str(solver.check())
    if status != "sat":
        return dict(
            status="no_witness_in_domain" if status == "unsat" else "solver_unknown",
            domains=domains,
        )
    model = solver.model()
    values = {}
    for f in fields:
        value = model.eval(variable(f), model_completion=True)
        kind = FIELDS[f]["type"]
        values[f] = (
            z3.is_true(value)
            if kind == "boolean"
            else (
                value.as_string()
                if kind == "string"
                else (
                    date.fromordinal(value.as_long()).isoformat()
                    if kind == "date"
                    else value.numerator_as_long() // value.denominator_as_long()
                )
            )
        )
    q = dict(
        status="ready",
        assertion=True,
        values=[dict(field=k, value=v) for k, v in values.items()],
    )
    pa = interpret(predicted, q, allowed)["decision"]
    ra = interpret(reference, q, allowed)["decision"]
    assert (
        pa != ra
        and pa in ["supported", "contradicted"]
        and ra in ["supported", "contradicted"]
    )
    return dict(
        status="witness", domains=domains, profile=q, predicted=pa, reference=ra
    )


def previous_audit():
    out = []
    for split, refname in [
        ("official", "official_reference.json"),
        ("composition", "composition.json"),
        ("capability_extended", "capability_reference.json"),
    ]:
        data = json.loads((bridge.PREVIOUS / refname).read_text())
        preds = json.loads(
            (bridge.PREVIOUS / f"results/{split}/predictions.json").read_text()
        )
        for r in preds:
            if r["stage"] != "policy":
                continue
            b = next(b for b in data["bundles"] if b["id"] == r["id"])
            w = witness(r["parsed"], b["reference_policy"], list(b["evidence"]))
            out.append(
                dict(
                    split=split, bundle=b["id"], model=r["model"], scope=b["scope"], **w
                )
            )
    return out


if __name__ == "__main__":
    data = previous_audit()
    (bridge.ROOT / "results/counterexamples.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        {
            s: sum(r["status"] == s for r in data)
            for s in sorted({r["status"] for r in data})
        }
    )
