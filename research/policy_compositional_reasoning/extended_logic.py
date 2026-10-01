"""Typed DNF + bounded rational linear definitions, executed with Z3.

A solver proves consequences of the extracted program, never its faithfulness to text.
No model-produced code, eval(), network or filesystem operations are executed.
"""

from datetime import date
from fractions import Fraction
import json
import math
from pathlib import Path
import z3
from logic import FIELDS as BASE_FIELDS, OPS

ROOT = Path(__file__).parent
FIELDS = BASE_FIELDS | json.loads((ROOT / "extended_fields.json").read_text())
DERIVED = {
    f"derived_{i}": {
        "type": "number",
        "description": "근거에서 추출한 수치 선형합. 신청자 입력 필드가 아니다.",
    }
    for i in range(3)
}
ALL_FIELDS = FIELDS | DERIVED


def scalar_ok(field, v):
    kind = ALL_FIELDS[field]["type"]
    if kind == "number":
        return type(v) in [int, float] and abs(v) <= 1e15 and math.isfinite(v)
    if kind == "boolean":
        return type(v) is bool
    if kind == "date":
        if not isinstance(v, str):
            return False
        try:
            return date.fromisoformat(v).isoformat() == v
        except ValueError:
            return False
    return isinstance(v, str) and 0 < len(v.strip()) <= 120


def number(v):
    return z3.RealVal(str(Fraction(str(v))))


def variable(f):
    kind = ALL_FIELDS[f]["type"]
    return (
        z3.Bool(f)
        if kind == "boolean"
        else (
            z3.String(f)
            if kind == "string"
            else z3.Int(f) if kind == "date" else z3.Real(f)
        )
    )


def literal(f, v):
    kind = ALL_FIELDS[f]["type"]
    return (
        z3.BoolVal(v)
        if kind == "boolean"
        else (
            z3.StringVal(v)
            if kind == "string"
            else (
                z3.IntVal(date.fromisoformat(v).toordinal())
                if kind == "date"
                else number(v)
            )
        )
    )


def compare(x, op, vals):
    if op == "in":
        return z3.Or([x == v for v in vals])
    if op == "not_in":
        return z3.Not(z3.Or([x == v for v in vals]))
    if op == "eq":
        return x == vals
    if op == "ne":
        return x != vals
    if op == "lt":
        return x < vals
    if op == "le":
        return x <= vals
    if op == "gt":
        return x > vals
    return x >= vals


def compile_program(p, allowed):
    if not isinstance(p, dict) or p.get("status") not in ["ready", "insufficient"]:
        return None, [], "invalid_policy_schema"
    if p["status"] == "insufficient":
        return None, [], "policy_insufficient"
    definitions = p.get("derived")
    clauses = p.get("clauses")
    if (
        not isinstance(definitions, list)
        or len(definitions) > 3
        or not isinstance(clauses, list)
        or not 1 <= len(clauses) <= 12
    ):
        return None, [], "invalid_policy_schema"
    derived = {}
    for d in definitions:
        if (
            not isinstance(d, dict)
            or set(d) != {"field", "constant", "terms", "evidence"}
            or d["field"] not in DERIVED
            or d["field"] in derived
        ):
            return None, [], "invalid_definition"
        if (
            not scalar_ok(d["field"], d["constant"])
            or not isinstance(d["terms"], list)
            or not 1 <= len(d["terms"]) <= 8
        ):
            return None, [], "invalid_definition"
        if (
            not isinstance(d["evidence"], list)
            or not d["evidence"]
            or any(e not in allowed for e in d["evidence"])
        ):
            return None, [], "invalid_evidence"
        terms = []
        for t in d["terms"]:
            if (
                not isinstance(t, dict)
                or set(t) != {"field", "numerator", "denominator"}
                or t["field"] not in FIELDS
                or FIELDS[t["field"]]["type"] != "number"
            ):
                return None, [], "invalid_linear_term"
            if (
                not scalar_ok(t["field"], t["numerator"])
                or not scalar_ok(t["field"], t["denominator"])
                or t["denominator"] == 0
            ):
                return None, [], "invalid_coefficient"
            terms.append(
                variable(t["field"]) * number(t["numerator"]) / number(t["denominator"])
            )
        derived[d["field"]] = number(d["constant"]) + z3.Sum(terms)
    groups = []
    atoms = []
    for c in clauses:
        if not isinstance(c, list) or not 1 <= len(c) <= 16:
            return None, [], "invalid_clause"
        group = []
        for a in c:
            if not isinstance(a, dict) or set(a) != {
                "field",
                "op",
                "value",
                "evidence",
            }:
                return None, [], "invalid_atom"
            f, op, v = a["field"], a["op"], a["value"]
            if f not in ALL_FIELDS or op not in OPS:
                return None, [], "invalid_operator"
            if f in DERIVED and f not in derived:
                return None, [], "undefined_derived"
            if (
                not isinstance(a["evidence"], list)
                or not a["evidence"]
                or any(e not in allowed for e in a["evidence"])
            ):
                return None, [], "invalid_evidence"
            if op in ["in", "not_in"]:
                if (
                    not isinstance(v, list)
                    or not v
                    or not all(scalar_ok(f, x) for x in v)
                ):
                    return None, [], "invalid_value"
                target = [literal(f, x) for x in v]
            else:
                if not scalar_ok(f, v):
                    return None, [], "invalid_value"
                target = literal(f, v)
            if op in ["lt", "le", "gt", "ge"] and ALL_FIELDS[f]["type"] not in [
                "number",
                "date",
            ]:
                return None, [], "invalid_comparison"
            expression = compare(
                derived[f] if f in derived else variable(f), op, target
            )
            group.append(expression)
        atoms.append(group)
        groups.append(z3.And(group))
    return z3.Or(groups), atoms, "valid"


def parse_profile(q):
    if (
        not isinstance(q, dict)
        or q.get("status") not in ["ready", "unsupported"]
        or type(q.get("assertion")) is not bool
    ):
        return None, None, "invalid_profile_schema"
    if q["status"] == "unsupported":
        return None, None, "query_unsupported"
    if not isinstance(q.get("values"), list):
        return None, None, "invalid_profile_schema"
    vals = {}
    for x in q["values"]:
        if (
            not isinstance(x, dict)
            or set(x) != {"field", "value"}
            or x["field"] not in FIELDS
            or x["field"] in vals
        ):
            return None, None, "invalid_profile_field"
        if x["value"] is not None and not scalar_ok(x["field"], x["value"]):
            return None, None, "invalid_profile_value"
        vals[x["field"]] = x["value"]
    return vals, q["assertion"], "valid"


def solve(formula, values):
    s = z3.Solver()
    s.set(timeout=3000)
    s.add([variable(f) == literal(f, v) for f, v in values.items() if v is not None])
    outcomes = []
    for wanted in [formula, z3.Not(formula)]:
        s.push()
        s.add(wanted)
        outcomes.append(str(s.check()))
        s.pop()
    if "unknown" in outcomes:
        return None, outcomes
    if outcomes == ["sat", "unsat"]:
        return True, outcomes
    if outcomes == ["unsat", "sat"]:
        return False, outcomes
    if outcomes == ["sat", "sat"]:
        return None, outcomes
    raise AssertionError("Inconsistent typed profile constraints")


def interpret(p, q, allowed):
    formula, atoms, ps = compile_program(p, allowed)
    values, pol, qs = parse_profile(q)
    if formula is None or values is None:
        semantic = ps in ["valid", "policy_insufficient"] and qs in [
            "valid",
            "query_unsupported",
        ]
        return dict(
            decision="not_established" if semantic else "abstain",
            policy_status=ps,
            profile_status=qs,
            trace=[],
            satisfiability=[],
        )
    value, sat = solve(formula, values)
    label = (
        "not_established"
        if value is None
        else "supported" if value == pol else "contradicted"
    )
    trace = [{"atoms": [solve(a, values)[0] for a in group]} for group in atoms]
    return dict(
        decision=label,
        policy_status=ps,
        profile_status=qs,
        trace=trace,
        satisfiability=sat,
    )


def equivalent(p, r, allowed):
    a, _, ps = compile_program(p, allowed)
    b, _, rs = compile_program(r, allowed)
    if a is None or b is None:
        return ps == rs == "policy_insufficient"
    s = z3.Solver()
    s.set(timeout=3000)
    s.add(z3.Xor(a, b))
    v = s.check()
    return True if v == z3.unsat else False if v == z3.sat else None
