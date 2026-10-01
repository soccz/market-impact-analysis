"""Typed DNF execution with explicit unknowns and no closed-world assumptions."""

from datetime import date
import json
from pathlib import Path

FIELDS = json.loads((Path(__file__).parent / "fields.json").read_text())
OPS = ["eq", "ne", "lt", "le", "gt", "ge", "in", "not_in"]


def scalar_ok(field, value):
    kind = FIELDS[field]["type"]
    if kind == "number":
        return type(value) in [int, float] and float("-inf") < value < float("inf")
    if kind == "boolean":
        return type(value) is bool
    if kind == "date":
        if not isinstance(value, str):
            return False
        try:
            return date.fromisoformat(value).isoformat() == value
        except ValueError:
            return False
    return isinstance(value, str) and bool(value.strip())


def policy(value, allowed):
    if not isinstance(value, dict) or value.get("status") not in [
        "ready",
        "insufficient",
    ]:
        return None, "invalid_policy_schema"
    if value["status"] == "insufficient":
        return None, "policy_insufficient"
    clauses = value.get("clauses")
    if not isinstance(clauses, list) or not clauses:
        return None, "empty_policy"
    if len(clauses) > 12:
        return None, "excess_branches"
    for clause in clauses:
        if not isinstance(clause, list) or not clause or len(clause) > 16:
            return None, "invalid_clause"
        for atom in clause:
            if not isinstance(atom, dict) or set(atom) != {
                "field",
                "op",
                "value",
                "evidence",
            }:
                return None, "invalid_atom"
            field, op, v = atom["field"], atom["op"], atom["value"]
            if field not in FIELDS or op not in OPS:
                return None, "invalid_operator"
            if op in ["in", "not_in"]:
                if (
                    not isinstance(v, list)
                    or not v
                    or not all(scalar_ok(field, x) for x in v)
                ):
                    return None, "invalid_value"
            elif not scalar_ok(field, v):
                return None, "invalid_value"
            if op in ["lt", "le", "gt", "ge"] and FIELDS[field]["type"] not in [
                "number",
                "date",
            ]:
                return None, "invalid_comparison"
            if (
                not isinstance(atom["evidence"], list)
                or not atom["evidence"]
                or any(i not in allowed for i in atom["evidence"])
            ):
                return None, "invalid_evidence"
    return clauses, "valid"


def profile(value):
    if (
        not isinstance(value, dict)
        or value.get("status") not in ["ready", "unsupported"]
        or type(value.get("assertion")) is not bool
    ):
        return None, None, "invalid_profile_schema"
    if value["status"] == "unsupported":
        return None, None, "query_unsupported"
    rows = value.get("values")
    if not isinstance(rows, list):
        return None, None, "invalid_profile_schema"
    result = {}
    for r in rows:
        if (
            not isinstance(r, dict)
            or set(r) != {"field", "value"}
            or r["field"] not in FIELDS
        ):
            return None, None, "invalid_profile_value"
        if r["field"] in result:
            return None, None, "duplicate_profile_field"
        if r["value"] is not None and not scalar_ok(r["field"], r["value"]):
            return None, None, "invalid_profile_type"
        result[r["field"]] = r["value"]
    return result, value["assertion"], "valid"


def atom_value(atom, values):
    v = values.get(atom["field"])
    if v is None:
        return None
    target = atom["value"]
    op = atom["op"]
    if op == "eq":
        return v == target
    if op == "ne":
        return v != target
    if op == "lt":
        return v < target
    if op == "le":
        return v <= target
    if op == "gt":
        return v > target
    if op == "ge":
        return v >= target
    if op == "in":
        return v in target
    return v not in target


def evaluate(clauses, values):
    trace = []
    for clause in clauses:
        atoms = [atom_value(a, values) for a in clause]
        result = False if False in atoms else None if None in atoms else True
        trace.append({"atoms": atoms, "value": result})
    value = (
        True
        if any(r["value"] is True for r in trace)
        else None if any(r["value"] is None for r in trace) else False
    )
    return value, trace


def interpret(parsed_policy, parsed_profile, allowed, semantics="finite"):
    from worlds import execute

    clauses, pr = policy(parsed_policy, allowed)
    values, assertion, qr = profile(parsed_profile)
    if clauses is None or values is None:
        semantic_unknown = pr in ["valid", "policy_insufficient"] and qr in [
            "valid",
            "query_unsupported",
        ]
        return {
            "decision": "not_established" if semantic_unknown else "abstain",
            "policy_status": pr,
            "profile_status": qr,
            "trace": [],
            "missing_fields": [],
        }
    if semantics == "finite":
        result, trace, influential, execution, count = execute(clauses, values)
    else:
        result, trace = evaluate(clauses, values)
        influential, execution, count = None, "kleene", 0
    label = (
        "not_established"
        if result is None
        else "supported" if result == assertion else "contradicted"
    )
    # Unknown fields are listed only when their unresolved branch can affect the answer.
    missing = (
        sorted(
            {
                a["field"]
                for clause, row in zip(clauses, trace)
                if row["value"] is None
                for a, v in zip(clause, row["atoms"])
                if v is None
            }
        )
        if result is None
        else []
    )
    return {
        "decision": label,
        "policy_status": pr,
        "profile_status": qr,
        "trace": trace,
        "missing_fields": influential if influential is not None else missing,
        "execution": execution,
        "partition_worlds": count,
    }
