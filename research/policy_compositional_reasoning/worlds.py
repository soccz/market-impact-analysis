"""Finite truth partitions for missing fields; preserve correlations across atoms."""

from datetime import date, timedelta
from itertools import product
from logic import FIELDS, evaluate

MAX_WORLDS = 4096


def candidates(field, clauses):
    values = []
    for clause in clauses:
        for a in clause:
            if a["field"] == field:
                values.extend(
                    a["value"] if a["op"] in ["in", "not_in"] else [a["value"]]
                )
    kind = FIELDS[field]["type"]
    if kind == "boolean":
        return [False, True]
    if kind == "number":
        v = sorted(set(values))
        return sorted(
            set(v + [v[0] - 1, v[-1] + 1] + [(a + b) / 2 for a, b in zip(v, v[1:])])
        )
    if kind == "date":
        days = {date.fromisoformat(v) for v in values}
        expanded = set(days)
        for d in days:
            if d > date.min:
                expanded.add(d - timedelta(days=1))
            if d < date.max:
                expanded.add(d + timedelta(days=1))
        return sorted(d.isoformat() for d in expanded)
    other = "__unlisted_value__"
    while other in values:
        other += "x"
    return sorted(set(values)) + [other]


def execute(clauses, values):
    initial, trace = evaluate(clauses, values)
    missing = sorted(
        {
            a["field"]
            for clause in clauses
            for a in clause
            if values.get(a["field"]) is None
        }
    )
    if initial is not None:
        return initial, trace, [], "kleene_decided", 0
    domains = [candidates(f, clauses) for f in missing]
    count = 1
    for domain in domains:
        count *= len(domain)
    if count > MAX_WORLDS:
        active = sorted(
            {
                a["field"]
                for clause, t in zip(clauses, trace)
                if t["value"] is None
                for a, v in zip(clause, t["atoms"])
                if v is None
            }
        )
        return None, trace, active, "budget_fallback", count
    rows = []
    for assignment in product(*domains):
        v, _ = evaluate(clauses, {**values, **dict(zip(missing, assignment))})
        assert v is not None
        rows.append((assignment, v))
    results = {v for _, v in rows}
    if len(results) == 1:
        return results.pop(), trace, [], "finite_partition", count
    influential = []
    for i, field in enumerate(missing):
        groups = {}
        for assignment, v in rows:
            key = assignment[:i] + assignment[i + 1 :]
            groups.setdefault(key, set()).add(v)
        if any(len(v) > 1 for v in groups.values()):
            influential.append(field)
    return None, trace, influential, "finite_partition", count
