"""Bounded nested policy interpretation and source-only document retrieval.

No gold labels, applicant profiles, or reference programs enter compilation.
This is an experimental reader, not a whole-program eligibility service.
"""

import itertools
import json
import math
import re

OPS = {"eq", "ne", "ge", "gt", "le", "lt"}


def validate(program, fields, evidence):
    if not isinstance(program, dict) or program.get("status") != "ready":
        raise ValueError("insufficient_program")
    nodes = program.get("nodes", [])
    if not 1 <= len(nodes) <= 80:
        raise ValueError("invalid_node_count")
    by_id = {n["id"]: n for n in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("duplicate_node")
    visiting, visited = set(), set()

    def visit(key):
        if key in visiting or key not in by_id:
            raise ValueError("cycle_or_missing_reference")
        if key in visited:
            return
        visiting.add(key)
        n = by_id[key]
        if not n.get("evidence") or not set(n["evidence"]) <= set(evidence):
            raise ValueError("unavailable_evidence")
        op = n["op"]
        if op == "atom":
            field, value = n["field"], n["value"]
            if field not in fields or n["cmp"] not in OPS or n["children"]:
                raise ValueError("invalid_atom")
            if fields[field]["type"] == "boolean":
                if type(value) is not bool or n["cmp"] not in {"eq", "ne"}:
                    raise ValueError("invalid_boolean")
            elif type(value) not in {int, float} or not math.isfinite(value):
                raise ValueError("invalid_number")
        elif op in {"all", "any", "not", "ref"}:
            if not n["children"] or (op in {"not", "ref"} and len(n["children"]) != 1):
                raise ValueError("invalid_children")
            for child in n["children"]:
                visit(child)
        else:
            raise ValueError("invalid_operator")
        visiting.remove(key)
        visited.add(key)

    visit(program["root"])
    if visited != set(by_id):
        raise ValueError("unreachable_node")
    return by_id


def complete(program, facts, nodes):
    def evaluate(key):
        n = nodes[key]
        if n["op"] == "atom":
            x, y = facts[n["field"]], n["value"]
            return {
                "eq": lambda: x == y,
                "ne": lambda: x != y,
                "ge": lambda: x >= y,
                "gt": lambda: x > y,
                "le": lambda: x <= y,
                "lt": lambda: x < y,
            }[n["cmp"]]()
        children = [evaluate(c) for c in n["children"]]
        return {
            "all": lambda: all(children),
            "any": lambda: any(children),
            "not": lambda: not children[0],
            "ref": lambda: children[0],
        }[n["op"]]()

    return evaluate(program["root"])


def decide(program, facts, fields, evidence):
    try:
        nodes = validate(program, fields, evidence)
        used = {n["field"] for n in nodes.values() if n["op"] == "atom"}
        values = {}
        for field in sorted(used):
            value = facts.get(field)
            spec = fields[field]
            if value is not None:
                valid = (
                    type(value) is bool
                    if spec["type"] == "boolean"
                    else type(value) in {int, float}
                    and math.isfinite(value)
                    and value >= 0
                )
                if not valid:
                    raise ValueError("invalid_fact_type")
                values[field] = [value]
            elif spec["type"] == "boolean":
                values[field] = [False, True]
            else:
                bounds = sorted(
                    {0.0}
                    | {
                        float(n["value"])
                        for n in nodes.values()
                        if n["op"] == "atom" and n["field"] == field
                    }
                )
                points = bounds + [(a + b) / 2 for a, b in zip(bounds, bounds[1:])]
                points += [bounds[-1] + 1]
                values[field] = sorted({x for x in points if x >= 0})
        if math.prod(map(len, values.values())) > 100000:
            raise ValueError("completion_budget")
        outcomes = set()
        for row in itertools.product(*values.values()):
            outcomes.add(complete(program, dict(zip(values, row)), nodes))
            if len(outcomes) == 2:
                return {"decision": "undetermined", "reason": "missing_relevant_facts"}
        return {
            "decision": "eligible" if True in outcomes else "ineligible",
            "reason": "all_completions_agree",
        }
    except (ValueError, KeyError, TypeError, RecursionError) as exc:
        return {"decision": "abstain", "reason": str(exc)}


def retrieve(document, k=4):
    """Character-bigram overlap; then explicit section/annex references, no gold."""

    def grams(text):
        text = re.sub(r"\s+", "", text)
        return {text[i : i + 2] for i in range(len(text) - 1)}

    query = grams(document["scope"])
    evidence = document["evidence"]
    scores = {
        key: len(query & grams(text)) / math.sqrt(max(1, len(grams(text))))
        for key, text in evidence.items()
    }
    selected = set(sorted(scores, key=lambda key: (-scores[key], key))[:k])
    expanded = set(selected)
    # Titles must occur at paragraph start; ordinary in-text mentions are not titles.
    titles = {}
    for key, text in evidence.items():
        match = re.match(r"\s*(제\d+조|별표\s*\d+)", text)
        if match:
            titles[re.sub(r"\s", "", match[1])] = key
    for _ in range(3):
        for key in list(expanded):
            for ref in re.findall(r"제\d+조|별표\s*\d+", evidence[key]):
                target = titles.get(re.sub(r"\s", "", ref))
                if target:
                    expanded.add(target)
    return {
        "ranked": list(sorted(selected)),
        "expanded": list(sorted(expanded)),
        "scores": scores,
    }


def literal_rules(document):
    """Transparent weak baseline: explicit labeled bounds + flat conjunction.

    Its guarded variant rejects exception/disjunction/reference language. It is
    not a competitive Korean parser, and failure is not proof of LLM novelty.
    """
    nodes = []
    comparisons = {"이상": "ge", "초과": "gt", "이하": "le", "미만": "lt"}
    text = "\n".join(document["evidence"].values())
    for key, line in document["evidence"].items():
        for field, spec in document["fields"].items():
            if spec["type"] == "boolean":
                continue
            for match in re.finditer(
                re.escape(spec["label"])
                + r"\s*[:：]?\s*(\d+(?:\.\d+)?)\s*(?:세|개월|만원|억원|%|원)?\s*(이상|초과|이하|미만)",
                line,
            ):
                nodes.append(
                    dict(
                        id=f"n{len(nodes)}",
                        op="atom",
                        field=field,
                        cmp=comparisons[match[2]],
                        value=float(match[1]),
                        children=[],
                        evidence=[key],
                    )
                )
    if not nodes:
        return {"status": "insufficient", "root": "", "nodes": []}, True
    children = [n["id"] for n in nodes]
    nodes.append(
        dict(
            id="root",
            op="all",
            field="",
            cmp="eq",
            value=0,
            children=children,
            evidence=list(document["evidence"]),
        )
    )
    guarded = bool(re.search(r"다만|제외|예외|또는|거나|별표|제\d+조|아니|않", text))
    return {"status": "ready", "root": "root", "nodes": nodes}, guarded


def public_document(document):
    return {k: v for k, v in document.items() if k != "evidence"}
