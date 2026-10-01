"""Independent Python predicates check the provisional graphs' calculation.

These do not provide an independent human interpretation of source text.
"""

import itertools
import json
from pathlib import Path
import unittest

from method import complete, validate
from report import semantic_audit
from test_method import atom, group, program

ROOT = Path(__file__).parent


def oracle(key, f):
    if key.startswith("auth-exception"):
        protected = (
            (f["disabled"] or f["carer"])
            if key.endswith("any")
            else (f["disabled"] and f["carer"])
        )
        basic = f["age"] >= 19 and (f["residence"] >= 6 or f["student"])
        return basic and (f["income"] <= 2000 or protected)
    if key.startswith("auth-subject"):
        income = f["own_income"] if key.endswith("self") else f["spouse_income"]
        return (
            f["age"] >= 19
            and (f["student"] or f["worker"])
            and (income < 3000 or (f["disabled"] and not f["sanction"]))
        )
    if key == "work24":
        employer_exclusion = f["large"] and f["wage"] >= 300 and f["age"] < 45
        benefit_exclusion = f["livelihood"] and not (f["conditional"] or f["deferred"])
        return not employer_exclusion and not benefit_exclusion
    return (
        (f["korean_self"] or f["korean_spouse"] or f["korean_child"])
        and not f["dependent_other"]
        and not (f["professional_self"] or f["professional_spouse"])
    )


def grid_check():
    data = json.loads((ROOT / "reference.json").read_text())
    count = 0
    domains_by_field = dict(
        age=[18, 19, 20, 44, 45, 46],
        residence=[5, 6, 7],
        income=[1999, 2000, 2001],
        own_income=[2999, 3000, 3001],
        spouse_income=[2999, 3000, 3001],
        wage=[299, 300, 301],
    )
    for d in data["documents"]:
        domains = {
            field: (
                [False, True] if spec["type"] == "boolean" else domains_by_field[field]
            )
            for field, spec in d["fields"].items()
        }
        nodes = validate(d["reference_program"], d["fields"], d["evidence"])
        for row in itertools.product(*domains.values()):
            facts = dict(zip(domains, row))
            assert complete(d["reference_program"], facts, nodes) == oracle(
                d["id"], facts
            ), (d["id"], facts)
            count += 1
    return count


class ReferenceTests(unittest.TestCase):
    def test_independent_grid(self):
        self.assertGreater(grid_check(), 1000)

    def test_counterexample_detects_boundary_error(self):
        fields = dict(age=dict(type="number", label="age"))
        a = program([atom("root", "age", 19, "ge")])
        b = program([atom("root", "age", 19, "gt")])
        result = semantic_audit(a, b, fields, {"p1": ""})
        self.assertEqual(result["status"], "different")
        self.assertEqual(result["witness"]["facts"], {"age": 19})

    def test_equivalence_not_syntactic_identity(self):
        fields = dict(w=dict(type="boolean", label="w"))
        a = program([atom("root", "w", True)])
        b = program([atom("w", "w", False), group("root", "not", ["w"])])
        self.assertEqual(
            semantic_audit(a, b, fields, {"p1": ""})["status"], "equivalent"
        )


if __name__ == "__main__":
    unittest.main()
