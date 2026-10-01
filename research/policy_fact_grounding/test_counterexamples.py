import unittest
from counterexamples import witness


def rule(op, value):
    return {
        "status": "ready",
        "derived": [],
        "clauses": [
            [{"field": "age_years", "op": op, "value": value, "evidence": ["E"]}]
        ],
    }


class WitnessTests(unittest.TestCase):
    def test_boundary_counterexample(self):
        w = witness(rule("lt", 39), rule("le", 39), ["E"])
        self.assertEqual(w["status"], "witness")
        self.assertEqual(w["profile"]["values"], [{"field": "age_years", "value": 39}])

    def test_equivalent_within_integer_domain(self):
        self.assertEqual(
            witness(rule("lt", 40), rule("le", 39), ["E"])["status"],
            "no_witness_in_domain",
        )

    def test_invalid_not_proved(self):
        self.assertEqual(witness(None, rule("le", 39), ["E"])["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
