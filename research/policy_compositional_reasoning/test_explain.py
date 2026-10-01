import unittest
from build_development import atom, prof, rule
from explain import explain
from logic import evaluate


class ExplainTests(unittest.TestCase):
    def test_two_alternative_minimal_fact_sets(self):
        p = rule([[atom("pledge", "eq", True)], [atom("residence_months", "ge", 18)]])
        x = explain(
            p,
            prof(
                {"pledge": True, "residence_months": 24, "parent_residence_months": 90}
            ),
            ["E"],
        )
        self.assertEqual(
            x["minimal_fact_sets"], [{"pledge": True}, {"residence_months": 24}]
        )

    def test_unknown_witnesses_are_concrete_opposite_results(self):
        p = rule(
            [
                [
                    atom("income_pct", "le", 150),
                    atom("residence_city", "eq", "광주광역시"),
                ]
            ]
        )
        known = {"income_pct": 100}
        x = explain(p, prof(known), ["E"])
        self.assertEqual(x["candidate_fields"], ["residence_city"])
        for name, completion in x["witnesses"].items():
            self.assertNotIn("income_pct", completion)
            self.assertEqual(
                evaluate(p["clauses"], known | completion)[0], name == "passes"
            )

    def test_unsatisfiable_program_needs_no_observed_fact(self):
        p = rule([[atom("income_pct", "lt", 100), atom("income_pct", "ge", 100)]])
        self.assertEqual(
            explain(p, prof({"income_pct": 150}), ["E"])["minimal_fact_sets"], [{}]
        )


if __name__ == "__main__":
    unittest.main()
