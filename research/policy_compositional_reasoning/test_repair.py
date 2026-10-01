import unittest
from repair import feedback, builder
from build_development import atom, rule


class RepairTests(unittest.TestCase):
    def test_missing_source_is_not_forced_to_ready(self):
        self.assertIsNone(feedback({"status": "insufficient", "clauses": []}, ["E"]))

    def test_schema_and_contradiction_are_selected(self):
        self.assertEqual(feedback(None, ["E"])["kind"], "schema_or_type")
        r = rule([[atom("age_years", "gt", 25), atom("age_years", "le", 25)]])
        self.assertFalse(feedback(r, ["E"])["condition_value"])

    def test_satisfiable_nonconstant_not_selected(self):
        self.assertIsNone(feedback(rule([[atom("age_years", "ge", 25)]]), ["E"]))

    def test_selection_does_not_read_reference_or_claim(self):
        records = [dict(model="qwen", stage="policy", id="b", parsed=None)]
        fn = builder(records, False)
        b = dict(
            id="b",
            scope="scope",
            evidence={"E": "source"},
            reference_policy="GOLD-SECRET",
        )
        r = fn(b, {"claim": "CLAIM-SECRET", "label": "GOLD-SECRET"}, "qwen", "policy")
        self.assertNotIn("GOLD-SECRET", str(r))
        self.assertNotIn("CLAIM-SECRET", str(r))


if __name__ == "__main__":
    unittest.main()
