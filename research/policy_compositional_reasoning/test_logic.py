import unittest
from logic import policy, profile, interpret
from worlds import execute
from infer import request


def a(field, op, v):
    return {"field": field, "op": op, "value": v, "evidence": ["E"]}


class LogicTests(unittest.TestCase):
    def test_false_and_missing_is_false(self):
        value, _, missing, _, _ = execute(
            [[a("age_years", "ge", 19), a("income_pct", "le", 150)]], {"age_years": 18}
        )
        self.assertFalse(value)
        self.assertEqual(missing, [])

    def test_true_or_missing_is_true(self):
        self.assertTrue(
            execute(
                [[a("pledge", "eq", True)], [a("residence_months", "ge", 12)]],
                {"pledge": True},
            )[0]
        )

    def test_shared_unknown_tautology(self):
        self.assertTrue(
            execute([[a("pledge", "eq", True)], [a("pledge", "eq", False)]], {})[0]
        )

    def test_shared_unknown_contradiction(self):
        self.assertFalse(
            execute([[a("income_pct", "gt", 150), a("income_pct", "le", 150)]], {})[0]
        )

    def test_equal_boundary_and_negated_claim(self):
        p = {
            "status": "ready",
            "clauses": [[a("age_years", "ge", 19), a("age_years", "le", 39)]],
        }
        q = {
            "status": "ready",
            "values": [{"field": "age_years", "value": 39}],
            "assertion": False,
        }
        self.assertEqual(interpret(p, q, ["E"])["decision"], "contradicted")

    def test_subject_binding(self):
        v = execute(
            [[a("residence_months", "ge", 12)]], {"parent_residence_months": 24}
        )
        self.assertIsNone(v[0])
        self.assertEqual(v[2], ["residence_months"])

    def test_influential_fields_only(self):
        v = execute(
            [
                [a("pledge", "eq", True), a("age_years", "gt", 0)],
                [a("pledge", "eq", False), a("age_years", "gt", 0)],
            ],
            {},
        )
        self.assertIsNone(v[0])
        self.assertEqual(v[2], ["age_years"])

    def test_reject_boolean_as_amount(self):
        self.assertIsNone(
            policy(
                {"status": "ready", "clauses": [[a("income_krw", "le", True)]]}, ["E"]
            )[0]
        )

    def test_reject_foreign_id_and_duplicate_profile(self):
        self.assertIsNone(
            policy({"status": "ready", "clauses": [[a("pledge", "eq", True)]]}, ["X"])[
                0
            ]
        )
        q = {
            "status": "ready",
            "assertion": True,
            "values": [
                {"field": "age_years", "value": 20},
                {"field": "age_years", "value": 30},
            ],
        }
        self.assertEqual(profile(q)[2], "duplicate_profile_field")

    def test_profile_does_not_receive_source_and_policy_has_no_claim(self):
        b = {"scope": "age", "evidence": {"E": "TARGET-SOURCE"}, "reference": "LEAK"}
        c = {"claim": "TARGET-CLAIM", "reference": "LEAK"}
        for stage in ["policy", "profile", "direct"]:
            r = request(b, c, "qwen", stage)
            text = r["messages"][1]["content"]
            self.assertNotIn("LEAK", text)
            if stage == "policy":
                self.assertNotIn("TARGET-CLAIM", text)
            if stage == "profile":
                self.assertNotIn("TARGET-SOURCE", text)

    def test_missing_policy_is_unknown(self):
        r = interpret(
            {"status": "insufficient", "clauses": []},
            {"status": "ready", "assertion": True, "values": []},
            ["E"],
        )
        self.assertEqual(r["decision"], "not_established")

    def test_invalid_query_is_not_hidden_by_missing_policy(self):
        self.assertEqual(
            interpret({"status": "insufficient"}, None, ["E"])["decision"], "abstain"
        )

    def test_date_partition_closed_boundary(self):
        clauses = [
            [a("joined_on", "le", "2025-01-01")],
            [a("joined_on", "gt", "2025-01-01")],
        ]
        self.assertTrue(execute(clauses, {})[0])

    def test_partition_budget_is_conservative(self):
        from unittest.mock import patch

        clauses = [[a("income_pct", "gt", 100), a("age_years", "gt", 20)]]
        with patch("worlds.MAX_WORLDS", 1):
            result = execute(clauses, {})
        self.assertIsNone(result[0])
        self.assertEqual(result[3], "budget_fallback")


if __name__ == "__main__":
    unittest.main()
