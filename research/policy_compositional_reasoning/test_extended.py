import unittest, json
from pathlib import Path
from fractions import Fraction
from extended_logic import interpret, equivalent
from build_development import prof, atom
from infer_extended import request


class ExtendedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(
            (Path(__file__).parent / "capability_reference.json").read_text()
        )
        cls.rent = cls.data["bundles"][0]["reference_policy"]
        cls.education = cls.data["bundles"][1]["reference_policy"]

    def test_rational_boundary_and_grid(self):
        for deposit in [0, 20000000, 24000000, 24000240, 100000000, 110000000]:
            for rent in [0, 600000, 600001, 700000, 750000, 800000]:
                expected = (deposit <= 100000000 and rent <= 600000) or (
                    rent > 600000 and Fraction(rent) + Fraction(deposit, 240) <= 800000
                )
                got = interpret(
                    self.rent,
                    prof({"deposit_krw": deposit, "monthly_rent_krw": rent}),
                    ["D-RENT"],
                )["decision"]
                self.assertEqual(got, "supported" if expected else "contradicted")

    def test_unknown_amount_has_both_possible_answers(self):
        r = interpret(self.rent, prof({"monthly_rent_krw": 700000}), ["D-RENT"])
        self.assertEqual(r["decision"], "not_established")
        self.assertEqual(r["satisfiability"], ["sat", "sat"])

    def test_previous_school_is_not_current(self):
        q = prof(
            {
                "education_status": "enrolled",
                "school_kind": "general",
                "prior_graduation": True,
            }
        )
        self.assertEqual(
            interpret(self.education, q, ["G-EDUCATION", "G-EDUCATION-EXCEPTIONS"])[
                "decision"
            ],
            "contradicted",
        )

    def test_unknown_prior_graduation(self):
        q = prof({"education_status": "enrolled", "school_kind": "remote"})
        self.assertEqual(
            interpret(self.education, q, ["G-EDUCATION", "G-EDUCATION-EXCEPTIONS"])[
                "decision"
            ],
            "not_established",
        )

    def test_undefined_derived_is_invalid(self):
        p = {
            "status": "ready",
            "derived": [],
            "clauses": [[atom("derived_0", "le", 1)]],
        }
        self.assertEqual(interpret(p, prof({}), ["E"])["decision"], "abstain")

    def test_zero_denominator_is_invalid(self):
        p = json.loads(json.dumps(self.rent))
        p["derived"][0]["terms"][1]["denominator"] = 0
        self.assertEqual(interpret(p, prof({}), ["D-RENT"])["decision"], "abstain")

    def test_rational_equivalence(self):
        p = json.loads(json.dumps(self.rent))
        p["derived"][0]["terms"][1].update(numerator=1, denominator=240)
        self.assertTrue(equivalent(p, self.rent, ["D-RENT"]))
        p["derived"][0]["terms"][1].update(numerator=0.0041667, denominator=1)
        self.assertFalse(equivalent(p, self.rent, ["D-RENT"]))

    def test_profile_cannot_inject_derived_value(self):
        q = prof({"derived_0": 1})
        self.assertEqual(interpret(self.rent, q, ["D-RENT"])["decision"], "abstain")

    def test_request_isolation(self):
        b = {
            "scope": "scope",
            "evidence": {"E": "SOURCE-TEXT"},
            "reference_policy": "GOLD-LEAK",
        }
        c = {"claim": "CLAIM-TEXT", "label": "GOLD-LEAK"}
        for stage in ["policy", "profile", "direct"]:
            r = request(b, c, "qwen", stage)
            t = str(r["messages"])
            self.assertNotIn("GOLD-LEAK", t)
            if stage == "policy":
                self.assertNotIn("CLAIM-TEXT", t)
            if stage == "profile":
                self.assertNotIn("SOURCE-TEXT", t)
                self.assertNotIn("derived_0", str(r["format"]))


if __name__ == "__main__":
    unittest.main()
