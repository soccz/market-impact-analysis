"""Behavioral contracts: boundaries, exceptions, missingness and attribution."""

import unittest
from engine import compare, evaluate, housing_2023


def profile(d=30_000_000, r=450_000, b=None):
    return {"deposit_won": d, "rent_won": r, "benefits": [] if b is None else b}


class Contracts(unittest.TestCase):
    def test_official_old_examples(self):
        self.assertEqual(evaluate(profile(), "2022")["state"], "pass")
        self.assertEqual(evaluate(profile(r=480_000), "2022")["state"], "fail")

    def test_exception_survives(self):
        p = profile()
        self.assertEqual(compare(p)["change"], "retained")
        self.assertEqual(compare(p, ablation="drop_exception")["change"], "included")

    def test_two_directions_and_minimal_causes(self):
        a = compare(profile(d=100_000_000, r=700_000))
        b = compare(profile(b=["livelihood"]))
        self.assertEqual(
            (a["change"], a["minimal_rule_changes"]), ("included", [["housing"]])
        )
        self.assertEqual(
            (b["change"], b["minimal_rule_changes"]), ("excluded", [["welfare"]])
        )

    def test_old_deposit_boundary_applies_outside_exception(self):
        self.assertEqual(
            evaluate(profile(d=50_000_000, r=400_000), "2022")["state"], "pass"
        )
        self.assertEqual(evaluate(profile(d=50_000_001, r=0), "2022")["state"], "fail")

    def test_new_inclusive_limit_and_small_transaction_branch(self):
        self.assertEqual(housing_2023(100_000_000, 1_000_000), ("pass", 200_000_000))
        self.assertEqual(housing_2023(100_000_000, 1_000_001)[0], "fail")
        self.assertEqual(housing_2023(0, 499_999), ("pass", 34_999_930))
        self.assertEqual(housing_2023(0, 500_000), ("pass", 50_000_000))

    def test_candidate_rounding_readings(self):
        result = evaluate(profile(r=460_000), "2022")
        self.assertEqual(result["state"], "unknown")
        self.assertEqual(
            result["components"]["housing"]["readings"],
            {"thousand_floor": "fail", "published_table": "pass"},
        )
        self.assertEqual(evaluate(profile(r=457_000), "2022")["state"], "pass")
        self.assertEqual(evaluate(profile(r=457_001), "2022")["state"], "unknown")

    def test_missing_fields_are_not_zero(self):
        self.assertEqual(evaluate(profile(d=None), "2022")["state"], "unknown")
        p = profile()
        p["benefits"] = None
        self.assertEqual(evaluate(p, "2023")["next_checks"], ["missing_benefits"])

    def test_decisive_failure_and_multiple_benefits(self):
        result = evaluate(profile(d=None, b=["education", "housing"]), "2023")
        self.assertEqual((result["state"], result["next_checks"]), ("fail", []))
        self.assertEqual(evaluate(profile(b=["education"]), "2023")["state"], "pass")

    def test_evidence_removal_is_not_repeal(self):
        a = evaluate(profile(), "2022", hidden=["2022-housing"])
        self.assertEqual(a["state"], "unknown")
        self.assertEqual(a["next_checks"], ["evidence_hidden"])
        self.assertEqual(
            evaluate(profile(b=["housing"]), "2022", hidden=["2022-housing"])["state"],
            "fail",
        )

    def test_invalid_values(self):
        for p in [
            profile(d=-1),
            profile(r=True),
            profile(d=1.1),
            profile(b=["other"]),
            profile(b=["housing", "housing"]),
        ]:
            with self.assertRaises(ValueError):
                evaluate(p, "2022")


if __name__ == "__main__":
    unittest.main()
