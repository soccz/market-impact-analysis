"""Behavioral checks for the distinction between change, error and stale cache."""

import unittest
from evaluate import score


class MetricTests(unittest.TestCase):
    def fixture(self, old="contradicted", fresh="supported", target="supported"):
        b = {"id": "b", "before": {"O1": "조건 유지"}, "after": {"N1": "조건 유지"}}
        c = {
            "id": "c",
            "bundle": "b",
            "claim": "조건 충족",
            "reference": {
                "before": "supported",
                "after": target,
                "route": "keep",
                "before_evidence": ["O1"],
                "after_evidence": ["N1"],
            },
        }
        r = {
            "map": {},
            "before": {"c": {"parsed": {"decision": old, "evidence": ["O1"]}}},
            "after": {"c": {"parsed": {"decision": fresh, "evidence": ["N1"]}}},
            "direct_gate": {"c": {"parsed": {"route": "keep", "evidence": ["N1"]}}},
            "map_gate": {"c": {"parsed": {"route": "keep", "evidence": ["N1"]}}},
        }
        return score({"bundles": [b], "cases": [c]}, r)

    def test_stable_reference_flip_can_fix_old_error(self):
        x = self.fixture()["refresh_all"]["metrics"]
        self.assertEqual(
            (x["false_flip"], x["corrected_old_error"], x["harmful_refresh"]), (1, 1, 0)
        )

    def test_correct_keep_routing_can_preserve_old_error(self):
        x = self.fixture()["change_map_gate"]["metrics"]
        self.assertEqual(
            (x["missed_recheck"], x["retained_old_error"], x["correct"]), (0, 1, 0)
        )

    def test_needless_refresh_can_introduce_error(self):
        x = self.fixture(old="supported", fresh="contradicted")["refresh_all"][
            "metrics"
        ]
        self.assertEqual((x["harmful_refresh"], x["correct"]), (1, 0))


if __name__ == "__main__":
    unittest.main()
