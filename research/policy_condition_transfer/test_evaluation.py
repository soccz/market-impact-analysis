"""Prevent abstention, numeric-only support and partial spans inflating scores."""

import unittest
from evaluate import covers, score


class EvaluationTests(unittest.TestCase):
    def test_partial_span_is_not_complete(self):
        required = [[dict(page=2, start=10, end=20)]]
        self.assertFalse(covers([dict(page=2, start=10, end=19)], required))
        self.assertFalse(covers([dict(page=1, start=0, end=100)], required))
        self.assertTrue(covers([dict(page=2, start=10, end=20)], required))

    def test_branches_are_required_but_alternative_locations_are_not(self):
        base = dict(page=1, start=0, end=5)
        alternative = dict(page=2, start=0, end=5)
        branch = dict(page=2, start=6, end=20)
        groups = [[base, alternative], [branch]]
        self.assertFalse(covers([base], groups))
        self.assertTrue(covers([alternative, branch], groups))

    def test_unmentioned_unrestricted_and_empty_exclusions_differ(self):
        refs = [
            dict(document="d", field=f, value=v, support_groups=[])
            for f, v in [("a", None), ("b", {"unrestricted": True}), ("c", [])]
        ]
        pred = {
            "d": {
                "m": {
                    f: dict(value=None, evidence=[], reason="no_parse")
                    for f in ["a", "b", "c"]
                }
            }
        }
        rows, metrics, _ = score(pred, refs, ["m"])
        self.assertEqual(
            [r["outcome"] for r in rows], ["absence_abstain", "abstain", "abstain"]
        )
        self.assertEqual(metrics["m"]["answerable"], 2)
        self.assertEqual(metrics["m"]["correct_joint"], 0)

    def test_numeric_match_cannot_hide_missing_scope(self):
        refs = [
            dict(
                document="d",
                field="age",
                value=[20, 23],
                support_groups=[
                    [dict(page=1, start=0, end=5)],
                    [dict(page=1, start=6, end=20)],
                ],
            )
        ]
        pred = {
            "d": {
                "m": {
                    "age": dict(
                        value=[20, 23],
                        evidence=[dict(page=1, start=0, end=5)],
                        reason="parsed",
                    )
                }
            }
        }
        rows, metrics, _ = score(pred, refs, ["m"])
        self.assertEqual(rows[0]["outcome"], "value_only")
        self.assertEqual(metrics["m"]["correct_joint"], 0)


if __name__ == "__main__":
    unittest.main()
