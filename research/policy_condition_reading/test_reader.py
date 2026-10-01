"""Engineering invariants, separate from empirical parser failure cases."""

import json
import unittest

from reader import chunks, compact, execute, parse, predict
from posthoc import restrict


class ReaderTests(unittest.TestCase):
    def test_missing_input_stays_unknown(self):
        self.assertEqual(
            execute("income", {"all": 150}, {"income_percent": None}), "unknown"
        )
        self.assertEqual(execute("welfare", ["housing"], {"benefits": None}), "unknown")

    def test_absent_rule_never_becomes_unrestricted(self):
        for field in ("age", "birth", "income", "welfare"):
            self.assertEqual(execute(field, None, {}), "unknown")

    def test_conditional_threshold_requires_scope(self):
        rule = {"first": 150, "returning": 120}
        self.assertEqual(execute("income", rule, {"income_percent": 130}), "unknown")
        self.assertEqual(
            execute("income", rule, {"income_percent": 130, "recipient": "first"}),
            "pass",
        )
        self.assertEqual(
            execute("income", rule, {"income_percent": 130, "recipient": "returning"}),
            "fail",
        )

    def test_date_inclusive_boundaries(self):
        rule = ["2001-01-01", "2004-12-31"]
        for day, expected in [
            ("2000-12-31", "fail"),
            (rule[0], "pass"),
            (rule[1], "pass"),
            ("2005-01-01", "fail"),
        ]:
            self.assertEqual(execute("birth", rule, {"birth": day}), expected)

    def test_evidence_offsets_survive_section_mask(self):
        text = "1.개요다른문장2.신청자격나이19~39세3.신청방법접수"
        masked, _ = restrict([text])
        self.assertEqual(len(masked[0]), len(text))
        p = predict(masked, "document_first")["age"]
        e = p["evidence"][0]
        self.assertEqual(text[e["start"] : e["end"]], "19~39세")

    def test_invalid_date_does_not_crash(self):
        text = "2001.2.30.~2004.12.31.출생자"
        self.assertIsNone(parse(chunks([text])[0], "birth"))

    def test_empty_text_abstains(self):
        self.assertTrue(
            all(p["value"] is None for p in predict([""], "document_first").values())
        )

    def test_normalization_is_idempotent(self):
        text = "신청 자격\n19 ~ 39세"
        self.assertEqual(compact(compact(text)), compact(text))


if __name__ == "__main__":
    unittest.main()
