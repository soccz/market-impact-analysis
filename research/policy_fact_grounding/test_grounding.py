import unittest
from grounding import money, spans, audit, correct


class GroundingTests(unittest.TestCase):
    def test_equivalent_formats(self):
        for s in [
            "2400만 240원",
            "24,000,240원",
            "0.2400024억원",
            "2천4백만2백4십원",
            "24000240원",
        ]:
            self.assertEqual(money(s), 24000240)

    def test_mixed_large_units(self):
        self.assertEqual(money("1억 1천만원"), 110000000)
        self.assertEqual(money("0.8억원"), 80000000)
        self.assertEqual(money("천원"), 1000)

    def test_invalid_formats(self):
        for s in [
            "24,00원",
            "약 2만원",
            "2만~3만원",
            "2달러",
            "1.5원",
            "2억3억원",
            "2천3천원",
            "-5원",
            "이천원",
            "2..4만원",
        ]:
            with self.assertRaises(ValueError, msg=s):
                money(s)

    def test_subject_and_units(self):
        es = spans(
            "부모의 보증금은 9천만원, 내 보증금은 2400만 240원, 월세는 70만원이다."
        )
        self.assertEqual(
            [e["owner"] for e in es], ["external", "applicant", "applicant"]
        )
        self.assertEqual([e["value"] for e in es], [90000000, 24000240, 700000])

    def test_correction_is_claim_grounded(self):
        p = {
            "status": "ready",
            "assertion": True,
            "values": [{"field": "deposit_krw", "value": 24002400}],
        }
        text = "내 보증금은 2400만 240원이다."
        a = audit(text, p)
        q = correct(p, a)
        self.assertEqual(q["values"][0]["value"], 24000240)
        self.assertEqual(p["values"][0]["value"], 24002400)
        for e in a["evidence"]:
            self.assertEqual(text[e["start"] : e["end"]], e["text"])

    def test_no_external_invention(self):
        p = {
            "status": "ready",
            "assertion": True,
            "values": [{"field": "deposit_krw", "value": 90000000}],
        }
        a = audit("부모의 보증금은 9천만원이다.", p)
        self.assertFalse(a["patches"])
        self.assertEqual(a["status"], "not_covered")

    def test_ambiguous_mentions_not_resolved(self):
        p = {"status": "ready", "assertion": True, "values": []}
        self.assertFalse(
            audit(
                "내 보증금은 2천만원에서 3천만원으로 바뀌었다. 내 보증금은 3천만원이다.",
                p,
            )["patches"]
        )

    def test_explicit_unknown(self):
        p = {
            "status": "ready",
            "assertion": False,
            "values": [{"field": "deposit_krw", "value": 100}],
        }
        a = audit("내 보증금은 밝히지 않았다.", p)
        self.assertIn("deposit_krw", a["patches"])
        self.assertIsNone(a["patches"]["deposit_krw"])

    def test_no_cross_sentence_owner(self):
        self.assertEqual(
            spans("내 상황은 다음과 같다. 보증금은 2만원이다.")[0]["owner"],
            "unresolved",
        )

    def test_zero_units(self):
        self.assertEqual(money("0만원"), 0)
        self.assertEqual(money("0억원"), 0)

    def test_change_expression_not_a_fact(self):
        p = {"status": "ready", "assertion": True, "values": []}
        self.assertFalse(
            audit("내 보증금은 2천만원에서 3천만원으로 바뀌었다.", p)["patches"]
        )

    def test_domestic_is_not_first_person(self):
        self.assertEqual(spans("국내 보증금은 2만원이다.")[0]["owner"], "unresolved")

    def test_duplicate_field_no_patch(self):
        p = {
            "status": "ready",
            "assertion": True,
            "values": [
                {"field": "deposit_krw", "value": 1},
                {"field": "deposit_krw", "value": 2},
            ],
        }
        self.assertEqual(
            audit("내 보증금은 3만원이다.", p)["status"], "profile_invalid"
        )


if __name__ == "__main__":
    unittest.main()
