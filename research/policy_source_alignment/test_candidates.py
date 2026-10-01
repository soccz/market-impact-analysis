import unittest
from candidates import extract, compile_selection, rule_selection
from bridge import interpret


def bundle(text, scope="선택 문장의 요건만."):
    return dict(scope=scope, evidence={"E": text})


def run(text, values, scope="선택 문장의 요건만."):
    b = bundle(text, scope)
    cat = extract(b)
    p = compile_selection(rule_selection(b, cat), cat)
    q = dict(
        status="ready",
        assertion=True,
        values=[dict(field=k, value=v) for k, v in values.items()],
    )
    return interpret(p, q, ["E"])["decision"]


class SourceTests(unittest.TestCase):
    def test_mixed_unit_bound(self):
        c = extract(bundle("보증금 9천만원 이하."))
        self.assertEqual(c["atoms"][0]["atom"]["value"], 90000000)

    def test_exact_source_span(self):
        b = bundle("내규: 보증금 1억 1원 이하, 월 임차료 60만원 미만.")
        c = extract(b)
        for a in c["atoms"]:
            self.assertEqual(
                b["evidence"][a["source"]][a["start"] : a["end"]], a["text"]
            )

    def test_wrong_field_unavailable(self):
        c = extract(bundle("자동차 가격 3천만원 이하."))
        self.assertFalse(c["atoms"])

    def test_decimal_income_range(self):
        s = "가구 중위소득 60% 초과 150% 이하."
        self.assertEqual(run(s, {"income_pct": 60}), "contradicted")
        self.assertEqual(run(s, {"income_pct": 60.01}), "supported")

    def test_no_rent_exclusion(self):
        s = "임차보증금 1억원 이하 및 월 임차료 60만원 이하. 월세가 없는 전세계약은 신청 불가."
        self.assertEqual(
            run(s, {"deposit_krw": 0, "monthly_rent_krw": 0}), "contradicted"
        )

    def test_absent_deposit_is_not_required(self):
        s = "보증금 1억원 이하, 월세 60만원 이하. 보증금이 없는 월세는 신청 가능하나 월세가 없는 전세계약은 신청 불가."
        self.assertEqual(
            run(s, {"deposit_krw": 100000000, "monthly_rent_krw": 600000}), "supported"
        )

    def test_percentage_coefficient(self):
        s = "임차보증금 3억원 이하. 환산보증금 : 임차보증금 + [(월임대료 × 12) ÷ 6.4%]"
        self.assertEqual(
            run(s, {"deposit_krw": 112500000, "monthly_rent_krw": 1000000}), "supported"
        )
        self.assertEqual(
            run(s, {"deposit_krw": 112500001, "monthly_rent_krw": 1000000}),
            "contradicted",
        )

    def test_benefit_exclusion_from_scope_is_flagged(self):
        c = extract(
            bundle(
                "기초생활수급자(생계·의료·주거·교육)", "4종 급여 수급자 제외 요건만."
            )
        )
        self.assertEqual(len(c["atoms"]), 4)
        self.assertEqual({a["origin"] for a in c["atoms"]}, {"scope"})

    def test_no_unsupported_id(self):
        c = extract(bundle("보증금 1억원 이하."))
        self.assertEqual(
            compile_selection(dict(status="ready", clauses=[["unseen"]]), c)["status"],
            "insufficient",
        )

    def test_residence_after_amount(self):
        self.assertEqual(
            run("본인이 24개월 이상 연속 거주해야 한다.", {"residence_months": 24}),
            "supported",
        )

    def test_and_or_same_facts(self):
        base = "거주기간 24개월 이상과 가구 중위소득 120% 이하를 "
        facts = {"residence_months": 24, "income_pct": 121}
        self.assertEqual(run(base + "모두 충족해야 한다.", facts), "contradicted")
        self.assertEqual(run(base + "하나라도 충족해야 한다.", facts), "supported")

    def test_unresolved_exception_abstains(self):
        s = "보증금 1억원 이하. 다만 재난 피해자는 예외다."
        self.assertEqual(
            rule_selection(bundle(s), extract(bundle(s)))["status"], "insufficient"
        )


if __name__ == "__main__":
    unittest.main()
