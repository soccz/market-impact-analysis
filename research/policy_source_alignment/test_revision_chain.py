import unittest
from revision_chain import source_contract, ground_contract, claim_slots, update


class RevisionTests(unittest.TestCase):
    def test_boundary_and_missing(self):
        old, _ = ground_contract({"O": "최대30만원 한도"}, None)
        new, _ = ground_contract(
            {
                "N": "최대40만원 한도. 2025. 3. 31. 이후 보증 가입자만 해당(이전 보증 가입자는 최대 30만원)"
            },
            None,
        )
        for day, expected in [
            ("2025-03-30", "contradicted"),
            ("2025-03-31", "supported"),
            (None, "not_established"),
        ]:
            q = dict(status="ready", joined_on=day, asserted_cap=400000)
            self.assertEqual(update(old, new, q, "contradicted")["updated"], expected)

    def test_stable_wrong_answer_repaired(self):
        old, _ = ground_contract({"O": "최대30만원 한도"}, None)
        x = update(
            old,
            old,
            dict(status="ready", joined_on=None, asserted_cap=200000),
            "supported",
        )
        self.assertFalse(x["revision_changes_decision"])
        self.assertTrue(x["prior_answer_disagrees"])
        self.assertEqual(x["updated"], "contradicted")

    def test_application_date_not_join_date(self):
        q = claim_slots(
            "내 신청일은 2025년 4월 1일이다. 내 보증 가입일은 밝히지 않았다. 이 보증료 지원 상한은 40만원이다."
        )
        self.assertEqual(q, dict(status="ready", joined_on=None, asserted_cap=400000))

    def test_parent_date(self):
        q = claim_slots(
            "부모의 보증 가입일은 2025년 4월 1일이다. 내 보증 가입일은 2025년 3월 30일이다. 이 보증료 지원 상한은 40만원이다."
        )
        self.assertEqual(q["joined_on"], "2025-03-30")

    def test_invalid_date(self):
        self.assertEqual(
            claim_slots(
                "내 보증 가입일은 2025년 2월 30일이다. 이 보증료 지원 상한은 40만원이다."
            )["status"],
            "unsupported",
        )

    def test_conflicting_dates(self):
        self.assertEqual(
            claim_slots(
                "내 보증 가입일은 2025년 3월 30일이다. 내 보증 가입일은 2025년 4월 1일이다. 이 보증료 지원 상한은 40만원이다."
            )["status"],
            "unsupported",
        )

    def test_unsupported_source(self):
        self.assertIsNone(
            source_contract({"E": "최대30만원, 다만 특별 대상은 별도 산정"})
        )


if __name__ == "__main__":
    unittest.main()
