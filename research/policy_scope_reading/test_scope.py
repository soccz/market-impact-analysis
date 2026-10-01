"""Test inference isolation, selective checks and formal-reference boundaries."""

import unittest

from evaluate import consensus, parsed, score
from guard import apply
from infer import messages
from logic import all_of, any_of, oracle


class ScopeTests(unittest.TestCase):
    def test_gold_and_oracle_never_enter_prompt(self):
        a = {
            "sentences": {"S1": "공개 입력"},
            "claim": "공개 주장",
            "reference": {"decision": "SECRET_LABEL"},
            "oracle_facts": {"hidden": 123},
        }
        text = str(messages(a, "prompt"))
        self.assertNotIn("SECRET_LABEL", text)
        self.assertNotIn("hidden", text)

    def test_false_and_unknown_does_not_become_unknown(self):
        self.assertFalse(all_of([False, None]))
        self.assertIsNone(all_of([True, None]))

    def test_true_or_unknown_does_not_become_unknown(self):
        self.assertTrue(any_of([True, None]))
        self.assertIsNone(any_of([False, None]))

    def test_versions_bind_before_numeric_comparison(self):
        f = dict(old=True, first=True, income=125)
        self.assertFalse(oracle("version_scope", f))
        f["old"] = False
        self.assertTrue(oracle("version_scope", f))

    def test_semantic_unknown_is_distinct_from_system_abstention(self):
        c = {"sentences": {"S1": "미확정"}}
        p = parsed({"parsed": {"decision": "not_established", "evidence": ["S1"]}}, c)
        self.assertEqual(p["decision"], "not_established")
        self.assertEqual(
            consensus(p, {"decision": "supported", "evidence": ["S1"]})["decision"],
            "abstain",
        )

    def test_unknown_citation_fails_closed(self):
        self.assertEqual(
            parsed(
                {"parsed": {"decision": "supported", "evidence": ["invented"]}},
                {"sentences": {"S1": "x"}},
            )["decision"],
            "abstain",
        )

    def test_false_numeric_trace_is_rejected_without_label(self):
        p = {"decision": "supported", "evidence": ["S1"]}
        r = {"parsed": {"rule": "140 ≤ 120, 충족"}}
        self.assertEqual(
            apply({"claim": "조건을 충족한다."}, r, p)["decision"], "abstain"
        )

    def test_correct_numeric_trace_can_disagree_with_decision(self):
        p = {"decision": "supported", "evidence": ["S1"]}
        r = {"parsed": {"rule": "140 > 120, 이하 조건 불충족"}}
        self.assertEqual(
            apply({"claim": "조건을 충족한다."}, r, p)["decision"], "abstain"
        )

    def test_guard_does_not_claim_to_solve_other_language(self):
        p = {"decision": "supported", "evidence": ["S1"]}
        r = {"parsed": {"rule": "조건 충족"}}
        self.assertEqual(apply({"claim": "조건을 충족한다."}, r, p), p)

    def test_sufficient_evidence_and_extra_citations_separate(self):
        row = {
            "id": "a",
            "family": "x",
            "sentences": {"S1": "근거", "S2": "부가"},
            "reference": {"decision": "supported", "evidence": ["S1"]},
        }
        out = score([row], {"a": {"decision": "supported", "evidence": ["S1", "S2"]}})
        self.assertEqual(out["metrics"]["joint"], 1)
        self.assertEqual(out["metrics"]["extra_citations"], 1)


if __name__ == "__main__":
    unittest.main()
