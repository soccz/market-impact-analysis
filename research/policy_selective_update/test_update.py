import copy
import json
import unittest

from core import decision, refresh_needed, updated
from infer import request


class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.bundle = {
            "id": "b",
            "before": {"O1": "대상은 34세 이하이다."},
            "after": {
                "N1": "대상은 34세 이하이다.",
                "N2": "제대군인의 상한은 3년 연장한다.",
            },
            "reference": {"secret": "REFERENCE_SENTINEL"},
        }
        self.case = {
            "claim": "군 복무가 없는 30세인 사람은 나이 조건을 충족한다.",
            "reference": "REFERENCE_SENTINEL",
        }
        self.old = {"decision": "supported", "evidence": ["O1"], "reason": "valid"}
        self.new = {"decision": "contradicted", "evidence": ["N1"], "reason": "valid"}

    def test_reference_never_enters_any_request(self):
        for stage in ["before", "after", "map", "direct_gate", "map_gate"]:
            payload = request(
                self.bundle, self.case, "qwen", stage, self.old, {"changes": []}
            )
            self.assertNotIn("REFERENCE_SENTINEL", json.dumps(payload))

    def test_keep_preserves_actual_old_error(self):
        result = updated(
            "change_map_gate",
            self.old,
            self.new,
            self.bundle,
            {"route": "keep", "evidence": ["N1"]},
        )
        self.assertEqual(result["prediction"], self.old)
        self.assertFalse(result["refreshed"])

    def test_unknown_or_invalid_gate_refreshes(self):
        for gate in [
            None,
            {"route": "uncertain", "evidence": ["N1"]},
            {"route": "keep", "evidence": ["FAKE"]},
        ]:
            self.assertTrue(
                refresh_needed("direct_scope_gate", self.old, self.bundle, gate)
            )

    def test_invalid_cache_is_never_retained(self):
        old = decision(None, self.bundle["before"])
        self.assertTrue(
            refresh_needed(
                "change_map_gate",
                old,
                self.bundle,
                {"route": "keep", "evidence": ["N1"]},
            )
        )

    def test_added_exception_is_a_real_lexical_gate_blind_spot(self):
        self.assertFalse(refresh_needed("citation_text_gate", self.old, self.bundle))
        changed = copy.deepcopy(self.bundle)
        changed["after"]["N1"] = "대상은 37세 이하이다."
        self.assertTrue(refresh_needed("citation_text_gate", self.old, changed))

    def test_uncertainty_is_not_invalid_output(self):
        p = decision(
            {"decision": "not_established", "evidence": ["N1"]}, self.bundle["after"]
        )
        self.assertEqual(p["reason"], "valid")

    def test_old_prompt_has_no_future_evidence(self):
        p = request(self.bundle, self.case, "qwen", "before")
        self.assertNotIn("N2", json.dumps(p, ensure_ascii=False))

    def test_map_contains_no_claim_or_previous_prediction(self):
        p = request(self.bundle, None, "qwen", "map")
        self.assertNotIn(self.case["claim"], str(p))
        self.assertNotIn("previous_model_evidence_ids", str(p))


if __name__ == "__main__":
    unittest.main()
