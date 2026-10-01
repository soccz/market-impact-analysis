import unittest, copy, json
from infer_candidates_v2 import request
from repair_alignment import feedback
from candidates import extract
from bridge import sha


class IsolationTests(unittest.TestCase):
    def test_reference_and_claim_not_in_policy_request(self):
        b = dict(
            id="x",
            scope="소득 상하한만.",
            evidence={"E": "중위소득 60% 초과 150% 이하."},
            reference_policy={"secret_reference": "first"},
        )
        first = request(b, None, "qwen", "policy")
        b["reference_policy"] = {"secret_reference": "changed"}
        self.assertEqual(first, request(b, {"claim": "hidden claim"}, "qwen", "policy"))
        self.assertNotIn("secret_reference", json.dumps(first))

    def test_reference_cannot_change_feedback(self):
        b = dict(
            id="x",
            scope="소득 상하한만.",
            evidence={"E": "중위소득 60% 초과 150% 이하."},
            reference_policy={},
        )
        selection = dict(status="ready", clauses=[["a0"], ["a1"]])
        a = feedback(b, selection)
        b["reference_policy"] = {"clauses": "arbitrary"}
        self.assertEqual(a, feedback(b, selection))
        self.assertEqual(a["reason"], "source_parser_disagreement")

    def test_correct_source_conjunction_no_retry(self):
        b = dict(
            id="x",
            scope="소득 상하한만.",
            evidence={"E": "중위소득 60% 초과 150% 이하."},
        )
        self.assertIsNone(feedback(b, dict(status="ready", clauses=[["a0", "a1"]])))


if __name__ == "__main__":
    unittest.main()
