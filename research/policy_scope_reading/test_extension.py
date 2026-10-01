"""Verify that adaptive prompts cannot see labels or privileged source locators."""

import unittest
from extension_infer import make_request
from evaluate import parsed
from class_diagnostics import diagnose
from thinking_infer import request_for as thinking_request
from sampling_infer import request_for as sampling_request
from citation_infer import make_request as citation_request


class ExtensionTests(unittest.TestCase):
    def test_reference_isolation_in_dynamic_schema(self):
        c = dict(
            sentences={"A": "내용", "B": "정보"},
            claim="주장",
            reference={"decision": "SECRET"},
            source_locators={"private": "SECRET"},
        )
        request = make_request(c, "qwen", "evidence")
        self.assertNotIn("SECRET", str(request))
        self.assertEqual(
            request["format"]["properties"]["evidence"]["items"]["enum"], ["A", "B"]
        )
        self.assertFalse(request["think"])

    def test_baseline_contract_unchanged(self):
        request = make_request(
            dict(sentences={"S1": "값"}, claim="주장"), "qwen", "baseline"
        )
        self.assertNotIn("coverage", request["format"]["properties"])
        self.assertEqual(request["options"]["num_predict"], 256)

    def test_coverage_is_not_a_gold_label_override(self):
        c = dict(sentences={"S1": "정보"})
        result = parsed(
            dict(
                parsed=dict(
                    coverage="insufficient", decision="supported", evidence=["S1"]
                )
            ),
            c,
        )
        self.assertEqual(result["decision"], "supported")

    def test_mode_comparisons_change_only_the_mode(self):
        case = dict(sentences={"S1": "명시된 조건"}, claim="확인할 주장")
        for builder in [thinking_request, sampling_request]:
            a, b = builder(case, False), builder(case, True)
            self.assertFalse(a.pop("think"))
            self.assertTrue(b.pop("think"))
            self.assertEqual(a, b)

    def test_citation_ablation_keeps_the_baseline_prompt_and_decoder(self):
        case = dict(
            sentences={"S1": "실제 문장"},
            claim="주장",
            reference={"decision": "SECRET"},
        )
        baseline = make_request(case, "qwen", "baseline")
        limited = citation_request(case, "qwen", "citation-only")
        self.assertEqual(baseline["messages"], limited["messages"])
        self.assertEqual(baseline["options"], limited["options"])
        self.assertEqual(
            set(baseline["format"]["properties"]), set(limited["format"]["properties"])
        )
        self.assertNotIn("SECRET", str(limited))

    def test_no_answer_has_no_selective_accuracy(self):
        rows = [
            dict(reference=x, prediction=dict(decision="abstain"), correct=False)
            for x in ["supported", "contradicted", "not_established"]
        ]
        result = diagnose(rows)
        self.assertIsNone(result["selective_accuracy"])
        self.assertEqual(result["output_coverage"], 0)
        self.assertEqual(result["macro_f1"], 0)

    def test_unknown_only_controls_do_not_get_three_class_macro_f1(self):
        result = diagnose(
            [
                dict(
                    reference="not_established",
                    prediction=dict(decision="not_established"),
                    correct=True,
                )
            ]
        )
        self.assertIsNone(result["macro_f1"])
        self.assertEqual(result["selective_accuracy"], 1)


if __name__ == "__main__":
    unittest.main()
